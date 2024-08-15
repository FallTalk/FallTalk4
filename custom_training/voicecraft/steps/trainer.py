import time
import os, random
import torch
import math, pickle

from torch.utils.data import RandomSampler, SequentialSampler, BatchSampler
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import logging
from steps.WarmUpScheduler import WarmUpScheduler

from data import gigaspeech
from models import voicecraft, voicecraft_bak
from models.LoRA import LinearWithDoRAMerged

from .trainer_utils import AverageMeter, print_model_info
from .optim import ScaledAdam, Eden


# Function to check if weights have changed
def check_weights_changed(initial_weights, updated_weights):
    for initial, updated in zip(initial_weights, updated_weights):
        if torch.equal(initial, updated):
            return False
    return True


def freeze_linear_layers(model):
    for child in model.children():
        if isinstance(child, nn.Linear) and not isinstance(child, nn.modules.linear.NonDynamicallyQuantizableLinear):
            for param in child.parameters():
                param.requires_grad = False
        else:
            # Recursively freeze linear layers in children modules
            freeze_linear_layers(child)


def freeze_all_layers(model):
    for name, param in model.named_parameters():
        if ("mask_embedding" in name or "audio_embedding" in name or "text_embedding" in name
                or "eog" in name or "eos" in name
                or "text_positional_embedding" in name or "audio_positional_embedding" in name):
            logging.debug(f"Freezing param {name}")
            param.requires_grad = False
        else:
            #print(f"not freezing {name}")
            pass


def replace_linear_with_dora(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and not isinstance(module, nn.modules.linear.NonDynamicallyQuantizableLinear):
            logging.debug(f"replacing layer {name}, module {module}")
            # Create a new DoRALayer with the same dimensions
            setattr(model, name, LinearWithDoRAMerged(module))
        else:
            # Recursively apply this function to submodules
            replace_linear_with_dora(module)


def replace_decoder_with_dora(transformer_decoder):
    if isinstance(transformer_decoder, nn.TransformerDecoder):
        for i, decoder_layer in enumerate(transformer_decoder.layers):
            for name, module in decoder_layer.named_children():
                if isinstance(module, nn.Linear) and not isinstance(module,
                                                                    nn.modules.linear.NonDynamicallyQuantizableLinear):
                    logging.debug(f"replacing layer {name}, module {module}")
                    # Create a new DoRALayer with the same dimensions
                    setattr(decoder_layer, name, LinearWithDoRAMerged(module))
                elif not isinstance(module, nn.MultiheadAttention):
                    # Recursively apply this function to submodules, skipping MultiheadAttention
                    replace_decoder_with_dora(module)
    elif isinstance(transformer_decoder, nn.Module):
        for name, module in transformer_decoder.named_children():
            if isinstance(module, nn.Linear) and not isinstance(module,
                                                                nn.modules.linear.NonDynamicallyQuantizableLinear):
                logging.debug(f"replacing layer {name}, module {module}")
                # Create a new DoRALayer with the same dimensions
                setattr(transformer_decoder, name, LinearWithDoRAMerged(module))
            elif not isinstance(module, nn.MultiheadAttention):
                # Recursively apply this function to submodules, skipping MultiheadAttention
                replace_decoder_with_dora(module)


def replace_predict_with_dora(predict_layer):
    for i, module in enumerate(predict_layer):
        if isinstance(module, nn.Linear) and not isinstance(module, nn.modules.linear.NonDynamicallyQuantizableLinear):
            logging.debug(f"replacing layer at index {i}, module {module}")
            # Create a new DoRALayer with the same dimensions
            predict_layer[i] = LinearWithDoRAMerged(module)
        elif isinstance(module, nn.ModuleList) or isinstance(module, nn.Sequential):
            # Recursively apply this function to submodules
            replace_predict_with_dora(module)
        elif isinstance(module, nn.Module):
            # Handle non-iterable modules like GELU
            for name, sub_module in module.named_children():
                if isinstance(sub_module, nn.Linear) and not isinstance(sub_module,
                                                                        nn.modules.linear.NonDynamicallyQuantizableLinear):
                    logging.debug(f"replacing sub-module {name}, module {sub_module}")
                    setattr(module, name, LinearWithDoRAMerged(sub_module))
                else:
                    replace_predict_with_dora(sub_module)


class Trainer:

    def __init__(self, args, world_size, rank):
        self.start_time = time.time()
        self.args = args
        self.world_size, self.rank = world_size, rank
        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(self.device)
        if self.rank == 0:
            self.writer = SummaryWriter(args.exp_dir)
        self.seed_everything(seed=self.args.seed)
        self.meters = self._setup_meters()

        self.ds_enabled = True

        self.progress, self.total_progress = self._setup_progress()

        self.model, self.trainables, self.optim_states, self.scheduler_states = self._setup_models()

        self.train_dataset_length, self.train_sampler, self.train_loader, self.valid_loader = self._setup_dataloader()
        if self.args.num_steps != None:
            self.total_step = self.args.num_steps
            self.args.num_epochs = math.ceil(self.total_step / math.floor(
                self.train_dataset_length / self.args.batch_size)) if not self.args.dynamic_batching else None
        else:
            self.total_step = int(math.floor(self.train_dataset_length / self.args.batch_size)) * self.args.num_epochs

        if self.ds_enabled:
            print("setting up deepspeed")
            import deepspeed
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            os.environ['LOCAL_RANK'] = '0'
            os.environ["DS_ACCELERATOR"] = 'cuda'
            torch.distributed.init_process_group(backend='gloo', rank=0, world_size=1, init_method='env://')

            ds_params = []
            self.model.parameters()
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.is_leaf:
                    ds_params.append(param)

            self.optimizer = DeepSpeedCPUAdam(self.trainables, lr=self.args.lr)
            self.ds_config = {
                "train_micro_batch_size_per_gpu": self.args.batch_size,
                "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
                "flops_profiler": {
                    "enabled": False,
                },
                "fp16": {
                    "enabled": True,
                    "auto_cast": False,
                    "loss_scale": 0,
                    "loss_scale_window": 1000,
                    "hysteresis": 2,
                    "min_loss_scale": 1,
                },
                "zero_optimization": {
                    "stage": 2,
                    "offload_optimizer": {
                        "device": "cpu",
                        "pin_memory": True
                    },
                    "offload_param": {
                        "device": "cpu",
                        "pin_memory": True
                    },
                    "allgather_partitions": True,
                    "allgather_bucket_size": 2e8,
                    "reduce_scatter": True,
                    "reduce_bucket_size": 2e8,
                    "overlap_comm": True,
                    "contiguous_gradients": True
                },
                "tensorboard": {
                    "enabled": True,
                    "output_path": os.path.join(self.args.exp_dir, "ds_logs"),
                    "job_name": "training"
                },
                "gradient_clipping": 1.0,
                "distributed_backend": "gloo",
                "wall_clock_breakdown": False,
                "accelerator": "cuda",
                "steps_per_print": 50,
            }

            warmup_steps = self.total_step * self.args.warmup_fraction
            after_scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=int(self.total_step - warmup_steps // 6),T_mult=1, eta_min=1e-9)
            self.scheduler = WarmUpScheduler(self.optimizer, self.total_step, 1e-9, self.args.lr, self.args.warmup_fraction, after_scheduler)

            self.model, self.optimizer, _, _ = deepspeed.initialize(
                model=self.model,
                config=self.ds_config,
                dist_init_required=False,
                optimizer=self.optimizer
            )

        else:
            self.optimizer, self.scheduler = self._setup_optimizer()
            self.scaler = torch.cuda.amp.GradScaler()

        self.model = self.model.to(self.device)

        if self.rank == 0:
            self.early_stop_accu_steps = 0
            if self.args.dynamic_batching:
                logging.info(
                    f"max number of tokens per GPU in a training batch: {self.args.max_num_tokens}, max number of tokens per GPU in a inference batch: {self.args.val_max_num_tokens}")
            else:
                logging.info(f"batch size (summed over all GPUs): {self.args.batch_size}")

    def train(self):
        flag = True
        skip_flag = False
        data_start_time = time.time()

        while flag:
            #self.train_sampler.set_epoch(self.progress['epoch'])
            for i, batch in tqdm(iterable=enumerate(self.train_loader), total=len(self.train_loader),
                                 desc=f"Training Epoch {self.progress['epoch']}"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                data_end_time = time.time()
                self.model.train()
                if self.progress['step'] > self.total_step:
                    logging.info("Running Final Validation and Save")
                    flag = False
                    self.validate_and_save()
                    if self.rank == 0:
                        self.writer.close()
                    break

                if isinstance(self.scheduler, Eden):
                    self.scheduler.step_epoch(self.progress['step'] // self.args.pseudo_epoch_size + 1)
                if self.args.optimizer_name == "ScaledAdam":
                    cur_lr = self.scheduler.get_last_lr()[0]
                else:
                    lrs = [param_group['lr'] for param_group in self.optimizer.param_groups]
                    if len(lrs) > 1:
                        assert lrs[0] == lrs[1]
                        cur_lr = lrs[0]
                    else:
                        cur_lr = lrs[0]

                if self.rank == 0 and self.progress['step'] % self.args.tb_write_every_n_steps == 0:
                    self.writer.add_scalar("train/lr", cur_lr, self.progress['step'])

                sum_losses = 0
                sum_top10acc = 0
                sum_ntoken = 0
                sum_top10acc_cbi = [0 for _ in range(self.args.n_codebooks)]

                if not self.ds_enabled:
                    all_inds = list(range(len(batch['y'])))

                    for j in range(self.args.gradient_accumulation_steps):
                        cur_ind = all_inds[j::self.args.gradient_accumulation_steps]
                        cur_batch = {key: batch[key][cur_ind] for key in batch}

                        with torch.cuda.amp.autocast(
                                dtype=torch.float16 if self.args.precision == "float16" else torch.float32):
                            out = self.model(cur_batch)
                            if out == None:
                                continue

                        is_nan = torch.tensor(int(torch.isnan(out['loss']).any()), dtype=torch.float32,
                                              device=self.rank)

                        # check if loss is nan
                        if is_nan.item() > 0:
                            logging.info(f"loss at step {self.progress['step']} is nan, therefore skip this batch")
                            skip_flag = True
                            continue

                        sum_losses += out['loss'].item()
                        sum_top10acc += out['top10acc'].item()
                        sum_ntoken += out['effective_ntoken'].item()

                        if 'top10acc_by_codebook' in out:
                            for cb in range(self.args.n_codebooks):
                                sum_top10acc_cbi[cb] += out['top10acc_by_codebook'][cb].item()

                        average_loss = sum_losses / sum_ntoken
                        average_top10acc = sum_top10acc / sum_ntoken
                        self.meters['train_loss'].update(average_loss, batch['x'].shape[0] * self.world_size)
                        self.meters['train_top10acc'].update(average_top10acc, batch['x'].shape[0] * self.world_size)

                        average_top10acc_cbi = [sum_top10acc_cbi[cb] / sum_ntoken * self.args.n_codebooks for cb in
                                                range(self.args.n_codebooks)]
                        for cb in range(self.args.n_codebooks):
                            self.meters[f'train_top10acc_cb{cb + 1}'].update(average_top10acc_cbi[cb],
                                                                             batch['x'].shape[0] * self.world_size)

                        if self.progress['step'] % self.args.tb_write_every_n_steps == 0:
                            self.writer.add_scalar('train/loss', average_loss, self.progress['step'])
                            self.writer.add_scalar('train/top10acc', average_top10acc, self.progress['step'])
                            self.writer.add_scalar("train/ntokens", sum_ntoken, self.progress['step'])
                            for cb in range(self.args.n_codebooks):
                                self.writer.add_scalar(f'train/top10acc_cb{cb + 1}', average_top10acc_cbi[cb],
                                                       self.progress['step'])

                        if not self.ds_enabled:
                            if self.args.optimizer_name == "ScaledAdam":
                                self.scaler.scale(out['loss']).backward()
                            else:
                                self.scaler.scale(out['loss'] / out['effective_ntoken']).backward()
                        else:
                            self.model.backward(out['loss'])

                        del cur_batch, out, is_nan
                else:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        out = self.model(batch)

                    sum_ntoken += out['effective_ntoken'].item()
                    self.meters['train_loss'].update(out['loss'].item() / sum_ntoken, batch['x'].shape[0] * self.world_size)
                    self.meters['train_top10acc'].update(out['top10acc'].item() / sum_ntoken, batch['x'].shape[0] * self.world_size)
                    self.writer.add_scalar('train/loss', out['loss'].item() / sum_ntoken, self.progress['step'])
                    self.writer.add_scalar('train/top10acc', out['top10acc'].item() / sum_ntoken, self.progress['step'])
                    self.writer.add_scalar("train/ntokens", sum_ntoken, self.progress['step'])
                    self.model.backward(out['loss'] / sum_ntoken)
                    del out

                if skip_flag and not self.ds_enabled:
                    self.optimizer.zero_grad()
                    skip_flag = False
                    continue

                if not self.ds_enabled:
                    if self.args.optimizer_name != "ScaledAdam":
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clip_val)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.model.step()

                if not self.ds_enabled:
                    self.optimizer.zero_grad()
                else:
                    pass
                    #self.model.zero_grad()

                if not self.ds_enabled:
                    if self.args.optimizer_name == "ScaledAdam":
                        self.scheduler.step_batch(self.progress['step'])
                    else:
                        self.scheduler.step()
                else:
                    self.scheduler.step()

                if self.rank == 0:
                    self.meters['data_time'].update(data_end_time - data_start_time)
                    self.meters['train_time'].update(time.time() - data_end_time)
                    if self.progress['step'] % self.args.tb_write_every_n_steps == 0:
                        self.writer.add_scalar("train/data_time", data_end_time - data_start_time,
                                               self.progress['step'])
                        self.writer.add_scalar("train/train_time", time.time() - data_end_time, self.progress['step'])

                    # logging
                    if self.progress['step'] % self.args.print_every_n_steps == 0:
                        log_out = {}
                        log_out[
                            'cur_epoch'] = f"{self.progress['epoch']}/{self.args.num_epochs}" if self.args.num_epochs is not None else f"{self.progress['epoch']}"
                        log_out['cur_step'] = f"{int(self.progress['cur_step'] + 1)}"
                        log_out['total_step'] = f"{self.progress['step']}/{self.args.num_steps}"
                        log_out['lr'] = f"{cur_lr:.7f}"
                        log_out['ntokens'] = f"{sum_ntoken}"
                        for key in self.meters:
                            if self.meters[key].val != 0 or self.meters[key].avg != 0:
                                log_out[key] = f"{self.meters[key].val:.4f} ({self.meters[key].avg:.4f})" if isinstance(
                                    self.meters[key].val, float) else f"{self.meters[key].val}"
                        logging.info(log_out)
                        if np.isnan(self.meters['train_loss'].avg):
                            logging.warning("training diverged...")
                            raise RuntimeError("training diverged...")

                # validation and save models
                if self.progress['step'] % self.args.val_every_n_steps == 0:
                    self.validate_and_save()

                self.progress['step'] += 1
                self.progress['cur_step'] += 1

                data_start_time = time.time()

            logging.info(f"Completed Epoch {self.progress['epoch']}")
            self.progress['epoch'] += 1
            self.progress['cur_step'] = 0  # reset cur_step to be 0

    def save_lora_weights(self, save_path):
        lora_weights = {}
        for name, param in self.model.named_parameters():
            if 'lora' in name:
                lora_weights[name] = param.data
        torch.save(lora_weights, save_path)
        logging.info(f"save {len(lora_weights)} LoRA weights at {save_path} at global step {self.progress['step']}")

    def validate_and_save(self):
        torch.cuda.empty_cache()

        logging.info(f"validate_and_save")

        self.model.eval()

        score = self.validate(self.valid_loader)

        if self.args.early_stop_threshold > 0:
            if self.progress['best_score'] - score < self.args.early_stop_threshold:
                self.early_stop_accu_steps += self.args.val_every_n_steps
                if self.early_stop_accu_steps >= self.args.early_stop_step - 1:
                    logging.info(
                        f"early stop based on self.args.early_stop_threshold: {self.args.early_stop_threshold}, and self.args.early_stop_step: {self.args.early_stop_step}")
                    logging.info(
                        f"best validation score at step: {self.progress['best_step']}, and the score is {self.progress['best_score']:.4f}")
                    raise RuntimeError("early stop")
                else:
                    self.early_stop_accu_steps = 0

        if (score < self.progress['best_score']):
            self.progress['best_step'] = self.progress['step']
            self.progress['best_score'] = score
            save_path = os.path.join(self.args.exp_dir, "best_lora_weights.pth")
            # Extract and save only the LoRA weights
            self.save_lora_weights(save_path)

        self._save_progress()
        save_path = os.path.join(self.args.exp_dir, (f"{self.progress['step']}_lora_weights.pth"))
        self.save_lora_weights(save_path)

        if (score < self.progress['best_score']):
            self.progress['best_step'] = self.progress['step']
            self.progress['best_score'] = score
            save_path = os.path.join(self.args.exp_dir, "best_bundle.pth")
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "config": self.args,
                    "phn2num": self.train_loader.dataset.phn2num
                }, save_path
            )
            logging.info(f"save *best* models at {save_path} at global step {self.progress['step']}")
        self._save_progress()
        save_path = os.path.join(self.args.exp_dir, "bundle.pth")
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "config": self.args,
                "phn2num": self.train_loader.dataset.phn2num
            }, save_path
        )
        logging.info(
            f"save models, indices, acc and other statistics at {save_path} and {self.args.exp_dir}/progress.pkl at global step {self.progress['step']}")

    def validate(self, valid_loader=None):
        logging.info(f"Validate {valid_loader}")
        if valid_loader is None:
            valid_loader = self.valid_loader
        self.model.eval()

        start_val_time = time.time()
        sum_losses = 0
        sum_top10acc = 0
        sum_ntoken = 0
        sum_top10acc_cbi = [0 for _ in range(self.args.n_codebooks)]

        logging.info("Running torch")

        with torch.no_grad():
            for batch in tqdm(valid_loader, disable=False):
                # Move all tensors in the batch to the GPU in a single operation
                batch = {k: v.to(self.device) for k, v in batch.items()}

                out = self.model(batch)
                sum_losses += out['loss'].item()
                sum_top10acc += out['top10acc'].item()
                sum_ntoken += out['effective_ntoken'].item()
                if 'top10acc_by_codebook' in out:
                    for cb in range(self.args.n_codebooks):
                        sum_top10acc_cbi[cb] += out['top10acc_by_codebook'][cb].item()

                # Create a list of keys to iterate over
                keys = list(batch.keys())
                # Delete the tensors after they are used
                for k in keys:
                    del batch[k]


        val_loss = sum_losses / sum_ntoken
        val_top10acc = sum_top10acc / sum_ntoken
        print(f"sum_losses {sum_losses}")
        print(f"sum_ntoken {sum_ntoken}")
        print(f"sum_top10acc {sum_top10acc}")
        print(f"val_loss {val_loss}")
        print(f"val_top10acc {val_top10acc}")

        if np.isnan(val_loss) or np.isinf(val_loss):
            logging.warning("training diverged...")
            raise RuntimeError("training diverged...")

        # Logging
        self.meters['val_loss'].update(val_loss)
        logging.info(f"Val loss: {val_loss:.5f}")
        self.writer.add_scalar("val/loss", val_loss, self.progress['step'])

        self.meters['val_top10acc'].update(val_top10acc)
        logging.info(f"Val top10acc: {val_top10acc:.5f}")
        self.writer.add_scalar("val/top10acc", val_top10acc, self.progress['step'])

        for cb in range(self.args.n_codebooks):
            average_top10acc_cbi = sum_top10acc_cbi[cb] / sum_ntoken * self.args.n_codebooks
            self.meters[f'val_top10acc_cb{cb + 1}'].update(average_top10acc_cbi)
            self.writer.add_scalar(f'val/top10acc_cb{cb + 1}', average_top10acc_cbi, self.progress['step'])

        logging.info(f"Validation takes: {time.time() - start_val_time:.2f}s")
        logging.info(
            f"Step [{self.progress['step']}/{self.total_step}]\t Time elapsed {(time.time() - self.start_time) / 3600.:.2f}h, Val Loss: {val_loss:.4f}, Val Top10Acc: {val_top10acc:.4f}")

        torch.cuda.empty_cache()

        return val_loss

    def _setup_meters(self):
        meters = {}
        meter_names = ['train_loss', 'val_loss', 'train_top10acc', 'val_top10acc', 'data_time', 'train_time']
        meter_names += ['train_dur_loss', 'train_dur_acc', 'val_dur_loss', 'val_dur_acc']
        meter_names += [f'train_top10acc_cb{cb + 1}' for cb in range(self.args.n_codebooks)]
        meter_names += [f'val_top10acc_cb{cb + 1}' for cb in range(self.args.n_codebooks)]
        for name in meter_names:
            meters[name] = AverageMeter()
        return meters

    def _setup_progress(self):
        progress = {}
        progress['best_step'] = 1
        progress['best_score'] = np.inf  # this records loss value
        progress['step'] = 1
        progress['epoch'] = 1
        progress['cur_step'] = 0  # step in the current epoch, for resuming the sampler
        total_progress = []
        # if self.args.resume or self.args.validate:
        if self.args.resume:
            progress_pkl = "%s/progress.pkl" % self.args.exp_dir
            with open(progress_pkl, "rb") as f:
                total_progress = pickle.load(f)
                progress['best_step'], progress['best_score'], progress['step'], progress['epoch'], progress[
                    'cur_step'], _ = total_progress[-1]
            if self.rank == 0:
                logging.info("\nResume training from:")
                logging.info("  epoch = %s" % progress['epoch'])
                logging.info("  cur_step = %s" % progress['cur_step'])
                logging.info("  step = %s" % progress['step'])
                logging.info("  best_step = %s" % progress['best_step'])
                logging.info("  best_score = %s" % progress['best_score'])
        return progress, total_progress

    def _save_progress(self):
        self.total_progress.append(
            [self.progress['best_step'], self.progress['best_score'], int(self.progress['step'] + 1),
             self.progress['epoch'], int(self.progress['cur_step'] + 1), time.time() - self.start_time])
        with open("%s/progress.pkl" % self.args.exp_dir, "wb") as f:
            pickle.dump(self.total_progress, f)

    def _setup_dataloader(self):
        assert self.args.dataset == 'gigaspeech', "only gigaspeech is supported for now"
        train_dataset, val_dataset = gigaspeech.dataset(self.args, 'train'), gigaspeech.dataset(self.args, 'validation')
        print(f"number of data points for train split {len(train_dataset)}")
        print(f"number of val points for train split {len(val_dataset)}")
        train_sampler_r = RandomSampler(train_dataset, replacement=True, num_samples=len(train_dataset))
        train_sampler = BatchSampler(train_sampler_r, batch_size=self.args.batch_size, drop_last=True)
        valid_sampler = SequentialSampler(val_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_sampler=train_sampler,
                                                   num_workers=self.args.num_workers,
                                                   collate_fn=train_dataset.collate,
                                                   persistent_workers=True,
                                                   pin_memory=True
                                                   )
        valid_loader = torch.utils.data.DataLoader(val_dataset,
                                                   batch_size=self.args.batch_size,
                                                   sampler=valid_sampler,
                                                   num_workers=self.args.num_workers,
                                                   collate_fn=val_dataset.collate,
                                                   persistent_workers=True,
                                                   pin_memory=True
                                                   )
        return len(train_dataset), train_sampler, train_loader, valid_loader

    def _setup_models(self):
        model = voicecraft.VoiceCraft(self.args)

        trainable_count = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Before layer freezing {trainable_count(model)=}...")
        freeze_all_layers(model)

        trainable_count = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"After layer freezing {trainable_count(model)=}...")

        replace_predict_with_dora(model.predict_layer)
        replace_decoder_with_dora(model.decoder)

        freeze_linear_layers(model)

        if self.rank == 0:
            logging.info(f"-----------------------------------------------------------------------------")
            for name, param in model.named_parameters():
                logging.debug(f"{name}: requires_grad:{param.requires_grad}")
            print_model_info(model, True, True)

        if self.progress['step'] > 1:
            bundle = torch.load(os.path.join(self.args.exp_dir, "bundle.pth"), map_location="cpu")
            model.load_state_dict(bundle['model'], strict=False)
            optim_states = bundle['optimizer']
            scheduler_states = bundle['scheduler']
            if self.rank == 0:
                logging.info("loaded parameters and data indices from epoch %d, global step %d" % (
                    self.progress['epoch'], self.progress['step']))
            del bundle['model']
        else:
            optim_states = None
            scheduler_states = None

        if self.args.load_model_from != None and self.progress['step'] <= 1:
            sd = torch.load(self.args.load_model_from, map_location="cpu")['model']
            if '830M_TTSEnhanced' in self.args.load_model_from:
                model.text_embedding.word_embeddings = torch.nn.Embedding(121, 2048)
                print("Updating Word Embeddings")
            model.load_state_dict(sd, strict=False)
            del sd
        if self.args.optimizer_name == "ScaledAdam":
            trainables = [p for p in model.parameters() if p.requires_grad]
        else:
            no_decay = [".bias", ".audio_embeddings.weight", ".text_embeddings.weight", ".norm.weight",
                        ".norm1.weight",
                        ".norm2.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if
                               not any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                               any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": 0.0,
                },
            ]

            if len(optimizer_grouped_parameters[1]['params']) == 0:
                logging.info(
                    "there is no embedding weights, bias, and layernorm parameters in the model, which should be True, check model parameter names")
                trainables = [optimizer_grouped_parameters[0]]
            else:
                trainables = optimizer_grouped_parameters
        #model.to(self.device)

        return model, trainables, optim_states, scheduler_states

    def _setup_optimizer(self):
        if self.args.optimizer_name == "ScaledAdam":
            parameters_names = []
            parameters_names.append([n for n, p in self.model.named_parameters() if p.requires_grad])
            optimizer = ScaledAdam(
                self.trainables,
                lr=self.args.lr,
                betas=(0.9, 0.95),
                clipping_scale=2.0,
                parameters_names=parameters_names,
                show_dominant_parameters=False,
                clipping_update_period=self.args.clipping_update_period,
            )
            scheduler = Eden(optimizer, self.args.reduce_lr_start_step, self.args.reduce_lr_start_epoch,
                             warmup_batches=self.total_step * self.args.warmup_fraction)

        else:
            optimizer = AdamW(self.trainables, lr=self.args.lr)
            warmup_steps = self.total_step * self.args.warmup_fraction
            logging.info(f"warmup_steps = {warmup_steps}")

            def lr_lambda(current_step: int):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return max(
                    0.0, float(self.total_step - current_step) / float(max(1, self.total_step - warmup_steps))
                )

            after_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=int(self.total_step - warmup_steps // 6),
                                                          T_mult=1, eta_min=1e-9)
            scheduler = WarmUpScheduler(optimizer, self.total_step, 1e-7, self.args.lr, self.args.warmup_fraction,
                                        after_scheduler)

            #scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)

        # if resume
        if self.progress['step'] > 1:
            optimizer.load_state_dict(self.optim_states)
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            del self.optim_states

            scheduler.load_state_dict(self.scheduler_states)

        optimizer.zero_grad()
        return optimizer, scheduler

    def seed_everything(self, seed=1):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
