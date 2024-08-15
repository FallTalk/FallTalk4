import logging
import time

import torch

from tts_engines.voicecraft.data.tokenizer import (
    tokenize_audio,
    tokenize_text
)

@torch.no_grad()
def inference_one_sample(model, model_args, phn2num, text_tokenizer, audio_tokenizer, audio_fn, target_text, device,
                         decode_config, prompt_end_frame):
    # phonemize
    text_tokens = [phn2num[phn] for phn in
                   tokenize_text(
                       text_tokenizer, text=target_text.strip()
                   ) if phn in phn2num
                   ]
    text_tokens = torch.LongTensor(text_tokens).unsqueeze(0)
    text_tokens_lens = torch.LongTensor([text_tokens.shape[-1]])

    # encode audio
    encoded_frames = tokenize_audio(audio_tokenizer, audio_fn, offset=0, num_frames=prompt_end_frame)
    original_audio = encoded_frames[0][0].transpose(2, 1)  # [1,T,K]
    assert original_audio.ndim == 3 and original_audio.shape[0] == 1 and original_audio.shape[
        2] == model_args.n_codebooks, original_audio.shape
    logging.info(
        f"original audio length: {original_audio.shape[1]} codec frames, which is {original_audio.shape[1] / decode_config['codec_sr']:.2f} sec.")

    # forward
    stime = time.time()
    if decode_config['sample_batch_size'] <= 1:
        logging.info(f"running inference with batch size 1")
        concat_frames, gen_frames = model.inference_tts(
            text_tokens.to(device),
            text_tokens_lens.to(device),
            original_audio[..., :model_args.n_codebooks].to(device),  # [1,T,8]
            top_k=decode_config['top_k'],
            top_p=decode_config['top_p'],
            temperature=decode_config['temperature'],
            stop_repetition=decode_config['stop_repetition'],
            kvcache=decode_config['kvcache'],
            silence_tokens=eval(decode_config['silence_tokens']) if type(decode_config['silence_tokens']) == str else
            decode_config['silence_tokens']
        )  # output is [1,K,T]
    else:
        logging.info(
            f"running inference with batch size {decode_config['sample_batch_size']}, i.e. return the shortest among {decode_config['sample_batch_size']} generations.")
        concat_frames, gen_frames = model.inference_tts_batch(
            text_tokens.to(device),
            text_tokens_lens.to(device),
            original_audio[..., :model_args.n_codebooks].to(device),  # [1,T,8]
            top_k=decode_config['top_k'],
            top_p=decode_config['top_p'],
            temperature=decode_config['temperature'],
            stop_repetition=decode_config['stop_repetition'],
            kvcache=decode_config['kvcache'],
            batch_size=decode_config['sample_batch_size'],
            silence_tokens=eval(decode_config['silence_tokens']) if type(decode_config['silence_tokens']) == str else
            decode_config['silence_tokens']
        )  # output is [1,K,T]
    logging.info(f"inference on one sample take: {time.time() - stime:.4f} sec.")

    logging.info(
        f"generated encoded_frames.shape: {gen_frames.shape}, which is {gen_frames.shape[-1] / decode_config['codec_sr']} sec.")

    # for timestamp, codes in enumerate(gen_frames[0].transpose(1,0)):
    #     logging.info(f"{timestamp}: {codes.tolist()}")
    # decode (both original and generated)
    concat_sample = audio_tokenizer.decode(
        [(concat_frames, None)]  # [1,T,8] -> [1,8,T]
    )
    gen_sample = audio_tokenizer.decode(
        [(gen_frames, None)]
    )
    # Empty cuda cache between runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # return
    return concat_sample, gen_sample


