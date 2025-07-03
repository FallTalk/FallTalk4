# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_data_files
import PyInstaller.config
PyInstaller.config.CONF['include_source_code'] = False


py3langid_hiddenimports = collect_submodules('py3langid')
fairseq_hiddenimports = collect_submodules('fairseq')
scipy_hiddenimports = collect_submodules('scipy')
TTS_hiddenimports = collect_submodules('TTS')
xformers_hiddenimports = collect_submodules('xformers')
pyannote_hiddenimports = collect_submodules('pyannote')
wordsegment_hiddenimports = collect_submodules('wordsegment')
g2p_en_hiddenimports = collect_submodules('g2p_en')
whisperx_hiddenimports = collect_submodules('whisperx')
inflect_hiddenimports = collect_submodules('inflect')
re_hiddenimports = collect_submodules('re')
torchaudio_hiddenimports = collect_submodules('torchaudio')
torio_hiddenimports = collect_submodules('torio')
ffmpeg_hiddenimports = collect_submodules('ffmpeg')
phonemizer_hiddenimports = collect_submodules('phonemizer')
triton_hiddenimports = collect_submodules('triton')
monotonic_align_hiddenimports = collect_submodules('monotonic_align')
transformers_hiddenimports = collect_submodules('transformers')
torchao_hiddenimports = collect_submodules('torchao')
flash_attn_hiddenimports = collect_submodules('flash_attn')
speechbrain_hiddenimports  = collect_submodules('speechbrain')
accelerate_hiddenimports  = collect_submodules('accelerate')


transformers_datas = collect_data_files('transformers', include_py_files=True)
scipy_datas = collect_data_files('scipy')
py3langid_datas = collect_data_files('py3langid')
fairseq_datas = collect_data_files('fairseq', include_py_files=True)
py3langid_datas = collect_data_files('py3langid')
xformers_datas = collect_data_files('xformers', include_py_files=True)
pyannote_datas = collect_data_files('pyannote', include_py_files=True)
wordsegment_datas = collect_data_files('wordsegment', include_py_files=True)
g2p_en_datas = collect_data_files('g2p_en', include_py_files=True)
whisperx_datas = collect_data_files('whisperx', include_py_files=True)
inflect_datas = collect_data_files('inflect', include_py_files=True)
re_datas = collect_data_files('re', include_py_files=True)
torchaudio_datas = collect_data_files('torio', include_py_files=True)
ffmpeg_datas  = collect_data_files('ffmpeg', include_py_files=True)
phonemizer_datas = collect_data_files('phonemizer', include_py_files=True)
config_datas = collect_data_files('config', include_py_files=True)
demucs_datas = collect_data_files('demucs', include_py_files=True)
torchao_datas = collect_data_files('torchao', include_py_files=True)
flash_attn_datas = collect_data_files('flash_attn', include_py_files=True)
speechbrain_datas = collect_data_files('speechbrain', include_py_files=True)
triton_datas = collect_data_files('triton', include_py_files=True)

def collect_module_data(module_path):
    data_files = []
    for root, dirs, files in os.walk(module_path):
        if '__pycache__' in dirs:
            dirs.remove('__pycache__')
        for file in files:
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, module_path)
            data_files.append((full_path, os.path.join('src', 'tts_engines', os.path.dirname(relative_path))))
    return data_files

# Manually specify the data files to be included
module_path = os.path.abspath('src/tts_engines/')
tts_engines_datas_modules = collect_module_data(module_path)

def collect_submodules(module_path, dest_name=None, exclude_exts=None):
    if exclude_exts is None:
        exclude_exts = {'.mp3', '.toml', '.ipynb'}

    data_files = []
    if dest_name is None:
        dest_name = os.path.basename(module_path)

    for root, dirs, files in os.walk(module_path):
        for skip in ('__pycache__', '.git', '.github', 'docs'):
            if skip in dirs:
                dirs.remove(skip)
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() in exclude_exts:
                continue

            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, module_path)
            target_path = os.path.join(dest_name, os.path.dirname(relative_path))
            data_files.append((full_path, target_path))

    print(f'Collected data files for {module_path}:')
    for src, dest in data_files:
        print(f'  {src} -> {dest}')

    return data_files

# Manually specify the data files to be included
third_party_datas = collect_submodules('third_party')


lightning_fabric_data = collect_data_files('lightning_fabric', include_py_files=True)
language_tags_data = collect_data_files('language_tags', include_py_files=True)
TTS_datas = collect_data_files('TTS', include_py_files=True)

a = Analysis(
    ['main.py'],
    pathex=['src',
    'third_party',
    ],
    binaries=[],
    datas=wordsegment_datas
    +pyannote_datas
    +xformers_datas
    +g2p_en_datas
    +TTS_datas
    +lightning_fabric_data
    +language_tags_data
    +fairseq_datas
    +py3langid_datas
    +whisperx_datas
    +inflect_datas
    +torchaudio_datas
    +re_datas
    +triton_datas
    +torchao_datas
    +flash_attn_datas
    +speechbrain_datas
    +transformers_datas
    +third_party_datas
    +phonemizer_datas
    +config_datas
    +demucs_datas
    +tts_engines_datas_modules,
    hiddenimports=scipy_hiddenimports
    +fairseq_hiddenimports
    +TTS_hiddenimports
    +py3langid_hiddenimports
    +xformers_hiddenimports
    +g2p_en_hiddenimports
    +whisperx_hiddenimports
    +pyannote_hiddenimports
    +torio_hiddenimports
    +inflect_hiddenimports
    +re_hiddenimports
    +phonemizer_hiddenimports
    +torchaudio_hiddenimports
    +wordsegment_hiddenimports
    +monotonic_align_hiddenimports
    +speechbrain_hiddenimports
    +accelerate_hiddenimports
    +transformers_hiddenimports
    +torchao_hiddenimports
    +flash_attn_hiddenimports
    +triton_hiddenimports,
    hookspath=[],
    hooksconfig={},
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    runtime_hooks=[],
    excludes=[],
    noarchive=True,
    optimize=0,
    module_collection_mode={
        'transformers': 'py',
    },
    parallel=32
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,  # This ensures that the executable is not bundled into a single file
    name='FallTalk',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # Enable UPX compression
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['resource\\falltalk.ico'],
)

# Add a COLLECT section to collect all the binaries and data files into a directory
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,  # Enable UPX compression
    upx_exclude=[],
    name='FallTalk',
)
