import torch
from faster_whisper import WhisperModel

MODEL_PATH = "./models/whisper-ct2"
_asr_model = None

def get_asr_model():
    global _asr_model
    if _asr_model is None:
        print("加载 ASR 模型...")
        _asr_model = WhisperModel(
            MODEL_PATH,
            device="cuda",
            compute_type="int8_float16",
            cpu_threads=4,
            num_workers=1
        )
        print("✅ ASR 加载完成")
    return _asr_model

def transcribe_audio(file_path: str) -> str:
    """转写音频，返回完整文本"""
    model = get_asr_model()
    segments, info = model.transcribe(
        file_path,
        beam_size=5,
        vad_filter=True,
        language="zh"
    )
    full_text = "".join(seg.text for seg in segments)
    return full_text

def release_asr_model():
    """释放 ASR 显存"""
    global _asr_model
    if _asr_model is not None:
        del _asr_model
        _asr_model = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print("ASR 显存已释放")