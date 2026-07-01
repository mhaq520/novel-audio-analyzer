import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_PATH = "./models/qwen2.5-7b"

_llm_model = None
_tokenizer = None


def load_llm():
    global _llm_model, _tokenizer
    if _llm_model is None:
        # 强制设置设备并打印信息
        torch.cuda.set_device(0)
        print(f"显存状态（加载前）: {torch.cuda.memory_summary()}")
        print("加载 LLM 4-bit 模型...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        _llm_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb_config,
            device_map="cuda:0",
            max_memory={0: "6GB"},
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        print(f"✅ LLM 加载完成，显存占用: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    return _llm_model, _tokenizer


def analyze_text(text: str):
    model, tokenizer = load_llm()

    max_len = 2000
    if len(text) > max_len:
        text = text[:max_len] + "..."

    prompt = f"""请根据以下小说剧本的转写文本，生成：
1. 剧情总结（故事主线与关键情节，200字以内）
2. 行为关键词（人物动作、事件标签，如"逃亡""对峙""密谋"，列出5-8个）

文本内容：
{text}

输出格式（必须严格遵守）：
摘要：...
关键词：词1, 词2, 词3, ...
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            # 移除 min_new_tokens，避免版本兼容问题
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n===== 模型原始输出 =====\n", result, "\n=========================\n")

    # 解析摘要（更宽容）
    summary = ""
    keywords = []
    lines = result.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith("摘要：") or line.startswith("摘要:"):
            summary = line.split("：")[-1].split(":")[-1].strip()
        elif line.startswith("关键词：") or line.startswith("关键词:"):
            kw_part = line.split("：")[-1].split(":")[-1].strip()
            keywords = [kw.strip() for kw in kw_part.split(",") if kw.strip()]

    # 如果仍然无效，取整个输出的前200字符作为摘要
    if not summary:
        summary = result[:200] if len(result) > 200 else result
        summary = summary.replace("摘要：", "").replace("摘要:", "").strip()
        if not summary:
            summary = "（模型未生成有效摘要，请检查音频内容）"

    if not keywords:
        keywords = ["关键词提取失败"]

    return summary, keywords

def release_llm():
    """释放 LLM 显存"""
    global _llm_model, _tokenizer
    if _llm_model is not None:
        del _llm_model
        _llm_model = None
    if _tokenizer is not None:
        del _tokenizer
        _tokenizer = None
    torch.cuda.empty_cache()
    print("LLM 显存已释放")