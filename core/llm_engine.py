import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_PATH = "./models/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored"

_llm_model = None
_tokenizer = None


def load_llm():
    global _llm_model, _tokenizer
    if _llm_model is None:
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

    prompt = f"""根据文本生成摘要和行为关键词。

文本：{text}

输出：
摘要："""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    input_len = inputs.input_ids.shape[1]
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    result = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    print("\n===== 模型原始输出 =====\n", result, "\n=========================\n")

    # 按行为关键词/关键词/关键点切分
    summary = ""
    keywords = []
    for sep in ["\n关键词", "\n行为关键词", "\n关键点"]:
        if sep in result:
            parts = result.split(sep, maxsplit=1)
            summary = parts[0].strip()
            # 取关键词行，去掉末尾垃圾字符
            kw_text = parts[1].split("\n")[0]
            kw_text = re.sub(r'^[：:]\s*', '', kw_text)
            kw_text = re.sub(r'\s*[\[<\]>].*$', '', kw_text)  # 清除垃圾后缀
            items = re.split(r'[、,，]\s*', kw_text)
            keywords = [kw.strip().rstrip("。") for kw in items if kw.strip() and len(kw.strip()) <= 20]
            break

    if not summary:
        summary = result.strip()[:300]

    if not keywords:
        match = re.search(r'(?:关键词|行为关键词)[：:]\s*(.+?)(?:\n|$)', result)
        if match:
            kw_text = match.group(1).strip()
            kw_text = re.sub(r'\s*[\[<\]>].*$', '', kw_text)
            items = re.split(r'[、,，]\s*', kw_text)
            keywords = [kw.strip().rstrip("。") for kw in items if kw.strip() and len(kw.strip()) <= 20]
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