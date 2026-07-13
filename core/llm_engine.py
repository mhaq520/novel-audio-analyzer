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

    # 修改4：截断改为头尾各取 1000 字
    max_len = 2000
    if len(text) > max_len:
        text = text[:1000] + "\n...(省略中段)...\n" + text[-1000:]

    prompt = f"""以下是一部小说的转写文本，请用一段话概括剧情，再列出5-8个行为关键词。

文本：{text}

摘要："""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    input_len = inputs.input_ids.shape[1]
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    def _generate(temperature):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=temperature,
                do_sample=True,
                repetition_penalty=1.15,
                no_repeat_ngram_size=5,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

    # 修改3：首次生成 + 失败检测 + 重试
    result = _generate(0.7)
    print("\n===== 模型原始输出 =====\n", result, "\n=========================\n")

    # 失败条件：含原文片段（未摘要）或 无分隔符
    if text[:100] in result or ("行为关键词" not in result and "关键词" not in result):
        print("⚠️ 首次生成异常，降温度重试...")
        result = _generate(0.3)
        print("\n===== 重试输出 =====\n", result, "\n=========================\n")

    summary = ""
    keywords = []

    for sep in ["行为关键词", "关键词"]:
        if sep in result:
            parts = result.split(sep, maxsplit=1)
            summary = parts[0].strip()

            kw_block = parts[1].strip()
            kw_block = re.sub(r'^[：:\s\n\r]*', '', kw_block)
            kw_block = kw_block.split('\n\n')[0] if '\n\n' in kw_block else kw_block

            # 修改2+5：逐行解析 + 去重 + 格式校验（含:或>10字跳过）
            found = []
            for line in kw_block.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                kw = re.sub(r'^\s*\d+\s*[.、)）\s]\s*', '', line)
                kw = kw.split('。')[0].strip()
                if kw and len(kw) <= 10 and '：' not in kw and ':' not in kw and kw not in found:
                    found.append(kw)

            # 修改2：行解析不足时按标点二次切分 + 去重
            if len(found) < 2:
                found = []
                for kw in re.split(r'[、，,\n]', kw_block):
                    kw = kw.strip()
                    if kw and len(kw) <= 10 and '：' not in kw and ':' not in kw and kw not in found:
                        found.append(kw)

            keywords = found
            break

    # 修改2：截断到 8 个
    if len(keywords) > 8:
        keywords = keywords[:8]

    if not summary:
        summary = result.strip()[:300]
    if not keywords:
        keywords = ["关键词提取失败"]

    return summary, keywords

def release_llm():
    global _llm_model, _tokenizer
    if _llm_model is not None:
        del _llm_model
        _llm_model = None
    if _tokenizer is not None:
        del _tokenizer
        _tokenizer = None
    torch.cuda.empty_cache()
    print("LLM 显存已释放")
