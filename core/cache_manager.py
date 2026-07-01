import hashlib
import os
import json

CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_key(file_path: str) -> str:
    """根据文件路径和修改时间生成唯一键"""
    stat = os.stat(file_path)
    key_str = f"{file_path}_{stat.st_mtime}"
    return hashlib.md5(key_str.encode()).hexdigest()

def get_cached_transcript(file_path: str):
    """返回缓存的转写文本，若无则返回 None"""
    key = get_cache_key(file_path)
    cache_file = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("transcript")
    return None

def save_cached_transcript(file_path: str, transcript: str):
    key = get_cache_key(file_path)
    cache_file = os.path.join(CACHE_DIR, f"{key}.json")
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump({"transcript": transcript}, f, ensure_ascii=False, indent=2)