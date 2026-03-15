import hashlib
import json
import os

def get_file_hash(file_path: str) -> str:
    """计算文件MD5哈希"""
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def load_hash_config(config_path: str) -> dict:
    """加载hash配置文件"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_hash_config(config_path: str, data: dict):
    """保存hash配置文件"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def check_hash_changed(input_file: str, config_path: str) -> tuple[bool, str]:
    """
    检查文件是否变化
    Returns: (是否变化, 当前hash)
    """
    current_hash = get_file_hash(input_file)
    config = load_hash_config(config_path)
    old_hash = config.get("douban_hash")
    
    if old_hash == current_hash:
        return False, current_hash
    return True, current_hash

def update_hash_config(config_path: str, file_hash: str, model_name: str):
    """更新hash配置"""
    save_hash_config(config_path, {
        "douban_hash": file_hash,
        "model": model_name
    })