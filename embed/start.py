import os
import sys

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from hash_check import check_hash_changed, update_hash_config
from embedding import DoubanEmbeddingProcessor

def main():
    # 路径配置
    input_file = os.path.join(current_dir, "douban.txt")
    config_path = os.path.join(current_dir, "embedding_config.json")
    vector_store_path = os.path.join(current_dir, "vector_store")
    embeddings_path = os.path.join(current_dir, "embeddings", "douban_embeddings.pkl")
    
    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 文件不存在 {input_file}")
        return
    
    # 检查hash是否变化
    changed, current_hash = check_hash_changed(input_file, config_path)
    
    if not changed:
        print("文件未变化，跳过处理")
        return
    
    print("检测到文件变化，开始重新生成embedding...")
    
    # 执行处理
    processor = DoubanEmbeddingProcessor()
    success = processor.process(input_file, vector_store_path, embeddings_path)
    
    # 成功后更新hash
    if success:
        update_hash_config(config_path, current_hash, processor.model_name)
        print(f"配置已更新: {config_path}")

if __name__ == "__main__":
    main()