import os
import pickle
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
import re

class DoubanEmbeddingProcessor:
    def __init__(self):
        model_name = os.path.abspath(os.path.join(__file__, "..", "..", "bge-small-zh-v1.5"))
        self.model_name = model_name

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu', 'local_files_only': True},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.text_splitter = CharacterTextSplitter(
            chunk_size=500, chunk_overlap=50, separator="\n"
        )
        
    def load_douban_data(self, file_path: str) -> List[str]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                texts = [line.strip() for line in content.split('\n') if line.strip()]
            return texts
        except FileNotFoundError:
            print(f"文件 {file_path} 未找到")
            return []
        except Exception as e:
            print(f"读取文件时发生错误: {e}")
            return []
    
    def create_documents(self, texts: List[str]) -> List[Document]:
        documents = []
        for i, text in enumerate(texts):
            movie_name = self._extract_movie_name(text)
            doc = Document(
                page_content=text,
                metadata={"source": "douban", "movie_name": movie_name, "index": i}
            )
            documents.append(doc)
        return documents
    
    def _extract_movie_name(self, text: str) -> str:
        try:
            match = re.search(r'《([^》]+)》', text)
            return match.group(1) if match else "未知电影"
        except:
            return "未知电影"
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        try:
            return FAISS.from_documents(documents=documents, embedding=self.embeddings)
        except Exception as e:
            print(f"创建向量存储时发生错误: {e}")
            return None
    
    def save_vector_store(self, vector_store: FAISS, save_path: str):
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            vector_store.save_local(save_path)
            print(f"向量存储已保存到: {save_path}")
        except Exception as e:
            print(f"保存向量存储时发生错误: {e}")
    
    def save_embeddings(self, embeddings: List[List[float]], texts: List[str], save_path: str):
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            data = {'embeddings': embeddings, 'texts': texts, 'model_name': self.model_name}
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"Embeddings已保存到: {save_path}")
        except Exception as e:
            print(f"保存embeddings时发生错误: {e}")
    
    def process(self, input_file: str, vector_store_path: str, embeddings_path: str):
        """执行embedding处理（不含hash检查）"""
        print(f"开始处理文件: {input_file}")
        print(f"使用模型: {self.model_name}")
        
        texts = self.load_douban_data(input_file)
        if not texts:
            print("没有加载到任何数据")
            return False
        print(f"成功加载 {len(texts)} 条数据")
        
        documents = self.create_documents(texts)
        print(f"创建 {len(documents)} 个Document对象")
        
        vector_store = self.create_vector_store(documents)
        if vector_store:
            self.save_vector_store(vector_store, vector_store_path)
        
        try:
            embeddings = self.embeddings.embed_documents([doc.page_content for doc in documents])
            self.save_embeddings(embeddings, texts, embeddings_path)
        except Exception as e:
            print(f"保存embeddings时发生错误: {e}")
        
        print("处理完成！")
        return True