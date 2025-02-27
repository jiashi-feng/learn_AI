from volcenginesdkarkruntime import Ark
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PythonTeachingRAG:
    def __init__(self):
        """初始化教学助手"""
        self.knowledge_base = []
        self.vector_index = None
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        print("初始化Python教学助手...")

    def add_document(self, text: str, title: str) -> None:
        """添加教学文档到知识库"""
        # 简单的文本分割：按段落分割
        paragraphs = text.split('\n\n')
        for p in paragraphs:
            if p.strip():
                self.knowledge_base.append({
                    'content': p.strip(),
                    'title': title
                })
        # 更新向量索引
        self._update_vector_index()
        print(f"已添加文档：{title}")

    def _update_vector_index(self):
        """更新向量索引"""
        embeddings = [self.model.encode(doc['content']) for doc in self.knowledge_base]
        embeddings = np.array(embeddings).astype('float32')
        self.vector_index = faiss.IndexFlatL2(embeddings.shape[1])
        self.vector_index.add(embeddings)

    def search_relevant_docs(self, query: str, top_k: int = 2) -> list:
        """搜索相关文档（向量检索）"""
        query_embedding = self.model.encode(query).reshape(1, -1).astype('float32')
        distances, indices = self.vector_index.search(query_embedding, top_k)
        return [self.knowledge_base[i] for i in indices[0]]

    def generate_answer(self, query: str, search: bool = True) -> str:
        """生成答案"""
        logging.info(f"处理问题：{query}")
        
        # 搜索相关文档
        relevant_docs = self.search_relevant_docs(query)
        
        if not relevant_docs:
            context = "没有找到相关的参考资料。"
        else:
            context = "\n\n".join([f"参考资料（来自{doc['title']}）：\n{doc['content']}" 
                                 for doc in relevant_docs])
        
        # 构建提示词
        if search:
            prompt = f"""作为Python教学助手，请基于以下参考资料回答问题。
            
    {context}
    
    问题：{query}
    
    请提供详细的解释，如果可能的话，给出代码示例。如果参考资料中没有相关信息，请基于你的知识给出准确的回答。"""
        else:
            prompt = query
        
        try:
            logging.info(f"final prompt: {prompt}")
            # 初始化豆包API客户端
            api_key = "3b5b5a22-4c24-4bbe-b496-7d88cb6fe6cf"
            client = Ark(api_key=api_key)
            # 调用豆包API生成答案
            completion = client.chat.completions.create(
                model="ep-20250226095516-cxq6t",
                messages=[
                    {"role": "system", "content": "你是豆包，是由字节跳动开发的 AI 人工智能助手"},
                    {"role": "user", "content": prompt},
                ],
            )
            return completion.choices[0].message.content
        except Exception as e:
            logging.error(f"生成答案时发生错误：{str(e)}")
            return f"生成答案时发生错误：{str(e)}"

def demo(search):
    """演示RAG系统的使用"""
    # 创建RAG系统
    rag = PythonTeachingRAG()
    
    # 添加示例文档
    python_basics = """
Python是一种高级编程语言，以其简洁的语法和强大的功能而闻名。
Python的设计哲学强调代码的可读性，其语法允许程序员用更少的代码表达概念。

Python是一种解释型语言，这意味着代码在运行时被直接解释执行，不需要预先编译。
这种特性使得开发过程更加快速和灵活。
"""

    python_functions = """
Python中的函数使用def关键字定义。函数可以接收参数并返回值。
基本语法如下：
def function_name(parameter1, parameter2):
    # 函数体
    return result

函数可以有默认参数值，这使得函数调用更加灵活。
例如：def greet(name="World"):
    print(f"Hello, {name}!")
"""
    
    rag.add_document(python_basics, "Python基础介绍")
    rag.add_document(python_functions, "Python函数教程")
    
    print("\n=== Python教学助手已就绪 ===")
    print("您可以询问任何Python相关的问题。")
    
    # 交互式问答
    while True:
        question = input("\n请输入您的问题（输入'退出'结束）：")
        if question.lower() == '退出':
            break
            
        answer = rag.generate_answer(question, search)
        print("\n答案：")
        print(answer)

if __name__ == "__main__":
    search = True
    demo(search)