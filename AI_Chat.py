import os
import streamlit as st
from volcenginesdkarkruntime import Ark
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 设置 API Key
api_key = "your api_key"
client = Ark(api_key=api_key)

# 初始化对话历史
if 'messages' not in st.session_state:
    st.session_state.messages = []

# 初始化学习目标、时间安排和历史学习记录
if 'learning_goal' not in st.session_state:
    st.session_state.learning_goal = ""
if 'time_schedule' not in st.session_state:
    st.session_state.time_schedule = ""
if 'learning_history' not in st.session_state:
    st.session_state.learning_history = []

# 定义一个函数来调用大模型
def call_model(prompt):
    try:
        # 合并历史消息和当前用户输入
        messages = [
            {"role": "system", "content": "你是牛马AI，是由小冯开发的 AI 人工智能助手"}
        ] + st.session_state.messages + [{"role": "user", "content": prompt}]
        completion = client.chat.completions.create(
            model="your model key",
            messages=messages,
        )
        return completion.choices[0].message.content
    except Exception as e:
        logging.error(f"调用大模型时发生错误: {e}")
        return f"发生错误: {e}"

# 智能学习计划生成
def generate_learning_plan(learning_goal, time_schedule):
    prompt = f"请根据学习目标 '{learning_goal}' 和时间安排 '{time_schedule}' 生成一个个性化的学习计划。"
    return call_model(prompt)

# 动态内容推荐
def recommend_learning_materials():
    try:
        # 从本地知识库获取基础内容
        file_path = 'D:/AI_CHAT/knowledge_base.txt'
        base_materials = []
        if os.path.exists(file_path):
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.split_documents(documents)
            base_materials = [doc.page_content for doc in texts]

        # 使用大模型生成补充学习材料
        if st.session_state.learning_goal:
            ai_prompt = f"""基于学习目标 '{st.session_state.learning_goal}'，请生成一份补充学习材料，
            包含以下内容：
            1. 相关的基础知识点
            2. 实践建议
            3. 常见问题和解决方案
            请确保内容简洁且实用。"""
            
            ai_generated_material = call_model(ai_prompt)
            base_materials.append(ai_generated_material)

        if not base_materials:
            return []

        # 使用 TfidfVectorizer 进行文本向量化
        vectorizer = TfidfVectorizer()
        doc_vectors = vectorizer.fit_transform(base_materials)
        
        # 获取历史记录
        history_text = " ".join(st.session_state.learning_history)
        if not history_text:
            return base_materials[:3]
            
        # 计算相似度并返回最相关的内容
        query_vector = vectorizer.transform([history_text])
        similarities = cosine_similarity(query_vector, doc_vectors)[0]
        top_k = 3
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        return [base_materials[i] for i in top_indices]
        
    except Exception as e:
        logging.error(f"推荐学习材料时出现错误: {e}")
        print(f"推荐学习材料时出现错误: {e}")
        return []

# 检索增强生成功能
def retrieval_augmented_generation(query):
    relevant_docs = recommend_learning_materials()
    if not relevant_docs:
        context = "没有找到相关的参考资料。"
    else:
        context = "\n\n".join(relevant_docs)

    prompt = f"""请基于以下参考资料回答问题。
    
    {context}
    
    问题：{query}
    
    请提供详细的解释，如果可能的话，给出代码示例。如果参考资料中没有相关信息，请基于你的知识给出准确的回答。"""
    return call_model(prompt)

# 创建 Streamlit 应用
st.title("牛马AI - 个性化学习助手")

# 输入学习目标和时间安排
st.session_state.learning_goal = st.text_input("请输入你的学习目标", st.session_state.learning_goal)
st.session_state.time_schedule = st.text_input("请输入你的时间安排", st.session_state.time_schedule)

# 生成学习计划
if st.button("生成学习计划"):
    if st.session_state.learning_goal and st.session_state.time_schedule:
        plan = generate_learning_plan(st.session_state.learning_goal, st.session_state.time_schedule)
        st.write("个性化学习计划:")
        st.write(plan)
        # 将学习计划添加到对话历史
        st.session_state.messages.append({"role": "user", "content": f"基于学习目标：{st.session_state.learning_goal}，时间安排：{st.session_state.time_schedule} 生成学习计划"})
        st.session_state.messages.append({"role": "assistant", "content": f"个性化学习计划:\n{plan}"})

# 动态内容推荐
if st.button("推荐学习材料"):
    materials = recommend_learning_materials()
    if materials:
        st.write("推荐的学习材料:")
        # 将推荐材料添加到对话历史
        combined_materials = "推荐的学习材料:\n" + "\n".join(materials)
        st.session_state.messages.append({"role": "assistant", "content": combined_materials})
        for material in materials:
            st.write(material)
    else:
        st.write("未找到相关学习材料。")
        st.session_state.messages.append({"role": "assistant", "content": "未找到相关学习材料。"})

# 创建侧边栏，并添加清空历史记录按钮
# 将清空按钮放在侧边栏顶部
if st.sidebar.button("清空历史记录"):
    st.session_state.messages = []
    st.session_state.learning_history = []
    st.experimental_rerun()

# 创建对话历史展开面板
with st.sidebar.expander("对话历史", expanded=True):
    # 显示对话历史
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.write(f"你: {message['content']}")
        else:
            st.write(f"牛马AI: {message['content']}")
