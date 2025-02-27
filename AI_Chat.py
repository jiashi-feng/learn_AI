import os
import streamlit as st
from volcenginesdkarkruntime import Ark
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 设置 API Key
api_key = "3b5b5a22-4c24-4bbe-b496-7d88cb6fe6cf"
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
            model="ep-20250226095516-cxq6t",
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
    file_path = 'D:/AI_CHAT/knowledge_base.txt'
    if not os.path.exists(file_path):
        logging.error(f"错误：文件 {file_path} 不存在。")
        print(f"错误：文件 {file_path} 不存在。")
        return []
    try:
        # 加载知识库，指定编码为 utf-8
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.from_documents(texts, embeddings)

        # 根据历史学习记录进行检索
        history_text = " ".join(st.session_state.learning_history)
        relevant_docs = docsearch.similarity_search(history_text)
        if not relevant_docs:
            logging.info("未找到相关学习材料。")
            print("未找到相关学习材料。")
        return [doc.page_content for doc in relevant_docs]
    except Exception as e:
        logging.error(f"加载文件 {file_path} 时出现错误: {e}")
        print(f"加载文件 {file_path} 时出现错误: {e}")
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

# 动态内容推荐
if st.button("推荐学习材料"):
    materials = recommend_learning_materials()
    if materials:
        st.write("推荐的学习材料:")
        for material in materials:
            st.write(material)
    else:
        st.write("未找到相关学习材料。")

# 创建侧边栏，并添加展开/收起功能
with st.sidebar.expander("对话历史", expanded=True):
    # 显示对话历史
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.write(f"你: {message['content']}")
        else:
            st.write(f"牛马AI: {message['content']}")

    # 添加清空历史记录按钮
    if st.button("清空历史记录"):
        st.session_state.messages = []
        st.session_state.learning_history = []
        st.experimental_rerun()

# 输入用户问题
user_input = st.text_input("请输入你的问题")

# 创建一个按钮，用于触发模型调用
if st.button("提交"):
    if user_input:
        # 保存用户输入到对话历史
        st.session_state.messages.append({"role": "user", "content": user_input})
        # 保存到历史学习记录
        st.session_state.learning_history.append(user_input)
        # 调用检索增强生成功能
        result = retrieval_augmented_generation(user_input)
        # 保存AI回复到对话历史
        st.session_state.messages.append({"role": "assistant", "content": result})
        # 显示结果
        st.write("模型回复:")
        st.write(result)