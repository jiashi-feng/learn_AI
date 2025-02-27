import os
import streamlit as st
from volcenginesdkarkruntime import Ark

# 设置 API Key
api_key = "3b5b5a22-4c24-4bbe-b496-7d88cb6fe6cf"
client = Ark(api_key=api_key)

# 初始化对话历史
if 'messages' not in st.session_state:
    st.session_state.messages = []

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
        return f"发生错误: {e}"

# 创建 Streamlit 应用
st.title("牛马AI")

# 创建侧边栏，并添加展开/收起功能
with st.sidebar.expander("对话历史", expanded=True):
    # 显示对话历史
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.write(f"你: {message['content']}")
        else:
            st.write(f"牛马AI: {message['content']}")

# 创建一个输入框，让用户输入问题
user_input = st.text_input("请输入你的问题", "常见的十字花科植物有哪些？")

# 创建一个按钮，用于触发模型调用
if st.button("提交"):
    # 保存用户输入到对话历史
    st.session_state.messages.append({"role": "user", "content": user_input})
    # 调用大模型并获取结果
    result = call_model(user_input)
    # 保存AI回复到对话历史
    st.session_state.messages.append({"role": "assistant", "content": result})
    # 显示结果
    st.write("模型回复:")
    st.write(result)