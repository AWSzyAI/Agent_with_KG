import streamlit as st
import networkx as nx
from pyvis.network import Network
import os
import sys

sys.path.append("./src/")
from graph_construction import construct_knowledge_graph, graph_to_dict
from question_answering import retrieve_from_graph

# 同理，如果不需要这两个 API 或函数名不同，可注释或替换
sys.path.append("./src/api")
# from gpt_api import send_request_to_api
from zhipu_api import ask_zhipu_with_kg


# ====================== Streamlit 配置 ======================
st.set_page_config(page_title="LLM meets KG", layout="wide")

# 如果 Session State 不存在，则初始化
if "uploaded_graph" not in st.session_state:
    st.session_state["uploaded_graph"] = {"graph": None, "graph_dict": None}

# 存储所有对话轮次消息的列表
if "dialog_history" not in st.session_state:
    st.session_state["dialog_history"] = []


# 自定义 CSS 用于简单美化
st.markdown(
    """
    <style>
    /* 背景色可自行调整 */
    body {
        background-color: #F0F2F6;
    }
    /* 用户消息、AI 消息的气泡 */
    .message-user {
        background-color: #ECECEC;
        padding: 8px;
        border-radius: 10px;
        margin-bottom: 5px;
        color: #000000;
    }
    .message-assistant {
        background-color: #D5E8FF;
        padding: 8px;
        border-radius: 10px;
        margin-bottom: 15px;
        color: #000000;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def list_csv_files(directory="./csv/"):
    """列出指定目录下的所有 CSV 文件"""
    return [f for f in os.listdir(directory) if f.endswith(".csv")]

def draw_interactive_graph(graph):
    """使用 PyVis 绘制交互式知识图谱，返回 network.Network 对象"""
    net = Network(height="500px", width="100%", notebook=True, cdn_resources='remote')
    for node in graph.nodes:
        net.add_node(node, label=str(node))
    for edge in graph.edges(data=True):
        source, target, attr = edge
        label = attr.get("label", "")
        net.add_edge(source, target, title=label, label=label)
    return net


# ============ 三栏布局：左(上传&操作)，中(对话)，右(知识图谱) ============
col_left, col_center, col_right = st.columns([2, 3, 2])

# ------------------------- 左栏：上传 & 加载知识图谱 -------------------------
with col_left:
    st.header("上传与配置")

    # 列出 ./csv/ 下已有的 CSV
    csv_files = list_csv_files("./csv/")
    if csv_files:
        selected_file = st.selectbox("选择已有CSV文件 (可选)", ["无"] + csv_files)
    else:
        selected_file = "无"

    # 上传 CSV
    uploaded_file = st.file_uploader("或上传新 CSV 文件", type=["csv"])
    final_file = None

    # 先判断是否上传了新文件
    if uploaded_file:
        final_file = f"./csv/{uploaded_file.name}"
        with open(final_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"已保存文件: {uploaded_file.name}")
    # 若无上传，则看是否在下拉框中选择了已有文件
    elif selected_file != "无":
        final_file = f"./csv/{selected_file}"

    # 点击按钮，加载知识图谱
    if final_file and st.button("加载知识图谱"):
        try:
            with open(final_file, "rb") as f:
                graph = construct_knowledge_graph(f)
                graph_dict = graph_to_dict(graph)
                st.session_state["uploaded_graph"]["graph"] = graph
                st.session_state["uploaded_graph"]["graph_dict"] = graph_dict
                st.success(f"知识图谱已加载：{os.path.basename(final_file)}")
        except Exception as e:
            st.error(f"知识图谱加载失败：{e}")


# ------------------------- 中栏：多轮对话 -------------------------
with col_center:
    st.header("多轮对话")

    # 1. 先把所有历史消息显示出来
    for msg in st.session_state["dialog_history"]:
        if msg["role"] == "user":
            st.markdown(f"<div class='message-user'>用户: {msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='message-assistant'>系统: {msg['content']}</div>", unsafe_allow_html=True)

    # 2. 输入框
    user_input = st.text_input("请输入问题", key="user_input_text")
    if st.button("提交问题"):
        # 判空
        if not user_input.strip():
            st.warning("请输入问题后再提交")
        else:
            # 将用户输入添加到对话历史
            st.session_state["dialog_history"].append({"role": "user", "content": user_input})

            # 检索图谱
            g = st.session_state["uploaded_graph"]["graph"]
            if g:
                matches = retrieve_from_graph(g, user_input)
            else:
                matches = []

            # 调用大模型接口 (此处仅示例，若无此需求可改为其它逻辑)
            answer = ask_zhipu_with_kg(matches, user_input)

            # 添加模型回复到历史
            st.session_state["dialog_history"].append({"role": "assistant", "content": answer})

            # 在 Streamlit 中，可选让页面立即刷新以展示最新对话
            # 通常不加也能看到最新对话，因为脚本会从头执行
            # st.experimental_rerun()


# ------------------------- 右栏：可视化知识图谱 -------------------------
with col_right:
    st.header("知识图谱")

    # 如果已经加载了图谱，就展示
    if st.session_state["uploaded_graph"]["graph"] is not None:
        graph_dict = st.session_state["uploaded_graph"]["graph_dict"]
        st.json(graph_dict, expanded=False)

        try:
            # 生成 PyVis 图
            graph = st.session_state["uploaded_graph"]["graph"]
            interactive_graph = draw_interactive_graph(graph)
            interactive_graph.show("static/graph.html")
            # 嵌入 HTML
            st.components.v1.html(open("static/graph.html", "r").read(), height=500)
        except Exception as e:
            st.error(f"图谱可视化失败: {e}")
    else:
        st.info("尚未加载知识图谱。")