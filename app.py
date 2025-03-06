import streamlit as st
import networkx as nx
from pyvis.network import Network
import os
import sys

# 如果你的项目中还有其他 Python 文件放在 src/ 子目录，需要加:
sys.path.append("./src/")
from graph_construction import construct_knowledge_graph, graph_to_dict
from question_answering import retrieve_from_graph

# 如果有大模型API文件放在 src/api 中:
sys.path.append("./src/api")
from zhipu_api import ask_zhipu_with_kg


# ========== Streamlit 基础配置 ==========
st.set_page_config(
    page_title="LLM meets KG",
    layout="wide",  
    initial_sidebar_state="expanded",
)

# ========== 初始化 session_state ==========
if "uploaded_graph" not in st.session_state:
    st.session_state["uploaded_graph"] = {"graph": None, "graph_dict": None}

if "dialog_history" not in st.session_state:
    st.session_state["dialog_history"] = []


# ========== 辅助函数 ==========
def list_csv_files(directory="./csv/"):
    """列出指定目录下的所有 CSV 文件"""
    if not os.path.exists(directory):
        return []
    return [f for f in os.listdir(directory) if f.endswith(".csv")]

def draw_interactive_graph(graph):
    """使用 PyVis 绘制交互式知识图谱，返回 Network 对象"""
    net = Network(
        height="500px",
        width="100%",
        notebook=True,
        cdn_resources="remote",
        bgcolor="#ffffff",
    )
    for node in graph.nodes:
        net.add_node(node, label=str(node))
    for edge in graph.edges(data=True):
        source, target, attr = edge
        label = attr.get("label", "")
        net.add_edge(source, target, title=label, label=label)
    return net


# ========== 侧边栏：CSV 上传 & 加载知识图谱 ==========
st.sidebar.title("知识图谱管理")

csv_files = list_csv_files("./csv/")
selected_file = st.sidebar.selectbox(
    "选择已有 CSV 文件",
    ["无"] + csv_files if csv_files else ["无"]
)

uploaded_file = st.sidebar.file_uploader("或上传新的 CSV 文件", type=["csv"])
final_file = None

# 如果用户上传了新的文件，则优先使用
if uploaded_file:
    final_file = f"./csv/{uploaded_file.name}"
    with open(final_file, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"已保存文件: {uploaded_file.name}")
elif selected_file != "无":
    final_file = f"./csv/{selected_file}"

# 加载按钮
if final_file and st.sidebar.button("加载知识图谱"):
    try:
        with open(final_file, "rb") as f:
            graph = construct_knowledge_graph(f)
            graph_dict = graph_to_dict(graph)
            st.session_state["uploaded_graph"]["graph"] = graph
            st.session_state["uploaded_graph"]["graph_dict"] = graph_dict
        st.sidebar.success(f"已加载: {os.path.basename(final_file)}")
    except Exception as e:
        st.sidebar.error(f"加载失败: {e}")


# ========== 侧边栏：显示知识图谱 (默认展开) ==========
if st.session_state["uploaded_graph"]["graph"] is not None:
    # 显示 JSON (可折叠/展开参数可根据需要)
    try:
        graph_in_sidebar = st.session_state["uploaded_graph"]["graph"]
        interactive_graph = draw_interactive_graph(graph_in_sidebar)
        graph_html_path_side = "static/graph_sidebar.html"
        interactive_graph.show(graph_html_path_side)

        with open(graph_html_path_side, "r", encoding="utf-8") as f:
            side_html_code = f.read()

        # 要在侧边栏中显示 HTML，需要用: with st.sidebar:
        with st.sidebar:
            st.components.v1.html(side_html_code, height=400)
    except Exception as e:
        st.sidebar.error(f"侧栏图谱渲染失败: {e}")


# ========== 主区：标题 & 多轮对话 (默认折叠) ==========
st.title("多轮对话 Demo (LLM + 知识图谱)")

# ========== 主区：知识图谱可视化 (可折叠) ==========
with st.expander("查看知识图谱 (主区)", expanded=False):
    if st.session_state["uploaded_graph"]["graph"] is not None:
        st.json(st.session_state["uploaded_graph"]["graph_dict"], expanded=False)

        try:
            main_graph = st.session_state["uploaded_graph"]["graph"]
            interactive_graph_main = draw_interactive_graph(main_graph)
            graph_html_path_main = "static/graph_main.html"
            interactive_graph_main.show(graph_html_path_main)

            with open(graph_html_path_main, "r", encoding="utf-8") as f:
                html_code_main = f.read()
            st.components.v1.html(html_code_main, height=500)
        except Exception as e:
            st.error(f"主区图谱可视化失败: {e}")
    else:
        st.info("尚未加载知识图谱")
# 多轮对话放在一个 expander 里，默认折叠
with st.expander("多轮对话", expanded=True):
    # 清空对话按钮
    if st.button("清空对话记录"):
        st.session_state["dialog_history"] = []

    # 展示已有对话消息
    for msg in st.session_state["dialog_history"]:
        if hasattr(st, "chat_message"):
            # 若 Streamlit >= 1.22
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        else:
            # 老版本自定义
            if msg["role"] == "user":
                st.markdown(
                    f"<div style='background-color:#ECECEC; padding:8px; "
                    f"border-radius:6px; margin-bottom:5px; color:#000;'>"
                    f"<b>用户：</b>{msg['content']}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div style='background-color:#D5E8FF; padding:8px; "
                    f"border-radius:6px; margin-bottom:5px; color:#000;'>"
                    f"<b>系统：</b>{msg['content']}</div>",
                    unsafe_allow_html=True
                )

    # 输入框 (若 Streamlit < 1.23，就用 st.text_input)
    if hasattr(st, "chat_input"):
        user_text = st.chat_input("请输入问题...")
    else:
        user_text = st.text_input("请输入问题")

    # 当有新输入
    if user_text:
        # 追加到对话历史
        st.session_state["dialog_history"].append({
            "role": "user",
            "content": user_text
        })

        # 检索图谱
        g = st.session_state["uploaded_graph"]["graph"]
        if g:
            matches = retrieve_from_graph(g, user_text)
        else:
            matches = []

        matches = str(matches) if matches else "未找到匹配"
        print(matches)

        # 调用大模型
        answer = ask_zhipu_with_kg(matches, user_text)
        if not answer:
            answer = "对不起，大模型暂时无法回答"

        st.session_state["dialog_history"].append({
            "role": "assistant",
            "content": answer
        })

        # 可选：强制刷新
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()

