import streamlit as st
import networkx as nx
from pyvis.network import Network
import os
import sys

# 如果有自定义代码目录，可保留，否则注释掉
sys.path.append("./src/")
from graph_construction import construct_knowledge_graph, graph_to_dict
from question_answering import retrieve_from_graph

# 同理，若不需要请注释
sys.path.append("./src/api")
from gpt_api import send_request_to_api
from zhipu_api import ask_zhipu_with_kg


# 如果不存在，就初始化
if "uploaded_graph" not in st.session_state:
    st.session_state["uploaded_graph"] = {"graph": None, "graph_dict": None}

def list_csv_files(directory="./csv/"):
    """列出指定目录下的所有 CSV 文件"""
    return [f for f in os.listdir(directory) if f.endswith(".csv")]

def draw_interactive_graph(graph):
    """
    使用 PyVis 绘制交互式知识图谱，返回 PyVis 的 Network 对象
    """
    net = Network(height="500px", width="100%", notebook=True, cdn_resources='remote')
    
    for node in graph.nodes:
        net.add_node(node, label=str(node))
    for edge in graph.edges(data=True):
        source, target, attr = edge
        label = attr.get("label", "")
        net.add_edge(source, target, title=label, label=label)
    return net

def render_chat_history(chat_history):
    """
    渲染多轮对话历史，确保文字可见（浅色文字对深色背景）
    """
    for entry in chat_history:
        # 用户消息
        st.markdown(
            f"""
            <div style="background-color:#444444; color:#ffffff; padding:10px; border-radius:10px; margin-bottom:5px;">
                <strong>用户:</strong> {entry['user']}
            </div>
            """, 
            unsafe_allow_html=True
        )
        # 系统回复
        st.markdown(
            f"""
            <div style="background-color:#0D0D0D; color:#ffffff; padding:10px; border-radius:10px; margin-bottom:15px;">
                <strong>系统:</strong> {entry['assistant']}
            </div>
            """, 
            unsafe_allow_html=True
        )

def main():
    st.title("LLM meet KG")

    # 同样先确保 session_state 中的图对象存在
    if "uploaded_graph" not in st.session_state:
        st.session_state["uploaded_graph"] = {"graph": None, "graph_dict": None}
    # 如果你希望多轮对话从一开始就为空，可确保 dialog_history 存在
    if "dialog_history" not in st.session_state:
        st.session_state["dialog_history"] = []

    # ====== 文件上传与加载知识图谱 =======
    csv_files = list_csv_files()
    selected_file = None
    if csv_files:
        selected_file = st.selectbox("请选择服务器端的 CSV 文件（可选）", ["无"] + csv_files)

    # 支持用户上传 CSV 文件
    uploaded_file = st.file_uploader("或者上传文件（CSV 格式, 可通过 pdf2csv.py 生成）", type=["csv"])

    # 确定最终使用的文件
    final_file = None
    if uploaded_file:
        # 用户上传新文件，保存到 ./csv/ 后再使用
        final_file = f"./csv/{uploaded_file.name}"
        with open(final_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"已成功上传并保存：{uploaded_file.name}")
    elif selected_file and selected_file != "无":
        # 如果用户选择了已有文件
        final_file = f"./csv/{selected_file}"

    # 加载按钮
    if final_file and st.button("加载知识图谱"):
        try:
            with open(final_file, "rb") as f:
                graph = construct_knowledge_graph(f)
                graph_dict = graph_to_dict(graph)
                st.session_state["uploaded_graph"]["graph"] = graph
                st.session_state["uploaded_graph"]["graph_dict"] = graph_dict
                st.success(f"知识图谱已成功加载: {os.path.basename(final_file)}")

                # 可视化知识图谱 (第一次展示)
                st.json(graph_dict, expanded=False)
                interactive_graph = draw_interactive_graph(graph)
                try:
                    interactive_graph.show("static/graph.html")
                    st.components.v1.html(open("static/graph.html", "r").read(), height=500)
                except Exception as e:
                    st.error(f"图形显示失败: {e}")
        except Exception as e:
            st.error(f"知识图谱加载失败：{e}")

    # 如果上传了文件，且尚未构建过图，就尝试构建一次
    if uploaded_file:
        if st.session_state["uploaded_graph"]["graph"] is None:
            try:
                graph = construct_knowledge_graph(uploaded_file)
                graph_dict = graph_to_dict(graph)
                st.session_state["uploaded_graph"]["graph"] = graph
                st.session_state["uploaded_graph"]["graph_dict"] = graph_dict
                st.success("知识图谱已成功构建！")

                # 可视化知识图谱
                st.json(graph_dict, expanded=False)
                interactive_graph = draw_interactive_graph(graph)
                try:
                    interactive_graph.show("static/graph.html")
                    st.components.v1.html(open("static/graph.html", "r").read(), height=500)
                except Exception as e:
                    st.error(f"图形显示失败: {e}")
            except Exception as e:
                st.error(f"知识图谱构建失败：{e}")

    # ====== 如果已构建了图，就允许用户进行多轮对话 =======
    if st.session_state["uploaded_graph"]["graph"] is not None:
        st.subheader("智能对话")

        # 渲染已经存在的聊天记录
        render_chat_history(st.session_state["dialog_history"])

        # 输入框 (注意设置一个 key，否则刷新时会丢失)
        user_input = st.text_input("请输入问题", key="user_input_box")

        if st.button("提交") and user_input.strip():
            # 读取当前图
            graph = st.session_state["uploaded_graph"]["graph"]
            
            # 先在图里检索
            matches = retrieve_from_graph(graph, user_input)
            if matches:
                # 如果有匹配的三元组，就把它们可视化一下
                matches_graph = nx.DiGraph()
                for match in matches:
                    source, relation, target = match
                    matches_graph.add_edge(source, target, label=relation)

                interactive_matches_graph = draw_interactive_graph(matches_graph)
                interactive_matches_graph.show("static/graph.html")
                st.components.v1.html(open("static/graph.html", "r").read(), height=500)

            context = matches if matches else "未找到相关信息"

            # 调用大模型接口获取答案
            answer = ask_zhipu_with_kg(context, user_input)

            # 更新会话历史
            st.session_state["dialog_history"].append({"user": user_input, "assistant": answer})

            # 重新渲染对话历史
            render_chat_history(st.session_state["dialog_history"])

            # 手动清理输入框中的内容
            st.session_state["user_input_box"] = ""  # 这会让输入框清空

if __name__ == "__main__":
    main()