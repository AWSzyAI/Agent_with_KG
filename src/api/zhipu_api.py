import os
from dotenv import load_dotenv
from zhipuai import ZhipuAI

# 加载环境变量
load_dotenv()

# 读取 API Key
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")

# 初始化 ZhipuAI 客户端
client = ZhipuAI(api_key=ZHIPU_API_KEY)
MODEL_NAME = "glm-4"

def ask_zhipu_with_kg(knowledge_graph_context, question, model=MODEL_NAME):
    """
    使用知识图谱内容和用户问题调用智谱AI
    :param knowledge_graph_context: 知识图谱上下文（文本描述）
    :param question: 用户输入的问题
    :param model: 智谱 AI 模型名称
    :return: 智谱AI的回答
    """
    messages = [
        {"role": "system", "content": "你是一名知识图谱专家，能够结合知识图谱回答用户问题。"},
        {"role": "user", "content": f"以下是知识图谱内容：\n{knowledge_graph_context}"},
        {"role": "assistant", "content": "我已经获取了知识图谱内容，请告诉我您的问题。"},
        {"role": "user", "content": question}
    ]
    print(messages)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"智谱 API 请求失败: {e}")
        return None

# Usage example
if __name__ == "__main__":
    sample_kg = "在知识图谱中，‘牛顿’与‘万有引力’的关系是‘提出’。"
    question = "牛顿与万有引力的关系是什么？"
    answer = ask_zhipu_with_kg(sample_kg, question)
    print(f"智谱AI回答: {answer}")