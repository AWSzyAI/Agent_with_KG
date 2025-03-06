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
        {
            "role": "system",
            "content": (
                "你是一名专业的知识图谱专家。"
                "你的可用信息仅限于用户提供的知识图谱内容以及对话上下文。"
                "请严格根据知识图谱进行回答，如果知识图谱中没有相关信息，请在回答中明确指出。"
            )
        },
        {
            "role": "user",
            "content": f"<知识图谱内容>\n{{knowledge_graph_context}}</知识图谱内容> {{question}}"
        }
    ]
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