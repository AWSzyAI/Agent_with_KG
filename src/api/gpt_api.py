import os
import csv
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()

# 读取 API Key 和 Base URL
GPT_API_KEY = os.getenv("GPT_API_KEY")
GPT_BASE_URL = os.getenv("GPT_BASE_URL")

# 初始化 OpenAI 客户端
client = OpenAI(api_key=GPT_API_KEY, base_url=GPT_BASE_URL)
MODEL_NAME = "gpt-4-turbo"

def send_request_to_api(knowledge_graph_content, user_query, temperature=0.1, model=MODEL_NAME):
    """
    通过 API 向大模型发送请求，生成结果。
    :param knowledge_graph_content: 提示词内容，用于生成知识图谱。
    :param user_query: 用户查询内容。
    :param temperature: 控制输出多样性的参数。
    :param model: 使用的大语言模型。
    :return: 模型生成的响应。
    """
    messages = [
        {"role": "system", "content": "You are an AI assistant for building structured knowledge graphs."},
        {"role": "user", "content": knowledge_graph_content},
        {"role": "user", "content": user_query}
    ]

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return completion.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"API Request failed: {e}")
        return None

def construct_knowledge_graph_by_LLM(text, output_csv="knowledge_graph_LLM.csv"):
    """
    解析输入文本，通过大语言模型生成知识图谱并存储为 CSV 文件。
    :param text: 输入的文本内容。
    :param output_csv: 保存知识图谱的 CSV 文件路径。
    :return: 生成的 CSV 文件路径。
    """
    prompt = (
        "根据以下文本内容，提取有意义的知识图谱关系并按如下格式生成：\n"
        "\"Source\", \"Target\", \"Relation\"\n\n"
        f"文本内容：{text}\n\n"
        "注意：确保生成的关系有逻辑性，适用于知识图谱的构建。"
    )

    try:
        result = send_request_to_api(prompt, "生成知识图谱", temperature=0.2)
        if not result:
            return None

        lines = result.strip().split("\n")

        with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Source", "Target", "Relation"])
            for line in lines:
                if "," in line:
                    cleaned_line = [col.strip().strip('"') for col in line.split(",")]
                    writer.writerow(cleaned_line)

        return output_csv
    
    except Exception as e:
        print(f"Error while generating knowledge graph: {e}")
        return None

# Usage example
if __name__ == "__main__":
    sample_text = "量子计算是一种新型计算范式，它利用量子力学原理来处理信息。"
    csv_path = construct_knowledge_graph_by_LLM(sample_text)
    print(f"知识图谱已生成: {csv_path}")