
# PDF,markdown,txt -> csv (三元组)

import os
import argparse
import PyPDF2
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import sys
sys.path.append("./src/api")
from gpt_api import construct_knowledge_graph_by_LLM

# 加载环境变量
load_dotenv()

def extract_text_from_pdf(pdf_path):
    """ 从 PDF 文件提取文本 """
    try:
        with open(pdf_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in tqdm(pdf_reader.pages, desc="提取 PDF 文本"):
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
        return text.strip()
    except Exception as e:
        print(f"❌ 读取 PDF 文件失败: {e}")
        return None

def extract_text_from_md(md_path):
    """ 从 Markdown 文件提取文本 """
    try:
        with open(md_path, "r", encoding="utf-8") as md_file:
            return md_file.read().strip()
    except Exception as e:
        print(f"❌ 读取 Markdown 文件失败: {e}")
        return None

def extract_text_from_txt(txt_path):
    """ 从 TXT 文件提取文本 """
    try:
        with open(txt_path, "r", encoding="utf-8") as txt_file:
            return txt_file.read().strip()
    except Exception as e:
        print(f"❌ 读取 TXT 文件失败: {e}")
        return None

def process_file(file_path, output_csv):
    """ 处理文件并生成知识图谱 CSV """
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return
    
    ext = file_path.split('.')[-1].lower()
    text = None

    if ext == "pdf":
        text = extract_text_from_pdf(file_path)
    elif ext == "md":
        text = extract_text_from_md(file_path)
    elif ext == "txt":
        text = extract_text_from_txt(file_path)
    else:
        print(f"❌ 不支持的文件格式: {ext}")
        return

    if not text:
        print("⚠️ 未能提取任何文本，知识图谱无法生成。")
        return

    print("✅ 文本提取成功，正在生成知识图谱...")
    output_path = construct_knowledge_graph_by_LLM(text, output_csv=output_csv)

    if output_path:
        print(f"✅ 知识图谱已生成: {output_path}")
    else:
        print("❌ 知识图谱生成失败！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从 PDF、Markdown、TXT 提取文本并生成知识图谱 CSV")
    parser.add_argument("--file", type=str, required=True, help="输入文件路径（支持 PDF、MD、TXT）")
    parser.add_argument("--output", type=str, default="knowledge_graph.csv", help="输出 CSV 文件名")
    
    args = parser.parse_args()

    process_file(args.file, args.output)

# python file2csv.py --file ./test/example.pdf --output ./test/output.csv
