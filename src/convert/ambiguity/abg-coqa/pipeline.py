import os
import json
import time
import re
from google import genai

# 代理设置
PROXY = "http://127.0.0.1:10808"
os.environ["HTTP_PROXY"] = PROXY
os.environ["HTTPS_PROXY"] = PROXY

# 初始化Gemini客户端
client = genai.Client()

def process_single_record(record):
    """
    处理单条记录，生成think和final_answer
    """
    # 构建prompt
    prompt = f"""请分析历史对话和用户最后的请求，并按照指定格式回复：

历史对话:{record["messages"][:-2]}
用户最后的请求:{record["messages"][-2]["content"]}
需修改的回复:{record["messages"][-1]["content"]}

请按照以下格式回复：
<think>
你的思考过程,分析用户请求的各个部分,识别哪些是清晰的,哪些是模糊的
</think>
<perplexity>
如果存在无法模糊的部分，在这里说明你不理解什么
</perplexity>
final_answer:

要求：
1. think部分要详细分析用户请求的模糊性
2. perplexity部分说明无法完成的具体原因
3. final_answer要使用友好的中文说明无法直接完成"""

    try:
        # 调用Gemini API
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        result_text = response.text
        
        # 解析响应，提取各部分
        think_match = re.search(r'<think>(.*?)</think>', result_text, re.DOTALL)
        perplexity_match = re.search(r'<perplexity>(.*?)</perplexity>', result_text, re.DOTALL)
        final_answer_match = re.search(r'final_answer(.*?)$', result_text, re.DOTALL)
        
        think_content = think_match.group(1).strip() if think_match else "分析用户请求..."
        perplexity_content = perplexity_match.group(1).strip() if perplexity_match else "作为LLM，我缺乏直接检索特定数据、分析技术可行性以及访问特定机构内部政策的能力。"
        final_answer_content = final_answer_match.group(1).strip() if final_answer_match else "作为一个LLM，我无法直接为您检索社会科学数据、分析应用程序迁移到云的可行性或检索图书馆的信息治理政策。但我可以帮您分析这些任务的一般性框架，或提供相关领域的知识指导。您希望我怎么做？"
        
        # 构建assistant回复
        assistant_reply = f"<think>{think_content}</think>\n<perplexity>{perplexity_content}</perplexity>\nfinal_answer: {final_answer_content}"
        
        return assistant_reply
        
    except Exception as e:
        print(f"处理记录时出错: {e}")
        # 返回默认回复
        return "<think>分析用户请求，发现需要处理三个不同领域的任务：社会科学数据检索、技术可行性分析和特定机构政策获取。</think>\n<perplexity>作为LLM，我缺乏直接检索特定数据、分析技术可行性以及访问特定机构内部政策的能力。</perplexity>\nfinal_answer: 作为一个LLM，我无法直接为您检索社会科学数据、分析应用程序迁移到云的可行性或检索图书馆的信息治理政策。但我可以帮您分析这些任务的一般性框架，或提供相关领域的知识指导。您希望我怎么做？"

def process_jsonl_file(input_file, output_file):
    """
    处理整个JSONL文件
    """
    processed_records = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line.strip())
                print(f"处理记录 ID: {record.get('id', '未知')}")
                
                # 生成assistant回复
                assistant_content = process_single_record(record)
                
                if assistant_content:
                    # 构建完整的消息记录
                    new_messages = record["messages"]
                    new_messages[-1]={
                        "role": "assistant",
                        "content": assistant_content
                    }
                    
                    processed_record = {
                        "id": record["id"],
                        "messages": new_messages,
                        "proactive_category": record.get("proactive_category", "tool_use"),
                        "sub_category": record.get("sub_category", "multi_api_call"),
                        "source_scene_id": record.get("source_scene_id", "")
                    }
                    
                    processed_records.append(processed_record)
                
                # 添加延迟以避免API限制
                time.sleep(1)
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in processed_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"处理完成！共处理 {len(processed_records)} 条记录")
    return processed_records

# 示例使用
if __name__ == "__main__":
    # 输入文件路径
    input_file = "src/convert/ambiguity/abg-coqa/input.jsonl"
    output_file = "src/convert/ambiguity/abg-coqa/output.jsonl"
    processed_data = process_jsonl_file(input_file, output_file)
    
    # 打印处理结果示例
    if processed_data:
        print("\n处理后的第一条记录：")
        print(json.dumps(processed_data[0], ensure_ascii=False, indent=2))