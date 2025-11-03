import os
import json
import time
from openai import OpenAI
from datasets import load_dataset
from dotenv import load_dotenv

# 代理设置 (如果需要)
PROXY = "http://127.0.0.1:10808"
os.environ["HTTP_PROXY"] = PROXY
os.environ["HTTPS_PROXY"] = PROXY

# --- 配置 ---
# 从数据集中选取并处理的样本数量
NUM_SAMPLES_TO_GENERATE = 2
# 对每个问题的提问次数
NUM_TRIALS_PER_QUESTION = 10
# 判断模型具备知识的正确率阈值 (0.7 表示 70%)
ACCURACY_THRESHOLD = 0.7

MODEL_NAME = "qwen/qwen3-8b" 
OUTPUT_FILENAME = "dataset/epistemic_uncertainty/trivia_qa/adaptive/knowledge_assessment_from_qwen3-8b.json"


def is_answer_correct(model_answer, ground_truth_answer_obj):
    """
    检查模型的答案是否与标准答案或其任何别名匹配（不区分大小写）。
    """
    if not model_answer:
        return False
        
    ground_truth_value = ground_truth_answer_obj.get("value", "")
    ground_truth_aliases = ground_truth_answer_obj.get("aliases", [])
    
    possible_answers = {ground_truth_value.lower()}
    for alias in ground_truth_aliases:
        possible_answers.add(alias.lower())

    model_answer_lower = model_answer.lower()
    for answer in possible_answers:
        if answer in model_answer_lower:
            return True
            
    return False

def query_openrouter(client, question, model):
    """向 OpenRouter API 发送请求并获取模型的回答"""
    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "TriviaQA Knowledge Test (Multi-Trial)",
            },
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer the following question concisely and accurately in a few words.",
                },
                {"role": "user", "content": question},
            ],
            max_tokens=500,
            temperature=0.2 
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"    !! API 调用失败: {e}")
        return None

def main():
    # 步骤 1: 加载环境变量
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("错误：请在 .env 文件中设置 OPENROUTER_API_KEY")
        return

    # 步骤 2: 配置 OpenAI 客户端
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # 步骤 3: 加载 trivia_qa 数据集
    try:
        print(f"正在加载 {NUM_SAMPLES_TO_GENERATE} 个 'trivia_qa' 样本...")
        dataset = load_dataset("trivia_qa", "rc.nocontext", split=f'train[:{NUM_SAMPLES_TO_GENERATE}]')
        print("数据集加载成功。")
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        return

    # 步骤 4: 遍历数据集，对每个问题进行多次提问并评估
    final_dataset = []
    for i, item in enumerate(dataset):
        question = item["question"]
        question_id = item["question_id"]
        ground_truth = item["answer"]
        correct_answer_text = ground_truth['value']

        print(f"\n[{i+1}/{NUM_SAMPLES_TO_GENERATE}] 开始评估问题: {question}")
        print(f"  -> 标准答案: {correct_answer_text}")

        correct_count = 0
        # 对同一个问题进行多次提问
        for trial in range(NUM_TRIALS_PER_QUESTION):
            print(f"  - 第 {trial+1}/{NUM_TRIALS_PER_QUESTION} 次提问...")
            model_answer = query_openrouter(client, question, MODEL_NAME)
            
            if is_answer_correct(model_answer, ground_truth):
                correct_count += 1
                print(f"    -> 回答: '{model_answer}' (评估: 正确)")
            else:
                print(f"    -> 回答: '{model_answer}' (评估: 错误)")
            
            time.sleep(1) # 避免触发 API 速率限制

        # 计算正确率并进行判断
        accuracy = correct_count / NUM_TRIALS_PER_QUESTION
        print(f"  => 完成评估。正确率: {accuracy:.0%}")

        if accuracy >= ACCURACY_THRESHOLD:
            print("  => 结论: 模型具备相关知识。")
            think_content = f"<think>模型在 {NUM_TRIALS_PER_QUESTION} 次提问中答对了 {correct_count} 次，正确率达到 {accuracy:.0%}，超过了 {ACCURACY_THRESHOLD:.0%} 的阈值。可以认为它具备相关知识。</think>"
            perplexity_content = "<perplexity></perplexity>"
            # ** 关键改动：final_answer 返回正确答案 **
            final_answer_content = correct_answer_text
        else:
            print("  => 结论: 模型不具备相关知识。")
            think_content = f"<think>模型在 {NUM_TRIALS_PER_QUESTION} 次提问中仅答对了 {correct_count} 次，正确率只有 {accuracy:.0%}，低于 {ACCURACY_THRESHOLD:.0%} 的阈值。可以认为它不具备相关知识。</think>"
            perplexity_content = f"<perplexity>模型似乎缺乏关于 '{question}' 的相关知识。</perplexity>"
            final_answer_content = "模型不具备回答此问题的相关知识。"

        assistant_final_content = f"{think_content}\n{perplexity_content}\n{final_answer_content}"

        # 构建数据点
        data_point = {
            "id": question_id,
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": assistant_final_content}
            ],
            "proactive_category": "knowledge_gap",
            "sub_category": "",
            "source_id": ""
        }
        final_dataset.append(data_point)

    # 步骤 5: 保存结果
    with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, indent=4, ensure_ascii=False)

    print(f"\n处理完成！数据集已保存至 '{OUTPUT_FILENAME}'")
    if final_dataset:
        print("\n第一个样本预览:")
        print(json.dumps(final_dataset[0], indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()