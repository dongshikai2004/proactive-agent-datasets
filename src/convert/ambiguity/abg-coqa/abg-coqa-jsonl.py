# convert_coref_to_proactive.py
import json
import os
import re 

# --- 配置 ---
INPUT_FILE = "data/coqa_abg_train.json" # 您提供的输入文件路径
OUTPUT_FILE = "dataset/contextual_ambiguity/proactive_annotations_from_coqa_abg_all_with_story.jsonl" # 输出 JSONL 文件路径

# 确保输出目录存在
output_dir = os.path.dirname(OUTPUT_FILE)
os.makedirs(output_dir, exist_ok=True)

def clean_filename(filename):
    """
    清理文件名，移除或替换不安全的字符。
    """
    # 替换不安全的字符为空下划线或直接移除
    cleaned = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # 移除可能导致问题的前导或尾随空格和点
    cleaned = cleaned.strip('. ')
    # 限制文件名长度（可选，防止过长）
    max_length = 200
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length]
    return cleaned

def convert_coref_to_proactive_item(coref_item, new_id):
    """
    将单个 coref 数据项转换为主动对话训练项。
    """
    # 检查 coref_item 是否为字典 (现在应该在 data 列表内部了)
    if not isinstance(coref_item, dict):
        print(f"  - 跳过非字典类型的项目: {type(coref_item)}")
        return None

    story_content = coref_item.get("story", "").strip() # 获取故事内容并去除首尾空白
    history_turns = coref_item.get("history_turns", [])
    target_turn = coref_item["target_turn"]
    has_clarification = bool(coref_item.get("clarification_turn"))

    # --- 构建对话历史 ---
    messages = []

    # 1. 处理第一条用户消息：将 story 与第一个问题合并
    if history_turns:
        # 如果有历史对话，将 story 与 history_turns[0] 合并
        first_question = history_turns[0]["question"]
        if story_content:
            combined_first_user_message = f"According to the story:\n\n{story_content}\n\nAnswer the following question:\n\n{first_question}"
        else:
            # 如果没有 story，直接使用第一个问题
            combined_first_user_message = first_question

        messages.append({"role": "user", "content": combined_first_user_message})
        # 添加第一个问题对应的助手回答
        messages.append({"role": "assistant", "content": history_turns[0]["answer"]})

        # 2. 添加剩余的历史对话 (从第二个开始，因为第一个已经处理过了)
        for turn in history_turns[1:]:
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": "<think>"+turn["rationale"]+"</think>"+turn["answer"]})

        # 3. 添加 target_turn 的问题
        messages.append({"role": "user", "content": target_turn["question"]})

    else:
        # 如果没有历史对话，将 story 与 target_turn 合并作为第一条用户消息
        if story_content:
            combined_first_user_message = f"According to the story:\n\n{story_content}\n\nAnswer the following question:\n\n{target_turn['question']}"
        else:
            # 如果没有 story，直接使用 target_turn 的问题
            combined_first_user_message = target_turn["question"]

        messages.append({"role": "user", "content": combined_first_user_message})



    if has_clarification:
        # 有澄清回合
        clarification_data = coref_item["clarification_turn"]
        clarification_question = clarification_data["question"]
        
        # 助手的回复是澄清问题
        assistant_reply = f"<think></think>\n<perplexity></perplexity>\nfinal_answer:{clarification_question}"
        messages.append({"role": "assistant", "content": assistant_reply})
        
        proactive_category = "clarification"
        sub_category = "coreference_ambiguity" # 或根据 ambiguity 字段判断 "contextual_ambiguity"
    else:
        # 无澄清回合，直接回答
        # 助手的回复是 target_turn 的答案，无需 perplexity 标签
        assistant_reply = "<think></think>\nfinal_answer:"+target_turn["answer"]
        messages.append({"role": "assistant", "content": assistant_reply})
        
        proactive_category = "direct_answer" # 新增类别
        sub_category = coref_item.get("ambiguity", "non_ambiguous") # 使用原数据的 ambiguity 或默认

    # --- 构建最终训练项 ---
    # 使用顺序编号作为 ID
    safe_id = f"converted_item_{new_id:05d}" # 例如: converted_item_00000, converted_item_00001
    
    proactive_item = {
        "id": safe_id, # 使用顺序编号 ID
        "messages": messages, # 包含合并了 story 的第一条消息和后续独立消息
        "proactive_category": proactive_category,
        "sub_category": sub_category,
        "source_dataset_id": coref_item["id"], # 记录来源
    }

    return proactive_item

def main():
    print(f"Loading data from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            full_data = json.load(f) # 加载整个 JSON
    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析错误: {e}")
        print(f"  请检查文件 {INPUT_FILE} 的格式是否为有效的 JSON。")
        return
    except FileNotFoundError:
        print(f"❌ 文件未找到: {INPUT_FILE}")
        return

    # 检查加载的数据是否为字典，并包含 'data' 键
    if not isinstance(full_data, dict) or "data" not in full_data:
        print(f"❌ 加载的数据不是预期的字典格式或缺少 'data' 键: {type(full_data)}")
        print(f"  请确保 {INPUT_FILE} 文件内容形如 {{'version': '...', 'data': [...]}}")
        return

    # 提取 'data' 列表
    coref_data_list = full_data["data"]

    # 检查 data 键对应的值是否为列表
    if not isinstance(coref_data_list, list):
        print(f"❌ 'data' 键对应的值不是列表类型: {type(coref_data_list)}")
        print(f"  请确保 {INPUT_FILE} 文件中的 'data' 字段是一个 JSON 数组，例如 [...]")
        return

    print(f"Found {len(coref_data_list)} items in 'data' field. Converting...")

    # 打开输出文件以写入 JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        converted_count = 0
        for i, item in enumerate(coref_data_list):
            # 再次检查 item 是否为字典 (来自 data 列表)
            if not isinstance(item, dict):
                print(f"  - 跳过 'data' 列表中索引 {i} 处的非字典项目: {type(item)}")
                continue
            # 传递顺序编号 (i) 作为 new_id
            proactive_item = convert_coref_to_proactive_item(item, i)
            if proactive_item:
                # 将单个 JSON 对象写入文件，并以换行符分隔
                f_out.write(json.dumps(proactive_item, ensure_ascii=False) + "\n")
                converted_count += 1
                # 可选：打印进度
                if converted_count % 1000 == 0:
                    print(f"  - Converted {converted_count} items...")

    print(f"Conversion complete! {converted_count} items saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()