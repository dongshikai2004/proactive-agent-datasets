import json
import os
import re 

# --- 配置 ---
INPUT_FILE = "data/coqa_abg_train.json" # 您提供的输入文件路径
OUTPUT_DIR = "dataset/proactive_annotations_from_coqa_abg_all_with_story"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    story_content = coref_item.get("story", "") # 获取故事内容
    history_turns = coref_item.get("history_turns", [])
    target_turn = coref_item["target_turn"]
    has_clarification = bool(coref_item.get("clarification_turn"))

    # --- 构建对话历史 ---
    messages = []
    # 1. 添加故事内容作为第一条用户消息
    if story_content:
        messages.append({"role": "user", "content": story_content})

    # 2. 添加历史对话
    for turn in history_turns:
         # 从 turn 中提取问题和答案来重构对话
         # 模拟：问题来自用户，答案来自助手
         messages.append({"role": "user", "content": turn["question"]})
         messages.append({"role": "assistant", "content": turn["answer"]})

    # 3. 添加 target_turn 的问题
    messages.append({"role": "user", "content": target_turn["question"]})

    # --- 根据是否有澄清回合构建助手的回复和相关字段 ---
    final_answer_content = target_turn["answer"] # 默认使用 target_turn 的答案

    if has_clarification:
        # 有澄清回合
        clarification_data = coref_item["clarification_turn"]
        clarification_question = clarification_data["question"]
        
        # 助手的回复是澄清问题
        assistant_reply = f"<perplexity>{clarification_question}</perplexity>"
        messages.append({"role": "assistant", "content": assistant_reply})
        
        # 最终答案是澄清问题本身
        final_answer_content = clarification_question
        
        proactive_category = "clarification"
        sub_category = "contextual_ambiguity" 
        uncertainty_type = None
        requires_tool = False
        thinking_process = {
            "intent_understanding": f"助手识别到用户的问题 '{target_turn['question']}' 存在指代不明或歧义，可能指向多个对象或概念。",
            "self_reflection": {
                "information_check": "助手分析当前对话历史和故事内容，发现用户问题中的指代对象不明确。",
                "knowledge_check": "助手知道根据上下文，问题的答案可能指向多个候选。",
                "tool_check": "此任务为语言理解，无需外部工具。"
            }
        }
    else:
        # 无澄清回合，直接回答
        # 助手的回复是 target_turn 的答案，无需 perplexity 标签
        assistant_reply = target_turn["answer"]
        messages.append({"role": "assistant", "content": assistant_reply})
        
        proactive_category = "direct_answer" # 新增类别
        sub_category = coref_item.get("ambiguity", "non_ambiguous") # 使用原数据的 ambiguity 或默认
        uncertainty_type = None
        requires_tool = False
        thinking_process = {
            "intent_understanding": f"助手理解了用户的问题 '{target_turn['question']}'，并根据上下文找到了答案。",
            "self_reflection": {
                "information_check": "助手确认当前对话历史和故事内容足以回答问题。",
                "knowledge_check": f"助手找到了问题 '{target_turn['question']}' 的答案。",
                "tool_check": "此任务为语言理解，无需外部工具。"
            }
        }

    # --- 构建最终训练项 ---
    # 使用顺序编号作为 ID
    safe_id = f"converted_item_{new_id:05d}" # 例如: converted_item_00000, converted_item_00001
    
    proactive_item = {
        "id": safe_id, # 使用顺序编号 ID
        "messages": messages, # 包含故事、历史对话和 target_turn 问题
        "proactive_category": proactive_category,
        "sub_category": sub_category,
        "uncertainty_type": uncertainty_type,
        "requires_tool": requires_tool,
        "thinking_process": thinking_process,
        "final_answer": final_answer_content, # 最终的答案内容
        "source_dataset_id": coref_item["id"], # 记录来源
        "original_coref_data": coref_item # 可选：保留原始数据引用
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
    converted_count = 0
    for i, item in enumerate(coref_data_list):
        # 再次检查 item 是否为字典 (来自 data 列表)
        if not isinstance(item, dict):
             print(f"  - 跳过 'data' 列表中索引 {i} 处的非字典项目: {type(item)}")
             continue
        # 传递顺序编号 (i) 作为 new_id
        proactive_item = convert_coref_to_proactive_item(item, i)
        if proactive_item:
            # 使用顺序编号 ID 作为文件名
            output_path = os.path.join(OUTPUT_DIR, f"{proactive_item['id']}.json")
            try:
                with open(output_path, "w", encoding="utf-8") as f_out:
                    json.dump(proactive_item, f_out, indent=2, ensure_ascii=False)
                converted_count += 1
                print(f"  - Converted and saved: {output_path}")
            except OSError as e:
                print(f"  - 保存文件失败 (可能因文件名问题): {output_path}, Error: {e}")
                # 可以选择跳过这个项目或尝试其他处理方式

    print(f"Conversion complete! {converted_count} items saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()