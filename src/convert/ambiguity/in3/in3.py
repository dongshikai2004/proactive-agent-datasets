# convert_vague_task_to_proactive.py
import json
import os

# --- 配置 ---
INPUT_FILE = "data/interaction_data_train.jsonl" # 您提供的输入 JSONL 文件路径
OUTPUT_FILE = "dataset/contextual_ambiguity/proactive_annotations_from_in3.jsonl" # 输出 JSONL 文件路径

# 确保输出目录存在
output_dir = os.path.dirname(OUTPUT_FILE)
os.makedirs(output_dir, exist_ok=True)

def convert_vague_task_to_proactive_items(vague_task_item, base_id):
    """
    将单个 vague task 数据项转换为多个主动对话训练项。
    每个助手的回复（无论 type）都会生成一个独立的样本。
    """
    # 检查 vague_task_item 是否为字典
    if not isinstance(vague_task_item, dict):
        print(f"  - 跳过非字典类型的项目: {type(vague_task_item)}")
        return []

    task = vague_task_item.get("task", "").strip()
    actions = vague_task_item.get("actions", [])
    category = vague_task_item.get("category", "unknown")

    if not actions:
        print(f"  - 跳过无 actions 的项目: {category}")
        return []

    proactive_items = []
    assistant_action_count = 0 # 计数助手的所有动作，用于生成 ID

    # 遍历 actions，找到所有助手的回复
    for i, action in enumerate(actions):
        role = action.get("role")
        content = action.get("content")
        thought = action.get("thought")
        action_type = action.get("type")

        if role == "assistant":
            # --- 构建当前助手回复的上下文历史 ---
            # 从 actions[0] 到 action[i-1]，构建一个对话历史
            context_messages = []
            for j in range(i):
                hist_action = actions[j]
                context_messages.append({
                    "role": hist_action.get("role"),
                    "content": hist_action.get("content")
                })

            # --- 构建当前样本的 messages ---
            # 1. 用户初始任务
            user_initial_message = task
            # 2. 上下文对话历史 (包含到当前助手回复之前的所有历史)
            # 3. 助手的当前回复

            messages = [
                {"role": "user", "content": user_initial_message},
                *context_messages, # 展开历史消息
            ]

            # --- 根据 action_type 设置类别和内容 ---
            if action_type == "New":
                # 这是一个澄清问题
                proactive_category = "ambiguity"
                sub_category = "detail_request"
                assistant_reply_content = f"<think>{thought}</think>\n<perplexity></perplexity>\nfinal_answer:{content}"

            else:
                # 这是一个总结或其他类型的直接回答
                proactive_category = "direct_answer"
                sub_category = "task_summary" if action_type == "summary" else "task_execution"
                assistant_reply_content =f"<think>{thought}</think>\nfinal_answer:{content}"
                

            messages.append({"role": "assistant", "content": assistant_reply_content})

            # --- 构建最终训练项 ---
            safe_id = f"vague_task_{base_id:05d}_assistant_action_{assistant_action_count}"

            proactive_item = {
                "id": safe_id,
                "messages": messages,
                "proactive_category": proactive_category,
                "sub_category": sub_category,
                "source_dataset_id": category,
            }

            proactive_items.append(proactive_item)
            assistant_action_count += 1

    return proactive_items

def main():
    print(f"Loading data from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"❌ 文件未找到: {INPUT_FILE}")
        return

    print(f"Found {len(lines)} lines in JSONL file. Converting...")

    # 打开输出文件以写入 JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        converted_count = 0
        for i, line in enumerate(lines):
            try:
                item = json.loads(line.strip())
            except json.JSONDecodeError:
                print(f"  - 跳过第 {i+1} 行，JSON 解析失败.")
                continue

            proactive_items = convert_vague_task_to_proactive_items(item, i)
            for proactive_item in proactive_items:
                f_out.write(json.dumps(proactive_item, ensure_ascii=False) + "\n")
                converted_count += 1

            if proactive_items and converted_count % 100 == 0: # 只在有输出时打印进度
                print(f"  - Converted {converted_count} items...")

    print(f"Conversion complete! {converted_count} items saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()