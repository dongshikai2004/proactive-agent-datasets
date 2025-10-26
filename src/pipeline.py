import os
import json
import time
import re
from google import genai

PROXY = "http://127.0.0.1:10808"
os.environ["HTTP_PROXY"] = PROXY
os.environ["HTTPS_PROXY"] = PROXY

client = genai.Client() 

# 其他配置
SCENES_FILE = "data/proactive_scenarios.json" # 输入场景定义
ANNOTATIONS_DIR = "dataset/proactive_annotations" # 输出训练集
PROMPT_TEMPLATE_FILE = "data/proactive_prompt_template.txt" # 提示词模板
DELAY = 1

# 读取提示词模板
with open(PROMPT_TEMPLATE_FILE, "r", encoding="utf-8") as f:
    PROMPT_TEMPLATE = f.read().strip()

def generate_annotation(scene):
    """
    生成单个场景的注释数据。
    """
    prompt = PROMPT_TEMPLATE.format(
        proactive_category=scene["category"],
        scenario_description=scene["description"],
        initial_user_query=scene.get("initial_user_query", ""),
        required_assistant_behavior=scene.get("required_behavior", ""),
        example_dialogue=scene.get("example_dialogue", ""),
        json_schema=json.dumps(PROACTIVE_JSON_SCHEMA, indent=2)
    )

    try:
        response = client.models.generate_content(
            model = 'gemini-2.0-flash', 
            contents=prompt,
        )

        # 提取生成的文本内容
        generated_text = response.text
        print(f"Raw Gemini response: {generated_text[:200]}...")

        # 尝试直接解析 JSON
        try:
            annotation_data = json.loads(generated_text)
        except json.JSONDecodeError:
            json_match = re.search(r'```json\s*(.*?)\s*```', generated_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = generated_text

            annotation_data = json.loads(json_str)

        final_answer = ""
        messages = annotation_data.get("messages", [])
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                final_answer = msg.get("content", "")
                break

        final_output = {
            "id": scene.get("id", f"dlg_{hash(str(scene)) % 10000}_turn0"), # 生成一个 ID
            "messages": messages,
            "proactive_category": scene["category"],
            "sub_category": annotation_data.get("sub_category", ""),
            "uncertainty_type": annotation_data.get("uncertainty_type", None),
            "requires_tool": annotation_data.get("requires_tool", False),
            "thinking_process": annotation_data.get("thinking_process", {}),
            "final_answer": final_answer,
            "source_scene_id": scene.get("id")
        }

        return final_output

    except json.JSONDecodeError as e:
        print(f"JSON 解析失败: {e}")
        return None
    except Exception as e:
        print(f"生成失败: {e}")
        return None

PROACTIVE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "messages": {
            "type": "array",
            "description": "包含用户和助手消息的对话历史。格式为 [{'role': 'user'/'assistant', 'content': '...'}, ...]",
            "items": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "enum": ["user", "assistant"]},
                    "content": {"type": "string"}
                },
                "required": ["role", "content"]
            }
        },
        "sub_category": {
            "type": "string",
            "description": "更细粒度的主动行为分类，如 'lexical_ambiguity', 'epistemic_uncertainty', 'capability_limitation' 等。"
        },
        "uncertainty_type": {
            "type": ["string", "null"],
            "description": "模型不确定性的类型，如 'epistemic', 'aleatoric', 'capability_limitation'，对于非 help_seeking 类别可为 null。"
        },
        "requires_tool": {
            "type": "boolean",
            "description": "对话是否最终需要调用外部工具来完成任务。"
        },
        "thinking_process": {
            "type": "object",
            "description": "模型内部的思考过程，包含意图理解、自我反思等。",
            "properties": {
                "intent_understanding": {"type": "string"},
                "self_reflection": {
                    "type": "object",
                    "properties": {
                        "information_sufficiency": {"type": "string"},
                        "knowledge_status": {"type": "string"},
                        "capability_status": {"type": "string"}
                    }
                }
            }
        }
    },
    "required": ["messages", "thinking_process", "sub_category"]
}


def main():
    try:
        with open(SCENES_FILE, "r", encoding="utf-8") as f:
            scenes = json.load(f).get("scenarios", [])
    except FileNotFoundError:
        print(f"输入文件未找到: {SCENES_FILE}")
        return
    except json.JSONDecodeError:
        print(f"输入文件 JSON 格式错误: {SCENES_FILE}")
        return

    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

    print(f"开始处理 {len(scenes)} 个场景...")
    for scene in scenes:
        scene_id = scene.get("id", "unknown_id")
        output_path = os.path.join(ANNOTATIONS_DIR, f"{scene_id}.json")

        if os.path.exists(output_path):
            print(f"已存在，跳过: {scene_id}")
            continue

        print(f"正在处理: {scene_id}")
        annotation = generate_annotation(scene)
        if annotation:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(annotation, f, indent=2, ensure_ascii=False)
            print(f"成功: {output_path}")
        else:
            print(f"失败: {scene_id}")

        time.sleep(DELAY)

    print("数据集构建完成")

if __name__ == "__main__":
    main()