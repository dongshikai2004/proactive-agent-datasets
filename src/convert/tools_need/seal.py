import json
import sys

def load_api_descriptions(tools_file_path):
    """Loads API descriptions and parameters from tools.jsonl."""
    api_descriptions = {}
    try:
        with open(tools_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tool_data = json.loads(line)
                api_name = tool_data.get("api_name")
                description = tool_data.get("api_description", "")
                if api_name:
                    api_descriptions[api_name] = description
    except FileNotFoundError:
        print(f"Warning: Tools file '{tools_file_path}' not found. Using empty map.", file=sys.stderr)
    except json.JSONDecodeError as e:
        print(f"Error parsing tools file '{tools_file_path}': {e}", file=sys.stderr)
    return api_descriptions

def convert_to_perplexity_training_format(input_file_path, output_file_path, tools_file_path):
    """
    Converts a JSONL file with API call information into a dialogue training format
    where the assistant first expresses perplexity about its lack of capability,
    including details about required parameters, then provides an empty clarifying response.

    Args:
        input_file_path (str): Path to the input JSONL file.
        output_file_path (str): Path to the output JSONL file.
        tools_file_path (str): Path to the tools JSONL file.
    """
    api_description_map = load_api_descriptions(tools_file_path)

    with open(input_file_path, 'r', encoding='utf-8') as infile, \
         open(output_file_path, 'w', encoding= 'utf-8') as outfile:

        for line_num, line in enumerate(infile, start=1):
            try:
                line = line.strip()
                if not line:
                    continue

                original_data = json.loads(line)

                # --- 1. Extract Information from Original Data ---
                query_text = original_data.get("query", "")
                api_calls = original_data.get("calling", [])
                scene_id = original_data.get("id", f"converted_line_{line_num}")

                # --- 2. Determine Category and Tool Requirement ---
                # Assume all these queries require tools based on the example structure
                proactive_category = "tool_use"
                sub_category = "direct_api_call" # Or "multi_api_call" if multiple calls are present
                if len(api_calls) > 1:
                    sub_category = "multi_api_call"

                # --- 3. Identify the Required Tool/API Names and Details ---
                required_apis = [call.get("api", "Unknown API") for call in api_calls]
                required_capabilities_details = []
                for api_name in set(required_apis): # Use set to avoid duplicates
                    api_desc = api_description_map.get(api_name)
                    required_capabilities_details.append(api_desc)
                
                # --- 5. Construct Messages with Perplexity ---
                user_message = {"role": "user", "content": query_text}
                
                # Perplexity message - includes parameter details
                assistant_content = f"<think></think>\n<perplexity>As an LLM, I lack the capability to directly {', and '.join(required_capabilities_details)}. I would need access to specific tools or APIs to fulfill this request.</perplexity>\n"
                assistant_message = {"role": "assistant", "content": assistant_content}

                messages = [user_message, assistant_message]

                # --- 6. Create Final Output Dictionary ---
                output_dict = {
                    "id": scene_id,
                    "messages": messages,
                    "proactive_category": proactive_category,
                    "sub_category": sub_category,
                    "source_scene_id": scene_id
                }

                # --- 7. Write to Output File ---
                outfile.write(json.dumps(output_dict, ensure_ascii=False) + '\n')

            except json.JSONDecodeError:
                print(f"Error: Could not parse JSON on line {line_num}: {line.strip()}", file=sys.stderr)
            except KeyError as e:
                print(f"Error: Missing expected key {e} on line {line_num}", file=sys.stderr)
            except Exception as e:
                print(f"Unexpected error processing line {line_num}: {e}", file=sys.stderr)

# --- Example Usage ---
if __name__ == "__main__":
    input_path = "data/Seal-Tools_Dataset/train.jsonl"  # Replace with your input file path
    tools_path = "data/Seal-Tools_Dataset/tool.jsonl"  # Replace with your tools file path
    output_path = "dataset/capability_limitation/converted_perplexity_training_data.jsonl" # Replace with your desired output file path

    convert_to_perplexity_training_format(input_path, output_path, tools_path)
    print(f"Perplexity conversion complete. Output saved to {output_path}")