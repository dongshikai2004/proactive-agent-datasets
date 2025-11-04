import os
import json
import time
import re
from google import genai
import pandas as pd
from google.genai import types
from typing import Dict, Any, Optional, Tuple
from openai import OpenAI

SYS_PROMPT_PROACTIVE_JUDGE = r"""
You are a QA classification assistant that analyzes question–answer pair to determine whether a proactive self-assessment step is needed for a model.

Your task is to decide whether the given QA pair requires the model to perform a proactive capability or boundary awareness check.

You are given:
- A **question** asked by a user.
- A **ground truth (GT) answer** that is correct or authoritative.

You must judge whether the question–answer pair falls inside or outside the model’s normal knowledge/ability boundary.

---

### Decision Logic

1. If the question is **clearly answerable** using stable, factual, or general knowledge (e.g., math, definitions, history, or reasoning that does not depend on time, privacy, or external data),
   → it is **within normal capability** → `need_proactive = false`, `proactive_category = safe_to_answer`.

2. If the question:
   - Involves information **beyond the model’s likely knowledge cutoff** (e.g., “What happened in 2026?”),
   - Depends on **real-time or external tools** (search, code execution, API calls),
   - Is **ambiguous or incomplete**, requiring clarification,
   - Is **restricted or sensitive** (e.g., personal data, medical, financial, private content),
   → then it **requires proactive boundary recognition** → `need_proactive = true`.

---

### Output Format

You must output a single JSON object in the following structure:

{
  "need_proactive": true or false,
  "proactive_category": "<one of: knowledge_gap | clarification_needed | out_of_scope | safe_to_answer>",
  "reason": "<short explanation for your decision>"
}

Guidelines:
- Keep the reasoning concise (1–3 sentences).
- Always output **only JSON**, no extra commentary.
- Assume the model’s knowledge cutoff is around mid-2024 and it cannot access the internet or real-time data.
"""

USER_PROMPT_PROACTIVE_JUDGE = r"""
You are given a question and its verified ground truth (GT) answer.  
Decide whether this QA pair requires a proactive boundary check based on the model’s ability and knowledge scope.

Question:
{question}

Answer:
{answer}

Please analyze:
- Whether the question requires the model to perform self-assessment about its knowledge or capability (i.e., a proactive check).  
- If yes, identify the correct proactive category and explain briefly why.

Output a single JSON object in this exact format:

{{
  "need_proactive": true or false,
  "proactive_category": "<one of: knowledge_gap | clarification_needed | out_of_scope | safe_to_answer>",
  "reason": "<short explanation>"
}}
"""


USER_PROMPT_PROACTIVE = r"""
You are given a question and its verified ground truth (GT) answer.  
Your task is to generate the reasoning, proactive self-assessment, and final answer fields as described in the system instructions.

Question:
{question}

Ground Truth Answer:
{answer}

Please analyze carefully and produce your output in **exactly** the following JSON format:

{{
    "think": "<your generated internal reasoning>",
    "proactive": "<your generated proactive self-assessment>",
    "final_answer": "<the ground truth answer or a slightly adjusted version reflecting awareness>"
}}
"""


SYS_PROMPT_PROACTIVE = """
You are a reasoning and introspection assistant that enriches QA datasets by adding self-awareness, internal reasoning, and boundary recognition.  
Your purpose is to help a model understand its own knowledge scope and limitations.

You are given a user question and its corresponding ground truth (GT) answer — a verified, high-quality response that is considered correct or authoritative.  
Your task is NOT to judge or rewrite this answer, but to generate additional reasoning and self-assessment fields that describe *how a model would internally think and evaluate its own capability* when producing such an answer.

Specifically, you must:
- Generate a **think** section: a short, realistic reasoning process about how the model interprets the user's question and derives an understanding from available context.  
  This should include the model’s internal step-by-step analysis of the problem based on its existing knowledge, without repeating or giving the final answer itself.
- Generate a **proactive** section: the model’s explicit self-assessment of whether it has the ability or sufficient knowledge to answer the question accurately.  
  This section reflects the model’s *awareness of its own boundaries* — such as not being able to access external tools, handle real-time data, or confirm uncertain facts.  
  It should describe whether the model feels confident, uncertain, or unable to answer, and justify that judgment clearly.
- Generate the **final_answer**: a user-facing response that combines the reasoning and self-assessment above.  
  Since the provided answer is ground truth (GT), your final_answer should be based on it — you may rephrase slightly for naturalness or to incorporate self-awareness (e.g., uncertainty disclaimers), but do not alter its factual meaning.

Output must be valid JSON in the following structure:

{
    "think": "<your generated internal reasoning>",
    "proactive": "<your generated proactive self-assessment>",
    "final_answer": "<the ground truth answer or a slightly adjusted version reflecting awareness>"
}

Guidelines:
- “think” = concise internal reasoning or deduction steps derived from the question and available knowledge.
- “proactive” = explicit reflection on capability boundaries, awareness of uncertainty, or inability to access real-time/external data.
- “final_answer” = the final user-facing response, informed by reasoning and boundary awareness, built upon the given GT answer.
- The given answer is always *correct*; do not contradict or re-evaluate its content.
- Keep all fields natural, concise (2–5 sentences each), and coherent.
- Always output **only JSON**, no explanations or commentary outside of the JSON structure.
"""

SYS_PROMPT_ONLY_REASONING = """
You are a reasoning and introspection assistant that enriches QA datasets by adding internal reasoning and user-facing answers.  
Your purpose is to help a model understand how it reasons when answering, while respecting its knowledge scope and limitations.

You are given a user question and its corresponding ground truth (GT) answer — a verified, high-quality response that is considered correct or authoritative.  
Your task is NOT to judge or rewrite this answer, but to generate additional reasoning fields that describe *how a model would internally think* when producing such an answer.

Specifically, you must:
- Generate a **think** section: a short, realistic reasoning process about how the model interprets the user's question and derives an understanding from available context.  
  This should include the model’s internal step-by-step analysis of the problem based on its existing knowledge, without repeating or giving the final answer itself.
- Generate the **final_answer**: a user-facing response that combines the reasoning and appropriate caveats or uncertainty (if any).  
  Since the provided answer is ground truth (GT), your final_answer should be based on it — you may rephrase slightly for naturalness or to incorporate light self-awareness (e.g., “based on general knowledge”), but do not alter its factual meaning.

Output must be valid JSON in the following structure:

{
    "think": "<your generated internal reasoning>",
    "final_answer": "<the ground truth answer or a slightly adjusted version reflecting awareness>"
}

Guidelines:
- “think” = concise internal reasoning or deduction steps derived from the question and available knowledge.
- “final_answer” = the final user-facing response, informed by the reasoning above and built upon the given GT answer.
- The given answer is always *correct*; do not contradict or re-evaluate its content.
- Keep both fields natural, concise (2–5 sentences each), and coherent.
- Do not include any separate proactive self-assessment field (it has already been judged unnecessary upstream).
- Always output **only JSON**, no explanations or commentary outside of the JSON structure.
"""

USER_PROMPT_ONLY_REASONING = r"""
You are given a question and its verified ground truth (GT) answer.  
Your task is to generate the internal reasoning and the final user-facing answer fields as described in the system instructions.

Question:
{question}

Ground Truth Answer:
{answer}

Please analyze carefully and produce your output in **exactly** the following JSON format:

{{
    "think": "<your generated internal reasoning>",
    "final_answer": "<the ground truth answer or a slightly adjusted version reflecting awareness>"
}}
"""


class Evalutor:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt4.1",
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable must be set or api_key provided"
            )

        self.model = model

        # self.client = genai.Client(api_key=self.api_key)、
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self.client = OpenAI(**client_kwargs)

    def evaluate_single_judge(self, question: str, answer: str) -> Optional[dict]:
        try:
            user_prompt = USER_PROMPT_PROACTIVE_JUDGE.format(
                question=question, answer=answer
            )

            # response = self.client.models.generate_content(
            #     model=self.model,
            #     contents=types.Part.from_text(text=user_prompt),
            #     config=types.GenerateContentConfig(
            #         system_instruction=SYS_PROMPT_PROACTIVE_JUDGE,
            #         temperature=0.1,
            #         response_mime_type="application/json",
            #         max_output_tokens=2048,
            #         thinking_config=types.ThinkingConfig(thinking_budget=0),
            #     ),
            # )
            messages = [
                {"role": "system", "content": SYS_PROMPT_PROACTIVE_JUDGE},
                {"role": "user", "content": user_prompt},
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,  # Low temperature for consistency
                max_tokens=500,
                response_format={"type": "json_object"},
            )

            result_text = response.choices[0].message.content
            result_json = json.loads(result_text)
            return result_json

        except Exception as e:
            print(f"    !! API 调用失败: {e}")
            return None

    def evaluate_single(self, question: str, answer: str) -> Optional[dict]:
        try:
            user_prompt = USER_PROMPT_PROACTIVE.format(question=question, answer=answer)

            messages = [
                {"role": "system", "content": SYS_PROMPT_PROACTIVE},
                {"role": "user", "content": user_prompt},
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,  # Low temperature for consistency
                max_tokens=500,
                response_format={"type": "json_object"},
            )

            result_text = response.choices[0].message.content
            result_json = json.loads(result_text)
            return result_json

        except Exception as e:
            print(f"    !! API 调用失败: {e}")
            return None

    def evaluate_single_only_reasoning(
        self, question: str, answer: str
    ) -> Optional[dict]:
        try:
            user_prompt = USER_PROMPT_ONLY_REASONING.format(
                question=question, answer=answer
            )

            messages = [
                {"role": "system", "content": SYS_PROMPT_ONLY_REASONING},
                {"role": "user", "content": user_prompt},
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,  # Low temperature for consistency
                max_tokens=500,
                response_format={"type": "json_object"},
            )

            result_text = response.choices[0].message.content
            result_json = json.loads(result_text)
            return result_json

        except Exception as e:
            print(f"    !! API 调用失败: {e}")
            return None


def main():
    qa_data = pd.read_json(
        "./data/kass/data.jsonl",
        lines=True,
    )
    evaluator = Evalutor(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model="gpt-4.1",
    )

    qa_data["question_text"] = qa_data["questions"].apply(
        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None
    )
    qa_data = qa_data.drop_duplicates(subset=["question_text"]).dropna(
        subset=["question_text"]
    )
    print(qa_data.columns)

    sample_size = min(150, len(qa_data))
    qa_sample = qa_data.sample(n=sample_size, random_state=42)
    question_set = set()

    results = []
    # read from existing results to avoid duplicate processing
    with open(
        "proactive_knowledge_gap_judge_results.jsonl", "r", encoding="utf-8"
    ) as f:
        for line in f:
            if line.strip():
                record = json.loads(line.strip())
                existing_question = record["messages"][0]["content"]
                question_set.add(existing_question)
                results.append(record)

    for index, row in qa_sample.iterrows():
        question = row["questions"][0]
        if question in question_set:
            continue
        question_set.add(question)
        answer = row["answers"][0]

        print(f"\n[{index+1}/{len(qa_data)}] 开始评估问题: {question}")
        result = evaluator.evaluate_single_judge(question, answer)
        print(f"    -> 评估结果: {result}")

        result_detailed = None

        if result["need_proactive"]:
            result_detailed = evaluator.evaluate_single(question, answer)
            print(f"    -> 详细结果: {result_detailed}")
        else:
            result_detailed = evaluator.evaluate_single_only_reasoning(question, answer)
            print(f"    -> 仅推理结果: {result_detailed}")

        time.sleep(1)
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        results.append(
            {
                "id": row["idx"],
                "messages": messages,
                "answer": result_detailed,
                "proactive_category": result["proactive_category"],
                "sub_category": "qa",
                "reason_category": result["reason"],
            }
        )
        if len(results) % 10 == 0:
            print(f"已处理 {len(results)} 条记录")

        if len(results) >= 150:
            break

    output_file = "proactive_knowledge_gap_judge_results.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
