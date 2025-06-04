import numpy as np
import json
from sentence_transformers import SentenceTransformer
from typing import List, Dict

# 加载 once 保持全局调用效率
_embedding_index = None
_metadata_index = None
_model = None

def load_index(
    embedding_path: str = "../../../Datasets/medreason_train_embeddings_bge_m3.npy",
    metadata_path: str = "../../../Datasets/medreason_train_metadata.jsonl",
    model_name: str = "BAAI/bge-m3"
):
    global _embedding_index, _metadata_index, _model

    if _embedding_index is None:
        print("Loading embeddings...")
        _embedding_index = np.load(embedding_path)

    if _metadata_index is None:
        print("Loading metadata...")
        _metadata_index = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                _metadata_index.append(json.loads(line))

    if _model is None:
        print("Loading embedding model...")
        # _model = SentenceTransformer(model_name)
        _model = SentenceTransformer(model_name, device='cpu') 

def get_retrieved_examples(
    question: str,
    options: Dict[str, str],  # 修复类型声明
    k: int = 5
) -> List[Dict]:
    """
    检索与给定问题最相似的K个训练样本
    """
    load_index()  # 确保索引和模型已加载

    # 格式化 options 为字符串
    options_str = "\n".join([f"{k}. {v}" for k, v in options.items()])

    # 构造查询文本
    query_text = question.strip() + "\n" + options_str.strip()
    query_vec = _model.encode([query_text], normalize_embeddings=True)

    # 相似度计算
    scores = np.dot(_embedding_index, query_vec.T).squeeze()

    # 获取 top-k
    top_k_indices = scores.argsort()[-k:][::-1]
    # # ✅ 插入打印检索得分与问题预览（调试用）
    # print(f"\n[Retrieval Debug Info]")
    # print(f"Query: {question[:160]}...")
    # for i in range(min(k, len(scores))):
    #     idx = top_k_indices[i]
    #     retrieved_q = _metadata_index[idx]["question"]
    #     sim = scores[idx]
    #     print(f"{i+1}: score={sim:.4f}, question={retrieved_q[:160]}")
    # print("-" * 40)


    similarity_threshold = 0.97
    results = []
    for idx in top_k_indices:
        if scores[idx] > similarity_threshold:
            continue  
        results.append(_metadata_index[idx])
        if len(results) == k:
            break

    return results

import numpy as np
import json
from sentence_transformers import SentenceTransformer
from typing import List, Dict

# 全局缓存变量（用于 COT embedding 检索）
_cot_embedding_index = None
_cot_metadata_index = None
_model = None  # 共享模型

def load_cot_index(
    embedding_path: str = "../../../Datasets/medreason_train_embeddings_cot_bge_m3.npy",
    metadata_path: str = "../../../Datasets/medreason_train_metadata_cot.jsonl",
    model_name: str = "BAAI/bge-m3"
):
    global _cot_embedding_index, _cot_metadata_index, _model

    if _cot_embedding_index is None:
        print("Loading COT embeddings...")
        _cot_embedding_index = np.load(embedding_path)

    if _cot_metadata_index is None:
        print("Loading COT metadata...")
        _cot_metadata_index = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                _cot_metadata_index.append(json.loads(line))

    if _model is None:
        print("Loading embedding model...")
        # _model = SentenceTransformer(model_name)
        _model = SentenceTransformer(model_name, device='cpu') 

def get_retrieved_cot_examples(
    reasoning_text: str,
    k: int = 5
) -> List[Dict]:
    """
    检索与给定 reasoning 最相似的 K 个训练样本。
    输入为一段推理文本，返回的是最相似的 CoT 样本。
    """
    load_cot_index()

    query_vec = _model.encode([reasoning_text.strip()], normalize_embeddings=True)
    scores = np.dot(_cot_embedding_index, query_vec.T).squeeze()
    top_k_indices = scores.argsort()[-k:][::-1]

    results = []
    for idx in top_k_indices:
        results.append(_cot_metadata_index[idx])
        if len(results) == k:
            break

    return results


def to_float(conf: str) -> float:
    return float(conf.strip().replace("%", "")) if conf else 0.0


def build_fewshot_prompt(
    retrieved_examples: List[Dict],
    target_question: str,
    target_options: Dict[str, str]
) -> str:
    """
    构建 few-shot CoT prompt，加入显式提示：模仿现有推理风格，生成当前问题的推理和答案。
    """
    option_letters = list(target_options.keys())
    format_hint = f"Thought: [your detailed step-by-step reasoning]\nConfidence: [Return a number with a percentage sign, e.g., 90%]\nAnswer: [One of {', '.join(option_letters)}]"

    # prompt = (
    #     "You are a medical expert assistant. Your task is to solve clinical multiple-choice questions.\n"
    #     "Below are several solved examples that show step-by-step reasoning and final answers.\n"
    #     "You should carefully analyze these examples, learn from their reasoning process, and apply similar thinking patterns to solve the new question.\n"
    #     "In your response, first provide a step-by-step Thought, then provide your final Answer.\n"
    #     f"You must strictly follow this format:\n'''{format_hint}'''\n\n"
    # )

    prompt = (
    "You are a medical expert assistant. Your task is to solve clinical multiple-choice questions.\n"
    "Below are several solved examples that show step-by-step reasoning and final answers.\n"
    "You should carefully analyze these examples, learn from their reasoning process, and apply similar thinking patterns to solve the new question.\n"
    "In your response, first provide a step-by-step Thought, then report your Confidence, and finally provide your final Answer.\n\n"
    "You must rate your confidence based on three criteria:\n"
    "1. How complete and clear your reasoning is.\n"
    "2. Whether the question provides enough information to support a conclusion.\n"
    "3. How familiar you are with the topic as a medical professional.\n\n"
    "Use the following scoring guidance:\n"
    "- 90–100%: Confident, well-supported reasoning and clear evidence.\n"
    "- 75–89%: Reasonable certainty with minor ambiguities.\n"
    "- 60–74%: Somewhat unsure, reasoning has gaps or question is vague.\n"
    "- <60%: Low confidence, incomplete reasoning or unclear information.\n\n"
    "Return a final confidence score in this format:\n"
    "Confidence: 87%\n\n"
    f"You must strictly follow this format:\n'''{format_hint}'''\n\n"
    )
    # prompt = (
    # "You are a medical expert assistant. Your task is to solve clinical multiple-choice questions.\n"
    # "Below are several solved examples that show step-by-step reasoning and final answers.\n"
    # "You should carefully analyze these examples, learn from their reasoning process, and apply similar thinking patterns to solve the new question.\n"
    # "In your response, first provide a step-by-step Thought, then report your Confidence, and finally provide your final Answer.\n\n"
    # "You must rate your confidence based on three criteria:\n"
    # "1. **Reasoning quality**: Is your step-by-step reasoning complete, logical, and free of major assumptions?"
    # "2. **Information sufficiency**: Does the question provide enough evidence to justify your conclusion?"
    # "3. **Domain familiarity**: How familiar are you with the topic based on your medical knowledge?"
    # "4. **Conclusion reliability**: How well does your final answer align with all key symptoms, lab findings, and clinical context? Would a medical expert agree that your diagnosis fully explains the case?"
    # "Use the following scoring guidance:"
    # "- **90–100%**: All criteria are fully met; confident in reasoning and conclusion is strongly supported."
    # "- **75–89%**: Mostly confident, minor uncertainty in either reasoning or conclusion."
    # "- **60–74%**: Some doubt due to incomplete reasoning or unclear fit with clinical presentation."
    # "- **Below 60%**: Significant uncertainty; possible gaps in reasoning or mismatch between conclusion and key findings."
    # "Return a final confidence score in this format:\n"
    # "Confidence: 87%\n\n"
    # f"You must strictly follow this format:\n'''{format_hint}'''\n\n"
    # )



    # Add few-shot examples
    # for example in retrieved_examples:
    for i, example in enumerate(retrieved_examples):
        q = example.get("question", "").strip()
        options_str = example.get("options", "").strip()
        reasoning = example.get("reasoning", "").strip()
        answer = example.get("answer", "").strip()

        if '.' in answer:
            answer_letter = answer.split('.')[0].strip()
        else:
            answer_letter = answer.strip()

        prompt += f"Example {i+1}:\n"
        prompt += f"Question:\n{q}\n"
        prompt += f"Options:\n{options_str}\n"
        prompt += f"Thought: {reasoning}\n"
        prompt += f"Answer: {answer_letter}\n\n"


    prompt += (
    f"Now, using a similar step-by-step reasoning process, solve the following question.\n"
    f"Question:\n{target_question.strip()}\n"
    f"Options:\n"
    )

    for k, v in target_options.items():
        prompt += f"{k}. {v.strip()}\n"

    prompt += "Thought:"

    return prompt


# def build_fewshot_prompt_no_confi(
#     retrieved_examples: List[Dict],
#     target_question: str,
#     target_options: Dict[str, str]
# ) -> str:
#     option_letters = list(target_options.keys())
#     format_hint = f"Thought: [your detailed step-by-step reasoning]\nAnswer: [One of {', '.join(option_letters)}]"

#     prompt = (
#         "You are a medical expert assistant. Your task is to solve clinical multiple-choice questions.\n"
#         "Below are several solved examples that show step-by-step reasoning and final answers.\n"
#         "You should carefully analyze these examples, learn from their reasoning process, and apply similar thinking patterns to solve the new question.\n"
#     )


#     for i, example in enumerate(retrieved_examples):
#         q = example.get("question", "").strip()
#         options_str = example.get("options", "").strip()
#         reasoning = example.get("reasoning", "").strip()
#         answer = example.get("answer", "").strip()

#         if '.' in answer:
#             answer_letter = answer.split('.')[0].strip()
#         else:
#             answer_letter = answer.strip()

#         prompt += f"Reference Example {i+1}:\n"
#         prompt += f"Question:\n{q}\n"
#         prompt += f"Options:\n{options_str}\n"
#         prompt += f"Thought: {reasoning}\n"
#         prompt += f"Answer: {answer_letter}\n\n"


#     prompt += (
#     f"#####################################################\n"
#     f"Now, solve the following question:\n"
#     f"Question:\n{target_question.strip()}\n"
#     f"Options:\n"
#     )

#     for k, v in target_options.items():
#         prompt += f"{k}. {v.strip()}\n"

#     # prompt += "Thought:"
#     prompt += (
#        "In your response, first provide a step-by-step Thought, then provide your final Answer.\n"
#         f"You must strictly follow this format:\n'''{format_hint}'''\n\n"
#         f"The new Question you need to analysis is:\n{target_question.strip()}\n"
#         )
#     for k, v in target_options.items():
#         prompt += f"{k}. {v.strip()}\n"
    

#     return prompt
def build_fewshot_prompt_no_confi(
    retrieved_examples: List[Dict],
    target_question: str,
    target_options: Dict[str, str]
) -> str:
    option_letters = list(target_options.keys())
    format_hint = f"Thought: [your detailed step-by-step reasoning]\nAnswer: [One of {', '.join(option_letters)}]"
    prompt = (
        "Your task is to solve one following clinical multiple-choice question:\n"
        f"Question:\n{target_question.strip()}\n"
        "Options:\n")
    for k, v in target_options.items():
        prompt += f"{k}. {v.strip()}\n"

    prompt += ("Below are solved examples showing detailed thinking processes.\n"
        "Each example includes two parts:\n"
        "1. **Finding Reasoning Paths**: Brainstorm possible paths to approach the question.\n"
        "2. **Reasoning Process**: Provide structured, step-by-step logical reasoning.\n"
        "\n"
        "⚠️ Please focus **only on learning the reasoning paths and structured thinking**.\n"
        "⚠️ For the new question, you must generate your own detailed Thought and strictly answer with a single letter (A/B/C/...).\n"
        "- Even if the information seems insufficient or ambiguous, you must select the best answer among the provided options.\n"
        "- You are not allowed to answer 'none', 'unknown', or leave the answer blank.\n"
        "- Always choose the most appropriate option based on the given context."
        "Provide your final answer by following this strict format:\n"
        f"'''\n{format_hint}\n'''\n"
    )
    for i, example in enumerate(retrieved_examples):

        q = example.get("question", "").strip()
        options_str = example.get("options", "").strip()
        reasoning = example.get("reasoning", "").strip()
        if "Reference Example" in reasoning:
            print(f"[Contaminated COT sample detected]: {reasoning[:100]}")

        prompt += f"\nReference Example {i+1}:\n"
        prompt += f"{q}\n"
        prompt += f"{options_str}\n"
        prompt += f"Thought:\n{reasoning}\n"

    return prompt

def build_format_prompt(output_s: str) -> str:
    prompt = (
        "You are tasked with reorganizing the following reasoning output into a strict structure without losing any information.\n\n"
        "Here is the original model output that needs reformatting:\n"
        f"'''{output_s}'''\n\n"
        "Your task:\n"
        "- Carefully retain all reasoning details from the original output without simplifying, summarizing, or omitting information.\n"
        "- If the original output lacks a clear step-by-step reasoning (CoT), create a reasonable and logically consistent Thought based on the available information.\n"
        "- You must produce an output following exactly this format:\n"
        "Thought: [The detailed step-by-step reasoning]\n"
        "Answer: [one of the option letters: A, B, C, D, etc.]\n\n"
        "Important:\n"
        "- Do not create new content beyond reorganizing or completing what is missing.\n"
        "- Always choose an answer, even if the original output is ambiguous or incomplete.\n"
        "- Never leave Thought or Answer blank.\n"
    )
    return prompt



def build_confidence_prompt(
    question: str,
    options: Dict[str, str],
    answer: str,
    thought: str,
) -> str:
    """
    构造 verifier agent 的评分 prompt，根据已有 reasoning 和 answer 来判断可靠性。
    """
    # 格式化 options
    options_str = "\n".join([f"{k}. {v.strip()}" for k, v in options.items()])

    # 构造问题部分
    prompt = (
        "A clinical AI agent has answered the following multiple-choice question:\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Options:\n{options_str}\n\n"
        f"The agent provided the following reasoning:\nThought:\n{thought.strip()}\n\n"
        f"Final Answer: {answer.strip()}\n\n"
    )

    prompt += (
        "You are a critical-thinking medical reviewer. "
        "Your task is to assign a reliability score from 1 to 5 based on how well the reasoning supports the answer.\n"
        "Use this scale:\n"
        "5 - Reasoning is complete, medically accurate, and fully supports the answer.\n"
        "4 - Mostly correct with minor issues, but the answer is still justified.\n"
        "3 - Reasoning has some issues or omissions, but partially supports the answer.\n"
        "2 - Reasoning is flawed or incomplete; answer is weakly supported.\n"
        "1 - Reasoning is incorrect or misleading; answer is not justified.\n\n"
        "Return your result in the format:\n"
        "Score: [1-5]"
    )

    return prompt

import re

def extract_score_from_verifier_output(output: str) -> int:
    """
    从 Verifier Agent 输出中提取评分分数（1~5）。
    输出格式预期为：Score: 4 或包含类似 'Score: 3' 的段落。
    """
    try:
        match = re.search(r"Score\s*:\s*([1-5])", output)
        if match:
            return int(match.group(1))
        else:
            return 0  # 如果没匹配到，返回0作为“无评分”处理
    except:
        return 0

# D:\BaiduNetdiskDownload\Agent\medagents-benchmark\baselines\MedAgents\wjh_cot.py