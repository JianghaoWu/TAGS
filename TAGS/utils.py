from prompt_utils import *
from data_utils import *
import time
from tags import get_retrieved_examples, build_fewshot_prompt, get_retrieved_cot_examples, to_float, build_fewshot_prompt_no_confi,build_confidence_prompt, extract_score_from_verifier_output,build_format_prompt
from collections import defaultdict


def fully_decode(qid, realqid, question, options, gold_answer, handler, handler_eval, args, dataobj):
    start_time = time.time()

    question_domains, options_domains, question_analyses, option_analyses, syn_report, output = "", "", "", "", "", ""
    vote_history, revision_history, syn_repo_history = [], [], []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    if args.method == "base_direct":
        direct_prompt = get_direct_prompt(question, options)
        output, usage = handler.get_output_multiagent(user_input=direct_prompt, temperature=0, system_role="")
        total_prompt_tokens += usage.prompt_tokens
        total_completion_tokens += usage.completion_tokens
        ans, output = cleansing_final_output(output)
    elif args.method == "base_cot":
        cot_prompt = get_cot_prompt(question, options)
        output, usage = handler.get_output_multiagent(user_input=cot_prompt, temperature=0, system_role="")
        total_prompt_tokens += usage.prompt_tokens
        total_completion_tokens += usage.completion_tokens
        ans, output = cleansing_final_output(output)
    elif args.method == "TAGS":
        stage_1_k = 2

        question_classifier, prompt_get_joint_domain = get_joint_domain_prompt(question, options, num_fields = 1)
        raw_joint_domain, usage = handler.get_output_multiagent(user_input=prompt_get_joint_domain, temperature=0, system_role=question_classifier)
        total_prompt_tokens += usage.prompt_tokens
        total_completion_tokens += usage.completion_tokens
        question_domains = [domain.strip() for domain in raw_joint_domain.split("Field:")[-1].strip().split("|")]
    
        retrieved_examples = get_retrieved_examples(question, options, k=stage_1_k)  # return [{'question': ..., 'options': ..., 'reasoning': ..., 'answer': ...}, ...]
        fewshot_prompt = build_fewshot_prompt_no_confi(retrieved_examples, question, options)

        domain = question_domains[0]
        option_letters = list(options.keys())

        Specialist_classifier = (
        f"You are an experienced specialist in {domain}. "
        f"Your role is to carefully analyze clinical multiple-choice questions from the standpoint of a {domain.lower()} expert. "
        f"You should reason by focusing on the interpretation of symptoms, underlying pathophysiology, and domain-specific diagnostic principles. "
        f"First, review the provided reference examples and understand their reasoning patterns. "
        f"Then, based on your specialist knowledge, perform structured, step-by-step reasoning for the new question. "
        f"You must strictly follow this response format:\n"
        f"Thought: [your detailed step-by-step reasoning]\nAnswer: [One of {', '.join(option_letters)}]" )

        GP_classifier = (
            "You are a general practitioner (GP) trained to manage a wide range of clinical conditions across all specialties. "
            "Your task is to evaluate clinical multiple-choice questions with broad, cross-disciplinary medical knowledge. "
            "Focus on extracting key clinical findings, ruling out unlikely diagnoses, and applying general reasoning principles. "
            "First, analyze the given reference examples to understand their diagnostic thought process. "
            "Then, perform a step-by-step analysis for the new question using general medical knowledge. "
            f"You must strictly follow this response format:\n"
            f"Thought: [your detailed step-by-step reasoning]\nAnswer: [One of {', '.join(option_letters)}]" )
        
        format_classifier = (
            "You are an expert assistant specializing in formatting reasoning outputs for clinical questions.\n"
            "Your task is to transform model outputs into a structured format without changing their original meaning.\n"
            "Strictly follow the format:\n"
            "Thought: [reasoning]\n"
            "Answer: [A/B/C/D/...]\n"
            "Preserve the original intent and content while improving the organization.\n")

        output_s, usage = handler.get_output_multiagent(user_input=fewshot_prompt, temperature=0, system_role=Specialist_classifier)
        total_prompt_tokens += usage.prompt_tokens
        total_completion_tokens += usage.completion_tokens
        output_g, usage = handler.get_output_multiagent(user_input=fewshot_prompt, temperature=0, system_role=GP_classifier)
        total_prompt_tokens += usage.prompt_tokens
        total_completion_tokens += usage.completion_tokens
        # format_prompt_s = build_format_prompt(output_s)
        # format_prompt_g = build_format_prompt(output_g)
        # output_s, usage_s = handler_eval.get_output_multiagent(user_input=format_prompt_s, temperature=0, system_role=format_classifier)
        # output_g, usage_g = handler_eval.get_output_multiagent(user_input=format_prompt_g, temperature=0, system_role=format_classifier)

        parsed_s = OutputParser.parse(output_s)
        parsed_g = OutputParser.parse(output_g)
        
        ans_s, thought_s = parsed_s["answer"], parsed_s["thought"]
        ans_g, thought_g = parsed_g["answer"], parsed_g["thought"]


        ######################### secondary reasoning #########################
        retrieved_examples_s = get_retrieved_cot_examples(thought_s, k=stage_1_k)  
        retrieved_examples_g = get_retrieved_cot_examples(thought_g, k=stage_1_k)  

        fewshot_prompt_s = build_fewshot_prompt_no_confi(retrieved_examples_s, question, options)
        fewshot_prompt_g = build_fewshot_prompt_no_confi(retrieved_examples_g, question, options)
        output_ss, usage = handler.get_output_multiagent(user_input=fewshot_prompt_s, temperature=0, system_role=Specialist_classifier)
        total_prompt_tokens += usage.prompt_tokens
        total_completion_tokens += usage.completion_tokens
        output_gg, usage = handler.get_output_multiagent(user_input=fewshot_prompt_g, temperature=0, system_role=GP_classifier)
        total_prompt_tokens += usage.prompt_tokens
        total_completion_tokens += usage.completion_tokens
        # format_prompt_ss = build_format_prompt(output_ss)
        # output_ss, usage_ss = handler_eval.get_output_multiagent(user_input=format_prompt_ss, temperature=0, system_role=format_classifier)
        # format_prompt_gg = build_format_prompt(output_gg)
        # output_gg, usage_gg = handler_eval.get_output_multiagent(user_input=format_prompt_gg, temperature=0, system_role=format_classifier)
        parsed_ss = OutputParser.parse(output_ss)
        parsed_gg = OutputParser.parse(output_gg)
        ans_ss, thought_ss = parsed_ss["answer"], parsed_ss["thought"]
        ans_gg, thought_gg = parsed_gg["answer"], parsed_gg["thought"]
        ######################### get confidence #########################
        Confidence_classifier = (
            "You are a medical reasoning evaluator. You are given a reasoning chain and a final answer generated by another agent. "
            "Your task is to assign a reliability score from 1 to 5 based on how well the reasoning supports the answer and aligns with medical knowledge.\n"
            "Do not score based on how well-written or structured the text is. Only score based on medical correctness and how justifiable the conclusion is.\n"
            "Use this scale:\n"
            "5 - Reasoning is complete, medically accurate, and fully supports the answer.\n"
            "4 - Mostly correct with minor issues, but the answer is still justified.\n"
            "3 - Reasoning has some issues or omissions, but partially supports the answer.\n"
            "2 - Reasoning is flawed or incomplete; answer is weakly supported.\n"
            "1 - Reasoning is incorrect or misleading; answer is not justified.\n"
            "Return your result in the format: Score: [1-5]"
        )

        confidence_prompt_ss = build_confidence_prompt(question, options, ans_ss, thought_ss)
        confidence_prompt_gg = build_confidence_prompt(question, options, ans_gg, thought_gg)
        confidence_prompt_s = build_confidence_prompt(question, options, ans_s, thought_s)
        confidence_prompt_g = build_confidence_prompt(question, options, ans_g, thought_g)
        confidence_ss, usage = handler.get_output_multiagent(user_input=confidence_prompt_ss, temperature=0, system_role=Confidence_classifier)
        total_prompt_tokens += usage.prompt_tokens
        total_completion_tokens += usage.completion_tokens
        confidence_gg, usage = handler.get_output_multiagent(user_input=confidence_prompt_gg, temperature=0, system_role=Confidence_classifier)
        total_prompt_tokens += usage.prompt_tokens
        total_completion_tokens += usage.completion_tokens
        confidence_s, usage = handler.get_output_multiagent(user_input=confidence_prompt_s, temperature=0, system_role=Confidence_classifier)
        total_prompt_tokens += usage.prompt_tokens
        total_completion_tokens += usage.completion_tokens
        confidence_g, usage = handler.get_output_multiagent(user_input=confidence_prompt_g, temperature=0, system_role=Confidence_classifier)
        total_prompt_tokens += usage.prompt_tokens
        total_completion_tokens += usage.completion_tokens

        conf_s = extract_score_from_verifier_output(confidence_s)
        conf_g = extract_score_from_verifier_output(confidence_g)
        conf_ss = extract_score_from_verifier_output(confidence_ss)
        conf_gg = extract_score_from_verifier_output(confidence_gg)

        weighted_votes = defaultdict(float)
        weighted_votes[ans_s] += conf_s
        weighted_votes[ans_g] += conf_g
        weighted_votes[ans_ss] += conf_ss
        weighted_votes[ans_gg] += conf_gg
        ans = max(weighted_votes.items(), key=lambda x: x[1])[0]


    end_time = time.time()
    total_time = end_time - start_time

    data_info = {
        'id':qid,
        'realidx': realqid,
        'question': question,
        'options': options,
        'predicted_answer': ans,
        'confidence': '100%',
        'answer_idx': gold_answer,
        'token_usage': {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
        },
        'time_elapsed': total_time
    }
    
    return data_info
