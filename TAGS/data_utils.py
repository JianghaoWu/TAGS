import os
import json
import re
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer
import re
from typing import Dict, Optional

class QADataset:
    def __init__(self, args, traindata_obj=None):
        if hasattr(args, 'start_pos'):
            self.start_pos = args.start_pos
        if hasattr(args, 'end_pos'):
            self.end_pos = args.end_pos
        if hasattr(args, 'model_name'):
            self.model_name = args.model_name
        self.dataset_name = args.dataset_name
        self.dir_path = args.dataset_dir
        self.split = args.split  # train / test / sampled_50_hard / sampled_50 / test_hard
        self.load() # load dataset -> load data in self.data

        self.build_choice_ref_MedQA()
        

    def load(self): # load dataset -> self.data
        filename = os.path.join(self.dir_path, self.split + '.jsonl')
        self.data = []
        with open(filename) as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)

    def get_by_idx(self, idx):
        data = self.data[idx]
        data['id'] = idx
        data['realidx'] = data['realidx'] if 'realidx' in data else idx
        return data

    def __len__(self):
        return len(self.data)

    def build_ref(self):
        self.ref = []
        for i in range(len(self)):
            item = self.get_by_idx(i)
            self.ref.append({'answers': {'text': item['answer']}, 'id': i})
    
    def build_choice_ref_MedQA(self):
        self.choice_ref = []
        for i in range(len(self)):
            item = self.get_by_idx(i)
            self.choice_ref.append({
                'answers': {'text': item['options'][item['answer_idx']],'choice': item['answer_idx']}, 
                'options': item['options'], 
                'type': item.get('meta_info', ''),
                'id': i})

    def build_choice_ref_MedMCQA(self):
        self.choice_ref = []
        for i in range(len(self)):
            item = self.get_by_idx(i)
            self.choice_ref.append({
                'answers': {'text': item['answer'],
                'choice': item['answer_idx']}, 
                'options': item['options'], 
                'id': item['realidx'] if 'realidx' in item else i})

    def compute_rougescore(self, preds):
        sum_score = 0.0
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        for i, answer in enumerate(preds):
            correct_answer = self.ref[i]['answers']['text']
            # correct_answer = correct_answer.replace('\n', ' ')
            score = scorer.score(answer, correct_answer)
            sum_score += score['rouge1'].fmeasure
        return sum_score / len(preds)

    def compute_accuracy(self, preds):
        if 'PubMedQA'.lower() in self.dir_path.lower():
            correct_num = 0.0
            all_num = 0.0
            for i, answer in enumerate(preds):
                all_num += 1
                correct_choice = self.choice_ref[i]['answers']['choice']
                correct_answer = self.choice_ref[i]['answers']['text']
                if answer == correct_choice or correct_answer in answer:
                    correct_num += 1
            print(f"correct_num: {correct_num}, all_num: {all_num}")
            return correct_num / all_num
        elif 'MedQA'.lower() in self.dir_path.lower():
            correct_num = {'step1': 0.0, 'step2&3': 0.0, 'all': 0.0}
            all_num = {'step1': 0.0, 'step2&3': 0.0, 'all': 0.0}
            for i, answer in enumerate(preds):
                answer = answer.strip()
                all_num['all'] += 1
                correct_choice = self.choice_ref[i]['answers']['choice']
                correct_answer = self.choice_ref[i]['answers']['text']
                type = self.choice_ref[i]['type']
                all_num[type] += 1
                if answer == correct_choice or (correct_choice in answer and answer != 'ERROR') or correct_answer in answer:
                    correct_num[type] += 1
                    correct_num['all'] += 1
            print(f"correct_num: {correct_num}, all_num: {all_num}")
            return [correct_num[key] / all_num[key] for key in ['step1', 'step2&3', 'all']]
        elif 'MedMCQA'.lower() in self.dir_path.lower() or 'MMLU'.lower() in self.dir_path.lower():
            correct_num = 0.0
            all_num = 0.0
            for i, answer in enumerate(preds):
                all_num += 1
                correct_choice = self.choice_ref[i]['answers']['choice']
                correct_answer = self.choice_ref[i]['answers']['text']
                if answer == correct_choice or correct_answer in answer:
                    correct_num += 1
            print(f"correct_num: {correct_num}, all_num: {all_num}")
            return correct_num / all_num

def remove_incomplete_sentence(text):
    sentences = sent_tokenize(text)
    if len(sentences) > 1 and sentences[-1][-1] != '.':
        return ' '.join(sentences[:-1]) + '.'   #remove the last sentence
    else:
        return text

def cleansing_analysis(analyses, domains, type):
    analysis = {}
    
    for i, item in enumerate(analyses):
        if item == "ERROR.":
            item = f"There is no analysis for this {type}."
        item = remove_incomplete_sentence(item)
        if "as an ai language model" in item.lower():
            end_index = item.lower().find("as an ai language model")+len("as an ai language model")
            item= item[end_index:].strip().strip(',').strip()
        analysis[domains[i]] = item
    
    return analysis


def cleansing_syn_report(question, options, raw_synthesized_report):

    tmp = raw_synthesized_report.split("Total Analysis:")
    try:
        total_analysis_text = tmp[1].strip()
    except:
        total_analysis_text = tmp[0].strip()
    if "Key Knowledge" in tmp:
        key_knowledge_text = tmp[0].split("Key Knowledge:")[-1].strip()
        final_syn_repo = f"Question: {question} \n" \
            f"Options: {options} \n" \
            f"Key Knowledge: {key_knowledge_text} \n" \
            f"Total Analysis: {total_analysis_text} \n"
    else:
        final_syn_repo = f"Question: {question} \n" \
            f"Options: {options} \n" \
            f"Total Analysis: {total_analysis_text} \n"
    
    return final_syn_repo

def cleansing_final_output_with_confidence(output):
    try:
        ans = output.split(":")[-1]
        ans = re.findall(r'A|B|C|D|E|F|G|H|I|J|K|L|M|N|O|P|Q|R|S|T|U|V|W|X|Y|Z', ans)
        if len(ans) == 0:
            ans = ""
        else:
            ans = ans[0]
    except:
        ans = ""
    
    try:
        match = re.search(r"[Cc]onfidence\s*:\s*(\d{1,3})%", output)
        confidence = f"{match.group(1)}%" if match else ""
    except:
        confidence = ""

    return ans, confidence, output

class OutputParser:
    """
    A unified parser to extract structured fields (thought, confidence, answer)
    from multi-agent model output in medical QA settings.
    """

    @staticmethod
    def extract_thought(output: str) -> str:
        """Extract the 'Thought' reasoning from the output."""
        try:
            match = re.search(r"Thought\s*:\s*(.*?)(?:\nConfidence\s*:|\nAnswer\s*:)", output, re.DOTALL)
            return match.group(1).strip() if match else ""
        except:
            return ""

    @staticmethod
    def extract_answer(output: str) -> str:
        """Extract the final selected answer (A/B/C/...) from the output."""
        try:
            # Match the last occurrence of 'Answer: <letter>'
            match = re.search(r"Answer\s*:\s*([A-Z])", output)
            return match.group(1) if match else ""
        except:
            return ""

    @classmethod
    def parse(cls, output: str) -> Dict[str, Optional[str]]:
        """Parse the full output into structured components."""
        return {
            "thought": cls.extract_thought(output),
            "answer": cls.extract_answer(output)
        }

def cleansing_final_output(output: str):
    try:
        cleaned_output = output.strip().replace("'''", "").replace('"""', "")
        match = re.search(r"(?:Answer\s*[:：]?\s*)?(?:.*\n)?\s*Option\s*[:：]?\s*([A-J])\b", cleaned_output, re.IGNORECASE)
        if match:
            return match.group(1).upper(), output
        match = re.search(r"Answer\s*[:：]?\s*([A-J])\b", cleaned_output, re.IGNORECASE)
        if match:
            return match.group(1).upper(), output
        match = re.search(r"\(\s*([A-J])\s*\)", cleaned_output)
        if match:
            return match.group(1).upper(), output
        match = re.search(r"\b([A-J])\s*\.", cleaned_output)
        if match:
            return match.group(1).upper(), output
        match = re.search(r"correct answer is\s+([A-J])\b", cleaned_output, re.IGNORECASE)
        if match:
            return match.group(1).upper(), output
        if "thank you" in cleaned_output.lower():
            return "Z", output
        if re.search(r"\b([A-J])(?:,\s*[A-J]){1,}\b", cleaned_output):
            return "Z", output
        lines = [line.strip() for line in cleaned_output.splitlines()]
        for line in lines:
            if re.fullmatch(r"[A-J]", line):
                return line, output
        match = re.findall(r"\b([A-J])\b", cleaned_output)
        if match:
            return match[0].upper(), output

    except Exception as e:
        print(f"[Error] cleansing output failed: {e}")

    return "Z", output

def cleansing_voting(output):
    output = output.lower()
    ans = re.findall(r'yes|no', output)
    if len(ans) == 0:
        ans = "yes"
    else:
        ans = ans[0]
    return ans


def transform_dict2text(analyses, type, content):
    if type == "question":
        report = ""
        i = 0
        for _domain, _analysis in analyses.items():
            report += f"Report{i} \n" \
                f"Question: {content} \n" \
                f"Domain: {_domain} \n" \
                f"Analysis: {_analysis} \n\n"
            i += 1
    elif type == "options":
        report = ""
        i = 0
        for _domain, _analysis in analyses.items():
            report += f"Report{i}: \n" \
                f"Options: {content} \n" \
                f"Domain: {_domain} \n" \
                f"Analysis: {_analysis} \n\n"
            i += 1
    return report


def build_answer_extraction_prompt(raw_output, option_letters):
    return (
        "You are a medical exam grader. Your task is to extract the FINAL answer "
        "from the following response.\n\n"
        "The response contains analysis and reasoning for a multiple-choice question. "
        "You must read the response carefully and extract the FINAL single-letter answer.\n\n"
        "⚠️ You must follow these STRICT rules:\n"
        "- ONLY respond with a single uppercase letter from this list: "
        f"{', '.join(option_letters)}.\n"
        "- DO NOT include explanations, thoughts, or any additional text.\n"
        "- DO NOT reprint the question, options, or reasoning.\n"
        "- If multiple letters appear, select the one most clearly stated as final.\n"
        "- If the correct answer is ambiguous or not explicitly mentioned, guess the most likely one, "
        "but still output ONLY ONE LETTER.\n"
        "- Output must match this exact format:\n"
        "'''\nA\n'''\n(replace A with the final selected letter)\n\n"
        "Now extract the final answer from this response:\n"
        "--------------------\n"
        f"{raw_output.strip()}\n"
        "--------------------\n"
        "Your final answer:"
    )
