from string import punctuation
import argparse
import tqdm
import json
from utils import *
from data_utils import QADataset
from api_utils import api_handler
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import os

load_dotenv()

def process_sample(idx, raw_sample, realqid, handler, handler_eval, args, dataobj):
    question = raw_sample['question'] if raw_sample['question'][-1] in punctuation else raw_sample['question']
    options = raw_sample['options']
    gold_answer = raw_sample['answer_idx']
    try:
        output = fully_decode(idx, realqid, question, options, gold_answer, handler, handler_eval, args, dataobj)
        if len(output['predicted_answer']) != 1:
            return None
        return output
    except Exception as e:
        print(f"Error processing sample: {e}")
        return None



def save_results(results, existing_output_file):
    results = sorted(results, key=lambda x: x['id'])
    # Write results in the original order of id
    with open(existing_output_file, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    print("Starting the script...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='deepseek-v3')
    parser.add_argument('--dataset_name', default='medexqa')
    parser.add_argument('--dataset_dir', default='../../data/medexqa/')
    parser.add_argument('--split', default='test_hard')
    parser.add_argument('--start_pos', type=int, default=0)
    parser.add_argument('--end_pos', type=int, default=-1)
    parser.add_argument('--output_files_folder', default='./output/')
    parser.add_argument('--num_processes', type=int, default=4)
    

    parser.add_argument('--method', type=str, default='syn_verif', choices=['syn_verif', 'syn_only', 'anal_only', 'base_direct', 'base_cot', 'retrieval_cot', 'TAGS'])
    parser.add_argument('--max_attempt_vote', type=int, default=3)
    args = parser.parse_args()


    # get handler
    if args.model_name in ['gpt-4o', 'gpt-4o-mini', 'deepseek-r1', 'deepseek-v3', 'Qwen2.5-7B-Instruct']: # select the model
        handler = api_handler(args.model_name)
    else:
        raise ValueError
    handler_eval = api_handler('deepseek-v3')

    # get dataobj
    dataobj = QADataset(args, traindata_obj=None)
    os.makedirs(args.output_files_folder, exist_ok=True)
    subfolder = os.path.join(args.output_files_folder, args.dataset_name)
    os.makedirs(subfolder, exist_ok=True)


    # get existing output file
    existing_output_file = os.path.join(subfolder, f"{args.model_name}-{args.dataset_name}-{args.split}-{args.method}.json")
    if os.path.exists(existing_output_file):
        print(f"Existing output file found: {existing_output_file}")
        with open(existing_output_file, 'r') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} results from existing file.")
    else:
        results = []

    # set test range
    end_pos = len(dataobj) if args.end_pos == -1 else args.end_pos
    test_range = range(args.start_pos, end_pos)  # closed interval
    results_ids = [result['id'] for result in results]
    results_realidx = [result.get('realidx', None) for result in results]
    test_range = [idx for idx in test_range if dataobj.get_by_idx(idx)['id'] not in results_ids and dataobj.get_by_idx(idx)['realidx'] not in results_realidx]

    # run multi-threading
    with ThreadPoolExecutor(max_workers=args.num_processes) as executor:
        futures = []
        # Submit tasks to the executor
        for idx in tqdm.tqdm(test_range, desc=f"{args.start_pos} ~ {end_pos}"):
            raw_sample = dataobj.get_by_idx(idx)
            realqid = raw_sample['id']
            # Submit each fully_decode task to the thread pool
            futures.append(executor.submit(process_sample, idx, raw_sample, realqid, handler, handler_eval, args, dataobj))

        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Collecting results"):
            try:
                data_info = future.result()
                if data_info is not None:
                    results.append(data_info)  # Store result with its index
                    save_results(results, existing_output_file)
            except Exception as e:
                print(f"Error processing sample: {e}")

