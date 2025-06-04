#!/bin/bash
#SBATCH --job-name=medagents_experiments
#SBATCH --output=logs/medagents_experiments.log
#SBATCH --error=logs/medagents_experiments.err
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=scavenge

LOGS_DIR=logs
DATA_DIR=/home/zile/nas/code/mllm/Agent/medagents-benchmark/data/
for dataset in medqa pubmedqa medmcqa medbullets mmlu mmlu-pro medxpertqa-u medxpertqa-r medexqa; do
    mkdir -p $LOGS_DIR/$dataset
    for model in gpt-4o deepseek-r1; do
        for split in test_hard; do
            for difficulty in adaptive; do
                log_file=$LOGS_DIR/$dataset/${model}_${dataset}_${split}_${difficulty}.log
                error_file=$LOGS_DIR/$dataset/${model}_${dataset}_${split}_${difficulty}.err
                echo "Running $model on $dataset $split with difficulty $difficulty"
                python main.py --dataset_name $dataset --dataset_dir $DATA_DIR/$dataset/ --split $split --model $model --method TAGS --output_files_folder ./output/ --num_processes 4 > $log_file 2> $error_file
            done
        done
    done
done