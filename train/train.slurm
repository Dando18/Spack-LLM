#!/bin/bash
#SBATCH -J train
#SBATCH -o train-%A.out
#SBATCH -A bhatele-lab-aac
#SBATCH -t 00:15:00
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=128000
#SBATCH -p gpu
#SBATCH --gpus=a100:4

cd ~/spackllm/train

module load cuda/12.1.1/gcc/ python
source ~/scratch/spackllm/.env/bin/activate

export HF_HOME=~/scratch/.cache/huggingface

accelerate launch \
    --config_file=accelerate-config.yaml \
    --machine_rank=0 \
    train.py \
        --model_name codellama/CodeLlama-7b-hf \
        --dataset_name daniellnichols/package-metadata \
        --dataset_text_field text \
        --learning_rate 1.41e-5 \
        --batch_size 1 \
        --seq_length 16384 \
        --num_train_epochs 3 \
        --output_dir output-7b \
        --save_steps 100 \
        --save_total_limit 1 \
        --gradient_checkpointing True \
        --mixed_precision bf16 \
        --num_workers 4 \
        --packing False