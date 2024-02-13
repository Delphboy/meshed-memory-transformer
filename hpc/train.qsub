#!/bin/bash
#$ -l h_rt=24:00:00
#$ -l h_vmem=11G
#$ -pe smp 8
#$ -l gpu=1
#$ -l gpu_type=ampere
#$ -wd /data/home/$USER/meshed-memory-transformer
#$ -j y
#$ -m ea
#$ -o logs/


# Load modules
module load python/3.8.5
module load cuda/11.6.2
module load cudnn/8.4.1-cuda11.6
module load gcc/6.3.0
module load java/1.8.0_382-openjdk

# Activate virtual environment
source .venv/bin/activate


# Run!
python3 train.py --dataset "coco" \
                --dataset_feat_path "/path/to/butd/features" \
                --dataset_ann_path "/path/to/dataset_coco.json" \
                --checkpoint_location "/data/scratch/$USER/m2-checkpoints/" \
                --feature_limit 50 \
                --exp_name "My cool experiment name" \
                --m 40 \
                --n 3 \
                --workers 4 \
                --max_epochs 30 \
                --batch_size 64 \
                --learning_rate 1 \
                --warmup 10000 \
                --seed 42 \
                --patience -1 \
                --force_rl_after -1 \
                --meshed_emb_size 2048 \
                --dropout 0.1 \
