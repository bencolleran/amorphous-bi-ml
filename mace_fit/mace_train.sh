#!/bin/bash
#$ -cwd
#$ -l s_rt=100:00:00
#$ -pe smp 16
#$ -l nvidia_a100=1
#$ -P gpu
#$ -o $JOB_ID.log
#$ -e $JOB_ID.err
 
export CUDA_VISIBLE_DEVICES=$SGE_GPU
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

conda activate autoplex
cd /u/vld/sedm7085/test/mace_train_files

mace_run_train \
    --name="MACE_model_rss" \
    --train_file="train.extxyz" \
    --valid_fraction=0.05 \
    --test_file="test.extxyz" \
    --config_type_weights='{"Default":1.0}' \
    --model="MACE" \
    --energy_key="REF_energy" \
    --forces_key="REF_forces" \
    --stress_key="REF_stress" \
    --hidden_irreps='128x0e + 128x1o' \
    --r_max=6.0 \
    --batch_size=16 \
    --max_num_epochs=2000 \
    --scheduler_patience=15 \
    --patience=30 \
    --eval_interval=4 \
    --ema \
    --ema_decay=0.99 \
    --swa \
    --start_swa=1200 \
    --default_dtype="float64"\
    --amsgrad \
    --error_table='PerAtomMAE' \
    --loss="huber" \
    --device=cuda \
    --restart_latest \
    --enable_cueq=True \
    --wandb \
    --wandb_name='mace_fit' \
 