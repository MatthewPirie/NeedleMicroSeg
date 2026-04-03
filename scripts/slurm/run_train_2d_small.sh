#!/bin/bash
#SBATCH --job-name=needle2d
#SBATCH --output=/home/pirie03/projects/aip-medilab/pirie03/NeedleMicroSeg/scripts/slurm/logs_2d/needle2d_%j.out
#SBATCH --error=/home/pirie03/projects/aip-medilab/pirie03/NeedleMicroSeg/scripts/slurm/logs_2d/needle2d_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

echo "job started: $(date)"
echo "job id: $SLURM_JOB_ID"
echo "hostname: $(hostname)"
echo "pwd before cd: $(pwd)"

cd /home/pirie03/projects/aip-medilab/pirie03/NeedleMicroSeg || exit 1
mkdir -p /home/pirie03/projects/aip-medilab/pirie03/NeedleMicroSeg/scripts/slurm/logs_2d

echo "pwd after cd: $(pwd)"

echo "activating venv: $(date)"
source /home/pirie03/envs/prostate_microseg/bin/activate
echo "venv activated: $(date)"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python scripts/run_train_2d.py \
  --train_config /home/pirie03/projects/aip-medilab/pirie03/NeedleMicroSeg/configs/train_2d/full_frame_resize_test_2.yaml \
  --runs_dir runs_2d \
  --run_name "full_frame_resize_test_${SLURM_JOB_ID}" \
  --splits_file "/home/pirie03/projects/aip-medilab/pirie03/NeedleMicroSeg/dataset/multicenter_custom_split.json" \
  --model_variant "small" \
  --num_workers 8

echo "job finished: $(date)"