#!/bin/bash
#SBATCH --job-name=generate_train_data
#SBATCH --output=/lustre/fswork/projects/rech/rqn/ugy38tw/.local/main/%x_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/rqn/ugy38tw/.local/main/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -C h100
#SBATCH --gres=gpu:4
#SBATCH --hint=nomultithread
#SBATCH --time=1:00:00
#SBATCH --account=yye@h100
#SBATCH --qos=qos_gpu_h100-dev

echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

module purge
module load miniforge/24.9.0
module load cuda/12.6.3
conda activate baselinesenv
export LD_LIBRARY_PATH=/linkhome/rech/genltc01/ugy38tw/.local/lib/python3.12/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH

chmod +x generate_train_data.py

srun generate_train_data.py --synthetic_data_dir test_generated_data.jsonl --get_label --use_adaptive_binary --model_dir "../.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-70B/snapshots/b1c0b44b4369b597ad119a196caf79a9c40e141e/"
