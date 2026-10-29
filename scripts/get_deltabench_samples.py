#!/bin/bash
#SBATCH --job-name=deltabench
#SBATCH --output=/lustre/fswork/projects/rech/rqn/ugy38tw/.local/main/%x_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/rqn/ugy38tw/.local/main/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --account=rqn@h100
#SBATCH --qos=qos_gpu_h100-dev

echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

module purge
module load miniforge/24.9.0
module load cuda/12.6.3
conda activate baselinesenv
export LD_LIBRARY_PATH=/linkhome/rech/genltc01/ugy38tw/.local/lib/python3.12/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH

chmod +x get_deltabench_partial_chains.py 
srun get_deltabench_partial_chains.py

echo "Job ended at: $(date)"
