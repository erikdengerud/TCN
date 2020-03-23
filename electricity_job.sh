#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-imf
#SBATCH --time=30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=20000
#SBATCH --job-name=electricity-pytorch
#SBATCH --mail-user=eriko1306@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
 
module load GCC/8.3.0  CUDA/10.1.243  OpenMPI/3.1.4 PyTorch/1.3.1-Python-3.7.4 

python3 electricity/run_electricity.py --num_workers=4 --model_save_path='electricity/models/tcn_ohe_li.pt' --writer_path='electricity/runs/tcn_ohe_li' --epochs=500 --leveledinit --no-time_covariates --no-clip --no-print --one_hot_id
