#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-imf
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=20000
#SBATCH --job-name=electricity-small-li-tc
#SBATCH --mail-user=eriko1306@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
 
module load GCC/8.3.0  CUDA/10.1.243  OpenMPI/3.1.4 PyTorch/1.3.1-Python-3.7.4 

python3 electricity_dglo_data/run_electricity.py --num_workers 4 --model_save_path electricity_dglo_data/models/tcn_small_li_tc.pt --writer_path electricity_dglo_data/runs/tcn_small_li_tc --epochs 100 --clip --log_interval=500 --print --leveledinit --time_covariates
