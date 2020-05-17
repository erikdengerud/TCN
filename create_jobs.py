import pandas as pd
import argparse


def create_job_file(run_command: str, fn: str, time_limit: str) -> None:
    top = f"""#!/bin/sh\n#SBATCH --partition=GPUQ\n#SBATCH --account=share-ie-imf\n#SBATCH --time={time_limit}\n#SBATCH --nodes=1\n#SBATCH --ntasks-per-node=1\n#SBATCH --mem=20000\n#SBATCH --job-name=revenue\n#SBATCH --mail-user=eriko1306@gmail.com\n#SBATCH --mail-type=ALL\n#SBATCH --gres=gpu:1 
    """
    venv = "source venv/bin/activate"
    modules = (
        "module load GCC/8.3.0  CUDA/10.1.243  OpenMPI/3.1.4 PyTorch/1.3.1-Python-3.7.4"
    )
    command = run_command
    # write file
    with open(fn, "w") as f:
        f.write(top)
        f.write("\n")
        f.write(modules)
        f.write("\n")
        f.write(venv)
        f.write("\n")
        f.write(command)


def create_bash_for_jobs(csv_path: str, fn: str) -> None:
    # read csv
    df = pd.read_csv(csv_path)
    jobs = {}
    for i in range(len(df)):
        command = [
            "python3",
            f"{df.dataset[i]}/run_{df.dataset[i]}.py",
            "--num_workers 4",
            f"--model_save_path {df.dataset[i]}/models/{df.model[i]}_{df.layers[i]}_{df.kernel_size[i]}_{df.filter_size[i]}_{df.dilations[i]}_embed_{df.embed[i]}_dim_{df.embed_dim[i]}",
            f"--writer_path {df.dataset[i]}/runs/{df.model[i]}_{df.layers[i]}_{df.kernel_size[i]}_{df.filter_size[i]}_{df.dilations[i]}_embed_{df.embed[i]}_dim_{df.embed_dim[i]}",
            f"--epochs {df.max_epochs[i]}",
            f"--tenacity {df.tenacity[i]} ",
            "--clip --log_interval 1000 --print",
            "--train_start 2007-01-01 --train_end 2017-01-01",
            "--num_rolling_periods 2 --length_rolling 4 --v_batch_size 32 --h_batch_size 3",
            f"--num_layers {df.layers[i]}",
            f"--kernel_size {df.kernel_size[i]}",
            f"--res_block_size {df.filter_size[i]}",
            f"{'--' if df.dilations[i] else '--no-'}dilated_convolutions",
        ]
        if df.embed[i]:
            command.append("--embed post")
            command.append(f"--embedding_dim {df.embed_dim[i]}")

        jobs[f"{df.dataset[i]}_job_{i}.sh"] = " ".join(command)

    for i, job in enumerate(jobs.keys()):
        create_job_file(jobs[job], job, df.time_limit[i])

    with open(fn, "w") as f:
        f.write("#!/bin/sh")

        for job in jobs.keys():
            f.write("\n")
            f.write(f"chmod u+x {job}")
        for i, job in enumerate(jobs.keys()):
            f.write("\n")
            f.write(f"RES{i}=$(sbatch --parsable {job})")
        f.write('\necho "')
        for i, _ in enumerate(jobs.keys()):
            f.write(f"$RES{i}\\n")
        f.write('" >> created_jobs_names.log')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create jobs")
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--fn", type=str, default="jobs.sh")
    args = parser.parse_args()
    if args.csv_path is None:
        print("Exception: No path specified.")
    else:
        create_bash_for_jobs(args.csv_path, args.fn)
