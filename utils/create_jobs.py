import pandas as pd
import argparse
import uuid


def create_job_file(run_command: str, fn: str, time_limit: str) -> None:
    top = f"""#!/bin/sh\n#SBATCH --partition=GPUQ\n#SBATCH --account=share-ie-imf\n#SBATCH --time={time_limit}\n#SBATCH --nodes=1\n#SBATCH --ntasks-per-node=1\n#SBATCH --mem=20000\n#SBATCH --job-name=revenue\n#SBATCH --mail-user=eriko1306@gmail.com\n#SBATCH --mail-type=ALL\n#SBATCH --gres=gpu:1\n#SBATCH --output=/tmp/$SLURM_JOB_ID.out
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


def create_model_name(df, i):
    # model_name_list = []
    # for c in df.columns:
    #    print(c)
    #    print(df[[c]].values)
    #    model_name_list.append(f"{c}-{df[[c]].values[i][0]}")
    # model_name = "_".join(model_name_list)
    return uuid.uuid4().hex  # model_name


def create_bash_for_jobs(csv_path: str, fn: str) -> None:
    # read csv

    df = pd.read_csv(csv_path)
    jobs = {}
    model_names = {}
    for i in range(len(df)):
        model_name = create_model_name(df, i)
        model_names[i] = model_name
        command = [
            "python3",
            f"{df.dataset[i]}/run_{df.dataset[i]}.py",
            f"--num_workers {df.num_workers[i]}",
            f"--model_save_path {df.dataset[i]}/models/{model_name}.pt",
            f"--writer_path {df.dataset[i]}/runs/{model_name}",
            f"--epochs {df.epochs[i]}",
            f"--tenacity {df.tenacity[i]}",
            f"{'--' if df.clip_gradient[i] else '--no-'}clip",
            f"--log_interval {df.log_interval[i]}",
            f"{'--' if df.print[i] else '--no-'}print",
            f"--train_start {df.train_start[i]}",
            f"--train_end {df.train_end[i]}",
            f"--num_rolling_periods {df.num_rolling_periods[i]}",
            f"--length_rolling {df.length_rolling[i]}",
            f"--v_batch_size {df.v_batch_size[i]}",
            f"--h_batch_size {df.h_batch_size[i]}",
            f"{'--' if df.dilations[i] else '--no-'}dilated_convolutions",
            f"{'--' if df.scale[i] else '--no-'}data_scale",
            f"{'--' if df.time_covariates[i] else '--no-'}time_covariates",
            f"{'--' if df.one_hot_id[i] else '--no-'}one_hot_id",
            f"{'--' if df.cluster_covariate[i] else '--no-'}cluster_covariate",
            f"--representation {df.representation[i]}",
            f"--similarity {df.similarity[i]}",
            f"--clustering {df.clustering[i]}",
            f"--num_clusters {df.num_clusters[i]}",
            f"--num_components {df.num_components[i]}",
            f"--num_layers {df.num_layers[i]}",
            f"--kernel_size {df.kernel_size[i]}",
            f"--res_block_size {df.res_block_size[i]}",
            f"{'--' if df.bias[i] else '--no-'}bias",
            f"--lr {df.lr[i]}",
            f"--dropout {df.dropout[i]}",
            f"{'--' if df.leveledinit[i] else '--no-'}leveledinit",
            f"--stride {df.stride[i]}",
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
            f.write(f"$RES{i},{model_names[i]},")
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
