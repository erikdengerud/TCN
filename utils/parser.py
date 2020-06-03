# parser.py
import argparse


def add_bool_arg(parser, name, default=False):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--" + name, dest=name, action="store_true")
    group.add_argument("--no-" + name, dest=name, action="store_false")
    parser.set_defaults(**{name: default})


def parse():
    parser = argparse.ArgumentParser(description="TCN Problem")

    # Dataset
    parser.add_argument("--train_start", type=str, default="2012-01-01")
    parser.add_argument("--train_end", type=str, default="2014-12-16")
    parser.add_argument("--v_batch_size", type=int, default=32)
    parser.add_argument("--h_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    add_bool_arg(parser, name="time_covariates", default=False)
    add_bool_arg(parser, name="one_hot_id", default=False)
    parser.add_argument("--mean", type=float, default=0)
    parser.add_argument("--var", type=float, default=1.0)
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--t", type=int, default=1000)
    add_bool_arg(parser, name="scale", default=False)
    add_bool_arg(parser, name="cluster_covariate", default=False)
    add_bool_arg(parser, name="random_covariate", default=False)
    parser.add_argument("--representation", type=str, default="pca")
    parser.add_argument("--similarity", type=str, default="euclidean")
    parser.add_argument("--clustering", type=str, default="KMeans")
    parser.add_argument("--num_clusters", type=int, default=10)
    parser.add_argument("--num_components", type=int, default=10)

    # Model architecture
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--kernel_size", type=int, default=7)
    parser.add_argument("--res_block_size", type=int, default=32)
    parser.add_argument("--type_res_blocks", type=str, default="erik")
    add_bool_arg(parser, name="bias", default=True)
    parser.add_argument("--embed", type=str, default=None)
    parser.add_argument("--embedding_dim", type=int, default=3)
    add_bool_arg(parser, name="dilated_convolutions", default=True)
    add_bool_arg(parser, name="embed_sector", default=False)

    # Training parameters
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--tenacity", type=int, default=7)
    add_bool_arg(parser, name="leveledinit", default=False)
    add_bool_arg(parser, name="clip", default=False)

    # Test parameters
    parser.add_argument("--num_rolling_periods", type=int, default=7)
    parser.add_argument("--length_rolling", type=int, default=24)

    # Logging
    parser.add_argument(
        "--model_save_path", type=str, default="electricity/models/tcn_electricity.pt"
    )
    parser.add_argument("--writer_path", type=str, default="electricity/runs/")
    parser.add_argument("--log_interval", type=int, default=1)
    add_bool_arg(parser, name="print", default=False)

    args = parser.parse_args()
    return args


def print_args(args):
    dic = vars(args)
    # print(dic)
    for key in dic.keys():
        print(f"{key:20s} : {dic[key]}")


if __name__ == "__main__":
    args = parse()
    print_args(args)
