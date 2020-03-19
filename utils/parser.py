# parser.py
import argparse

def add_bool_arg(parser, name, default=False):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true')
    group.add_argument('--no-' + name, dest=name, action='store_false')
    parser.set_defaults(**{name:default})

def parse():
    parser = argparse.ArgumentParser(description='Adding Problem')

    # Dataset
    parser.add_argument('--train_start', type=str, default='2012-01-01')
    parser.add_argument('--train_end', type=str, default='2014-12-17')
    parser.add_argument('--v_batch_size', type=int, default=32)
    parser.add_argument('--h_batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=0)
    add_bool_arg(parser, name='time_covariates', default=True)
    add_bool_arg(parser, name='one_hot_id', default=True)
    
    # Model architecture
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--kernel_size', type=int, default=7)
    parser.add_argument('--res_block_size', type=int, default=32)
    add_bool_arg(parser, name='bias', default=True)

    # Training parameters
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--stride', type=int, default=1)
    add_bool_arg(parser, name='leveledinit', default=True)
    add_bool_arg(parser, name='clip', default=True)

    # Test parameters
    parser.add_argument('--num_rolling_periods', type=int, default=7)
    parser.add_argument('--length_rolling', type=int, default=24)

    # Logging
    parser.add_argument('--model_save_path', type=str, default='electricity/models/tcn_electricity.pt')
    parser.add_argument('--writer_path', type=str, default='electricity/runs/')
    parser.add_argument('--log_interval', type=int, default=5)
    add_bool_arg(parser, name='print', default=True)

    args = parser.parse_args()
    return args

def print_args(args):
    dic = vars(args)
    #print(dic)
    for key in dic.keys():
        print(f"{key:20s} : {dic[key]}")



if __name__ == "__main__":
    args = parse()
    print_args(args)