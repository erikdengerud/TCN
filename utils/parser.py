import argparse

def parse():
    parser = argparse.ArgumentParser(description='Adding Problem')
    parser.add_argument(
        '--train_start', type=str, default='2012-01-01', metavar='train_start')
    parser.add_argument(
        '--train_end', type=str, default='2014-12-17', metavar='train_end')
    #parser.add_argument(
    #    '--test_start', type=str, default='2014-08-01', metavar='test_start')
    #parser.add_argument(
    #    '--test_end', type=str, default='2014-10-01', metavar='test_end')
    parser.add_argument(
        '--v_batch_size', type=int, default=32, metavar='v_batch_size')
    parser.add_argument(
        '--h_batch_size', type=int, default=256, metavar='h_batch_size')
    parser.add_argument(
        '--num_layers', type=int, default=5, metavar='num_layers')
    parser.add_argument(
        '--in_channels', type=int, default=8, metavar='in_channels')
    parser.add_argument(
        '--out_channels', type=int, default=1, metavar='out_channels')
    parser.add_argument(
        '--kernel_size', type=int, default=7, metavar='kernel_size')
    parser.add_argument(
        '--res_block_size', type=int, default=32, metavar='res_block_size')
    parser.add_argument(
        '--bias', type=bool, default=True, metavar='bias')
    parser.add_argument(
        '--dropout', type=float, default=0.0, metavar='dropout')
    parser.add_argument(
        '--stride', type=int, default=1, metavar='stride')
    parser.add_argument(
        '--leveledinit', type=bool, default=False, metavar='leveledinit')
    parser.add_argument(
        '--model_save_path', type=str, default='electricity/models/tcn_electricity.pt', 
        metavar='model_save_path')
    parser.add_argument(
        '--epochs', type=int, default=300, metavar='epochs')
    parser.add_argument(
        '--lr', type=float, default=5e-4, metavar='lr')
    parser.add_argument(
        '--clip', type=bool, default=False, metavar='clip')
    parser.add_argument(
        '--log_interval', type=int, default=5, metavar='log_interval')
    parser.add_argument(
        '--writer_path', type=str, default='electricity/runs/', 
        metavar='writer_path')
    parser.add_argument(
        '--writer_comment', type=str, default='test', 
        metavar='writer_comment')
    parser.add_argument(
        '--print', type=bool, default=False, metavar='print')
    parser.add_argument(
        '--num_workers', type=int, default=0, metavar='num_workers')
    parser.add_argument(
        '--num_rolling_periods', type=int, default=7, metavar='num_rolling')
    parser.add_argument(
        '--length_rolling', type=int, default=24, metavar='length_rolling')
    parser.add_argument(
        '--time_covariates', type=bool, default=True, metavar='time_covariates'
    )

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