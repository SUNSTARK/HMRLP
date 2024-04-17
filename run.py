import argparse
import numpy as np
import pandas as pd
import cmdstanpy


def parse_args():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Routine main program')
    parser.add_argument('--dataset_path', type=str, required=True, help="dataset path")
    parser.add_argument('--res_path', type=str, required=True, help="res output path")
    parser.add_argument('--seed', type=int, default=1234, help="random seed")
    args = parser.parse_args()
    return args


def read_data(filepath):
    # Read data from CSV file
    data_df = pd.read_parquet(filepath)
    return data_df


def train_model(data_df, stan_file, args):
    # Set model training parameters
    total_iter = 100
    num_warmup = 70
    tree_depth = 8
    N = len(data_df['uid'].unique())
    M = len(data_df)
    W = max(data_df['w'])
    J = max(data_df['dh'])
    model_data = {
        'N': N,
        'M': M,
        'W': W,
        'J': J,
        'y': np.array(data_df['y'].tolist()),
    }

    cpp_options = {
        'STAN_THREADS': True
    }

    stm = cmdstanpy.CmdStanModel(stan_file=stan_file, cpp_options=cpp_options)

    # Run Stan sampling
    fit = stm.sample(data=model_data, chains=1, iter_sampling=total_iter - num_warmup,
                     iter_warmup=num_warmup, refresh=1, max_treedepth=tree_depth, seed=args.seed,
                     threads_per_chain=16, show_console=False)

    res = fit.summary()
    res.to_csv(args.res_path)


def main():
    args = parse_args()
    data_df = read_data(args.dataset_path)
    stan_file = 'routineness.stan'
    train_model(data_df, stan_file, args)


if __name__ == '__main__':
    main()
