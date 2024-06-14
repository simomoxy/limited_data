import os
import joblib
import pandas as pd
import optuna
import argparse
from argparse import ArgumentParser


def parse_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument("--models", default=['cnn'], type=list)
    parser.add_argument("--down_sample_rates", default=[0.0, 0.2, 0.4, 0.6, 0.8, 0.85, 0.9, 0.95, 0.99], type=list)
    parser.add_argument("--data_seeds", default=[0, 1, 2, 3, 4], type=list)
    parser.add_argument("--path", default='./optuna', type=str)
    parser.add_argument("--num_study_samples", default=100, type=int)
    
    return parser


def main():
    parser = ArgumentParser()
    parser = parse_args(parser)

    args = parser.parse_args()
    print(args)
    
    #print(os.getcwd())

    for model in args.models:   
        path = f'{args.path}/{model}/multi/bayesian'
        pareto_path = f'{args.path}/{model}/pareto'
        
        for file in os.listdir(path):
            if not ('.log' in file):
                continue

            study_name = file.split('.log')[0]
            l = study_name.split('_')
            dsr = l[4]
            ds = l[6]

            storage = optuna.storages.JournalStorage(optuna.storages.JournalFileStorage(f"{path}/{study_name}.log"))
            study = optuna.load_study(study_name=study_name, storage=storage)

            if not len(study.trials) >= 0:#args.num_study_samples:
                print(f"not enough trials for {study_name}")
                print(len(study.trials))
                print()
                continue

            pareto = study.best_trials

            pareto_idx = [p.number for p in pareto]
            df = study.trials_dataframe()
            pareto_df = df.loc[pareto_idx, :]
            print(study_name, len(pareto_idx))

            p_path = pareto_path + f"/{dsr}/{ds}"
            if not os.path.exists(p_path):
                os.makedirs(p_path)

                for i,num in enumerate(pareto_idx):
                    if num > args.num_study_samples:
                        continue

                    trial = pareto_df.iloc[i, :]
                    params = pd.DataFrame({i.replace('params_', ''):[trial[i]] for i in trial.index if 'params' in i})
                    params = params.to_dict('list')
                    params = {k:v[0] for k,v in params.items()}
                    with open(f'{p_path}/pareto{i}.pkl', 'wb') as f:
                        joblib.dump(params, f)


if __name__ == "__main__":
    main()  