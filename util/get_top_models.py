import os
from pathlib import Path
import pandas as pd
import json
import joblib
import argparse
from argparse import ArgumentParser


def parse_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument("--models", default=['cnn'], type=list)
    parser.add_argument("--down_sample_rates", default=[0.0, 0.2, 0.4, 0.6, 0.8, 0.85, 0.9, 0.95, 0.99], type=list)
    parser.add_argument("--data_seeds", default=[0, 1, 2, 3, 4], type=list)
    parser.add_argument("--path", default='./ray/', type=str)
    parser.add_argument("--num_study_samples", default=100, type=int)
    parser.add_argument("--topk", default=5, type=int)
    parser.add_argument("--sota", default=False, type=bool)
    
    return parser


def unite_dicts(list_of_dicts):
    result = {k: [] for k in list_of_dicts[0].keys()}
    print(result)
    for d in list_of_dicts:
        for k, v in d.items():
            #print(k, v)
            if k == 0:
                continue
            result[k].append(v)
    return pd.DataFrame(result)


def save_max_dicts(path, model, p_df, objective='single', optimization='ASHA', sota=True, topk=5, num_study_samples=100):
    for dsr in p_df['down_sample_rate'].unique():
        for data_seed in p_df['data_seed'].unique():
            if not os.path.exists(f'{path}/top{topk}/{objective}/{optimization}/{sota}/{dsr}/{data_seed}'):
                os.makedirs(f'{path}/top{topk}/{objective}/{optimization}/{sota}/{dsr}/{data_seed}')
        
            df = p_df[(p_df['model'] == model) & (p_df['down_sample_rate'] == dsr) & (p_df['objective'] == objective) & (p_df['optimization'] == optimization) & (p_df['sota'] == sota) & (p_df['data_seed'] == data_seed)]
            if len(df) >= num_study_samples :
                df = df.sample(num_study_samples)
            else:
                print(f'{model}: [{len(df)}|{num_study_samples}]for dsr={dsr} ds={data_seed} sota={sota}')
                continue
            df = df.sort_values('score', ascending=False)
            df = df.head(topk)

            if len(df) < topk:
                print(f'Not enough models for {dsr}')
                continue
            for i in range(topk):
                with open(f'{path}/top{topk}/{objective}/{optimization}/{sota}/{dsr}/{data_seed}/top{i+1}.pkl', 'wb') as f:
                    joblib.dump(df.iloc[i, :].to_dict(), f)


def main():
    parser = ArgumentParser()
    parser = parse_args(parser)

    args = parser.parse_args()
    print(args)

    for model in args.models:
        path = f"{args.path}/{model}" 


        params = []
        results = []
        for dir_ in os.listdir(path):
            if not os.path.isdir(f"{path}/{dir_}" ):
                continue
            
            if 'ray' not in dir_:
                continue
            _, _, objective, optimization, dsr, sota, data_seed, id_ = dir_.split('_')
            dsr = float(dsr)
            id_ = int(id_.split('=')[-1])
            
            
            for folder in os.listdir(f"{path}/{dir_}" ):
                if not os.path.isdir(f"{path}/{dir_}/{folder}" ):
                    continue
                if 'params.json' not in os.listdir(f"{path}/{dir_}/{folder}" ) or 'result.json' not in os.listdir(f"{path}/{dir_}/{folder}" ):
                    continue
                with open(f"{path}/{dir_}/{folder}/params.json") as f:
                    try:
                        params.append(json.load(f))
                    except:
                        continue
                params[-1]['down_sample_rate'] = dsr
                params[-1]['id'] = id_
                params[-1]['data_seed'] = data_seed
                params[-1]['objective'] = objective
                params[-1]['optimization'] = optimization
                if sota == 'True':
                    params[-1]['sota'] = True
                else:
                    params[-1]['sota'] = False
            
                results.append(pd.read_json(f"{path}/{dir_}/{folder}/result.json", lines=True))
        p_df = unite_dicts(params)
        max_performance = []
        for r in results:
            try:
                max_performance.append(r['val_macro'].max())
            except KeyError:
                max_performance.append(0)
                
        p_df['score'] = max_performance

        p_df = p_df.drop(columns='SOTA_ARCHITECTURE')
        p_df.to_csv(f'{path}/{model}_ray.csv', index=False)


        save_max_dicts(path, model, p_df, topk=args.topk, num_study_samples=args.num_study_samples, sota=args.sota)
    

if __name__ == "__main__":
    main()