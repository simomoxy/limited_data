from platform import architecture
from pyexpat import model
import time
from datetime import timedelta
import logging
import os
import pdb
from os.path import exists, join, dirname
import argparse
from argparse import ArgumentParser
import pickle
import joblib

import math
import sys

import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.optim import AdamW
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint #, EarlyStopping
from pytorch_lightning.utilities.model_summary import ModelSummary

from mlxtend.evaluate import confusion_matrix
from clinical_ts.create_logger import create_logger
from clinical_ts.eval_utils_cafa import eval_scores, eval_scores_bootstrap
from clinical_ts.timeseries_utils import aggregate_predictions
from ecg_datamodule import ECGDataModule

import optuna
from optuna.storages import JournalStorage, JournalFileStorage
from optimization_utils.optuna_config import *
from optimization_utils.PytorchLightningPruningCallback import PyTorchLightningPruningCallback


import torch.multiprocessing as mp

logger = create_logger(__name__)


def parse_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--dataset",
        dest="target_folder",
        help="used dataset for training",
    )
    parser.add_argument("--logdir", default="./logs")
    parser.add_argument("--study_path", default="")
    parser.add_argument("--label_class", default="label_all")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--input_size", type=int, default=250)
    parser.add_argument("--gpu", action="store_true", default=True)
    parser.add_argument("--nomemmap", action="store_true", default=False)
    parser.add_argument("--test_folds", nargs="+", default=[9, 10], type=int)
    parser.add_argument("--filter_label")
    parser.add_argument("--combination",  default="both")
    parser.add_argument("--test_only", action="store_true", default=False)
    parser.add_argument("--model", default='xresnet1d50')
    parser.add_argument("--normalize", action='store_true', default=False)
    parser.add_argument("--use_meta_information_in_head", action='store_true', default=False)
    parser.add_argument("--down_sample_rate", default=0.0, type=float)
    parser.add_argument("--accelerator", default="gpu", type=str)
    parser.add_argument("--num_study_samples", default=6, type=int)
    parser.add_argument("--SOTA_config", action="store_true", default=False)
    parser.add_argument("--num_gpus", default=1, type=int)
    parser.add_argument("--limited_complexity", action="store_true", default=False)
    parser.add_argument("--objectives", default='single', type=str)
    parser.add_argument("--optimization", default='random', type=str)
    parser.add_argument("--percent_valid_examples", default=0.1, type=float)
    parser.add_argument("--id", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--study_name", default='test', type=str)
    parser.add_argument("--load_from_memory", action='store_true', default=False)
    parser.add_argument("--data_seed", default=1, type=int)
    parser.add_argument("--num_jobs", default=10, type=int)
    return parser


def init_logger(debug=False, log_dir="./experiment_logs"):
    level = logging.INFO

    if debug:
        level = logging.DEBUG

    # remove all handlers to change basic configuration
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(
        filename=os.path.join(log_dir, "info.log"),
        level=level,
        format="%(asctime)s %(name)s:%(lineno)s %(levelname)s:  %(message)s  ",
    )
    return logging.getLogger(__name__)


def get_config(args, trial):
    if args.model == 'cnn':
        config=cnn_config(trial=trial)
        config["model"] = 'cnn'
        config["SOTA_ARCHITECTURE"] = args.SOTA_config
    elif args.model == 'lstm':
        config=lstm_config(trial=trial)
        config["model"] = 'lstm'
        config["SOTA_ARCHITECTURE"] = args.SOTA_config
    elif args.model == 'enc':
        config=transformerEnc_config(trial=trial)
        config["model"] = 'enc'
        config["SOTA_ARCHITECTURE"] = args.SOTA_config
    elif args.model == 'transformer':
        config=transformer_config(trial=trial)
        config["model"] = 'transformer'
        config["SOTA_ARCHITECTURE"] = args.SOTA_config
    elif args.model == 's4':
        config=s4_config(trial, args.SOTA_config)
        config["model"] = 's4'
        config["SOTA_ARCHITECTURE"] = args.SOTA_config
    elif args.model == 'xresnet':
        config=xresnet_config(trial, args.SOTA_config)
        config["model"] = 'xresnet'
        config["SOTA_ARCHITECTURE"] = args.SOTA_config
    elif args.model == 'cpc':
        config=cpc_config(trial, args.SOTA_config)
        config["model"] = 'cpc'
        config["SOTA_ARCHITECTURE"] = args.SOTA_config
    elif args.model == 'rff':
        config=rff_config(trial=trial)
        config["model"] = 'rff'
        config["SOTA_ARCHITECTURE"] = args.SOTA_config
    elif args.model == 'convrff':
        config=convrff_config(trial=trial)
        config["model"] = 'convrff'
        config["SOTA_ARCHITECTURE"] = args.SOTA_config
    else:
        raise NotImplemented("Only ray_cnn, ray_lstm, ray_transformerEnc, and ray_transformer supported.")

    return config


def get_datamodule(args, config):
    # data
    datamodule = ECGDataModule(
        config["batch_size"],
        args.target_folder,
        label_class=args.label_class,
        num_workers=args.num_workers,
        test_folds=args.test_folds,
        nomemmap=args.nomemmap,
        combination=args.combination,
        filter_label=args.filter_label,
        data_input_size=args.input_size,
        normalize=args.normalize,
        use_meta_information_in_head=args.use_meta_information_in_head,
        down_sample_rate=args.down_sample_rate,
        data_seed=args.data_seed,
    )

    config["num_classes"] = datamodule.num_classes
    config["num_samples"] = datamodule.num_samples
    config["down_sample_rate"] = datamodule.down_sample_rate

    return datamodule, config


def get_model(args, config):
    if args.model == 'cnn':
        from dl_models.modules import CNN_module
        model = CNN_module(config)
    elif args.model == 'lstm':
        from dl_models.modules import LSTM_module
        model = LSTM_module(config)
    elif args.model == 'enc':
        from dl_models.modules import Transformer_module
        model = Transformer_module(config)
    elif args.model == 'transformer':
        from dl_models.modules import Transformer_module
        model = Transformer_module(config)
    elif args.model == 's4':
        from dl_models.modules import S4_module
        model = S4_module(config)
    elif args.model == 'xresnet':
        from dl_models.modules import XRESNET_module
        model = XRESNET_module(config)
    elif args.model == 'cpc':
        from dl_models.modules import CPC_module
        model = CPC_module(config)
    elif args.model == 'rff':
        from dl_models.modules import RFF_module
        model = RFF_module(config)
    elif args.model == 'convrff':
        from dl_models.modules import ConvRFF_module
        model = ConvRFF_module(config)
    else:
        raise Exception("model {} not found".format(args.model))
    
    summary = ModelSummary(model)
    config["trainable_parameters"] = summary.trainable_parameters
    config["model_size"] = summary.model_size

    return model, config


def get_sampler(args, seed, study_dir):
    if args.optimization == 'bayesian':
        if os.path.exists(f"{study_dir}_sampler.pkl"):
            sampler = joblib.load(open(f"{study_dir}_sampler.pkl", "rb"))
            print('\nloaded sampler from memory\n')
        else:
            sampler = optuna.samplers.TPESampler(seed=seed)
    elif args.optimization == 'random':
        if os.path.exists(f"{study_dir}_sampler.pkl"):
            sampler = joblib.load(open(f"{study_dir}_sampler.pkl", "rb"))
            print('\nloaded sampler from memory\n')
        else:
            sampler = optuna.samplers.RandomSampler(seed=seed)
    elif args.optimization == 'population':
        if os.path.exists(f"{study_dir}_sampler.pkl"):
            sampler = joblib.load(open(f"{study_dir}_sampler.pkl", "rb"))
            print('\nloaded sampler from memory\n')
        else:
            sampler = optuna.samplers.NSGAIISampler(seed=seed)
    return sampler


def get_study_single(args, study_dir, direction, sampler, study_name, pruner=None, storage_name=None):
    if args.load_from_memory: # true if study was completed
        if os.path.exists(f"{study_dir}.pkl"):
            with open(f"{study_dir}.pkl", "rb") as fp:
                study = joblib.load(fp)
                study.sampler = sampler
                print("\nloaded study from memory\n")
        else:
            if pruner:
                study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner, study_name=study_name)
            else:
                study = optuna.create_study(direction=direction, sampler=sampler, study_name=study_name)
    else: # true if study hasn't been completed yet, aka, no pickle file
        #raise NotImplementedError("load_from_memory=False not implemented yet")
        if pruner:
            study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner, study_name=study_name, storage=storage_name, load_if_exists=True)
        else:
            study = optuna.create_study(direction=direction, sampler=sampler, study_name=study_name, storage=storage_name, load_if_exists=True)
        
    return study
        
    


def get_study_multi(args, study_dir, directions, sampler, study_name, storage_name=None):
    if args.load_from_memory:  # true if study was completed
        if os.path.exists(f"{study_dir}.pkl"):
            with open(f"{study_dir}.pkl", "rb") as fp:
                study = joblib.load(fp)
                study.sampler = sampler
        else:
                study = optuna.create_study(directions=directions, sampler=sampler, study_name=study_name)
    else:
        #raise NotImplementedError("load_from_memory=False not implemented yet")
        study = optuna.create_study(directions=directions, sampler=sampler, study_name=study_name, storage=storage_name, load_if_exists=True)           
        
    return study

# hard threshold due to memory issues
def valid_model_size(config):
    if config["model_size"] < 0:
        return False
    elif config["model_size"] > 750: # 0.8348
        return False
    else:
        return True
    

def calc_out_conv(len_, kz, p, s):
    return math.floor(((len_ + 2*p - 1*(kz-1) -1) / s) + 1)
def calc_out_pool(len_, kz, p, s, pool):
    if pool == 'max':
        return math.floor(((len_ + 2*p - 1*(kz-1) -1) / s) + 1)
    else:
        return math.floor(((len_ + 2*p - 1*(kz)) / s) + 1)


def poolings(pool):
    if pool == 'max':
        return nn.MaxPool1d
    elif pool == 'avg':
        return nn.AvgPool1d
    elif pool == 'None':
        return None
    else:
        raise NotImplemented("Currently no other activation functions under consideration")


def check_cnn_stop_trial(config):
    in_l = config["in_length"] = 250
    num_layers = config["num_layers"] 
    kz = [config["kernel_sizes"]] * num_layers
    s = [config["strides"]] * num_layers
    p = config['padding']
    pool = config['pool']
    pool_fn = poolings(pool)
    
    out_l = calc_out_conv(in_l, kz=kz[0], s=s[0], p=p)
    if pool_fn:
        out_l = calc_out_pool(out_l, kz=kz[0], s=s[0], p=p, pool=pool)
    for i in range(1, num_layers):
        out_l = calc_out_conv(out_l, kz=kz[i], s=s[i], p=p)
        if out_l <= 0:
            return True
        if pool_fn:
            out_l = calc_out_pool(out_l, kz=kz[i],s=s[i], p=p, pool=pool)
            if out_l <= 0:
                return True
    
    if out_l <= 0:
        return True
    return False


def check_convrff_stop_trial(config):
    length = config["in_length"] = 250
    num_layers = config["num_layers"]
    kz = config["kernel_size"]
    s = config["stride"]
    p = config['padding']
    
    for i in range(num_layers*2):
        length = ((length + 2 * p - 1 * (kz - 1) - 1) / s) + 1
        length = math.floor(length)

    if length <= 0:
        return True
    else:   
        return False
    
    
def max_trial_callback(study, trial):
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE or t.state == optuna.trial.TrialState.RUNNING])
    print()
    print(f'current num completes: {n_complete}')
    print()
    if n_complete >= 100:
        study.stop()


def objective(args, trial: optuna.trial.Trial) -> float:
    config = get_config(args, trial)
    
    if args.model == 'cnn':
        if check_cnn_stop_trial(config):
            if args.objectives == 'multi':
                return -np.inf, np.inf
            else:
                return -np.inf
            
    elif args.model == 'convrff':
        if check_convrff_stop_trial(config):
            if args.objectives == 'multi':
                return -np.inf, np.inf
            else:
                return -np.inf

    
    if args.model == 'enc':
        if not(config['valid_embed_dim']):
            if args.objectives == 'multi':
                return -np.inf, np.inf
            else:
                return -np.inf
    

    datamodule, config = get_datamodule(args, config)
    
    model, config = get_model(args, config)
    #early_stopping = EarlyStopping('val_loss')
    
    # configure trainer
    tb_logger = TensorBoardLogger(
        args.logdir, name=args.experiment_name, version="",) if not args.test_only else None

    if args.objectives == 'multi':
        trainer = pl.Trainer(
            logger=tb_logger,
            limit_val_batches=args.percent_valid_examples,
            enable_checkpointing=False,
            max_epochs=args.epochs,
            accelerator="auto",
            devices="auto",
            )
    else:
        if not valid_model_size(config):
            return -np.inf

        trainer = pl.Trainer(
        logger=tb_logger,
        limit_val_batches=args.percent_valid_examples,
        enable_checkpointing=False,
        max_epochs=args.epochs,
        accelerator=args.accelerator, devices=[args.device] if args.accelerator == 'gpu' else "auto",
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val/val_macro")],
        deterministic=True,
        )
        
    trainer.logger.log_hyperparams(config)
    trainer.fit(model, datamodule=datamodule)
    
    try: 
        if trainer.callback_metrics["val/val_macro"] is None:
            if args.objectives == 'multi':
               return -np.inf, np.inf
            else:
                return -np.inf
        else:
            if args.objectives == 'multi':
                return trainer.callback_metrics["val/val_macro"].item(), config["model_size"]
            else:
                return trainer.callback_metrics["val/val_macro"].item()
    except: 
        if args.objectives == 'multi':
            return -np.inf, np.inf
        else:
            return -np.inf


def multi_run(args, seed, study_name, study_dir, storage, n_trials):
    sampler = get_sampler(args, seed, study_dir)
    study = optuna.load_study(sampler=sampler, study_name=study_name, storage=storage) 
    study.optimize(lambda trial: objective(args, trial), n_trials=n_trials, n_jobs=1, callbacks=[max_trial_callback])



def cli_main():

    start = time.time()
    parser = ArgumentParser()
    parser = parse_args(parser)

    args = parser.parse_args()
    print(args)
    
    torch.manual_seed(args.seed)

    mp.set_start_method('spawn')# good solution !!!!

    if args.model == 'cpc':
        experiment_name = args.study_name + f'_id={args.id}_'
    else:
        experiment_name = args.study_name + f'_id={args.id}' #get_experiment_name(args)
    args.experiment_name = experiment_name

    if not os.path.isdir(f"{args.study_path}/{args.objectives}/{args.optimization}"):
        os.makedirs(f"{args.study_path}/{args.objectives}/{args.optimization}")
        print('created dir: ', f"{args.study_path}/{args.objectives}/{args.optimization}")

    init_logger(log_dir=join(f'{args.logdir}'))
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = args.experiment_name  # Unique identifier of the study.
    study_dir = f"{args.study_path}/{args.objectives}/{args.optimization}/{study_name}"

    storage = JournalStorage(JournalFileStorage(f"{study_dir}.log"))
    if args.objectives == 'multi':
        study = optuna.create_study(directions=['maximize', 'minimize'], study_name=study_name, storage=storage, load_if_exists=True)
        
        processes = []
        n_trials = args.num_study_samples // args.num_jobs
        print(f'Running optuna in parallel for {n_trials} trials.')

        for i in range(args.num_jobs):
            p = mp.Process(target = multi_run, args=(args, args.seed+i, study_name, study_dir, storage, n_trials))
            print(f"Process {i} start.")
            p.start()
            processes.append(p)

        for i,p in enumerate(processes):
            p.join()
            print(f"Process {i} finish.")

        print(f'After MPI: {len(study.get_trials())}')

        remaining_trials = 100 % args.num_jobs # 100 max trials with 16 cpus
        if remaining_trials > 0:
            print(f"running remaining {remaining_trials} trials")
            sampler = get_sampler(args, args.seed, study_dir)
            study = optuna.load_study(sampler=sampler, study_name=study_name, storage=storage)
            study.optimize(lambda trial: objective(args, trial), n_trials=remaining_trials, n_jobs=1, callbacks=[max_trial_callback])

        # Save the sampler with pickle to be loaded later.
        with open(f"{study_dir}_sampler.pkl", "wb") as fout:
            joblib.dump(study.sampler, fout)

        del study.sampler
        with open(f"{study_dir}.pkl", "wb") as fp:
            joblib.dump(study, fp)

        #optuna.visualization.plot_pareto_front(study, target_names=["AUC", "model_size"])
        print(f"Number of trials on the Pareto front: {len(study.best_trials)}")

        trial_with_highest_auc = max(study.best_trials, key=lambda t: t.values[0])
        print(f"Trial with highest accuracy: ")
        print(f"\tnumber: {trial_with_highest_auc.number}")
        print(f"\tparams: {trial_with_highest_auc.params}")
        print(f"\tvalues: {trial_with_highest_auc.values}")
        
        end = time.time()
        print("\nTime elapsed ", timedelta(seconds=end-start))
    
    else:
        pruner = optuna.pruners.MedianPruner() #optuna.pruners.NopPruner()

        """if args.optimization == 'bayesian':
            sampler = optuna.samplers.TPESampler(seed=args.seed)
        elif args.optimization == 'random':
            sampler = optuna.samplers.RandomSampler(seed=args.seed)
        elif args.optimization == 'population':
            sampler = optuna.samplers.NSGAIISampler(seed=args.seed)
        else:
            raise NotImplementError('Only bayesian, random, and population samplers are supported.')
        """
        sampler = get_sampler(args, study_dir)
        study = get_study_single(args, study_dir, direction="maximize", sampler=sampler, pruner=pruner, study_name=study_name, storage=storage, load_if_exists=True)

        study.optimize(lambda trial: objective(args, trial), n_trials=args.num_study_samples, n_jobs=1, callbacks=[max_trial_callback])
        
        processes = []
        n_trials = args.num_study_samples // args.num_jobs
        print(f'Running optuna in parallel for {n_trials} trials.')

        for i in range(args.num_jobs):
            p = mp.Process(target = multi_run, args=(args, args.seed+i, study_name, study_dir, storage, n_trials))
            print(f"Process {i} start.")
            p.start()
            processes.append(p)

        for i,p in enumerate(processes):
            p.join()
            print(f"Process {i} finish.")

        print(f'After MPI: {len(study.get_trials())}')

        remaining_trials = 100 % args.num_jobs # 100 max trials with 16 cpus
        if remaining_trials > 0:
            print(f"running remaining {remaining_trials} trials")
            sampler = get_sampler(args, args.seed, study_dir)
            study = optuna.load_study(sampler=sampler, study_name=study_name, storage=storage)
            study.optimize(lambda trial: objective(args, trial), n_trials=remaining_trials, n_jobs=1, callbacks=[max_trial_callback])

        # Save the sampler with pickle to be loaded later.
        with open(f"{study_dir}_sampler.pkl", "wb") as fout:
            joblib.dump(study.sampler, fout)

        del study.sampler
        with open(f"{study_dir}.pkl", "wb") as fp:
            joblib.dump(study, fp)

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))



if __name__ == "__main__":
    cli_main()



