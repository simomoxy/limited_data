from platform import architecture
from pyexpat import model
import time
import logging
import os
import pdb
from os.path import exists, join, dirname
import argparse
from argparse import ArgumentParser
import pickle

import math

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
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.model_summary import ModelSummary

from mlxtend.evaluate import confusion_matrix
from clinical_ts.create_logger import create_logger
from clinical_ts.eval_utils_cafa import eval_scores, eval_scores_bootstrap
from clinical_ts.timeseries_utils import aggregate_predictions
from ecg_datamodule import ECGDataModule
from dl_models.modules import *

### RAY
from optimization_utils.ray_config import *

from ray import air, tune
from ray.air import session
from ray.air.config import  ScalingConfig
#from ray.train import CheckpointConfig
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler


logger = create_logger(__name__)

def parse_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument("--study_path", default="")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--dataset",
        dest="target_folder",
        help="used dataset for training",
    )
    parser.add_argument("--logdir", default="./logs")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--label_class", default="label_all")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--input_size", type=int, default=250)
    parser.add_argument("--gpu", action="store_true", default=False)
    parser.add_argument("--nomemmap", action="store_true", default=False)
    parser.add_argument("--test_folds", nargs="+", default=[9, 10], type=int)
    parser.add_argument("--filter_label")
    parser.add_argument("--combination",  default="both")
    parser.add_argument("--test_only", action="store_true", default=False)
    parser.add_argument("--model", default='xresnet1d50')
    parser.add_argument("--rate", default=1.0, type=float)
    parser.add_argument("--d_state", default=8, type=int)
    parser.add_argument("--d_model", default=512, type=int)
    parser.add_argument("--n_layers", default=4, type=int)
    parser.add_argument("--s4_dropout", default=0.2, type=float)
    parser.add_argument("--bn", action='store_true', default=False)
    parser.add_argument("--binary_classification",
                        action='store_true', default=False)
    parser.add_argument("--concat_pooling",
                        action='store_true', default=False)
    parser.add_argument("--normalize", action='store_true', default=False)
    parser.add_argument("--use_meta_information_in_head", action='store_true', default=False)
    parser.add_argument("--cpc_bn_encoder", action='store_true', default=False)
    parser.add_argument("--down_sample_rate", default=0.0, type=float)
    parser.add_argument("--device", default=0)
    parser.add_argument("--accelerator", default="gpu", type=str)
    parser.add_argument("--objectives", default='single', type=str)
    parser.add_argument("--optimization", default='ASHA', type=str)
    parser.add_argument("--num_study_samples", default=10, type=int)
    parser.add_argument("--num_worker", default=1, type=int)
    parser.add_argument("--cpus_per_worker", default=1, type=int)
    parser.add_argument("--gpus_per_worker", default=1, type=float)
    parser.add_argument("--SOTA_config", action="store_true", default=False)
    parser.add_argument("--id", default=0, type=int)
    parser.add_argument("--study_name", default="ray_study")
    parser.add_argument("--data_seed", default=1, type=int)
    parser.add_argument("--seed", default=0, type=int)
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


class NumValidTrialsStopper(tune.Stopper):
    def __init__(self, num_trials=100):
        self.valid_trials = set()
        self.num_trials = num_trials

    def __call__(self, trial_id: str, result: dict) -> bool:
        if not np.isnan(result["val_macro"]) and result["done"]:
            self.valid_trials.add(trial_id)

    def stop_all(self) -> bool:
        return len(self.valid_trials) >= self.num_trials


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


def cli_main():
    parser = ArgumentParser()
    parser = parse_args(parser)

    args = parser.parse_args()
    print(args)
    torch.manual_seed(args.seed)
    
    experiment_name = args.study_name + f'_id={args.id}' #get_experiment_name(args)
    args.experiment_name = experiment_name
     
   
    # configure trainer
    tb_logger = TensorBoardLogger(
        args.logdir, name=experiment_name, version="") if not args.test_only else None
    

    ##### NEEEDED TO CHANGE SINCE RAY.TRAIN.LIGHNTING NOT SUPPORTED
    ##### FOLLOW : https://docs.ray.io/en/latest/tune/examples/tune-vanilla-pytorch-lightning.html#tune-vanilla-pytorch-lightning-ref
    #####
   
    def train(config, args, logger):
            
        if args.model == 'cnn':
            if check_cnn_stop_trial(config):
                session.report({"val_macro": np.nan, "done": True}) 
                
                return 
                
        elif args.model == 'convrff':
            if check_convrff_stop_trial(config):
                session.report({"val_macro": np.nan, "done": True}) 
                
                return
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

        if args.model == 'cnn':
            model = CNN_module(config)
        elif args.model == 'lstm':
            model = LSTM_module(config)
        elif args.model == 'enc':
            model = Transformer_module(config)
        elif args.model == 'transformer':
            model = Transformer_module(config)
        elif args.model == 's4':
            model = S4_module(config)
        elif args.model == 'xresnet':
            model = XRESNET_module(config)
        elif args.model == 'cpc':
            model = CPC_module(config)
        elif args.model == 'rff':
            model = RFF_module(config)
        elif args.model == 'convrff':
            model = ConvRFF_module(config)
        #elif args.model == 'ray_dgp':
        #    model = GP_ECGLightningModel(config, datamodule)
        else:
            raise NotImplemented("Only ray_cnn, ray_lstm, ray_transformerEnc, and ray_transformer supported.")
        
   
        summary = ModelSummary(model)
        config["trainable_parameters"] = summary.trainable_parameters
        config["model_size"] = summary.model_size

        trainer = Trainer(
        max_epochs=args.epochs,
        # If fractional GPUs passed in, convert to int.
        gpus=math.ceil(args.gpus_per_worker),
        logger=logger,
        callbacks=[
            TuneReportCallback(
                {
                    #'model_size': "model_size",
                    "val_loss": "val/total_loss",
                    "val_macro": "val/val_macro",
                    "val_macro_agg": "val/val_macro_agg",
                    "val_macro_agg_sig": "val/val_macro_agg_sig",
                    "val_acc": "val/val_acc",
                    "train_loss": "train/total_loss",
                    "train_macro": "train/train_macro",
                    "train_macro_agg": "train/train_macro_agg",
                    "train_macro_agg_sig": "train/train_macro_agg_sig", 
                    "train_acc": "train/train_acc",
                    
                },
                on="validation_end")
        ],
        accelerator=args.accelerator, devices=[args.device] if args.accelerator == 'gpu' else "auto",
        val_check_interval=1.0,
        )
        trainer.fit(model, datamodule)

    def tune_model(args, logger):
        if args.model == 'cnn':
            config=cnn_config()
            config["model"] = 'cnn'  
            config["SOTA_ARCHITECTURE"] = args.SOTA_config
        elif args.model == 'lstm':
            config=lstm_config()   
            config["model"] = 'lstm'
            config["SOTA_ARCHITECTURE"] = args.SOTA_config   
        elif args.model == 'enc':
            config=transformerEnc_config()
            config["model"] = 'enc'
            config["SOTA_ARCHITECTURE"] = args.SOTA_config 
        elif args.model == 'transformer':
            config=transformer_config()
            config["model"] = 'transformer' 
            config["SOTA_ARCHITECTURE"] = args.SOTA_config
        elif args.model == 's4':
            config=s4_config(args.SOTA_config)
            config["model"] = 's4' 
            config["SOTA_ARCHITECTURE"] = args.SOTA_config
        elif args.model == 'xresnet':
            config=xresnet_config(args.SOTA_config)
            config["model"] = 'xresnet'
            config["SOTA_ARCHITECTURE"] = args.SOTA_config 
        elif args.model == 'cpc':
            config=cpc_config(args.SOTA_config)
            config["model"] = 'cpc'
            config["SOTA_ARCHITECTURE"] = args.SOTA_config 
        elif args.model == 'rff':
            config=rff_config(args.SOTA_config)
            config["model"] = 'rff'
            config["SOTA_ARCHITECTURE"] = args.SOTA_config
        elif args.model == 'convrff':
            config=convrff_config(args.SOTA_config)
            config["model"] = 'convrff'
            config["SOTA_ARCHITECTURE"] = args.SOTA_config
        else: 
            raise Exception("model {} not found".format(args.model))

        scheduler = ASHAScheduler(
        max_t=args.epochs,
        grace_period=1,
        reduction_factor=2)
        
        reporter = CLIReporter(
        parameter_columns=list(config.keys()),
        metric_columns=[
            'training_iteration', #'model_size',
            'val_macro', 'val_macro_agg', 'val_macro_agg_sig', 'val_loss', 'val_acc',
            'train_macro', 'train_macro_agg', 'train_macro_agg_sig', 'train_loss', 'train_acc'    
        ]
        )
 
        train_fn_with_parameters = tune.with_parameters(train,
                                                    args=args,
                                                    logger=logger, 
                                                    )
        resources_per_trial = {"num_workers": args.num_workers, "cpu": args.cpus_per_worker, "gpu": args.gpus_per_worker}
        

        tuner = tune.Tuner(
            tune.with_resources(
            train_fn_with_parameters,
            resources=resources_per_trial
            ),
            tune_config=tune.TuneConfig(
            metric="val_macro",
            mode="max",
            scheduler=scheduler,
            num_samples=args.num_study_samples,
            ),
            run_config=air.RunConfig(
            local_dir=args.study_path,
            name=args.experiment_name,
            progress_reporter=reporter,
            stop=NumValidTrialsStopper(args.num_study_samples),
            checkpoint_config=air.CheckpointConfig(
                    num_to_keep=1
                ),
            ),
            param_space = config, 
        )
        results = tuner.fit()

        print("Best hyperparameters found were: ", results.get_best_result().config)


    tune_model(args, tb_logger)

if __name__ == "__main__":
    cli_main()
