import numpy as np
from ray import tune


###########################################################################################
###########################################################################################
###########################################################################################

def cnn_config():
    return {
        "in_features": tune.choice([12]),
        "in_length": tune.choice([250]),
        "num_layers": tune.choice([1, 2, 3, 4, 5]),
        "feature_sizes": tune.sample_from(lambda _: 2**np.random.randint(2, 8)),
        "kernel_sizes": tune.choice([3, 5, 7]),
        "strides": tune.choice([1, 2]),
        "act_fun": tune.choice(["relu", "gelu", "elu"]),
        "padding": tune.choice([0, 1]),
        "pool": tune.choice(["avg", "max", "None"]),
        "normalization": tune.choice(["BN", "LN", "None"]),
        "batch_size": tune.sample_from(lambda _: 2**np.random.randint(2, 7)),
        "global_pool": tune.choice([True, False]),
        "dropout": tune.uniform(0.0, 0.75),
        "lr": tune.loguniform(1e-5, 1e-1),
        "wd": tune.loguniform(1e-5, 1e-1),
    }

###########################################################################################
###########################################################################################
###########################################################################################

def lstm_config():
    return {
        "input_size": tune.choice([12]),
        "hidden_size": tune.sample_from(lambda _: 2**np.random.randint(1, 9)),
        "num_layers": tune.choice([1, 2, 3, 4, 5]),
        "lstm_bias": tune.choice([True, False]),
        "lstm_dropout": tune.uniform(0.0, 0.75),
        "bidirectional": tune.choice([True, False]),
        "act_fun": tune.choice(["relu", "gelu", "elu"]),
        "normalization": tune.choice(["BN", "LN", "None"]),
        "batch_size": tune.sample_from(lambda _: 2**np.random.randint(2, 7)),
        "lr": tune.loguniform(1e-5, 1e-1),
        "wd": tune.loguniform(1e-5, 1e-1),
    }

###########################################################################################
###########################################################################################
###########################################################################################

def transformerEnc_config():
    config = {
        "batch_size": tune.sample_from(lambda _: 2**np.random.randint(2, 7)),
        "input_channels": 12,
        "seq_length": 250,
        "d_model": tune.sample_from(lambda _: 2**np.random.randint(1, 9)),
        "nhead": tune.choice([1, 2, 4, 8]),
        "num_layers": tune.choice([1, 2, 3, 4, 5]),
        "dropout": tune.uniform(0.0, 0.75),
        "decode": False,
        "masking": False,
        "clf_pool": tune.choice([None, 'self_attention', 'adaptive_concat_pool']),#,, 'mean']),
        "lr": tune.loguniform(1e-5, 1e-1),
        "wd": tune.loguniform(1e-5, 1e-1),
    }
    #config["valid_embed_dim"] = (config["d_model"] % config["nhead"]) == 0
    return config
    

###########################################################################################
###########################################################################################
###########################################################################################

def transformer_config():
    return {
        "batch_size": tune.sample_from(lambda _: 2**np.random.randint(2, 7)),
        "input_channels": 12,
        "seq_length": 250,
        "d_model": tune.sample_from(lambda _: 2**np.random.randint(1, 9)),
        "nhead": tune.choice([1, 2, 4, 8]),
        "num_layers": tune.choice([1, 2, 3, 4, 5]),
        "dropout": tune.uniform(0.0, 0.75),
        "decode": True,
        "masking": tune.choice([False, True]),
        "clf_pool": tune.choice([None, 'self_attention', 'adaptive_concat_pool']),# 'mean']),
        "lr": tune.loguniform(1e-5, 1e-1),
        "wd": tune.loguniform(1e-5, 1e-1),
   }

###########################################################################################
###########################################################################################
###########################################################################################

def s4_config(SOTA_config=False):
    if SOTA_config:
        return {
            'd_input': 12,
            'l_max': 250,
            'd_state': 8,
            'd_model': 512,
            'num_layers': 4,
            'dropout': 0.2,
            'normalization': "LN",
            'bidirectional': False,
            "batch_size": tune.sample_from(lambda _: 2**np.random.randint(2, 7)),
            "lr": tune.loguniform(1e-5, 1e-1),
            "wd": tune.loguniform(1e-5, 1e-1),
        }
    else:
        return {
            'd_input': 12,
            'l_max': 250,
            'd_state': tune.sample_from(lambda _: 2**np.random.randint(1, 9)),
            'd_model': tune.sample_from(lambda _: 2**np.random.randint(1, 9)),
            "num_layers": tune.choice([1, 2, 3, 4, 5]),
            "dropout": tune.uniform(0.0, 0.75),
            'normalization': tune.choice(["BN", "LN", None]),#[True, False]),
            'bidirectional': tune.choice([True, False]),
            "batch_size": tune.sample_from(lambda _: 2**np.random.randint(2, 7)),
            "lr": tune.loguniform(1e-5, 1e-1),
            "wd": tune.loguniform(1e-5, 1e-1),
        }
    
###########################################################################################
###########################################################################################
###########################################################################################

def xresnet_config(SOTA_config=False):
    if SOTA_config:
        return {
            "xresnet": 'xresnet1d50',
            "batch_size": tune.sample_from(lambda _: 2**np.random.randint(2, 7)),
            "lr": tune.loguniform(1e-5, 1e-1),
            "wd": tune.loguniform(1e-5, 1e-1),
    }
    else:
        return {
            "xresnet": tune.choice(['xresnet1d18', 'xresnet1d18_deep', 'xresnet1d18_deeper', 'xresnet1d34', 'xresnet1d34_deep', 'xresnet1d34_deeper', 'xresnet1d50', 'xresnet1d50_deep', 'xresnet1d50_deeper', 'xresnet1d101']),
            "batch_size": tune.sample_from(lambda _: 2**np.random.randint(2, 7)),
            "lr": tune.loguniform(1e-5, 1e-1),
            "wd": tune.loguniform(1e-5, 1e-1),

   }
    
###########################################################################################
###########################################################################################
###########################################################################################

def cpc_config(SOTA_config=False):
    if SOTA_config:
        return {
            "batch_size": tune.sample_from(lambda _: 2**np.random.randint(2, 7)),
            'input_channels': 12,
            'strides': [1]*4,#[5,4,2,2,2],
            'kss': [1]*4,#[10,8,4,4,4],
            'features': [512]*4,
            'bn_encoder': True,
            'n_hidden': 512,
            'n_layers': 2,
            'mlp': False,
            'lstm': True,
            'bias_proj': False,
            'concat_pooling': True,
            'ps_head': 0.5,
            'lin_ftrs_head': [512],
            'bn_head': True,
            'skip_encoder': False,
            "lr": tune.loguniform(1e-5, 1e-1),
            "wd": tune.loguniform(1e-5, 1e-1),
        }

    else:
        depth = np.random.randint(1, 6),
        kernels = [2, 4, 6, 8, 10, 12]
        strides = [6, 5, 4, 3, 2, 1]

        return {
            "batch_size": tune.sample_from(lambda _: 2**np.random.randint(2, 7)),
            'input_channels': 12,
            'strides': tune.sample_from(lambda _: np.random.choice(strides, size=depth)),
            'kss': tune.sample_from(lambda _: np.random.choice(kernels, size=depth)),
            'features': tune.sample_from(lambda _: 2**np.random.randint(4, 11, size=depth)),
            'bn_encoder': tune.choice([True, False]),
            'n_hidden': tune.sample_from(lambda _: 2**np.random.randint(2, 11)),
            'n_layers': tune.choice([1, 2, 3, 4, 5]),
            'mlp': tune.choice([True, False]),
            'lstm': tune.choice([True, False]),
            'bias_proj': tune.choice([True, False]),
            'concat_pooling': tune.choice([True, False]),
            'ps_head': tune.choice([0.0, 0.1, 0.25, 0.5, 0.75]),
            'lin_ftrs_head': tune.sample_from(lambda _: [2**np.random.randint(2, 11)]),
            'bn_head': tune.choice([True, False]),
            'skip_encoder': tune.choice([True, False]),
            "lr": tune.loguniform(1e-5, 1e-1),
            "wd": tune.loguniform(1e-5, 1e-1),

        }
    
###########################################################################################
###########################################################################################
###########################################################################################

def rff_config(SOTA_config=False):
    return {
        "kernel": tune.choice(['rbf', 'arccos']),
        "mc": tune.choice([1, 5, 10, 15, 20]),
        "num_layers": tune.choice([1, 2, 3, 4, 5]),
        "in_dims": 250,
        "N_RFs": tune.sample_from(lambda _: 2**np.random.randint(1, 6)),
        #"out_dims": [],
        "lr": tune.loguniform(1e-5, 1e-1),
        "wd": tune.loguniform(1e-5, 1e-1),
        "batch_size": tune.sample_from(lambda _: 2**np.random.randint(2, 7)),
    }

###########################################################################################
###########################################################################################
###########################################################################################

def convrff_config(SOTA_config=False):
    return {
        "kernel": tune.choice(['rbf', 'arccos']),
        "mc": tune.choice([1, 5, 10, 15, 20]),
        "num_layers": tune.choice([1, 2, 3, 4, 5]),
        "in_channels": 12,
        "feature_sizes": tune.sample_from(lambda _: 2**np.random.randint(1, 9)),
        #"out_dim":
        "length": 250,
        "global_pool": tune.choice([False, True]),
        "kernel_size": tune.choice([3, 5, 7]),
        "stride": tune.choice([1, 2]),
        "padding": tune.choice([0, 1]),
        "group": 1,
        "lr": tune.loguniform(1e-5, 1e-1),
        "wd": tune.loguniform(1e-5, 1e-1),
        "batch_size": tune.sample_from(lambda _: 2**np.random.randint(2, 7)),
    }
