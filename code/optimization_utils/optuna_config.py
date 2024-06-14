import numpy as np
import optuna


###########################################################################################
###########################################################################################
###########################################################################################

def cnn_config(trial: optuna.trial.Trial):
    config = dict()
    config["batch_size"] = trial.suggest_categorical("batch_size", [2**2, 2**3, 2**4, 2**5, 2**6, 2**7])
    config["in_features"] = 12
    config["in_length"] = 250
    
    config["num_layers"] = trial.suggest_int("num_layers", 1, 5)
    config["feature_sizes"] = trial.suggest_categorical("feature_sizes", [2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8])
    config["kernel_sizes"] = trial.suggest_categorical("kernel_sizes", [3, 5, 7])
    config["padding"] = trial.suggest_int("padding", 0, 1)
    config["strides"] = trial.suggest_int("strides", 1, 2)
    config["dropout"] = trial.suggest_float("dropout", 0.0, 0.75)
    
    config["pool"] = trial.suggest_categorical("pool", ["avg", "max", "None"])
    config["normalization"] = trial.suggest_categorical("normalization", ["BN", "LN", "None"])
    config["act_fun"] = trial.suggest_categorical("act_fun", ["relu", "gelu", "elu"])
    config["global_pool"] = trial.suggest_categorical("global_pool", [True, False])
    
    config["lr"] = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    config["wd"] = trial.suggest_float("wd", 1e-5, 1e-1, log=True)
    

    return config

###########################################################################################
###########################################################################################
###########################################################################################

def lstm_config(trial: optuna.trial.Trial): 
    config = dict()
    config["batch_size"] = trial.suggest_categorical("batch_size", [2**2, 2**3, 2**4, 2**5, 2**6, 2**7])
    config["input_size"] = 12
    #config["in_length"] = 250
    config["hidden_size"] = trial.suggest_categorical("hidden_size", [2, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9])
    config["num_layers"] = trial.suggest_int("num_layers", 1, 5)
    config["lstm_bias"] = trial.suggest_categorical("lstm_bias", [True, False])
    config["lstm_dropout"] = trial.suggest_float("lstm_dropout", 0.0, 0.75)
    config["bidirectional"] = trial.suggest_categorical("bidirectional", [True, False])
    config["act_fun"] = trial.suggest_categorical("act_fun", ["relu", "gelu", "elu"])
    config["normalization"] = trial.suggest_categorical("normalization", ["BN", "LN", "None"])

    config["lr"] = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    config["wd"] = trial.suggest_float("wd", 1e-5, 1e-1, log=True)

    return config

###########################################################################################
###########################################################################################
###########################################################################################

def transformerEnc_config(trial: optuna.trial.Trial):
    config = dict()
    config["batch_size"] = trial.suggest_categorical("batch_size", [2**2, 2**3, 2**4, 2**5, 2**6, 2**7])
    config["input_channels"] = 12
    config["seq_length"] = 250
    config["d_model"] = trial.suggest_categorical("d_model", [2, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9])
    config["nhead"] = trial.suggest_categorical("nhead", [1, 2, 4, 8])
    config["valid_embed_dim"] = (config["d_model"] % config["nhead"]) == 0
    config["num_layers"] = trial.suggest_int("num_layers", 1, 5)
    config["dropout"] = trial.suggest_float("dropout", 0.0, 0.75)
    config["decode"] = False
    config["masking"] = False
    config["clf_pool"] = trial.suggest_categorical("clf_pool", [None, 'self_attention', 'adaptive_concat_pool'])

    config["lr"] = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    config["wd"] = trial.suggest_float("wd", 1e-5, 1e-1, log=True)

    return config


###########################################################################################
###########################################################################################
###########################################################################################

def transformer_config(trial: optuna.trial.Trial):
    config = dict()
    config["batch_size"] = trial.suggest_categorical("batch_size", [2**2, 2**3, 2**4, 2**5, 2**6, 2**7])
    config["input_channels"] = 12
    config["seq_length"] = 250
    config["d_model"] = trial.suggest_categorical("d_model", [2, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9])
    config["nhead"] = trial.suggest_categorical("nhead", [1, 2, 4, 8])
    config["num_layers"] = trial.suggest_int("num_layers", 1, 5)
    config["dropout"] = trial.suggest_float("dropout", 0.0, 0.75)
    config["decode"] = True
    config["masking"] = trial.suggest_categorical("masking", [True, False])
    config["clf_pool"] = trial.suggest_categorical("clf_pool", [None, 'self_attention', 'adaptive_concat_pool'])

    config["lr"] = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    config["wd"] = trial.suggest_float("wd", 1e-5, 1e-1, log=True)

    return config

###########################################################################################
###########################################################################################
###########################################################################################

def s4_config(trial: optuna.trial.Trial, SOTA_config: bool=False):
    config = dict()
    config["batch_size"] = trial.suggest_categorical("batch_size", [2**2, 2**3, 2**4, 2**5, 2**6, 2**7])
    config["d_input"] = 12
    config["l_max"] = 250

    if SOTA_config:
        config["d_model"] = 512
        config["d_state"] = 8
        config["num_layers"] = 4
        config["dropout"] = 0.2
        config['normalization'] = "LN"
        config['bidirectional'] = False
    
    else:
        config["d_model"] = trial.suggest_categorical("d_model", [2, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9])
        config["d_state"] = trial.suggest_categorical("d_state", [2, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9])
        config["num_layers"] = trial.suggest_int("num_layers", 1, 5)
        config["dropout"] = trial.suggest_float("dropout", 0.0, 0.75)
        config['normalization'] = trial.suggest_categorical("normalization", ["BN", "LN", "None"])
        config['bidirectional'] = trial.suggest_categorical("bidirectional", [True, False])

    config["lr"] = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    config["wd"] = trial.suggest_float("wd", 1e-5, 1e-1, log=True)

    return config
    
###########################################################################################
###########################################################################################
###########################################################################################

def xresnet_config(trial: optuna.trial.Trial, SOTA_config: bool=False):
    config = dict()
    config["batch_size"] = trial.suggest_categorical("batch_size", [2**2, 2**3, 2**4, 2**5, 2**6, 2**7])

    if SOTA_config:
        config['xresnet'] = 'xresnet1d50'
    else:
        config['xresnet'] = trial.suggest_categorical("xresnet", ['xresnet1d18', 'xresnet1d18_deep', 'xresnet1d18_deeper', 'xresnet1d34', 'xresnet1d34_deep', 'xresnet1d34_deeper', 'xresnet1d50', 'xresnet1d50_deep', 'xresnet1d50_deeper', 'xresnet1d101']) 
    
    config["lr"] = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    config["wd"] = trial.suggest_float("wd", 1e-5, 1e-1, log=True)

    return config
    
###########################################################################################
###########################################################################################
###########################################################################################

def cpc_config(trial: optuna.trial.Trial, SOTA_config: bool=False):
    config = dict()
    config["batch_size"] = trial.suggest_categorical("batch_size", [2**2, 2**3, 2**4, 2**5, 2**6, 2**7])
    config["input_channels"] = 12

    if SOTA_config:
        config['strides'] = [1]*4
        config['kss'] = [1]*4
        config['features'] = [512]*4
        config['bn_encoder'] = True
        config['n_hidden'] = 512
        config['n_layers'] = 2
        config['mlp'] = False
        config['lstm'] = True
        config['bias_proj'] = False
        config['concat_pooling'] = True
        config['lin_ftrs_head'] = [512]
        config['bn_head'] = True
        config['skip_encoder'] = False
        config['ps_head'] = 0.5

    else:
        kernels = [2, 4, 6, 8, 10, 12]
        strides = [6, 5, 4, 3, 2, 1]

        config['depth'] = trial.suggest_int('depth', 1, 6)
        config['strides'] = trial.suggest_categorical("strides", strides)
        config['kss'] = trial.suggest_categorical("kernels", kernels)
        config['features'] = trial.suggest_categorical("features", [2, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9])
        config['bn_encoder'] = trial.suggest_categorical("bn_encoder", [True, False])
        config['n_hidden'] = trial.suggest_categorical("n_hidden", [2, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9])
        config['n_layers'] = trial.suggest_int("num_layers", 1, 5)
        config['mlp'] = trial.suggest_categorical("mlp", [True, False])
        config['lstm'] = trial.suggest_categorical("lstm", [True, False])
        config['bias_proj'] = trial.suggest_categorical("bias_proj", [True, False])
        config['concat_pooling'] = trial.suggest_categorical("concat_pooling", [True, False])
        config['lin_ftrs_head'] = [trial.suggest_categorical("lin_ftrs_head", [2, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9])] # needs to be a list
        config['bn_head'] = trial.suggest_categorical("bn_head", [True, False])
        config['skip_encoder'] = trial.suggest_categorical("skip_encoder", [True, False])
        config["ps_head"] = trial.suggest_float("dropout", 0.0, 0.75)

    config["lr"] = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    config["wd"] = trial.suggest_float("wd", 1e-5, 1e-1, log=True)

    return config
    
###########################################################################################
###########################################################################################
###########################################################################################

def rff_config(trial: optuna.trial.Trial):
    config = dict()
    config["batch_size"] = trial.suggest_categorical("batch_size", [2**2, 2**3, 2**4, 2**5, 2**6, 2**7])
    config['in_dims'] = 250
    config["kernel"] = trial.suggest_categorical("kernel", ["rbf", "arccos"])
    config['mc'] = trial.suggest_categorical('mc', [1, 5, 10, 15, 20])
    config['num_layers'] = trial.suggest_int('num_layers', 1, 5)
    config['N_RFs'] = trial.suggest_categorical("N_RFs", [2, 2**2, 2**3, 2**4, 2**5, 2**6])
    config['global_pool'] = trial.suggest_categorical('global_pool', [False, True])
    
    config["lr"] = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    config["wd"] = trial.suggest_float("wd", 1e-5, 1e-1, log=True)

    return config

###########################################################################################
###########################################################################################
###########################################################################################

def convrff_config(trial: optuna.trial.Trial):
    config = dict()
    config["batch_size"] = trial.suggest_categorical("batch_size", [2**2, 2**3, 2**4, 2**5, 2**6, 2**7])
    config['in_channels'] = 12
    config['length'] = 250
    config["kernel"] = trial.suggest_categorical("kernel", ["rbf", "arccos"])
    config['mc'] = trial.suggest_categorical('mc', [1, 5, 10, 15, 20])
    config['num_layers'] = trial.suggest_int('num_layers', 1, 5)
    config['feature_sizes'] = trial.suggest_categorical("feature_sizes", [2, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9])
    config['global_pool'] = trial.suggest_categorical('global_pool', [False, True])
    config["kernel_size"] = trial.suggest_categorical("kernel_size", [3, 5, 7])
    config["stride"] = trial.suggest_categorical("stride", [1, 2])
    config["padding"] = trial.suggest_categorical("padding", [0, 1])
    config["group"] = 1

    config["lr"] = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    config["wd"] = trial.suggest_float("wd", 1e-5, 1e-1, log=True)

    return config
