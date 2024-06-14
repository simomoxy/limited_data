import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch import Tensor
from torch.optim import AdamW

from mlxtend.evaluate import confusion_matrix
from dl_models.models import S4Model, LSTM, CNN, Transformer, RFF, ConvRFF
from dl_models.cpc import CPCModel
from dl_models.ecg_resnet import ECGResNet

import sys
from pathlib import Path
# Get the parent directory
# Add the parent directory to sys.path
P = Path(__file__).parents[1] / '/clinical_ts/'
sys.path.append(P)
from clinical_ts.eval_utils_cafa import eval_scores, eval_scores_bootstrap
from clinical_ts.timeseries_utils import aggregate_predictions


def evaluate_macro(preds, targets, trainer, idmap):
    # for val sanity check TODO find cleaner solution
    idmap = idmap[: preds.shape[0]]
    lbl_itos = trainer.datamodule.lbl_itos
    scores = eval_scores(targets, preds, classes=lbl_itos, parallel=True)
    preds_agg, targs_agg = aggregate_predictions(preds, targets, idmap)
    scores_agg = eval_scores(targs_agg, preds_agg,
                             classes=lbl_itos, parallel=True)
    macro = scores["label_AUC"]["macro"]
    macro_agg = scores_agg["label_AUC"]["macro"]
    return macro, macro_agg, scores_agg


class ECGLightningModel(pl.LightningModule):
    def __init__(
        self,
        model,
        batch_size,
        num_samples,
        lr=0.001,
        wd=0.001,
        loss_fn=F.binary_cross_entropy_with_logits,  # nn.CrossEntropyLoss()
        opt=AdamW,
        cal_acc=True,
        rate=1.0,
        sigmoid_eval=True,
        save_preds=False,
        use_meta_information_in_head=False,
        model_size=None,
        **kwargs
    ):
        """
        Args:
            create_model: function that returns torch model that is used for training
            batch_size: batch size
            lr: learning rate
            loss_fn : loss function
            opt: optimizer

        """

        super(ECGLightningModel, self).__init__()
        self.save_hyperparameters(logger=True)
        self.model = model
        self.epoch = 0
        self.ce = isinstance(loss_fn, nn.CrossEntropyLoss)
        self.cal_acc = self.ce or cal_acc
        self.train_scores = None
        self.val_scores = None
        self.test_scores = None
        self.sigmoid_eval=sigmoid_eval
        self.save_preds = save_preds
        self.use_meta_information_in_head = use_meta_information_in_head
        if use_meta_information_in_head:
            logger.info("Use meta information in Lightning Model")

        self.train_preds = []
        self.train_targets = []
        self.train_loss = 0
        self.train_macro = 0
        self.train_macro_agg = 0
        self.train_sigmacro = 0
        self.train_acc = 0

        self.model_size = model_size

    def configure_optimizers(self):
        optimizer = self.hparams.opt(self.model.parameters(
        ), lr=self.hparams.lr, weight_decay=self.hparams.wd)

        return optimizer

    def forward(self, x, eval=False):
        return self.model(x.float())

    def forward_with_meta(self, x, meta_feats, eval=False):
        return self.model.forward_with_meta(x.float(), meta_feats)

    def training_step(self, batch, batch_idx):
        if self.use_meta_information_in_head:
            x, targets, meta = batch
            preds = self.forward_with_meta(x, meta)
        else:
            x, targets = batch
            preds = self(x)

        if self.ce:
            preds = nn.Softmax(dim=1)(preds)
            loss = self.hparams.loss_fn(preds, targets)
        else:
            loss = self.hparams.loss_fn(preds, targets)

        self.train_preds.append(preds.detach().cpu())
        self.train_targets.append(targets.detach().cpu())

        return {"loss": loss.cpu()}

    def training_epoch_end(self, outputs):
        epoch_loss = torch.tensor(
            [get_loss_from_output(output, key="loss") for output in outputs]
        ).mean()
        
        if epoch_loss.isnan():
            self.train_loss = None
            self.train_macro = None
            self.train_macro_agg = None
            self.train_sigmacro = None
            self.train_acc = None
            
            return
        
        self.log("train/total_loss", epoch_loss, on_step=False, on_epoch=True)
        self.log("lr", self.hparams.lr, on_step=False, on_epoch=True)

        preds = torch.cat(self.train_preds, dim=0).numpy()
        targets = torch.cat(self.train_targets, dim=0).numpy()

        macro, macro_agg, scores_agg = evaluate_macro(
            preds, targets, self.trainer, self.trainer.datamodule.train_idmap)
        if self.sigmoid_eval:
            preds = torch.sigmoid(Tensor(preds)).numpy()
        sigmacro, sigmacro_agg, sigscores_agg = evaluate_macro(
            preds, targets, self.trainer, self.trainer.datamodule.train_idmap)


        hard_preds = np.zeros(preds.shape)
        hard_preds[torch.arange(len(hard_preds)), preds.argmax(axis=1)] = 1
        acc = (hard_preds.argmax(axis=1) ==
                   targets.argmax(axis=1)).sum()/len(targets)

        #self.log("train/train_acc", acc)

        self.train_preds.clear()
        self.train_targets.clear()


        self.train_loss = epoch_loss
        self.train_macro = macro
        self.train_macro_agg = macro_agg
        self.train_sigmacro = sigmacro_agg
        self.train_acc = acc

    def validation_step(self, batch, batch_idx):
        if self.use_meta_information_in_head:
            x, targets, meta = batch
            preds = self.forward_with_meta(x, meta)
        else:
            x, targets = batch
            preds = self(x)
        if self.ce:
            preds = nn.Softmax(dim=1)(preds)
            loss = self.hparams.loss_fn(preds, targets)
        else:
            loss = self.hparams.loss_fn(preds, targets)
        results = {
            "val_loss": loss.cpu(),
            "preds": preds.cpu(),
            "targets": targets.cpu(),
        }
        return results

    def test_step(self, batch, batch_idx):
        if self.use_meta_information_in_head:
            x, targets, meta = batch
            preds = self.forward_with_meta(x, meta)
        else:
            x, targets = batch
            preds = self(x)
        if self.ce:
            preds == nn.Softmax(dim=1)(preds)
            loss = self.hparams.loss_fn(preds, targets)
        else:
            loss = self.hparams.loss_fn(preds, targets)
        results = {
            "test_loss": loss.cpu(),
            "preds": preds.cpu(),
            "targets": targets.cpu(),
        }
        return results

    def validation_epoch_end(self, outputs):
        # outputs[0] when using multiple datasets
        if np.isnan(cat(outputs, "preds")).any() or self.train_loss is None:
            log = {
                "val/total_loss": None,
                "val/val_macro": None,
                "val/val_macro_agg": None,
                "val/val_macro_agg_sig": None,
                "val/val_acc": None,
                "train/total_loss": None,
                "train/train_macro": None,
                "train/train_macro_agg": None,
                "train/train_macro_agg_sig": None,
                "train/train_acc": None,
            }
            return {"val_loss": np.inf, "log": log, "progress_bar": log}
            
        
        preds = cat(outputs, "preds")
        targets = cat(outputs, "targets")
        macro, macro_agg, scores_agg = evaluate_macro(
            preds, targets, self.trainer, self.trainer.datamodule.idmap)

        if self.sigmoid_eval:
            preds = torch.sigmoid(Tensor(preds)).numpy()
        sigmacro, sigmacro_agg, sigscores_agg = evaluate_macro(
            preds, targets, self.trainer, self.trainer.datamodule.idmap)
        val_loss = mean(outputs, "val_loss")

        if True:#self.cal_acc:
            hard_preds = np.zeros(preds.shape)
            hard_preds[torch.arange(len(hard_preds)), preds.argmax(axis=1)] = 1
            acc = (hard_preds.argmax(axis=1) ==
                   targets.argmax(axis=1)).sum()/len(targets)
            #log["val/val_acc"] = acc

            preds_agg, targs_agg = aggregate_predictions(
                preds, targets, self.trainer.datamodule.idmap[: preds.shape[0]])
            preds1 = np.argmax(preds_agg, axis=1)
            targs1 = np.argmax(targs_agg, axis=1)
            cm = confusion_matrix(y_target=targs1,
                                  y_predicted=preds1,
                                  binary=False)
            self.cm = cm
        log = {
            "val/total_loss": val_loss,
            "val/val_macro": macro,
            "val/val_macro_agg": macro_agg,
            "val/val_macro_agg_sig": sigmacro_agg,
            "val/val_acc": acc,
            "train/total_loss": self.train_loss,
            "train/train_macro": self.train_macro,
            "train/train_macro_agg": self.train_macro_agg,
            "train/train_macro_agg_sig": self.train_sigmacro,
            "train/train_acc": self.train_acc,
        }
        if self.model_size:
            log['model_size'] = self.model_size.cpu()
        self.val_scores = scores_agg
        
        
        #log = {k: v.cpu().to(torch.float32) if isinstance(v, torch.Tensor) else v for k, v in log.items()}
        #print()
        #print([(k, v, type(v)) for k,v in log.items()])
        #print()
        log = {k: torch.tensor(v, dtype=torch.float32).cpu() for k, v in log.items()}
        self.log_dict(log)
        return {"val_loss": val_loss, "log": log, "progress_bar": log}

    def test_epoch_end(self, outputs):
        # outputs[0] when using multiple datasets
        preds = cat(outputs, "preds")
        targets = cat(outputs, "targets")
        macro, macro_agg, scores_agg = evaluate_macro(
            preds, targets, self.trainer, self.trainer.datamodule.test_idmap)
        if self.sigmoid_eval:
            preds = torch.sigmoid(Tensor(preds)).numpy()
        if self.save_preds:
            self.preds = preds
            self.targets = targets
        sigmacro, sigmacro_agg, sigscores_agg = evaluate_macro(
            preds, targets, self.trainer, self.trainer.datamodule.test_idmap)
        test_loss = mean(outputs, "test_loss").cpu()
        log = {
            "test/total_loss": test_loss,
            "test/test_macro": macro,
            "test/test_macro_agg": macro_agg,
            "test/test_macro_agg_sig": sigmacro_agg,
        }
        if self.cal_acc:
            hard_preds = np.zeros(preds.shape)
            hard_preds[torch.arange(len(hard_preds)), preds.argmax(axis=1)] = 1
            acc = (hard_preds.argmax(axis=1) ==
                   targets.argmax(axis=1)).sum()/len(targets)
            log["test/test_acc"] = acc
            # preds1 = np.argmax(preds_agg, axis=1)
            # targs1 = np.argmax(targs_agg, axis=1)
            # cm = confusion_matrix(y_target=targs1,
            #                       y_predicted=preds1,
            #                       binary=False)
        log = {k: v.cpu().to(torch.float32) for k, v in log.items() if isinstance(v, torch.Tensor)}
        self.log_dict(log)
        self.test_scores = sigscores_agg

        return {"test_loss": test_loss, "log": log, "progress_bar": log}

    def on_train_start(self):
        self.epoch = 0

    def on_train_epoch_end(self):
        self.epoch += 1

def get_loss_from_output(out, key="minimize"):
    return out[key] if isinstance(out, dict) else get_loss_from_output(out[0], key)


def mean(res, key1, key2=None):
    if key2 is not None:
        return torch.stack([x[key1][key2] for x in res]).mean()
    return torch.stack(
        [x[key1] for x in res if type(x) == dict and key1 in x.keys()]
    ).mean()


def cat(res, key):
    """print(res, key)
    if type(x) == dict and key in x.keys():
        return torch.cat(
            [x[key]]
        )
    else:
        return torch.zeros_like(x)"""
    return torch.cat(
        [x[key] for x in res if type(x) == dict and key in x.keys()]
    ).detach().numpy()

###########################################################################################
###########################################################################################
###########################################################################################

class RFFECGLightningModel(ECGLightningModel):
    def __init__(
        self,
        model,
        batch_size,
        num_samples,
        lr=0.001,
        wd=0.001,
        loss_fn=F.binary_cross_entropy_with_logits,  # nn.CrossEntropyLoss()
        opt=AdamW,
        cal_acc=False,
        rate=1.0,
        sigmoid_eval=True,
        save_preds=False,
        use_meta_information_in_head=False,
        model_size=None,
        **kwargs
    ):
        """
        Args:
            create_model: function that returns torch model that is used for training
            batch_size: batch size
            lr: learning rate
            loss_fn : loss function
            opt: optimizer

        """

        super(RFFECGLightningModel, self).__init__(model, batch_size, num_samples, lr, wd, loss_fn, opt, cal_acc, rate, sigmoid_eval, save_preds, use_meta_information_in_head, model_size)
        self.loss_fn = loss_fn
        
    def training_step(self, batch, batch_idx):
        if self.use_meta_information_in_head:
            x, targets, meta = batch
            preds = self.forward_with_meta(x, meta)
        else:
            x, targets = batch
            x = x.unsqueeze(2)
            bz = x.shape[0]
            preds = self(x)
            preds = preds.view(self.model.mc, bz, -1)

        ell = self.loss_fn(preds, targets.repeat(self.model.mc, 1, 1).float())
        kl = self.model.get_kl()
        loss = kl + ell #kl - ell
        #print(ell, kl, loss)

        self.train_preds.append(preds.mean(dim=0).detach().cpu())
        self.train_targets.append(targets.detach().cpu())

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        if self.use_meta_information_in_head:
            x, targets, meta = batch
            preds = self.forward_with_meta(x, meta)
        else:
            x, targets = batch
            x = x.unsqueeze(2)
            bz = x.shape[0]
            preds = self(x)
            preds = preds.view(self.model.mc, bz, -1)

        ell = self.loss_fn(preds, targets.repeat(self.model.mc, 1, 1).float())
        kl = self.model.get_kl()
        loss = kl + ell  #kl - ell

        #print('preds', preds.shape, 'targets', targets.shape)

        results = {
            "val_loss": loss,
            "preds": preds.mean(dim=0).cpu(),
            "targets": targets.cpu(),
        }

        return results

    def test_step(self, batch, batch_idx):
        if self.use_meta_information_in_head:
            x, targets, meta = batch
            preds = self.forward_with_meta(x, meta)
        else:
            x, targets = batch
            bz = x.shape[0]
            preds = self(x)
            preds = preds.view(self.model.mc, bz, -1)
            
        ell = self.loss_fn(preds, targets.repeat(self.model.mc, 1, 1).float())
        kl = self.model.get_kl()
        loss = kl + ell #kl - ell

        results = {
            "test_loss": loss,
            "preds": preds.mean(dim=0).cpu(),
            "targets": targets.cpu(),
        }

        return results

###########################################################################################
###########################################################################################
###########################################################################################


class CPC_module(ECGLightningModel):
    def __init__(
        self,
        config,
        loss_fn=F.binary_cross_entropy_with_logits,  # nn.CrossEntropyLoss()
        opt=AdamW,
        cal_acc=False,
        rate=1.0,
        sigmoid_eval=True,
        save_preds=False,
        use_meta_information_in_head=False,
        **kwargs
    ):
        input_channels = config["input_channels"]
        strides = list(config["strides"])
        kss = list(config["kss"])
        features = list(config["features"])
        bn_encoder = config["bn_encoder"]
        n_hidden = config["n_hidden"]
        n_layers = config["n_layers"]
        mlp = config["mlp"]
        lstm = config["lstm"]
        bias_proj = config["bias_proj"]
        num_classes = config["num_classes"]
        concat_pooling = config['concat_pooling']
        ps_head = config["ps_head"]
        lin_ftrs_head = list(config['lin_ftrs_head'])
        bn_head = config['bn_head']
        skip_encoder = config['skip_encoder']
        s4 = False
        model = CPCModel(input_channels, strides=strides, kss=kss, features=features, bn_encoder=bn_encoder, n_hidden=n_hidden, n_layers=n_layers, mlp=mlp, lstm=lstm, bias_proj=bias_proj, num_classes=num_classes, concat_pooling=concat_pooling, ps_head=ps_head, lin_ftrs_head=lin_ftrs_head, bn_head=bn_head, skip_encoder=skip_encoder)

        super(CPC_module, self).__init__(
            model,
            config["batch_size"],
            config["num_samples"],
            lr=config["lr"],
            wd=config["wd"],
            loss_fn=F.binary_cross_entropy_with_logits,  # nn.CrossEntropyLoss()
            opt=AdamW,
            cal_acc=False,
            rate=1.0,
            sigmoid_eval=True,
            save_preds=False,
            use_meta_information_in_head=False,
            #model_size=config['model_size'],
            **kwargs
        )



###########################################################################################
###########################################################################################
###########################################################################################


class XRESNET_module(ECGLightningModel):
    def __init__(
        self,
        config,
        loss_fn=F.binary_cross_entropy_with_logits,  # nn.CrossEntropyLoss()
        opt=AdamW,
        cal_acc=False,
        rate=1.0,
        sigmoid_eval=True,
        save_preds=False,
        use_meta_information_in_head=False,
        **kwargs
    ):
        model = ECGResNet(config["xresnet"], config["num_classes"], big_input=False, use_meta_information_in_head=False)
        super(XRESNET_module, self).__init__(
            model,
            config["batch_size"],
            config["num_samples"],
            lr=config["lr"],
            wd=config["wd"],
            loss_fn=F.binary_cross_entropy_with_logits,  # nn.CrossEntropyLoss()
            opt=AdamW,
            cal_acc=False,
            rate=1.0,
            sigmoid_eval=True,
            save_preds=False,
            use_meta_information_in_head=False,
            #model_size=config['model_size'],
            **kwargs
        )


###########################################################################################
###########################################################################################
###########################################################################################



class S4_module(ECGLightningModel):
    def __init__(
        self,
        config,
        loss_fn=F.binary_cross_entropy_with_logits,  # nn.CrossEntropyLoss()
        opt=AdamW,
        cal_acc=False,
        rate=1.0,
        sigmoid_eval=True,
        save_preds=False,
        use_meta_information_in_head=False,
        backend='GPU',
        **kwargs
    ):
        model = S4Model(config)
        super(S4_module, self).__init__(
            model,
            config["batch_size"],
            config["num_samples"],
            lr=config["lr"],
            wd=config["wd"],
            loss_fn=F.binary_cross_entropy_with_logits,  # nn.CrossEntropyLoss()
            opt=AdamW,
            cal_acc=False,
            rate=1.0,
            sigmoid_eval=True,
            save_preds=False,
            use_meta_information_in_head=False,
            backend=backend,    
            #model_size=config['model_size'],
            **kwargs
        )


###########################################################################################
###########################################################################################
###########################################################################################


class CNN_module(ECGLightningModel):
    def __init__(
        self,
        config,
        loss_fn=F.binary_cross_entropy_with_logits,  # nn.CrossEntropyLoss()
        opt=AdamW,
        cal_acc=False,
        rate=1.0,
        sigmoid_eval=True,
        save_preds=False,
        use_meta_information_in_head=False,
        **kwargs
    ):
        model = CNN(config)
        super(CNN_module, self).__init__(
            model,
            config["batch_size"],
            config["num_samples"],
            lr=config["lr"],
            wd=config["wd"],
            loss_fn=F.binary_cross_entropy_with_logits,  # nn.CrossEntropyLoss()
            opt=AdamW,
            cal_acc=False,
            rate=1.0,
            sigmoid_eval=True,
            save_preds=False,
            use_meta_information_in_head=False,
            #model_size=config['model_size'],
            **kwargs
        )

###########################################################################################
###########################################################################################
###########################################################################################


class LSTM_module(ECGLightningModel):
    def __init__(
        self,
        config,
        loss_fn=F.binary_cross_entropy_with_logits,  # nn.CrossEntropyLoss()
        opt=AdamW,
        cal_acc=False,
        rate=1.0,
        sigmoid_eval=True,
        save_preds=False,
        use_meta_information_in_head=False,
        **kwargs
    ):
        model = LSTM(config)
        super(LSTM_module, self).__init__(
            model,
            config["batch_size"],
            config["num_samples"],
            lr=config["lr"],
            wd=config["wd"],
            loss_fn=F.binary_cross_entropy_with_logits,  # nn.CrossEntropyLoss()
            opt=AdamW,
            cal_acc=False,
            rate=1.0,
            sigmoid_eval=True,
            save_preds=False,
            use_meta_information_in_head=False,
            #model_size=config['model_size'],
            **kwargs
        )

###########################################################################################
###########################################################################################
###########################################################################################


class Transformer_module(ECGLightningModel):
    def __init__(
        self,
        config,
        loss_fn=F.binary_cross_entropy_with_logits,  # nn.CrossEntropyLoss()
        opt=AdamW,
        cal_acc=False,
        rate=1.0,
        sigmoid_eval=True,
        save_preds=False,
        use_meta_information_in_head=False,
        **kwargs
    ):
        model = Transformer(config)
        super(Transformer_module, self).__init__(
            model,
            config["batch_size"],
            config["num_samples"],
            lr=config["lr"],
            wd=config["wd"],
            loss_fn=F.binary_cross_entropy_with_logits,  # nn.CrossEntropyLoss()
            opt=AdamW,
            cal_acc=False,
            rate=1.0,
            sigmoid_eval=True,
            save_preds=False,
            use_meta_information_in_head=False,
            #model_size=config['model_size'],
            **kwargs
        )

###########################################################################################
###########################################################################################
###########################################################################################
        

class RFF_module(RFFECGLightningModel):
    def __init__(
        self,
        config,
        loss_fn=F.binary_cross_entropy_with_logits,  # nn.CrossEntropyLoss()
        opt=AdamW,
        cal_acc=False,
        rate=1.0,
        sigmoid_eval=True,
        save_preds=False,
        use_meta_information_in_head=False,
        **kwargs
    ):
        model = RFF(config)
        super(RFF_module, self).__init__(
            model,
            config["batch_size"],
            config["num_samples"],
            lr=config["lr"],
            wd=config["wd"],
            loss_fn=F.binary_cross_entropy_with_logits,  # nn.CrossEntropyLoss()
            opt=AdamW,
            cal_acc=cal_acc,
            rate=1.0,
            sigmoid_eval=True,
            save_preds=False,
            use_meta_information_in_head=False,
            #model_size=config['model_size'],
            **kwargs
        )


###########################################################################################
###########################################################################################
###########################################################################################


class ConvRFF_module(RFFECGLightningModel):
    def __init__(
        self,
        config,
        loss_fn=F.binary_cross_entropy_with_logits,  # nn.CrossEntropyLoss()
        opt=AdamW,
        cal_acc=False,
        rate=1.0,
        sigmoid_eval=True,
        save_preds=False,
        use_meta_information_in_head=False,
        **kwargs
    ):
        model = ConvRFF(config)
        super(ConvRFF_module, self).__init__(
            model,
            config["batch_size"],
            config["num_samples"],
            lr=config["lr"],
            wd=config["wd"],
            loss_fn=F.binary_cross_entropy_with_logits,  # nn.CrossEntropyLoss()
            opt=AdamW,
            cal_acc=False,
            rate=1.0,
            sigmoid_eval=True,
            save_preds=False,
            use_meta_information_in_head=False,
            #model_size=config['model_size'],
            **kwargs
        )
