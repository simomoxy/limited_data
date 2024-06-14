"""
    Code adapted from: https://github.com/tmehari/ssm_ecg 
"""

import os

import torch
from pytorch_lightning import LightningDataModule

from clinical_ts.ecg_dataset_wrapper import ECGDataSetWrapper


class ECGDataModule(LightningDataModule):

    name = 'ecg_dataset'
    extra_args = {}

    def __init__(
            self,
            batch_size,
            target_folder,
            label_class="label_all",
            data_dir: str = None,
            num_workers: int = 8,
            data_input_size=250,
            shuffle_train=True,
            nomemmap=False,
            test_folds=[8, 9],
            combination='both',
            filter_label=None,
            val_stride=None,
            normalize=False,
            use_meta_information_in_head=False,
            down_sample_rate=0.0,
            data_seed=0,
            model=None,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.dims = (12, data_input_size)
        # self.val_split = val_split
        self.target_folder = target_folder
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.label_class = label_class
        self.data_input_size = data_input_size
        self.shuffle_train = shuffle_train
        self.nomemmap = nomemmap
        self.test_folds = test_folds
        self.combination = combination
        self.filter_label = filter_label
        self.val_stride = val_stride
        self.normalize=normalize
        self.use_meta_information_in_head = use_meta_information_in_head
        self.down_sample_rate = down_sample_rate
        self.data_seed = data_seed
        self.model = model
        self.set_params()

    def set_params(self):
        dataset, _, _ = get_dataset(self.batch_size, self.num_workers, self.target_folder, label_class=self.label_class,
                                    input_size=self.data_input_size, nomemmap=self.nomemmap, test_folds=self.test_folds, combination=self.combination, filter_label=self.filter_label, val_stride=self.val_stride,
                                    down_sample_rate=self.down_sample_rate, data_seed=self.data_seed, model=self.model)
        self.num_samples = dataset.train_ds_size
        self.lbl_itos = dataset.lbl_itos
        self.num_classes = len(self.lbl_itos)
        self.train_idmap = dataset.train_ds_idmap
        self.idmap = dataset.val_ds_idmap

    def prepare_data(self):
        pass

    def train_dataloader(self):
        _, train_loader, _ = get_dataset(self.batch_size, self.num_workers, self.target_folder, label_class=self.label_class,
                                         input_size=self.data_input_size, shuffle_train=self.shuffle_train, nomemmap=self.nomemmap, test_folds=self.test_folds, combination=self.combination, filter_label=self.filter_label, normalize=self.normalize, use_meta_information_in_head=self.use_meta_information_in_head, down_sample_rate=self.down_sample_rate,
                                         data_seed=self.data_seed)
        return train_loader

    def val_dataloader(self):
        _, _, valid_loader = get_dataset(self.batch_size, self.num_workers, self.target_folder, label_class=self.label_class,
                                         input_size=self.data_input_size, nomemmap=self.nomemmap, test_folds=self.test_folds, combination=self.combination, filter_label=self.filter_label, val_stride=self.val_stride, normalize=self.normalize, use_meta_information_in_head=self.use_meta_information_in_head,
                                         data_seed=self.data_seed)
        return valid_loader

    def test_dataloader(self):
        dataset, _, valid_loader = get_dataset(self.batch_size, self.num_workers, self.target_folder, test=True,
                                         label_class=self.label_class, input_size=self.data_input_size, nomemmap=self.nomemmap, test_folds=self.test_folds, combination=self.combination, filter_label=self.filter_label, normalize=self.normalize, use_meta_information_in_head=self.use_meta_information_in_head,
                                         data_seed=self.data_seed)
        self.test_idmap = dataset.val_ds_idmap
        return valid_loader

    def default_transforms(self):
        pass


def get_dataset(batch_size, num_workers, target_folder, test=False, label_class="label_all",
                input_size=250, shuffle_train=False, nomemmap=False, test_folds=[8, 9],
                combination='both', filter_label=None, val_stride=None, normalize=False, use_meta_information_in_head=False,
                down_sample_rate=0.0, data_seed=0, model=None):

    dataset = ECGDataSetWrapper(
        batch_size, num_workers, target_folder, test=test, label=label_class,
        input_size=input_size, shuffle_train=shuffle_train, nomemmap=nomemmap,
        normalize=normalize, test_folds=test_folds, combination=combination,
        filter_label=filter_label, val_stride=val_stride, use_meta_information_in_head=use_meta_information_in_head,
        down_sample_rate=down_sample_rate, data_seed=data_seed, model=model)

    train_loader, valid_loader = dataset.get_data_loaders()
    return dataset, train_loader, valid_loader