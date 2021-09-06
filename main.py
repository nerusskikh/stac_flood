# GPU limitation stuff
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = 'cuda'

#deep learning
import segmentation_models_pytorch as smp
import torch


from torch import nn
from pytorch_toolbelt.inference import tta
from pytorch_toolbelt.losses import BinaryLovaszLoss, BinaryFocalLoss, JointLoss
from catalyst.utils.swa import get_averaged_weights_by_path_mask
from catalyst.dl import (
    SupervisedRunner, 
    CriterionCallback, 
    EarlyStoppingCallback, 
    SchedulerCallback,
    IOUCallback,
    CheckpointCallback,
)


#data management
import cv2
import rasterio
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from data import STACDataset, get_aug


#maintenance
import glob
import random
import os
import json

from utils import seed_everything, colorcode_STAC_errors
from collections import OrderedDict
from datetime import datetime
from tqdm import tqdm

#configuration
from config import *


def model_setup(encoder_name, decoder_name):
    encoder_weights = 'noisy-student' if encoder_name.startswith('timm-efficentnet') else 'imagenet'
    model = getattr(smp, decoder_name)(
                                       encoder_name,
                                       in_channels=2,
                                       activation=None,
                                       encoder_weights=encoder_weights).cuda()
    optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=LR),
])
    return model, optimizer


class STACRunner(SupervisedRunner):
    def handle_batch(self, batch):
        x = batch[self._input_key]
        target_ = batch[self._target_key]
        mask_ = 1.*(target_!=255)

        out_ = self.model(x)
        out_ = out_*mask_
        target_ = target_*mask_
        self.batch = {self._input_key: x, self._output_key: out_, self._target_key: target_}
        self.input = {self._input_key: x, self._target_key: target_}
        self.output = {self._output_key: out_}
    
    def predict_batch(self, batch):
        x = batch[self._input_key]
        target_ = batch[self._target_key]
        mask_ = 1.*(target_!=255)

        with torch.no_grad():
            out_ = tta.d4_image2mask(self.model, input_.cuda())
            out_ = out_*mask_
        self.batch = {self._input_key: x, self._output_key: out_, self._target_key: target_}
        return self.batch


class SymmetricLovaszLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_loss = BinaryLovaszLoss()
    def forward(self, out, target):
        return 0.5*(self.base_loss(out, target)+self.base_loss(-out, 1-target))



def IandU(pred, targ, threshold=0.):
    pred_bin = pred>threshold
    targ_bin = targ>threshold
    
    intersection = torch.sum((pred_bin*targ_bin)>0).item()
    union = torch.sum((pred_bin+targ_bin)>0).item()
    
    return intersection, union
    

def validateSTACDataloader(model, dataloader, threshold=0.):
    val_losses = []
    val_intersections = []
    val_unions = []
    model.eval()
    for input_, target_  in tqdm(dataloader, leave=False):
        input_ = input_.cuda()
        target_ = target_.cuda()
        mask_ = 1.*(target_!=255)

        with torch.no_grad():
            out_ = tta.d4_image2mask(model, input_)
            out_ = out_*mask_
        target_ = target_*mask_
        loss = criterion(out_, target_)
        val_losses.append(loss.item())
        i_, u_ = IandU(out_, target_, threshold=threshold)
        val_intersections.append(i_)
        val_unions.append(u_)
        
    aggregated_metric = np.sum(val_intersections)/(np.sum(val_unions)+1e-8)
        
    return np.mean(val_losses), aggregated_metric

if __name__ == "__main__":
    seed_everything(SEED)
    criterion = JointLoss(SymmetricLovaszLoss(), BinaryFocalLoss(), second_weight=FOCAL_WEIGHT)
    
    ### Training
    current_time = datetime.now().strftime("%b%d_%H_%M")


    for fold_ in range(nfolds):

        prefix = f"{run_prefix}/{current_time}/{fold_}"

        log_dir = os.path.join("runs", prefix)
        os.makedirs(log_dir, exist_ok=False)

        dataset_train = STACDataset(fold=fold_, train=True, tfms=get_aug())
        dataloader_train = DataLoader(
                                      dataset_train,
                                      batch_size=bs,
                                      shuffle=True,
                                      num_workers=NUM_WORKERS
                                      )

        dataset_val = STACDataset(fold=fold_, train=False)
        dataloader_val = DataLoader(
                                    dataset_val,
                                    batch_size=bs,
                                    shuffle=False,
                                    num_workers=NUM_WORKERS
                                    )

        loaders = OrderedDict()
        loaders["train"] = dataloader_train
        loaders["valid"] = dataloader_val

        model, optimizer = model_setup(encoder_name, decoder_name)

        runner = STACRunner(
            input_key="features", output_key="scores", target_key="targets", loss_key="loss"
                            )

        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loaders=loaders,
            num_epochs=N_EPOCHS,
            callbacks=[
                IOUCallback(input_key="scores", target_key="targets", threshold=0),
                CheckpointCallback(
                                   logdir = os.path.join(log_dir, 'checkpoints'),
                                   loader_key='valid',
                                   metric_key='loss',
                                   minimize=True,
                                   use_runner_logdir=False,
                                   save_n_best=SAVE_N_BEST),
            ],
            logdir=log_dir,
            valid_loader="valid",
            valid_metric="iou",
            minimize_valid_metric=False,
            verbose=False,
            amp=True

    )


    # SWA and validation
    fold_metrics = []
    for fold_ in range(nfolds):
        prefix = f"{run_prefix}/{current_time}"

        log_dir = os.path.join("runs", prefix, str(fold_))
        dataset_val = STACDataset(fold=fold_, train=False)
        dataloader_val = DataLoader(dataset_val,
                                    batch_size=val_bs,
                                    shuffle=False,
                                    num_workers=NUM_WORKERS
                                    )

        model, optimizer = model_setup(encoder_name, decoder_name)

        swa_weights = get_averaged_weights_by_path_mask(
                                                        logdir=log_dir,
                                                        path_mask='train*.pth')
        torch.save(swa_weights, os.path.join(
                                            "runs",
                                            prefix,
                                            run_prefix+f'_fold_{fold_}_swa.pth'))
        model.load_state_dict(swa_weights)

        fold_val_loss, fold_val_metric = validateSTACDataloader(
                                                                model.cuda(),
                                                                dataloader_val, 
                                                                threshold=0.0
                                                                )
        fold_metrics.append(fold_val_metric)

    crossvalidation_results = {}
    for fold_, result_ in enumerate(fold_metrics):
        crossvalidation_results[f'fold_{fold_}'] = result_
    crossvalidation_results['mean'] = np.mean(fold_metrics)
    crossvalidation_results['std'] = np.std(fold_metrics)

    with open(os.path.join("runs", prefix, run_prefix+'_cv_results.json'), 'w') as outfile:
        json.dump(crossvalidation_results, outfile)
    