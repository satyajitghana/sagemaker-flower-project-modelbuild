from typing import Any, Dict, Optional, Tuple

import os
import subprocess
import torch
import timm
import json

import pytorch_lightning as pl
import torchvision.transforms as T
import torch.nn.functional as F

from pathlib import Path
from torchvision.datasets import ImageFolder
from pytorch_lightning.plugins.environments import LightningEnvironment
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import accuracy
from pytorch_lightning import loggers as pl_loggers
from datetime import datetime

from model import LitResnet
from dataset import FlowerDataModule

sm_output_dir = Path(os.environ.get("SM_OUTPUT_DIR"))
sm_model_dir = Path(os.environ.get("SM_MODEL_DIR"))
num_cpus = int(os.environ.get("SM_NUM_CPUS"))

train_channel = os.environ.get("SM_CHANNEL_TRAIN")
test_channel = os.environ.get("SM_CHANNEL_TEST")

ml_root = Path("/opt/ml")


def get_training_env():
    sm_training_env = os.environ.get("SM_TRAINING_ENV")
    sm_training_env = json.loads(sm_training_env)
    
    return sm_training_env


def train(model, datamodule, sm_training_env):
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=ml_root / "output" / "tensorboard" / sm_training_env["job_name"])
    
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="auto",
        logger=[tb_logger]
    )
    
    trainer.fit(model, datamodule)
    
    return trainer

def save_scripted_model(model, output_dir):
    script = model.to_torchscript()

    # save for use in production environment
    torch.jit.save(script, output_dir / "model.scripted.pt")


def save_last_ckpt(trainer, output_dir):
    trainer.save_checkpoint(output_dir / "last.ckpt")


if __name__ == '__main__':
    
    img_dset = ImageFolder(train_channel)
    
    print(":: Classnames: ", img_dset.classes)
    
    datamodule = FlowerDataModule(train_data_dir=train_channel, test_data_dir=test_channel, num_workers=num_cpus)
    datamodule.setup()
    
    model = LitResnet(num_classes=datamodule.num_classes)
    
    sm_training_env = get_training_env()
    
    print(":: Training ...")
    trainer = train(model, datamodule, sm_training_env)
    
    print(":: Saving Model Ckpt")
    save_last_ckpt(trainer, sm_model_dir)
    
    print(":: Saving Scripted Model")
    save_scripted_model(model, sm_model_dir)

