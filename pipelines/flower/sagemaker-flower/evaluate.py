from typing import Any, Dict, Optional, Tuple

import os
import subprocess
import torch
import timm
import json
import tarfile

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

ml_root = Path("/opt/ml")

model_artifacts = ml_root / "processing" / "model"
dataset_dir = ml_root / "processing" / "test"

def eval_model(trainer, model, datamodule):
    test_res = trainer.test(model, datamodule)[0]
    
    report_dict = {
        "multiclass_classification_metrics": {
            "accuracy": {
                "value": test_res["test/acc"],
                "standard_deviation": "0",
            },
        },
    }
    
    eval_folder = ml_root / "processing" / "evaluation"
    eval_folder.mkdir(parents=True, exist_ok=True)
    
    out_path = eval_folder / "evaluation.json"
    
    print(f":: Writing to {out_path.absolute()}")
    
    with out_path.open("w") as f:
        f.write(json.dumps(report_dict))
        
        
if __name__ == '__main__':
    
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    
    datamodule = FlowerDataModule(
        train_data_dir=dataset_dir.absolute(),
        test_data_dir=dataset_dir.absolute(),
        num_workers=os.cpu_count()
    )
    datamodule.setup()
    
    model = LitResnet.load_from_checkpoint(checkpoint_path="last.ckpt")
    
    trainer = pl.Trainer(
        accelerator="auto",
    )
    
    print(":: Evaluating Model")
    eval_model(trainer, model, datamodule)