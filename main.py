import argparse
import os 
import shutil

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomApply
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from models import ModelWrapper
from datasets import DatasetWrapper
from datasets.transforms import (
    GriddedRandomMask,
    RandomMask,
    SuperResolution,
    Noise
)
from lit_tasks import EBMSSLTask, EBMLinearProbeTask, EBMFineTuneTask

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def cli_main():
    # ------------
    # args
    # ------------
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--dataset', type=str, default="imagenet")
    parser.add_argument('--data_root', type=str, default="/mnt/d/datasets")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    
    # Model
    parser.add_argument('--backbone', type=str, default="resnet50")
    parser.add_argument('--from_ckpt', type=str, default="")
    parser.add_argument('--pretrained', type=str2bool, default=False)
    
    # SSL Task
    parser.add_argument('--base_alpha', type=float, default=0.1)
    parser.add_argument('--num_steps', type=int, default=2)
    parser.add_argument('--transform', type=str, default="gridded_mask", choices=["gridded_mask", "random_mask", "super_resolution", "noise", "all", "random"])
    parser.add_argument('--mask_size', type=int, default=16)
    parser.add_argument('--mask_ratio', type=float, default=0.5)
    parser.add_argument('--scale_factor', type=int, default=2)
    parser.add_argument('--ssl_epochs', type=int, default=100)
    parser.add_argument('--ssl_lr', type=float, default=0.0001) # SSL learning rate
    parser.add_argument('--ssl_acc_batches', type=int, default=8) # Accumulate batches for SSL
    
    # Linear Probe & Fine-tune Task
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--sl_epochs', type=int, default=100)
    parser.add_argument('--sl_lr', type=float, default=0.001) # Linear probe/Finetune learning rate
    parser.add_argument('--sl_acc_batches', type=int, default=1) # Accumulate batches for linear probe/finetune
    
    # Optimizer
    parser.add_argument('--optimizer', type=str, default="AdamW", choices=["Adam", "SGD", "AdamW"])
    
    # Trainer
    parser.add_argument('--precision', type=str, default='16-mixed')
    parser.add_argument('--accelerator', type=str, default="gpu")
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--task', type=str, default="ssl", choices=["ssl", "ssl+linear", "ssl+finetune", "linear", "finetune"])

    # Misc
    parser.add_argument('--seed', type=int, default=2023)
    
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    
    # ------------
    # data
    # ------------
    data = DatasetWrapper(args.dataset, os.path.join(args.data_root, args.dataset), args.seed)
    train_dataset = data("train")
    finetune_dataset = data("finetune")
    test_dataset = data("test")

    data_loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True, 
        generator=torch.Generator().manual_seed(args.seed)
    )
        
    train_loader = DataLoader(train_dataset, shuffle=True, **data_loader_kwargs)
    val_loader = DataLoader(test_dataset, shuffle=False, **data_loader_kwargs)
    finetune_loader = DataLoader(finetune_dataset, shuffle=False, **data_loader_kwargs)
    
    # ------------
    # model
    # ------------
    model = ModelWrapper(
        args.backbone,
        args.base_alpha,
        args.from_ckpt,
        args.pretrained,
        args.num_classes         
    )

    # ------------
    # training
    # ------------
    gridded_mask = GriddedRandomMask(args.mask_size, args.mask_ratio)
    random_mask = RandomMask(args.mask_size, args.mask_ratio)
    super_resolution = SuperResolution(args.scale_factor)
    noise = Noise()
    
    match args.transform:
        case "gridded_mask":
            transform = gridded_mask
        
        case "random_mask":
            transform = random_mask
        
        case "super_resolution":
            transform = super_resolution
        
        case "noise":
            transform = noise
        
        case "all":
            transform = Compose([
                gridded_mask,
                random_mask,
                super_resolution,
                noise
            ])
        
        case "random":
            transform = Compose([
                RandomApply(gridded_mask, p=0.1),
                RandomApply(random_mask, p=0.1),
                RandomApply(super_resolution, p=0.1),
                RandomApply(noise, p=0.1)
            ])
            
        case _:
            raise NotImplementedError
        
    checkpoint_path = f"./checkpoints/{args.backbone}/{args.dataset}"
    if os.path.exists(checkpoint_path):
        shutil.rmtree(checkpoint_path)
    os.makedirs(checkpoint_path, exist_ok=True)
    
    trainer_kwargs = dict(
        logger=WandbLogger(
            project="EBMSSL",
            name=f"{args.backbone}-{args.dataset}",
            log_model=False,
            save_dir="."
        ) if args.wandb else None,
        callbacks=[
            ModelCheckpoint(dirpath=checkpoint_path, save_top_k=2, monitor="train-rec-loss", mode="min"),
            LearningRateMonitor(logging_interval='step')
        ],
        precision=args.precision,
        accelerator=args.accelerator,
        devices=args.devices,
        deterministic=True,
    )
    
    if "ssl" in args.task: 
        ssl_task = EBMSSLTask(
            model = model,
            transform = transform,
            num_steps = 2,
            optimizer = getattr(optim, args.optimizer),
            base_lr = args.ssl_lr
        )
        
        trainer = pl.Trainer(
            max_epochs=args.ssl_epochs,
            accumulate_grad_batches=args.ssl_acc_batches,
            **trainer_kwargs
        )
        
        trainer.fit(ssl_task, train_loader, val_loader)

    if "linear" in args.task:
        linear_probe_task = EBMLinearProbeTask(
            model = model,
            optimizer = getattr(optim, args.optimizer),
            base_lr = args.sl_lr
        )
        
        trainer = pl.Trainer(
            max_epochs=args.sl_epochs,
            accumulate_grad_batches=args.sl_acc_batches,
            **trainer_kwargs
        )
        
        trainer.fit(linear_probe_task, finetune_loader, val_loader)
    
    elif "finetune" in args.task:
        finetune_task = EBMFineTuneTask(
            model = model,
            optimizer = getattr(optim, args.optimizer),
            base_lr = args.sl_lr
        )
        
        trainer = pl.Trainer(
            max_epochs=args.sl_epochs,
            accumulate_grad_batches=args.sl_acc_batches,
            **trainer_kwargs
        )
        
        trainer.fit(finetune_task, finetune_loader, val_loader)
        
if __name__ == '__main__':
    cli_main()