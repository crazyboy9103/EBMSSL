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
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_gpus', type=int, default=1)
    # Model
    parser.add_argument('--backbone', type=str, default="resnet18")
    parser.add_argument('--from_ckpt', type=str, default="")
    parser.add_argument('--pretrained', type=str2bool, default=False)
    
    # SSL Task
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--alpha_0', type=float, default=0.1)
    parser.add_argument('--num_steps', type=int, default=2)
    parser.add_argument('--transform', type=str, default="gridded_mask", choices=["gridded_mask", "random_mask", "super_resolution", "noise", "all", "random"])
    parser.add_argument('--mask_size', type=int, default=16)
    parser.add_argument('--mask_ratio', type=float, default=0.5)
    parser.add_argument('--scale_factor', type=int, default=2)
    
    # Linear Probe & Fine-tune Task
    parser.add_argument('--eval', type=str, default="linear", choices=["linear", "finetune"])
    parser.add_argument('--num_classes', type=int, default=10)
    
    
    # Optimizer
    parser.add_argument('--optimizer', type=str, default="Adam", choices=["Adam", "SGD", "AdamW"])
    parser.add_argument('--ssl_lr', type=float, default=0.0001) # SSL learning rate
    parser.add_argument('--sl_lr', type=float, default=0.001) # Linear probe/Finetune learning rate
    
    
    # Misc
    parser.add_argument('--seed', type=int, default=2023)
    
    # Trainer
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--precision', type=int, default=32)
    parser.add_argument('--accelerator', type=str, default="gpu")
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--gpus', type=int, default=1)
    
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    
    logger = None  
    if args.wandb:
        project_name = "EBMSSL"
        experiment_name = f"{args.backbone}-{args.dataset}"
        
        logger = WandbLogger(
            project=project_name,
            name=experiment_name,
            log_model=False,
            save_dir="."
        )
    
    checkpoint_path = f"./checkpoints/{args.backbone}/{args.dataset}"
    if os.path.exists(checkpoint_path):
        shutil.rmtree(checkpoint_path)
    os.makedirs(checkpoint_path, exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(dirpath=checkpoint_path, save_top_k=2, monitor="train-rec-loss", mode="min"),
        LearningRateMonitor(logging_interval='step')
    ]
    
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=args.max_epochs,
        precision=args.precision,
        accelerator=args.accelerator,
        devices=args.gpus,
        deterministic=True,
    )
    # ------------
    # data
    # ------------
    data = DatasetWrapper(args.dataset, os.path.join(args.data_root, args.dataset))
    train_dataset = data("train")
    finetune_dataset = data("finetune")
    test_dataset = data("test")

    data_loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True, 
        generator=torch.Generator().manual_seed(args.seed)
    )
        
    train_loader = DataLoader(train_dataset, **data_loader_kwargs)
    val_loader = DataLoader(test_dataset, **data_loader_kwargs)
    finetune_loader = DataLoader(finetune_dataset, **data_loader_kwargs)
    

    # ------------
    # model
    # ------------
    model = ModelWrapper(
        args.backbone,
        args.alpha_0,
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
        
        
    ssl_task = EBMSSLTask(
        model = model,
        transform = transform,
        num_steps = 2,
        optimizer = getattr(optim, args.optimizer),
        base_lr = args.ssl_lr
    )
    
    trainer.fit(ssl_task, train_loader, val_loader)

    if args.eval == "linear":
        linear_probe_task = EBMLinearProbeTask(
            model = model,
            optimizer = getattr(optim, args.optimizer),
            base_lr = args.sl_lr
        )
        
        trainer.fit(linear_probe_task, finetune_loader, val_loader)
    
    elif args.eval == "finetune":
        finetune_task = EBMFineTuneTask(
            model = model,
            optimizer = getattr(optim, args.optimizer),
            base_lr = args.sl_lr
        )
        
        trainer.fit(finetune_task, finetune_loader, val_loader)
        
    
    # ------------
    # testing
    # ------------
    
if __name__ == '__main__':
    cli_main()