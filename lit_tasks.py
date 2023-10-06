from typing import Callable, Type

import wandb
from torch.nn import functional as F
from torch import optim
from torch import nn
from torch import Tensor
from torch import autograd
import pytorch_lightning as pl
from torchvision.transforms.functional import to_pil_image
import torchmetrics
import umap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class EBMBaseTask(pl.LightningModule):
    def __init__(
        self, 
        model: nn.Module,
        # SSL Task
        transform: Callable[[Tensor], Tensor] = lambda x: x,
        num_steps: int = 2, 
        # Optimizer
        optimizer: Type = optim.Adam,
        base_lr: float = 0.001, 
    ):
        super().__init__()
        self.model = model
        self.transform = transform
        
        self.num_steps = num_steps
        
        self.optimizer = optimizer
        self.lr = base_lr
        
        # Initialize metrics
        num_classes = model.linear_head.out_features
        
        metric_kwargs = dict(
            task = "multiclass",
            num_classes = num_classes,
        )
        self.accuracy = torchmetrics.Accuracy(**metric_kwargs)
        self.precision = torchmetrics.Precision(average='weighted', **metric_kwargs)
        self.recall = torchmetrics.Recall(average='weighted', **metric_kwargs)
        self.f1 = torchmetrics.F1Score(average='weighted', **metric_kwargs)
        self.confmat = torchmetrics.ConfusionMatrix(**metric_kwargs)
    
    def reconstruct(self, batch, beta=1.0):
        images, _ = batch
        corrupted_images = self.transform(images)
        loss = 0
        for _ in range(self.num_steps):
            corrupted_images = corrupted_images.detach()
            corrupted_images.requires_grad_(True)
            # stop gradients between inner-loop steps.
            energy_score = self.model(corrupted_images, return_energy=True)

            # energy score with shape [n, 1]
            im_grad = autograd.grad(energy_score.sum(), corrupted_images)[0]
            
            # compute the gradient of input pixels along the direction
            # of energy maximization
            corrupted_images = corrupted_images - self.model.alpha * im_grad
            # gradient descent along the direction of energy minimization
            loss += F.smooth_l1_loss(corrupted_images, images, beta=beta)
            
        return loss, corrupted_images
    
    def features(self, batch):
        images, labels = batch
        features = self.model(images)
        return features
    
    def logits(self, batch):
        images, labels = batch
        logits = self.model(images, return_logits=True)
        loss = F.cross_entropy(logits, labels)
        return loss, logits
    
    def accumulate_metrics(self, y, y_hat):
        self.accuracy(y_hat, y)
        self.precision(y_hat, y)
        self.recall(y_hat, y)
        self.f1(y_hat, y)
        self.confmat(y_hat, y)
    
    def reset_metrics(self):
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        self.confmat.reset()    
        
    def metrics(self):
        return {
            "accuracy": self.accuracy.compute(),
            "precision": self.precision.compute(),
            "recall": self.recall.compute(),
            "f1": self.f1.compute(),
            "confmat": self.confmat.compute()
        }
        
        
        
    
class EBMSSLTask(EBMBaseTask):
    def __init__(self, model: nn.Module, transform: Callable[[Tensor], Tensor] = lambda x: x, num_steps: int = 2, optimizer: Type = optim.Adam, base_lr: float = 0.001):
        super().__init__(model, transform, num_steps, optimizer, base_lr)
        
    def validation_step(self, batch, batch_idx):
        # rec_loss, _ = self.reconstruct(batch)
        # ce_loss, logits = self.logits(batch)
        features = self.features(batch)
        
        y = batch[1]
        
        # Convert PyTorch tensor to NumPy array
        features_np = features.detach().cpu().numpy()
        labels_np = y.detach().cpu().numpy()

        # UMAP dimensionality reduction
        umap_model = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='correlation')
        umap_result = umap_model.fit_transform(features_np)

        # t-SNE dimensionality reduction
        tsne_model = TSNE(n_components=2, verbose=1, perplexity=min(5, len(y)), n_iter=300)
        tsne_result = tsne_model.fit_transform(features_np)

        # Create a Matplotlib figure and axis
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))  # Increased size for colorbar

        # UMAP plot
        umap_scatter = ax[0].scatter(umap_result[:, 0], umap_result[:, 1], c=labels_np, cmap='viridis')
        ax[0].set_title('UMAP Projection')
        fig.colorbar(umap_scatter, ax=ax[0])  # Add colorbar to the first subplot

        # t-SNE plot
        tsne_scatter = ax[1].scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels_np, cmap='viridis')
        ax[1].set_title('t-SNE Projection')
        fig.colorbar(tsne_scatter, ax=ax[1])  # Add colorbar to the second subplot

        # Log the Matplotlib figure to wandb
        self.logger.experiment.log({"Feature Visualization": wandb.Image(fig)})

        # Close the figure to free up resources
        plt.close(fig)

            
    
    def training_step(self, batch, batch_idx):
        rec_loss, rec_images = self.reconstruct(batch)
        
        if batch_idx == 0:
            for idx, image in enumerate(rec_images):
                pil_image = to_pil_image(image.detach().cpu())
                orig_image = to_pil_image(batch[0][idx].detach().cpu())
                self.logger.experiment.log({
                    "reconstruction": wandb.Image(pil_image), 
                    "original": wandb.Image(orig_image)
                })
        
        self.log('train-rec-loss', rec_loss.item())
        return rec_loss

    def configure_optimizers(self):
        optimizer = self.optimizer(self.model.parameters(), lr=self.lr)
        return optimizer
    
class EBMLinearProbeTask(EBMBaseTask):
    def __init__(self, model: nn.Module, transform: Callable[[Tensor], Tensor] = lambda x: x, num_steps: int = 2, optimizer: Type = optim.Adam, base_lr: float = 0.001):
        super().__init__(model, transform, num_steps, optimizer, base_lr)
        
    def validation_step(self, batch, batch_idx):
        ce_loss, logits = self.logits(batch)
        self.log('valid-linear-loss', ce_loss.item())

        y_hat = logits.argmax(dim=1)
        y = batch[1]
        self.accumulate_metrics(y, y_hat)
        return ce_loss 
    
    def training_step(self, batch, batch_idx):
        ce_loss, _ = self.logits(batch)
        self.log('train-linear-loss', ce_loss.item())
        return ce_loss
    
    def configure_optimizers(self):
        optimizer = self.optimizer(self.model.linear_head.parameters(), lr=self.lr)
        return optimizer
    
    def on_validation_epoch_end(self):
        metrics = self.metrics()
        confmat = metrics.pop("confmat")
        self.logger.experiment.log({'confusion_matrix': confmat})

        for name, val in metrics.items():
            self.log(f"valid-{name}", val)

        self.reset_metrics()
        
class EBMFineTuneTask(EBMBaseTask):
    def validation_step(self, batch, batch_idx):
        ce_loss, logits = self.logits(batch)
        self.log('valid-finetune-loss', ce_loss.item())
        
        y_hat = logits.argmax(dim=1)
        y = batch[1]
        self.accumulate_metrics(y, y_hat)
        return ce_loss 
    
    def training_step(self, batch, batch_idx):
        ce_loss, _ = self.logits(batch)
        self.log('train-finetune-loss', ce_loss.item())
        return ce_loss 
    
    def configure_optimizers(self):
        optimizer = self.optimizer(tuple(self.model.backbone.parameters()) + tuple(self.model.linear_head.parameters()), lr=self.lr)
        return optimizer
        
        #  # following milestones, warmup_iters are arbitrarily chosen
        # first, second = self.steps_per_epoch * int(self.total_epochs * 4/6), self.steps_per_epoch * int(self.total_epochs * 5/6)
        # warmup_iters = self.steps_per_epoch * int(self.total_epochs * 1/6)
        # scheduler = LinearWarmUpMultiStepDecay(optimizer, milestones=[first, second], gamma=1/3, warmup_iters=warmup_iters)
        # scheduler_config = {
        #     "scheduler": scheduler,
        #     "interval": "step",
        # }
        # return [optimizer], [scheduler_config]

    def on_validation_epoch_end(self):
        metrics = self.metrics()
        confmat = metrics.pop("confmat")
        self.logger.experiment.log({'confusion_matrix': confmat})

        for name, val in metrics.items():
            self.log(f"valid-{name}", val)

        self.reset_metrics()
        