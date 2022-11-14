"""DataModule based RLDataset, to batch the experience get in the training-game"""
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from .exp_replay import ExperienceReplay, RLDataset


class RLLitDataModule(pl.LightningDataModule):
    """Lightning DataModule with train/val dataloaders

    Parameters
    ----------
    memory_size : int, optional
        Memory size for ExperienceReplay, by default 3000
    buffer_size : int, optional
        Buffer size (total batch on each epoch), by default 2000
    batch_size : int, optional
        Batch size to use, by default 64

    Attributes
    ----------
    train_dataset : torch.utils.data.dataset
        Train dataset
    val_dataset : torch.utils.data.dataset
        Validation dataset
    """

    def __init__(self, memory_size: int = 3000, buffer_size: int = 2000,
                 batch_size: int = 64, **kwargs):
        """
        Update docstring in google format to make Lightning happy

        Args:
            memory_size: Memory size for ExperienceReplay, by default 3000
            buffer_size: Buffer size (total batch on each epoch), by default 2000
            batch_size: Batch size to use, by default 64

        """
        super().__init__()
        self.save_hyperparameters(ignore=[x for x in kwargs])

    def setup(self, stage):
        self.train_dataset = RLDataset(
            buffer=ExperienceReplay(max_memory=self.hparams.memory_size),
            buffer_size=self.hparams.buffer_size)
        self.val_dataset = RLDataset(
            buffer=ExperienceReplay(max_memory=self.hparams.memory_size),
            buffer_size=self.hparams.buffer_size)

        # Populate each dataloader
        self.trainer.populate(self.train_dataset.buffer.memory.maxlen)
        return super().setup(stage)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size)
