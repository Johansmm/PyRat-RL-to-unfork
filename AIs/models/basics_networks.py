from typing import List, Tuple, Union
import numpy as np

import torch
import torch.nn as nn

import pytorch_lightning as pl


class BaseLit(pl.LightningModule):
    """The abstact class adapted to Pytorch lightning

    Parameters
    ----------
    x_example : Union[torch.Tensor, np.ndarray]
        Input example, to calculate the input shape.
    out_shape : Union[int, List[int]]
        Output shape.
    """

    hparams: dict

    def __init__(self, x_sample: Union[np.ndarray, torch.Tensor],
                 out_shape: Union[int, List[int]] = None, **kwargs):
        super().__init__()
        kwargs["actions"] = kwargs.get("actions", np.prod(out_shape))
        self.save_hyperparameters(ignore=["ignore", "x_sample"] + kwargs.get("ignore", []))
        self.model = self.build_model(x_sample=x_sample)
        self.criterion = self.build_criterion()

    def build_model(self, x_sample: torch.Tensor):
        """Build model given an input sample

        Parameters
        ----------
        x_sample : torch.Tensor
            sample to extract input shape

        Returns
        -------
        nn.Module
            The model
        """
        raise NotImplementedError("This method must be overwritten by the child")

    def build_criterion(self):
        """Build criterion

        Returns
        -------
        nn.Module
            Criterion
        """
        raise NotImplementedError("This method must be overwritten by the child")

    def forward(self, x: torch.Tensor):
        """Forward step

        Parameters
        ----------
        x : torch.Tensor
            The input tensor

        Returns
        -------
        torch.Tensor
            Output of the network
        """
        raise NotImplementedError("This method must be overwritten by the child")

    def get_epsilon(self):
        """Return a value between [0,1] with the linear proporcional step

        Returns
        -------
        float
            Relative step
        """
        return 1.0 - self.current_epoch / self.trainer.max_epochs

    @torch.no_grad()
    def inference(self, x: torch.Tensor, epsilon: float = 0.0):
        """Inference step

        Parameters
        ----------
        x : torch.Tensor
            The input tensor

        Returns
        -------
        torch.Tensor
            Output of the network
        """
        if np.random.random() < epsilon:
            action = np.random.choice(self.hparams.actions, size=(
                x.shape[0], self.hparams.out_shape))
        else:
            action = self.forward(x).detach().cpu().numpy()
        return action.argmax(axis=-1)

    def configure_optimizers(self):
        """Configure optimizer over model's weights

        Returns
        -------
        torch.nn
            Optimizer
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch: Union[List[torch.Tensor], torch.Tensor], batch_idx: int):
        """Training step

        Parameters
        ----------
        train_batch : Union[List[torch.Tensor], torch.Tensor]
            Train batch
        batch_idx : int
            Index batch

        Returns
        -------
        Union[dict[torch.Tensor], List[torch.Tensor], torch.Tensor]
            Loss and/or metrics
        """
        raise NotImplementedError("This method must be overwritten by the child")

    def validation_step(self, val_batch: Union[List[torch.Tensor], torch.Tensor], batch_idx: int):
        """Validation step

        Parameters
        ----------
        train_batch : Union[List[torch.Tensor], torch.Tensor]
            Train batch
        batch_idx : int
            Index batch

        Returns
        -------
        Union[dict[torch.Tensor], List[torch.Tensor], torch.Tensor]
            Loss and/or metrics
        """
        raise NotImplementedError("This method must be overwritten by the child")


class PerceptronLit(BaseLit):
    """The simplest model adapted to Pytorch lightning

    Parameters
    ----------
    x_example : Tuple[torch.Tensor, np.ndarray]
        Input example, to calculate the input shape
    out_shape : int, optional
        Number of actions, by default 4
    dropout : float, optional
        Dropout, by default 0.01.
    """

    def __init__(self, x_sample: Tuple[torch.Tensor, np.ndarray],
                 out_shape: int = 4, dropout: float = 0.01, **kwargs):
        super().__init__(x_sample=x_sample, out_shape=out_shape, dropout=dropout)

    def build_model(self, x_sample: torch.Tensor):
        in_features = x_sample.reshape(-1).shape[0]
        return nn.Sequential(nn.Dropout(p=self.hparams.dropout),
                             nn.Linear(in_features, self.hparams.out_shape))

    def build_criterion(self):
        return nn.MSELoss()

    def forward(self, x: torch.Tensor):
        x = x.view(x.shape[0], -1)
        action = self.model(x)
        return action

    def _common_step(self, batch: Union[List[torch.Tensor], torch.Tensor], batch_idx: int):
        states, actions, rewards, next_states = batch
        y_pred = self.forward(states).gather(1, actions.long().unsqueeze(-1)).squeeze(-1)
        return self.criterion(y_pred, rewards)

    def training_step(self, train_batch: Union[List[torch.Tensor], torch.Tensor], batch_idx: int):
        loss = self._common_step(batch=train_batch, batch_idx=batch_idx)
        self.log("train_loss", loss)
        self.log("epsilon", self.get_epsilon(), prog_bar=True)

        # At the end, make a simplest step into the game
        self.trainer.play_step(epsilon=self.get_epsilon())
        return loss

    def validation_step(self, val_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        loss = self._common_step(batch=val_batch, batch_idx=batch_idx)
        self.log("val_loss", loss)
        return loss


class MLPLit(PerceptronLit):
    """MLP network transform to Pytorch Lightning

    Parameters
    ----------
    x_example : Tuple[torch.Tensor, np.ndarray]
        Input example, to calculate the input shape
    hidden_size : int
        Size of hidden layers
    out_shape : int, optional
        Number of actions, by default 4
    dropout : float, optional
        Dropout, by default 0.01.
    """

    def __init__(self, *args, hidden_size: int, **kwargs):
        super().__init__(*args, hidden_size=hidden_size, **kwargs)

    def build_model(self, x_sample):
        in_features = x_sample.reshape(-1).shape[0]
        return nn.Sequential(
            nn.Linear(in_features, self.hparams.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=self.hparams.dropout),
            nn.Linear(self.hparams.hidden_size, self.hparams.out_shape),
        )


if __name__ == "__main__":
    # A fake unit test
    x = torch.rand((1, 10, 10, 3))

    # Test models
    model1 = PerceptronLit(x[0])
    y = model1(x)
    assert y.shape == (1, 4)

    for eps in [0.0, 1.0]:
        y_infer = model1.inference(x, eps)
        assert isinstance(y_infer, np.ndarray) and y_infer.shape == (1,)

    # Test second model
    model2 = MLPLit(x[0], hidden_size=100)
    y = model2(x)
    assert y.shape == (1, 4)
