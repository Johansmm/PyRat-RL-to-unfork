"""Trainer based on Pytorch Lightning, that update memory in-place,
given the actions of the player"""
import torch
import pytorch_lightning as pl
from pytorch_lightning.trainer.trainer import (
    Union, Logger, Iterable, Optional, List, Callback, Dict, timedelta, Accelerator,
    Strategy, Path, _PATH, Profiler, PLUGIN_INPUT, _LITERAL_WARN, _defaults_from_env_vars)


class RLTrainer(pl.Trainer):
    """Inspire of lightning_examples/reinforce-learning-DQN.html, extend classic
    trainer with two functions:

        1. :func:`populate` : Normally used by dataloaders in :func:`setup`.
        2. :func:`play_step` : Usefull by the agent.
    """

    @_defaults_from_env_vars
    def __init__(self, logger: Union[Logger, Iterable[Logger], bool] = True,
                 enable_checkpointing: bool = True,
                 callbacks: Optional[Union[List[Callback], Callback]] = None,
                 default_root_dir: Optional[_PATH] = None,
                 gradient_clip_val: Optional[Union[int, float]] = None,
                 gradient_clip_algorithm: Optional[str] = None,
                 num_nodes: int = 1,
                 num_processes: Optional[int] = None,
                 devices: Optional[Union[List[int], str, int]] = None,
                 gpus: Optional[Union[List[int], str, int]] = None,
                 auto_select_gpus: bool = False,
                 tpu_cores: Optional[Union[List[int], str, int]] = None,
                 ipus: Optional[int] = None,
                 enable_progress_bar: bool = True,
                 overfit_batches: Union[int, float] = 0,
                 track_grad_norm: Union[int, float, str] = -1,
                 check_val_every_n_epoch: Optional[int] = 1,
                 fast_dev_run: Union[int, bool] = False,
                 accumulate_grad_batches: Optional[Union[int, Dict[int, int]]] = None,
                 max_epochs: Optional[int] = None,
                 min_epochs: Optional[int] = None,
                 max_steps: int = -1,
                 min_steps: Optional[int] = None,
                 max_time: Optional[Union[str, timedelta, Dict[str, int]]] = None,
                 limit_train_batches: Optional[Union[int, float]] = None,
                 limit_val_batches: Optional[Union[int, float]] = None,
                 limit_test_batches: Optional[Union[int, float]] = None,
                 limit_predict_batches: Optional[Union[int, float]] = None,
                 val_check_interval: Optional[Union[int, float]] = None,
                 log_every_n_steps: int = 50,
                 accelerator: Optional[Union[str, Accelerator]] = None,
                 strategy: Optional[Union[str, Strategy]] = None,
                 sync_batchnorm: bool = False,
                 precision: Union[int, str] = 32,
                 enable_model_summary: bool = True,
                 num_sanity_val_steps: int = 2,
                 resume_from_checkpoint: Optional[Union[Path, str]] = None,
                 profiler: Optional[Union[Profiler, str]] = None,
                 benchmark: Optional[bool] = None,
                 deterministic: Optional[Union[bool, _LITERAL_WARN]] = None,
                 reload_dataloaders_every_n_epochs: int = 0,
                 auto_lr_find: Union[bool, str] = False,
                 replace_sampler_ddp: bool = True,
                 detect_anomaly: bool = False,
                 auto_scale_batch_size: Union[str, bool] = False,
                 plugins: Optional[Union[PLUGIN_INPUT, List[PLUGIN_INPUT]]] = None,
                 amp_backend: str = "native",
                 amp_level: Optional[str] = None,
                 move_metrics_to_cpu: bool = False,
                 multiple_trainloader_mode: str = "max_size_cycle",
                 inference_mode: bool = True,
                 enviroment=None) -> None:
        super().__init__(logger, enable_checkpointing, callbacks, default_root_dir,
                         gradient_clip_val, gradient_clip_algorithm, num_nodes, num_processes,
                         devices, gpus, auto_select_gpus, tpu_cores, ipus, enable_progress_bar,
                         overfit_batches, track_grad_norm, check_val_every_n_epoch, fast_dev_run,
                         accumulate_grad_batches, max_epochs, min_epochs, max_steps, min_steps,
                         max_time, limit_train_batches, limit_val_batches, limit_test_batches,
                         limit_predict_batches, val_check_interval, log_every_n_steps, accelerator,
                         strategy, sync_batchnorm, precision, enable_model_summary,
                         num_sanity_val_steps, resume_from_checkpoint, profiler, benchmark,
                         deterministic, reload_dataloaders_every_n_epochs, auto_lr_find,
                         replace_sampler_ddp, detect_anomaly, auto_scale_batch_size, plugins,
                         amp_backend, amp_level, move_metrics_to_cpu, multiple_trainloader_mode,
                         inference_mode)
        self.enviroment = enviroment

    def populate(self, steps: int = 1000) -> None:
        """Carries out several random steps through the environment to initially
        fill up the replay buffer with experiences.

        Parameters
        ----------
        steps : int
            Number of random steps to populate the buffer with
        """
        for _ in range(steps):
            self.play_step(epsilon=1.0)

    @torch.no_grad()
    def play_step(self, epsilon: float = 0.0):
        """Carries out a single interaction step between the agent and the environment.

        Parameters
        ----------
        epsilon : float, optional
            Take an action by the agent (closer to 0.0) or randomly (closer to 1.0),
            by default 0.0

        Returns
        -------
        float
            Reward gain by the action
        bool
            Indicates if the action ended with the game
        """
        # Play the action over the current state of enviroment
        state = self.enviroment.observe()
        torch_state = torch.from_numpy(state).to(torch.float32)
        action = self.model.inference(torch_state, epsilon=epsilon).squeeze()
        action_str = self.enviroment.action_space[action]
        new_state, reward, game_over = self.enviroment.act(action_str)

        # And save the current status
        self.datamodule.train_dataset.remember((state[0], action, reward, new_state[0]), game_over)

        # If game is over, create a new game
        if game_over:
            self.enviroment.reset()

        return reward, game_over
