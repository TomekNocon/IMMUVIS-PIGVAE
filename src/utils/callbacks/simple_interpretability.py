# src/utils/callbacks/minimal_test.py

import torch
import pytorch_lightning as pl
from lightning.pytorch.callbacks import Callback
from typing import Any, Dict, Optional

class SimpleInterpretabilityCallback(Callback):
    """
    Minimal test callback - properly inherits from Lightning Callback.
    """
    
    def __init__(self, log_every_n_steps: int = 10):
        # CRITICAL: Must call parent constructor
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        print("MinimalTestCallback initialized!")
    
    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """Called when fit, validate, test, predict, or tune begins"""
        print(f"MinimalTestCallback: Setup called for stage: {stage}")
    
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when training starts"""
        print("MinimalTestCallback: Training started!")
    
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Called after each training batch"""
        if batch_idx % self.log_every_n_steps == 0:
            print(f"MinimalTestCallback: Batch {batch_idx} completed")
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called at the end of validation epoch"""
        print(f"MinimalTestCallback: Validation epoch {trainer.current_epoch} completed")
    
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when training ends"""
        print("MinimalTestCallback: Training completed!")
    
    def state_dict(self) -> Dict[str, Any]:
        """
        CRITICAL: This method must exist and return a dict for Lightning's validation.
        Called when saving checkpoints.
        """
        return {
            'log_every_n_steps': self.log_every_n_steps,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        CRITICAL: This method must exist for Lightning's validation.
        Called when loading from checkpoints.
        """
        self.log_every_n_steps = state_dict.get('log_every_n_steps', 10)
    
    def on_save_checkpoint(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        checkpoint: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Called when saving checkpoint"""
        return self.state_dict()
    
    def on_load_checkpoint(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        callback_state: Dict[str, Any]
    ) -> None:
        """Called when loading checkpoint"""
        self.load_state_dict(callback_state)