import sys
import os

import numpy as np

import torch
import torch.nn as nn
import torch.functional as F

class NoamOptim(object):
    """ 
    OpenAI "Attention is All You Kneed" Optimizer wrapper for learning rate scheduling.
    """

    def __init__(self, optimizer, d_model, factor=2, n_warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.factor = factor
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.n_steps += 1
        lr = self.get_lr()
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        self.optimizer.step()

    def get_lr(self):
        return self.factor * (
                self.d_model ** (-0.5)
                * min(self.n_steps ** (-0.5), self.n_steps * self.n_warmup_steps ** (-1.5))
        )
        
    def get_state_dict(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "optimizer_decay_step": self.step(),
            "optimizer_d_model": self.d_model,
            "optimizer_n_warmup_steps": self.n_warmup_steps,
            "optimizer_factor": self.factor,
        }