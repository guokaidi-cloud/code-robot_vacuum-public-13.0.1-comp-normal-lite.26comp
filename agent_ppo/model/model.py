#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Simple MLP policy network for Robot Vacuum.
清扫大作战策略网络。
"""

import torch
import torch.nn as nn

from agent_ppo.conf.conf import Config


def _make_fc(in_dim, out_dim, gain=1.41421):
    """Create a linear layer with orthogonal initialization.

    创建正交初始化的线性层。
    """
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class Model(nn.Module):
    """Dual-head MLP for Robot Vacuum.

    清扫大作战双头 MLP 策略网络。
    """

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "robot_vacuum"
        self.device = device

        obs_dim = Config.DIM_OF_OBSERVATION  # 69
        act_num = Config.ACTION_NUM  # 8

        # Shared backbone / 共享骨干网络
        self.backbone = nn.Sequential(
            _make_fc(obs_dim, 128),
            nn.ReLU(),
            _make_fc(128, 64),
            nn.ReLU(),
        )

        # Actor head: outputs action logits / 策略头：输出动作 logits
        self.actor_head = _make_fc(64, act_num, gain=0.01)

        # Critic head: outputs single state value / 价值头：输出单个状态价值
        self.critic_head = _make_fc(64, 1, gain=0.01)

    def forward(self, s, inference=False):
        """Forward pass.

        前向传播。
        """
        x = s.to(torch.float32)
        h = self.backbone(x)
        logits = self.actor_head(h)
        value = self.critic_head(h)
        return [logits, value]

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
