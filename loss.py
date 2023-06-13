# -*- coding: utf-8 -*-
"""Loss function.

* Author: Minseong Kim(tyui592@gmail.com)
"""

import torch
import torch.nn.functional as F
from typing import List


def calc_meanstd_loss(features: List[torch.Tensor],
                      targets: List[torch.Tensor],
                      weights: List[float] = None) -> torch.Tensor:
    """Calculate mean std loss with list of features."""
    if weights is None:
        weights = [1/len(features)] * len(features)

    loss = 0
    for f, t, w in zip(features, targets, weights):
        f_std, f_mean = torch.std_mean(f.flatten(2), dim=2)
        t_std, t_mean = torch.std_mean(t.flatten(2), dim=2)
        loss += (F.mse_loss(f_std, t_std) + F.mse_loss(f_mean, t_mean)) * w
    return loss / len(features)


def calc_l2_loss(features: List[torch.Tensor],
                 targets: List[torch.Tensor],
                 weights: List[float] = None) -> torch.Tensor:
    """Calculate content(L2) loss with list of features."""
    if weights is None:
        weights = [1/len(features)] * len(features)

    loss = 0
    for f, t, w in zip(features, targets, weights):
        loss += F.mse_loss(f, t) * w
    return loss / len(features)


def calc_uncorrelation_loss(features: List[torch.Tensor],
                            weights: List[float] = None,
                            eps: float = 1e-5) -> torch.Tensor:
    """Calculate uncorrealtion loss with list of features."""
    if weights is None:
        weights = [1/len(features)] * len(features)

    loss = 0
    for f, w in zip(features, weights):
        # flatten a feature map to the vector
        v = f.flatten(2)

        # mean vector
        m = torch.mean(v, dim=2, keepdim=True)

        # move to zero mean
        zm = v - m

        # calculate the covariance
        cov = torch.bmm(zm, zm.transpose(2, 1))

        # correlation coefficient
        zm_std = torch.sqrt(torch.sum(torch.pow(zm, 2), dim=2, keepdim=True))
        denominator = torch.bmm(zm_std, zm_std.transpose(2, 1))
        corr = cov / (denominator + eps)

        # sum all off-diagonal terms
        num_ch = corr.shape[1]
        ones = torch.ones(num_ch).unsqueeze(0).type_as(corr)
        diag = torch.eye(num_ch).unsqueeze(0).type_as(corr)
        offdiag = ones - diag

        # normalize with the number of off-diagonals
        uncorr_loss = torch.sum(torch.abs(corr) * offdiag) / torch.sum(offdiag)
        loss += uncorr_loss * w
    return loss / len(features)


def calc_channel_loss(feature: torch.Tensor,
                      eps: float = 1e-5) -> torch.Tensor:
    """Calculate the number of nonzero response (l0-norm) channel."""
    B, C = feature.shape[:2]

    vector = feature.flatten(2)

    l2_norm = torch.norm(vector, p=2, dim=2)

    l0_norm = l2_norm / (l2_norm + eps)

    channel_loss = torch.sum(l0_norm)
    return channel_loss / (B * C)


def calc_xor_loss(feature: torch.Tensor,
                  eps: float = 1e-5) -> torch.Tensor:
    """Calculate positional variation of nonzero/zero response channel."""
    B, C = feature.shape[:2]

    vector = feature.flatten(2)

    l2_norm = torch.norm(vector, p=2, dim=2)

    l0_norm = l2_norm / (l2_norm + eps)

    l0_norm_adddim = l0_norm.unsqueeze(0)

    diff = torch.abs(l0_norm_adddim - l0_norm_adddim.transpose(1, 0))
    xor = torch.sum(diff, dim=2)

    xor_loss = torch.sum(torch.triu(xor))
    return xor_loss / (C * B * (B - 1) / 2)
