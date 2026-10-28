# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA, RLE_WEIGHT
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast

from .metrics import bbox_iou, probiou
from .tal import bbox2dist, rbox2dist


class VarifocalLoss(nn.Module):
    """Varifocal loss by Zhang et al.

    Implements the Varifocal Loss function for addressing class imbalance in object detection by focusing on
    hard-to-classify examples and balancing positive/negative samples.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (float): The balancing factor used to address class imbalance.

    References:
        https://arxiv.org/abs/2008.13367
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        """Initialize the VarifocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Compute varifocal loss between predictions and ground truth."""
        weight = self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5).

    Implements the Focal Loss function for addressing class imbalance by down-weighting easy examples and focusing on
    hard negatives during training.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (torch.Tensor): The balancing factor used to address class imbalance.
    """

    def __init__(self, gamma: float = 1.5, alpha: float = 0.25):
        """Initialize FocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss with modulating factors for class imbalance."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if (self.alpha > 0).any():
            self.alpha = self.alpha.to(device=pred.device, dtype=pred.dtype)
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Criterion class for computing Distribution Focal Loss (DFL)."""

    def __init__(self, reg_max: int = 16) -> None:
        """Initialize the DFL module with regularization maximum."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return sum of left and right DFL losses from https://ieeexplore.ieee.org/document/9792391."""
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


def _bbox_center_distance_ratio(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute normalized center distance term rho^2/c^2 for xyxy boxes."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)

    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    c2 = cw.pow(2) + ch.pow(2) + eps
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4
    return rho2 / c2


def _bbox_axis_distance_ratio(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute axis-aware normalized center distance dx^2/cw^2 + dy^2/ch^2 for xyxy boxes."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)

    cw = (b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)).clamp(min=eps)
    ch = (b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)).clamp(min=eps)
    dx2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2)) / 4
    dy2 = ((b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4
    return dx2 / cw.pow(2) + dy2 / ch.pow(2)


def _bbox_size_aware_center_ratio(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute center distance normalized by object scales, giving more weight to small-object offsets."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)

    w1 = (b1_x2 - b1_x1).clamp(min=eps)
    h1 = (b1_y2 - b1_y1).clamp(min=eps)
    w2 = (b2_x2 - b2_x1).clamp(min=eps)
    h2 = (b2_y2 - b2_y1).clamp(min=eps)
    scale2 = ((w1 + w2) / 2).pow(2) + ((h1 + h2) / 2).pow(2) + eps
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4
    return rho2 / scale2


def _bbox_shape_mismatch(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute shape mismatch term using log-scale width/height ratios."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)

    w1 = (b1_x2 - b1_x1).clamp(min=eps)
    h1 = (b1_y2 - b1_y1).clamp(min=eps)
    w2 = (b2_x2 - b2_x1).clamp(min=eps)
    h2 = (b2_y2 - b2_y1).clamp(min=eps)

    dw = (w1 / w2).log().abs()
    dh = (h1 / h2).log().abs()
    return dw + dh


def _bbox_corner_distance_ratio(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute normalized corner-distance mismatch for xyxy boxes."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)

    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    c2 = cw.pow(2) + ch.pow(2) + eps

    tl = (b1_x1 - b2_x1).pow(2) + (b1_y1 - b2_y1).pow(2)
    br = (b1_x2 - b2_x2).pow(2) + (b1_y2 - b2_y2).pow(2)
    return (tl + br) / (2.0 * c2)


def _bbox_scale_consistency(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute width-height scale mismatch for xyxy boxes."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)

    w1 = (b1_x2 - b1_x1).clamp(min=eps)
    h1 = (b1_y2 - b1_y1).clamp(min=eps)
    w2 = (b2_x2 - b2_x1).clamp(min=eps)
    h2 = (b2_y2 - b2_y1).clamp(min=eps)

    return ((w1 - w2).abs() / (w2 + eps)) + ((h1 - h2).abs() / (h2 + eps))


def _interpolate_boxes(pred: torch.Tensor, target: torch.Tensor, steps: int) -> torch.Tensor:
    """Linearly interpolate boxes from pred to target for InterpIoU computation."""
    alphas = torch.linspace(0.0, 1.0, steps, device=pred.device, dtype=pred.dtype).view(steps, 1, 1)
    return pred.unsqueeze(0) * (1.0 - alphas) + target.unsqueeze(0) * alphas


class BboxLoss(nn.Module):
    """Criterion class for computing training losses for bounding boxes."""

    def __init__(
        self,
        reg_max: int = 16,
        iou_type: str = "wiouv3",
        wiou_gamma: float = 1.9,
        wiou_delta: float = 3.0,
        wiou_half_life: float = 7000.0,
        dynciouwiou_tau: float = 1.0,
        dynciouwiou_pow: float = 1.0,
        dynciouwiou_min: float = 0.0,
        dynciouwiou_max: float = 1.0,
        cwiou_stable_min: float = 0.6,
        cwiou_stable_max: float = 2.5,
        qualitycwiou_k: float = 8.0,
        qualitycwiou_t: float = 0.5,
        focalcwiou_gamma: float = 0.2,
        focalcwiou_pow: float = 2.0,
        cwiouv19_temp_base: float = 0.8,
        cwiouv19_temp_range: float = 0.6,
        cwiouv19_hardness_pow: float = 1.0,
        cwiouv19_dist_cap: float = 0.0,
        awciouv1_axis_w: float = 0.20,
        awciouv1_shape_w: float = 0.05,
        qgciou_ciou_w: float = 1.0,
        qgciou_axis_w: float = 0.20,
        qgciou_shape_w: float = 0.05,
    ):
        """Initialize the BboxLoss module with regularization maximum and IoU loss type."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.iou_type = (iou_type or "wiouv3").lower()
        self.qgciou_ciou_w = 1.0
        if self.iou_type in {"qgciou", "qg-ciou"}:
            # Paper Eq. (8): r * (omega1 * L_CIoU + omega2 * L_axis
            #                       + omega3 * L_shape).
            self.iou_type = "awciouv1"
            self.qgciou_ciou_w = float(qgciou_ciou_w)
            awciouv1_axis_w = float(qgciou_axis_w)
            awciouv1_shape_w = float(qgciou_shape_w)
            if min(self.qgciou_ciou_w, awciouv1_axis_w, awciouv1_shape_w) < 0:
                raise ValueError("QG-CIoU weights must be nonnegative")
        self.awciouv1_axis_w = awciouv1_axis_w
        self.awciouv1_shape_w = awciouv1_shape_w
        if self.iou_type not in {
            "wiouv3",
            "wiou",
            "ciou",
            "cwiou",
            "cwiouv2",
            "cwiouv3",
            "cwiouv4",
            "cwiouv5",
            "cwiouv6",
            "cwiouv7",
            "cwiouv8",
            "cwiouv9",
            "cwiouv10",
            "cwiouv11",
            "cwiouv12",
            "cwiouv13",
            "cwiouv14",
            "cwiouv15",
            "cwiouv16",
            "cwiouv17",
            "cwiouv18",
            "cwiouv19",
            "cwiouv20",
            "cwiouv21",
            "cwiouv22",
            "cwiouv23",
            "awciouv1",
            "awciouv2",
            "stablecwiou",
            "qualitycwiou",
            "focalcwiou",
            "dynciouwiou",
            "piouv2",
            "interpiou",
        }:
            raise ValueError(
                f"Unsupported box_iou_type='{iou_type}'. Use 'wiouv3', 'wiou', 'ciou', 'cwiou', 'cwiouv2', "
                "'cwiouv3', 'cwiouv4', 'cwiouv5', 'cwiouv6', 'cwiouv7', 'cwiouv8', 'stablecwiou', "
                "'cwiouv9', 'cwiouv10', 'cwiouv11', 'cwiouv12', 'cwiouv13', 'cwiouv14', "
                "'cwiouv15', 'cwiouv16', 'cwiouv17', 'cwiouv18', 'cwiouv19', 'cwiouv20', "
                "'cwiouv21', 'cwiouv22', 'cwiouv23', 'awciouv1', 'awciouv2', "
                "'qualitycwiou', 'focalcwiou', "
                "'dynciouwiou', 'piouv2', or 'interpiou'."
            )
        self.piou_lambda = 1.3
        self.interp_steps = 10
        # WIoU v3 non-monotonic focusing parameters.
        self.wiou_gamma = float(wiou_gamma)
        self.wiou_delta = float(wiou_delta)
        self.wiou_half_life = float(wiou_half_life)
        self.wiou_momentum = 1 - 0.5 ** (1 / self.wiou_half_life)
        self.dynciouwiou_tau = float(dynciouwiou_tau)
        self.dynciouwiou_pow = float(dynciouwiou_pow)
        self.dynciouwiou_min = float(dynciouwiou_min)
        self.dynciouwiou_max = float(dynciouwiou_max)
        self.cwiou_stable_min = float(cwiou_stable_min)
        self.cwiou_stable_max = float(cwiou_stable_max)
        self.qualitycwiou_k = float(qualitycwiou_k)
        self.qualitycwiou_t = float(qualitycwiou_t)
        self.focalcwiou_gamma = float(focalcwiou_gamma)
        self.focalcwiou_pow = float(focalcwiou_pow)
        self.cwiouv19_temp_base = float(cwiouv19_temp_base)
        self.cwiouv19_temp_range = float(cwiouv19_temp_range)
        self.cwiouv19_hardness_pow = float(cwiouv19_hardness_pow)
        self.cwiouv19_dist_cap = float(cwiouv19_dist_cap)
        if self.wiou_gamma <= 0:
            raise ValueError(f"wiou_gamma must be > 0, got {self.wiou_gamma}")
        if self.wiou_delta <= 0:
            raise ValueError(f"wiou_delta must be > 0, got {self.wiou_delta}")
        if self.wiou_half_life <= 0:
            raise ValueError(f"wiou_half_life must be > 0, got {self.wiou_half_life}")
        if self.dynciouwiou_tau <= 0:
            raise ValueError(f"dynciouwiou_tau must be > 0, got {self.dynciouwiou_tau}")
        if self.dynciouwiou_pow <= 0:
            raise ValueError(f"dynciouwiou_pow must be > 0, got {self.dynciouwiou_pow}")
        if self.dynciouwiou_min > self.dynciouwiou_max:
            raise ValueError(
                "dynciouwiou_min must be <= dynciouwiou_max, "
                f"got {self.dynciouwiou_min} > {self.dynciouwiou_max}"
            )
        if self.cwiou_stable_min <= 0:
            raise ValueError(f"cwiou_stable_min must be > 0, got {self.cwiou_stable_min}")
        if self.cwiou_stable_min > self.cwiou_stable_max:
            raise ValueError(
                "cwiou_stable_min must be <= cwiou_stable_max, "
                f"got {self.cwiou_stable_min} > {self.cwiou_stable_max}"
            )
        if self.qualitycwiou_k <= 0:
            raise ValueError(f"qualitycwiou_k must be > 0, got {self.qualitycwiou_k}")
        if not (0.0 <= self.qualitycwiou_t <= 1.0):
            raise ValueError(f"qualitycwiou_t must be in [0, 1], got {self.qualitycwiou_t}")
        if self.focalcwiou_gamma < 0:
            raise ValueError(f"focalcwiou_gamma must be >= 0, got {self.focalcwiou_gamma}")
        if self.focalcwiou_pow <= 0:
            raise ValueError(f"focalcwiou_pow must be > 0, got {self.focalcwiou_pow}")
        if self.cwiouv19_temp_base <= 0:
            raise ValueError(f"cwiouv19_temp_base must be > 0, got {self.cwiouv19_temp_base}")
        if self.cwiouv19_temp_range < 0:
            raise ValueError(f"cwiouv19_temp_range must be >= 0, got {self.cwiouv19_temp_range}")
        if self.cwiouv19_hardness_pow <= 0:
            raise ValueError(f"cwiouv19_hardness_pow must be > 0, got {self.cwiouv19_hardness_pow}")
        if self.cwiouv19_dist_cap < 0:
            raise ValueError(f"cwiouv19_dist_cap must be >= 0, got {self.cwiouv19_dist_cap}")
        self.register_buffer("wiou_iou_mean", torch.tensor(1.0))

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for bounding boxes."""
        self.awciouv1_stats = None
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        eps = 1e-7
        pred_fg = pred_bboxes[fg_mask]
        target_fg = target_bboxes[fg_mask]
        if self.iou_type == "ciou":
            iou = bbox_iou(pred_fg, target_fg, xywh=False, CIoU=True)
            loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        elif self.iou_type in {
            "cwiou",
            "cwiouv2",
            "cwiouv3",
            "cwiouv4",
            "cwiouv5",
            "cwiouv6",
            "cwiouv7",
            "cwiouv8",
            "cwiouv9",
            "cwiouv10",
            "cwiouv11",
            "cwiouv12",
            "cwiouv13",
            "cwiouv14",
            "cwiouv15",
            "cwiouv16",
            "cwiouv17",
            "cwiouv18",
            "cwiouv19",
            "cwiouv20",
            "cwiouv21",
            "cwiouv22",
            "cwiouv23",
            "awciouv1",
            "awciouv2",
            "stablecwiou",
            "qualitycwiou",
            "focalcwiou",
        }:
            # cwIoU: use WIoU-style dynamic factor as weight for CIoU loss.
            ciou = bbox_iou(pred_fg, target_fg, xywh=False, CIoU=True)
            ciou_loss = (1.0 - ciou).clamp(min=eps)

            iou = bbox_iou(pred_fg, target_fg, xywh=False)
            iou_loss = (1.0 - iou).clamp(min=eps)
            dist_weight = _bbox_center_distance_ratio(pred_fg, target_fg, eps=eps).exp()

            if self.training:
                with torch.no_grad():
                    mean_iou_loss = iou_loss.detach().mean().to(self.wiou_iou_mean.dtype)
                    self.wiou_iou_mean.mul_(1.0 - self.wiou_momentum).add_(self.wiou_momentum * mean_iou_loss)

            iou_mean = self.wiou_iou_mean.to(iou_loss.dtype).clamp(min=eps)
            beta = (iou_loss.detach() / iou_mean).clamp(min=eps)
            alpha = self.wiou_delta * torch.pow(iou_loss.new_tensor(self.wiou_gamma), beta - self.wiou_delta)
            focus = beta / (alpha + eps)
            if self.iou_type == "cwiouv2":
                # cwiouv2 removes distance weighting and keeps only dynamic focus term.
                wiou_weight = focus
            elif self.iou_type == "awciouv1":
                # QG-CIoU (legacy identifier: awciouv1). The quality weight is
                # detached through beta; geometry terms retain gradients.
                wiou_weight = focus
                axis_geo = _bbox_axis_distance_ratio(pred_fg, target_fg, eps=eps)
                shape_geo = _bbox_shape_mismatch(pred_fg, target_fg, eps=eps)
                ciou_loss = (
                    self.qgciou_ciou_w * ciou_loss
                    + self.awciouv1_axis_w * axis_geo
                    + self.awciouv1_shape_w * shape_geo
                )
                self.awciouv1_stats = {
                    "awciouv1_ciou_mean": ciou_loss.detach().mean().item(),
                    "awciouv1_axis_mean": axis_geo.detach().mean().item(),
                    "awciouv1_axis_w_mean": (axis_geo * self.awciouv1_axis_w).detach().mean().item(),
                    "awciouv1_shape_mean": shape_geo.detach().mean().item(),
                    "awciouv1_shape_w_mean": (shape_geo * self.awciouv1_shape_w).detach().mean().item(),
                }
            elif self.iou_type == "awciouv2":
                # AWCIoUv2: keep cwiouv2 weighting, strengthen geometry with corner and scale consistency.
                wiou_weight = focus
                hard_gate = (1.0 - iou.detach()).clamp(0.0, 1.0)
                axis_geo = _bbox_axis_distance_ratio(pred_fg, target_fg, eps=eps)
                shape_geo = _bbox_shape_mismatch(pred_fg, target_fg, eps=eps)
                corner_geo = _bbox_corner_distance_ratio(pred_fg, target_fg, eps=eps)
                scale_geo = _bbox_scale_consistency(pred_fg, target_fg, eps=eps)
                geo = (0.15 + 0.15 * hard_gate) * axis_geo + 0.05 * shape_geo + 0.10 * corner_geo + 0.05 * scale_geo
                ciou_loss = ciou_loss + geo
            elif self.iou_type == "cwiouv3":
                # cwiouv3: re-introduce distance with softened strength for stability.
                wiou_weight = focus * dist_weight.sqrt()
            elif self.iou_type == "cwiouv4":
                # cwiouv4: hard samples get stronger distance emphasis, easy samples stay close to cwiouv2.
                hard_gate = (1.0 - iou.detach()).clamp(0.0, 1.0)
                wiou_weight = focus * (1.0 + hard_gate * (dist_weight - 1.0))
            elif self.iou_type == "cwiouv5":
                # cwiouv5: blend plain and stabilized weighting to avoid extreme penalties.
                wiou_raw = dist_weight * focus
                wiou_stable = torch.log1p(wiou_raw).clamp(self.cwiou_stable_min, self.cwiou_stable_max)
                blend = (1.0 - iou.detach()).clamp(0.0, 1.0)
                wiou_weight = (1.0 - blend) * wiou_stable + blend * wiou_raw
            elif self.iou_type == "cwiouv6":
                # cwiouv6: linear-mild dist enhancement around 1.0 to improve AP50 without over-penalizing.
                wiou_weight = focus * (1.0 + 0.5 * (dist_weight - 1.0))
            elif self.iou_type == "cwiouv7":
                # cwiouv7: capped dist enhancement to suppress outlier gradients from very large center offsets.
                wiou_weight = focus * dist_weight.clamp(min=1.0, max=3.0)
            elif self.iou_type == "cwiouv8":
                # cwiouv8: hardness-adaptive dist exponent; harder samples get stronger distance emphasis.
                hard_gate = (1.0 - iou.detach()).clamp(0.0, 1.0)
                dist_pow = 0.5 + 1.5 * hard_gate
                wiou_weight = focus * torch.pow(dist_weight, dist_pow)
            elif self.iou_type == "cwiouv9":
                # cwiouv9: stronger capped distance enhancement, usually beneficial for AP50-focused optimization.
                wiou_weight = focus * dist_weight.clamp(min=1.0, max=4.0)
            elif self.iou_type == "cwiouv10":
                # cwiouv10: hardness-adaptive cap, hard samples get larger distance cap than easy ones.
                hard_gate = (1.0 - iou.detach()).clamp(0.0, 1.0)
                dist_cap = 2.0 + 2.0 * hard_gate
                wiou_weight = focus * torch.minimum(dist_weight, dist_cap)
            elif self.iou_type == "cwiouv11":
                # cwiouv11: emphasize near-threshold samples (IoU around 0.5) to target mAP50 gains.
                threshold_gate = torch.sigmoid(10.0 * (0.55 - iou.detach()))
                dist_boost = dist_weight.clamp(min=1.0, max=4.0) - 1.0
                wiou_weight = focus * (1.0 + threshold_gate * dist_boost)
            elif self.iou_type == "cwiouv12":
                # cwiouv12: low dynamic cap avoids collapse with cwiouv9 and keeps gradients controlled.
                hard_gate = (1.0 - iou.detach()).clamp(0.0, 1.0)
                dist_cap = 1.15 + 0.30 * hard_gate
                wiou_weight = focus * torch.minimum(dist_weight, dist_cap)
            elif self.iou_type == "cwiouv13":
                # cwiouv13: emphasize IoU~0.5 band (AP50-sensitive region) with bounded distance boost.
                mid_gate = torch.exp(-((iou.detach() - 0.50).pow(2)) / 0.02)
                dist_boost = (dist_weight - 1.0).clamp(min=0.0, max=1.5)
                wiou_weight = focus * (1.0 + mid_gate * dist_boost)
            elif self.iou_type == "cwiouv14":
                # cwiouv14: adaptive exponent with pre-clamped distance term for robust hard-sample emphasis.
                hard_gate = (1.0 - iou.detach()).clamp(0.0, 1.0)
                dist_base = dist_weight.clamp(min=1.0, max=2.0)
                dist_pow = 0.70 + 0.80 * hard_gate
                wiou_weight = focus * torch.pow(dist_base, dist_pow)
            elif self.iou_type == "cwiouv15":
                # cwiouv15: axis-aware distance weighting to separately penalize x/y misalignment.
                axis_dist = _bbox_axis_distance_ratio(pred_fg, target_fg, eps=eps)
                wiou_weight = focus * axis_dist.exp()
            elif self.iou_type == "cwiouv16":
                # cwiouv16: size-aware center distance, usually stronger for small-object localization.
                size_dist = _bbox_size_aware_center_ratio(pred_fg, target_fg, eps=eps)
                wiou_weight = focus * size_dist.exp()
            elif self.iou_type == "cwiouv17":
                # cwiouv17: coupled distance-shape weighting for both center drift and box geometry mismatch.
                center_dist = _bbox_center_distance_ratio(pred_fg, target_fg, eps=eps)
                shape_dist = _bbox_shape_mismatch(pred_fg, target_fg, eps=eps)
                wiou_weight = focus * (center_dist + 0.5 * shape_dist).exp()
            elif self.iou_type == "cwiouv18":
                # cwiouv18: exp(dist) baseline with hardness boost for low-IoU samples.
                hard_gate = (1.0 - iou.detach()).clamp(0.0, 1.0)
                wiou_weight = focus * dist_weight * (1.0 + 0.35 * hard_gate)
            elif self.iou_type == "cwiouv19":
                # cwiouv19: adaptive exp temperature around baseline to reduce over-penalty on easy samples.
                hard_gate = (1.0 - iou.detach()).clamp(0.0, 1.0)
                center_dist = _bbox_center_distance_ratio(pred_fg, target_fg, eps=eps)
                hard_gate = hard_gate.pow(self.cwiouv19_hardness_pow)
                temp = self.cwiouv19_temp_base + self.cwiouv19_temp_range * hard_gate
                if self.cwiouv19_dist_cap > 0:
                    center_dist = center_dist.clamp(max=self.cwiouv19_dist_cap)
                wiou_weight = focus * torch.exp(center_dist * temp)
            elif self.iou_type == "cwiouv20":
                # cwiouv20: threshold-aware blend, keeps strong exp(dist) for hard cases and relaxes easy cases.
                threshold_gate = torch.sigmoid(10.0 * (0.60 - iou.detach()))
                wiou_weight = focus * (threshold_gate * dist_weight + (1.0 - threshold_gate))
            elif self.iou_type == "cwiouv21":
                # cwiouv21: v19-style adaptive temperature with extra AP50-band boost.
                hard_gate = (1.0 - iou.detach()).clamp(0.0, 1.0)
                center_dist = _bbox_center_distance_ratio(pred_fg, target_fg, eps=eps)
                temp = 0.85 + 0.55 * hard_gate
                ap50_gate = torch.sigmoid(12.0 * (0.58 - iou.detach()))
                wiou_weight = focus * torch.exp(center_dist * temp) * (1.0 + 0.20 * ap50_gate)
            elif self.iou_type == "cwiouv22":
                # cwiouv22: v19-style temperature driven by beta hardness instead of IoU only.
                center_dist = _bbox_center_distance_ratio(pred_fg, target_fg, eps=eps)
                beta_hard = (beta - 1.0).tanh().clamp(-0.9, 0.9)
                temp = 0.85 + 0.45 * beta_hard
                wiou_weight = focus * torch.exp(center_dist * temp)
            elif self.iou_type == "cwiouv23":
                # cwiouv23: blend v19 adaptive exp with capped baseline for hard-sample stability.
                hard_gate = (1.0 - iou.detach()).clamp(0.0, 1.0)
                center_dist = _bbox_center_distance_ratio(pred_fg, target_fg, eps=eps)
                wiou_v19 = torch.exp(center_dist * (0.85 + 0.55 * hard_gate))
                wiou_cap = dist_weight.clamp(min=1.0, max=3.0)
                blend = torch.sigmoid(10.0 * (0.62 - iou.detach()))
                wiou_weight = focus * ((1.0 - blend) * wiou_v19 + blend * wiou_cap)
            else:
                wiou_weight = dist_weight * focus
            if self.iou_type in {"stablecwiou", "qualitycwiou", "focalcwiou"}:
                # Stable-CWIoU: smooth and clamp WIoU weight to prevent over-aggressive gradients.
                wiou_weight = torch.log1p(wiou_weight).clamp(self.cwiou_stable_min, self.cwiou_stable_max)
            if self.iou_type == "qualitycwiou":
                # Quality-Aware CWIoU: easy samples (high IoU) rely more on plain CIoU behavior.
                quality_gate = torch.sigmoid(self.qualitycwiou_k * (iou.detach() - self.qualitycwiou_t))
                wiou_weight = (1.0 - quality_gate) * wiou_weight + quality_gate
            base_cwiou = wiou_weight * ciou_loss
            if self.iou_type == "focalcwiou":
                # Lightweight focal regression term to boost hard low-IoU samples and improve recall.
                focal_term = self.focalcwiou_gamma * iou_loss.pow(self.focalcwiou_pow)
                total_reg = base_cwiou + focal_term
            else:
                total_reg = base_cwiou

            loss_iou = (total_reg * weight).sum() / target_scores_sum
        elif self.iou_type == "dynciouwiou":
            # Dynamic fusion: loss = a * WIoU + (1-a) * CIoU, with a adapted by sample hardness beta.
            ciou = bbox_iou(pred_fg, target_fg, xywh=False, CIoU=True)
            ciou_loss = (1.0 - ciou).clamp(min=eps)

            iou = bbox_iou(pred_fg, target_fg, xywh=False)
            iou_loss = (1.0 - iou).clamp(min=eps)
            dist_weight = _bbox_center_distance_ratio(pred_fg, target_fg, eps=eps).exp()

            if self.training:
                with torch.no_grad():
                    mean_iou_loss = iou_loss.detach().mean().to(self.wiou_iou_mean.dtype)
                    self.wiou_iou_mean.mul_(1.0 - self.wiou_momentum).add_(self.wiou_momentum * mean_iou_loss)

            iou_mean = self.wiou_iou_mean.to(iou_loss.dtype).clamp(min=eps)
            beta = (iou_loss.detach() / iou_mean).clamp(min=eps)
            alpha = self.wiou_delta * torch.pow(iou_loss.new_tensor(self.wiou_gamma), beta - self.wiou_delta)
            focus = beta / (alpha + eps)

            wiou_loss = iou_loss * dist_weight * focus
            dyn_ratio = beta / (self.dynciouwiou_tau + beta)
            dyn_a = dyn_ratio.pow(self.dynciouwiou_pow).clamp(self.dynciouwiou_min, self.dynciouwiou_max)
            fused_loss = dyn_a * wiou_loss + (1.0 - dyn_a) * ciou_loss
            loss_iou = (fused_loss * weight).sum() / target_scores_sum
        elif self.iou_type == "interpiou":
            interp_boxes = _interpolate_boxes(pred_fg, target_fg, self.interp_steps)  # [K, N, 4]
            k, n, _ = interp_boxes.shape
            interp_boxes = interp_boxes.reshape(-1, 4)
            target_repeat = target_fg.unsqueeze(0).expand(k, -1, -1).reshape(-1, 4)

            ious = bbox_iou(interp_boxes, target_repeat, xywh=False).view(k, n)
            step_weights = torch.linspace(0.1, 1.0, k, device=pred_fg.device, dtype=pred_fg.dtype).view(k, 1)
            weighted_iou = (ious * step_weights).sum(dim=0) / step_weights.sum()
            loss_interp = 1.0 - weighted_iou
            loss_iou = (loss_interp.unsqueeze(-1) * weight).sum() / target_scores_sum
        elif self.iou_type == "piouv2":
            iou = bbox_iou(pred_fg, target_fg, xywh=False)

            w_gt = (target_fg[..., 2:3] - target_fg[..., 0:1]).clamp(min=eps)
            h_gt = (target_fg[..., 3:4] - target_fg[..., 1:2]).clamp(min=eps)

            dw1 = (pred_fg[..., 0:1] - target_fg[..., 0:1]).abs()
            dw2 = (pred_fg[..., 2:3] - target_fg[..., 2:3]).abs()
            dh1 = (pred_fg[..., 1:2] - target_fg[..., 1:2]).abs()
            dh2 = (pred_fg[..., 3:4] - target_fg[..., 3:4]).abs()

            p = (dw1 / w_gt + dw2 / w_gt + dh1 / h_gt + dh2 / h_gt) / 4.0
            f_p = 1.0 - torch.exp(-(p.pow(2)))
            l_piou = (1.0 - iou).clamp(min=0.0) + f_p

            q = torch.exp(-p)
            att = 3.0 * (self.piou_lambda * q) * torch.exp(-((self.piou_lambda * q).pow(2)))
            loss_iou = (att * l_piou * weight).sum() / target_scores_sum
        else:
            # WIoU v3: L = r(beta) * exp(rho^2/c^2) * (1 - IoU).
            iou = bbox_iou(pred_fg, target_fg, xywh=False)
            iou_loss = (1.0 - iou).clamp(min=eps)
            dist_weight = _bbox_center_distance_ratio(pred_fg, target_fg, eps=eps).exp()

            if self.training:
                with torch.no_grad():
                    mean_iou_loss = iou_loss.detach().mean().to(self.wiou_iou_mean.dtype)
                    self.wiou_iou_mean.mul_(1.0 - self.wiou_momentum).add_(self.wiou_momentum * mean_iou_loss)

            iou_mean = self.wiou_iou_mean.to(iou_loss.dtype).clamp(min=eps)
            beta = (iou_loss.detach() / iou_mean).clamp(min=eps)
            alpha = self.wiou_delta * torch.pow(iou_loss.new_tensor(self.wiou_gamma), beta - self.wiou_delta)
            focus = beta / (alpha + eps)
            loss_iou = (iou_loss * dist_weight * focus * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            # normalize ltrb by image size
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl


class RLELoss(nn.Module):
    """Residual Log-Likelihood Estimation Loss.

    Attributes:
        size_average (bool): Option to average the loss by the batch_size.
        use_target_weight (bool): Option to use weighted loss.
        residual (bool): Option to add L1 loss and let the flow learn the residual error distribution.

    References:
        https://arxiv.org/abs/2107.11291
        https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/losses/regression_loss.py
    """

    def __init__(self, use_target_weight: bool = True, size_average: bool = True, residual: bool = True):
        """Initialize RLELoss with target weight and residual options.

        Args:
            use_target_weight (bool): Whether to use target weights for loss calculation.
            size_average (bool): Whether to average the loss over elements.
            residual (bool): Whether to include residual log-likelihood term.
        """
        super().__init__()
        self.size_average = size_average
        self.use_target_weight = use_target_weight
        self.residual = residual

    def forward(
        self, sigma: torch.Tensor, log_phi: torch.Tensor, error: torch.Tensor, target_weight: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            sigma (torch.Tensor): Output sigma, shape (N, D).
            log_phi (torch.Tensor): Output log_phi, shape (N).
            error (torch.Tensor): Error, shape (N, D).
            target_weight (torch.Tensor): Weights across different joint types, shape (N).
        """
        log_sigma = torch.log(sigma)
        loss = log_sigma - log_phi.unsqueeze(1)

        if self.residual:
            loss += torch.log(sigma * 2) + torch.abs(error)

        if self.use_target_weight:
            assert target_weight is not None, "'target_weight' should not be None when 'use_target_weight' is True."
            if target_weight.dim() == 1:
                target_weight = target_weight.unsqueeze(1)
            loss *= target_weight

        if self.size_average:
            loss /= len(loss)

        return loss.sum()


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses for rotated bounding boxes."""

    def __init__(self, reg_max: int):
        """Initialize the RotatedBboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for rotated bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = rbox2dist(
                target_bboxes[..., :4], anchor_points, target_bboxes[..., 4:5], reg_max=self.dfl_loss.reg_max - 1
            )
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = rbox2dist(target_bboxes[..., :4], anchor_points, target_bboxes[..., 4:5])
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl


class MultiChannelDiceLoss(nn.Module):
    """Criterion class for computing multi-channel Dice losses."""

    def __init__(self, smooth: float = 1e-6, reduction: str = "mean"):
        """Initialize MultiChannelDiceLoss with smoothing and reduction options.

        Args:
            smooth (float): Smoothing factor to avoid division by zero.
            reduction (str): Reduction method ('mean', 'sum', or 'none').
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate multi-channel Dice loss between predictions and targets."""
        assert pred.size() == target.size(), "the size of predict and target must be equal."

        pred = pred.sigmoid()
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice
        dice_loss = dice_loss.mean(dim=1)

        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss


class BCEDiceLoss(nn.Module):
    """Criterion class for computing combined BCE and Dice losses."""

    def __init__(self, weight_bce: float = 0.5, weight_dice: float = 0.5):
        """Initialize BCEDiceLoss with BCE and Dice weight factors.

        Args:
            weight_bce (float): Weight factor for BCE loss component.
            weight_dice (float): Weight factor for Dice loss component.
        """
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = MultiChannelDiceLoss(smooth=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate combined BCE and Dice loss between predictions and targets."""
        _, _, mask_h, mask_w = pred.shape
        if tuple(target.shape[-2:]) != (mask_h, mask_w):  # downsample to the same size as pred
            target = F.interpolate(target, (mask_h, mask_w), mode="nearest")
        return self.weight_bce * self.bce(pred, target) + self.weight_dice * self.dice(pred, target)


class KeypointLoss(nn.Module):
    """Criterion class for computing keypoint losses."""

    def __init__(self, sigmas: torch.Tensor) -> None:
        """Initialize the KeypointLoss class with keypoint sigmas."""
        super().__init__()
        self.sigmas = sigmas

    def forward(
        self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor, kpt_mask: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Calculate keypoint loss factor and Euclidean distance loss for keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses for YOLOv8 object detection."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):  # model must be de-paralleled
        """Initialize v8DetectionLoss with model parameters and task-aligned assignment settings."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )
        self.bbox_loss = BboxLoss(
            m.reg_max,
            iou_type=getattr(h, "box_iou_type", "wiouv3"),
            wiou_gamma=getattr(h, "wiou_gamma", 1.9),
            wiou_delta=getattr(h, "wiou_delta", 3.0),
            wiou_half_life=getattr(h, "wiou_half_life", 7000.0),
            dynciouwiou_tau=getattr(h, "dynciouwiou_tau", 1.0),
            dynciouwiou_pow=getattr(h, "dynciouwiou_pow", 1.0),
            dynciouwiou_min=getattr(h, "dynciouwiou_min", 0.0),
            dynciouwiou_max=getattr(h, "dynciouwiou_max", 1.0),
            cwiou_stable_min=getattr(h, "cwiou_stable_min", 0.6),
            cwiou_stable_max=getattr(h, "cwiou_stable_max", 2.5),
            qualitycwiou_k=getattr(h, "qualitycwiou_k", 8.0),
            qualitycwiou_t=getattr(h, "qualitycwiou_t", 0.5),
            focalcwiou_gamma=getattr(h, "focalcwiou_gamma", 0.2),
            focalcwiou_pow=getattr(h, "focalcwiou_pow", 2.0),
            cwiouv19_temp_base=getattr(h, "cwiouv19_temp_base", 0.8),
            cwiouv19_temp_range=getattr(h, "cwiouv19_temp_range", 0.6),
            cwiouv19_hardness_pow=getattr(h, "cwiouv19_hardness_pow", 1.0),
            cwiouv19_dist_cap=getattr(h, "cwiouv19_dist_cap", 0.0),
            awciouv1_axis_w=getattr(h, "awciouv1_axis_w", 0.20),
            awciouv1_shape_w=getattr(h, "awciouv1_shape_w", 0.05),
            qgciou_ciou_w=getattr(h, "qgciou_ciou_w", 1.0),
            qgciou_axis_w=getattr(h, "qgciou_axis_w", 0.20),
            qgciou_shape_w=getattr(h, "qgciou_shape_w", 0.05),
        ).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets by converting to tensor format and scaling coordinates."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size and return foreground mask and
        target indices.
        """
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if hasattr(self.bbox_loss, "awciouv1_stats"):
            self.bbox_loss.awciouv1_stats = None
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )  # loss(box, cls, dfl)

    def parse_output(
        self, preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Parse model predictions to extract features."""
        return preds[1] if isinstance(preds, tuple) else preds

    def __call__(
        self,
        preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        return self.loss(self.parse_output(preds), batch)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate detection loss using assigned targets."""
        batch_size = preds["boxes"].shape[0]
        loss, loss_detach = self.get_assigned_targets_and_loss(preds, batch)[1:]
        return loss * batch_size, loss_detach


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 segmentation."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):  # model must be de-paralleled
        """Initialize the v8SegmentationLoss class with model parameters and mask overlap setting."""
        super().__init__(model, tal_topk, tal_topk2)
        self.overlap = model.args.overlap_mask
        self.bcedice_loss = BCEDiceLoss(weight_bce=0.5, weight_dice=0.5)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the combined loss for detection and segmentation."""
        pred_masks, proto = preds["mask_coefficient"].permute(0, 2, 1).contiguous(), preds["proto"]
        loss = torch.zeros(5, device=self.device)  # box, seg, cls, dfl, semseg
        if isinstance(proto, tuple) and len(proto) == 2:
            proto, pred_semseg = proto
        else:
            pred_semseg = None
        (fg_mask, target_gt_idx, target_bboxes, _, _), det_loss, _ = self.get_assigned_targets_and_loss(preds, batch)
        # NOTE: re-assign index for consistency for now. Need to be removed in the future.
        loss[0], loss[2], loss[3] = det_loss[0], det_loss[1], det_loss[2]

        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        if fg_mask.sum():
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                # masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]
                proto = F.interpolate(proto, masks.shape[-2:], mode="bilinear", align_corners=False)

            imgsz = (
                torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_masks.dtype) * self.stride[0]
            )
            loss[1] = self.calculate_segmentation_loss(
                fg_mask,
                masks,
                target_gt_idx,
                target_bboxes,
                batch["batch_idx"].view(-1, 1),
                proto,
                pred_masks,
                imgsz,
            )
            if pred_semseg is not None:
                sem_masks = batch["sem_masks"].to(self.device)  # NxHxW
                sem_masks = F.one_hot(sem_masks.long(), num_classes=self.nc).permute(0, 3, 1, 2).float()  # NxCxHxW

                if self.overlap:
                    mask_zero = masks == 0  # NxHxW
                    sem_masks[mask_zero.unsqueeze(1).expand_as(sem_masks)] = 0
                else:
                    batch_idx = batch["batch_idx"].view(-1)  # [total_instances]
                    for i in range(batch_size):
                        instance_mask_i = masks[batch_idx == i]  # [num_instances_i, H, W]
                        if len(instance_mask_i) == 0:
                            continue
                        sem_masks[i, :, instance_mask_i.sum(dim=0) == 0] = 0

                loss[4] = self.bcedice_loss(pred_semseg, sem_masks)
                loss[4] *= self.hyp.box  # seg gain

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss
            if pred_semseg is not None:
                loss[4] += (pred_semseg * 0).sum()

        loss[1] *= self.hyp.box  # seg gain
        return loss * batch_size, loss.detach()  # loss(box, seg, cls, dfl, semseg)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (N, H, W), where N is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (N, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (N, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (N,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if self.overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int = 10):  # model must be de-paralleled
        """Initialize v8PoseLoss with model parameters and keypoint-specific loss functions."""
        super().__init__(model, tal_topk, tal_topk2)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the total loss and detach it for pose estimation."""
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
        loss = torch.zeros(5, device=self.device)  # box, kpt_location, kpt_visibility, cls, dfl
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, batch)
        )
        # NOTE: re-assign index for consistency for now. Need to be removed in the future.
        loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

        batch_size = pred_kpts.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype) * self.stride[0]

        # Pboxes
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        # Keypoint loss
        if fg_mask.sum():
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch["batch_idx"].view(-1, 1),
                stride_tensor,
                target_bboxes,
                pred_kpts,
            )

        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain

        return loss * batch_size, loss.detach()  # loss(box, pose, kobj, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def _select_target_keypoints(
        self,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        target_gt_idx: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Select target keypoints for each anchor based on batch index and target ground truth index.

        Args:
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).

        Returns:
            (torch.Tensor): Selected keypoints tensor, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        return selected_keypoints

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        # Select target keypoints using helper method
        selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)

        # Divide coordinates by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            target_bboxes /= stride_tensor
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class PoseLoss26(v8PoseLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation with RLE loss support."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):  # model must be de-paralleled
        """Initialize PoseLoss26 with model parameters and keypoint-specific loss functions including RLE loss."""
        super().__init__(model, tal_topk, tal_topk2)
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        self.rle_loss = None
        self.flow_model = model.model[-1].flow_model if hasattr(model.model[-1], "flow_model") else None
        if self.flow_model is not None:
            self.rle_loss = RLELoss(use_target_weight=True).to(self.device)
            self.target_weights = (
                torch.from_numpy(RLE_WEIGHT).to(self.device) if is_pose else torch.ones(nkpt, device=self.device)
            )

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the total loss and detach it for pose estimation."""
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
        loss = torch.zeros(
            6 if self.rle_loss else 5, device=self.device
        )  # box, kpt_location, kpt_visibility, cls, dfl[, rle]
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, batch)
        )
        # NOTE: re-assign index for consistency for now. Need to be removed in the future.
        loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

        batch_size = pred_kpts.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype) * self.stride[0]

        pred_kpts = pred_kpts.view(batch_size, -1, *self.kpt_shape)  # (b, h*w, 17, 3)

        if self.rle_loss and preds.get("kpts_sigma", None) is not None:
            pred_sigma = preds["kpts_sigma"].permute(0, 2, 1).contiguous()
            pred_sigma = pred_sigma.view(batch_size, -1, self.kpt_shape[0], 2)  # (b, h*w, 17, 2)
            pred_kpts = torch.cat([pred_kpts, pred_sigma], dim=-1)  # (b, h*w, 17, 5)

        pred_kpts = self.kpts_decode(anchor_points, pred_kpts)

        # Keypoint loss
        if fg_mask.sum():
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            keypoints_loss = self.calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch["batch_idx"].view(-1, 1),
                stride_tensor,
                target_bboxes,
                pred_kpts,
            )
            loss[1] = keypoints_loss[0]
            loss[2] = keypoints_loss[1]
            if self.rle_loss is not None:
                loss[5] = keypoints_loss[2]

        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        if self.rle_loss is not None:
            loss[5] *= self.hyp.rle  # rle gain

        return loss * batch_size, loss.detach()  # loss(box, kpt_location, kpt_visibility, cls, dfl[, rle])

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., 0] += anchor_points[:, [0]]
        y[..., 1] += anchor_points[:, [1]]
        return y

    def calculate_rle_loss(self, pred_kpt: torch.Tensor, gt_kpt: torch.Tensor, kpt_mask: torch.Tensor) -> torch.Tensor:
        """Calculate the RLE (Residual Log-likelihood Estimation) loss for keypoints.

        Args:
            pred_kpt (torch.Tensor): Predicted kpts with sigma, shape (N, num_keypoints, kpts_dim) where kpts_dim >= 4.
            gt_kpt (torch.Tensor): Ground truth keypoints, shape (N, num_keypoints, kpts_dim).
            kpt_mask (torch.Tensor): Mask for valid keypoints, shape (N, num_keypoints).

        Returns:
            (torch.Tensor): The RLE loss.
        """
        pred_kpt_visible = pred_kpt[kpt_mask]
        gt_kpt_visible = gt_kpt[kpt_mask]
        pred_coords = pred_kpt_visible[:, 0:2]
        pred_sigma = pred_kpt_visible[:, -2:]
        gt_coords = gt_kpt_visible[:, 0:2]

        target_weights = self.target_weights.unsqueeze(0).repeat(kpt_mask.shape[0], 1)
        target_weights = target_weights[kpt_mask]

        pred_sigma = pred_sigma.sigmoid()
        error = (pred_coords - gt_coords) / (pred_sigma + 1e-9)

        # Filter out NaN and Inf values to prevent MultivariateNormal validation errors
        valid_mask = ~(torch.isnan(error) | torch.isinf(error)).any(dim=-1)
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred_kpt.device)

        error = error[valid_mask]
        error = error.clamp(-100, 100)  # Prevent numerical instability
        pred_sigma = pred_sigma[valid_mask]
        target_weights = target_weights[valid_mask]

        log_phi = self.flow_model.log_prob(error)

        return self.rle_loss(pred_sigma, log_phi, error, target_weights)

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
            rle_loss (torch.Tensor): The RLE loss.
        """
        # Select target keypoints using inherited helper method
        selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)

        # Divide coordinates by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0
        rle_loss = 0

        if masks.any():
            target_bboxes /= stride_tensor
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if self.rle_loss is not None and (pred_kpt.shape[-1] == 4 or pred_kpt.shape[-1] == 5):
                rle_loss = self.calculate_rle_loss(pred_kpt, gt_kpt, kpt_mask)
                rle_loss = rle_loss.clamp(min=0)
            if pred_kpt.shape[-1] == 3 or pred_kpt.shape[-1] == 5:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss, rle_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses for classification."""

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the classification loss between predictions and true labels."""
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        return loss, loss.detach()


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model, tal_topk=10, tal_topk2: int | None = None):
        """Initialize v8OBBLoss with model, assigner, and rotated bbox loss; model must be de-paralleled."""
        super().__init__(model, tal_topk=tal_topk)
        self.assigner = RotatedTaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets for oriented bounding box detection."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the loss for oriented bounding box detection."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl, angle
        pred_distri, pred_scores, pred_angle = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
            preds["angle"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        batch_size = pred_angle.shape[0]  # batch size

        dtype = pred_scores.dtype
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * float(imgsz[1]), targets[:, 5] * float(imgsz[0])
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo26n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )
            weight = target_scores.sum(-1)[fg_mask]
            loss[3] = self.calculate_angle_loss(
                pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum
            )  # angle loss
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        loss[3] *= self.hyp.angle  # angle gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl, angle)

    def bbox_decode(
        self, anchor_points: torch.Tensor, pred_dist: torch.Tensor, pred_angle: torch.Tensor
    ) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)

    def calculate_angle_loss(self, pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum, lambda_val=3):
        """Calculate oriented angle loss.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes with shape [N, 5] (x, y, w, h, theta).
            target_bboxes (torch.Tensor): Target bounding boxes with shape [N, 5] (x, y, w, h, theta).
            fg_mask (torch.Tensor): Foreground mask indicating valid predictions.
            weight (torch.Tensor): Loss weights for each prediction.
            target_scores_sum (torch.Tensor): Sum of target scores for normalization.
            lambda_val (int): Controls the sensitivity to aspect ratio.

        Returns:
            (torch.Tensor): The calculated angle loss.
        """
        w_gt = target_bboxes[..., 2]
        h_gt = target_bboxes[..., 3]
        pred_theta = pred_bboxes[..., 4]
        target_theta = target_bboxes[..., 4]

        log_ar = torch.log((w_gt + 1e-9) / (h_gt + 1e-9))
        scale_weight = torch.exp(-(log_ar**2) / (lambda_val**2))

        delta_theta = pred_theta - target_theta
        delta_theta_wrapped = delta_theta - torch.round(delta_theta / math.pi) * math.pi
        ang_loss = torch.sin(2 * delta_theta_wrapped[fg_mask]) ** 2

        ang_loss = scale_weight[fg_mask] * ang_loss
        ang_loss = ang_loss * weight

        return ang_loss.sum() / target_scores_sum


class E2EDetectLoss:
    """Criterion class for computing training losses for end-to-end detection."""

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]


class E2ELoss:
    """Criterion class for computing training losses for end-to-end detection."""

    def __init__(self, model, loss_fn=v8DetectionLoss):
        """Initialize E2ELoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = loss_fn(model, tal_topk=10)
        self.one2one = loss_fn(model, tal_topk=7, tal_topk2=1)
        self.updates = 0
        self.total = 1.0
        # init gain
        self.o2m = 0.8
        self.o2o = self.total - self.o2m
        self.o2m_copy = self.o2m
        # final gain
        self.final_o2m = 0.1

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = self.one2many.parse_output(preds)
        one2many, one2one = preds["one2many"], preds["one2one"]
        loss_one2many = self.one2many.loss(one2many, batch)
        loss_one2one = self.one2one.loss(one2one, batch)
        return loss_one2many[0] * self.o2m + loss_one2one[0] * self.o2o, loss_one2one[1]

    def update(self) -> None:
        """Update the weights for one-to-many and one-to-one losses based on the decay schedule."""
        self.updates += 1
        self.o2m = self.decay(self.updates)
        self.o2o = max(self.total - self.o2m, 0)

    def decay(self, x) -> float:
        """Calculate the decayed weight for one-to-many loss based on the current update step."""
        return max(1 - x / max(self.one2one.hyp.epochs - 1, 1), 0) * (self.o2m_copy - self.final_o2m) + self.final_o2m


class TVPDetectLoss:
    """Criterion class for computing training losses for text-visual prompt detection."""

    def __init__(self, model, tal_topk=10, tal_topk2: int | None = None):
        """Initialize TVPDetectLoss with task-prompt and visual-prompt criteria using the provided model."""
        self.vp_criterion = v8DetectionLoss(model, tal_topk, tal_topk2)
        # NOTE: store following info as it's changeable in __call__
        self.hyp = self.vp_criterion.hyp
        self.ori_nc = self.vp_criterion.nc
        self.ori_no = self.vp_criterion.no
        self.ori_reg_max = self.vp_criterion.reg_max

    def parse_output(self, preds) -> dict[str, torch.Tensor]:
        """Parse model predictions to extract features."""
        return self.vp_criterion.parse_output(preds)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt detection."""
        return self.loss(self.parse_output(preds), batch)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt detection."""
        if self.ori_nc == preds["scores"].shape[1]:
            loss = torch.zeros(3, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        preds["scores"] = self._get_vp_features(preds)
        vp_loss = self.vp_criterion(preds, batch)
        box_loss = vp_loss[0][1]
        return box_loss, vp_loss[1]

    def _get_vp_features(self, preds: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """Extract visual-prompt features from the model output."""
        scores = preds["scores"]
        vnc = scores.shape[1]

        self.vp_criterion.nc = vnc
        self.vp_criterion.no = vnc + self.vp_criterion.reg_max * 4
        self.vp_criterion.assigner.num_classes = vnc
        return scores


class TVPSegmentLoss(TVPDetectLoss):
    """Criterion class for computing training losses for text-visual prompt segmentation."""

    def __init__(self, model, tal_topk=10):
        """Initialize TVPSegmentLoss with task-prompt and visual-prompt criteria using the provided model."""
        super().__init__(model)
        self.vp_criterion = v8SegmentationLoss(model, tal_topk)
        self.hyp = self.vp_criterion.hyp

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt segmentation."""
        return self.loss(self.parse_output(preds), batch)

    def loss(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt segmentation."""
        if self.ori_nc == preds["scores"].shape[1]:
            loss = torch.zeros(4, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        preds["scores"] = self._get_vp_features(preds)
        vp_loss = self.vp_criterion(preds, batch)
        cls_loss = vp_loss[0][2]
        return cls_loss, vp_loss[1]
