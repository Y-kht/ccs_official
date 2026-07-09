#!/usr/bin/env python3
"""CCS OOD Detection Evaluation.

Tests whether the learned covering set can distinguish ID from OOD:
  - For each test point, generate K positives and K negatives
    (augmentations / batch instances, or label-based sampling)
  - Compute metric distances under the learned CCS score
  - OOD score = -(mean negative distance) / (ID-calibrated threshold);
    ID anchors push negatives far away, OOD anchors do not

Compares against standard OOD baselines (MSP, Energy, Mahalanobis, KNN)
and SOTA methods (CombOOD, D-KNN, CIDER, ViM, NAC) on supported ID datasets.

Supported ID datasets: CIFAR-100 (repvgg_a2), STL10 (resnet18_simclr),
and ImageNet-100 (resnet18), with OOD sets SVHN, CIFAR-10, CIFAR-100,
NINCO, and DTD (textures).

Note: ViM and NAC require a supervised classification head and are
automatically skipped for self-supervised backbones (e.g. resnet18_simclr).

Usage:
    python tables_ood.py --config configs/config_ood_c_rep.json --note NOTE
"""
import os
# Optionally pin/reorder GPUs before importing torch, e.g.:
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import argparse
import json
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from types import SimpleNamespace
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from utils import (
    RepVGGBlock, _RepVGGA2Features, get_feat_dim, get_transforms, get_base_model,
    compute_t_hard,
    CCSModel, get_model_name, get_run_tag, set_seed,
    LabelSampler, generate_samples
)

from ood_baselines_utils import combood, d_knn, cider, vim, nac

# =========================================================
# Parallelisation
# =========================================================

def maybe_dataparallel(model):
    """Wrap model in DataParallel if multiple GPUs are available."""
    if torch.cuda.device_count() > 1:
        print(f"  Using DataParallel on {torch.cuda.device_count()} GPUs")
        return nn.DataParallel(model)
    return model

# =========================================================
# RepVGGA2
# =========================================================
class RepVGGA2(nn.Module):
    """RepVGG-A2 (stages + global avg pool + optional linear head). Output: 1408-d or 100-d logits."""
    NUM_BLOCKS = [2, 4, 14, 1]
    WIDTH_MULT = [1.5, 1.5, 1.5, 2.75]
    PRETRAINED_URL = ('https://github.com/chenyaofo/pytorch-cifar-models/'
                      'releases/download/repvgg/cifar100_repvgg_a2-8e71b1f8.pt')

    def __init__(self, pretrained=False, keep_head=False, num_classes=100):
        super().__init__()
        self.keep_head = keep_head
        self.in_planes = min(64, int(64 * self.WIDTH_MULT[0]))
        self.stage0 = RepVGGBlock(3, self.in_planes)
        self.stage1 = self._stage(int(64 * self.WIDTH_MULT[0]),  self.NUM_BLOCKS[0], 1)
        self.stage2 = self._stage(int(128 * self.WIDTH_MULT[1]), self.NUM_BLOCKS[1], 2)
        self.stage3 = self._stage(int(256 * self.WIDTH_MULT[2]), self.NUM_BLOCKS[2], 2)
        self.stage4 = self._stage(int(512 * self.WIDTH_MULT[3]), self.NUM_BLOCKS[3], 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Add the classification head if requested
        if self.keep_head:
            self.linear = nn.Linear(int(512 * self.WIDTH_MULT[3]), num_classes)

        if pretrained:
            self._load_pretrained()

    def _stage(self, planes, n, stride):
        blocks = []
        for s in [stride] + [1] * (n - 1):
            blocks.append(RepVGGBlock(self.in_planes, planes, stride=s))
            self.in_planes = planes
        return nn.Sequential(*blocks)

    def _load_pretrained(self):
        from torch.hub import load_state_dict_from_url
        sd = load_state_dict_from_url(self.PRETRAINED_URL, progress=True)
        
        if not self.keep_head:
            # drop fc / linear keys (we only use the feature extractor)
            sd = {k: v for k, v in sd.items()
                  if not k.startswith(('fc.', 'linear.', 'flatten.', 'gap.'))}
            
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if missing:
            raise RuntimeError(f"RepVGG-A2 pretrained weights missing keys: {missing[:5]}...")
        if unexpected and not self.keep_head:
            print(f"  Note: {len(unexpected)} unused keys in checkpoint (fc/linear head)")

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        out = self.gap(x).flatten(1)   # (B, 1408)
        
        # Apply the head if kept
        if self.keep_head:
            out = self.linear(out)
            
        return out

# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='CCS OOD Detection')
    parser.add_argument('--config', type=str, default='config_ood.json')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Override seeds from config')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Override save_dir from config')
    parser.add_argument('--note', type=str, help="Notes for the file names")
                        
    return parser.parse_args()


# =============================================================================
# OOD DATASETS
# =============================================================================

def _ensure_downloaded(dataset_cls, root):
    """Trigger pytorch-ood download without keeping the dataset object."""
    try:
        dataset_cls(root, download=True)
    except Exception:
        pass  # Already downloaded or extraction done


def get_ood_loader(name, batch_size, num_workers, transform,
                   imagenet100_root=None):
    """Load an OOD test dataset (auto-downloads if needed).

    For NINCO, uses pytorch-ood for download then loads via ImageFolder
    to preserve real class labels (needed for label-based sampling).
    Other datasets use their native torchvision loaders.

    Supported: svhn, cifar10, cifar100, dtd/textures, imagenet100, ninco
    """
    from torchvision.datasets import ImageFolder
    root = './data'

    if name == 'svhn':
        ds = datasets.SVHN(root, split='test', download=True, transform=transform)
    elif name == 'cifar10':
        ds = datasets.CIFAR10(root, train=False, download=True, transform=transform)
    elif name == 'cifar100':
        ds = datasets.CIFAR100(root, train=False, download=True, transform=transform)
    elif name in ('dtd', 'textures'):
        ds = datasets.DTD(root, split='test', download=True, transform=transform)
    elif name == 'imagenet100':
        imgnet_root = imagenet100_root or 'path/to/imagenet-100'  # set 'imagenet100_root' in the config
        val_path = os.path.join(imgnet_root, 'val.X')
        if not os.path.isdir(val_path):
            raise FileNotFoundError(
                f"ImageNet-100 val directory not found at {val_path}. "
                f"Set 'imagenet100_root' in config or use --train_dir/--val_dir.")
        ds = ImageFolder(val_path, transform=transform)
        print(f"  ImageNet-100 (OOD): {len(ds)} images, {len(ds.classes)} classes")
    elif name == 'ninco':
        from pytorch_ood.dataset.img import NINCO
        _ensure_downloaded(NINCO, root)
        ninco_path = os.path.join(root, 'NINCO', 'NINCO_OOD_classes')
        if not os.path.isdir(ninco_path):
            raise FileNotFoundError(
                f"NINCO class directory not found at {ninco_path}. "
                f"Check pytorch-ood extraction path.")
        ds = ImageFolder(ninco_path, transform=transform)
        print(f"  NINCO: {len(ds)} images, {len(ds.classes)} classes "
              f"(via ImageFolder)")
    else:
        raise ValueError(f"Unknown OOD dataset: {name}. "
                         f"Supported: svhn, cifar10, cifar100, dtd, textures, "
                         f"imagenet100, ninco")
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers)

# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

@torch.no_grad()
def extract_features(loader, backbone, device):
    """Extract features and labels using a frozen backbone."""
    backbone.eval()
    feats, labs = [], []
    for imgs, labels in loader:
        z = backbone(imgs.to(device).half()).view(imgs.size(0), -1).float()
        feats.append(z.cpu())
        labs.append(labels)
    return torch.cat(feats), torch.cat(labs)


@torch.no_grad()
def extract_augmented_features(loader, backbone, K, sampling, device):
    """Extract backbone features for anchors, K positives, and K negatives.

    Generates augmentations/samples ONCE via generate_samples() (shared with
    conformal_evaluate in utils.py) and runs them through the frozen backbone.
    Returns CPU tensors that can be reused across CCS methods.

    Returns:
        z_a:   (N, D)    anchor features
        z_pos: (N, K, D) positive features
        z_neg: (N, K, D) negative features
    """
    backbone.eval()
    sampler = LabelSampler(loader.dataset) if sampling == 'label' else None

    all_za, all_zp, all_zn = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        N = imgs.shape[0]

        # Extract backbone features
        z_a = backbone(imgs.half()).view(N, -1).float()
        pos = generate_samples(imgs, labels, K, sampler, positive=True).to(device)
        C, H, W = pos.shape[2:]
        z_p = backbone(pos.view(-1, C, H, W).half()).view(N, K, -1).float()
        del pos; torch.cuda.empty_cache()
        neg = generate_samples(imgs, labels, K, sampler, positive=False).to(device)
        z_n = backbone(neg.view(-1, C, H, W).half()).view(N, K, -1).float()
        del neg; torch.cuda.empty_cache()
        del imgs; torch.cuda.empty_cache()
        all_za.append(z_a.cpu())
        all_zp.append(z_p.cpu())
        all_zn.append(z_n.cpu())

    return torch.cat(all_za), torch.cat(all_zp), torch.cat(all_zn)


def build_encoder(model_name, input_size, pretrained=True, keep_head=False):
    """Return (features_module, feat_dim) or (full_model, feat_dim) for the chosen backbone."""
    if model_name == 'repvgg_a2':
        if keep_head:
            # Replace `RepVGGA2` with your actual full model class constructor if named differently
            model = RepVGGA2(pretrained=pretrained, keep_head=keep_head)
            return model, 1408
        else:
            features = _RepVGGA2Features(pretrained=pretrained)
            return features, 1408
        
    elif model_name == 'resnet18_simclr':
        # SimCLR-pretrained ResNet18 for STL10
        from torchvision.models import resnet18
        backbone = resnet18(weights=None)
        if pretrained:
            # Path to a SimCLR-pretrained ResNet18 checkpoint for STL10;
            # set the SIMCLR_CKPT environment variable or edit the default.
            ckpt_path = os.environ.get('SIMCLR_CKPT',
                                       'path/to/ResNet_simclr_STL10.ckpt')
            checkpoint = torch.load(ckpt_path, map_location='cpu')

            for key in ('model_state_dict', 'state_dict', 'model', 'encoder'):
                if key in checkpoint:
                    checkpoint = checkpoint[key]
                    break

            skip_prefixes = ('projection', 'head', 'fc.', 'projector', 'classifier', 'linear')
            state = {k: v for k, v in checkpoint.items() if not k.startswith(skip_prefixes)}

            backbone_keys = set(backbone.state_dict().keys())
            if state and not (set(state.keys()) & backbone_keys):
                sample_key = next(iter(state))
                for prefix in ['convnet.', 'encoder.', 'backbone.', 'model.', 
                               'feature_extractor.', 'resnet.', 'net.', 
                               'base_encoder.', 'f.']:
                    if sample_key.startswith(prefix):
                        state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
                        print(f"  SimCLR: stripped key prefix '{prefix}'")
                        break

            state = {k: v for k, v in state.items()
                     if k in backbone_keys and state[k].shape == backbone.state_dict()[k].shape}

            missing, unexpected = backbone.load_state_dict(state, strict=False)
            n_loaded = len(backbone.state_dict()) - len(missing)
            print(f"  SimCLR: loaded {n_loaded}/{len(backbone.state_dict())} params "
                  f"({len(missing)} missing, {len(unexpected)} unexpected)")
            if n_loaded == 0:
                raise RuntimeError(f"SimCLR checkpoint loaded 0 parameters! Keys: {list(checkpoint.keys())[:5]}")
                
        if keep_head:
            return backbone, 512
        else:
            # FIX: Add nn.Flatten(1) to the sequential
            features = nn.Sequential(
                *list(backbone.children())[:-1],
                nn.Flatten(1)
            )
            return features, 512
            
    else:  # resnet18 (ImageNet pretrained)
        from torchvision.models import resnet18, ResNet18_Weights
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        if input_size == 32:
            backbone.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            backbone.maxpool = nn.Identity()
            
        if keep_head:
            return backbone, 512
        else:
            # FIX: Add nn.Flatten(1) to the sequential
            features = nn.Sequential(
                *list(backbone.children())[:-1],
                nn.Flatten(1)
            )
            return features, 512
        

# =============================================================================
# LINEAR PROBE (for MSP / Energy)
# =============================================================================

def train_linear_probe(z_train, y_train, n_classes=100, epochs=20, device='cpu'):
    """Train a linear classifier on frozen features."""
    D = z_train.shape[1]
    probe = nn.Linear(D, n_classes).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    ds = TensorDataset(z_train, y_train)
    loader = DataLoader(ds, batch_size=512, shuffle=True)

    probe.train()
    for ep in range(epochs):
        correct, total = 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = probe(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
    print(f"  Linear probe accuracy: {correct/total:.1%}")
    probe.eval()
    return probe


# =============================================================================
# BASELINE OOD SCORING
# All return numpy arrays where HIGHER = more OOD.
# =============================================================================

@torch.no_grad()
def score_msp(probe, z, device='cpu', batch_size=512):
    """MSP: score = 1 - max softmax probability."""
    scores = []
    for i in range(0, len(z), batch_size):
        logits = probe(z[i:i+batch_size].to(device))
        scores.append(1.0 - logits.softmax(-1).max(-1).values.cpu())
    return torch.cat(scores).numpy()


@torch.no_grad()
def score_energy(probe, z, T=1.0, device='cpu', batch_size=512):
    """Energy: score = -T * logsumexp(logits/T). More negative = more ID."""
    scores = []
    for i in range(0, len(z), batch_size):
        logits = probe(z[i:i+batch_size].to(device))
        energy = -T * torch.logsumexp(logits / T, dim=-1)
        scores.append(energy.cpu())
    return torch.cat(scores).numpy()


def fit_mahalanobis(z, y, n_classes=100):
    """Fit class-conditional means + shared precision matrix."""
    class_means, all_diffs = [], []
    for c in range(n_classes):
        mask = (y == c)
        if mask.sum() == 0:
            class_means.append(torch.zeros(z.shape[1]))
            continue
        mu = z[mask].mean(0)
        class_means.append(mu)
        all_diffs.append(z[mask] - mu)
    all_diffs = torch.cat(all_diffs)
    cov = (all_diffs.T @ all_diffs) / len(all_diffs)
    precision = torch.linalg.inv(cov + 1e-4 * torch.eye(z.shape[1]))
    return class_means, precision


@torch.no_grad()
def score_mahalanobis(z, class_means, precision, device='cpu', batch_size=512):
    """Mahalanobis: score = min_c (z-mu_c)^T Sigma^{-1} (z-mu_c)."""
    prec = precision.to(device)
    means = [mu.to(device) for mu in class_means]
    scores = []
    for i in range(0, len(z), batch_size):
        zb = z[i:i+batch_size].to(device)
        best = torch.full((zb.shape[0],), float('inf'), device=device)
        for mu in means:
            diff = zb - mu
            maha = (diff @ prec * diff).sum(-1)
            best = torch.min(best, maha)
        scores.append(best.cpu())
    return torch.cat(scores).numpy()


@torch.no_grad()
def score_knn_l2(z, z_ref, k=50, batch_size=200):
    """KNN (L2): score = k-th nearest Euclidean distance."""
    scores = []
    for i in range(0, len(z), batch_size):
        dists = torch.cdist(z[i:i+batch_size], z_ref)
        kth = dists.kthvalue(min(k, dists.shape[1]), dim=1).values
        scores.append(kth)
    return torch.cat(scores).numpy()


# =============================================================================
# CCS COVERING-SET OOD SCORING (from pre-extracted features)
# =============================================================================

@torch.no_grad()
def _apply_projector(model, z, device, batch_size=1024):
    """Apply the InfoNCE projector to pre-extracted backbone features.
    z can be (N, D) or (N, K, D). Returns same shape, on CPU."""
    base = get_base_model(model)
    if not base.add_infonce:
        return z
    projector = base.projector.to(device)
    shape = z.shape
    z_flat = z.reshape(-1, shape[-1])  # (N*K, D) or (N, D)
    out = []
    for i in range(0, len(z_flat), batch_size):
        out.append(projector(z_flat[i:i+batch_size].to(device)).cpu())
    return torch.cat(out).reshape(shape)


@torch.no_grad()
def calibrate_from_features(model, z_a, z_pos, alpha, device, batch_size=512):
    """Calibrate covering-set threshold from pre-extracted backbone features.

    Args:
        model:  loaded CCSModel (for projector + metric params)
        z_a:    (N, D) anchor backbone features (CPU)
        z_pos:  (N, K, D) positive backbone features (CPU)
        alpha:  miscoverage rate
    Returns:
        threshold (float)
    """
    base = get_base_model(model)
    m, p = base.get_params()

    # Apply projector if needed
    z_a = _apply_projector(model, z_a, device)
    z_pos = _apply_projector(model, z_pos, device)

    all_dists = []
    N = z_a.shape[0]
    for i in range(0, N, batch_size):
        za_b = z_a[i:i+batch_size].to(device)
        zp_b = z_pos[i:i+batch_size].to(device)
        dist_p = base.compute_distance(zp_b, za_b, m, p)  # (B, K)
        all_dists.append(dist_p.cpu())

    all_dists = torch.cat(all_dists, dim=0).reshape(-1)
    threshold = compute_t_hard(all_dists, alpha).item()
    print(f"  Calibrated threshold: t={threshold:.4f} "
          f"(from {all_dists.numel()} positive distances)")
    return threshold


@torch.no_grad()
def score_from_features(model, z_a, z_pos, z_neg, threshold, device,
                        batch_size=512):
    """Compute covering-set OOD scores from pre-extracted backbone features.

    OOD score = -(mean negative distance) / threshold. For ID anchors the
    learned metric pushes negatives far beyond the calibrated threshold
    (very negative score); for OOD anchors it does not (higher score).

    Args:
        model:      loaded CCSModel
        z_a:        (N, D) anchor backbone features (CPU)
        z_pos:      (N, K, D) positive backbone features (CPU)
        z_neg:      (N, K, D) negative backbone features (CPU)
        threshold:  calibrated threshold t
    Returns:
        scores: numpy array (N,), HIGHER = more OOD
    """
    base = get_base_model(model)
    m, p = base.get_params()

    # Apply projector if needed (z_pos is kept in the signature for API
    # symmetry with calibrate_from_features but is not used by this score)
    z_a = _apply_projector(model, z_a, device)
    z_neg = _apply_projector(model, z_neg, device)

    scores = []
    N = z_a.shape[0]
    for i in range(0, N, batch_size):
        za_b = z_a[i:i+batch_size].to(device)
        zn_b = z_neg[i:i+batch_size].to(device)

        dist_n = base.compute_distance(zn_b, za_b, m, p)  # (B, K)

        ood_score = -(dist_n.mean(dim=1) / threshold)
        scores.append(ood_score.cpu())

    scores = torch.cat(scores).numpy()
    print(f"  Mean score: {scores.mean():.4f}")
    return scores


# =============================================================================
# METRICS
# =============================================================================

def auroc(id_scores, ood_scores):
    """AUROC via Wilcoxon rank-sum (no sklearn needed)."""
    n0, n1 = len(id_scores), len(ood_scores)
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.zeros(n0), np.ones(n1)])
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Handle ties
    sorted_s = scores[order]
    i = 0
    while i < len(sorted_s):
        j = i
        while j < len(sorted_s) and sorted_s[j] == sorted_s[i]:
            j += 1
        avg_rank = ranks[order[i:j]].mean()
        ranks[order[i:j]] = avg_rank
        i = j
    return (ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)


def fpr_at_tpr(id_scores, ood_scores, tpr_target=0.95):
    """FPR when TPR = tpr_target (OOD = positive class)."""
    threshold = np.percentile(ood_scores, 100 * (1 - tpr_target))
    return (id_scores >= threshold).mean()


def compute_ood_metrics(id_scores, ood_scores):
    """Return AUROC (%) and FPR@95 (%)."""
    return {
        'auroc': auroc(id_scores, ood_scores) * 100,
        'fpr95': fpr_at_tpr(id_scores, ood_scores) * 100,
    }


# =============================================================================
# CCS CHECKPOINT LOADING
# =============================================================================

def build_ccs_args(config, method, seed, save_dir):
    """Build a minimal args namespace to locate a CCS checkpoint."""
    args = SimpleNamespace()
    args.dataset = config['id_dataset']
    args.model = config['model']
    args.embed_dim = get_feat_dim(config['model'])
    args.seed = seed
    args.scale = config.get('scale', 7.0)
    args.save_dir = save_dir
    args.n_norms = config["n_norms"]
    args.sampling = config.get("sampling", "augmentation")

    for k, v in method.items():
        if k not in ('name', 'type'):
            setattr(args, k, v)

    defaults = {
        'alpha': 0.05, 'k': 10, 'r': 0.5, 'sampling': 'augmentation',
        'epochs': 50, 'lr': 0.01, 'model_lr': 0.001,
        'add_infonce': False, 'alternating_training': False,
        'contrastive_weight': 1.0, 'volume_weight': 0.1,
        'nce_temperature': 0.1, 'delta': 1.0,
        'n_norms': 1, 'problem': 'vol', 'scale': 4.0, 'lam': None,
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    return args


def load_ccs_model(config, method, seed, save_dir, device, note):
    """Instantiate CCSModel and load a trained checkpoint."""
    args = build_ccs_args(config, method, seed, save_dir)
    args.note = note
    if args.dataset == 'cifar100':
        input_size = 32
    elif args.dataset == 'stl10':
        input_size = 96
    else:
        input_size = 224
    model = CCSModel(
        metric=args.metric,
        add_infonce=args.add_infonce,
        n_norms=args.n_norms,
        input_size=input_size,
        delta=args.delta,
        model_name=args.model,
    ).to(device)

    # Locate checkpoint
    ckpt_path = method.get('checkpoint')
    if ckpt_path is None:
        tag = get_run_tag(args)
        model_name = get_model_name(args)
        ckpt_path = os.path.join("results", 'checkpoints', tag,
                                 f'{model_name}.pt')

    if not os.path.exists(ckpt_path):
        # Checkpoints are produced by tables_coverage.py; run it first with a
        # matching config and --note so the file names line up.
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    base = get_base_model(model)
    base.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"  Loaded checkpoint: {ckpt_path}")
    return model, args


# =============================================================================
# LaTeX TABLE
# =============================================================================

def generate_latex(all_results, config, seeds, method_names, all_times):
    """Rows = methods, col-groups = OOD datasets (AUROC / FPR95 each)."""
    ood_datasets = config['ood_datasets']
    n_ood = len(ood_datasets)

    lines = []
    lines.append(r"\begin{table}[h!]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{OOD detection on {config['id_dataset'].upper()} (ID). "
        rf"Backbone: {config['model']}, "
        rf"seeds$={list(seeds)}$.}}")
    lines.append(r"\footnotesize")

    cols = "l" + "cc" * n_ood
    lines.append(rf"\begin{{tabular}}{{{cols}}}")
    lines.append(r"\toprule")

    # Header row 1
    hdr = r"\textbf{Method}"
    for ds in ood_datasets:
        hdr += rf" & \multicolumn{{2}}{{c}}{{\textbf{{{ds.upper()}}}}}"
    lines.append(hdr + r" \\")

    # Header row 2 (sub-columns)
    for i, _ in enumerate(ood_datasets):
        lo = 2 * i + 2
        hi = lo + 1
        lines.append(rf"\cmidrule(lr){{{lo}-{hi}}}")
    sub = ""
    for _ in ood_datasets:
        sub += r" & AUROC$\uparrow$ & FPR95$\downarrow$"
    sub += r" &"
    lines.append(sub + r" \\")
    lines.append(r"\midrule")
    newline = '\n'
    # Data rows
    for mname in method_names:
        ds_results = all_results[mname]
        row = mname
        for ds in ood_datasets:
            vals = ds_results.get(ds, {})
            if not vals.get('auroc'):
                row += rf" {newline}& -- {newline}& --"
                continue
            a = np.array(vals['auroc'])
            f = np.array(vals['fpr95'])
            row += (rf" {newline}& ${a.mean():.1f} \pm {a.std():.1f}$"
                    rf" {newline}& ${f.mean():.1f} \pm {f.std():.1f}$")
        lines.append(row + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    cli = parse_args()

    with open(cli.config) as f:
        config = json.load(f)

    seeds = cli.seeds or config.get('seeds', [42])
    save_dir = cli.save_dir or config.get('save_dir', './results')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    id_dataset = config['id_dataset']
    ood_names = config['ood_datasets']
    backbone_name = config['model']
    batch_size = config.get('batch_size', 256)
    num_workers = config.get('num_workers', 4)
    knn_k = config.get('knn_k', 50)
    methods = config['methods']
    method_names = [m['name'] for m in methods]
    sampling = config.get('sampling', 'augmentation')

    # Identify unique CCS (sampling, K) groups for shared feature extraction
    ccs_methods = [m for m in methods if m['type'] == 'ccs']
    ccs_groups = {}  # (sampling, K) -> list of method dicts
    for m in ccs_methods:
        key = (sampling, m['k'])
        ccs_groups.setdefault(key, []).append(m)
    has_ccs = len(ccs_methods) > 0

    print(f"ID: {id_dataset} | OOD: {ood_names} | Backbone: {backbone_name}")
    print(f"Seeds: {seeds} | Methods: {len(methods)} | Device: {device}")
    if has_ccs:
        groups_str = ", ".join(f"(sampling={s}, K={k})"
                               for s, k in ccs_groups.keys())
        print(f"CCS groups: {groups_str}")
    print(f"knn_k={knn_k}")
    print("=" * 70)

    # --- Data loaders ---
    if id_dataset == 'cifar100':
        input_size = 32
    elif id_dataset == 'stl10':
        input_size = 96
    else:
        input_size = 224
    tf_val = get_transforms(input_size, train=False, dataset=id_dataset,
                            model_name=backbone_name)

    if id_dataset == 'cifar100':
        id_train_ds = datasets.CIFAR100('./data', train=True,  download=True,
                                        transform=tf_val)
        id_test_ds  = datasets.CIFAR100('./data', train=False, download=True,
                                        transform=tf_val)
        n_classes = 100
    elif id_dataset == 'stl10':
        from torchvision.datasets import STL10
        id_train_ds = STL10('./data', split='train', download=True,
                            transform=tf_val)
        id_test_ds  = STL10('./data', split='test',  download=True,
                            transform=tf_val)
        n_classes = 10
    elif id_dataset == 'imagenet100':
        from torchvision.datasets import ImageFolder
        imgnet_root = config.get('imagenet100_root', 'path/to/imagenet-100')
        id_train_ds = ImageFolder(os.path.join(imgnet_root, 'train.X'),
                                  transform=tf_val)
        id_test_ds  = ImageFolder(os.path.join(imgnet_root, 'val.X'),
                                  transform=tf_val)
        n_classes = 100
    else:
        raise ValueError(f"Unsupported ID dataset: {id_dataset}")

    id_train_loader = DataLoader(id_train_ds, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers)
    id_test_loader  = DataLoader(id_test_ds, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers)

    # Split ID test into calibration (first half) and evaluation (second half)
    n_test = len(id_test_ds)
    n_cal = n_test // 2
    id_cal_ds  = torch.utils.data.Subset(id_test_ds, range(n_cal))
    id_eval_ds = torch.utils.data.Subset(id_test_ds, range(n_cal, n_test))
    id_cal_loader  = DataLoader(id_cal_ds, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers)
    id_eval_loader = DataLoader(id_eval_ds, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers)

    ood_loaders = {}
    imgnet_root = config.get('imagenet100_root', 'path/to/imagenet-100')
    for name in ood_names:
        ood_loaders[name] = get_ood_loader(name, batch_size, num_workers, tf_val,
                                           imagenet100_root=imgnet_root)

    # --- Extract plain features once for baselines ---
    has_baselines = any(m['type'] in ('msp', 'energy', 'mahalanobis', 'knn')
                        for m in methods)
    z_train = z_eval = y_train = y_eval = None
    z_ood = {}

    if has_baselines:
        print("\nExtracting features with frozen backbone (for baselines)...")
        backbone, feat_dim = build_encoder(backbone_name, input_size,
                                           pretrained=True)
        backbone = maybe_dataparallel(backbone.half().to(device)).eval()

        z_train, y_train = extract_features(id_train_loader, backbone, device)
        z_eval, y_eval   = extract_features(id_eval_loader, backbone, device)
        print(f"  ID train: {z_train.shape}  |  ID eval: {z_eval.shape}")

        for name in ood_names:
            z_ood[name], _ = extract_features(ood_loaders[name], backbone,
                                              device)
            print(f"  OOD {name}: {z_ood[name].shape}")

        del backbone
        torch.cuda.empty_cache()
    else:
        print("\nNo baseline methods — skipping plain feature extraction.")

    # --- Results container ---
    all_results = {m['name']: {ds: {'auroc': [], 'fpr95': []}
                               for ds in ood_names}
                   for m in methods}
    all_times = {m['name']: [] for m in methods}

    # === Seed loop ===
    for si, seed in enumerate(seeds):
        print(f"\n{'#'*70}")
        print(f"# SEED {seed} ({si+1}/{len(seeds)})")
        print(f"{'#'*70}")
        set_seed(seed)

        # Subsample reference set for KNN (if baselines present)
        z_ref = None
        if has_baselines and z_train is not None:
            ref_idx = torch.randperm(len(z_train))[:config.get('n_ref', 2000)]
            z_ref = z_train[ref_idx]

        # Shared baseline objects (lazy-init per seed)
        probe = None
        class_means, precision = None, None

        # --- Pre-extract augmented features ONCE per seed per (sampling, K) ---
        # ccs_feats[(sampling, K)] -> {loader_key: (z_a, z_pos, z_neg)}
        ccs_feats = {}
        ccs_extract_times = {}  # (sampling, K) -> seconds
        if has_ccs:
            bb, _ = build_encoder(backbone_name, input_size, pretrained=True)
            bb = maybe_dataparallel(bb.half().to(device)).eval()

            for (samp, K) in ccs_groups.keys():
                gkey = (samp, K)
                ccs_feats[gkey] = {}
                print(f"\n  Pre-extracting features "
                      f"(sampling={samp}, K={K})...")

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_extract_start = time.perf_counter()

                print("    ID cal set...")
                ccs_feats[gkey]['id_cal'] = extract_augmented_features(
                    id_cal_loader, bb, K, samp, device)
                print(f"      anchors: {ccs_feats[gkey]['id_cal'][0].shape}, "
                      f"pos: {ccs_feats[gkey]['id_cal'][1].shape}, "
                      f"neg: {ccs_feats[gkey]['id_cal'][2].shape}")

                print("    ID eval set...")
                ccs_feats[gkey]['id_eval'] = extract_augmented_features(
                    id_eval_loader, bb, K, samp, device)

                for ds in ood_names:
                    print(f"    OOD {ds}...")
                    ccs_feats[gkey][ds] = extract_augmented_features(
                        ood_loaders[ds], bb, K, samp, device)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                ccs_extract_times[gkey] = time.perf_counter() - t_extract_start
                print(f"  Extraction time for (sampling={samp}, K={K}): "
                      f"{ccs_extract_times[gkey]:.2f}s")

            del bb
            torch.cuda.empty_cache()
            print("  Feature extraction complete.\n")
        
        # --- Method loop ---
        sota_model = None
        m_K = 000
        m_sampling = "aug"
        for mi, method in enumerate(methods):
            mname = method['name']
            mtype = method['type']
            print(f"\n--- [{si+1}/{len(seeds)}] [{mi+1}/{len(methods)}] "
                  f"{mname} ---")

            try:
                # ========== BASELINE METHODS (feature-based) ==========
                if mtype == 'msp':
                    if probe is None:
                        print("  Training linear probe...")
                        probe = train_linear_probe(z_train, y_train,
                                                   n_classes, device=device)
                    t_start = time.perf_counter()
                    id_scores = score_msp(probe, z_eval, device)
                    ood_scorers = {ds: score_msp(probe, z_ood[ds], device)
                                   for ds in ood_names}

                elif mtype == 'energy':
                    if probe is None:
                        print("  Training linear probe...")
                        probe = train_linear_probe(z_train, y_train,
                                                   n_classes, device=device)
                    T = method.get('temperature', 1.0)
                    t_start = time.perf_counter()
                    id_scores = score_energy(probe, z_eval, T, device)
                    ood_scorers = {ds: score_energy(probe, z_ood[ds], T, device)
                                   for ds in ood_names}

                elif mtype == 'mahalanobis':
                    if class_means is None:
                        print("  Fitting class-conditional Gaussians...")
                        class_means, precision = fit_mahalanobis(
                            z_train, y_train, n_classes)
                    t_start = time.perf_counter()
                    id_scores = score_mahalanobis(z_eval, class_means,
                                                  precision, device)
                    ood_scorers = {
                        ds: score_mahalanobis(z_ood[ds], class_means,
                                              precision, device)
                        for ds in ood_names}

                elif mtype == 'knn':
                    t_start = time.perf_counter()
                    id_scores = score_knn_l2(z_eval, z_ref, knn_k)
                    ood_scorers = {ds: score_knn_l2(z_ood[ds], z_ref, knn_k)
                                   for ds in ood_names}
                    
                # ========== SOTA METHODS ==========
                elif mtype in ('combood', 'd_knn', 'cider', 'vim', 'nac'):
                    requires_head = mtype in ('vim', 'nac')

                    # ViM and NAC require a supervised classification head.
                    # Self-supervised backbones (e.g. SimCLR) don't have one,
                    # so skip these methods gracefully.
                    _no_head_backbones = ('resnet18_simclr',)
                    if requires_head and backbone_name in _no_head_backbones:
                        print(f"  SKIPPING {mname}: requires a supervised "
                              f"classification head, but {backbone_name} is "
                              f"self-supervised. Remove from config or use a "
                              f"supervised backbone.")
                        continue
                    
                    # Only load/reload if model is missing or has the wrong head configuration
                    if sota_model is None or getattr(sota_model, '_has_head', None) != requires_head:
                        print(f"  Loading model for SOTA methods (keep_head={requires_head})...")
                        sota_model, _ = build_encoder(backbone_name, input_size, pretrained=True, keep_head=requires_head)
                        sota_model = sota_model.to(device).eval()
                        sota_model._has_head = requires_head  # Tag it to avoid unnecessary reloads
                    
                    method['id_train_loader'] = id_train_loader
                    method['model_name'] = backbone_name
                    sota_func_map = {
                        'combood': combood, 
                        'd_knn': d_knn, 
                        'cider': cider, 
                        'vim': vim, 
                        'nac': nac
                    }
                    
                    ood_scorers = {}
                    t_start = time.perf_counter()
                    for ds in ood_names:
                        print(f"  Running {mname} on {ds}...")
                        id_sc, ood_sc = sota_func_map[mtype](sota_model, id_eval_loader, ood_loaders[ds], method)
                        id_scores = id_sc
                        ood_scorers[ds] = ood_sc

                # ========== CCS (from pre-extracted features) ==========
                elif mtype == 'ccs':
                    model, cargs = load_ccs_model(
                        config, method, seed, save_dir, device, cli.note)

                    # Read per-method params
                    m_sampling = sampling
                    m_K = method['k']
                    m_alpha = method.get('alpha', 0.05)
                    gkey = (m_sampling, m_K)

                    # Unpack pre-extracted features for this (sampling, K) group
                    z_cal_a, z_cal_pos, z_cal_neg = ccs_feats[gkey]['id_cal']
                    z_ev_a, z_ev_pos, z_ev_neg = ccs_feats[gkey]['id_eval']

                    # Step 1: Calibrate threshold (using this method's alpha)
                    t_start = time.perf_counter()
                    print(f"  Calibrating threshold (alpha={m_alpha})...")
                    threshold = calibrate_from_features(
                        model, z_cal_a, z_cal_pos, m_alpha, device)

                    # Step 2: Score ID eval
                    print("  Scoring ID eval...")
                    id_scores = score_from_features(
                        model, z_ev_a, z_ev_pos, z_ev_neg,
                        threshold, device)

                    # Step 3: Score each OOD dataset
                    ood_scorers = {}
                    for ds in ood_names:
                        z_ood_a, z_ood_pos, z_ood_neg = ccs_feats[gkey][ds]
                        print(f"  Scoring OOD {ds}...")
                        ood_scorers[ds] = score_from_features(
                            model, z_ood_a, z_ood_pos, z_ood_neg,
                            threshold, device)

                    # Diagnostic
                    print(f"  ID mean: {id_scores.mean():.4f} | OOD: "
                          + ", ".join(f"{ds}={ood_scorers[ds].mean():.4f}"
                                      for ds in ood_names))

                    del model
                    torch.cuda.empty_cache()

                else:
                    raise ValueError(f"Unknown method type: {mtype}")

                # ========== Record timing ==========
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_elapsed = time.perf_counter() - t_start

                # For CCS methods, include the shared sample-generation
                # and feature-extraction time for fair comparison.
                if mtype == 'ccs':
                    gkey = (m_sampling, method['k'])
                    t_extract = ccs_extract_times.get(gkey, 0.0)
                    t_elapsed += t_extract
                    print(f"  Method time: {t_elapsed - t_extract:.2f}s "
                          f"(+ {t_extract:.2f}s extraction = {t_elapsed:.2f}s)")
                else:
                    print(f"  Method time: {t_elapsed:.2f}s")

                all_times[mname].append(t_elapsed)

                # ========== Compute metrics per OOD dataset ==========
                for ds_name in ood_names:
                    ood_scores = ood_scorers[ds_name]
                    met = compute_ood_metrics(id_scores, ood_scores)
                    all_results[mname][ds_name]['auroc'].append(met['auroc'])
                    all_results[mname][ds_name]['fpr95'].append(met['fpr95'])
                    print(f"  {ds_name}: AUROC={met['auroc']:.1f}%  "
                          f"FPR95={met['fpr95']:.1f}%")

            except Exception as e:
                print(f"  !! FAILED: {e}")
                import traceback; traceback.print_exc()
                torch.cuda.empty_cache()

        # Free augmented features at end of seed
        del ccs_feats
        if sota_model is not None:
            del sota_model
        torch.cuda.empty_cache()

    # === Summary ===
    print(f"\n{'='*70}")
    print("AGGREGATED RESULTS (mean +/- std)")
    print(f"{'='*70}")
    for mname in method_names:
        print(f"\n  {mname}:")
        for ds in ood_names:
            vals = all_results[mname][ds]
            if not vals['auroc']:
                print(f"    {ds}: NO RUNS")
                continue
            a = np.array(vals['auroc'])
            f = np.array(vals['fpr95'])
            print(f"    {ds}: AUROC={a.mean():.1f}+/-{a.std():.1f}%  |  "
                  f"FPR95={f.mean():.1f}+/-{f.std():.1f}%")
        times = all_times.get(mname, [])
        if times:
            t = np.array(times)
            print(f"    Time: {t.mean():.2f}+/-{t.std():.2f}s")

    # === LaTeX ===
    latex = generate_latex(all_results, config, seeds, method_names, all_times)
    print(f"\n{'='*70}")
    print("LATEX TABLE")
    print(f"{'='*70}")
    print(latex)

    # === Save ===
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f'ood_results_{id_dataset}_{backbone_name}_{seeds}seeds_{m_K}k_{m_sampling}.json')
    with open(out_path, 'w') as f:
        json.dump({'seeds': seeds, 'config': config, 'results': all_results,
                   'times': all_times},
                  f, indent=2, default=str)

    tex_path = os.path.join(save_dir, f'ood_table_{id_dataset}_{backbone_name}_{seeds}seeds_{m_K}k_{m_sampling}.tex')
    with open(tex_path, 'w') as f:
        f.write(latex)

    print(f"\nSaved results: {out_path}")
    print(f"Saved LaTeX:   {tex_path}")


if __name__ == '__main__':
    main()