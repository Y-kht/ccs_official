#!/usr/bin/env python3
"""
CCS Simulation: 2D/3D cluster data with learned metrics.
Self-contained — no encoder, works directly on 2D/3D points.
"""
import math
import os
import random
import torch
import torch.nn as nn
import numpy as np
from scipy.spatial import ConvexHull
from utils import save_checkpoint, load_checkpoint

LOG2 = math.log(2.0)


# =============================================================================
# SEED
# =============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_clusters(n_per_cluster=500, d=2, n_clusters=5, separation=2.5, std=1.0,
                      seed=42, distribution='isotropic'):
    """
    Generate `n_clusters` labelled clusters in R^d.

    Parameters
    ----------
    n_per_cluster : int
        Samples per cluster.
    d : int
        Dimensionality (2 or 3).
    n_clusters : int
        Number of clusters / labels.
    separation : float
        Distance between adjacent cluster centres.
    std : float
        Noise scale.
    seed : int
        Random seed.
    distribution : str
        'isotropic'   — spherical Gaussians (baseline-friendly)
        'anisotropic' — random stretched covariance per cluster
        'banana'      — arc/crescent shapes per cluster
        'mixed'       — first half anisotropic, second half banana
        'rings'       — concentric rings/shells at different radii

    Returns
    -------
    z : Tensor (N, d)
    labels : Tensor (N,) long
    """
    set_seed(seed)

    # Cluster centres on a circle (using first 2 dims; 3rd dim offset for 3D)
    centres = []
    for i in range(n_clusters):
        angle = 2 * math.pi * i / n_clusters
        c = [separation * math.cos(angle), separation * math.sin(angle)]
        if d >= 3:
            c.append(separation * 0.3 * math.sin(2 * angle))  # z variation
        c += [0.0] * (d - len(c))
        centres.append(c)
    centres = torch.tensor(centres, dtype=torch.float32)

    all_z, all_y = [], []

    for i in range(n_clusters):
        # Decide distribution type for this cluster
        if distribution == 'mixed':
            # First half anisotropic, second half banana
            dist_type = 'anisotropic' if i < n_clusters // 2 else 'banana'
        else:
            dist_type = distribution

        if dist_type == 'isotropic':
            pts = centres[i] + std * torch.randn(n_per_cluster, d)

        elif dist_type == 'anisotropic':
            # Random covariance: A @ A^T with random elongation
            A = torch.randn(d, d) * 0.5
            # Stretch one random axis by 3-5x
            stretch_dim = i % d
            A[stretch_dim] *= 1.0 + torch.rand(1).item() * 2.0
            cov_sqrt = A  # not symmetric, but A @ x gives correlated samples
            pts = centres[i] + std * (torch.randn(n_per_cluster, d) @ cov_sqrt.T)

        elif dist_type == 'banana':
            # Arc/crescent: sample angle along a half-circle, add noise
            theta = torch.rand(n_per_cluster) * math.pi  # [0, π]
            radius = 1.0 + std * 0.3 * torch.randn(n_per_cluster)
            if d == 2:
                x = radius * torch.cos(theta)
                y = radius * torch.sin(theta)
                pts_local = torch.stack([x, y], dim=-1)
            else:  # d >= 3
                # Arc in a random 2D plane, with noise in 3rd dim
                x = radius * torch.cos(theta)
                y = radius * torch.sin(theta)
                z_coord = std * 0.5 * torch.randn(n_per_cluster)
                pts_local = torch.stack([x, y, z_coord], dim=-1)
                if d > 3:
                    pts_local = torch.cat([pts_local,
                                           std * 0.3 * torch.randn(n_per_cluster, d - 3)], dim=-1)
            # Random rotation per cluster so bananas point different directions
            rot = torch.linalg.qr(torch.randn(d, d))[0]  # random orthogonal
            pts = centres[i] + (pts_local @ rot.T) + std * 0.2 * torch.randn(n_per_cluster, d)

        elif dist_type == 'rings':
            # Points on a ring/shell at a cluster-specific radius
            ring_radius = 1.0 + 0.5 * i
            if d == 2:
                theta = torch.rand(n_per_cluster) * 2 * math.pi
                x = ring_radius * torch.cos(theta)
                y = ring_radius * torch.sin(theta)
                pts_local = torch.stack([x, y], dim=-1)
            else:  # d >= 3
                # Points on a sphere
                raw = torch.randn(n_per_cluster, d)
                pts_local = ring_radius * raw / raw.norm(dim=-1, keepdim=True)
            pts = centres[i] + pts_local + std * 0.3 * torch.randn(n_per_cluster, d)

        else:
            raise ValueError(f"Unknown distribution: {dist_type}")

        all_z.append(pts)
        all_y.append(torch.full((n_per_cluster,), i, dtype=torch.long))

    z = torch.cat(all_z)
    labels = torch.cat(all_y)

    perm = torch.randperm(len(z))
    return z[perm], labels[perm]


# =============================================================================
# LABEL SAMPLER (on 2D points)
# =============================================================================

class SimLabelSampler:
    """Index points by label for positive/negative sampling.

    p_same: probability that a positive sample is drawn from the anchor's own
    cluster; with probability 1 - p_same it is drawn from the other clusters
    (per sample). Negatives always come from the other clusters.
    """
    def __init__(self, z, labels, p_same=1.0):
        self.z = z
        self.labels = labels
        self.p_same = p_same
        self.label_to_idx = {}
        for i in range(len(labels)):
            l = labels[i].item()
            self.label_to_idx.setdefault(l, []).append(i)
        for k in self.label_to_idx:
            self.label_to_idx[k] = torch.tensor(self.label_to_idx[k])

    def sample(self, anchor_labels, k, same_class=True, return_mask=False):
        """
        Returns (N, k, d) tensor of sampled points.
        If return_mask, also returns an (N, k) bool tensor marking samples
        drawn from the anchor's own cluster (class-consistent).
        """
        batch = []
        masks = []
        for l in anchor_labels.tolist():
            if same_class:
                pool = self.label_to_idx[l]
                idx = pool[torch.randint(len(pool), (k,))]
                mask = torch.ones(k, dtype=torch.bool)
                if self.p_same < 1.0:
                    other = torch.cat([v for key, v in self.label_to_idx.items()
                                       if key != l])
                    idx_other = other[torch.randint(len(other), (k,))]
                    mix = torch.rand(k) < self.p_same
                    idx = torch.where(mix, idx, idx_other)
                    mask = mix
            else:
                pool = torch.cat([v for key, v in self.label_to_idx.items()
                                  if key != l])
                idx = pool[torch.randint(len(pool), (k,))]
                mask = torch.zeros(k, dtype=torch.bool)
            batch.append(self.z[idx])           # (k, d)
            masks.append(mask)
        out = torch.stack(batch)                # (N, k, d)
        if return_mask:
            return out, torch.stack(masks)      # (N, k, d), (N, k)
        return out


# =============================================================================
# METRIC MODEL (no encoder — identity forward)
# =============================================================================

class SimMetricModel(nn.Module):
    """
    Same metric parameterizations as CCSModel, but no encoder.
    forward(z) = z  (identity).
    """
    def __init__(self, metric='single', d=2, n_norms=1):
        # n_norms is accepted for config compatibility but unused (kept at 1)
        super().__init__()
        self.metric = metric
        self.embed_dim = d
        self.add_infonce = False   # never for simulation

        if metric == 'single':
            self.A = nn.Parameter(torch.eye(d) + 0.1 * torch.randn(d, d))
            self.p_raw = nn.Parameter(torch.empty(1))
            nn.init.uniform_(self.p_raw, a=0.1, b=4.0)
        elif metric == 'generalized':
            self.a_raw = nn.Parameter(torch.ones(d))
            self.p_raw = nn.Parameter(torch.empty(d))
            nn.init.uniform_(self.p_raw, a=0.1, b=4.0)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def forward(self, z):
        return z

    def get_params(self):
        if self.metric == 'single':
            return self.A @ self.A.T, torch.clamp(torch.abs(self.p_raw) + 1e-3, 0.1, 10.0)
        if self.metric == 'generalized':
            m = self.a_raw ** 2 + 1e-3
            p = torch.clamp(torch.abs(self.p_raw) + 1e-3, 0.1, 10.0)
            return m, p
        raise ValueError(f"Unknown metric: {self.metric}")

    def compute_distance(self, z1, z2, m=None, p=None):
        if m is None or p is None:
            m, p = self.get_params()
        if z1.dim() == 3:
            z2 = z2.unsqueeze(1)
        diff = z1 - z2

        if self.metric == 'single':
            transformed = diff @ m
            return torch.pow((torch.abs(transformed) + 1e-8).pow(p).sum(-1), 1/p)

        if self.metric == 'generalized':
            return (m * (torch.abs(diff) + 1e-8).pow(p)).sum(-1)
        raise ValueError(f"Unknown metric: {self.metric}")


# =============================================================================
# VOLUME FORMULAS
# =============================================================================

def log_volume_generalized(p, m, t):
    inv_p = 1.0 / p
    sum_inv_p = inv_p.sum()
    return (sum_inv_p * torch.log(t + 1e-8)
            - torch.log(m + 1e-8).sum()
            + m.shape[0] * LOG2
            + torch.lgamma(1 + inv_p).sum()
            - torch.lgamma(1 + sum_inv_p))


def compute_log_volume(model, t):
    m, p = model.get_params()
    t_val = torch.clamp(torch.tensor(t, device=p.device)
                        if not torch.is_tensor(t) else t, min=1e-8, max=1e12)
    if model.metric == 'single':
        k = model.embed_dim
        log_det = -2 * torch.slogdet(model.A)[1]
        ball = k * LOG2 + k * torch.lgamma(1 + 1/p) - torch.lgamma(1 + k/p)
        return (log_det + k * torch.log(t_val + 1e-8) + ball).item()
    elif model.metric == 'generalized':
        return log_volume_generalized(p, m, t_val).item()
    raise ValueError(f"Unknown metric: {model.metric}")


# =============================================================================
# THRESHOLD COMPUTATION
# =============================================================================

def compute_t(dist_pos, alpha, window=5):
    flat = dist_pos.view(-1)
    n = flat.numel()
    k = min(int(math.ceil((1 - alpha) * n)), n - 1)
    sorted_vals, _ = torch.sort(flat)
    lo = max(0, k - window)
    hi = min(n, k + window + 1)
    center = float(k - lo)
    idx = torch.arange(hi - lo, device=flat.device, dtype=torch.float)
    weights = torch.softmax(-(idx - center).abs(), dim=0)
    return (weights * sorted_vals[lo:hi]).sum()


def compute_t_hard(dist_pos, alpha):
    flat = dist_pos.view(-1)
    k = min(int(math.ceil((1 - alpha) * flat.numel())), flat.numel() - 1) + 1
    return torch.kthvalue(flat, k)[0]


# =============================================================================
# LOSS
# =============================================================================

def compute_loss(z_anchor, z_pos, z_neg, model, alpha, problem, lam=0.05, scale=7.0):
    """Compute loss + metrics. Returns (loss, t, cov, exc, log_vol)."""
    m, p = model.get_params()
    dist_pos = model.compute_distance(z_pos, z_anchor, m, p)
    dist_neg = model.compute_distance(z_neg, z_anchor, m, p)

    t = compute_t(dist_pos, alpha)
    cov = (dist_pos <= t.detach()).float().mean().item()
    exc = (dist_neg > t.detach()).float().mean().item()

    if problem in ('vol', 'vol_neg'):
        t_clamped = torch.clamp(t, min=1e-4, max=1e12)
        if model.metric == 'single':
            k_dim = model.embed_dim
            log_det = -2 * torch.slogdet(model.A)[1]
            t_term = k_dim * torch.log(t_clamped + 1e-8)
            ball = k_dim * LOG2 + k_dim * torch.lgamma(1 + 1/p) - torch.lgamma(1 + k_dim/p)
            loss_vol = log_det + t_term + ball
        elif model.metric == 'generalized':
            loss_vol = log_volume_generalized(p, m, t_clamped)

    # regularize M away from infinity and p away from zero
    if problem in ('neg', 'vol_neg'):
        loss_neg = torch.sigmoid(scale * (t - dist_neg)).mean()

    reg_weight = 0.0
    # --- NEW STABLE REGULARIZATION ---
    # 1. L2 penalty on m: Minimizing this keeps m small (away from infinity)
    reg_m = torch.square(m).sum()
    
    # 2. Inverse penalty on p: Minimizing this creates a barrier stopping p from reaching 0
    reg_p = (1.0 / (p + 1e-8)).sum() 
    reg_tot = reg_m + reg_p

    if problem != 'vol_neg':
        if problem == 'vol':
            loss = loss_vol + reg_weight * (reg_tot)
                
        elif problem == 'neg':
            loss = loss_neg + reg_weight * (reg_tot)
        
    else:  # vol_neg
        loss = lam * loss_vol + loss_neg  + reg_weight * (reg_tot)

    log_vol = compute_log_volume(model, t.detach())
    return loss, t.item(), cov, exc, log_vol


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, z_train, labels_train, sampler, optimizer, args):
    model.train()
    device = args.device
    bs = args.batch_size
    n = len(z_train)
    perm = torch.randperm(n)

    total_loss, total_cov, total_exc, total_vol, nb = 0, 0, 0, 0, 0
    t_last = 0.0
    args.lam = getattr(args, 'lam', 0.05)  # default lambda for vol_neg if not set
    for i in range(0, n, bs):
        idx = perm[i:i+bs]
        z_a = z_train[idx].to(device)
        lab = labels_train[idx]

        z_p = sampler.sample(lab, args.k, same_class=True).to(device)
        z_n = sampler.sample(lab, args.k, same_class=False).to(device)

        loss, t, cov, exc, vol = compute_loss(
            z_a, z_p, z_n, model, args.alpha, args.problem, args.lam, args.scale)

        if torch.isnan(loss):
            continue

        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_cov += cov
        total_exc += exc
        total_vol += vol
        t_last = t
        nb += 1

    return (total_loss/nb if nb else 0, total_cov/nb if nb else 0,
            total_exc/nb if nb else 0, t_last, total_vol/nb if nb else 0)


@torch.no_grad()
def validate(model, z_val, labels_val, sampler_val, args):
    model.eval()
    device = args.device
    z_a = z_val.to(device)
    z_p = sampler_val.sample(labels_val, args.k, same_class=True).to(device)
    z_n = sampler_val.sample(labels_val, args.k, same_class=False).to(device)

    loss, t, cov, exc, vol = compute_loss(
        z_a, z_p, z_n, model, args.alpha, args.problem, args.lam, args.scale)
    return loss.item(), cov, exc, vol


def train_model(model, z_train, labels_train, z_val, labels_val,
                sampler_train, sampler_val, args):
    """Full training loop. Returns history dict."""
    if model.metric == 'generalized':
        params = [model.a_raw, model.p_raw]
    else:
        params = [model.A, model.p_raw]

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=5e-4)
    else:
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    history = {'loss': [], 'val_loss': [], 't': [], 'cov': [], 'exc': [],
               'log_vol': [], 'val_log_vol': [], 'p': []}
    val_exc_track = -1.0
    count = 0
    for epoch in range(args.epochs):
        count += 1
        loss, cov, exc, t, vol = train_epoch(
            model, z_train, labels_train, sampler_train, optimizer, args)
        val_loss, val_cov, val_exc, val_vol = validate(
            model, z_val, labels_val, sampler_val, args)
        scheduler.step()

        _, p = model.get_params()
        history['loss'].append(loss)
        history['val_loss'].append(val_loss)
        history['t'].append(t)
        history['cov'].append(val_cov)
        history['exc'].append(val_exc)
        history['log_vol'].append(vol)
        history['val_log_vol'].append(val_vol)
        history['p'].append(p.detach().cpu().numpy().copy() if p.dim() > 0 else p.item())

        p_str = f"[{p.min():.2f},{p.max():.2f}]" if p.dim() > 0 else f"{p.item():.2f}"
        print(f"  Ep {epoch+1:02d}/{args.epochs} | L:{loss:.3f}/{val_loss:.3f} | "
              f"t:{t:.3f} | Cov:{val_cov:.1%} | Exc:{val_exc:.1%} | "
              f"Vol:{val_vol/3:.1f} | p:{p_str}")
        
        ckpt_path = os.path.join("temp_sim_checkpoints", f"best_checkpoint.pt")
        # Report the model with lowest exclusion.
        if val_exc_track < val_exc:
            val_exc_track = val_exc
            save_checkpoint(model, ckpt_path)
            print(f"Saved a new best checkpoint: {ckpt_path}")
            count = 0
        
        if val_exc < 0.001 or count >= 8:
            load_checkpoint(model, ckpt_path, args.device)
            return history

    return history


# =============================================================================
# CONFORMAL EVALUATION
# =============================================================================

@torch.no_grad()
def conformal_evaluate(model, z_val, labels_val, sampler_val, args):
    """Split val in half: calibrate threshold on first, evaluate on second."""
    model.eval()
    device = args.device
    m, p = model.get_params()

    n = len(z_val) // 2
    z_cal, lab_cal = z_val[:n], labels_val[:n]
    z_test, lab_test = z_val[n:], labels_val[n:]

    # Build samplers for each half: calibration shares the training p,
    # the test half may use a different (possibly shifted) p_test.
    p_train = getattr(args, 'p_train', 1.0)
    p_test = getattr(args, 'p_test', p_train)
    sampler_cal = SimLabelSampler(z_val, labels_val, p_same=p_train)  # sample from full pool
    sampler_test = SimLabelSampler(z_val, labels_val, p_same=p_test)

    z_p_cal = sampler_cal.sample(lab_cal, args.k, same_class=True).to(device)
    dist_cal = model.compute_distance(z_p_cal, z_cal.to(device), m, p)
    threshold = compute_t_hard(dist_cal, args.alpha).item()

    # ONE draw of K positives under p_test; the mask records which of those K
    # happened to come from the anchor's own cluster (~p_test*K on average).
    z_p_test, pos_mask = sampler_test.sample(lab_test, args.k, same_class=True,
                                             return_mask=True)
    z_p_test = z_p_test.to(device)
    z_n_test = sampler_test.sample(lab_test, args.k, same_class=False).to(device)
    dist_pos = model.compute_distance(z_p_test, z_test.to(device), m, p)
    dist_neg = model.compute_distance(z_n_test, z_test.to(device), m, p)

    inside_pos = (dist_pos <= threshold)
    coverage = inside_pos.float().mean().item()
    exclusion = (dist_neg > threshold).float().mean().item()
    log_vol = compute_log_volume(model, threshold)

    # Cov_class: coverage restricted to the class-consistent positives of the
    # SAME draw (no re-sampling).
    pos_mask = pos_mask.to(inside_pos.device)
    n_class = pos_mask.sum().item()
    cov_class = ((inside_pos & pos_mask).sum().item() / n_class
                 if n_class > 0 else float('nan'))

    return threshold, coverage, exclusion, log_vol, cov_class


# =============================================================================
# BASELINES
# =============================================================================

@torch.no_grad()
def run_baseline(case, z_val, labels_val, sampler_val, args):
    """
    Baseline 0a (L2 ball) or 0b (Mahalanobis ellipsoid).
    Returns dict with threshold, coverage, exclusion, conformal_log_volume.
    """
    device = args.device
    d = z_val.shape[1]
    n = len(z_val) // 2

    # Split cal / test
    z_cal, lab_cal = z_val[:n], labels_val[:n]
    z_test, lab_test = z_val[n:], labels_val[n:]

    # Calibration shares the training p; the test half may use a shifted p_test.
    p_train = getattr(args, 'p_train', 1.0)
    p_test = getattr(args, 'p_test', p_train)
    sampler_cal = SimLabelSampler(z_val, labels_val, p_same=p_train)
    sampler_test = SimLabelSampler(z_val, labels_val, p_same=p_test)

    z_p_cal = sampler_cal.sample(lab_cal, args.k, same_class=True).to(device)
    # ONE draw of K positives under p_test; mask marks the class-consistent ones
    z_p_test, pos_mask = sampler_test.sample(lab_test, args.k, same_class=True,
                                             return_mask=True)
    z_p_test = z_p_test.to(device)
    pos_mask = pos_mask.to(device)
    z_n_test = sampler_test.sample(lab_test, args.k, same_class=False).to(device)

    z_cal_d = z_cal.to(device)
    z_test_d = z_test.to(device)

    if case == '0a':
        # L2 distance
        cal_dist = torch.norm(z_p_cal - z_cal_d.unsqueeze(1), dim=-1).view(-1)
        threshold = torch.sort(cal_dist)[0][int((1 - args.alpha) * len(cal_dist))].item()
        test_pos_dist = torch.norm(z_p_test - z_test_d.unsqueeze(1), dim=-1)
        test_neg_dist = torch.norm(z_n_test - z_test_d.unsqueeze(1), dim=-1)
        log_vol = ((d/2) * math.log(math.pi) - math.lgamma(d/2 + 1)
                   + d * math.log(threshold + 1e-8))
    elif case == '0b':  # 0b
        # Use first quarter for covariance, second quarter for calibration
        n_half = n // 2
        z_p_cov = sampler_cal.sample(lab_cal[:n_half], args.k, same_class=True).to(device)
        diffs_cov = (z_p_cov - z_cal_d[:n_half].unsqueeze(1)).view(-1, d)
        cov_mat = diffs_cov.T @ diffs_cov / len(diffs_cov)
        cov_inv = torch.linalg.inv(cov_mat + 1e-4 * torch.eye(d, device=device))

        z_p_cal2 = sampler_cal.sample(lab_cal[n_half:], args.k, same_class=True).to(device)
        diffs_cal = (z_p_cal2 - z_cal_d[n_half:].unsqueeze(1)).view(-1, d)
        cal_dist = torch.sqrt((diffs_cal @ cov_inv * diffs_cal).sum(-1))
        threshold = torch.sort(cal_dist)[0][int((1 - args.alpha) * len(cal_dist))].item()

        test_pos_diff = (z_p_test - z_test_d.unsqueeze(1)).view(-1, d)
        test_neg_diff = (z_n_test - z_test_d.unsqueeze(1)).view(-1, d)
        test_pos_dist = torch.sqrt((test_pos_diff @ cov_inv * test_pos_diff).sum(-1)).view(z_p_test.shape[:2])
        test_neg_dist = torch.sqrt((test_neg_diff @ cov_inv * test_neg_diff).sum(-1)).view(z_n_test.shape[:2])

        log_det_cov = -torch.linalg.slogdet(cov_inv)[1].item()
        log_vol = ((d/2) * math.log(math.pi) - math.lgamma(d/2 + 1)
                   + d * math.log(threshold + 1e-8) + 0.5 * log_det_cov)
        
    elif 'hull' in case.lower():
        print(f"Running method of convex hull shrinkage.")
        from scipy.spatial import Delaunay

        z_p_test_np = z_p_test.cpu().numpy()  # (N_test, k, d)
        z_n_test_np = z_n_test.cpu().numpy()  # (N_test, k, d)

        coverages = []
        exclusions = []
        target = 1.0 - args.alpha
        stride = 0.01
        mask_np = pos_mask.cpu().numpy()
        cls_num, cls_den = 0, 0

        for i in range(z_p_test_np.shape[0]):
            pos = z_p_test_np[i]  # (k, d)

            try:
                hull = ConvexHull(pos)
            except Exception:
                continue

            hull_pts = pos[hull.vertices]
            centroid = hull_pts.mean(axis=0)

            # Sweep shrink factor down; keep the last state where positive
            # coverage is still >= 1 - alpha (just above the target).
            last_cov = None
            last_exc = None
            last_inside = None
            shrink_factor = 1.0
            pos = z_p_test_np[i]  # (k, d)
            neg = z_n_test_np[i]  # (k, d)
            m = 1
            while shrink_factor > 0:
                shrunk = centroid + shrink_factor * (hull_pts - centroid)
                try:
                    delaunay = Delaunay(shrunk)
                except Exception:
                    break
                inside_pos = delaunay.find_simplex(pos) >= 0
                cov = float(inside_pos.mean())
                if cov < target:
                    if m == 1:
                        inside_neg = delaunay.find_simplex(neg) >= 0
                        last_cov = cov
                        last_exc = float((~inside_neg).mean())
                        last_inside = inside_pos
                    m = 0
                    break
                inside_neg = delaunay.find_simplex(neg) >= 0
                last_cov = cov
                last_exc = float((~inside_neg).mean())
                last_inside = inside_pos
                shrink_factor -= stride

            if last_cov is None:
                continue
            coverages.append(last_cov)
            exclusions.append(last_exc)
            cls_num += int((last_inside & mask_np[i]).sum())
            cls_den += int(mask_np[i].sum())

        coverage = float(np.mean(coverages)) if coverages else 0.0
        exclusion = float(np.mean(exclusions)) if exclusions else 0.0
        cov_class = cls_num / cls_den if cls_den > 0 else float('nan')
        log_vol = 0.0  # not a distance/volume-based method
        threshold = 0.0  # placeholder; expected by downstream code

        print(f"  Baseline {case}: Cov={coverage:.1%}, "
            f"Exc={exclusion:.1%}, LogVol={log_vol:.2f}")

        return {'threshold': threshold, 'coverage': coverage, 'exclusion': exclusion,
                'conformal_log_volume': log_vol, 'cov_class': cov_class,
                'cov_inv': None}
    
    else: #HIB
        # HIB models each input as Z ~ N(mu(x), Sigma(x)) with diagonal Sigma
        # (in the paper, mu and Sigma are produced by a CNN). With no encoder
        # in the simulation, we instantiate the same idea by fitting a
        # per-anchor diagonal Gaussian directly to the test point's positives
        # (mean = mu, sample variance = Sigma). The set predictor is the
        # k-sigma ellipsoid {z : (z-mu)^T Sigma^{-1} (z-mu) <= k^2}.
        import scipy.stats as scst
        print(f"Running HIB {2.0:.0f}-sigma ellipsoid baseline.")
        sigma_factor = 2 # math.sqrt(scst.chi2.ppf(1 - args.alpha, d))
        thresh_sq = sigma_factor ** 2

        z_p_test_np = z_p_test.cpu().numpy()  # (N_test, k, d)
        z_n_test_np = z_n_test.cpu().numpy()  # (N_test, k, d)

        coverages = []
        exclusions = []
        log_vols = []
        log_unit_ball = (d / 2) * math.log(math.pi) - math.lgamma(d / 2 + 1)
        mask_np = pos_mask.cpu().numpy()
        cls_num, cls_den = 0, 0

        for i in range(z_p_test_np.shape[0]):
            pos = z_p_test_np[i,:args.k//4]  # (k/2, d)
            neg = z_n_test_np[i]  # (k, d)

            mu = pos.mean(axis=0)
            var = np.maximum(pos.var(axis=0, ddof=1), 1e-8)
            pos = z_p_test_np[i] # (k, d)
            diff_p = pos - mu
            d2_p = ((diff_p ** 2) / var).sum(axis=1)
            diff_n = neg - mu
            d2_n = ((diff_n ** 2) / var).sum(axis=1)

            inside_p = d2_p <= thresh_sq
            coverages.append(float(inside_p.mean()))
            exclusions.append(float((d2_n > thresh_sq).mean()))
            cls_num += int((inside_p & mask_np[i]).sum())
            cls_den += int(mask_np[i].sum())

            # Vol of k-sigma ellipsoid = V_d * sqrt(det Sigma) * k^d.
            log_vols.append(log_unit_ball
                            + 0.5 * np.log(var).sum()
                            + d * math.log(sigma_factor + 1e-8))

        coverage = float(np.mean(coverages)) if coverages else 0.0
        exclusion = float(np.mean(exclusions)) if exclusions else 0.0
        cov_class = cls_num / cls_den if cls_den > 0 else float('nan')
        log_vol = float(np.mean(log_vols)) if log_vols else 0.0
        threshold = 0.0  # placeholder; HIB uses a per-point ellipsoid

        print(f"  Baseline {case}: Cov={coverage:.1%}, "
              f"Exc={exclusion:.1%}, LogVol={log_vol:.2f}")

        return {'threshold': threshold, 'coverage': coverage,
                'exclusion': exclusion, 'conformal_log_volume': log_vol,
                'cov_class': cov_class,
                'cov_inv': None}

    inside_pos = (test_pos_dist <= threshold)
    coverage = inside_pos.float().mean().item()
    exclusion = (test_neg_dist > threshold).float().mean().item()

    n_class = pos_mask.sum().item()
    cov_class = ((inside_pos & pos_mask).sum().item() / n_class
                 if n_class > 0 else float('nan'))

    print(f"  Baseline {case}: t={threshold:.4f}, Cov={coverage:.1%}, "
          f"Exc={exclusion:.1%}, LogVol={log_vol:.2f}")

    return {'threshold': threshold, 'coverage': coverage, 'exclusion': exclusion,
            'conformal_log_volume': log_vol, 'cov_class': cov_class,
            'cov_inv': cov_inv.cpu() if case == '0b' else None}


# =============================================================================
# PLOTTING
# =============================================================================

def plot_clusters(z, labels, title='Clusters', save_path=None):
    """Scatter plot of labelled clusters. 2D → matplotlib PNG, 3D → plotly HTML."""
    d = z.shape[1]
    z_np = z.numpy()

    if d >= 3:
        import plotly.graph_objects as go
        fig = go.Figure()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        for c in sorted(labels.unique().tolist()):
            mask = (labels == c).numpy()
            fig.add_trace(go.Scatter3d(
                x=z_np[mask, 0], y=z_np[mask, 1], z=z_np[mask, 2],
                mode='markers',
                marker=dict(size=2, color=colors[c % len(colors)], opacity=0.5),
                name=f'Class {c}'
            ))
        fig.update_layout(
            title=title,
            scene=dict(xaxis_title='z₁', yaxis_title='z₂', zaxis_title='z₃',
                       aspectmode='data'),
            legend=dict(itemsizing='constant'),
            margin=dict(l=0, r=0, t=40, b=0),
            width=900, height=800
        )
        if save_path:
            save_path = _ensure_html_ext(save_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.write_html(save_path)
        return save_path
    else:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 7))
        for c in sorted(labels.unique().tolist()):
            mask = (labels == c).numpy()
            ax.scatter(z_np[mask, 0], z_np[mask, 1],
                       s=8, alpha=0.5, label=f'Class {c}')
        ax.set_aspect('equal')
        ax.set_xlabel('$z_1$'); ax.set_ylabel('$z_2$')
        ax.legend()
        ax.set_title(title)
        ax.grid(True, alpha=0.2)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path


def _ensure_html_ext(path):
    """Replace extension with .html for interactive plots."""
    base, _ = os.path.splitext(path)
    return base + '.html'


def plot_training(history, title='Training', save_path=None):
    """Plot training curves."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs = range(1, len(history['loss']) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0,0].plot(epochs, history['loss'], 'b-', label='Train')
    axes[0,0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0,0].set_title('Loss'); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

    axes[0,1].plot(epochs, history['t'], 'g-')
    axes[0,1].set_title('Threshold t'); axes[0,1].grid(True, alpha=0.3)

    axes[1,0].plot(epochs, history['cov'], 'purple', label='Coverage')
    axes[1,0].plot(epochs, history['exc'], 'orange', label='Exclusion')
    axes[1,0].axhline(0.95, color='red', ls='--', alpha=0.5)
    axes[1,0].set_title('Coverage & Exclusion'); axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

    axes[1,1].plot(epochs, history['log_vol'], 'b-', label='Train')
    axes[1,1].plot(epochs, history['val_log_vol'], 'r-', label='Val')
    axes[1,1].set_title('Log Volume'); axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


@torch.no_grad()
def _exact_boundary(model, anchor, threshold, directions):
    """Exact boundary of {z : g(anchor, z) <= t} along rays from the anchor.

    The covering set is star-shaped around the anchor for every metric type,
    so the boundary point along a unit direction u is anchor + s*u where
    g(anchor, anchor + s*u) = t. s is found by vectorized bisection on the
    model's own compute_distance, so the plotted set is EXACTLY the trained
    one (single or generalized, convex or not).

    directions: (n, d) unit vectors. Returns (n, d) boundary points (torch).
    """
    device = next(model.parameters()).device
    anchor = anchor.to(device).float()
    U = directions.to(device).float()
    m, p = model.get_params()

    def dist_at(s):  # s: (n,) -> distances (n,)
        pts = anchor.unsqueeze(0) + s.unsqueeze(1) * U          # (n, d)
        return model.compute_distance(pts.unsqueeze(0),
                                      anchor.unsqueeze(0), m, p).reshape(-1)

    n = U.shape[0]
    hi = torch.ones(n, device=device)
    for _ in range(60):                       # grow until outside everywhere
        inside = dist_at(hi) <= threshold
        if not inside.any():
            break
        hi = torch.where(inside, hi * 2.0, hi)
    lo = torch.zeros(n, device=device)
    for _ in range(60):                       # bisection
        mid = 0.5 * (lo + hi)
        inside = dist_at(mid) <= threshold
        lo = torch.where(inside, mid, lo)
        hi = torch.where(inside, hi, mid)
    s = 0.5 * (lo + hi)
    return (anchor.unsqueeze(0) + s.unsqueeze(1) * U).cpu()


@torch.no_grad()
def plot_covering_set_paper(model, anchor, z_pos, z_neg, threshold,
                            z_all, labels_all, method_name='Method',
                            n_boundary=720, save_path=None):
    """Publication-quality PNG of one anchor's exact covering set (2D or 3D).

    Draws the data clusters as recessive context, the exact metric ball
    {z : g(anchor, z) <= t} as a filled region with a crisp boundary, and the
    anchor / positives / negatives on top. Colors are a validated
    colorblind-safe palette; light background for print.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Validated palette (light mode)
    C_FILL, C_EDGE = '#cde2fb', '#1c5cab'   # covering set: light blue + deep blue edge
    C_POS, C_NEG = '#1baf7a', '#e34948'     # aqua positives, red negatives
    C_ANCHOR = '#eda100'                    # yellow star
    C_CTX, C_INK, C_SUB = '#c9c8c2', '#0b0b0b', '#52514e'

    d = anchor.shape[0]
    a_np = anchor.cpu().numpy()
    pos_np, neg_np = z_pos.cpu().numpy(), z_neg.cpu().numpy()
    z_np = z_all.cpu().numpy()

    if d == 2:
        theta = torch.linspace(0, 2 * math.pi, n_boundary + 1)[:-1]
        U = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
        B = _exact_boundary(model, anchor, threshold, U).numpy()

        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
        fig.patch.set_facecolor('white'); ax.set_facecolor('white')

        ax.scatter(z_np[:, 0], z_np[:, 1], s=7, c=C_CTX, alpha=0.8,
                   linewidths=0, zorder=1, rasterized=True)
        ax.fill(B[:, 0], B[:, 1], color=C_FILL, alpha=0.55, zorder=2)
        ax.plot(np.append(B[:, 0], B[0, 0]), np.append(B[:, 1], B[0, 1]),
                color=C_EDGE, lw=2.0, zorder=3,
                label=rf'Covering set ($t={threshold:.2f}$)')
        ax.scatter(pos_np[:, 0], pos_np[:, 1], s=34, c=C_POS, marker='o',
                   edgecolors='white', linewidths=0.6, zorder=4,
                   label='Positives')
        ax.scatter(neg_np[:, 0], neg_np[:, 1], s=34, c=C_NEG, marker='X',
                   edgecolors='white', linewidths=0.6, zorder=4,
                   label='Negatives')
        ax.scatter([a_np[0]], [a_np[1]], s=280, c=C_ANCHOR, marker='*',
                   edgecolors=C_INK, linewidths=0.8, zorder=5, label='Anchor')

        # Window: covering set + samples, padded; context fills the rest
        pts = np.vstack([B, pos_np, neg_np, a_np[None]])
        lo_w, hi_w = pts.min(0), pts.max(0)
        pad = 0.18 * (hi_w - lo_w).max()
        ax.set_xlim(lo_w[0] - pad, hi_w[0] + pad)
        ax.set_ylim(lo_w[1] - pad, hi_w[1] + pad)
        ax.set_aspect('equal')
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(method_name, fontsize=13, color=C_INK, pad=10)
        leg = ax.legend(loc='upper right', fontsize=8.5, frameon=True,
                        framealpha=0.9, edgecolor='none',
                        labelcolor=C_SUB, borderpad=0.7)
        leg.set_zorder(6)

    elif d == 3:
        th = torch.linspace(0, 2 * math.pi, 121)
        ph = torch.linspace(1e-3, math.pi - 1e-3, 61)
        TH, PH = torch.meshgrid(th, ph, indexing='ij')
        U = torch.stack([torch.sin(PH) * torch.cos(TH),
                         torch.sin(PH) * torch.sin(TH),
                         torch.cos(PH)], dim=-1).reshape(-1, 3)
        B = _exact_boundary(model, anchor, threshold, U).numpy()
        X = B[:, 0].reshape(121, 61); Y = B[:, 1].reshape(121, 61)
        Z = B[:, 2].reshape(121, 61)

        fig = plt.figure(figsize=(7, 7), dpi=300)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(z_np[:, 0], z_np[:, 1], z_np[:, 2], s=4, c=C_CTX,
                   alpha=0.35, linewidths=0)
        ax.plot_surface(X, Y, Z, color=C_FILL, alpha=0.35, linewidth=0,
                        antialiased=True, shade=True)
        ax.scatter(pos_np[:, 0], pos_np[:, 1], pos_np[:, 2], s=26, c=C_POS,
                   marker='o', edgecolors='white', linewidths=0.4,
                   label='Positives')
        ax.scatter(neg_np[:, 0], neg_np[:, 1], neg_np[:, 2], s=26, c=C_NEG,
                   marker='X', edgecolors='white', linewidths=0.4,
                   label='Negatives')
        ax.scatter([a_np[0]], [a_np[1]], [a_np[2]], s=240, c=C_ANCHOR,
                   marker='*', edgecolors=C_INK, linewidths=0.8,
                   label='Anchor')
        ax.set_title(rf'{method_name}  ($t={threshold:.2f}$)',
                     fontsize=13, color=C_INK)
        ax.legend(loc='upper right', fontsize=8.5, labelcolor=C_SUB)
        ax.set_box_aspect((1, 1, 1))
    else:
        print(f"  plot_covering_set_paper: d={d} not supported (2D/3D only)")
        return None

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white')
    plt.close()
    return save_path


