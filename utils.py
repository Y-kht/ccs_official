"""CCS utilities: models, losses, data, training."""
import math
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.datasets import STL10
from torchvision.models import resnet18, ResNet18_Weights


def set_seed(seed):
    """Set random seed for reproducibility."""
    RNG = torch.Generator().manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return RNG

# =============================================================================
# DATA
# =============================================================================

def get_transforms(size=224, train=True, dataset='cifar100', model_name='resnet18'):
    """Get train/val transforms with normalization matching the model's pretraining."""
    # Normalization must match what the pretrained model was trained with:
    #   resnet18        → pretrained on ImageNet  → ImageNet stats
    #   repvgg_a2       → pretrained on CIFAR-100 → CIFAR-100 stats
    #   resnet18_simclr → pretrained with SimCLR  → (0.5, 0.5, 0.5)
    if model_name == 'repvgg_a2':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif model_name == 'resnet18_simclr':
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    else:  # resnet18 (ImageNet pretrained)
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    if train:
        if dataset == 'cifar100':
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        elif dataset == 'stl10':
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(size=96),
                transforms.RandomApply([
                    transforms.ColorJitter(0.5, 0.5, 0.5, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=9),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:  # imagenet100
            return transforms.Compose([
                transforms.RandomResizedCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
    return transforms.Compose([
        transforms.Resize(size), transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def get_loaders(dataset, batch_size, num_workers, train_dir=None, val_dir=None,
                r=1.0, model_name='resnet18', cal_use=False):
    """Get train/val data loaders. r controls the fraction of training data used."""
    DATASETPATH = './data'
    if dataset == 'cifar100':
        train_data = datasets.CIFAR100(DATASETPATH, train=True, download=True,
                                       transform=get_transforms(32, True, dataset, model_name))
        cal_data = datasets.CIFAR100(DATASETPATH, train=False, download=True,
                                     transform=get_transforms(32, False, dataset, model_name))
    elif dataset == 'stl10':
        # STL10: use 'train' (5k labeled) for training, 'test' (8k labeled) for val.
        # Single-view transforms — augmentation for positives is handled by generate_samples.
        train_data = STL10(root=DATASETPATH, split='train', download=True,
                           transform=get_transforms(96, True, dataset, model_name))
        cal_data = STL10(root=DATASETPATH, split='test', download=True,
                         transform=get_transforms(96, False, dataset, model_name))
    else:  # imagenet100
        train_data = datasets.ImageFolder(train_dir,
                                          transform=get_transforms(224, True, dataset, model_name))
        cal_data = datasets.ImageFolder(val_dir,
                                        transform=get_transforms(224, False, dataset, model_name))

    # Subsample training set according to ratio r
    if cal_use:
        n_cal = max(1, int(0.5 * len(cal_data)))
        indices = torch.randperm(len(cal_data)).tolist()
        # Split the randomized indices into two parts
        train_indices = indices[:n_cal]
        remaining_indices = indices[n_cal:]
        # Create both subsets
        val_data = Subset(cal_data, remaining_indices)
        train_data = Subset(cal_data, train_indices)
        print(f"Using {n_cal}/{len(train_data) + len(val_data)} "
              f"({0.5:.0%}) of calibration data")
    else:    
        if r < 1.0:
            n_train = max(1, int(r * len(train_data)))
            indices = torch.randperm(len(train_data))[:n_train].tolist()
            train_data = Subset(train_data, indices)
            print(f"Using {n_train}/{n_train + len(train_data) - n_train} "
                f"({r:.0%}) of training data")
        val_data = cal_data

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers)
    return train_loader, val_loader

class LabelSampler:
    """Index samples by label for supervised sampling."""
    def __init__(self, dataset):
        self.label_to_indices = {}
        self.images = []
        for idx, (img, label) in enumerate(dataset):
            self.label_to_indices.setdefault(label, []).append(idx)
            self.images.append(img)
        self.images = torch.stack(self.images)
        for k in self.label_to_indices:
            self.label_to_indices[k] = torch.tensor(self.label_to_indices[k])
        # Fix 2: cache the different-class index pool per label (built lazily,
        # once each) instead of concatenating it for every anchor, every batch.
        self._neg_pool_cache = {}

    def sample(self, labels, k, same_class=True, replacement=True):
        """Sample k images per label (same or different class)."""
        device = labels.device if torch.is_tensor(labels) else 'cpu'
        labels = labels.cpu().numpy() if torch.is_tensor(labels) else labels
        batch = []
        for label in labels:
            if same_class:
                pool = self.label_to_indices[label]
            else:
                key = int(label)
                pool = self._neg_pool_cache.get(key)
                if pool is None:
                    pool = torch.cat([v for l, v in self.label_to_indices.items()
                                      if l != label])
                    self._neg_pool_cache[key] = pool
            idx = pool[torch.randint(len(pool), (k,))] if replacement else pool[torch.randperm(len(pool))[:k]]
            batch.append(self.images[idx])
        return torch.stack(batch).to(device)

# GPU augmentation for positive generation
_aug_transform = None
def get_aug_transform(device):
    global _aug_transform
    if _aug_transform is None:
        import torchvision.transforms.v2 as T
        _aug_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        ])
    return _aug_transform

def generate_samples(imgs, labels, k, sampler=None, positive=True, replacement=True):
    """Generate positive/negative samples via augmentation or label-based sampling."""
    N = imgs.shape[0]
    device = imgs.device

    if sampler is not None:
        return sampler.sample(labels, k, same_class=positive, replacement=replacement).to(device)

    if positive:
        aug = get_aug_transform(device)
        out = torch.stack([aug(imgs) for _ in range(k)], dim=1)  # (N, K, C, H, W)
    else:
        # Use other batch images as negatives
        idx = torch.stack([torch.randperm(N) for _ in range(k)], dim=1)
        out = imgs[idx]
    return out

# =============================================================================
# MODELS
# =============================================================================

# --- RepVGG-A2 for CIFAR-100 ------------------------------------------------

def _conv_bn(in_ch, out_ch, ks, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_ch, out_ch, ks, stride, padding,
                                        groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(out_ch))
    return result

class RepVGGBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 groups=1, deploy=False):
        super().__init__()
        self.deploy, self.groups, self.in_channels = deploy, groups, in_ch
        self.nonlinearity = nn.ReLU()
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_ch, out_ch, kernel_size, stride,
                                         padding, groups=groups, bias=True)
        else:
            self.rbr_identity = (nn.BatchNorm2d(in_ch)
                                 if out_ch == in_ch and stride == 1 else None)
            self.rbr_dense = _conv_bn(in_ch, out_ch, kernel_size, stride, padding, groups)
            self.rbr_1x1 = _conv_bn(in_ch, out_ch, 1, stride, padding - kernel_size // 2, groups)

    def forward(self, x):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(x))
        id_out = 0 if self.rbr_identity is None else self.rbr_identity(x)
        return self.nonlinearity(self.rbr_dense(x) + self.rbr_1x1(x) + id_out)

class _RepVGGA2Features(nn.Module):
    """RepVGG-A2 feature extractor (stages + global avg pool). Output: 1408-d."""
    NUM_BLOCKS = [2, 4, 14, 1]
    WIDTH_MULT = [1.5, 1.5, 1.5, 2.75]
    PRETRAINED_URL = ('https://github.com/chenyaofo/pytorch-cifar-models/'
                      'releases/download/repvgg/cifar100_repvgg_a2-8e71b1f8.pt')

    def __init__(self, pretrained=False):
        super().__init__()
        self.in_planes = min(64, int(64 * self.WIDTH_MULT[0]))
        self.stage0 = RepVGGBlock(3, self.in_planes)
        self.stage1 = self._stage(int(64 * self.WIDTH_MULT[0]),  self.NUM_BLOCKS[0], 1)
        self.stage2 = self._stage(int(128 * self.WIDTH_MULT[1]), self.NUM_BLOCKS[1], 2)
        self.stage3 = self._stage(int(256 * self.WIDTH_MULT[2]), self.NUM_BLOCKS[2], 2)
        self.stage4 = self._stage(int(512 * self.WIDTH_MULT[3]), self.NUM_BLOCKS[3], 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
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
        # drop fc / linear keys (we only use the feature extractor)
        sd = {k: v for k, v in sd.items()
              if not k.startswith(('fc.', 'linear.', 'flatten.', 'gap.'))}
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if missing:
            raise RuntimeError(f"RepVGG-A2 pretrained weights missing keys: {missing[:5]}...")
        if unexpected:
            print(f"  Note: {len(unexpected)} unused keys in checkpoint (fc/linear head)")

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.gap(x).flatten(1)   # (B, 1408)

# --- Encoder builder ---------------------------------------------------------

def build_encoder(model_name, input_size, pretrained=True):
    """Return (features_module, feat_dim) for the chosen backbone."""
    if model_name == 'repvgg_a2':
        features = _RepVGGA2Features(pretrained=pretrained)
        return features, 1408
    elif model_name == 'resnet18_simclr':
        # SimCLR-pretrained ResNet18 for STL10
        backbone = resnet18(weights=None)
        if pretrained:
            # Path to a SimCLR-pretrained ResNet18 checkpoint for STL10;
            # set the SIMCLR_CKPT environment variable or edit the default.
            ckpt_path = os.environ.get('SIMCLR_CKPT',
                                       'path/to/ResNet_simclr_STL10.ckpt')
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)

            # Extract state dict from various checkpoint formats
            for key in ('model_state_dict', 'state_dict', 'model', 'encoder'):
                if key in checkpoint:
                    checkpoint = checkpoint[key]
                    break

            # Filter out projection head / fc keys
            skip_prefixes = ('projection', 'head', 'fc.', 'projector',
                             'classifier', 'linear')
            state = {k: v for k, v in checkpoint.items()
                     if not k.startswith(skip_prefixes)}

            # Auto-detect and strip common key prefixes to match bare ResNet18 keys
            backbone_keys = set(backbone.state_dict().keys())
            if state and not (set(state.keys()) & backbone_keys):
                # No overlap — keys are probably prefixed
                sample_key = next(iter(state))
                for prefix in ['convnet.', 'encoder.', 'backbone.', 'model.', 
                               'feature_extractor.', 'resnet.', 'net.', 
                               'base_encoder.', 'f.']:
                    if sample_key.startswith(prefix):
                        state = {k[len(prefix):]: v for k, v in state.items()
                                 if k.startswith(prefix)}
                        print(f"  SimCLR: stripped key prefix '{prefix}'")
                        break

            # Filter out keys that don't belong to the backbone
            state = {k: v for k, v in state.items()
                     if k in backbone_keys and state[k].shape == backbone.state_dict()[k].shape}

            missing, unexpected = backbone.load_state_dict(state, strict=False)
            n_loaded = len(backbone.state_dict()) - len(missing)
            print(f"  SimCLR: loaded {n_loaded}/{len(backbone.state_dict())} params "
                  f"({len(missing)} missing, {len(unexpected)} unexpected)")
            if n_loaded == 0:
                raise RuntimeError(
                    f"SimCLR checkpoint loaded 0 parameters! "
                    f"Checkpoint keys sample: {list(checkpoint.keys())[:5]}")
        features = nn.Sequential(*list(backbone.children())[:-1])
        return features, 512
    else:  # resnet18 (ImageNet pretrained)
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        if input_size == 32:
            backbone.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            backbone.maxpool = nn.Identity()
        features = nn.Sequential(*list(backbone.children())[:-1])
        return features, 512

def get_feat_dim(model_name):
    """Return the feature dimension for a given backbone."""
    return 1408 if model_name == 'repvgg_a2' else 512

# -----------------------------------------------------------------------------

class CCSModel(nn.Module):
    """
    CCS model parameterized by:
      metric:      'single' | 'generalized'
      add_infonce: False → metric-only training | True → adds projector + InfoNCE
    """
    def __init__(self, metric='single', add_infonce=False, n_norms=3,
                 input_size=224, delta=1.0, model_name='resnet18'):
        super().__init__()
        self.metric = metric
        self.add_infonce = add_infonce

        # Encoder — always pretrained and frozen
        self.features, feat_dim = build_encoder(model_name, input_size, pretrained=True)
        for p in self.features.parameters(): p.requires_grad = False
        self.features = self.features.half()

        self.embed_dim = feat_dim
        self.k_g = feat_dim
        self.proj_dim = n_norms if n_norms > 1 else feat_dim

        # Semantic projector — only trainable layer when add_infonce
        if add_infonce:
            self.projector = nn.Linear(feat_dim, self.proj_dim)
            self.register_buffer('delta', torch.tensor(float(delta)))
            # Metric parameters
            if metric == 'single':
                self.A = nn.Parameter(torch.randn(self.proj_dim, self.proj_dim) * 0.1)
                self.p_raw = nn.Parameter(torch.tensor(2.0))
            elif metric == 'generalized':
                # Per-dimension: m_j = a_j^2, p_j per dimension
                self.a_raw = nn.Parameter(torch.ones(self.proj_dim))
                self.p_raw = nn.Parameter(torch.full((self.proj_dim,), 2.0))
        else:
            # Metric parameters
            if metric == 'single':
                self.A = nn.Parameter(torch.randn(feat_dim, feat_dim) * 0.1)
                self.p_raw = nn.Parameter(torch.tensor(2.0))
            elif metric == 'generalized':
                # Per-dimension: m_j = a_j^2, p_j per dimension
                self.a_raw = nn.Parameter(torch.ones(feat_dim))
                self.p_raw = nn.Parameter(torch.full((feat_dim,), 2.0))

    def forward(self, x):
        with torch.no_grad():
            z = self.features(x.half()).view(x.size(0), -1).float()
        if self.add_infonce:
            z = self.projector(z)
        return z

    def get_params(self):
        """Return metric parameters (m, p)."""
        if self.metric == 'single':
            return torch.clamp(self.A @ self.A.T, min=1e-9), torch.clamp(torch.abs(self.p_raw) + 1e-3, 0.1, 10.0)
        if self.metric == 'generalized':
            m = torch.clamp(self.a_raw ** 2 + 1e-3, min=1e-9)          # (d,)
            p = torch.clamp(torch.abs(self.p_raw) + 1e-3, 0.1, 10.0)  # (d,)
            return m, p

    def compute_distance(self, z1, z2, m=None, p=None):
        """Compute metric distance g(z1, z2)."""
        if m is None or p is None:
            m, p = self.get_params()

        if z1.dim() == 3:
            z2 = z2.unsqueeze(1)
        diff = z1 - z2

        if self.metric == 'single':
            # Distance is ||M * (z1-z2)||_p where M = AA^T.
            transformed = diff @ m
            return torch.pow((torch.abs(transformed) + 1e-8).pow(p).sum(-1), 1/p)

        if self.metric == 'generalized':
            # d(z1,z2) = Σ_j m_j |Δz_j|^{p_j},  m: (d,), p: (d,)
            return (m * (torch.abs(diff) + 1e-8).pow(p)).sum(-1)

# =============================================================================
# LOSSES
# =============================================================================

LOG2 = math.log(2.0)

def log_volume_generalized(p, m, t):
    """Normalized volume V^(1/d) of generalized ℓ_p ball."""
    d = m.shape[0]
    inv_p = 1.0 / p
    sum_inv_p = inv_p.sum()
    log_vol = (sum_inv_p * torch.log(t + 1e-8)
               - torch.log(m + 1e-8).sum()
               + d * LOG2
               + torch.lgamma(1 + inv_p).sum()
               - torch.lgamma(1 + sum_inv_p))
    return log_vol

def compute_log_volume(model, t):
    """Compute normalized volume of the covering set for any metric type."""
    m, p = model.get_params()
    t_val = torch.clamp(torch.tensor(t, device=p.device) if not torch.is_tensor(t) else t, min=1e-8)
    if model.metric == 'single':
        k = model.embed_dim
        log_det = -2 * torch.slogdet(model.A)[1]
        ball = k * LOG2 + k * torch.lgamma(1 + 1/p) - torch.lgamma(1 + k/p)
        log_vol = log_det + k * torch.log(t_val + 1e-8) + ball
        return log_vol.item()
    elif model.metric == 'generalized':
        return log_volume_generalized(p, m, t_val).item()
    


def compute_t(dist_pos, alpha, window=5):
    """
    Soft (1-α) quantile threshold.
    Averages over a window of sorted values around the quantile point
    so that gradients flow through multiple samples instead of just one.
    """
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
    """Hard (1-α) quantile threshold for evaluation / conformal calibration."""
    flat = dist_pos.view(-1)
    k = min(int(math.ceil((1 - alpha) * flat.numel())), flat.numel() - 1) + 1
    return torch.kthvalue(flat, k)[0]

def info_nce_loss(z_anchor, z_pos, z_neg, temperature=0.1):
    """InfoNCE contrastive loss."""
    z_a = F.normalize(z_anchor, dim=-1)
    z_p = F.normalize(z_pos, dim=-1)
    z_n = F.normalize(z_neg, dim=-1)

    sim_pos = (z_a.unsqueeze(1) * z_p).sum(-1) / temperature
    sim_neg = (z_a.unsqueeze(1) * z_n).sum(-1) / temperature
    logits = torch.cat([sim_pos, sim_neg], dim=1)

    loss = -F.log_softmax(logits, dim=1)[:, :z_pos.shape[1]].mean()
    acc = (logits.argmax(1) < z_pos.shape[1]).float().mean().item()
    return loss, acc

def compute_loss(z_anchor, z_pos, z_neg, model, alpha, problem,
                 phase='joint', contrastive_weight=1.0, volume_weight=0.1,
                 temperature=0.1, scale=7.0, lam=None, reg_weight=0.0):
    """
    Unified loss for all metric/problem/infonce combinations.
      problem='vol': minimize covering set volume
      problem='neg': minimize negative inclusion rate
      lam: if not None, loss = neg + lam*vol (overrides problem)
      phase: 'joint' | 'metric' | 'model' (only for add_infonce + alternating)
    """
    m, p = model.get_params()
    dist_pos = model.compute_distance(z_pos, z_anchor, m, p)
    dist_neg = model.compute_distance(z_neg, z_anchor, m, p)

    # Threshold = (1-α) quantile of positive distances
    t = compute_t(dist_pos, alpha)

    cov = (dist_pos <= t.detach()).float().mean().item()
    exc = (dist_neg > t.detach()).float().mean().item()

    # --- Helper: compute volume loss ---
    def _vol_loss():
        t_clamped = torch.clamp(t, min=1e-4)
        if model.metric == 'single':
            k_dim = model.embed_dim
            log_det = -2 * torch.slogdet(model.A)[1]
            t_term = k_dim * torch.log(t_clamped + 1e-8)
            ball = k_dim * LOG2 + k_dim * torch.lgamma(1 + 1/p) - torch.lgamma(1 + k_dim/p)
            return log_det + t_term + ball
        elif model.metric == 'generalized':
            if phase == 'model':
                return log_volume_generalized(p.detach(), m.detach(), t_clamped)
            return log_volume_generalized(p, m, t_clamped)

    # --- Helper: compute neg loss ---
    def _neg_loss():
        return torch.sigmoid(scale * (t - dist_neg)).mean()

    # --- NEW STABLE REGULARIZATION ---
    # 1. L2 penalty on m: Minimizing this keeps m small (away from infinity)
    reg_m = torch.square(m).sum()
    # 2. Inverse penalty on p: Minimizing this creates a barrier stopping p from reaching 0
    reg_p = (1.0 / (p + 1e-8)).sum() 
    reg_tot = reg_m + reg_p
    # --- Problem-specific loss ---
    if lam is not None:
        loss = (1-lam) * _neg_loss() + lam * _vol_loss() + reg_weight * reg_tot
    elif problem == 'vol':
        loss = _vol_loss() + reg_weight * reg_tot
    else:  # problem == 'neg'
        loss = _neg_loss() + reg_weight * reg_tot

    # --- Add InfoNCE when encoder is being trained ---
    if model.add_infonce and phase != 'metric':
        nce, _ = info_nce_loss(z_anchor, z_pos, z_neg, temperature)
        if phase == 'model':
            loss = contrastive_weight * nce + volume_weight * loss
        else:
            loss = loss + contrastive_weight * nce

    # --- Always compute log-volume for monitoring ---
    log_vol = compute_log_volume(model, t.detach())

    return loss, t.item(), cov, exc, log_vol
@torch.no_grad()
def conformal_evaluate(model, val_cache, args, sampler=None):
    """
    Proper conformal evaluation with calibration/test split.
    Splits the validation set in half: first half for calibrating the threshold,
    second half for measuring coverage and exclusion.
    Operates on cached backbone features (z_a, z_p, z_n) from cache_features.
    """
    model.eval()
    base = get_base_model(model)
    m, p = base.get_params()

    z_a_all, z_p_all, z_n_all = val_cache
    all_dist_pos = []
    all_dist_neg = []
    for i in range(0, z_a_all.shape[0], args.batch_size):
        f_a = z_a_all[i:i + args.batch_size].to(args.device)
        f_p = z_p_all[i:i + args.batch_size].to(args.device)
        f_n = z_n_all[i:i + args.batch_size].to(args.device)

        z_a = _head(base, f_a)
        z_p = _head(base, f_p)
        z_n = _head(base, f_n)

        dist_p = base.compute_distance(z_p, z_a, m, p)  # (N, K)
        dist_n = base.compute_distance(z_n, z_a, m, p)  # (N, K)

        all_dist_pos.append(dist_p.cpu())
        all_dist_neg.append(dist_n.cpu())

    all_dist_pos = torch.cat(all_dist_pos, dim=0)
    all_dist_neg = torch.cat(all_dist_neg, dim=0)

    # Random cal/test partition (seeded by args.seed). Required for exchangeability
    # when val is class-grouped (e.g. ImageFolder for ImageNet-100); a no-op for
    # already-shuffled splits like CIFAR-100 test, modulo per-seed permutation.
    gen = torch.Generator().manual_seed(int(getattr(args, 'seed', 42)))
    perm = torch.randperm(len(all_dist_pos), generator=gen)
    all_dist_pos = all_dist_pos[perm]
    all_dist_neg = all_dist_neg[perm]

    # Split into calibration (first half) and test (second half)
    n = len(all_dist_pos) // 2
    cal_dist_pos = all_dist_pos[:n].reshape(-1)
    test_dist_pos = all_dist_pos[n:]
    test_dist_neg = all_dist_neg[n:]

    # Calibrate threshold on calibration set using hard quantile
    threshold = compute_t_hard(cal_dist_pos, args.alpha).item()

    # Evaluate on test set
    coverage = (test_dist_pos <= threshold).float().mean().item()
    exclusion = (test_dist_neg > threshold).float().mean().item()

    # Compute log-volume at calibrated threshold
    log_vol = compute_log_volume(base, threshold)

    return threshold, coverage, exclusion, log_vol

def get_run_tag(args):
    """Short descriptive tag: e.g. 'single_vol_infonce' or 'generalized_vol'."""
    lam = getattr(args, 'lam', None)
    if lam is not None:
        tag = f"{args.metric}_neg_lam{lam}"
    else:
        tag = f"{args.metric}_{args.problem}"
    if args.add_infonce:
        tag += "_infonce"
    if args.note:
        tag += f"_{args.note}"
    return tag

def get_model_name(args, baseline=None):
    """Generate model filename based on args (or baseline name)."""
    base = (f"{args.dataset}_{args.model}_a{args.alpha}_k{args.k}_d{args.embed_dim}"
            f"_{args.sampling}_s{args.seed}_r{args.r}")

    if baseline is not None:
        return f"{baseline}_{base}"

    tag = get_run_tag(args)
    parts = f"{tag}_{base}_ep{args.epochs}_lr{args.lr}_reg{args.reg_w}"
    if args.add_infonce:
        parts += (f"_mlr{args.model_lr}_cw{args.contrastive_weight}_"
                  f"t{args.nce_temperature}_d{args.delta}_p{args.n_norms}")
        parts += "_alt" if args.alternating_training else "_joint"
    lam = getattr(args, 'lam', None)
    if lam is not None:
        parts += f"_sc{args.scale}"
    elif args.problem == 'neg':
        parts += f"_sc{args.scale}"
    return parts

def save_checkpoint(model, path):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_checkpoint(model, path, device):
    """Load model checkpoint. Returns True if successful."""
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        return True
    return False

def get_base_model(model):
    """Get underlying model from DataParallel wrapper."""
    return model.module if hasattr(model, 'module') else model

# --- Fix 1: frozen-encoder feature caching -----------------------------------
# The backbone is frozen, so its features never change across epochs. We embed
# every anchor / positive / negative ONCE (before the training loop) and then
# train on the cached features, applying only the (tiny) trainable head.

@torch.no_grad()
def cache_features(base, loader, args, sampler=None):
    """Embed anchors + positive/negative samples for a whole loader once.

    Returns backbone features (pre-projector) on CPU:
      z_a: (Ntot, D)   z_p: (Ntot, K, D)   z_n: (Ntot, K, D)
    BatchNorm runs in eval mode so the cached features are well-defined.
    To bound peak GPU memory (ImageNet packs N*K 224x224 images per batch),
    the frozen encoder is split across GPUs (DataParallel) and run in
    sub-batches of `enc_bs` images; positives are encoded and freed before
    negatives are generated.
    """
    base.eval()
    encoder = base.features
    if torch.cuda.device_count() > 1:
        encoder = torch.nn.DataParallel(base.features)
    enc_bs = getattr(args, 'enc_bs', 512)

    def _encode(x):
        """Encode (M,C,H,W) on device in sub-batches -> (M, D) CPU float."""
        outs = []
        for j in range(0, x.shape[0], enc_bs):
            chunk = x[j:j + enc_bs]
            outs.append(encoder(chunk.half()).view(chunk.shape[0], -1).float().cpu())
        return torch.cat(outs)

    za, zp, zn = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(args.device)
        za.append(_encode(imgs))
        pos = generate_samples(imgs, labels, args.k, sampler, positive=True,
                               replacement=args.with_replacement)
        N, K, C, H, W = pos.shape
        zp.append(_encode(pos.view(-1, C, H, W)).view(N, K, -1))
        del pos; torch.cuda.empty_cache()
        neg = generate_samples(imgs, labels, args.k, sampler, positive=False,
                               replacement=args.with_replacement)
        zn.append(_encode(neg.view(-1, C, H, W)).view(N, K, -1))
        del neg; torch.cuda.empty_cache()
        del imgs; torch.cuda.empty_cache()
    return torch.cat(za), torch.cat(zp), torch.cat(zn)

def _head(base, f):
    """Apply the trainable head to cached backbone features (identity if none)."""
    return base.projector(f) if base.add_infonce else f

def train_epoch_cached(base, cache, optimizer, scaler, args, phase='joint'):
    """One training epoch over cached features (no encoder forward)."""
    base.train()
    z_a_all, z_p_all, z_n_all = cache
    perm = torch.randperm(z_a_all.shape[0])
    total_loss, total_cov, total_exc, total_vol, n, t = 0, 0, 0, 0, 0, 0.0
    for i in range(0, len(perm), args.batch_size):
        idx = perm[i:i + args.batch_size]
        f_a = z_a_all[idx].to(args.device)
        f_p = z_p_all[idx].to(args.device)
        f_n = z_n_all[idx].to(args.device)

        with torch.amp.autocast('cuda', enabled=args.use_amp):
            loss, t, cov, exc, log_vol = compute_loss(
                _head(base, f_a), _head(base, f_p), _head(base, f_n), base,
                args.alpha, args.problem, phase, args.contrastive_weight,
                args.volume_weight, args.nce_temperature, args.scale,
                lam=getattr(args, 'lam', None),reg_weight=args.reg_w)

        if torch.isnan(loss):
            continue

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(base.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item(); total_cov += cov; total_exc += exc
        total_vol += log_vol; n += 1

    return (total_loss/n if n > 0 else 0, total_cov/n if n > 0 else 0,
            total_exc/n if n > 0 else 0, t, total_vol/n if n > 0 else 0)

@torch.no_grad()
def validate_cached(base, cache, args):
    """Validation over cached features (no encoder forward)."""
    base.eval()
    z_a_all, z_p_all, z_n_all = cache
    total_loss, total_cov, total_exc, total_vol, n = 0, 0, 0, 0, 0
    for i in range(0, z_a_all.shape[0], args.batch_size):
        f_a = z_a_all[i:i + args.batch_size].to(args.device)
        f_p = z_p_all[i:i + args.batch_size].to(args.device)
        f_n = z_n_all[i:i + args.batch_size].to(args.device)

        loss, t, cov, exc, log_vol = compute_loss(
            _head(base, f_a), _head(base, f_p), _head(base, f_n), base,
            args.alpha, args.problem, scale=args.scale,
            lam=getattr(args, 'lam', None),reg_weight=args.reg_w)

        if not torch.isnan(loss):
            total_loss += loss.item(); total_cov += cov; total_exc += exc
            total_vol += log_vol; n += 1

    return (total_loss/n if n > 0 else 0, total_cov/n if n > 0 else 0,
            total_exc/n if n > 0 else 0, total_vol/n if n > 0 else 0)

def train_model(model, train_cache, val_cache, args, sampler=None):
    """Full training loop with checkpoint support."""
    import copy
    base = get_base_model(model)
    tag = get_run_tag(args)
    epochs = args.epochs
    
    eval_mode = getattr(args, 'eval_mode', False)
    
    # Check for existing checkpoint
    model_name = get_model_name(args)
    ckpt_dir = os.path.join(args.save_dir, 'checkpoints', tag)
    ckpt_path = os.path.join(ckpt_dir, f'{model_name}.pt')

    if load_checkpoint(base, ckpt_path, args.device):
        print(f"Loaded checkpoint: {ckpt_path}")
        return None  # Signal that training was skipped

    # Set up optimizer
    def _metric_params():
        if base.metric == 'generalized':
            return [base.a_raw, base.p_raw]
        else:  # single
            return [base.A, base.p_raw]

    if base.add_infonce:
        model_params = list(base.projector.parameters())
        metric_params = _metric_params()
        param_groups = [
            {'params': model_params, 'lr': args.model_lr},
            {'params': metric_params, 'lr': args.lr}
        ]
    else:
        metric_params = _metric_params()
        param_groups = [{'params': metric_params, 'lr': args.lr}]

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(param_groups, weight_decay=5e-4)
    else:
        for pg in param_groups:
            pg['momentum'] = 0.9
        optimizer = torch.optim.SGD(param_groups, weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=args.use_amp)

    history = {'loss': [], 'val_loss': [], 't': [], 'cov': [], 'exc': [],
               'log_vol': [], 'val_log_vol': [], 'p': [], 'm': []}
    val_exc_track = 0.0
    val_logvol_track = float('inf')
    count = 0

    for epoch in range(epochs):
        count += 1
        # Determine phase for alternating training
        if base.add_infonce and args.alternating_training:
            if epoch % 2 == 0:
                phase = 'model'
                for p in metric_params: p.requires_grad = False
                for p in model_params: p.requires_grad = True
            else:
                phase = 'metric'
                for p in metric_params: p.requires_grad = True
                for p in model_params: p.requires_grad = False
        else:
            phase = 'joint'

        loss, cov, exc, t, log_vol = train_epoch_cached(base, train_cache, optimizer, scaler, args, phase)
        val_loss, val_cov, val_exc, val_log_vol = validate_cached(base, val_cache, args)
        scheduler.step()

        # Record history
        _, p = base.get_params()
        history['loss'].append(loss)
        history['val_loss'].append(val_loss)
        history['t'].append(t)
        history['cov'].append(val_cov)
        history['exc'].append(val_exc)
        history['log_vol'].append(log_vol)
        history['val_log_vol'].append(val_log_vol)
        history['p'].append(p.detach().cpu().numpy().copy() if p.dim() > 0 else p.item())
        if base.metric == 'generalized':
            history['m'].append((base.a_raw ** 2).detach().cpu().numpy().copy())

        p_str = f"[{p.min():.2f},{p.max():.2f}]" if p.dim() > 0 else f"{p.item():.2f}"
        phase_str = f"[{phase:6s}] " if base.add_infonce else ""
        print(f"Ep {epoch+1:02d}/{epochs} {phase_str}| L:{loss:.3f}/{val_loss:.3f} | "
              f"t:{t:.3f} | Cov:{val_cov:.1%} | Exc:{val_exc:.1%} | "
              f"Vol:{val_log_vol:.1f} | p:{p_str}")
        
        if not eval_mode:
            if val_exc_track < val_exc:
                val_exc_track = val_exc
                save_checkpoint(base, ckpt_path)
                print(f"Saved a new best checkpoint: {ckpt_path}")
                count = 0
            
            if val_exc < 0.001 or count >= 8:
                load_checkpoint(base, ckpt_path, args.device)
                return history
            
        elif eval_mode == 'ad':
            if val_logvol_track > val_log_vol:
                val_logvol_track = val_log_vol
                save_checkpoint(base, ckpt_path)
                print(f"Saved a new best checkpoint: {ckpt_path}")
                count = 0
            
            if count >= 8:
                load_checkpoint(base, ckpt_path, args.device)
                return history

    # load for testing
    load_checkpoint(base, ckpt_path, args.device)

    return history

# =============================================================================
# CONFORMAL BASELINES
# =============================================================================

def run_baseline(case, val_cache, args, sampler=None):
    """Run conformal baseline (0a: hyperball, 0b: hyperellipsoid)."""
    device = args.device
    alpha = args.alpha
    k = args.k

    # Feature extractor
    if args.dataset == 'cifar100':
        input_size = 32
    elif args.dataset == 'stl10':
        input_size = 96
    else:  # imagenet100
        input_size = 224
    features, embed_dim = build_encoder(args.model, input_size, pretrained=True)
    features = features.half().to(device)

    for p in features.parameters(): p.requires_grad = False

    z_a_all, z_p_all, z_n_all = val_cache

    all_z = z_a_all
    all_pos = z_p_all
    all_neg = z_n_all

    # Split calibration/test
    n = len(all_z) // 2
    cal_z, cal_pos = all_z[:n], all_pos[:n]
    test_z, test_pos, test_neg = all_z[n:], all_pos[n:], all_neg[n:]
    # Split calibration over 2 for covariance estimation (0b)
    cal_z_half_cov, cal_pos_half_cov = cal_z[:n//2], cal_pos[:n//2]
    cal_z_0b, cal_pos_0b = cal_z[n//2:n], cal_pos[n//2:n]

    cov_inv = None
    if case == '0a':
        # L2 distance
        cal_dist = torch.norm(cal_pos - cal_z.unsqueeze(1), dim=-1).view(-1)
        threshold = torch.sort(cal_dist)[0][int((1-alpha) * len(cal_dist))].item()
        test_pos_dist = torch.norm(test_pos - test_z.unsqueeze(1), dim=-1)
        test_neg_dist = torch.norm(test_neg - test_z.unsqueeze(1), dim=-1)
    elif case == '0b':  # 0b
        # Mahalanobis distance
        diffs = (cal_pos_0b - cal_z_0b.unsqueeze(1)).view(-1, embed_dim)
        diffs_cov = (cal_pos_half_cov - cal_z_half_cov.unsqueeze(1)).view(-1, embed_dim)
        cov = diffs_cov.T @ diffs_cov / len(diffs_cov)
        cov_inv = torch.linalg.inv(cov + 1e-4 * torch.eye(embed_dim))
        cal_dist = torch.sqrt((diffs @ cov_inv * diffs).sum(-1))
        threshold = torch.sort(cal_dist)[0][int((1-alpha) * len(cal_dist))].item()

        test_pos_diff = (test_pos - test_z.unsqueeze(1)).view(-1, embed_dim)
        test_neg_diff = (test_neg - test_z.unsqueeze(1)).view(-1, embed_dim)
        test_pos_dist = torch.sqrt((test_pos_diff @ cov_inv * test_pos_diff).sum(-1)).view(test_pos.shape[:2])
        test_neg_dist = torch.sqrt((test_neg_diff @ cov_inv * test_neg_diff).sum(-1)).view(test_neg.shape[:2])

    elif 'hull' in case.lower():
        from scipy.spatial import ConvexHull
        print(f"Running method of convex hull shrinkage.")
        from scipy.spatial import Delaunay

        # Exact hulls are impossible in backbone dimension: qhull needs >= d+1
        # points and k//2 points span a zero-volume polytope in 512/1408-D
        # (every anchor raised -> all-zero results). Project to a low-dim PCA
        # basis fitted on CALIBRATION positives (no test leakage) and do the
        # hull + shrinkage + membership tests there. qhull is practical <= ~8D.
        hull_dim = min(int(getattr(args, 'hull_dim', 8)), max(2, k // 2 - 2))
        X_fit = cal_pos.reshape(-1, embed_dim).float()
        if len(X_fit) > 20000:
            X_fit = X_fit[torch.randperm(len(X_fit))[:20000]]
        pca_mean = X_fit.mean(0, keepdim=True)
        _, _, V = torch.pca_lowrank(X_fit - pca_mean, q=hull_dim, niter=4)

        def _proj(t3):  # (N, K, D) -> (N, K, hull_dim) numpy
            N_, K_, _ = t3.shape
            flat = (t3.reshape(-1, embed_dim).float() - pca_mean) @ V
            return flat.reshape(N_, K_, hull_dim).cpu().numpy()

        print(f"  Hull in {hull_dim}-D PCA space (fit on calibration positives).")
        z_p_test_np = _proj(test_pos)  # (N_test, k, hull_dim)
        z_n_test_np = _proj(test_neg)  # (N_test, k, hull_dim)

        coverages = []
        exclusions = []
        target = 1.0 - args.alpha
        stride = 0.01

        for i in range(z_p_test_np.shape[0]):
            pos = z_p_test_np[i,:args.k//2]  # (k, d)
            pos_cov = z_p_test_np[i]
            neg = z_n_test_np[i]  # (k, d)

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
            shrink_factor = 1.0
            while shrink_factor > 0:
                shrunk = centroid + shrink_factor * (hull_pts - centroid)
                try:
                    delaunay = Delaunay(shrunk)
                except Exception:
                    break
                inside_pos = delaunay.find_simplex(pos) >= 0
                cov = float(inside_pos.mean())
                if cov < target:
                    break
                inside_neg = delaunay.find_simplex(neg) >= 0
                inside_pos = delaunay.find_simplex(pos_cov) >= 0
                cov = float(inside_pos.mean())
                last_cov = cov
                last_exc = float((~inside_neg).mean())
                shrink_factor -= stride

            if last_cov is None:
                continue
            coverages.append(last_cov)
            exclusions.append(last_exc)

        coverage = float(np.mean(coverages)) if coverages else 0.0
        exclusion = float(np.mean(exclusions)) if exclusions else 0.0
        log_vol = 0.0  # not a distance/volume-based method
        threshold = 0.0  # placeholder; expected by downstream code

        print(f"  Baseline {case}: Cov={coverage:.1%}, "
            f"Exc={exclusion:.1%}, Normalised Vol={log_vol:.2f}")

        return {'threshold': threshold, 'coverage': coverage, 'exclusion': exclusion,
                'conformal_log_volume': log_vol,
                'cov_inv': None}
    
    elif 'mog' in case.lower():
        # HIB MoG variant (Sec 2.2 of the HIB paper): fit a C-component
        # diagonal-covariance Gaussian Mixture to each test point's k positives
        # in embedding space. The set predictor is the union of per-component
        # k-sigma ellipsoids:
        #     inside(z)  <=>  min_c (z - mu_c)^T Sigma_c^{-1} (z - mu_c) <= k^2.
        from sklearn.mixture import GaussianMixture
        from scipy.special import logsumexp
        import scipy.stats as scst
        d = embed_dim
        C = 2  # paper uses C=2
        sigma_factor = math.sqrt(scst.chi2.ppf(1 - args.alpha, d))
        thresh_sq = sigma_factor ** 2
        print(f"Running HIB MoG (C={C}) ellipsoid baseline.")

        z_p_test_np = test_pos.cpu().numpy()  # (N_test, k, d)
        z_n_test_np = test_neg.cpu().numpy()
        k_per = z_p_test_np.shape[1]

        coverages, exclusions, log_vols = [], [], []
        log_unit_ball = (d / 2) * math.log(math.pi) - math.lgamma(d / 2 + 1)

        for i in range(z_p_test_np.shape[0]):
            pos = z_p_test_np[i]  # (k, d)
            neg = z_n_test_np[i]  # (k, d)

            # Fit GMM; fall back to a single diagonal Gaussian (anchored at
            # test_z[i] to stay consistent with the single-Gaussian HIB case)
            # when k is too small or EM fails.
            use_mog = k_per >= 2 * C
            if use_mog:
                try:
                    gmm = GaussianMixture(n_components=C,
                                          covariance_type='diag',
                                          reg_covar=1e-4,
                                          n_init=1,
                                          random_state=0).fit(pos)
                    mu = gmm.means_                              # (C, d)
                    var = np.maximum(gmm.covariances_, 1e-8)     # (C, d)
                except Exception:
                    use_mog = False
            if not use_mog:
                mu = test_z[i].cpu().numpy()[None, :]            # (1, d)
                var = np.maximum(pos.var(axis=0, ddof=1,
                                         keepdims=True), 1e-8)   # (1, d)

            # Per-component Mahalanobis^2, take the min over components.
            diff_p = pos[:, None, :] - mu[None, :, :]               # (k, C', d)
            d2_p = ((diff_p ** 2) / var[None, :, :]).sum(-1).min(-1)
            diff_n = neg[:, None, :] - mu[None, :, :]
            d2_n = ((diff_n ** 2) / var[None, :, :]).sum(-1).min(-1)

            coverages.append(float((d2_p <= thresh_sq).mean()))
            exclusions.append(float((d2_n > thresh_sq).mean()))

            # Union of ellipsoids: log V_total <= logsumexp_c(log V_c).
            per_comp_logvol = (log_unit_ball
                               + 0.5 * np.log(var).sum(axis=1)
                               + d * math.log(sigma_factor + 1e-8))
            log_vols.append(float(logsumexp(per_comp_logvol)))

        coverage = float(np.mean(coverages)) if coverages else 0.0
        exclusion = float(np.mean(exclusions)) if exclusions else 0.0
        log_vol = float(np.mean(log_vols)) if log_vols else 0.0
        threshold = 0.0  # placeholder; MoG uses per-point mixture ellipsoids

        print(f"  Baseline {case}: Cov={coverage:.1%}, "
              f"Exc={exclusion:.1%}, Normalised Vol={log_vol:.2f}")

        return {'threshold': threshold, 'coverage': coverage,
                'exclusion': exclusion, 'conformal_log_volume': log_vol,
                'cov_inv': None}

    else: #HIB
        # HIB models each input as Z ~ N(mu(x), Sigma(x)) with diagonal Sigma
        # (in the paper, mu and Sigma are produced by a CNN).
        # We instantiate the same idea by fitting a
        # per-anchor diagonal Gaussian directly to the test point's positives
        # (mean = mu, sample variance = Sigma). The set predictor is the
        # k-sigma ellipsoid {z : (z-mu)^T Sigma^{-1} (z-mu) <= k^2}.
        d = embed_dim
        import scipy.stats as scst
        print(f"Running HIB {2.0:.0f}-sigma ellipsoid baseline.")
        sigma_factor = math.sqrt(scst.chi2.ppf(1 - args.alpha, d))
        thresh_sq = sigma_factor ** 2

        z_p_test_np = test_pos.cpu().numpy()  # (N_test, k, d)
        z_n_test_np = test_neg.cpu().numpy()  # (N_test, k, d)

        coverages = []
        exclusions = []
        log_vols = []
        log_unit_ball = (d / 2) * math.log(math.pi) - math.lgamma(d / 2 + 1)

        for i in range(z_p_test_np.shape[0]):
            pos = z_p_test_np[i]  # (k, d)
            neg = z_n_test_np[i]  # (k, d)

            mu = pos.mean(axis=0) #test_z[i].cpu().numpy()[None, :] #pos.mean(axis=0)
            var = np.maximum(pos.var(axis=0, ddof=1), 1e-8)
            pos = z_p_test_np[i] # (k, d)
            diff_p = pos - mu
            d2_p = ((diff_p ** 2) / var).sum(axis=1)
            diff_n = neg - mu
            d2_n = ((diff_n ** 2) / var).sum(axis=1)

            coverages.append(float((d2_p <= thresh_sq).mean()))
            exclusions.append(float((d2_n > thresh_sq).mean()))

            # Vol of k-sigma ellipsoid = V_d * sqrt(det Sigma) * k^d.
            log_vols.append(log_unit_ball
                            + 0.5 * np.log(var).sum()
                            + d * math.log(sigma_factor + 1e-8))

        coverage = float(np.mean(coverages)) if coverages else 0.0
        exclusion = float(np.mean(exclusions)) if exclusions else 0.0
        log_vol = float(np.mean(log_vols)) if log_vols else 0.0
        threshold = 0.0  # placeholder; HIB uses a per-point ellipsoid

        print(f"  Baseline {case}: Cov={coverage:.1%}, "
              f"Exc={exclusion:.1%}, Normalised Vol={log_vol:.2f}")

        return {'threshold': threshold, 'coverage': coverage,
                'exclusion': exclusion, 'conformal_log_volume': log_vol,
                'cov_inv': None}


    coverage = (test_pos_dist <= threshold).float().mean().item()
    exclusion = (test_neg_dist > threshold).float().mean().item()

    # Log-volume of the covering set
    d = embed_dim
    if case == '0a':
        # L2 ball: V = π^(d/2) / Γ(d/2+1) · t^d
        log_vol = (d/2) * math.log(math.pi) - math.lgamma(d/2 + 1) + d * math.log(threshold + 1e-8)
    else:  # 0b — Mahalanobis ellipsoid: L2 ball volume × sqrt(det(Σ))
        log_det_cov = -torch.linalg.slogdet(cov_inv)[1].item()  # log det(Σ) = -log det(Σ^{-1})
        log_vol = ((d/2) * math.log(math.pi) - math.lgamma(d/2 + 1)
                   + d * math.log(threshold + 1e-8) + 0.5 * log_det_cov)

    print(f"Baseline {case}: threshold={threshold:.4f}, Coverage={coverage:.1%}, "
          f"Exclusion={exclusion:.1%}, Normalised Vol={log_vol:.1f}")

    return {'threshold': threshold, 'coverage': coverage, 'exclusion': exclusion,
            'conformal_log_volume': log_vol}