#!/usr/bin/env python3
"""CCS: Sensitivity analysis evaluation with LaTeX friendly output."""
import os
import argparse
import json
import torch
import numpy as np
from types import SimpleNamespace

from utils import (
    get_loaders, LabelSampler, CCSModel,
    train_model, set_seed, get_feat_dim,
    conformal_evaluate, cache_features
)

def parse_args():
    parser = argparse.ArgumentParser(description='CCS Sensitivity Analysis')
    parser.add_argument('--config', type=str, default='configs/config_c_rep_a005_sensitivity.json', help='Path to config JSON')
    # Default to a single seed for sensitivity analysis to save time, but allow multiple
    parser.add_argument('--seeds', type=int, nargs='+', default=[42], help='List of random seeds')
    parser.add_argument('--save_dir', type=str, default='./results_sensitivity')
    parser.add_argument('--num_gpus', type=int, default=-1)
    # ImageNet100 directories (only needed when dataset == 'imagenet100');
    # point these to your local ImageNet100 train/val folders.
    parser.add_argument('--train_dir', type=str, default='path/to/imagenet-100/train')
    parser.add_argument('--val_dir', type=str, default='path/to/imagenet-100/val')
    return parser.parse_args()

def build_args(shared, method, seed, cli_args, override_params):
    """Build a full args namespace for a single method + seed run with parameter overrides."""
    args = SimpleNamespace()

    # 1. Load shared params
    for k, v in shared.items():
        setattr(args, k, v)

    # 2. Load Method-specific params
    for k, v in method.items():
        if k not in ('name', 'type', 'case'):
            setattr(args, k, v)

    # 3. Apply defaults for fields that may not be in config
    defaults = {
        'epochs': 5, 'train_epochs': 35, 'lr': 0.01, 'model_lr': 0.001,
        'alternating_training': False, 'add_infonce': False,
        'metric': None, 'problem': 'vol', 'r': 1.0
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    # 4. Apply Sensitivity Overrides (The specific param we are sweeping + fixed defaults)
    for k, v in override_params.items():
        setattr(args, k, v)

    # 5. Per-run setup
    args.seed = seed
    args.note = 'sensitivity'  # tag used in checkpoint/table file names
    args.save_dir = cli_args.save_dir
    args.train_dir = cli_args.train_dir
    args.val_dir = cli_args.val_dir
    args.num_gpus = cli_args.num_gpus
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.embed_dim = get_feat_dim(args.model)

    return args

def run_single(args, train_loader, val_loader, sampler):
    """Run the method and return (coverage, exclusion, log_vol)."""
    input_size = 32 if args.dataset == 'cifar100' else 224

    model = CCSModel(
        metric=args.metric,
        add_infonce=args.add_infonce,
        n_norms=args.n_norms,
        input_size=input_size,
        delta=args.delta,
        model_name=args.model
    ).to(args.device)

    if args.num_gpus > 1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Pre-extract frozen-backbone features (depends on k and seed, so rebuilt per run)
    train_cache = cache_features(model, train_loader, args, sampler)
    val_cache = cache_features(model, val_loader, args, sampler)

    # Train the metric parameters on the cached features
    train_model(model, train_cache, val_cache, args, sampler)

    # Evaluate the conformal covering set
    conf_t, conf_cov, conf_exc, conf_vol = conformal_evaluate(model, val_cache, args, sampler)

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return conf_cov, conf_exc, conf_vol

def main():
    cli_args = parse_args()

    with open(cli_args.config) as f:
        config = json.load(f)

    shared = config['shared']
    methods = config['methods']
    seeds = cli_args.seeds

    if cli_args.num_gpus < 0:
        cli_args.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    os.makedirs(cli_args.save_dir, exist_ok=True)

    # Find the target method
    target_method_name = "Generalized_Neg_vol_infonce"
    method = next((m for m in methods if m['name'] == target_method_name), None)
    if not method:
        raise ValueError(f"Method '{target_method_name}' not found in the config file.")

    # Define the parameter sweep grid
    sweep_grid = {
        'alpha': [0.05, 0.1, 0.15, 0.2],
        'scale': [1, 4, 7, 10],
        'contrastive_weight': [0.1, 0.5, 1.0],
        'nce_temperature': [0.001, 0.01, 0.1],
        'lam': [0.005, 0.05, 0.5],
        'k': [1, 5, 10, 15, 20]
    }

    # Define the fixed default base parameters
    base_defaults = {
        'alpha': 0.05,
        'scale': 7.0,
        'contrastive_weight': 1.0,
        'nce_temperature': 0.1,
        'lam': 0.05,
        'k': 20
    }

    print("=" * 70)
    print(f"Starting Sensitivity Analysis for: {target_method_name}")
    print(f"Base Defaults: {base_defaults}")
    print(f"Seeds: {seeds}")
    print("=" * 70)

    # Structure to hold results
    # results[param_name][param_value] = {'cov': [], 'exc': [], 'vol': []}
    results = {param: {val: {'cov': [], 'exc': [], 'vol': []} for val in vals} for param, vals in sweep_grid.items()}

    # Load data ONCE for efficiency (since none of the sweep params alter the base dataloader structure)
    # Note: 'k' alters generate_samples during training, not the dataloader batching
    dummy_args = build_args(shared, method, seeds[0], cli_args, base_defaults)
    train_loader, val_loader = get_loaders(
        dummy_args.dataset, dummy_args.batch_size, dummy_args.num_workers,
        dummy_args.train_dir, dummy_args.val_dir, r=dummy_args.r, model_name=dummy_args.model
    )
    sampler = LabelSampler(train_loader.dataset) if dummy_args.sampling == 'label' else None

    # Run the sweeps
    for param_name, values_to_sweep in sweep_grid.items():
        print(f"\n>>> Sweeping {param_name} over {values_to_sweep}")
        
        for val in values_to_sweep:
            for seed in seeds:
                print(f"  -> Testing {param_name} = {val} (Seed: {seed})")
                
                # Build override dictionary: start with defaults, override the active sweep parameter
                current_overrides = base_defaults.copy()
                current_overrides[param_name] = val
                
                # Build args and set seed
                args = build_args(shared, method, seed, cli_args, current_overrides)
                set_seed(seed)
                
                try:
                    cov, exc, vol = run_single(args, train_loader, val_loader, sampler)
                except Exception as e:
                    print(f"  !! FAILED for {param_name}={val}: {e} — skipping")
                    torch.cuda.empty_cache()
                    continue
                
                results[param_name][val]['cov'].append(cov)
                results[param_name][val]['exc'].append(exc)
                results[param_name][val]['vol'].append(vol)

    # ---------------------------------------------------------
    # Generate LaTeX Friendly Output 
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS RESULTS (LaTeX Friendly)")
    print("=" * 70)

    # Output formatting logic
    alias_map = {
        'alpha': 'a',
        'scale': 'scale',
        'contrastive_weight': 'cw',
        'nce_temperature': 'temp',
        'lam': 'lam',
        'k': 'k'
    }

    for param_name, values in sweep_grid.items():
        alias = alias_map[param_name]
        
        # --- Coverage ---
        print(f"{alias:<10} coverage")
        for val in values:
            if results[param_name][val]['cov']:
                mean_val = np.mean(results[param_name][val]['cov'])
                print(f"{val:<10.3g} {mean_val:.4f}")
        
        # --- Exclusion ---
        print(f"{alias:<10} exclusion")
        for val in values:
            if results[param_name][val]['exc']:
                mean_val = np.mean(results[param_name][val]['exc'])
                print(f"{val:<10.3g} {mean_val:.4f}")

        # --- LogVol ---
        print(f"{alias:<10} logvol")
        for val in values:
            if results[param_name][val]['vol']:
                mean_val = np.mean(results[param_name][val]['vol'])
                print(f"{val:<10.3g} {mean_val:.4f}")
        
        print("-" * 30)

if __name__ == '__main__':
    main()