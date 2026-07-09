#!/usr/bin/env python3
"""CCS: Multi-seed evaluation with LaTeX table output."""
import os
# Optionally pin/reorder GPUs before importing torch, e.g.:
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import argparse
import json
import torch
import numpy as np
from types import SimpleNamespace

from utils import (
    get_loaders, LabelSampler, CCSModel,
    train_model, run_baseline, set_seed,
    get_feat_dim, get_base_model,
    conformal_evaluate, cache_features,
    get_run_tag, get_model_name
)

def parse_args():
    parser = argparse.ArgumentParser(description='CCS Multi-Seed Table Generator')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config JSON')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='List of random seeds')
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--num_gpus', type=int, default=-1)
    # ImageNet100 directories (only needed when dataset == 'imagenet100');
    # point these to your local ImageNet100 train/val class folders.
    parser.add_argument('--train_dir', type=str, default='path/to/imagenet-100/train')
    parser.add_argument('--val_dir', type=str, default='path/to/imagenet-100/val')
    parser.add_argument('--note', type=str, help="Notes for the file names")

    return parser.parse_args()


def build_args(shared, method, seed, cli_args):
    """Build a full args namespace for a single method + seed run."""
    args = SimpleNamespace()

    # Shared params
    for k, v in shared.items():
        setattr(args, k, v)

    # Method-specific overrides
    for k, v in method.items():
        if k not in ('name', 'type', 'case'):
            setattr(args, k, v)

    # Defaults for fields that may not be in config
    defaults = {
        'epochs': 50, 'lr': 0.01, 'model_lr': 0.001,
        'alternating_training': False, 'add_infonce': False,
        'metric': None, 'problem': 'vol',
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    # Per-run
    args.seed = seed
    args.save_dir = cli_args.save_dir
    args.train_dir = cli_args.train_dir
    args.val_dir = cli_args.val_dir
    args.num_gpus = cli_args.num_gpus
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.embed_dim = get_feat_dim(args.model)
    args.note = cli_args.note

    return args


def run_single(method, model, args, train_cache, val_cache, sampler=None):
    """Run a single method and return (threshold, coverage, exclusion, log_vol)."""
    if method['type'] == 'baseline':
        res = run_baseline(method['case'], val_cache, args, sampler)
        return (res['threshold'], res['coverage'], res['exclusion'],
                res['conformal_log_volume'])

    if args.num_gpus > 1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    train_model(model, train_cache, val_cache, args, sampler)
    conf_t, conf_cov, conf_exc, conf_vol = conformal_evaluate(
        model, val_cache, args, sampler)

    # Free GPU memory
    del model
    torch.cuda.empty_cache()
    # Report the log-volume normalized by the feature dimension (LogVol/d)
    return conf_t, conf_cov, conf_exc, conf_vol / args.embed_dim


def generate_latex(all_results, shared, seeds, val_size):
    """Generate LaTeX table string from aggregated results."""
    n_test = val_size // 2  # conformal_evaluate uses second half

    lines = []
    lines.append(r"\begin{table}[h!]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{{shared['dataset'].upper()}, "
        rf"model={shared['model']}, "
        rf"$k={shared['k']}$, "
        rf"$\alpha={shared['alpha']}$, "
        rf"$n_\mathrm{{test}}={n_test}$, "
        rf"seeds={list(seeds)}.}}"
    )
    lines.append(r"\tiny")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Method} & \textbf{Threshold} & \textbf{Coverage} "
                 r"& \textbf{Exclusion} & \textbf{LogVol} \\")
    lines.append(r"\midrule")

    for name, vals in all_results.items():
        if not vals['threshold']:
            lines.append(rf"{name} & \multicolumn{{4}}{{c}}{{failed}} \\")
            continue
        t_arr = np.array(vals['threshold'])
        c_arr = np.array(vals['coverage']) * 100
        e_arr = np.array(vals['exclusion']) * 100
        v_arr = np.array(vals['log_vol'])
        newline = "\n"
        def fmt(arr, prec=2):
            return rf"${arr.mean():.{prec}f} \pm {arr.std():.{prec}f}$"

        lines.append(
            rf"{name} {newline}& {fmt(t_arr, 4)} {newline}& {fmt(c_arr, 1)}\% "
            rf"{newline}& {fmt(e_arr, 1)}\% {newline}& {fmt(v_arr, 2)} \\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


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

    print(f"Dataset: {shared['dataset']} | Model: {shared['model']} | "
          f"Seeds: {seeds} | Methods: {len(methods)}")
    print("=" * 70)

    # Collect results: {method_name: {threshold: [], coverage: [], ...}}
    all_results = {m['name']: {'threshold': [], 'coverage': [],
                               'exclusion': [], 'log_vol': []}
                   for m in methods}

    val_size = None  # will be set on first data load

    for si, seed in enumerate(seeds):
        print(f"\n{'#'*70}")
        print(f"# SEED {seed} ({si+1}/{len(seeds)})")
        print(f"{'#'*70}")
        set_seed(seed)

        # Fresh data loaders per seed (subsampling depends on seed)
        train_loader, val_loader = get_loaders(
            shared['dataset'], shared['batch_size'], shared['num_workers'],
            cli_args.train_dir, cli_args.val_dir, r=shared['r'], model_name=shared['model'], cal_use=False
        )
        print("Loaders built!\n")
        if val_size is None:
            val_size = len(val_loader.dataset)

        sampler_train = None
        sampler_val = None
        if shared['sampling'] == 'label':
            sampler_train = LabelSampler(train_loader.dataset)
            sampler_val = LabelSampler(val_loader.dataset)
        print("Samplers built!\n")
        caches_built = False
        for mi, method in enumerate(methods):
            name = method['name']
            print(f"\n--- [{si+1}/{len(seeds)}] [{mi+1}/{len(methods)}] "
                  f"{name} (seed={seed}) ---")

            # Build args and set seed BEFORE anything else
            args = build_args(shared, method, seed, cli_args)
            set_seed(seed)
            # Metric method — fresh model
            input_size = 32 if args.dataset == 'cifar100' else 224
            model = CCSModel(
                metric=args.metric,
                add_infonce=args.add_infonce,
                n_norms=args.n_norms,
                input_size=input_size,
                delta=args.delta,
                model_name=args.model
            ).to(args.device)

            if not caches_built:
                train_cache = None  # built lazily below, only if a metric method must train
                val_cache = cache_features(model, val_loader, args, sampler_val)
                print("Validation cached!\n")
                caches_built = True

            # Train cache is only needed when a metric method has no checkpoint yet
            if method['type'] == 'metric' and train_cache is None:
                ckpt_path = os.path.join(args.save_dir, 'checkpoints',
                                         get_run_tag(args),
                                         f'{get_model_name(args)}.pt')
                if not os.path.exists(ckpt_path):
                    train_cache = cache_features(model, train_loader, args, sampler_train)
                    print("Train cached!\n")

            # Run
            try:
                t, cov, exc, vol = run_single(method, model, args, train_cache, val_cache)
            except Exception as e:
                print(f"  !! FAILED: {e} — skipping this run")
                torch.cuda.empty_cache()
                continue

            all_results[name]['threshold'].append(t)
            all_results[name]['coverage'].append(cov)
            all_results[name]['exclusion'].append(exc)
            all_results[name]['log_vol'].append(vol)

            print(f"  -> t={t:.4f}, Cov={cov:.1%}, Exc={exc:.1%}, LogVol={vol:.1f}")

    # Print summary
    print(f"\n{'='*70}")
    print("AGGREGATED RESULTS (mean ± std)")
    print(f"{'='*70}")
    for name, vals in all_results.items():
        if not vals['threshold']:
            print(f"{name:35s} | NO SUCCESSFUL RUNS")
            continue
        t_m, t_s = np.mean(vals['threshold']), np.std(vals['threshold'])
        c_m, c_s = np.mean(vals['coverage'])*100, np.std(vals['coverage'])*100
        e_m, e_s = np.mean(vals['exclusion'])*100, np.std(vals['exclusion'])*100
        v_m, v_s = np.mean(vals['log_vol']), np.std(vals['log_vol'])
        print(f"{name:35s} | t={t_m:.4f}±{t_s:.4f} | "
              f"Cov={c_m:.1f}±{c_s:.1f}% | Exc={e_m:.1f}±{e_s:.1f}% | "
              f"Vol={v_m:.1f}±{v_s:.1f}")

    # Generate LaTeX
    latex = generate_latex(all_results, shared, seeds, val_size or 10000)
    print(f"\n{'='*70}")
    print("LATEX TABLE")
    print(f"{'='*70}")
    print(latex)

    # Save
    out_path = os.path.join(cli_args.save_dir, f'table_results_{args.dataset}_{args.model}.json')
    with open(out_path, 'w') as f:
        json.dump({'seeds': seeds, 'shared': shared, 'results': all_results},
                  f, indent=2, default=str)

    tex_path = os.path.join(cli_args.save_dir, f'table_{args.dataset}_{args.model}_{seeds}seeds_{args.note}_k{args.k}_r{args.r}.tex')
    with open(tex_path, 'w') as f:
        f.write(latex)

    print(f"\nSaved results to {out_path}")
    print(f"Saved LaTeX table to {tex_path}")


if __name__ == '__main__':
    main()