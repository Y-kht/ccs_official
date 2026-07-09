#!/usr/bin/env python3
"""CCS Simulation: Multi-seed evaluation with LaTeX table output."""
import argparse
import os
# Optionally pin a GPU before importing torch, e.g.:
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import torch
import numpy as np
from types import SimpleNamespace

from simulation_utils import (
    set_seed, generate_clusters, SimLabelSampler, SimMetricModel,
    train_model, conformal_evaluate, run_baseline,
    plot_clusters, plot_covering_set_paper
)


def parse_args():
    parser = argparse.ArgumentParser(description='CCS Simulation Table Generator')
    parser.add_argument('--config', type=str, default='config_sim.json',
                        help='Path to simulation config JSON')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                        help='List of random seeds')
    parser.add_argument('--save_dir', type=str, default='./results_sim')
    parser.add_argument('--note', type=str, help="Notes for the file names")

    return parser.parse_args()


def build_args(shared, method, seed):
    """Build a full args namespace for a single method + seed run."""
    args = SimpleNamespace()
    for k, v in shared.items():
        setattr(args, k, v)
    for k, v in method.items():
        if k not in ('name', 'type', 'case'):
            setattr(args, k, v)
    # Defaults
    defaults = {
        'epochs': 30, 'lr': 0.1, 'optimizer': 'sgd', 'grad_clip': 1.0,
        'metric': None, 'problem': 'vol', 'scale': 7.0, 'n_norms': 1,
        'p_train': 1.0, 'shift': False, 'plot': False,
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    # Test-half positive sampling defaults to the train/calibration p
    if not hasattr(args, 'p_test'):
        args.p_test = args.p_train
    args.seed = seed
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return args


def run_single(method, args, z_train, labels_train, z_val, labels_val,
               sampler_train, sampler_val, z_all=None, labels_all=None,
               plot_dir=None, plot_test_idx=None):
    """Run a single method. Returns (threshold, coverage, exclusion, log_vol).
    
    If plot_dir is not None, generates a covering-set plot for the given test point.
    """
    if method['type'] == 'baseline':
        res = run_baseline(method['case'], z_val, labels_val, sampler_val, args)
        t = res['threshold']
        cov_bl = res['coverage']
        exc_bl = res['exclusion']
        vol_bl = res['conformal_log_volume']
        cc_bl = res.get('cov_class', float('nan'))
        return t, cov_bl, exc_bl, vol_bl, cc_bl

    # Metric method — fresh model
    model = SimMetricModel(
        metric=args.metric, d=z_train.shape[1], n_norms=args.n_norms
    ).to(args.device)

    train_model(model, z_train, labels_train, z_val, labels_val,
                sampler_train, sampler_val, args)

    conf_t, conf_cov, conf_exc, conf_vol, conf_cc = conformal_evaluate(
        model, z_val, labels_val, sampler_val, args)

    # Exact covering-set figure (config flag "plot"), produced for one fixed
    # random test anchor on the first seed
    if (getattr(args, 'plot', False) and plot_dir is not None
            and z_all is not None and plot_test_idx is not None):
        anchor = z_val[plot_test_idx]
        anchor_label = labels_val[plot_test_idx]
        sampler = SimLabelSampler(z_val, labels_val,
                                  p_same=getattr(args, 'p_test', 1.0))
        n_show = 10  # number of positives/negatives shown in the figure
        z_pos_m = sampler.sample(anchor_label.unsqueeze(0), n_show,
                                 same_class=True)[0]
        z_neg_m = sampler.sample(anchor_label.unsqueeze(0), n_show,
                                 same_class=False)[0]
        safe_name = method['name'].replace(' ', '_').replace('$', '').replace('\\', '')
        path = plot_covering_set_paper(
            model, anchor, z_pos_m, z_neg_m, conf_t, z_all, labels_all,
            method_name=method['name'],
            save_path=os.path.join(plot_dir, f'covering_{safe_name}.png'))
        if path:
            print(f"  Saved covering-set figure: {path}")

    del model
    torch.cuda.empty_cache()
    return conf_t, conf_cov, conf_exc, conf_vol, conf_cc


def generate_latex(all_results, shared, seeds, n_test):
    """Generate LaTeX table string."""
    lines = []
    lines.append(r"\begin{table}[h!]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{Simulated {shared['d']}D clusters ({shared.get('distribution', 'isotropic')}): "
        rf"$n_\mathrm{{clusters}}={shared['n_clusters']}$, "
        rf"$n_\mathrm{{per\_cluster}}={shared['n_per_cluster']}$, "
        rf"separation$={shared['separation']}$, "
        rf"$k={shared['k']}$, "
        rf"$\alpha={shared['alpha']}$, "
        rf"$p_\mathrm{{train}}={shared.get('p_train', 1.0)}$, "
        rf"$p_\mathrm{{test}}={shared.get('p_test', shared.get('p_train', 1.0))}$, "
        rf"$n_\mathrm{{test}}={n_test}$, "
        rf"seeds={list(seeds)}.}}"
    )
    shift = shared.get('shift', False)
    n_cols = 5 if shift else 4
    lines.append(r"\tiny")
    lines.append(r"\begin{tabular}{l" + "c" * n_cols + r"}")
    lines.append(r"\toprule")
    header = r"\textbf{Method} & \textbf{Threshold} & \textbf{Coverage} "
    if shift:
        header += r"& \textbf{Cov\_class} "
    header += r"& \textbf{Exclusion} & \textbf{LogVol} \\"
    lines.append(header)
    lines.append(r"\midrule")
    newline = "\n"
    for name, vals in all_results.items():
        if not vals['threshold']:
            lines.append(rf"{name} & \multicolumn{{{n_cols}}}{{c}}{{failed}} \\")
            continue
        t_arr = np.array(vals['threshold'])
        c_arr = np.array(vals['coverage']) * 100
        e_arr = np.array(vals['exclusion']) * 100
        v_arr = np.array(vals['log_vol'])

        def fmt(arr, prec=2):
            return rf"${arr.mean():.{prec}f} \pm {arr.std():.{prec}f}$"

        row = (rf"{name} {newline}& {fmt(t_arr, 4)} {newline}& {fmt(c_arr, 1)}\% ")
        if shift:
            cc_arr = np.array(vals['cov_class'], dtype=float) * 100
            row += (rf"{newline}& ${np.nanmean(cc_arr):.1f} \pm "
                    rf"{np.nanstd(cc_arr):.1f}$\% ")
        row += rf"{newline}& {fmt(e_arr, 1)}\% {newline}& {fmt(v_arr/3, 2)} \\"
        lines.append(row)

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

    os.makedirs(cli_args.save_dir, exist_ok=True)

    p_train_cfg = shared.get('p_train', 1.0)
    p_test_cfg = shared.get('p_test', p_train_cfg)
    print(f"Simulation: {shared['n_clusters']} clusters × "
          f"{shared['n_per_cluster']} pts, d={shared['d']}, "
          f"separation={shared['separation']}, "
          f"distribution={shared.get('distribution', 'isotropic')}, "
          f"p_train={p_train_cfg}, p_test={p_test_cfg}")
    print(f"Seeds: {seeds} | Methods: {len(methods)}")
    print("=" * 70)

    # Collect results
    all_results = {m['name']: {'threshold': [], 'coverage': [],
                               'exclusion': [], 'log_vol': [],
                               'cov_class': []}
                   for m in methods}
    shift = shared.get('shift', False)

    n_test = None
    l = 0
    for si, seed in enumerate(seeds):
        print(f"\n{'#'*70}")
        print(f"# SEED {seed} ({si+1}/{len(seeds)})")
        print(f"{'#'*70}")

        # Generate fresh data for each seed
        set_seed(seed)
        n_total = shared['n_per_cluster'] * shared['n_clusters']
        if l == 0:
            z_all, labels_all = generate_clusters(
                n_per_cluster=shared['n_per_cluster'],
                d=shared['d'],
                n_clusters=shared['n_clusters'],
                separation=shared['separation'],
                std=shared.get('std', 1.0),
                seed=seed,
                distribution=shared.get('distribution', 'isotropic')
            )
            l=1

        # Split train / val
        train_ratio = shared.get('train_ratio', 0.6)
        n_train = int(train_ratio * len(z_all))
        z_train, labels_train = z_all[:n_train], labels_all[:n_train]
        z_val, labels_val = z_all[n_train:], labels_all[n_train:]

        if n_test is None:
            n_test = len(z_val) // 2

        p_train = shared.get('p_train', 1.0)
        sampler_train = SimLabelSampler(z_train, labels_train, p_same=p_train)
        sampler_val = SimLabelSampler(z_val, labels_val, p_same=p_train)

        # Plot clusters once (first seed)
        is_first_seed = (si == 0)
        plot_dir = None
        plot_test_idx = None
        if is_first_seed:
            fig_dir = os.path.join(cli_args.save_dir, 'figures')
            os.makedirs(fig_dir, exist_ok=True)
            cluster_path = plot_clusters(z_all, labels_all,
                          title=f'Simulated Clusters (seed={seed})',
                          save_path=os.path.join(fig_dir, 'clusters.png'))
            print(f"Saved cluster plot to {cluster_path}")
            plot_dir = fig_dir
            # Pick one random test point — same for all methods
            n_half = len(z_val) // 2
            plot_test_idx = n_half + torch.randint(0, len(z_val) - n_half, (1,)).item()
            print(f"Plot anchor: index={plot_test_idx}, "
                  f"label={labels_val[plot_test_idx].item()}")

        for mi, method in enumerate(methods):
            name = method['name']
            print(f"\n--- [{si+1}/{len(seeds)}] [{mi+1}/{len(methods)}] "
                  f"{name} (seed={seed}) ---")

            args = build_args(shared, method, seed)
            set_seed(seed)

            try:
                t, cov, exc, vol, cov_cls = run_single(
                    method, args, z_train, labels_train, z_val, labels_val,
                    sampler_train, sampler_val,
                    z_all=z_all, labels_all=labels_all,
                    plot_dir=plot_dir, plot_test_idx=plot_test_idx)
            except Exception as e:
                print(f"  !! FAILED: {e} — skipping this run")
                torch.cuda.empty_cache()
                continue

            all_results[name]['threshold'].append(t)
            all_results[name]['coverage'].append(cov)
            all_results[name]['exclusion'].append(exc)
            all_results[name]['log_vol'].append(vol)
            all_results[name]['cov_class'].append(cov_cls)

            cc_str = f", Cov_class={cov_cls:.1%}" if shift else ""
            print(f"  -> t={t:.4f}, Cov={cov:.1%}{cc_str}, Exc={exc:.1%}, LogVol={vol:.2f}")

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
        cc_str = ""
        if shift:
            cc_arr = np.array(vals['cov_class'], dtype=float) * 100
            cc_str = f" | Cov_class={np.nanmean(cc_arr):.1f}±{np.nanstd(cc_arr):.1f}%"
        print(f"{name:35s} | t={t_m:.4f}±{t_s:.4f} | "
              f"Cov={c_m:.1f}±{c_s:.1f}%{cc_str} | Exc={e_m:.1f}±{e_s:.1f}% | "
              f"Vol={v_m:.2f}±{v_s:.2f}")

    # LaTeX
    latex = generate_latex(all_results, shared, seeds, n_test or 0)
    print(f"\n{'='*70}")
    print("LATEX TABLE")
    print(f"{'='*70}")
    print(latex)

    # Save
    out_path = os.path.join(cli_args.save_dir, 'sim_table_results.json')
    with open(out_path, 'w') as f:
        json.dump({'seeds': seeds, 'shared': shared, 'results': all_results},
                  f, indent=2, default=str)

    tex_path = os.path.join(cli_args.save_dir, f'sim_table_{seeds}seeds_{cli_args.note}.tex')
    with open(tex_path, 'w') as f:
        f.write(latex)

    print(f"\nSaved results to {out_path}")
    print(f"Saved LaTeX table to {tex_path}")


if __name__ == '__main__':
    main()