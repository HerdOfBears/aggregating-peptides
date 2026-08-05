#!/usr/bin/env python3
"""
Multi-panel correlation-dimension figure over an experiment directory.

Given an EXPERIMENT_DIR whose immediate subdirectories are jobids (mj1, mj2, ...),
this script computes the correlation integral C(r) for the last frame of every
replica of every jobid, draws one log10(C(r)) vs log10(r/r0) panel per jobid, and
reports the mean slope +/- standard error (across replicas) inside each region
defined by the vertical guide lines.

Expected layout (matches the original snippet):

    EXPERIMENT_DIR/
        mj14/
            cg/                        <-- --subpath (default: "cg")
                replica-1/solvated.gro
                replica-1/prod.xtc
                replica-2/...
        mj15/
            cg/
                ...

Usage
-----
    python plot_correlation_dimension.py outputs/mj-Ceq45p2-neutral_term/ \
        --regions "1.2:1.6,1.675:1.9" \
        -o corrdim.png

Slopes are fit in log10(r/r0) space, so they are invariant to the r0 choice
(r0 only shifts the x-axis / intercepts). Region boundaries and the vertical
lines are given in the SAME units as the x-axis, i.e. log10(r/r0).
"""

import argparse
import glob
import math
import os
import re
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import MDAnalysis as mda
from MDAnalysis.analysis import distances
from sklearn.linear_model import LinearRegression

from aggrepep.morphology import compute_correlation_dimension

def chain_coms(protein, n_chains):
    """
    Mass-weighted center of mass of each contiguous chain.

    Parameters
    ----------
    protein : MDAnalysis AtomGroup
        The protein atoms (or beads) to compute COMs for.
    n_chains : int
        The number of contiguous chains in the protein. The atoms must be
        ordered such that the first n_atoms/n_chains atoms are chain 1, 
        the next n_atoms/n_chains atoms are chain 2, etc.
    
    Returns
    -------
    xyz_com : np.ndarray, shape (n_chains, 3)
    """
    n_atoms = protein.n_atoms
    assert n_atoms % n_chains == 0, "atoms don't split evenly into chains"
    per = n_atoms // n_chains

    pos  = protein.positions.reshape(n_chains, per, 3)   # (n_chains, per, 3)
    mass = protein.masses.reshape(n_chains, per)         # (n_chains, per)

    xyz_com = (pos * mass[:, :, None]).sum(axis=1) / mass.sum(axis=1)[:, None]
    return xyz_com


# ----------------------------------------------------------------------------
# Discovery helpers
# ----------------------------------------------------------------------------
def _natural_key(path):
    """Sort key so mj2 < mj14 (numeric-aware)."""
    name = os.path.basename(os.path.normpath(path))
    m = re.search(r"(\d+)", name)
    return (name[: m.start()] if m else name, int(m.group(1)) if m else -1)


def discover_jobids(experiment_dir, jobid_glob):
    dirs = [p for p in glob.glob(os.path.join(experiment_dir, jobid_glob))
            if os.path.isdir(p)]
    return sorted(dirs, key=_natural_key)


def discover_replicas(jobid_dir, subpath):
    reps = glob.glob(os.path.join(jobid_dir, subpath, "replica-*"))
    reps = [r for r in reps if os.path.isdir(r)]

    def rep_num(p):
        m = re.search(r"replica-(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else 0

    return sorted(reps, key=rep_num)


# ----------------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------------
def fit_region_slope(x, y, lo, hi):
    """Fit y = m*x + b for points with lo < x < hi. Returns (m, b) or (None, None)."""
    mask = (x > lo) & (x < hi) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return None, None
    reg = LinearRegression().fit(x[mask].reshape(-1, 1), y[mask].reshape(-1, 1))
    return float(reg.coef_[0, 0]), float(reg.intercept_[0])


def slope_stats(slopes):
    """Mean, sample std, standard error and 95% CI half-width for a list of slopes."""
    s = np.asarray([v for v in slopes if v is not None], dtype=float)
    n = s.size
    if n == 0:
        return dict(n=0, mean=np.nan, sd=np.nan, se=np.nan, ci95=np.nan)
    mean = float(s.mean())
    # ddof=1 => sample std (the original snippet used ddof=0; flip if you want to match exactly).
    sd = float(s.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n)
    return dict(n=n, mean=mean, sd=sd, se=se, ci95=1.96 * se)


# ----------------------------------------------------------------------------
# Per-jobid processing
# ----------------------------------------------------------------------------
def process_jobid(jobid_dir, args):
    """Return per-replica (x, y) log arrays and per-region slope lists for one jobid."""
    jobid = os.path.basename(os.path.normpath(jobid_dir))
    replicas = discover_replicas(jobid_dir, args.subpath)
    if not replicas:
        print(f"[{jobid}] no replicas found under '{args.subpath}/replica-*'", file=sys.stderr)
        return None

    log_r0 = np.log10(args.r0)
    per_replica = []                                   # list of (x, y) log arrays
    region_slopes = {i: [] for i in range(len(args.regions))}

    for rep_dir in replicas:
        gro = os.path.join(rep_dir, "solvated.gro")
        xtc = os.path.join(rep_dir, "prod.xtc")
        if not (os.path.exists(gro) and os.path.exists(xtc)):
            print(f"[{jobid}] skipping {os.path.basename(rep_dir)} (missing gro/xtc)",
                  file=sys.stderr)
            continue

        try:
            uni = mda.Universe(gro, xtc)
            uni.trajectory[args.frame]
            protein = uni.select_atoms(args.selection)
            coms = chain_coms(protein, args.chain_size)
            r_vals, Cr_vals = compute_correlation_dimension(
                coms, uni.dimensions,
                r_min=args.r_min, r_max=args.r_max, n_bins=args.n_bins,
            )
        except Exception as exc:  # keep going if one replica is broken
            print(f"[{jobid}] error on {os.path.basename(rep_dir)}: {exc}", file=sys.stderr)
            continue

        # log10(r/r0) vs log10(C(r)); drop non-positive C(r) so the log is finite.
        x = np.log10(r_vals) - log_r0
        with np.errstate(divide="ignore"):
            y = np.log10(Cr_vals)
        good = np.isfinite(y)
        per_replica.append((x[good], y[good]))

        for i, (lo, hi) in enumerate(args.regions):
            m, _ = fit_region_slope(x, y, lo, hi)
            if m is not None:
                region_slopes[i].append(m)

    if not per_replica:
        return None
    return dict(jobid=jobid, replicas=per_replica, region_slopes=region_slopes)


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def make_figure(results, args):
    n = len(results)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.8 * ncols, 4.2 * nrows),
                             squeeze=False)
    axes_flat = axes.ravel()

    # Vertical guide lines = unique region boundaries.
    vline_positions = sorted({b for region in args.regions for b in region})

    for ax, res in zip(axes_flat, results):
        for j, (x, y) in enumerate(res["replicas"]):
            ax.scatter(x, y, s=12, alpha=0.7, label=f"rep {j + 1}")

        for vx in vline_positions:
            ax.axvline(vx, linestyle="--", color="k", linewidth=1)

        # Per-region mean slope +/- SE, drawn as an annotation and a mean fit line.
        lines = []
        for i, (lo, hi) in enumerate(args.regions):
            st = slope_stats(res["region_slopes"][i])
            if st["n"] == 0:
                lines.append(f"[{lo:.3g},{hi:.3g}]: no fit")
                continue
            lines.append(
                f"[{lo:.3g},{hi:.3g}]: m = {st['mean']:.3f} $\\pm$ {st['se']:.3f} (n={st['n']})"
            )
            # Mean fit line: use mean slope through the region's mid-data intercept.
            intercepts = []
            for (x, y) in res["replicas"]:
                _, b = fit_region_slope(x, y, lo, hi)
                if b is not None:
                    intercepts.append(b)
            if intercepts:
                b_bar = float(np.mean(intercepts))
                xs = np.array([lo, hi])
                ax.plot(xs, st["mean"] * xs + b_bar, color="k", linewidth=2.25, alpha=1.0)

        ax.text(0.52, 0.02, "\n".join(lines), transform=ax.transAxes,
                fontsize=9, va="bottom", ha="left",
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))

        ax.set_title(f"{res['jobid']} correlation dimension", fontsize=13)
        ax.legend(fontsize=8, loc="upper left")

    # Hide any unused panels.
    for ax in axes_flat[n:]:
        ax.axis("off")

    # Shared axis labels.
    try:
        fig.supxlabel(r"$\log_{10}(r/r_0)$", fontsize=16)
        fig.supylabel(r"$\log_{10}(C(r))$", fontsize=16)
    except AttributeError:  # older matplotlib
        for ax in axes_flat[:n]:
            ax.set_xlabel(r"$\log_{10}(r/r_0)$", fontsize=12)
            ax.set_ylabel(r"$\log_{10}(C(r))$", fontsize=12)

    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"\nSaved figure -> {args.output}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_regions(spec):
    """'1.2:1.6,1.675:1.9' -> [(1.2, 1.6), (1.6, 1.9)] (units: log10(r/r0))."""
    regions = []
    for chunk in spec.split(","):
        lo, hi = chunk.split(":")
        lo, hi = float(lo), float(hi)
        if hi <= lo:
            raise argparse.ArgumentTypeError(f"region '{chunk}' must have hi > lo")
        regions.append((lo, hi))
    return regions


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment_dir", help="Directory containing jobid subdirs (mj*).")
    p.add_argument("-o", "--output", default=None, help="Output figure path (png/pdf/svg).")
    p.add_argument("--jobid-glob", default="mj*", help="Glob for jobid subdirs (default: mj*).")
    p.add_argument("--subpath", default="cg",
                   help="Subdir under each jobid holding replica-* (default: cg).")
    p.add_argument("--selection", default="name BB SC1 SC2 SC3",
                   help="MDAnalysis atom selection (default: 'name BB SC1 SC2 SC3').")
    p.add_argument("--chain-size", type=int, default=64,
                   help="Atoms/beads per chain passed to chain_coms (default: 64).")
    p.add_argument("--frame", type=int, default=-1, help="Trajectory frame index (default: -1).")
    p.add_argument("--r-min", type=float, default=17.5)
    p.add_argument("--r-max", type=float, default=200.0)
    p.add_argument("--n-bins", type=int, default=100)
    p.add_argument("--r0", type=float, default=1.0,
                   help="Length scale for x = log10(r/r0) (default: 1.0).")
    p.add_argument("--regions", type=parse_regions, default="1.2:1.6,1.6:1.9",
                   help="Comma-separated lo:hi regions in log10(r/r0) "
                        "(default: '1.2:1.6,1.675:1.9').")
    p.add_argument("--dpi", type=int, default=200)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if isinstance(args.regions, str):          # when default string wasn't type-converted
        args.regions = parse_regions(args.regions)

    if args.output is None:
        base = os.path.basename(os.path.normpath(args.experiment_dir)) or "experiment"
        args.output = f"{base}_corrdim.png"

    jobid_dirs = discover_jobids(args.experiment_dir, args.jobid_glob)
    if not jobid_dirs:
        sys.exit(f"No jobid dirs matching '{args.jobid_glob}' in {args.experiment_dir}")

    print(f"Found {len(jobid_dirs)} jobid(s): "
          f"{', '.join(os.path.basename(d) for d in jobid_dirs)}")

    results = []
    jobid_to_slope_stats = {}
    for jd in jobid_dirs:
        res = process_jobid(jd, args)
        if res is None:
            continue
        results.append(res)


        jobid_to_slope_stats[res['jobid']] = slope_stats(res["region_slopes"][i] for i in range(len(args.regions)))

        # Console summary.
        print(f"\n=== {res['jobid']} ===")
        for i, (lo, hi) in enumerate(args.regions):
            st = slope_stats(res["region_slopes"][i])
            if st["n"] == 0:
                print(f"  region [{lo}, {hi}]: no valid fits")
            else:
                print(f"  region [{lo}, {hi}]: m = {st['mean']:.4f} "
                      f"+/- {st['se']:.4f} (SE)  |  95% CI +/- {st['ci95']:.4f}  "
                      f"(n={st['n']})")

    if not results:
        sys.exit("No jobid produced usable data.")

    with open(os.path.join(args.experiment_dir, "corrdim_summary.csv"), "w") as f:
        f.write("JobID,Region,MeanSlope,StdErr,95%CI,NumReplicas\n")
        for res in results:
            jobid = res['jobid']
            for i, (lo, hi) in enumerate(args.regions):
                st = slope_stats(res["region_slopes"][i])
                f.write(f"{jobid},{lo}:{hi},{st['mean']:.4f},{st['se']:.4f},{st['ci95']:.4f},{st['n']}\n")

    make_figure(results, args)


if __name__ == "__main__":
    main()