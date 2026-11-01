"""
Within-building baseline fits on the pooled Cinvestav 3.5 GHz data.

Establishes the local reference that all transfer results are judged against:
per-building path loss exponent, obstruction losses, and residual sigma.

Model form (intercept FIXED at the 33.3 dB system-loss anchor, so no constant
term is estimated):

    PL - 33.3  =  10*n*log10(d)  +  sum_i L_i * N_i  +  beta * config_C2

Two feature sets are fitted side by side:
  FULL  - per-material obstruction counts. Better specified, but several
          coefficients are unidentifiable in Comms and SSE (zero variance).
  TOTAL - single aggregate N_total. Coarser, estimable everywhere. This is
          the one that will carry the three-scenario rotation.

Requires: pandas numpy statsmodels scipy
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

DATA_DIR = "/Users/ginacelentano/PathLossMLWireless/data/Re__Machine_Learning-Based_Propagation_Models"
POOLED = os.path.join(DATA_DIR, "pooled.csv")

WALL_COLS = [
    "Num_brick_wall",
    "Num_wood_wall",
    "Num_glass_wall",
    "Num_drywall",
    "Num_column",
    "Num_elevator",
]

FEATURES_FULL = ["tenlog10d"] + WALL_COLS + ["config_C2"]
FEATURES_TOTAL = ["tenlog10d", "N_total", "config_C2"]

N_BOOT = 500
RNG = np.random.default_rng(0)


def drop_degenerate(df, feats):
    """Drop features with no variance in this subset -- they are unidentifiable."""
    keep, dropped = [], []
    for f in feats:
        if df[f].std() > 1e-12:
            keep.append(f)
        else:
            dropped.append(f)
    return keep, dropped


def fit_anchored(df, feats, anchored=True):
    """Fit on the anchored target.

    anchored=True  -> through the origin (CI convention: intercept pinned at
                      the 33.3 dB system-loss anchor, nothing estimated).
    anchored=False -> free constant (COST231 / LDPLSM-MW convention). The
                      fitted constant is the offset FROM 33.3 dB.
    """
    X = df[feats].astype(float)
    if not anchored:
        X = sm.add_constant(X, has_constant="add")
    y = df["PL_excess"].astype(float)
    return sm.OLS(y, X).fit()


def vif_table(df, feats):
    """Variance inflation factors. High VIF on tenlog10d = exponent not identified."""
    X = df[feats].astype(float).values
    out = {}
    for i, f in enumerate(feats):
        others = np.delete(X, i, axis=1)
        if others.shape[1] == 0:
            out[f] = 1.0
            continue
        r2 = sm.OLS(X[:, i], sm.add_constant(others)).fit().rsquared
        out[f] = np.inf if r2 > 1 - 1e-12 else 1.0 / (1.0 - r2)
    return pd.Series(out)


def block_bootstrap_n(df, feats, anchored=True, n_boot=N_BOOT):
    """Resample grid-letter blocks (not rows) to respect spatial correlation.

    NOTE: blocked resampling is standard in spatial statistics and is the
    remedy cited in the spatial-CV literature, but I did not find a propagation
    paper that block-bootstraps the path loss exponent specifically. Defensible
    by analogy; not established practice in this field.
    """
    blocks = [g for _, g in df.groupby(["config", "grid_letter"], dropna=False)]
    if len(blocks) < 3:
        return np.array([])
    draws = []
    for _ in range(n_boot):
        idx = RNG.integers(0, len(blocks), len(blocks))
        samp = pd.concat([blocks[i] for i in idx], ignore_index=True)
        if samp[feats].std().min() <= 1e-12:
            continue
        try:
            draws.append(fit_anchored(samp, feats, anchored).params["tenlog10d"])
        except Exception:
            continue
    return np.array(draws)


def residual_diagnostics(df, resid):
    """Residual sigma, bias, trend vs log-distance, normality."""
    x = df["log10d"].astype(float)
    trend = sm.OLS(resid, sm.add_constant(x)).fit()
    slope = trend.params.iloc[1]
    lo, hi = trend.conf_int().iloc[1]
    return {
        "sigma_dB": float(np.std(resid, ddof=1)),
        "bias_dB": float(np.mean(resid)),
        "trend_slope": float(slope),
        "trend_ci": (float(lo), float(hi)),
        "trend_ok": bool(lo < 0 < hi),
        "shapiro_p": float(stats.shapiro(resid[:5000])[1]) if len(resid) >= 3 else np.nan,
    }


def report(df, label, feats, name, anchored=True):
    keep, dropped = drop_degenerate(df, feats)
    if "tenlog10d" not in keep:
        print(f"  [{name}] no distance variance -- skipped")
        return None

    m = fit_anchored(df, keep, anchored)
    n_hat = m.params["tenlog10d"]
    resid = m.resid.values
    d = residual_diagnostics(df, resid)
    boot = block_bootstrap_n(df, keep, anchored)

    print(f"\n  [{name}]  n = {len(df)}")
    if dropped:
        print(f"    unidentifiable, dropped: {', '.join(dropped)}")
    if not anchored:
        c = m.params["const"]
        print(f"    fitted offset from 33.3 dB anchor = {c:+.2f} dB  "
              f"(implied PL(1m) = {33.3 + c:.1f} dB)")
    print(f"    n_hat = {n_hat:.3f}", end="")
    if boot.size:
        print(f"   block-bootstrap 95% CI [{np.percentile(boot, 2.5):.3f}, "
              f"{np.percentile(boot, 97.5):.3f}]  (sd {boot.std():.3f})")
    else:
        print()

    print("    coefficients (dB per unit):")
    for f in keep:
        if f == "tenlog10d":
            continue
        print(f"      {f:18s} {m.params[f]:7.2f}   p={m.pvalues[f]:.3g}")
    print("      (COST231 note: these are calibration coefficients that absorb"
          " furniture, not physical wall losses)")

    print(f"    residual sigma = {d['sigma_dB']:.2f} dB   bias = {d['bias_dB']:.2f} dB")
    print(f"    residual trend vs log10d: slope={d['trend_slope']:.2f} "
          f"CI[{d['trend_ci'][0]:.2f},{d['trend_ci'][1]:.2f}] "
          f"{'OK' if d['trend_ok'] else '<-- STRUCTURE LEFT IN RESIDUALS'}")
    print(f"    Shapiro-Wilk p = {d['shapiro_p']:.3g}")

    v = vif_table(df, keep)
    worst = v.drop("tenlog10d", errors="ignore")
    print(f"    VIF(tenlog10d) = {v['tenlog10d']:.1f}"
          + (f"   max other VIF = {worst.max():.1f} ({worst.idxmax()})" if len(worst) else ""))

    return {
        "scope": label, "model": name, "anchored": anchored,
        "n": len(df), "n_hat": n_hat,
        "offset_dB": (m.params["const"] if not anchored else 0.0),
        "n_hat_lo": np.percentile(boot, 2.5) if boot.size else np.nan,
        "n_hat_hi": np.percentile(boot, 97.5) if boot.size else np.nan,
        "sigma_dB": d["sigma_dB"], "bias_dB": d["bias_dB"],
        "trend_ok": d["trend_ok"], "vif_dist": v["tenlog10d"],
        "dropped": ";".join(dropped),
    }


def main():
    df = pd.read_csv(POOLED)
    df["tenlog10d"] = 10.0 * df["log10d"]
    for c in WALL_COLS:
        if c not in df.columns:
            df[c] = 0.0

    rows = []
    variants = (("FULL", FEATURES_FULL), ("TOTAL", FEATURES_TOTAL))

    # Three subsets. LOS is reported for completeness but is underpowered:
    # ~57 points total, and they are distance-truncated by construction
    # (no wall in the path => receiver is close to the transmitter).
    subsets = {
        "ALL":  df,
        "NLOS": df[df["audit_is_LOS_geom"] == 0],
        "LOS":  df[df["audit_is_LOS_geom"] == 1],
    }

    for sub_name, sub in subsets.items():
        print("\n" + "#" * 70)
        print(f"# SUBSET: {sub_name}   (n = {len(sub)})")
        if sub_name == "LOS":
            print("# All obstruction counts are 0 here by definition, so the")
            print("# FULL and TOTAL models collapse to the same fit.")
            print("# Treat these exponents as indicative only.")
        print("#" * 70)

        # Both intercept conventions for ALL; anchored only for the subsets,
        # where a free constant would be fitting noise.
        conventions = (True, False) if sub_name == "ALL" else (True,)

        for anchored in conventions:
            tag = "ANCHORED at 33.3 dB (CI)" if anchored \
                  else "FLOATING constant (COST231 / LDPLSM-MW)"
            print(f"\n===== {tag} =====")

            use = variants if sub_name != "LOS" else (("TOTAL", FEATURES_TOTAL),)

            for b, g in sub.groupby("building"):
                if len(g) < 10:
                    print(f"\n--- {b} --- n={len(g)}, too few points, skipped")
                    continue
                print(f"\n--- {b} ---")
                for name, feats in use:
                    r = report(g, b, feats, name, anchored)
                    if r:
                        r["subset"] = sub_name
                        rows.append(r)

            print("\n--- POOLED (no building dummies) ---")
            for name, feats in use:
                r = report(sub, "POOLED", feats, name, anchored)
                if r:
                    r["subset"] = sub_name
                    rows.append(r)

    res = pd.DataFrame(rows)

    print("\n" + "=" * 70)
    print("DOES DROPPING THE 57 LOS POINTS MOVE THE ANSWER?")
    print("  (if ALL and NLOS agree, those points carry no leverage and the")
    print("   pooled-vs-split question is moot for this dataset)")
    print("=" * 70)
    a = res[(res.subset == "ALL") & (res.anchored)]
    nl = res[res.subset == "NLOS"]
    for name in ("FULL", "TOTAL"):
        for scope in ("Comms", "Library", "SSE", "POOLED"):
            va = a[(a.model == name) & (a.scope == scope)]["n_hat"]
            vn = nl[(nl.model == name) & (nl.scope == scope)]["n_hat"]
            if len(va) and len(vn):
                print(f"  {name:6s} {scope:8s} ALL={va.iloc[0]:.3f}  "
                      f"NLOS={vn.iloc[0]:.3f}  delta={vn.iloc[0]-va.iloc[0]:+.3f}")

    print("\n" + "=" * 70)
    print("EXPONENT SPREAD ACROSS BUILDINGS")
    print("  (target < 0.50 -- my criterion, no literature basis)")
    print("=" * 70)
    for sub_name in ("ALL", "NLOS", "LOS"):
        for anchored in (True, False):
            for name in ("FULL", "TOTAL"):
                sub = res[(res.subset == sub_name) & (res.model == name)
                          & (res.anchored == anchored) & (res.scope != "POOLED")]
                if len(sub) < 2:
                    continue
                spread = sub.n_hat.max() - sub.n_hat.min()
                verdict = "PASS" if spread < 0.5 else "FAIL"
                lab = "anchored" if anchored else "floating"
                vals = "  ".join(f"{s}={v:.2f}" for s, v in zip(sub.scope, sub.n_hat))
                print(f"  {sub_name:5s} {lab:9s} {name:6s} spread = {spread:.2f}  "
                      f"[{verdict}]   {vals}")
    print("\n  naive (no obstruction terms, anchored) was 1.73")

    print("\n" + "=" * 70)
    print("REFERENCE VALUES")
    print("=" * 70)
    print("  3GPP InH-Office NLOS (CI):  n = 3.19,  sigma_SF = 8.29 dB")
    print("  3GPP InH-Office LOS  (CI):  n = 1.73,  sigma_SF = 3.02 dB")
    print("  This dataset is 97.5% NLOS -- compare against the NLOS row only.")
    print("  WARNING: using a 43.3 dB anchor on this PL (which omits the two")
    print("  5 dBi antenna gains) biases n_hat DOWN by ~1.0 and spuriously")
    print("  lands near 3.19. That agreement would be an artifact.")

    out = os.path.join(DATA_DIR, "baseline_results.csv")
    res.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()