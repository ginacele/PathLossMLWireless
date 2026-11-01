"""
Pool the six Cinvestav 3.5 GHz received-power files into one modeling frame.

Design decisions baked in (see discussion):
  - C1 and C2 are pooled WITHIN a building; antenna config becomes a covariate.
  - PL = Ptx - Prx with Ptx = 10 dBm.  Antenna gains are NOT removed, so this is
    SYSTEM loss; the free-space anchor at 1 m is 33.3 dB, not 43.3 dB.
  - N_total (sum of all obstruction counts) is the transferable feature.
    Per-material counts are kept too, but they have zero variance in some
    buildings and will not transfer.
  - No building dummies. The building column is for SPLITTING only.
  - Coord is kept for spatial-block CV but must never be used as a feature.
  - LOS/NLOS Classification is held out as an audit column, not a feature.
"""

import glob
import os
import numpy as np
import pandas as pd

DATA_DIR = "/Users/ginacelentano/PathLossMLWireless/data/Re__Machine_Learning-Based_Propagation_Models"
PTX_DBM = 10.0
D0 = 1.0
FSPL_1M_SYSTEM = 33.3  # 20*log10(4*pi/lambda) at 3.5 GHz, minus 2 x 5 dBi

WALL_COLS = [
    "Num_brick_wall",
    "Num_wood_wall",
    "Num_glass_wall",
    "Num_drywall",
    "Num_column",
]

# Row counts reported in the paper, for a sanity check against the files on disk.
PAPER_COUNTS = {
    ("Comms", "C1"): 721,
    ("Comms", "C2"): 677,
    ("SSE", "C1"): 107,
    ("SSE", "C2"): 107,
    ("Library", "C1"): 343,
    ("Library", "C2"): 344,
}


def load_one(path):
    """Load one Prx_<Scenario>_<Config>_classified.csv and normalize its schema."""
    name = os.path.basename(path)
    parts = name.replace(".csv", "").split("_")
    scenario, config = parts[1], parts[2]

    # utf-8-sig strips the BOM on the leading 'Coord.' header.
    df = pd.read_csv(path, encoding="utf-8-sig")

    # Drop the stray unnamed/empty columns (SSE_C2 carries two).
    df = df.loc[:, [c for c in df.columns if str(c).strip() != ""]]
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

    # Strip units from headers: 'Distance (m)' -> 'Distance', 'P_rx (dBm)' -> 'P_rx'
    df.columns = [str(c).split("(")[0].strip() for c in df.columns]

    # Keep the original column AND a clearly-named copy. The copy is what the
    # modeling code reads, so it is unambiguous the label never enters a fit.
    if "LOS/NLOS Classification" in df.columns:
        df["audit_label"] = df["LOS/NLOS Classification"]

    df["building"] = scenario
    df["config"] = config
    df["source_file"] = name
    return df


def main():
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "Prx_*_classified.csv")))
    if not paths:
        raise FileNotFoundError(f"No Prx_*_classified.csv found in {DATA_DIR}")

    frames = [load_one(p) for p in paths]

    print("--- Row counts: file vs paper ---")
    for f in frames:
        b, c, n = f["building"].iloc[0], f["config"].iloc[0], len(f)
        exp = PAPER_COUNTS.get((b, c))
        flag = "" if exp == n else f"  <-- paper says {exp}"
        print(f"  {b:8s} {c}  {n:4d}{flag}")

    df = pd.concat(frames, ignore_index=True, sort=False)
    print(f"\nPooled rows: {len(df)}")

    # --- elevator: Library only ---
    if "Elevator" in df.columns:
        df["has_elevator_feature"] = df["Elevator"].notna().astype(int)
        df["Num_elevator"] = pd.to_numeric(df["Elevator"], errors="coerce").fillna(0)
        df = df.drop(columns=["Elevator"])
    else:
        df["Num_elevator"] = 0
        df["has_elevator_feature"] = 0

    # --- obstruction counts ---
    for c in WALL_COLS:
        df[c] = pd.to_numeric(df.get(c), errors="coerce").fillna(0)

    df["N_total"] = df[WALL_COLS].sum(axis=1) + df["Num_elevator"]

    # --- target ---
    df["P_rx"] = pd.to_numeric(df["P_rx"], errors="coerce")
    df["PL"] = PTX_DBM - df["P_rx"]

    # --- distance ---
    df["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")

    # --- audit labels (never features) ---
    if "audit_label" not in df.columns:
        df["audit_label"] = np.nan
    df["audit_label"] = df["audit_label"].astype(str).str.strip().str.upper()
    df["audit_is_LOS_geom"] = (df["N_total"] == 0).astype(int)

    # --- cleaning ---
    n0 = len(df)
    df = df[df["PL"].notna() & df["Distance"].notna()]
    print(f"Dropped {n0 - len(df)} rows with missing PL or Distance")

    n1 = len(df)
    df = df[df["Distance"] > D0]
    print(f"Dropped {n1 - len(df)} rows with d <= {D0} m")

    # --- derived features ---
    df["log10d"] = np.log10(df["Distance"])
    df["PL_excess"] = df["PL"] - FSPL_1M_SYSTEM
    df["config_C2"] = (df["config"] == "C2").astype(int)

    # Reference-level change (-50 -> -30 dBm) noted in Comments
    if "Comments" in df.columns:
        df["ref_level_changed"] = (
            df["Comments"].astype(str).str.contains("30", na=False).astype(int)
        )
    else:
        df["ref_level_changed"] = 0

    # Grid letter band, for spatial-block CV
    df["grid_letter"] = (
        df["Coord."].astype(str).str.extract(r"^([A-Za-z]+)")[0].str.upper()
    )

    # --- summary ---
    print("\n--- Per building x config ---")
    print(
        df.groupby(["building", "config"]).agg(
            n=("PL", "size"),
            d_min=("Distance", "min"),
            d_max=("Distance", "max"),
            PL_mean=("PL", "mean"),
            PL_sd=("PL", "std"),
            frac_LOS_geom=("audit_is_LOS_geom", "mean"),
        ).round(2)
    )

    print("\n--- Stated label vs geometry (N_total == 0) ---")
    print(pd.crosstab(df["audit_label"], df["audit_is_LOS_geom"]))

    print("\n--- Obstruction sd by building (0.00 = unidentifiable in that fold) ---")
    print(df.groupby("building")[WALL_COLS + ["Num_elevator"]].std().round(2))

    print("\n--- Naive anchored exponent, no obstruction terms ---")
    for b, g in df.groupby("building"):
        n_hat = np.polyfit(10 * g["log10d"], g["PL_excess"], 1)[0]
        print(f"  {b:10s} n_hat = {n_hat:.2f}   (n={len(g)})")
    n_all = np.polyfit(10 * df["log10d"], df["PL_excess"], 1)[0]
    print(f"  {'POOLED':10s} n_hat = {n_all:.2f}   (n={len(df)})")

    out = os.path.join(DATA_DIR, "pooled.csv")
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")

    return df


if __name__ == "__main__":
    main()