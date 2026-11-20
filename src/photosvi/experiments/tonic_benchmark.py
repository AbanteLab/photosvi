"""The tonic-response cohort benchmark.

The second cohort benchmark, kept apart from the rate benchmark so that each tool is tested
where it applies: simulated crossed cohorts in which the height P of the tonic response
differs between the two conditions, condition 1 at P0 and condition 2 at (1 + dP) P0, while
the transient rate does not (delta = 0). Four analyses each give one statistic per cohort,
and their thresholds are calibrated on the dP = 0 cohorts as in the rate benchmark:

  flmm      functional linear mixed model (fastFMM; Cui et al. 2022, as applied to fiber
            photometry by Loewinger et al. 2025) on dF/F in 1-min bins aligned to the
            injection, condition as fixed effect, random intercept per animal. Statistic:
            the largest |pointwise t| of the condition coefficient after the injection.
  zlp, zals the Bonaventura lab's tonic analysis (process_merged_data.m, 2026): per session,
            z-score dF/F against a baseline segment (median and standard deviation), take the
            slow component by (zlp) a 4th-order Butterworth low-pass at 0.1 Hz applied
            causally (MATLAB `filter`, scipy `lfilter`) or (zals) an asymmetric-least-squares
            baseline (lambda 1e6, unscaled; p 0.01; 10 iterations), and integrate it in
            5-min windows (sum / fs). Statistic: paired t-test across animals of the summed
            post-injection AUC, treated minus vehicle; the per-window AUCs are recorded too.
  twostage  PhotoSVI single-trace fits (scripts/cohort_twostage.py): per-session tonic peak
            P; paired t-test of log(P_treated / P_vehicle). Estimate: the mean log ratio.
  hier      PhotoSVI cohort model (scripts/cohort_hierarchical.py): the contrast of the
            per-condition tonic amplitude means, alpha_treated - alpha_vehicle with
            alpha = log P, over its posterior standard deviation. Estimate: the contrast.

The two PhotoSVI estimates are of log(1 + dP); bias and RMSE are reported on that axis.

The cohorts themselves come from the existing simulator: `tonic_config` writes one config
per dP, and the existing cohort scripts read it through CONFIG, so every arm regenerates the
same cohort from (config, seed) exactly as in the rate benchmark.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path

import numpy as np
import yaml

ARMS = ("flmm", "zlp", "zals", "twostage", "hier")
LABELS = {"flmm": "FLMM on dF/F", "zlp": "z-score AUC, low-pass", "zals": "z-score AUC, ALS",
          "twostage": "two-stage, paired", "hier": "hierarchical"}


def tonic_config(cfg: dict, dp: float) -> dict:
    """A copy of `cfg` whose second condition has tonic peak (1 + dp) times the first's."""
    c = copy.deepcopy(cfg)
    p0 = float(c["cohort"]["tonic_peak"][0])
    c["cohort"]["tonic_peak"] = [p0, p0 * (1.0 + float(dp))]
    c.setdefault("grid", {})["this_tonic_dp"] = float(dp)
    return c


def write_cell_config(base: str | Path, dp: float, out_dir: str | Path,
                      fit_overrides: dict | None = None) -> Path:
    """Write the config of one dP cell into `out_dir` and return its path.

    `fit_overrides` are merged into the `fit` block, e.g. {"tonic_block": True} for the
    opt-in joint block over the tonic means.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = tonic_config(yaml.safe_load(open(base)), dp)
    cfg["fit"].update(fit_overrides or {})
    p = out_dir / f"config_dp{dp:.2f}.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return p


def binned_traces(y: np.ndarray, inj_t, fs: float, pre_min: float = 10.0,
                  bin_s: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    """Mean dF/F per `bin_s` bin, aligned to each session's injection.

    `y` is [S, T]. Bins run from `pre_min` minutes before the injection to the shortest
    post-injection stretch of the cohort (and the shortest pre-injection one), so every session
    has the same grid. Returns the
    [S, L] matrix and the bin centres in minutes from the injection.
    """
    n = int(round(bin_s * fs))
    # As many pre-injection bins as every session has: with the injection jittered, the
    # earliest one may come before `pre_min`.
    n_pre = min(int(round(pre_min * 60 / bin_s)), min(int(t) for t in inj_t) // n)
    post = min(y.shape[1] - int(t) for t in inj_t)
    n_post = post // n
    rows = []
    for s in range(y.shape[0]):
        i0 = int(inj_t[s]) - n_pre * n
        seg = y[s, i0:i0 + (n_pre + n_post) * n]
        rows.append(seg.reshape(-1, n).mean(1))
    centres = (np.arange(-n_pre, n_post) + 0.5) * bin_s / 60.0
    return np.array(rows), centres


def _paired_t(d: np.ndarray) -> float:
    d = np.asarray(d, float)
    if d.size < 2 or not np.isfinite(d).all() or d.std(ddof=1) == 0:
        return float("nan")
    return float(d.mean() / (d.std(ddof=1) / math.sqrt(d.size)))


def _pair(values, animal, drug) -> np.ndarray:
    """Per animal, treated minus vehicle (drug 1 minus drug 0)."""
    out = []
    for a in sorted(set(int(x) for x in animal)):
        v = {int(d): float(x) for x, an, d in zip(values, animal, drug) if int(an) == a}
        if 0 in v and 1 in v:
            out.append(v[1] - v[0])
    return np.array(out)


def auc_arm(binned: np.ndarray, centres: np.ndarray, animal, drug) -> dict:
    """Post-injection minus pre-injection mean dF/F per session, paired across animals."""
    post, pre = centres > 0, centres < 0
    v = binned[:, post].mean(1) - binned[:, pre].mean(1)
    d = _pair(v, animal, drug)
    return {"stat": abs(_paired_t(d)), "t": _paired_t(d), "diff": float(d.mean())}


# ------------------------------------------------- the Bonaventura lab's tonic analysis
def als_baseline(y: np.ndarray, lam: float = 1e6, p: float = 0.01, n_iter: int = 10) -> np.ndarray:
    """Asymmetric least squares baseline, as `als_baseline` in process_merged_data.m."""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    L = y.size
    D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(L - 2, L))
    DtD = lam * (D.T @ D)
    w = np.ones(L)
    for _ in range(n_iter):
        z = spsolve((sparse.diags(w) + DtD).tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return z


def zscore_slow(y: np.ndarray, fs: float, base: tuple[int, int]) -> dict[str, np.ndarray]:
    """z-score against y[base[0]:base[1]], then the two slow components of the lab script."""
    from scipy.signal import butter, lfilter
    seg = y[base[0]:base[1]]
    z = (y - np.median(seg)) / np.std(seg, ddof=1)
    b, a = butter(4, 0.1 / (fs / 2), "low")
    return {"zlp": lfilter(b, a, z), "zals": als_baseline(z)}


def bfp_arms(y: np.ndarray, inj_t, fs: float, animal, drug, base_start_s: float = 300.0,
             win_s: float = 300.0) -> dict:
    """The lab's z-scored AUC, low-pass and ALS variants, on one cohort.

    The lab z-scores against minutes 5-15 of a protocol whose first injection is at 10 min;
    the simulated recordings have one injection, at `inj_t`, so the baseline is the
    pre-injection stretch from `base_start_s` to the injection, samples
    [round(base_start_s * fs), inj_t). AUC windows of `win_s` are aligned to the injection;
    the statistic sums the post-injection ones that every session has.
    """
    S = y.shape[0]
    n = int(round(win_s * fs))
    n_post = min((y.shape[1] - int(t)) // n for t in inj_t)
    n_pre = min(int(t) for t in inj_t) // n
    out = {}
    for var in ("zlp", "zals"):
        auc_post, auc_pre = [], []
        for s in range(S):
            t = int(inj_t[s])
            slow = zscore_slow(y[s], fs, (int(round(base_start_s * fs)), t))[var]
            auc_post.append([slow[t + k * n:t + (k + 1) * n].sum() / fs for k in range(n_post)])
            auc_pre.append([slow[t - (k + 1) * n:t - k * n].sum() / fs for k in range(n_pre)][::-1])
        tot = [sum(r) for r in auc_post]
        d = _pair(tot, animal, drug)
        tt = _paired_t(d)
        out[var] = {"stat": abs(tt), "t": tt, "diff": float(d.mean()),
                    "auc_post_windows": auc_post, "auc_pre_windows": auc_pre}
    return out


def twostage_arm(ts_rows: list[dict]) -> dict:
    """Paired t-test of log(P_treated / P_vehicle) from per-session single-trace fits."""
    lp = [math.log(r["tonic_peak"]) for r in ts_rows]
    d = _pair(lp, [r["animal"] for r in ts_rows], [r["drug"] for r in ts_rows])
    t = _paired_t(d)
    return {"stat": abs(t), "t": t, "est": float(d.mean()), "n_pairs": int(d.size)}


def hier_arm(hier: dict) -> dict:
    """|z| of the contrast of the per-condition tonic amplitude means (alpha = log P).

    Reads the contrast the model reports itself when there is one (alpha_treated -
    alpha_vehicle with its standard deviation, which a joint block over the tonic means must
    supply because the two are then correlated); otherwise the factorized guide's
    per-condition posteriors, whose contrast width is their quadrature sum.
    """
    # `tonic_contrast_alpha` = [loc, sd] from HierarchicalCohortModel.tonic_contrast(1, 0),
    # which uses the guide's covariance whether or not the tonic means are in a joint block
    # (fit.tonic_block); the older keys are read only if it is absent.
    tc = hier.get("tonic_contrast_alpha")
    if tc:
        return {"stat": abs(tc[0] / tc[1]), "est": tc[0], "scale": tc[1]}
    tc = hier.get("tonic_alpha_contrast")
    if tc:
        return {"stat": abs(tc["loc"] / tc["scale"]), "est": tc["loc"], "scale": tc["scale"]}
    q = hier["tonic_alpha_q"]
    names = list(q)
    loc = q[names[1]]["loc"] - q[names[0]]["loc"]
    scale = math.hypot(q[names[1]]["scale"], q[names[0]]["scale"])
    return {"stat": abs(loc / scale), "est": loc, "scale": scale}


# ------------------------------------------------------------------------------ FLMM
R_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "cohort_tonic_flmm.R"
RSCRIPT = os.environ.get("RSCRIPT", "Rscript")


def flmm_arm(binned: np.ndarray, centres: np.ndarray, animal, drug, workdir: str | Path) -> dict:
    """fastFMM on the binned dF/F: Y(s) ~ condition + (1 | animal).

    Statistic: max over post-injection bins of |beta_cond(s)| / its pointwise standard
    error. Also records whether the joint 95% band excludes zero anywhere after the
    injection, which is how the method is read in practice.
    """
    import subprocess
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    csv = workdir / "flmm_in.csv"
    out = workdir / "flmm_out.json"
    L = binned.shape[1]
    with open(csv, "w") as fh:
        fh.write("animal,cond," + ",".join(f"Y_{k + 1}" for k in range(L)) + "\n")
        for s in range(binned.shape[0]):
            fh.write(f"a{int(animal[s])},{int(drug[s])},"
                     + ",".join(f"{v:.8g}" for v in binned[s]) + "\n")
    r = subprocess.run([RSCRIPT, str(R_SCRIPT), str(csv), str(out)], capture_output=True,
                       text=True)
    if r.returncode != 0:
        raise RuntimeError(f"fastFMM failed:\n{r.stderr[-2000:]}")
    res = json.load(open(out))
    beta = np.array(res["beta_cond"])
    se = np.array(res["se_cond"])
    q = float(res["qn_cond"])
    post = centres > 0
    t = beta / se
    joint_excl = bool(np.any((np.abs(beta) > q * se)[post]))
    return {"stat": float(np.nanmax(np.abs(t[post]))), "joint_excludes_zero": joint_excl,
            "qn": q, "beta_post_mean": float(beta[post].mean())}
