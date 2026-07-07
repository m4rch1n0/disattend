#!/usr/bin/env python3
"""Phase 3 confirmatory analyzer: co-primaries, TOST verdict, controls.

Implements experiments/phase3_main/PREREG.md (incl. section-9 amendments)
verbatim. This file is the "analyzer" of the PREREG blinding rule (section 8):
it is frozen by commit BEFORE any cross-model statistic is computed on Phase-3
data. Development and validation used ONLY the Phase-2 pilot substrate
(--pilot mode, disjoint seeds, exploratory) and synthetic checks.

Frozen analysis choices (everything the PREREG left to the analyzer):

* Contrast convention, all three metrics benign-free where possible:
  erank_rv / flatness_ratio: contrast_i = rand_i - pgd_i (positive = PGD
  collapses more). entropy (echo): contrast_i = |pgd_i - ben_i| -
  |rand_i - ben_i| (the pilot-gate |shift| form; entropy's effect is
  two-sided so the raw difference has no stable sign).
* Co-primary B mass floor (PREREG 3, rider a): a pair (model1, model2) enters
  D only if BOTH sides have clamped total mass sum_l max(c_l,i, 0) > 0
  (S is 0/0-undefined at zero mass); no additional positive floor. Fixed from
  the pilot substrate: across candidate floors {0, .005, .01, .02, .033} the
  pilot per-seed D moves +0.159 -> +0.181, all far above SESOI-B = 0.10, so
  the minimal-exclusion rule is adopted (most seeds kept, smallest forecast
  D). The same floor grid is reported on Phase-3 data as sensitivity (no
  veto). Exclusion counts reported per model.
* Each family member contributes one p = max(p_t, p_Wilcoxon), both
  two-sided; Holm at alpha = .05 over the 5 members (PREREG 3).
* TOST: t-based 90% CI (mean +- t_{.95,n-1} SE) within +-SESOI (PREREG 4).
  Bootstrap percentile CIs (10k resamples, rng seed 21260705) reported for
  estimation (95%) alongside.
* NFE-transfer "Holm-significant" (PREREG 6): Holm at .05 across the 4
  transfer tests (A/B x NFE 50/100), separate from the confirmatory family.
* Matched-quality control (PREREG 6): survival for a co-primary = paired
  TOST-equivalence of DiT@FID95 vs DiT@FID68 within that co-primary's SESOI
  AND same sign of (DiT@FID95 - UNet) as the primary estimate.
* Convergence ablation (PREREG 6): read from <slug>.it{20,80} run dirs if
  present, else reported as PENDING (does not block the co-primaries; it
  gates the "40 iters suffices" justification only).

Pilot validation (--pilot): reproduces the DESIGN_REVIEW_20260706 substrate
numbers exactly (A: DiT .0329/SD .0288, UNet .0457/SD .0369, Delta -.0128/
SD .0318, corr +.56, d 1.14/1.24; B zero-fill D +.156/SD .233, profile
+.117) plus the frozen-rule D (+.159, n=94). Asserted with tolerances; any
drift fails loudly.

Usage:
  uv run python scripts/analyze_phase3.py --pilot     # validation, no Phase-3 data
  uv run python scripts/analyze_phase3.py             # confirmatory run (post-freeze)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from scipy import stats

ALPHA = 0.05
SESOI_A = 0.010
SESOI_B = 0.10
PRIMARY_EPS = 0.05
EPS_GRID = [0.01, 0.02, 0.05, 0.1]
N_BOOT = 10_000
BOOT_SEED = 21260705
NFE_T_MIN = 0.72
NFE_LIST = [50, 100]
FLOOR_GRID = [0.0, 0.005, 0.01, 0.02, 0.033]  # 0.0 = frozen primary rule
L2_RATIO_BAND = (0.8, 1.25)

DIT_ALL = [f"blocks.{i}.attn" for i in range(12)]
DIT_IN = DIT_ALL[:6]                      # blocks 0-5 (L_in/L = 6/12)
DIT_IN_MINUS = DIT_ALL[:5]                # sensitivity: 0-4 / 5-11
DIT_IN_PLUS = DIT_ALL[:7]                 # sensitivity: 0-6 / 7-11
UNET_IN = ["input_blocks.4.1", "input_blocks.5.1"]
UNET_OUT = [f"output_blocks.{i}.1" for i in (6, 7, 8)]
UNET_ALL = UNET_IN + UNET_OUT             # depth order, L_in/L = 2/5

MEMBERS = ["A_magnitude", "B_structure", "dose_response",
           "entropy_echo", "flatness_echo"]


def late_window(n_steps: int) -> list[int]:
    """0-indexed post-step window t = (i+1)/n >= NFE_T_MIN (PREREG 2/6)."""
    idx = [i for i in range(n_steps) if (i + 1) / n_steps >= NFE_T_MIN - 1e-9]
    assert idx and idx == list(range(idx[0], n_steps)), idx
    return idx


# ---------- substrate access ----------


def agg_scalar(stacked: dict, metric: str, steps: list[int] | None,
               token_filter: int | None = 256) -> np.ndarray:
    """Flat mean over (locus layer, step window, head) -> (B,) per sample.

    Equals the PREREG-2 nested mean (head count is constant within each
    model's N=256 locus). steps=None -> full trajectory.
    """
    per = []
    for name, d in stacked.items():
        if token_filter is not None and int(d["n_tokens"]) != token_filter:
            continue
        x = d[metric] if steps is None else d[metric][steps]
        per.append(x.permute(1, 0, 2).flatten(1))  # (B, T*H)
    assert per, f"no layers at token_filter={token_filter}"
    return torch.cat(per, dim=1).mean(dim=1).double().numpy()


def per_layer_contrast(rand: dict, pgd: dict, steps: list[int],
                       layers: list[str], metric: str = "erank_rv"
                       ) -> dict[str, np.ndarray]:
    """Per-layer (B,) contrast rand - pgd, mean over heads and the window."""
    out = {}
    for name in layers:
        r = rand[name][metric][steps].mean(dim=(0, 2)).double()
        p = pgd[name][metric][steps].mean(dim=(0, 2)).double()
        out[name] = (r - p).numpy()
    return out


def load_leg(out_dir: Path, eps_list: list[float], with_nfe: bool) -> dict:
    leg = {"ben": torch.load(out_dir / "benign.pt", map_location="cpu",
                             weights_only=False)}
    leg["eps"] = {e: torch.load(out_dir / f"eps_{e}.pt", map_location="cpu",
                                weights_only=False) for e in eps_list}
    leg["nfe"] = {}
    if with_nfe:
        for nfe in NFE_LIST:
            f = torch.load(out_dir / f"nfe{nfe}_eps{PRIMARY_EPS}.pt",
                           map_location="cpu", weights_only=False)
            assert f["step_indices"] == late_window(nfe) and f["t_min"] == NFE_T_MIN
            leg["nfe"][nfe] = f
    return leg


# ---------- statistics ----------


def paired_p(x: np.ndarray) -> dict:
    """Two-sided paired t and Wilcoxon on a difference vector; p = max."""
    t = stats.ttest_1samp(x, 0.0)
    if np.allclose(x, 0):
        w_p = 1.0
    else:
        w_p = float(stats.wilcoxon(x, zero_method="wilcox").pvalue)
    return {"t": float(t.statistic), "p_t": float(t.pvalue), "p_wilcoxon": w_p,
            "p": float(max(t.pvalue, w_p))}


def t_ci(x: np.ndarray, level: float) -> list[float]:
    n = len(x)
    half = stats.t.ppf(0.5 + level / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return [float(x.mean() - half), float(x.mean() + half)]


def boot_ci(x: np.ndarray, level: float, seed: int = BOOT_SEED) -> list[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (N_BOOT, len(x)))
    means = x[idx].mean(axis=1)
    q = (1 - level) / 2
    return [float(np.percentile(means, 100 * q)),
            float(np.percentile(means, 100 * (1 - q)))]


def member_stats(x: np.ndarray) -> dict:
    """Full inferential block for one paired difference vector."""
    return {
        "n": int(len(x)), "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)),
        "se": float(x.std(ddof=1) / np.sqrt(len(x))),
        **paired_p(x),
        "ci95_boot": boot_ci(x, 0.95), "ci90_t": t_ci(x, 0.90),
    }


def holm(pvals: dict[str, float], alpha: float = ALPHA) -> dict[str, dict]:
    """Holm step-down; returns per-key adjusted p and rejection."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running_max, rejecting = {}, 0.0, True
    for rank, (k, p) in enumerate(items):
        adj = min(1.0, max(running_max, (m - rank) * p))
        running_max = adj
        rejecting = rejecting and adj < alpha
        out[k] = {"p": p, "p_holm": adj, "reject": bool(rejecting)}
    return out


def tost_equivalent(x: np.ndarray, sesoi: float) -> dict:
    lo, hi = t_ci(x, 0.90)
    return {"ci90_t": [lo, hi], "sesoi": sesoi,
            "equivalent": bool(-sesoi < lo and hi < sesoi)}


# ---------- co-primary statistics ----------


def erank_contrast(leg: dict, eps: float, steps: list[int] | None,
                   metric: str = "erank_rv", token_filter: int | None = 256
                   ) -> np.ndarray:
    d = leg["eps"][eps]
    r = agg_scalar(d["attn_rand"], metric, steps, token_filter)
    p = agg_scalar(d["attn_pgd"], metric, steps, token_filter)
    return r - p


def entropy_echo_contrast(leg: dict, eps: float, steps: list[int]) -> np.ndarray:
    b = agg_scalar(leg["ben"]["attn"], "entropy", steps)
    d = leg["eps"][eps]
    r = agg_scalar(d["attn_rand"], "entropy", steps)
    p = agg_scalar(d["attn_pgd"], "entropy", steps)
    return np.abs(p - b) - np.abs(r - b)


def share_excess(pl: dict[str, np.ndarray], inp: list[str], alln: list[str]
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Per-seed excess input share S_i and clamped total mass (B,)."""
    cp = {n: np.maximum(pl[n], 0.0) for n in alln}
    tot = np.sum([cp[n] for n in alln], axis=0)
    ins = np.sum([cp[n] for n in inp], axis=0)
    with np.errstate(invalid="ignore"):
        s = ins / tot - len(inp) / len(alln)
    return s, tot


def share_profile(pl: dict[str, np.ndarray], inp: list[str],
                  alln: list[str]) -> float:
    """S computed on the seed-mean profile (nearly bias-free framing)."""
    cp = {n: max(float(pl[n].mean()), 0.0) for n in alln}
    tot = sum(cp[n] for n in alln)
    return sum(cp[n] for n in inp) / tot - len(inp) / len(alln)


def paired_D(s1: np.ndarray, tot1: np.ndarray, s2: np.ndarray,
             tot2: np.ndarray, floor: float) -> tuple[np.ndarray, dict]:
    keep = (tot1 > floor) & (tot2 > floor)
    excl = {"n_pairs_kept": int(keep.sum()),
            "n_excluded_side1": int((tot1 <= floor).sum()),
            "n_excluded_side2": int((tot2 <= floor).sum())}
    return s1[keep] - s2[keep], excl


def dose_slopes(leg: dict, steps: list[int]) -> np.ndarray:
    """Per-seed OLS slope of the absolute erank contrast on log(eps)."""
    x = np.log(np.array(EPS_GRID))
    xc = x - x.mean()
    ys = np.stack([erank_contrast(leg, e, steps) for e in EPS_GRID])  # (4, B)
    return (xc[:, None] * (ys - ys.mean(axis=0))).sum(axis=0) / (xc ** 2).sum()


# ---------- analysis blocks ----------


def coprimary_A(dit: dict, unet: dict, steps: list[int]) -> tuple[dict, np.ndarray]:
    c_d = erank_contrast(dit, PRIMARY_EPS, steps)
    c_u = erank_contrast(unet, PRIMARY_EPS, steps)
    delta = c_d - c_u
    res = {
        "contrast_dit": {"mean": float(c_d.mean()), "sd": float(c_d.std(ddof=1)),
                         "d_within": float(c_d.mean() / c_d.std(ddof=1))},
        "contrast_unet": {"mean": float(c_u.mean()), "sd": float(c_u.std(ddof=1)),
                          "d_within": float(c_u.mean() / c_u.std(ddof=1))},
        "pair_corr": float(np.corrcoef(c_d, c_u)[0, 1]),
        "delta": member_stats(delta),
        "tost": tost_equivalent(delta, SESOI_A),
    }
    return res, delta


def coprimary_B(dit: dict, unet: dict, steps: list[int]) -> tuple[dict, np.ndarray]:
    pl_d = per_layer_contrast(dit["eps"][PRIMARY_EPS]["attn_rand"],
                              dit["eps"][PRIMARY_EPS]["attn_pgd"], steps, DIT_ALL)
    pl_u = per_layer_contrast(unet["eps"][PRIMARY_EPS]["attn_rand"],
                              unet["eps"][PRIMARY_EPS]["attn_pgd"], steps, UNET_ALL)
    s_d, tot_d = share_excess(pl_d, DIT_IN, DIT_ALL)
    s_u, tot_u = share_excess(pl_u, UNET_IN, UNET_ALL)
    D, excl = paired_D(s_d, tot_d, s_u, tot_u, floor=0.0)
    D_prof = share_profile(pl_d, DIT_IN, DIT_ALL) - share_profile(pl_u, UNET_IN, UNET_ALL)

    floor_sens = {}
    for f in FLOOR_GRID:
        Df, ex = paired_D(s_d, tot_d, s_u, tot_u, floor=f)
        floor_sens[str(f)] = {"D_mean": float(Df.mean()), **ex}

    bound_sens = {}
    for lbl, inp in (("dit_in_0-4", DIT_IN_MINUS), ("dit_in_0-6", DIT_IN_PLUS)):
        s_alt, tot_alt = share_excess(pl_d, inp, DIT_ALL)
        Da, _ = paired_D(s_alt, tot_alt, s_u, tot_u, floor=0.0)
        bound_sens[lbl] = {"D_mean": float(Da.mean()),
                           "D_profile": float(share_profile(pl_d, inp, DIT_ALL)
                                              - share_profile(pl_u, UNET_IN, UNET_ALL))}

    dec = np.mean([pl_u[n] for n in UNET_OUT], axis=0)
    enc = np.mean([pl_u[n] for n in UNET_IN], axis=0)
    blk0 = np.maximum(pl_d["blocks.0.attn"], 0.0)
    with np.errstate(invalid="ignore"):
        blk0_share = np.where(tot_d > 0, blk0 / tot_d - 1 / 12, np.nan)
    blk0_share = blk0_share[~np.isnan(blk0_share)]

    res = {
        "D": member_stats(D), "D_profile": float(D_prof),
        "exclusions": excl,
        "tost": tost_equivalent(D, SESOI_B),
        "sesoi_pass_both": bool(abs(D.mean()) >= SESOI_B and abs(D_prof) >= SESOI_B),
        "floor_sensitivity": floor_sens,
        "boundary_sensitivity": bound_sens,
        "corroborations": {
            "unet_decoder_minus_encoder": member_stats(dec - enc),
            "dit_block0_excess_share": member_stats(blk0_share),
        },
        "profiles_mean": {
            "dit": {n: float(pl_d[n].mean()) for n in DIT_ALL},
            "unet": {n: float(pl_u[n].mean()) for n in UNET_ALL},
        },
    }
    return res, D


def verdict_class(reject_holm: bool, magnitude_pass: bool,
                  tost_eq: bool, control_survives: bool | None) -> str:
    """PREREG section-4 three-way rule (total function).

    magnitude_pass: |point estimate| >= SESOI on the registered framing(s) --
    for A the per-seed mean; for B BOTH the per-seed and the seed-mean-profile
    D (PREREG 5).
    """
    if reject_holm and magnitude_pass:
        if control_survives is None:
            return "PENDING-CONTROL"
        return "DISTINGUISHABLE" if control_survives else "TRACKS-CONVERGENCE/AMBIGUOUS"
    if reject_holm:
        return "DETECTABLE-BELOW-SESOI"
    return "EQUIVALENT-WITHIN-SESOI" if tost_eq else "INCONCLUSIVE"


# ---------- controls ----------


def control_fid95(dit: dict, dit95: dict, unet: dict, steps: list[int],
                  primary_sign_A: float, primary_sign_B: float) -> dict:
    c_95 = erank_contrast(dit95, PRIMARY_EPS, steps)
    c_68 = erank_contrast(dit, PRIMARY_EPS, steps)
    c_un = erank_contrast(unet, PRIMARY_EPS, steps)
    a_eq = tost_equivalent(c_95 - c_68, SESOI_A)
    a_sign = float(np.sign((c_95 - c_un).mean()))

    def pl(leg, layers):
        d = leg["eps"][PRIMARY_EPS]
        return per_layer_contrast(d["attn_rand"], d["attn_pgd"], steps, layers)
    s95, t95 = share_excess(pl(dit95, DIT_ALL), DIT_IN, DIT_ALL)
    s68, t68 = share_excess(pl(dit, DIT_ALL), DIT_IN, DIT_ALL)
    sun, tun = share_excess(pl(unet, UNET_ALL), UNET_IN, UNET_ALL)
    dB, exB = paired_D(s95, t95, s68, t68, floor=0.0)
    b_eq = tost_equivalent(dB, SESOI_B)
    d95u, _ = paired_D(s95, t95, sun, tun, floor=0.0)
    b_sign = float(np.sign(d95u.mean()))

    return {
        "A": {"delta_95_vs_68": member_stats(c_95 - c_68), "tost": a_eq,
              "sign_95_vs_unet": a_sign, "sign_matches_primary":
              bool(a_sign == primary_sign_A),
              "survives": bool(a_eq["equivalent"] and a_sign == primary_sign_A)},
        "B": {"D_95_vs_68": member_stats(dB), "tost": b_eq, "exclusions": exB,
              "sign_95_vs_unet": b_sign, "sign_matches_primary":
              bool(b_sign == primary_sign_B),
              "survives": bool(b_eq["equivalent"] and b_sign == primary_sign_B)},
    }


def control_nfe(dit: dict, unet: dict, primary_A: float, primary_B: float) -> dict:
    """Transfer of both co-primaries at NFE 50/100 on the registered window."""
    res, pvals = {}, {}
    for nfe in NFE_LIST:
        fd, fu = dit["nfe"][nfe], unet["nfe"][nfe]
        steps_all = list(range(len(fd["step_indices"])))  # files hold the window only
        cA_d = agg_scalar(fd["rand"], "erank_rv", steps_all) - \
            agg_scalar(fd["pgd"], "erank_rv", steps_all)
        cA_u = agg_scalar(fu["rand"], "erank_rv", steps_all) - \
            agg_scalar(fu["pgd"], "erank_rv", steps_all)
        deltaA = cA_d - cA_u
        s_d, tot_d = share_excess(per_layer_contrast(fd["rand"], fd["pgd"],
                                                     steps_all, DIT_ALL), DIT_IN, DIT_ALL)
        s_u, tot_u = share_excess(per_layer_contrast(fu["rand"], fu["pgd"],
                                                     steps_all, UNET_ALL), UNET_IN, UNET_ALL)
        DB, exB = paired_D(s_d, tot_d, s_u, tot_u, floor=0.0)
        res[str(nfe)] = {"A": member_stats(deltaA),
                         "B": {**member_stats(DB), "exclusions": exB}}
        pvals[f"A@{nfe}"] = res[str(nfe)]["A"]["p"]
        pvals[f"B@{nfe}"] = res[str(nfe)]["B"]["p"]
    hm = holm(pvals)
    for nfe in NFE_LIST:
        for key, primary in (("A", primary_A), ("B", primary_B)):
            r = res[str(nfe)][key]
            mag_ok = (abs(primary) / 2 <= abs(r["mean"]) <= abs(primary) * 2) \
                if primary != 0 else False
            r["holm"] = hm[f"{key}@{nfe}"]
            r["survives"] = bool(np.sign(r["mean"]) == np.sign(primary)
                                 and hm[f"{key}@{nfe}"]["reject"] and mag_ok)
    return res


def control_attack(dit: dict, unet: dict) -> dict:
    per_eps = {}
    for e in EPS_GRID:
        r = float(dit["eps"][e]["l2_out"]["pgd"].mean()
                  / unet["eps"][e]["l2_out"]["pgd"].mean())
        per_eps[str(e)] = {"l2_ratio_dit_over_unet": r,
                           "in_band": bool(L2_RATIO_BAND[0] <= r <= L2_RATIO_BAND[1])}
    return {"band": list(L2_RATIO_BAND), "per_eps": per_eps,
            "all_in_band": all(v["in_band"] for v in per_eps.values())}


def control_saturation(leg: dict, steps: list[int]) -> dict:
    c05 = float(erank_contrast(leg, 0.05, steps).mean())
    c10 = float(erank_contrast(leg, 0.1, steps).mean())
    return {"contrast@0.05": c05, "contrast@0.1": c10,
            "saturating": bool(c10 < 2 * c05)}


def control_convergence(base_dir: Path, slug: str, main_leg: dict,
                        steps: list[int]) -> dict:
    """{20,40,80}-iter plateau; ablation dirs <slug>.it{20,80} (optional)."""
    out = {"c40": float(erank_contrast(main_leg, PRIMARY_EPS, steps).mean())}
    for it in (20, 80):
        d = base_dir / f"{slug}.it{it}"
        if not (d / f"eps_{PRIMARY_EPS}.pt").exists():
            out[f"c{it}"] = None
            continue
        leg = load_leg(d, [PRIMARY_EPS], with_nfe=False)
        out[f"c{it}"] = float(erank_contrast(leg, PRIMARY_EPS, steps).mean())
    if out["c80"] is None:
        out["status"] = "PENDING (ablation runs not found)"
    else:
        plateau = abs(out["c80"] - out["c40"]) <= 0.10 * abs(out["c40"])
        out["plateau"] = bool(plateau)
        out["status"] = "plateau OK" if plateau else "NOT converged at 40"
    return out


# ---------- descriptive framings (no veto) ----------


def descriptives(dit: dict, unet: dict, steps: list[int]) -> dict:
    b_d = agg_scalar(dit["ben"]["attn"], "erank_rv", steps)
    b_u = agg_scalar(unet["ben"]["attn"], "erank_rv", steps)
    c_d = erank_contrast(dit, PRIMARY_EPS, steps)
    c_u = erank_contrast(unet, PRIMARY_EPS, steps)
    all_d = erank_contrast(dit, PRIMARY_EPS, steps, token_filter=None)
    all_u = erank_contrast(unet, PRIMARY_EPS, steps, token_filter=None)
    full_d = erank_contrast(dit, PRIMARY_EPS, None)
    full_u = erank_contrast(unet, PRIMARY_EPS, None)
    ben_by_N = {}
    for name, d in unet["ben"]["attn"].items():
        ben_by_N.setdefault(int(d["n_tokens"]), []).append(
            float(d["erank_rv"][steps].mean()))
    return {
        "benign_level": {"dit": float(b_d.mean()), "unet": float(b_u.mean())},
        "relative_contrast_over_mean_benign": {
            "dit": float(c_d.mean() / b_d.mean()),
            "unet": float(c_u.mean() / b_u.mean())},
        "cell_count_caveat": "per-seed cells/scalar: DiT 12x8x12=1152, UNet 5x8x4=160",
        "all_layer_locus_delta": {  # locus-contingency disclosure (PREREG 3)
            "dit": float(all_d.mean()), "unet": float(all_u.mean()),
            "delta": float((all_d - all_u).mean())},
        "full_trajectory_delta": {
            "dit": float(full_d.mean()), "unet": float(full_u.mean()),
            "delta": member_stats(full_d - full_u)},
        "per_eps_delta_robustness": {
            str(e): member_stats(erank_contrast(dit, e, steps)
                                 - erank_contrast(unet, e, steps))
            for e in EPS_GRID if e != PRIMARY_EPS},
        "unet_benign_erank_by_N": {str(k): float(np.mean(v))
                                   for k, v in sorted(ben_by_N.items())},
    }


# ---------- pilot validation ----------

PILOT_EXPECT = {  # DESIGN_REVIEW_20260706 substrate numbers
    "A_dit_mean": 0.0329, "A_dit_sd": 0.0288,
    "A_unet_mean": 0.0457, "A_unet_sd": 0.0369,
    "delta_mean": -0.0128, "delta_sd": 0.0318, "corr": 0.56,
    "d_dit": 1.14, "d_unet": 1.24,
    "D_zerofill": 0.156, "D_zerofill_sd": 0.233, "D_profile": 0.117,
    "D_floor0": 0.159,
}


def pilot_as_leg(slug: str) -> dict:
    """Adapt the pilot ab_results.pt (K=3 draws) to the Phase-3 leg layout."""
    ab = torch.load(REPO_ROOT / f"experiments/phase2_pilot/{slug}/ab_results.pt",
                    map_location="cpu", weights_only=False)
    rand_keys = sorted(k for k in ab["attn"] if k.startswith("rand"))
    rand_mean = {}
    for name in ab["attn"]["ben"]:
        rand_mean[name] = {
            k: torch.stack([ab["attn"][r][name][k] for r in rand_keys]).mean(0)
            for k in ("entropy", "flatness_ratio", "erank_rv")}
        rand_mean[name]["n_tokens"] = ab["attn"]["ben"][name]["n_tokens"]
    return {"ben": {"attn": ab["attn"]["ben"]},
            "eps": {PRIMARY_EPS: {"attn_rand": rand_mean,
                                  "attn_pgd": ab["attn"]["pgd"]}},
            "nfe": {}}


def validate_on_pilot() -> int:
    steps = late_window(25)
    dit, unet = pilot_as_leg("SiT-B-2"), pilot_as_leg("UNet-B")
    a, delta = coprimary_A(dit, unet, steps)
    b, D = coprimary_B(dit, unet, steps)

    got = {
        "A_dit_mean": a["contrast_dit"]["mean"], "A_dit_sd": a["contrast_dit"]["sd"],
        "A_unet_mean": a["contrast_unet"]["mean"], "A_unet_sd": a["contrast_unet"]["sd"],
        "delta_mean": a["delta"]["mean"], "delta_sd": a["delta"]["sd"],
        "corr": a["pair_corr"],
        "d_dit": a["contrast_dit"]["d_within"], "d_unet": a["contrast_unet"]["d_within"],
        "D_profile": b["D_profile"], "D_floor0": b["D"]["mean"],
    }
    # review's zero-fill convention, replicated for equivalence only
    pl_d = per_layer_contrast(dit["eps"][PRIMARY_EPS]["attn_rand"],
                              dit["eps"][PRIMARY_EPS]["attn_pgd"], steps, DIT_ALL)
    pl_u = per_layer_contrast(unet["eps"][PRIMARY_EPS]["attn_rand"],
                              unet["eps"][PRIMARY_EPS]["attn_pgd"], steps, UNET_ALL)
    s_d, tot_d = share_excess(pl_d, DIT_IN, DIT_ALL)
    s_u, tot_u = share_excess(pl_u, UNET_IN, UNET_ALL)
    zf = np.where(tot_d > 0, s_d, 0.0) - np.where(tot_u > 0, s_u, 0.0)
    got["D_zerofill"], got["D_zerofill_sd"] = float(zf.mean()), float(zf.std(ddof=1))

    ok = True
    for k, want in PILOT_EXPECT.items():
        tol = 0.005 if abs(want) < 1 else 0.01
        match = abs(got[k] - want) <= tol
        ok = ok and match
        print(f"  {k:16s} got {got[k]:+.4f}  want {want:+.4f}  "
              f"{'OK' if match else 'MISMATCH'}")
    print(f"  exclusions at floor 0: {b['exclusions']}")
    print(f"  floor sensitivity: "
          f"{ {f: round(v['D_mean'], 3) for f, v in b['floor_sensitivity'].items()} }")
    print(f"  boundary sensitivity: "
          f"{ {k: round(v['D_mean'], 3) for k, v in b['boundary_sensitivity'].items()} }")
    print(f"\npilot validation: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ---------- confirmatory run ----------


def run_confirmatory(base_dir: Path) -> int:
    steps = late_window(25)
    dit = load_leg(base_dir / "SiT-B-2", EPS_GRID, with_nfe=True)
    unet = load_leg(base_dir / "UNet-B", EPS_GRID, with_nfe=True)
    dit95 = load_leg(base_dir / "SiT-B-2-FID95", [PRIMARY_EPS], with_nfe=False)
    n = len(agg_scalar(dit["ben"]["attn"], "erank_rv", steps))
    assert n == 500, n

    a, delta = coprimary_A(dit, unet, steps)
    b, D = coprimary_B(dit, unet, steps)
    slopes = dose_slopes(dit, steps) - dose_slopes(unet, steps)
    ent = entropy_echo_contrast(dit, PRIMARY_EPS, steps) \
        - entropy_echo_contrast(unet, PRIMARY_EPS, steps)
    flat = erank_contrast(dit, PRIMARY_EPS, steps, metric="flatness_ratio") \
        - erank_contrast(unet, PRIMARY_EPS, steps, metric="flatness_ratio")

    family = {
        "A_magnitude": a["delta"], "B_structure": b["D"],
        "dose_response": member_stats(slopes),
        "entropy_echo": member_stats(ent), "flatness_echo": member_stats(flat),
    }
    hm = holm({k: family[k]["p"] for k in MEMBERS})
    for k in MEMBERS:
        family[k]["holm"] = hm[k]

    fid95 = control_fid95(dit, dit95, unet, steps,
                          primary_sign_A=float(np.sign(a["delta"]["mean"])),
                          primary_sign_B=float(np.sign(b["D"]["mean"])))
    nfe = control_nfe(dit, unet, primary_A=a["delta"]["mean"],
                      primary_B=b["D"]["mean"])
    attack = control_attack(dit, unet)
    saturation = {"dit": control_saturation(dit, steps),
                  "unet": control_saturation(unet, steps)}
    convergence = {"dit": control_convergence(base_dir, "SiT-B-2", dit, steps),
                   "unet": control_convergence(base_dir, "UNet-B", unet, steps)}

    verdicts = {
        "A": verdict_class(hm["A_magnitude"]["reject"],
                           abs(a["delta"]["mean"]) >= SESOI_A,
                           a["tost"]["equivalent"], fid95["A"]["survives"]),
        "B": verdict_class(hm["B_structure"]["reject"], b["sesoi_pass_both"],
                           b["tost"]["equivalent"], fid95["B"]["survives"]),
    }
    verdicts["overall_distinguishable"] = bool(
        verdicts["A"] == "DISTINGUISHABLE" or verdicts["B"] == "DISTINGUISHABLE")

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "scripts/analyze_phase3.py"],
            cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip())
    except OSError:
        commit, dirty = "unknown", True

    results = {
        "prereg": "experiments/phase3_main/PREREG.md",
        "analyzer_commit": commit, "analyzer_dirty": dirty,
        "alpha": ALPHA, "sesoi": {"A": SESOI_A, "B": SESOI_B},
        "primary_eps": PRIMARY_EPS, "late_steps": steps, "n_seeds": n,
        "coprimary_A": a, "coprimary_B": b,
        "family": family,
        "controls": {"fid95": fid95, "nfe_transfer": nfe, "attack": attack,
                     "saturation": saturation, "convergence": convergence},
        "descriptives": descriptives(dit, unet, steps),
        "verdicts": verdicts,
    }
    out = base_dir / "analysis.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"analyzer commit {commit[:9]}{' DIRTY' if dirty else ''}, n={n}")
    print(f"\nco-primary A (erank late, eps={PRIMARY_EPS}):")
    print(f"  DiT {a['contrast_dit']['mean']:+.4f}  UNet {a['contrast_unet']['mean']:+.4f}"
          f"  Delta {a['delta']['mean']:+.4f}  CI95 {a['delta']['ci95_boot']}"
          f"  p={a['delta']['p']:.2e} holm_reject={hm['A_magnitude']['reject']}")
    print(f"  TOST 90% CI {a['tost']['ci90_t']} vs +-{SESOI_A} -> "
          f"equivalent={a['tost']['equivalent']}")
    print(f"\nco-primary B (excess input share):")
    print(f"  D {b['D']['mean']:+.4f} (profile {b['D_profile']:+.4f})"
          f"  CI95 {b['D']['ci95_boot']}  p={b['D']['p']:.2e}"
          f"  holm_reject={hm['B_structure']['reject']}")
    print(f"  exclusions {b['exclusions']}  sesoi_pass_both={b['sesoi_pass_both']}")
    print(f"\nfamily (Holm): " + "  ".join(
        f"{k}={'R' if hm[k]['reject'] else '-'}" for k in MEMBERS))
    print(f"controls: fid95 A={fid95['A']['survives']} B={fid95['B']['survives']}"
          f"  attack_band={attack['all_in_band']}"
          f"  sat dit={saturation['dit']['saturating']}"
          f" unet={saturation['unet']['saturating']}")
    for nfe_k, r in nfe.items():
        print(f"  nfe{nfe_k}: A survives={r['A']['survives']} "
              f"B survives={r['B']['survives']}")
    print(f"  convergence: dit={convergence['dit']['status']} "
          f"unet={convergence['unet']['status']}")
    print(f"\nVERDICT  A: {verdicts['A']}   B: {verdicts['B']}")
    print(f"overall distinguishable: {verdicts['overall_distinguishable']}")
    print(f"\nwrote {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true",
                    help="validation on the Phase-2 pilot substrate only")
    ap.add_argument("--base-dir", default="experiments/phase3_main")
    args = ap.parse_args()
    if args.pilot:
        return validate_on_pilot()
    return run_confirmatory(REPO_ROOT / args.base_dir)


if __name__ == "__main__":
    sys.exit(main())
