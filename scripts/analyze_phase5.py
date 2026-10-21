#!/usr/bin/env python3
"""Phase 5 analyzer (post-thesis, exploratory): guidance ablation and official
SiT checkpoints. Reuses the Phase-3 statistics verbatim (imported from
scripts/analyze_phase3.py and scripts/analyze_phase3_detector.py); nothing
here is pre-registered and every output is labelled as such.

Per leg (any model, any depth): late-window N=256 effective-rank contrast
c = M_rand - M_pgd (mean, SD, Cohen's d, bootstrap CI95, paired p), benign
level and relative susceptibility, full-trajectory contrast, per-layer
profile and input-side excess share (first half of the blocks for a
transformer; the Phase-3 encoder set for the UNet), benign per-block entropy
and erank (attention-sink check), within-model detector (AUC, 5%-FPR
operating point) plus the transfer of the Phase-3 DiT z-unit threshold,
displacement and LPIPS ratios, and the energy LSE shift when the energy
probe ran. Paired readings: cfg 1.5 vs the Phase-3 cfg 1.0 leg (same seeds),
cross-model Delta_A and D at cfg 1.5, official B/2 vs our DiT.

Legs that do not exist yet are skipped, so the script can run while the
queue is still going. Writes <dir>/analysis.json for each of the two dirs.

  .venv/bin/python scripts/analyze_phase5.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch

from analyze_phase3 import (PRIMARY_EPS, SESOI_A, SESOI_B, UNET_ALL, UNET_IN,
                            agg_scalar, late_window, load_leg, member_stats,
                            paired_D, per_layer_contrast, share_excess,
                            share_profile, tost_equivalent)
from analyze_phase3_detector import FPR_TARGET, auc_ci, auc_low

P3 = REPO_ROOT / "experiments/phase3_main"
STEPS = late_window(25)


# ---------- helpers ----------

def layers_of(leg: dict) -> tuple[list[str], list[str]]:
    """(all N=256 layers in depth order, input-side subset)."""
    names = [n for n, d in leg["ben"]["attn"].items() if int(d["n_tokens"]) == 256]
    if all(n.startswith("blocks.") for n in names):
        names = sorted(names, key=lambda n: int(n.split(".")[1]))
        return names, names[: len(names) // 2]
    return UNET_ALL, UNET_IN


def load_opt(path: Path, eps_list=(PRIMARY_EPS,)) -> dict | None:
    if not (path / "meta.json").exists():
        return None
    leg = load_leg(path, list(eps_list), with_nfe=False)
    leg["meta"] = json.load(open(path / "meta.json"))
    unc = path / f"eps_{PRIMARY_EPS}_uncond.pt"
    if unc.exists():
        leg["uncond"] = torch.load(unc, map_location="cpu", weights_only=False)
        leg["uncond_ben"] = torch.load(path / "benign_uncond.pt", map_location="cpu",
                                       weights_only=False)
    en = path / f"energy_eps{PRIMARY_EPS}.pt"
    if en.exists():
        leg["energy"] = torch.load(en, map_location="cpu", weights_only=False)
    return leg


def contrast(leg: dict, steps=STEPS, token_filter=256) -> np.ndarray:
    d = leg["eps"][PRIMARY_EPS]
    return (agg_scalar(d["attn_rand"], "erank_rv", steps, token_filter)
            - agg_scalar(d["attn_pgd"], "erank_rv", steps, token_filter))


def energy_scalar(stacked: dict, steps=STEPS) -> np.ndarray:
    per = [d["row_lse"][steps].permute(1, 0, 2).flatten(1)
           for d in stacked.values() if int(d["n_tokens"]) == 256]
    return torch.cat(per, dim=1).mean(dim=1).double().numpy()


def with_d(x: np.ndarray) -> dict:
    out = member_stats(x)
    out["d"] = float(x.mean() / x.std(ddof=1))
    return out


# ---------- per-leg block ----------

def leg_summary(leg: dict, ref_dit_scores: dict | None) -> dict:
    alln, inp = layers_of(leg)
    d = leg["eps"][PRIMARY_EPS]
    ben = agg_scalar(leg["ben"]["attn"], "erank_rv", STEPS)
    pgd = agg_scalar(d["attn_pgd"], "erank_rv", STEPS)
    rand = agg_scalar(d["attn_rand"], "erank_rv", STEPS)
    c = rand - pgd
    pl = per_layer_contrast(d["attn_rand"], d["attn_pgd"], STEPS, alln)
    s, tot = share_excess(pl, inp, alln)
    s_ok = s[tot > 0]

    # benign per-block levels (late window), sink = block with the lowest erank
    blocks = {}
    for n in alln:
        bd = leg["ben"]["attn"][n]
        blocks[n] = {"erank_rv": float(bd["erank_rv"][STEPS].mean()),
                     "entropy": float(bd["entropy"][STEPS].mean())}
    sink = min(blocks, key=lambda n: blocks[n]["erank_rv"])
    flat = max(blocks, key=lambda n: blocks[n]["entropy"])

    # detector, within model and with the Phase-3 DiT z-threshold
    tau = float(np.quantile(ben, FPR_TARGET))
    det = {"auc_pgd": auc_low(ben, pgd), "auc_pgd_ci95": auc_ci(ben, pgd),
           "auc_rand": auc_low(ben, rand),
           "tau_5pct_fpr": tau, "tpr_pgd": float((pgd < tau).mean()),
           "tpr_rand": float((rand < tau).mean())}
    if ref_dit_scores is not None:
        mu, sd = ben.mean(), ben.std(ddof=1)
        det["transfer_from_phase3_dit"] = {
            "tau_z": ref_dit_scores["tau_z"],
            "fpr": float(((ben - mu) / sd < ref_dit_scores["tau_z"]).mean()),
            "tpr_pgd": float(((pgd - mu) / sd < ref_dit_scores["tau_z"]).mean()),
            "tpr_rand": float(((rand - mu) / sd < ref_dit_scores["tau_z"]).mean())}

    l2p, l2r = d["l2_out"]["pgd"].double().numpy(), d["l2_out"]["rand"].double().numpy()
    lpp, lpr = d["lpips"]["pgd"].double().numpy(), d["lpips"]["rand"].double().numpy()
    out = {
        "model": leg["meta"]["model"], "cfg_scale": leg["meta"]["cfg_scale"],
        "n_seeds": int(len(c)), "n_layers_locus": len(alln), "input_side": inp,
        "benign_level": float(ben.mean()),
        "contrast_late": with_d(c),
        "contrast_full_trajectory": with_d(contrast(leg, None)),
        "relative_susceptibility": float(c.mean() / ben.mean()),
        "pgd_shift_vs_benign": with_d(pgd - ben),
        "rand_shift_vs_benign": with_d(rand - ben),
        "profile_mean": {n: float(pl[n].mean()) for n in alln},
        "input_share_excess": {"profile": float(share_profile(pl, inp, alln)),
                               "per_seed": with_d(s_ok), "n_seeds_with_mass": int(len(s_ok))},
        "benign_blocks_late": blocks, "benign_min_erank_block": sink,
        "benign_max_entropy_block": flat,
        "detector": det,
        "displacement": {"l2_pgd_mean": float(l2p.mean()), "l2_rand_mean": float(l2r.mean()),
                         "ratio_of_means": float(l2p.mean() / l2r.mean()),
                         "lpips_pgd_mean": float(lpp.mean()), "lpips_rand_mean": float(lpr.mean())},
        "peak_vram_gb": leg["meta"].get("peak_vram_gb"),
        "wall_hours": round(leg["meta"].get("wall_seconds", 0) / 3600, 2),
    }
    if "uncond" in leg:
        cu = (agg_scalar(leg["uncond"]["attn_rand"], "erank_rv", STEPS)
              - agg_scalar(leg["uncond"]["attn_pgd"], "erank_rv", STEPS))
        out["uncond_half"] = {
            "benign_level": float(agg_scalar(leg["uncond_ben"]["attn"], "erank_rv", STEPS).mean()),
            "contrast_late": with_d(cu),
            "corr_with_cond_contrast": float(np.corrcoef(c, cu)[0, 1])}
    if "energy" in leg:
        e = leg["energy"]
        eb, ep, er = (energy_scalar(e[k]) for k in ("ben", "pgd", "rand"))
        out["energy_lse_late"] = {"benign_level": float(eb.mean()),
                                  "pgd_minus_benign": with_d(ep - eb),
                                  "rand_minus_benign": with_d(er - eb),
                                  "corr_lse_shift_vs_erank_contrast": float(np.corrcoef(ep - eb, c)[0, 1])}
    return out


# ---------- paired blocks ----------

def paired_same_seeds(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str) -> dict:
    n = min(len(a), len(b))
    diff = a[:n] - b[:n]
    return {"a": label_a, "b": label_b, "n": int(n), "a_mean": float(a[:n].mean()),
            "b_mean": float(b[:n].mean()), "a_minus_b": with_d(diff),
            "corr": float(np.corrcoef(a[:n], b[:n])[0, 1])}


def cross_model(dit: dict, unet: dict, tag: str) -> dict:
    c_d, c_u = contrast(dit), contrast(unet)
    n = min(len(c_d), len(c_u))
    delta = c_d[:n] - c_u[:n]
    alln_d, in_d = layers_of(dit)
    pl_d = per_layer_contrast(dit["eps"][PRIMARY_EPS]["attn_rand"],
                              dit["eps"][PRIMARY_EPS]["attn_pgd"], STEPS, alln_d)
    pl_u = per_layer_contrast(unet["eps"][PRIMARY_EPS]["attn_rand"],
                              unet["eps"][PRIMARY_EPS]["attn_pgd"], STEPS, UNET_ALL)
    s_d, tot_d = share_excess(pl_d, in_d, alln_d)
    s_u, tot_u = share_excess(pl_u, UNET_IN, UNET_ALL)
    D, excl = paired_D(s_d[:n], tot_d[:n], s_u[:n], tot_u[:n], floor=0.0)
    return {
        "setting": tag, "n_pairs": int(n),
        "A_delta": {**member_stats(delta), "tost_sesoi_A": tost_equivalent(delta, SESOI_A),
                    "pair_corr": float(np.corrcoef(c_d[:n], c_u[:n])[0, 1]),
                    "reading": "sensitivity of the registered test, not a new verdict"},
        "B_D": {**member_stats(D), "D_profile": float(share_profile(pl_d, in_d, alln_d)
                                                      - share_profile(pl_u, UNET_IN, UNET_ALL)),
                "tost_sesoi_B": tost_equivalent(D, SESOI_B), **excl},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg-dir", default="experiments/phase5_cfg")
    ap.add_argument("--pre-dir", default="experiments/phase5_pretrained")
    args = ap.parse_args()
    cfg_dir, pre_dir = REPO_ROOT / args.cfg_dir, REPO_ROOT / args.pre_dir

    # Phase-3 reference legs (cfg 1.0): paired baselines and the z-threshold
    p3 = {"SiT-B-2": load_leg(P3 / "SiT-B-2", [PRIMARY_EPS], with_nfe=False),
          "UNet-B": load_leg(P3 / "UNet-B", [PRIMARY_EPS], with_nfe=False),
          "SiT-B-2-FID95": load_leg(P3 / "SiT-B-2-FID95", [PRIMARY_EPS], with_nfe=False)}
    ben_d = agg_scalar(p3["SiT-B-2"]["ben"]["attn"], "erank_rv", STEPS)
    ref_z = {"tau_z": float(np.quantile((ben_d - ben_d.mean()) / ben_d.std(ddof=1), FPR_TARGET))}
    p3_c = {k: contrast(v) for k, v in p3.items()}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                            capture_output=True, text=True).stdout.strip()
    stamp = {"status": "exploratory, post-thesis, not pre-registered",
             "late_window_steps": STEPS, "primary_eps": PRIMARY_EPS,
             "git_commit": commit, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    # ----- 5a: guidance ablation
    legs = {s: load_opt(cfg_dir / s) for s in ("SiT-B-2", "UNet-B")}
    res = {**stamp, "legs": {}, "paired_vs_cfg1": {}, "cross_model_cfg15": None,
           "phase3_reference_cfg1": {k: with_d(v) for k, v in p3_c.items()}}
    for s, leg in legs.items():
        if leg is None:
            print(f"[5a] {s}: not finished, skipped"); continue
        res["legs"][s] = leg_summary(leg, ref_z)
        res["paired_vs_cfg1"][s] = paired_same_seeds(contrast(leg), p3_c[s],
                                                     f"{s} cfg{leg['meta']['cfg_scale']}", f"{s} cfg1.0")
    if legs["SiT-B-2"] is not None and legs["UNet-B"] is not None:
        res["cross_model_cfg15"] = cross_model(legs["SiT-B-2"], legs["UNet-B"], "cfg 1.5")
        res["cross_model_cfg1_reference"] = cross_model(p3["SiT-B-2"], p3["UNet-B"], "cfg 1.0 (Phase 3)")
    with open(cfg_dir / "analysis.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"[5a] wrote {cfg_dir / 'analysis.json'} ({len(res['legs'])} legs)")

    # ----- 5b: official checkpoints
    slugs = ["SiT-S-2-official", "SiT-B-2-official", "SiT-L-2-official",
             "SiT-XL-2-official", "SiT-XL-2-official.cfg15"]
    legs = {s: load_opt(pre_dir / s) for s in slugs}
    res = {**stamp, "legs": {}, "paired": {},
           "phase3_reference_cfg1": {k: with_d(v) for k, v in p3_c.items()},
           "phase3_dit_benign_level": float(ben_d.mean())}
    for s, leg in legs.items():
        if leg is None:
            print(f"[5b] {s}: not finished, skipped"); continue
        res["legs"][s] = leg_summary(leg, ref_z)
    if legs["SiT-B-2-official"] is not None:
        res["paired"]["official_B2_vs_our_DiT"] = paired_same_seeds(
            contrast(legs["SiT-B-2-official"]), p3_c["SiT-B-2"], "SiT-B/2 official", "our DiT (Phase 3)")
        res["paired"]["official_B2_vs_DiT_FID95"] = paired_same_seeds(
            contrast(legs["SiT-B-2-official"]), p3_c["SiT-B-2-FID95"], "SiT-B/2 official", "DiT@FID95")
    if legs["SiT-XL-2-official"] is not None and legs["SiT-XL-2-official.cfg15"] is not None:
        res["paired"]["XL_cfg15_vs_cfg1"] = paired_same_seeds(
            contrast(legs["SiT-XL-2-official.cfg15"]), contrast(legs["SiT-XL-2-official"]),
            "SiT-XL/2 cfg1.5", "SiT-XL/2 cfg1.0")
    size = {s: {"blocks": res["legs"][s]["n_layers_locus"],
                "contrast_mean": res["legs"][s]["contrast_late"]["mean"],
                "d": res["legs"][s]["contrast_late"]["d"],
                "benign_level": res["legs"][s]["benign_level"],
                "relative_susceptibility": res["legs"][s]["relative_susceptibility"],
                "input_share_profile": res["legs"][s]["input_share_excess"]["profile"],
                "auc_pgd": res["legs"][s]["detector"]["auc_pgd"]}
            for s in slugs[:4] if s in res["legs"]}
    res["size_curve_cfg1"] = size
    with open(pre_dir / "analysis.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"[5b] wrote {pre_dir / 'analysis.json'} ({len(res['legs'])} legs)")

    for name, r in (("5a", json.load(open(cfg_dir / "analysis.json"))),
                    ("5b", json.load(open(pre_dir / "analysis.json")))):
        for s, L in r["legs"].items():
            cl = L["contrast_late"]
            print(f"[{name}] {s:26s} cfg{L['cfg_scale']:<4} n={L['n_seeds']:3d} benign {L['benign_level']:.4f} "
                  f"c {cl['mean']:+.4f} (sd {cl['sd']:.4f}, d {cl['d']:.2f}, p {cl['p']:.1e}) "
                  f"rel {L['relative_susceptibility']:.3f} share {L['input_share_excess']['profile']:+.3f} "
                  f"AUC {L['detector']['auc_pgd']:.3f} min-erank {L['benign_min_erank_block']} "
                  f"max-entropy {L['benign_max_entropy_block']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
