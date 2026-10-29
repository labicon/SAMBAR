"""RETAIN baseline: alpha * SFT + (1 - alpha) * reference checkpoint.

Run with --sft CKPT --pretrained CKPT --alpha 0.5 --out OUTPUT.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil

import flax.traverse_util as traverse_util
import numpy as np
import orbax.checkpoint as ocp

import openpi.models.model as _model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sft", required=True, help="SFT checkpoint step dir (containing params/).")
    ap.add_argument("--pretrained", required=True, help="Pretrained checkpoint step dir (containing params/).")
    ap.add_argument("--out", required=True, help="Output checkpoint dir to create (must not exist).")
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight on SFT: merged = alpha*sft + (1-alpha)*pretrained. Default 0.5.",
    )
    ap.add_argument(
        "--assets-from",
        default=None,
        help="Dir whose assets/ (norm stats) to copy into <out>. Default: --sft "
        "(SFT and its pretrain share identical norm stats anyway).",
    )
    args = ap.parse_args()

    sft_dir = pathlib.Path(args.sft)
    pre_dir = pathlib.Path(args.pretrained)
    out_dir = pathlib.Path(args.out)
    alpha = args.alpha
    if not (0.0 <= alpha <= 1.0):
        raise SystemExit(f"--alpha must be in [0,1], got {alpha}")
    if out_dir.exists():
        raise SystemExit(f"--out already exists, refusing to overwrite: {out_dir}")

    print(f"[merge] SFT        = {sft_dir / 'params'}")
    sft = _model.restore_params(sft_dir / "params", restore_type=np.ndarray)
    print(f"[merge] pretrained = {pre_dir / 'params'}")
    pre = _model.restore_params(pre_dir / "params", restore_type=np.ndarray)

    fs = traverse_util.flatten_dict(sft)
    fp = traverse_util.flatten_dict(pre)
    if set(fs) != set(fp):
        only_s = sorted(set(fs) - set(fp))[:3]
        only_p = sorted(set(fp) - set(fs))[:3]
        raise SystemExit(f"param key mismatch — only in sft: {only_s}; only in pretrained: {only_p}")

    merged_flat = {}
    for k in fs:
        a, b = fs[k], fp[k]
        if a.shape != b.shape:
            raise SystemExit(f"shape mismatch at {'/'.join(k)}: {a.shape} vs {b.shape}")
        # interpolate in float64 for precision, cast back to the stored dtype.
        merged_flat[k] = (alpha * a.astype(np.float64) + (1.0 - alpha) * b.astype(np.float64)).astype(a.dtype)
    merged = traverse_util.unflatten_dict(merged_flat)
    print(
        f"[merge] interpolated {len(merged_flat)} arrays at alpha={alpha} "
        f"(merged = {alpha}*sft + {1 - alpha}*pretrained)"
    )

    out_dir.mkdir(parents=True, exist_ok=False)
    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save((out_dir / "params").resolve(), ocp.args.PyTreeSave({"params": merged}))
    print(f"[merge] saved params -> {out_dir / 'params'}")

    assets_src = pathlib.Path(args.assets_from) if args.assets_from else sft_dir
    if (assets_src / "assets").exists():
        shutil.copytree(assets_src / "assets", out_dir / "assets")
        print(f"[merge] copied assets <- {assets_src / 'assets'}")
    else:
        print(f"[merge] WARNING: no assets/ at {assets_src}; serving will rely on config norm stats")

    # sanity: reload the merged checkpoint exactly as serve_policy would.
    rt = traverse_util.flatten_dict(_model.restore_params(out_dir / "params", restore_type=np.ndarray))
    assert set(rt) == set(merged_flat), "reloaded merged params structure differs!"
    print(f"[merge] reload OK: {len(rt)} arrays. Serve with --policy.dir {out_dir}")


if __name__ == "__main__":
    main()
