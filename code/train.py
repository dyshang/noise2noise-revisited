"""Train a Noise2Noise denoiser under a chosen residual loss.

Two training modes:

* ``--data-root <clean folder>`` (default): synthetic training. Clean images
  are loaded and additive zero-mean Gaussian noise with sigma uniform in
  ``[sigma_lo, sigma_hi]`` is drawn independently for x1 and x2. The
  training noise distribution is held fixed across loss variants so that
  cross-noise behaviour at test time is attributable to the loss function
  alone.

* ``--real-pairs <SIDD-Medium root>``: real-noise N2N. Two distinct camera
  shots of the same scene are paired and a common random crop returns
  (x1, x2). GT is never read in this mode -- this is the canonical
  Noise2Noise training under signal-dependent real camera noise. Train
  scenes are the deterministic 128/32 sorted split (see
  ``data.sidd_scene_split``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import N2NPairedDataset, RealN2NPairDataset
from losses import lasso_penalty, make_loss
from model import UNet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--data-root",
        default=None,
        help="clean *colour* image folder for synthetic training (e.g. ../data/CBSD400)",
    )
    src.add_argument(
        "--real-pairs",
        default=None,
        help="SIDD-Medium root for real-noise N2N training, e.g. ../data/SIDD_Medium_Srgb",
    )
    p.add_argument("--out", required=True, help="output checkpoint path")
    p.add_argument("--loss", choices=["l2", "l1", "charbonnier", "huber"], default="l1")
    p.add_argument(
        "--huber-delta",
        type=float,
        default=1.0,
        help=(
            "smooth-L1 threshold for --loss huber. On [0,1] images the "
            "default 1.0 never leaves the quadratic regime (== 0.5*MSE); "
            "use ~0.05 (2 sigma at sigma=25/255) for a true L2/L1 interpolation"
        ),
    )
    p.add_argument("--lasso", type=float, default=0.0, help="Lasso penalty on conv weights (separate from loss)")
    p.add_argument(
        "--target-outlier-frac",
        type=float,
        default=0.0,
        help=(
            "diagnostic: fraction of TARGET (x2) pixels replaced by salt/pepper "
            "extremes each step, to probe loss robustness to training-target "
            "outliers (bounded-influence mechanism). 0 disables."
        ),
    )
    p.add_argument(
        "--log-grad-norms",
        action="store_true",
        help="record global grad norm and batch max |residual| per step into the ckpt json",
    )
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--sigma-lo", type=float, default=5.0)
    p.add_argument("--sigma-hi", type=float, default=50.0)
    p.add_argument(
        "--real-length",
        type=int,
        default=5000,
        help="virtual epoch length for --real-pairs mode (random crops drawn per epoch)",
    )
    p.add_argument(
        "--sidd-npy-root",
        default=None,
        help=(
            "native-resolution mmap mode for --real-pairs: mirror tree of "
            "uint8 .npy shots written by prep_sidd_npy.py. Disables the "
            "max_side=1024 downsampling (which attenuates real noise)."
        ),
    )
    p.add_argument(
        "--supervised-gt",
        action="store_true",
        help=(
            "N2C anchor mode for --real-pairs: the target is the scene GT "
            "(matched by shot index) instead of the second noisy shot. "
            "READS GROUND TRUTH -- not self-supervised; only for the "
            "disclosed supervised-ceiling control row, never for N2N rows."
        ),
    )
    p.add_argument(
        "--virtual-length",
        type=int,
        default=None,
        help=(
            "virtual epoch length for synthetic mode (random crops per epoch, "
            "images round-robin). Default None = legacy one crop per image. "
            "Set 5000 to match the --real-pairs step budget."
        ),
    )
    p.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        ),
    )
    # macOS fork+PIL+torch is unstable for multi-hour runs => default 0.
    # Windows / Linux can safely set 2-4 for a non-trivial speedup.
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers; default 0 for macOS stability, 2-4 on Win/Linux",
    )
    p.add_argument(
        "--residual",
        action="store_true",
        help="global input->output skip: network predicts a correction (DnCNN-style)",
    )
    p.add_argument(
        "--no-bn",
        action="store_true",
        help="drop BatchNorm from conv blocks (the original N2N U-Net has none)",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.real_pairs is None and args.data_root is None:
        # Colour, native resolution. The grayscale BSD400 set is NOT a valid
        # default here: training grayscale while evaluating colour caps every
        # method near 22 dB on Kodak24 (the channel-statistics confound the
        # paper documents), and it looks like undertraining rather than a bug.
        args.data_root = "../data/CBSD400"
    torch.manual_seed(args.seed)

    if args.real_pairs is not None:
        ds = RealN2NPairDataset(
            args.real_pairs,
            patch_size=args.patch_size,
            length=args.real_length,
            split="train",
            npy_root=args.sidd_npy_root,
            supervised=args.supervised_gt,
        )
        print(
            f"[{'N2C/supervised-GT' if args.supervised_gt else 'real-pairs'}] "
            f"{len(ds.scenes)} training scenes, "
            f"length={len(ds)} patches/epoch, patch={args.patch_size}, "
            f"resolution={'native/mmap' if args.sidd_npy_root else 'max_side=1024'}"
        )
    else:
        ds = N2NPairedDataset(
            args.data_root,
            patch_size=args.patch_size,
            sigma_lo=args.sigma_lo,
            sigma_hi=args.sigma_hi,
            length=args.virtual_length,
        )
        print(
            f"[synthetic] {len(ds.paths)} clean images, "
            f"length={len(ds)} patches/epoch, patch={args.patch_size}"
        )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    model = UNet(residual=args.residual, batchnorm=not args.no_bn).to(args.device)
    loss_fn = make_loss(args.loss, huber_delta=args.huber_delta).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    history = []
    grad_log: list[dict[str, float]] = []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        n = 0
        for x1, x2 in tqdm(loader, desc=f"epoch {epoch+1}/{args.epochs}", leave=False):
            x1 = x1.to(args.device, non_blocking=True)
            x2 = x2.to(args.device, non_blocking=True)
            if args.target_outlier_frac > 0:
                # Salt/pepper contamination of the TARGET only: one mask per
                # pixel location (shared across channels), value 0 or 1.
                b, _, h, w = x2.shape
                hit = torch.rand(b, 1, h, w, device=x2.device) < args.target_outlier_frac
                salt = (torch.rand(b, 1, h, w, device=x2.device) < 0.5).float()
                x2 = torch.where(hit, salt.expand_as(x2), x2)
            pred = model(x1)
            loss = loss_fn(pred, x2)
            if args.lasso > 0:
                loss = loss + args.lasso * lasso_penalty(model)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.log_grad_norms:
                gsq = 0.0
                for p_ in model.parameters():
                    if p_.grad is not None:
                        gsq += float((p_.grad.detach() ** 2).sum())
                grad_log.append(
                    {
                        "grad_norm": gsq ** 0.5,
                        "max_abs_residual": float((pred - x2).detach().abs().max()),
                    }
                )
            opt.step()
            running += loss.item() * x1.size(0)
            n += x1.size(0)
        sched.step()
        avg = running / max(n, 1)
        history.append({"epoch": epoch, "loss": avg, "lr": sched.get_last_lr()[0]})
        print(f"epoch {epoch+1}: loss={avg:.5f}")

    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "history": history,
        },
        out,
    )
    payload: dict = {"args": vars(args), "history": history}
    if grad_log:
        payload["grad_log"] = grad_log
    with out.with_suffix(".json").open("w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
