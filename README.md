# Noise2Noise Revisited: Training Pair Distributions Dominate Loss Choice in Self-Supervised Denoising

Code, scripts, and per-image metric dumps for the paper.

> Paper: [TODO arXiv link]
> Venue: 8th International Conference on Video, Signal and Image Processing
> (VSIP 2026), Zhenjiang, China, 6-8 November 2026.

Trained weights are not redistributed: every model retrains from scratch in
~16 minutes on one consumer GPU, and each checkpoint records its own RNG seed,
so the runs here are reproducible without downloading them.

## Setup

```bash
cd code
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # PyTorch w/ CUDA: see pytorch.org for your platform
```

Every model trains in ~16 minutes on a single RTX 3080 (10 GB).
Datasets: see `data/README.md` (all public; not redistributed here).

## Reproduce

| Step | Command (from `code/`) |
|---|---|
| Train all main models (5 synth losses, 5 real-pair losses, Noise2Void) | `bash train_all.sh` |
| Evaluation grid + native SIDD32 + official SIDD blocks (GPU) | `bash eval_all.sh gpu` |
| CBM3D baseline (CPU, hours; safe to run in parallel with the GPU pass) | `bash bm3d_baseline.sh` -- `bash eval_all.sh bm3d` is the same thing |
| Merge the baseline in and render the paper tables | `bash eval_all.sh tables` |
| Seed-gap table (needs `seeds_synth.sh` results) | `python render_tables.py --results results/cross_noise_kodak_v4.json --blocks-results results/sidd_val_blocks_v4.json --seeds-results results/kodak_seeds_v4.json --layout compact --out tables_compact.tex --seeds-out table_seeds.tex` |
| Diagnostics (Huber δ=0.05, gradient stats, contaminated targets, N2V-real, Lasso α sweep) | `bash diagnostics_batch.sh` |
| Synthetic-cell seed replication | `bash seeds_synth.sh` |
| N2C supervised anchor + real-cell seeds | `bash n2c_and_real_seeds.sh` |
| Wilcoxon + Holm significance tests | `python stats_tests.py` |
| SIDD cross-shot residual correlation | `python sidd_shot_correlation.py` |

All scripts are idempotent (existing checkpoints/results are skipped) and
record the RNG seed inside each checkpoint.

## Results

`code/results/` contains every number in the paper, including
**per-image metric dumps** (`*.per_image.json`) used for the paired
Wilcoxon tests, and the significance-test outputs (`wilcoxon_*.txt`).
Files ending `_cbm3d` are the colour BM3D baseline used in all tables
(`bm3d_rgb` with a wavelet-domain sigma estimate).

The canonical `cross_noise_kodak_v4.json` and `sidd_val_blocks_v4.json`
already have that baseline merged in as the `bm3d` row, so every table in
the paper re-renders straight from a fresh clone -- no GPU, no dataset
download:

```bash
cd code
python render_tables.py --results results/cross_noise_kodak_v4.json \
    --blocks-results results/sidd_val_blocks_v4.json \
    --layout compact --out tables_compact.tex
```

## Citation

```bibtex
@inproceedings{shang2026n2nrevisited,
  author    = {Shang, Dingyan and Xu, Zhenyu and Wang, Youting and
               Shen, Bonan and Ning, Tao and Liu, Bowen},
  title     = {Noise2Noise Revisited: Training Pair Distributions Dominate
               Loss Choice in Self-Supervised Denoising},
  booktitle = {Proc. 8th Int. Conf. Video, Signal and Image Processing (VSIP)},
  address   = {Zhenjiang, China},
  year      = {2026},
  pages     = {TODO after publication},
  doi       = {TODO after publication}
}
```

MIT License.
