"""Datasets: synthetic-noise N2N pairs, clean evaluation, real-noise eval, and
real-noise N2N training pairs (SIDD-Medium).

Public types:

- ``N2NPairedDataset``     -- (clean+n1, clean+n2), n1/n2 i.i.d. from training noise.
- ``CleanEvalDataset``     -- (clean, name), noise applied in eval loop.
- ``RealNoisePairDataset`` -- (noisy, clean, name) for SIDD-style real-noise eval.
- ``RealN2NPairDataset``   -- (noisy_a, noisy_b) cropped at the same coords from
                              two independent shots of the same SIDD scene. Never
                              touches GT -- this is the canonical N2N pair under
                              real (signal-dependent) camera noise.

For SIDD-Medium specifically, ``sidd_scene_split`` partitions the 160 scenes
into a sorted-then-sliced 128/32 train/eval split. This is fully deterministic
across machines and runs without any seed bookkeeping, and both training and
evaluation loaders read it -- they cannot drift apart.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from noise import gaussian_range


def _load_downsampled_uint8(path: Path, max_side: int | None) -> np.ndarray:
    """Load PNG, optionally bicubic-resize so its long side <= ``max_side``,
    return HWC uint8 numpy. Decoupling the decode+resize step from
    ``__getitem__`` lets a dataset front-load it once during construction;
    on Linux DataLoader workers fork from the parent and inherit the buffer
    copy-on-write, so subsequent random crops are zero-copy slicing.
    """
    img = Image.open(path).convert("RGB")
    if max_side is not None:
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    # PIL exposes a read-only buffer; copy() owns it so torch.from_numpy
    # doesn't warn about non-writable arrays and crops can stay zero-copy.
    return np.array(img, dtype=np.uint8, copy=True)


def _to_float_tensor(arr: np.ndarray) -> torch.Tensor:
    """HWC uint8 ndarray -> CHW float tensor in [0, 1]."""
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().float() / 255.0


def _list_images(root: str | Path) -> list[Path]:
    root = Path(root)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)


N_TRAIN_SCENES = 128  # SIDD-Medium has 160 scenes total; first 128 train, last 32 eval.


def sidd_scene_split(
    scene_names: list[str],
    split: str,
    *,
    n_train: int = N_TRAIN_SCENES,
) -> list[str]:
    """Deterministic 128/32 SIDD-Medium scene partition.

    Sort scene names lexicographically, take the first ``n_train`` for
    training and the remainder for evaluation. SIDD scene directory names
    begin with a zero-padded numeric scene-instance ID, so lexicographic
    order matches numeric order, and the split is byte-identical across
    machines, processes, and runs -- no seed required.
    """
    if split not in {"train", "eval", "all"}:
        raise ValueError(f"split must be train/eval/all, got {split!r}")
    ordered = sorted(scene_names)
    if split == "all":
        return ordered
    if split == "train":
        return ordered[:n_train]
    return ordered[n_train:]


def _sidd_data_root(root: str | Path) -> Path:
    """Auto-descend into SIDD's ``Data/`` subdir if present."""
    root = Path(root)
    if (root / "Data").is_dir():
        return root / "Data"
    return root


def _list_sidd_scenes(root: str | Path) -> list[Path]:
    data = _sidd_data_root(root)
    return sorted(p for p in data.iterdir() if p.is_dir())


def _warn_if_grayscale(paths: list[Path], root: str | Path, probe: int = 16) -> None:
    """Loudly flag a single-channel training set.

    ``Image.open(...).convert("RGB")`` promotes grayscale to three identical
    channels without complaint, so a grayscale training folder trains happily
    and only shows up as a hard PSNR ceiling on colour evaluation sets -- the
    channel-statistics confound described in the paper. This turns that silent
    failure into a visible warning at dataset construction.
    """
    step = max(1, len(paths) // probe)
    sampled = paths[::step][:probe]
    def _is_gray(path: Path) -> bool:
        with Image.open(path) as im:
            return im.mode in ("L", "1", "I", "F")

    gray = sum(_is_gray(p) for p in sampled)
    if gray == len(sampled):
        warnings.warn(
            f"{root}: all {len(sampled)} sampled images are single-channel. "
            "Training grayscale while evaluating colour caps PSNR regardless "
            "of loss or step budget and mimics undertraining. Use a colour "
            "set (e.g. CBSD400) unless this is deliberate.",
            RuntimeWarning,
            stacklevel=2,
        )


class N2NPairedDataset(Dataset):
    """Synthetic-Gaussian N2N pairs from a clean image folder.

    ``length`` sets a virtual epoch size (random crops per epoch, images
    visited round-robin), matching ``RealN2NPairDataset`` semantics so the
    synthetic and real-pair training arms can run identical step budgets.
    ``length=None`` preserves the legacy one-crop-per-image epoch.

    Images are decoded once at construction and cached as uint8; __getitem__
    is pure crop + noise, so DataLoader workers stay cheap.
    """

    def __init__(
        self,
        root: str | Path,
        patch_size: int = 128,
        sigma_lo: float = 5.0,
        sigma_hi: float = 50.0,
        length: int | None = None,
    ):
        self.paths = _list_images(root)
        if not self.paths:
            raise FileNotFoundError(f"no images under {root}")
        self.patch = patch_size
        self.sigma_lo = sigma_lo
        self.sigma_hi = sigma_hi
        self.length = length
        self.cache = [_load_downsampled_uint8(p, None) for p in self.paths]
        _warn_if_grayscale(self.paths, root)

    def __len__(self) -> int:
        return self.length if self.length is not None else len(self.paths)

    def _crop(self, img: torch.Tensor) -> torch.Tensor:
        _, h, w = img.shape
        if h < self.patch or w < self.patch:
            img = TF.resize(img, [max(self.patch, h), max(self.patch, w)], antialias=True)
            _, h, w = img.shape
        i = torch.randint(0, h - self.patch + 1, (1,)).item()
        j = torch.randint(0, w - self.patch + 1, (1,)).item()
        return img[:, i : i + self.patch, j : j + self.patch]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        clean = _to_float_tensor(self.cache[idx % len(self.cache)])
        clean = self._crop(clean)
        x1 = gaussian_range(clean, self.sigma_lo, self.sigma_hi)
        x2 = gaussian_range(clean, self.sigma_lo, self.sigma_hi)
        return x1, x2


class CleanEvalDataset(Dataset):
    def __init__(self, root: str | Path, max_side: int | None = None):
        self.paths = _list_images(root)
        if not self.paths:
            raise FileNotFoundError(f"no images under {root}")
        self.max_side = max_side

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.max_side is not None:
            w, h = img.size
            scale = self.max_side / max(w, h)
            if scale < 1.0:
                img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
        clean = TF.to_tensor(img)
        return clean, self.paths[idx].stem


class RealNoisePairDataset(Dataset):
    """Paired real-noise evaluation set (SIDD-Small or SIDD-Medium sRGB).

    Expected layout::

        root/[Data/]<scene_id>/
            *GT_SRGB_*.PNG       # clean reference(s)
            *NOISY_SRGB_*.PNG    # camera-captured noisy shot(s)

    For each scene we enumerate every available noisy image and pair it with
    a GT (matched by shot-index when filenames carry one, otherwise GT[0]).
    With SIDD-Medium's 2-noisy / 2-GT layout this gives 2 eval pairs per
    scene; SIDD-Small style 1+1 gives 1 pair per scene.

    ``split`` is forwarded to ``sidd_scene_split`` -- pass ``split='eval'``
    to restrict to the held-out 32 scenes (the unified eval set in this
    project), ``'train'`` for the 128 training scenes, or ``'all'``.
    """

    def __init__(
        self,
        root: str | Path,
        max_side: int | None = 1024,
        patch_size: int | None = None,
        gt_glob: str = "*GT_SRGB_*.PNG",
        noisy_glob: str = "*NOISY_SRGB_*.PNG",
        split: str = "all",
    ):
        scenes = _list_sidd_scenes(root)
        if not scenes:
            raise FileNotFoundError(f"no scene dirs under {root}")
        keep = set(sidd_scene_split([s.name for s in scenes], split))
        self.pairs: list[tuple[Path, Path, str]] = []
        for scene in scenes:
            if scene.name not in keep:
                continue
            gts = sorted(scene.glob(gt_glob))
            noisy = sorted(scene.glob(noisy_glob))
            if not gts or not noisy:
                continue
            gt_by_shot = _index_by_shot(gts)
            for n_path in noisy:
                shot = _shot_id(n_path)
                gt_path = gt_by_shot.get(shot, gts[0])
                self.pairs.append((n_path, gt_path, f"{scene.name}__{shot or n_path.stem}"))
        if not self.pairs:
            raise FileNotFoundError(
                f"no {gt_glob} / {noisy_glob} pairs found under {root} (split={split!r})"
            )
        self.max_side = max_side
        self.patch_size = patch_size

    def __len__(self) -> int:
        return len(self.pairs)

    def _load(self, path: Path) -> Image.Image:
        img = Image.open(path).convert("RGB")
        if self.max_side is not None:
            w, h = img.size
            scale = self.max_side / max(w, h)
            if scale < 1.0:
                img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
        return img

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        noisy_path, clean_path, name = self.pairs[idx]
        noisy = TF.to_tensor(self._load(noisy_path))
        clean = TF.to_tensor(self._load(clean_path))
        h = min(noisy.shape[1], clean.shape[1])
        w = min(noisy.shape[2], clean.shape[2])
        noisy, clean = noisy[:, :h, :w], clean[:, :h, :w]
        if self.patch_size is not None:
            ps = self.patch_size
            i = max(0, (h - ps) // 2)
            j = max(0, (w - ps) // 2)
            noisy = noisy[:, i : i + ps, j : j + ps]
            clean = clean[:, i : i + ps, j : j + ps]
        return noisy, clean, name


def _shot_id(path: Path) -> str:
    """Last numeric suffix of a SIDD filename, e.g. ``..._SRGB_010.PNG`` -> ``010``."""
    stem = path.stem
    last = stem.rsplit("_", 1)[-1]
    return last if last.isdigit() else ""


def _index_by_shot(paths: list[Path]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in paths:
        sid = _shot_id(p)
        if sid:
            out.setdefault(sid, p)
    return out


class RealN2NPairDataset(Dataset):
    """Real-noise N2N training pairs from SIDD-Medium.

    For each scene we collect *only* the noisy shots (GT is never read --
    that would defeat the whole point of N2N). ``__getitem__`` draws two
    distinct noisy shots of one scene and returns the same random
    ``patch_size`` crop from both. SIDD-Medium has only 2 noisy shots per
    scene, so the per-scene pair is essentially fixed; diversity comes from
    the random crop.

    ``max_side`` (default 1024) downsamples each noisy shot once during
    construction so that ``__getitem__`` is pure crop + tensor conversion.
    Skipping per-step PNG decode is roughly a 50x throughput win on
    consumer GPUs. Eval is also done at max_side=1024 (RealNoisePairDataset
    default), so train/eval scales match.

    ``length`` sets the virtual epoch size directly (default 5000). With
    ~128 scenes x 2 shots and a downsampled ~768x1024 area, every shot
    yields ~36 non-overlapping ``patch_size=128`` crops -- so 5-10k random
    samples per epoch comfortably revisits the unique patch space.

    ``npy_root``: native-resolution mode. Points at the mirror tree written
    by ``prep_sidd_npy.py``; shots are opened with ``np.load(mmap_mode='r')``
    so random crops read only the touched pages. ``max_side`` is ignored
    (no downsampling -- downsampling attenuates the real noise and was the
    core protocol flaw corrected in this revision). Full native cache
    (~13 GB) exceeds the WSL2 RAM budget, hence mmap instead of RAM cache.

    ``supervised``: N2C anchor mode -- the target becomes the scene GT
    (matched by shot index) instead of the second noisy shot. This READS
    GROUND TRUTH and is therefore not self-supervised; it exists solely
    for the disclosed supervised-ceiling control row. Never use it for
    any model reported as N2N.
    """

    def __init__(
        self,
        root: str | Path,
        patch_size: int = 128,
        length: int = 5000,
        max_side: int | None = 1024,
        noisy_glob: str = "*NOISY_SRGB_*.PNG",
        split: str = "train",
        npy_root: str | Path | None = None,
        supervised: bool = False,
        gt_glob: str = "*GT_SRGB_*.PNG",
    ):
        scenes = _list_sidd_scenes(root)
        keep = set(sidd_scene_split([s.name for s in scenes], split))
        self.supervised = supervised
        self.scenes: list[tuple[str, list, list | None]] = []

        def _as_npy(p: Path, scene: Path) -> Path:
            npy = Path(npy_root) / scene.name / f"{p.stem}.npy"
            if not npy.exists():
                raise FileNotFoundError(
                    f"{npy} missing -- run prep_sidd_npy.py first "
                    f"(for GT shots: --noisy-glob '*GT_SRGB_*.PNG')"
                )
            return npy

        for scene in scenes:
            if scene.name not in keep:
                continue
            noisy = sorted(scene.glob(noisy_glob))
            if len(noisy) < 2:
                continue
            targets: list | None = None
            if supervised:
                gts = sorted(scene.glob(gt_glob))
                if not gts:
                    continue
                gt_by_shot = _index_by_shot(gts)
                tgt_paths = [gt_by_shot.get(_shot_id(p), gts[0]) for p in noisy]
            if npy_root is not None:
                shots = [_as_npy(p, scene) for p in noisy]
                if supervised:
                    targets = [_as_npy(p, scene) for p in tgt_paths]
            else:
                shots = [_load_downsampled_uint8(p, max_side) for p in noisy]
                if supervised:
                    targets = [
                        _load_downsampled_uint8(p, max_side) for p in tgt_paths
                    ]
            self.scenes.append((scene.name, shots, targets))
        if not self.scenes:
            raise FileNotFoundError(
                f"no scene with >=2 noisy shots under {root} (split={split!r})"
            )
        self.patch = patch_size
        self.length = length
        # Per-process memmap handle cache. Handles must be opened lazily in
        # the process that uses them: with Python 3.14's forkserver default,
        # DataLoader workers receive the dataset by *pickle*, and pickling an
        # np.memmap materialises the full array (13 GB x num_workers = OOM,
        # which silently killed the first native-res batch run).
        self._handles: dict[Path, np.ndarray] = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def _shot(self, item: np.ndarray | Path) -> np.ndarray:
        if isinstance(item, np.ndarray):
            return item
        h = self._handles.get(item)
        if h is None:
            h = np.load(item, mmap_mode="r")
            self._handles[item] = h
        return h

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Cycle scenes round-robin so every scene is hit roughly equally per
        # epoch even when length isn't a multiple of len(self.scenes).
        name, noisy, targets = self.scenes[idx % len(self.scenes)]
        n = len(noisy)
        i = torch.randint(0, n, (1,)).item()
        if self.supervised:
            a, b = self._shot(noisy[i]), self._shot(targets[i])
        else:
            j = torch.randint(0, n - 1, (1,)).item()
            if j >= i:
                j += 1
            a, b = self._shot(noisy[i]), self._shot(noisy[j])
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        ps = self.patch
        if h < ps or w < ps:
            raise RuntimeError(
                f"scene {name}: shot too small for patch {ps} (got {h}x{w})"
            )
        ti = torch.randint(0, h - ps + 1, (1,)).item()
        tj = torch.randint(0, w - ps + 1, (1,)).item()
        # ascontiguousarray also materialises read-only memmap slices into
        # writable buffers, which torch.from_numpy requires.
        return (
            _to_float_tensor(np.ascontiguousarray(a[ti : ti + ps, tj : tj + ps])),
            _to_float_tensor(np.ascontiguousarray(b[ti : ti + ps, tj : tj + ps])),
        )
