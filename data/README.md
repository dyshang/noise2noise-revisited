# Datasets (download separately; not redistributed here)

Expected layout under `data/`:

| Folder | Contents | Source |
|---|---|---|
| `CBSD400/` | 400 RGB images, BSDS500 train+test splits | https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html |
| `Kodak24/` | 24 Kodak PhotoCD images (PNG) | http://r0k.us/graphics/kodak/ |
| `SIDD_Medium_Srgb/` | SIDD-Medium sRGB (160 scenes, 2 noisy + 2 GT each) | https://www.eecs.yorku.ca/~kamel/sidd/ |
| `SIDD_Blocks/` | ValidationNoisyBlocksSrgb.mat + ValidationGtBlocksSrgb.mat | https://www.eecs.yorku.ca/~kamel/sidd/ |

`code/prep_sidd_npy.py` converts SIDD-Medium to the memory-mapped `.npy`
tree used by training (`SIDD_Medium_Srgb_npy/`). The deterministic
128/32 scene split (sorted scene-instance names) is implemented in
`code/data.py`; no split file is needed.
