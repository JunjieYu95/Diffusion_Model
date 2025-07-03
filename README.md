# Diffusion Models for 2-D Geoscience & Image In-Painting

This repository contains a set of lightweight, **PyTorch + 🤗 Diffusers** examples that demonstrate how to train and use Denoising Diffusion Probabilistic Models (DDPM) for both toy image data (MNIST) and real-world 2-D geoscience data sets such as CO₂ plumes, SIS facies and fluvial channels.  
The project also showcases *RePaint* based in-painting / data-conditioning, making it a handy starting point for seismic tomography or geostatistical reconstruction tasks.

---

## Table of Contents
1. [Project Highlights](#project-highlights)  
2. [Directory Layout](#directory-layout)  
3. [Quick Start](#quick-start)  
4. [Training Workflows](#training-workflows)  
5. [RePaint In-painting](#repaint-in-painting)  
6. [Utilities & Visualisation](#utilities--visualisation)  
7. [Results](#results)  
8. [Troubleshooting](#troubleshooting)  
9. [Roadmap](#roadmap)  
10. [Citation](#citation) / [Licence](#licence)

---

## Project Highlights
• **Minimal code ‑ maximum clarity** – fewer than ~150 lines per experiment.  
• **Multiple data domains** – MNIST digits and several geoscience pixel grids.  
• **GPU-ready** – Automatically selects CUDA when available.  
• **Flexible output sizes** – thanks to fully-convolutional U-Net architecture.  
• **In-painting** – RePaint pipeline with configurable masks (random pixels, seismic ray paths, patches, …).

---

## Directory Layout
```text
.
├── DDPM/                    # All diffusion experiments
│   ├── MNIST_examples/
│   │   └── main_mnist.py
│   ├── CO2_PLUME_examples/
│   │   ├── main_co2.py
│   │   └── test_flexible_dimension.py
│   └── Geo_examples/
│       ├── main_geo.py
│       └── repaint.py
├── utility.py               # Data-loader helpers & mask generators
├── viz.py                   # Visualisation helpers
├── roadpath.md              # Development diary / notes
└── literatures/             # (optional) reading material & papers
```
Each experiment folder also contains a `monitor/` sub-folder where training curves and generated samples are saved.

---

## Quick Start
### 1. Clone & install
```bash
# clone
$ git clone <repo-url>
$ cd <repo-dir>

# create environment (conda or venv)
$ conda create -n ddpm python=3.10 -y
$ conda activate ddpm

# install core dependencies
$ pip install torch torchvision diffusers matplotlib tqdm
```
Optional extras: `jupyter`, `notebook`, `ipykernel` if you wish to run the notebooks in `./DDPM/*/Analysis.ipynb`.

### 2. Prepare data
All geoscience data sets are stored as *PyTorch tensors* (`.pt`) and **are NOT checked into the repo**.  
Update the absolute paths inside `utility.py` to point to your local copies:
```python
file = '/path/to/geo_dataset/uncond-sis-train.pt'
```

### 3. Kick off a run
```bash
# MNIST toy example (CPU or single-GPU)
$ python DDPM/MNIST_examples/main_mnist.py

# CO₂ plume (needs ~6 GB GPU mem)
$ python DDPM/CO2_PLUME_examples/main_co2.py

# Fluvial / SIS facies (64×64 grids)
$ python DDPM/Geo_examples/main_geo.py
```
Outputs (loss curves + sample grids) will be written to the corresponding `monitor/` directory.

---

## Training Workflows
All scripts follow the same pattern:
1. **Load data** via one of the helper dataloaders in `utility.py` (MNIST, SIS, fluvial, CO₂ plume, …).  
2. **Instantiate U-Net** (`UNet2DModel`) and configure channels/blocks.  
3. **Define scheduler** – `DDPMScheduler` with 1 000 steps (modifiable).  
4. **Optimise** with Adam + _MSE_ or _L1_ targets.  
5. **Monitor** – after every *k* epochs images are sampled through a `DDPMPipeline` and saved.

Hyper-parameters can be changed at the top of each `main_*.py` file.

---

## RePaint In-painting
The script `DDPM/Geo_examples/repaint.py` demonstrates conditioning a pre-trained DDPM on partial observations using **RePaint**:
```bash
$ python DDPM/Geo_examples/repaint.py
```
Key components:
* `get_random_pixel_mask`, `get_penetrate_mask`, `patched_mask` in `utility.py` create binary masks.  
* A trained model checkpoint (`monitor_channel/model_50.pt`) is loaded and passed to `RePaintPipeline`.
* Output grids + side-by-side comparisons are saved via `viz.compare_repaint`.

---

## Utilities & Visualisation
* **`utility.py`** – dataloaders and geometric mask generators (ray-path, patch, random pixel).  
* **`viz.py`** – helper functions for tidier matplotlib grids, conditional vs generated sample comparison, etc.

---

## Results
Sample outputs (generated digits, CO₂ plume reconstructions, patch masks, …) are committed as PNGs inside each `monitor*/` or `roadpath_images/` folder for quick inspection.

---

## Troubleshooting
• **White / saturated backgrounds** – make sure you rescale tensors to `[-1,1]` during training and clamp back to `[0,1]` for visualisation (see comments in `roadpath.md`).  
• **CUDA OOM** – reduce `batch_size` or `block_out_channels`; try `mixed_precision="fp16"`.

---

## Roadmap
See `roadpath.md` for the full development log.  Planned items:
1. Hyper-parameter sweep for better geological textures.  
2. DDIM & score-based variants for faster sampling.  
3. 3-D volume extension.

---

## Citation
If you build on this code, please cite:
```text
@misc{ddpm_geoscience,
  author       = {Your Name},
  title        = {Diffusion Models for Geoscience In-painting},
  year         = {2024},
  howpublished = {GitHub},
  url          = {https://github.com/<your-repo>}
}
```

## Licence
[MIT](LICENSE) © 2024 Your Name