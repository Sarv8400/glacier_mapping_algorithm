# Himalayan Glacier Segmentation — Deep Learning Framework

A multi-model deep learning framework for binary semantic segmentation of glaciers from multi-band satellite raster imagery across major Himalayan basins. This repository implements and compares three segmentation architectures — **GLAVITU**, **M-LandsNet**, and **UNet++** — along with a dedicated **post-processing pipeline** to refine predicted glacier masks.

---

## 📁 Repository Files

| File | Description |
|---|---|
| `glavitu_all.ipynb` | GLAVITU model — ResNet + Vision Transformer + U-Net hybrid |
| `m_landsnet_new_sato_all.ipynb` | M-LandsNet model — ResNet + Attention Gate U-Net |
| `unetplusplus_model.ipynb` | UNet++ model — Nested dense skip connection U-Net |
| `post_processing.ipynb` | Post-processing pipeline for refining DL segmentation outputs |

---

## ⚠️ BEFORE YOU START — Read This First

> Complete all steps below **before running any notebook cell**. Skipping steps will cause import errors, shape mismatches, or missing file crashes.

---

## ✅ Step 1 — System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.8 | 3.9 |
| RAM | 16 GB | 32 GB |
| GPU VRAM | 6 GB | 12 GB+ |
| Disk Space | 30 GB | 80 GB+ |
| OS | Ubuntu 20.04 / Windows 10 | Ubuntu 22.04 |

> **GPU is required for training.** All three models train up to 1000 epochs — CPU training is not practical.

---

## ✅ Step 2 — Create a Virtual Environment

```bash
# Option A: venv
python -m venv glacier_env
source glacier_env/bin/activate          # Linux/Mac
glacier_env\Scripts\activate             # Windows

# Option B: Conda (recommended)
conda create -n glacier python=3.9
conda activate glacier
```

---

## ✅ Step 3 — Install All Required Libraries

Install in this exact order to avoid dependency conflicts:

### 3a. Core Deep Learning
```bash
pip install tensorflow==2.12.0
pip install keras==2.12.0
```

### 3b. Segmentation Models
```bash
pip install segmentation-models
```

> **Critical:** Always set the framework before importing `segmentation_models`:
> ```python
> import os
> os.environ["SM_FRAMEWORK"] = "tf.keras"
> import segmentation_models as sm
> ```

### 3c. Geospatial Libraries
```bash
# Install GDAL first — required by rasterio
conda install -c conda-forge gdal          # Recommended
# OR (Linux only)
pip install gdal

pip install rasterio
pip install fiona
pip install shapely
pip install geopandas
pip install pyproj
```

> **Windows users:** Download GDAL prebuilt wheels from https://www.lfd.uci.edu/~gohlke/pythonlibs/ and install manually before rasterio.

### 3d. Image Processing & Scientific Libraries
```bash
pip install numpy
pip install scikit-learn
pip install scikit-image
pip install matplotlib
pip install patchify
pip install opencv-python
pip install scipy
pip install tqdm
```

### 3e. Install Everything at Once
```bash
pip install tensorflow==2.12.0 keras==2.12.0 segmentation-models \
    rasterio fiona shapely geopandas pyproj \
    numpy scikit-learn scikit-image matplotlib patchify \
    opencv-python scipy tqdm
```

### Verified Package Versions

| Package | Version |
|---|---|
| tensorflow | 2.12.0 |
| keras | 2.12.0 |
| segmentation-models | 1.0.1 |
| rasterio | 1.3.x |
| geopandas | 0.13.x |
| scikit-image | 0.20.x |
| numpy | 1.23.x |
| scikit-learn | 1.2.x |
| scipy | 1.10.x |
| opencv-python | 4.x |
| gdal | 3.x |

---

## ✅ Step 4 — Verify GPU Setup

Run this before opening any notebook:

```python
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPUs available:", tf.config.list_physical_devices('GPU'))
```

Expected:
```
TensorFlow version: 2.12.0
GPUs available: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

Required CUDA stack for TensorFlow 2.12:
```
CUDA 11.8  +  cuDNN 8.6
```

---

## ✅ Step 5 — Prepare Your Input Data

### Required Raster Specifications

| Property | Required |
|---|---|
| Format | GeoTIFF (`.tif`) |
| Bands | 18 (full models) or 6 (top-band variants) |
| Pixel values | Normalized 0–255 |
| CRS | Projected CRS, consistent across raster + mask pairs |
| Mask values | Binary: `1 = glacier`, `0 = non-glacier` |

### Data Preprocessing Checklist
- [ ] All rasters are normalized to 0–255
- [ ] Masks are strictly binary (0 and 1 only, no NoData)
- [ ] Raster and mask CRS are aligned (reproject with rasterio if needed)
- [ ] Raster and mask spatial extents match
- [ ] No NaN or NoData values in patch regions

### Input Rasters Used in This Study

| Raster | Region |
|---|---|
| `raster_cb__normalized.tif` | Chandra Bhaga Basin |
| `raster_satopanth_normalized.tif` | Uttarakhand (Satopanth) |
| `KASHMIR_NORMALIZED_255_final.tif` | Kashmir |
| `sikkim_chaudhary_255_final.tif` | Sikkim |
| `cb_vali_normalized_128.tif` | CB Validation |
| `NEW_UK_TIF_fixedB10.tif` | Alaknanda & Bhagirathi Basin |

---

## ✅ Step 6 — Set Up Directory Structure

The notebooks expect this folder layout. Patch subdirectories are auto-created; place raw rasters and masks manually.

```
sarvesh/
├── DATASETS_NORMALIZED/
│   ├── raster_cb__normalized.tif
│   ├── raster_satopanth_normalized.tif
│   ├── KASHMIR_NORMALIZED_255_final.tif
│   ├── sikkim_chaudhary_255_final.tif
│   ├── cb_vali_normalized_128.tif
│   ├── sikkim_mask.tif
│   ├── KASHMIR_MASK.tif
│   ├── CB_VALIDATION_MASK.tif
│   └── all_sau_net/
│       ├── gn2_all/
│       │   ├── rasterlayer/              ← Auto-generated (GLAVITU patches)
│       │   └── masklayer/
│       └── glavitu/
│           ├── sf/model/                 ← Saved GLAVITU models (.h5)
│           └── top6/
│               ├── top6band/
│               ├── rasterlayer/
│               ├── masklayer/
│               └── top6_sf/
│
├── sato_new/
│   ├── NEW_UK_TIF_fixedB10.tif
│   ├── glacier_mask_binary.tif
│   └── sato_sikkim_kashmir/
│       ├── rasterlayer/                  ← Auto-generated (UNet++ patches)
│       ├── vectorlayer/
│       ├── model/                        ← Saved UNet++ models (.h5)
│       └── top6/
│           ├── rasterlayer/
│           ├── masklayer/
│           └── model/
│
├── mlandsnet_sato_new/
│   ├── rasterlayer/                      ← Auto-generated (M-LandsNet patches)
│   ├── vectorlayer/
│   ├── model_18/                         ← Saved M-LandsNet models (.h5)
│   └── top_6/
│       ├── rasterlayer/
│       └── masklayer/
│
├── chandrabasin/
│   └── dataset/new/
│       └── masklayersato.tif
│
└── new_layers_/
    └── mask_reprojected/
        ├── KASHMIR_NORMALIZED_255_finalkasmircomplete_mask_reprojected.tif
        └── raster_cb__normalized_new_mask_reprojected.tif
```

---

## ✅ Step 7 — Launch Jupyter Notebook

```bash
pip install jupyter
jupyter notebook        # then open the desired .ipynb file
# OR
pip install jupyterlab
jupyter lab
```

---

## 🏔️ Study Areas

| Region | Role |
|---|---|
| Chandra Bhaga Basin | Training + Validation |
| Uttarakhand — Alaknanda & Bhagirathi | Training |
| Kashmir | Training + Testing |
| Sikkim | Training + Testing |
| Satopanth (Uttarakhand) | Testing |
| CB Validation (Miyer Valley) | Held-out Validation |

---

## 🧠 Model Descriptions

---

### 1. `glavitu_all.ipynb` — GLAVITU

**GLAcier Vision Transformer U-net** — A hybrid architecture combining residual encoding with a Vision Transformer bottleneck.

**Architecture:**
- **Encoder:** Residual blocks with skip connections
- **Bottleneck:** Vision Transformer (ViT) with multi-head self-attention + positional embeddings
- **Decoder:** Transposed convolution upsampling with skip connections
- **Output:** Pixel-wise softmax (binary: glacier / non-glacier)

**Key Parameters:**

| Parameter | Value |
|---|---|
| Input patch size | 64 × 64 |
| Full model bands | 18 |
| Top-band model bands | 6 (bands 8, 4, 5, 1, 11, 14) |
| Loss | Categorical Cross-Entropy |
| Optimizer | Adam (lr=1e-4, clipnorm=1.0) |
| Batch size | 16 |
| Max epochs | 1000 (EarlyStopping patience=20) |

**Notebook Structure:**
```
├── Data loading & patch visualization (18-band)
├── Dataset metrics & validation
├── Train-test split
├── Model definition (ResNet + ViT + U-Net)
├── Compile & train (18-band)
├── Training curves & prediction visualization
├── Full-scene inference & vectorization (18-band)
├── Evaluation metrics (18-band)
├── Gradient-based band contribution analysis
├── Top-6 band extraction
├── Compile & train (6-band)
├── Full-scene inference & vectorization (6-band)
└── Evaluation metrics (6-band)
```

**Saved Models:**

| Variant | Saved As |
|---|---|
| Full 18-band | `glavitu_all_area.h5` |
| Top-6 band | `top6_glavitu_all_area.h5` |

---

### 2. `m_landsnet_new_sato_all.ipynb` — M-LandsNet

**Multi-LandsNet** — A U-Net architecture enhanced with residual blocks and attention gates for fine-grained glacier boundary detection.

**Architecture:**
- **Encoder:** Residual blocks with dilated convolutions
- **Attention gates:** Suppress irrelevant feature responses at skip connections
- **Decoder:** Upsampling with attention-gated skip connections
- **Output:** Pixel-wise softmax (binary)

**Key Parameters:**

| Parameter | Value |
|---|---|
| Input patch size | 128 × 128 |
| Full model bands | 18 |
| Top-band model bands | 6 (bands 5, 4, 1, 8, 11, 6) |
| Loss | Dice Loss + Focal Loss |
| Optimizer | Adam (lr=1e-3) with mixed float16 precision |
| Batch size | 16 |
| Max epochs | 1000 (EarlyStopping patience=20) |

**Notebook Structure:**
```
├── Pre-evaluation visualization (18-band results)
├── Data patching & augmentation
├── Dataset metrics
├── Train-test split
├── M-LandsNet model definition
├── Compile & train (18-band)
├── Full-scene inference & vectorization (18-band)
├── Evaluation metrics (18-band)
├── Gradient-based band contribution analysis
├── Top-6 band extraction
├── Compile & train (6-band)
├── Full-scene inference & vectorization (6-band)
└── Evaluation metrics (6-band)
```

**Saved Models:**

| Variant | Saved As |
|---|---|
| Full 18-band | `m_landsnet_newsato_all_area.h5` |
| Top-6 band | `m_landsnet_top6.h5` |

---

### 3. `unetplusplus_model.ipynb` — UNet++

**UNet++** — A U-Net with nested, dense skip connections for improved feature reuse at multiple scales, applied to glacier segmentation.

**Architecture:**
- **Encoder:** Conv blocks (64→128→256→512→1024 filters)
- **Nested skip connections:** Dense paths between encoder and decoder nodes at each level
- **Decoder:** Transposed convolution upsampling with dense skip aggregation
- **Output:** Pixel-wise softmax (binary)

**Key Parameters:**

| Parameter | Value |
|---|---|
| Input patch size | 256 × 256 |
| Full model bands | 18 |
| Top-band model bands | 6 (bands 5, 17, 1, 2, 16, 7) |
| Loss | Dice Loss + Focal Loss |
| Optimizer | Adam (lr=1e-3) with mixed float16 precision |
| Batch size | 16 (full) / 8 (top-6) |
| Max epochs | 1000 (EarlyStopping patience=20) |

**Notebook Structure:**
```
├── Data patching & augmentation (256×256 patches)
├── Patch visualization
├── Dataset metrics
├── Train-test split
├── UNet++ model definition
├── Compile & train (18-band)
├── Training curves & prediction visualization
├── Full-scene inference & vectorization (18-band)
├── Gradient-based band contribution analysis
├── Evaluation metrics (18-band)
├── Top-6 band extraction
├── Compile & train (6-band)
├── Full-scene inference & vectorization (6-band)
└── Evaluation metrics (6-band)
```

**Saved Models:**

| Variant | Saved As |
|---|---|
| Full 18-band | `uk_sik_kash_unet_model.h5` |
| Top-6 band | `unet_sato_all_top6.h5` |

---

### 4. `post_processing.ipynb` — Post-Processing Pipeline

A standalone post-processing notebook that refines raw DL segmentation masks before final evaluation and vectorization. Applied after inference from any of the three models.

**Post-Processing Steps:**

1. **Load & align masks** — Reproject ground truth to match predicted raster grid using `rasterio.warp.reproject`
2. **Remove small fragments** — Delete glacier objects below minimum area threshold (default: 0.03 km²)
3. **Merge nearby fragments** — Connect glacier segments within a configurable distance (default: 10–40 m) using distance transform
4. **Morphological smoothing** — Apply binary dilation + erosion + morphological closing to smooth jagged boundaries
5. **Hole filling** — Fill internal holes in glacier polygons using `binary_fill_holes`
6. **Auto-parameter optimization** — Grid search over merge distances and scale factors to maximize IoU against ground truth (multiprocessing accelerated)
7. **Vectorization** — Convert processed raster to raw and smoothed Shapefiles (`.shp`) via `rasterio.features.shapes` + `geopandas`
8. **Buffer analysis** — Compute IoU at ±10m and ±20m boundary buffers to account for digitization uncertainty
9. **Visualization** — Zoomed patch comparisons, TP/FP/FN overlay maps, raw vs smooth boundary plots

**Key Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `min_area_km2` | 0.03 | Minimum glacier area to retain |
| `merge_distance_m` | 10–40 | Distance to merge nearby fragments |
| `pixel_resolution` | 10 | Pixel size in meters |
| `max_iterations` | 10–30 | Smoothing/merging iterations |

**Additional Libraries Required for Post-Processing:**
```bash
pip install scikit-image geopandas pyproj tqdm
```

---

## 🔄 Common Pipeline (All Models)

All three model notebooks share this workflow:

1. **Patch generation** — Sliding window extraction from raw rasters + masks with augmentation (H-flip, V-flip, 90°/180°/270° rotations)
2. **Dataset loading** — Load GeoTIFF patches → transpose to `(N, H, W, C)` → one-hot encode masks
3. **Train/test split** — 80/20 split (random state = 100)
4. **Train** — EarlyStopping + ModelCheckpoint save best model
5. **Inference** — Sliding window with 50% overlap + blend matrix on full raster scenes → georeferenced GeoTIFF output
6. **Vectorize** — Convert binary rasters to Shapefiles using `rasterio` + `fiona` + `shapely`
7. **Band analysis** — `tf.GradientTape` sensitivity scores → top 6 bands identified
8. **Retrain** — Repeat steps 1–6 with top-6 band rasters only
9. **Evaluate** — Compare both variants against ground truth masks

---

## 📊 Evaluation Metrics

All models are evaluated per region using:

| Metric | Description |
|---|---|
| Overall Accuracy | Pixel-level accuracy |
| IoU (Jaccard Score) | Intersection over Union |
| F1-Score (Dice) | Precision-Recall harmonic mean |
| Precision | Correctly identified glacier pixels |
| Recall | True glacier pixels captured |
| Cohen's Kappa | Agreement beyond chance |
| Confusion Matrix | TP / FP / TN / FN per region |

---

## 📦 Quick Reference — Model Comparison

| Model | Patch Size | Bands | Architecture Highlight |
|---|---|---|---|
| GLAVITU | 64 × 64 | 18 / 6 | Vision Transformer bottleneck |
| M-LandsNet | 128 × 128 | 18 / 6 | Attention gates on skip connections |
| UNet++ | 256 × 256 | 18 / 6 | Nested dense skip connections |

---

## 📌 Citation

If you use this work, please cite accordingly. Dataset regions cover major Himalayan glacier basins including Chandra Bhaga, Kashmir, Sikkim, and Uttarakhand (Alaknanda-Bhagirathi).

---

## 📄 License

MIT License — see `LICENSE` file for details.
