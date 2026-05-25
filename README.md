# Climate Grid Super-Resolution: Copernicus 1° → IMD 0.25°

A deep learning pipeline to upscale coarse rainfall forecasts from Copernicus (1° resolution) to high-resolution rainfall maps matching India Meteorological Department (IMD) observations (0.25° resolution).

## Overview

This notebook implements a **U-Net based super-resolution model** trained on seasonal climate forecast data. The model learns to transform low-resolution (coarse) rainfall patterns into fine-grained spatial detail while preserving physical accuracy.

### Key Features

- **6× more training data** by leveraging all 6 CDS forecast lead times
- **Proper normalization** with z-score standardization to stabilize gradients
- **U-Net with residual blocks** to recover fine-scale spatial patterns
- **Combined MAE + MSE loss** for robust handling of monsoon rainfall spikes
- **Adaptive learning rate scheduling** (ReduceLROnPlateau) and early stopping
- **True super-resolution pipeline**: low-resolution grid remains at 1° native scale
- **Spatial interpolation** for NaN handling instead of zero-filling
- **Extended evaluation**: RMSE, MAE, Pearson correlation, bias, and spatial error maps

## Dataset

### Input (Low-Resolution)
- **Source**: Copernicus Climate Change Service (CDS)
- **Product**: ECMWF System 51 seasonal hindcasts
- **Resolution**: 1° × 1° grid
- **Temporal**: Monthly means, 6 lead times (1–6 months ahead)
- **Region**: India (5°N–38°N, 65°E–100°E)

### Target (High-Resolution)
- **Source**: India Meteorological Department (IMD)
- **Resolution**: 0.25° × 0.25° grid
- **Temporal**: Daily observations, aggregated to monthly totals
- **Region**: India (5°N–38°N, 65°E–100°E)

### Data Size
- 2025 calendar year: 12 months × 6 lead times = ~72 forecast-observation pairs
- After train/val split (70/30): ~50 training, ~22 validation samples
- **Note**: This is a small dataset; temporal augmentation is essential

## Architecture

### U-Net Encoder-Decoder
```
Input (1° grid)
    ↓
Encoder: Conv2D (64) → ResBlock (64) → Conv2D (128, stride=2) → ResBlock (128)
    ↓
Bottleneck: ResBlock (128) × 2
    ↓
Decoder: Upsample (stride=2) → ResBlock (64) + Skip → Resize to HR grid
         → ResBlock (64) → ResBlock (32)
    ↓
Output (0.25° grid, 1 channel)
```

### Residual Blocks
- Pre-activation design (BatchNorm → ReLU → Conv → BatchNorm → ReLU → Conv)
- Skip connections to preserve spatial information
- He normal initialization for stable training

## Training

### Loss Function
```python
Loss = 0.7 * MAE + 0.3 * MSE
```
- **MAE** (70%): Robust to extreme rainfall events, avoids mean-collapse
- **MSE** (30%): Penalizes large errors

### Optimizer & Callbacks
- **Adam** with initial learning rate = 1e-3
- **ReduceLROnPlateau**: Halve LR if validation loss stagnates (patience=5)
- **EarlyStopping**: Stop at patience=15, restore best weights
- **ModelCheckpoint**: Save best model during training

### Data Handling
- **Normalization**: Z-score standardization (computed from training set only)
- **Train/Val Split**: Time-ordered 70/30 split (no random shuffle for temporal coherence)
- **Temporal Augmentation**: Additive Gaussian noise (~5% std) to mimic natural variability
- **Dropout**: 0.3–0.4 in model layers to prevent overfitting

## Evaluation Metrics

- **RMSE** (Root Mean Squared Error): Typical error magnitude (mm)
- **MAE** (Mean Absolute Error): Average absolute error (mm)
- **Bias**: Systematic over/under-prediction (mm)
- **Pearson r**: Spatial/temporal correlation between predicted and observed
- **Spatial error maps**: Per-grid-cell MAE to identify weak regions

### Location-Based Validation
- Pune-focused validation near monsoon convergence zone
- Compares model vs. naive bilinear interpolation baseline

## Installation

```bash
pip install -q netCDF4 xarray cfgrib cdsapi cartopy scikit-learn tensorflow
```

### CDS API Setup
```bash
# Create ~/.cdsapirc with your CDS credentials
url: https://cds.climate.copernicus.eu/api
key: <YOUR_CDS_API_KEY>
```

Get your free API key: https://cds.climate.copernicus.eu/

## Usage

1. **Configure CDS credentials** in cell 5
2. **Run cells 1–3**: Install packages and set up imports
3. **Run cells 4–8**: Download and load IMD observations + CDS forecasts
4. **Run cells 9–11**: Build training pairs and normalize data
5. **Run cells 12–17**: Build and train U-Net model
6. **Run cells 18–24**: Evaluate on validation set and generate plots

## Outputs

- `sr_unet_final.keras`: Trained model
- `normalisation_stats.json`: Mean/std for inference
- `training_curves.png`: Loss/MAE/RMSE over epochs
- `spatial_mae.png`: Spatial distribution of validation error
- `pune_timeseries.png`: Pune location time-series comparison
- `sr_comparison.png`: Side-by-side LR bilinear vs. model vs. IMD

## Known Issues & Improvements

### Current Limitations
- **Tiny dataset**: 72 samples limits generalization; model may overfit
- **Single year**: 2025 only; no inter-annual variability learned
- **Monsoon specificity**: Cannot extrapolate to different climate regimes

### Recommended Improvements
- Integrate multi-year data (ERA5 reanalysis, 1980–present) → 2500+ samples
- Replace spatial flipping with temporal augmentation (preserves geography)
- Add uncertainty quantification (dropout-based Bayesian inference)
- Domain adaptation for generalization across regions
- Ensemble methods combining multiple models
- Physics-informed constraints (e.g., mass conservation, energy balance)

## References

- U-Net for Image-to-Image Translation: Ronneberger et al. (2015)
- Residual Networks: He et al. (2016)
- ERA5 Reanalysis: Hersbach et al. (2020)
- Climate Downscaling: van der Linden & Lenderink (2009)

## License

MIT License — feel free to use, modify, and distribute.

## Author

Joel Panjimaram

## Changelog

### v2 (Current)
- Removed physically invalid spatial flipping
- Added temporal augmentation with noise
- Improved NaN handling with spatial interpolation
- Added Pune-focused validation
- Extended evaluation metrics

### v1
- Baseline U-Net + single lead time

---

Questions or issues? Open an issue on GitHub or contact the author.
