---
title: Kosi Flood SAR Segmentation
emoji: 🌊
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Kosi River Flood Segmentation (SAR U-Net)

**Author:** Satwik Shreshth

A U-Net based deep learning model for segmenting flood extent from Synthetic Aperture Radar (SAR) imagery, trained on the 2024 Kosi river flood in Bihar, India.

## Overview

This model takes a 6-band SAR feature stack as input and predicts a binary flood mask. The task is framed as pre/post-flood change detection rather than single-date classification, using dual-polarization Sentinel-1 imagery captured before and after the flood event.

## Data

- **Source:** Sentinel-1 GRD, IW mode, VV+VH polarization, descending orbit (Google Earth Engine)
- **Region:** Kosi river basin, Bihar, India
- **Pre-flood window:** August 1-20, 2024
- **Post-flood window:** September 1-20, 2024
- **Resolution:** 10m, EPSG:4326
- **Input bands:** VV_post, VH_post, VV_pre, VH_pre, log_ratio (post-pre backscatter change), elevation (SRTM DEM)
- **Ground truth:** Flood mask generated via a fixed log-ratio threshold, with permanent water bodies (JRC Global Surface Water) excluded

## Model

- **Architecture:** U-Net, encoder-decoder with skip connections, features [64, 128, 256, 512]
- **Loss:** Combined BCE and Dice loss
- **Training:** Kaggle T4x2 GPUs, 87 epochs with early stopping

## Performance (held-out test set)

| Metric | Score |
|---|---|
| Dice | 0.9593 |
| IoU | 0.9218 |
| Precision | 0.9591 |
| Recall | 0.9600 |
| Specificity | 0.9959 |

## Usage

Upload a 6-band GeoTIFF patch (256x256 pixels or smaller) with bands in the order: VV_post, VH_post, VV_pre, VH_pre, log_ratio, elevation. The model returns a binary flood mask and the predicted flood coverage percentage.

## Limitations

This model was trained and evaluated on a single flood event in one river basin. Performance on other regions, sensors, or flood events is untested and would require separate validation before operational use.