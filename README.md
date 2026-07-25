# Kosi Flood SAR Segmentation

Pixel-wise flood extent segmentation over the Kosi river basin (Bihar, India) using a
six-channel U-Net trained on bi-temporal Sentinel-1 SAR imagery from the August–September
2024 flood event.

**Test Dice: 0.9593 · Test IoU: 0.9218** — evaluated on a 548-patch held-out test set, against
a label-informed threshold baseline (Dice 0.8129). See the paper for the full methodology,
including an explicit discussion of the circularity between the threshold-derived reference
labels and the baseline comparison.

**[Try the live demo](https://huggingface.co/spaces/satwikshreshth1/kosi-flood-sar-segmentation)** — upload a six-band GeoTIFF and get a predicted flood mask back.

**Links:** [Dataset (Kaggle)](https://www.kaggle.com/datasets/satwikshreshth01/bihar-kosi-flood-sar-datase) · [Live demo (Hugging Face Space)](https://huggingface.co/spaces/satwikshreshth1/kosi-flood-sar-segmentation)

---

## What's in this repo

```
.
├── Results/                              # Figures, metrics, and qualitative outputs from training/eval
├── Data_Export.js                        # Google Earth Engine script — generates the six-band SAR stack + label
├── flood-mapping-prepration.ipynb        # Data audit, mosaicking, patch extraction, label noise correction
├── kosi-training.ipynb                   # U-Net training, evaluation, and baseline comparison
├── Shreshth_Kosi_SAR_UNet_Flood_Mapping.pdf  # Paper
├── index.html                            # Static landing page + live inference demo (calls the HF Space)
└── README.md
```

The trained model weights and the Gradio inference app (`app.py`, `best_model.pt`) live in the
[Hugging Face Space](https://huggingface.co/spaces/satwikshreshth1/kosi-flood-sar-segmentation)
rather than in this repo, since the checkpoint is too large to track comfortably without Git LFS.
`index.html` calls that Space directly from the browser, so it works as a static page — e.g. via
GitHub Pages — with no server of its own.

## Dataset

The processed SAR patch archive is published on Kaggle:
**[Bihar Kosi Flood SAR Dataset](https://www.kaggle.com/datasets/satwikshreshth01/bihar-kosi-flood-sar-datase)**

> ⚠️ Worth double-checking that URL before publishing — it currently ends in `...sar-datase`
> rather than `...sar-dataset`. If that's really the slug Kaggle assigned, it's fine as-is;
> if it's a typo, best to fix it now before it's linked from the repo.

To regenerate the raw six-band stack and label from scratch (rather than using the Kaggle
archive directly), run `Data_Export.js` in the [Google Earth Engine code editor](https://code.earthengine.google.com/).
It pulls Sentinel-1 GRD (COPERNICUS/S1_GRD), the JRC Global Surface Water dataset, and SRTM
elevation, and exports the six-channel stack plus the thresholded flood label over the study
region (86.0–87.5°E, 25.5–26.8°N).

## Pipeline

1. **`Data_Export.js`** (Earth Engine) — composites pre-flood (1–20 Aug 2024) and post-flood
   (1–20 Sep 2024) Sentinel-1 VV/VH, computes the VV log-ratio change layer, masks out
   permanent water and terrain above 200 m, and exports the six-band stack + flood label.
2. **`flood-mapping-prepration.ipynb`** — audits the exported tiles for no-data artefacts,
   mosaics them into a virtual raster, extracts 256×256 patches, compares median-filter vs.
   morphological-opening strategies for speckle-noise correction in the label, and produces
   the stratified train/validation/test split.
3. **`kosi-training.ipynb`** — trains the six-channel U-Net (31.0M parameters) under a combined
   BCE + Dice loss, evaluates it on the held-out test set, and compares it against
   per-patch Otsu, global Otsu, and label-informed threshold baselines.
4. **`Results/`** — the metrics, plots, and qualitative prediction figures produced by the
   notebooks above.
5. **`index.html`** — a static front-end that summarises the pipeline and lets a visitor drop
   in a six-band GeoTIFF for live inference, calling the Hugging Face Space directly from the
   browser.

## Model

- **Architecture:** standard U-Net, 4 encoder/decoder stages (64→128→256→512, bottleneck 1024),
  modified to accept a 6-channel input.
- **Input stack:** post-flood VV/VH, pre-flood VV/VH, VV log-ratio, SRTM elevation.
- **Loss:** 0.5 × BCE + 0.5 × Dice.
- **Training:** Adam, lr 1e-4 with plateau decay, batch size 16, dual T4 GPUs, early stopping
  on validation Dice (best checkpoint at epoch 72 of 87).

| Metric | U-Net | Threshold baseline |
|---|---|---|
| Dice | 0.9593 | 0.8129 |
| IoU | 0.9218 | 0.6848 |
| Precision | 0.9591 | 0.6852 |
| Recall | 0.9600 | 0.9992 |

Full results, ablations, and a discussion of what the network's advantage does and doesn't
demonstrate (given that the reference labels are threshold-derived) are in the paper.

## Limitations

This model is trained and evaluated on a single flood event over a single river basin. It is
a research demonstrator, not an operational flood-warning system, and hasn't been validated
against independently surveyed ground truth. See the paper (Section 7 / 4.5–4.6) for the full
discussion.

## Citation

If you use this work, please cite:

```
Shreshth, S. (2026). Delineating the Sorrow of Bihar: A Six-Channel U-Net for Semantic
Segmentation of the August–September 2024 Kosi River Flood from Multi-Temporal Sentinel-1
SAR Imagery.
```

## Contact

Satwik Shreshth — satwikshreshth2002@gmail.com
