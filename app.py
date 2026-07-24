# Kosi River Flood Segmentation - SAR U-Net
# Author: Satwik Shreshth
# Model trained on Sentinel-1 SAR imagery over the Kosi river basin, Bihar, India

import spaces
import torch
import torch.nn as nn
import numpy as np
import gradio as gr
import rasterio

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)

# Model is loaded on CPU at startup. ZeroGPU allocates a GPU only for the
# duration of functions decorated with @spaces.GPU, so the model is moved
# to CUDA inside the predict function itself, not at module load time.
model = UNet(in_channels=6, out_channels=1)
state_dict = torch.load("best_model.pt", map_location="cpu")
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()

PATCH_SIZE = 256
BAND_NAMES = ["VV_post", "VH_post", "VV_pre", "VH_pre", "log_ratio", "elevation"]

@spaces.GPU
def predict(tif_file):
    with rasterio.open(tif_file.name) as src:
        arr = src.read().astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0)

    if arr.shape[0] != 6:
        return None, f"Expected 6 bands (VV_post, VH_post, VV_pre, VH_pre, log_ratio, elevation), got {arr.shape[0]}"

    h, w = arr.shape[1], arr.shape[2]
    pad_h = max(0, PATCH_SIZE - h)
    pad_w = max(0, PATCH_SIZE - w)
    arr = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
    arr = arr[:, :PATCH_SIZE, :PATCH_SIZE]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tensor = torch.from_numpy(arr).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        pred = (torch.sigmoid(output) > 0.5).cpu().numpy()[0, 0]

    flood_pct = 100 * pred.sum() / pred.size
    return (pred * 255).astype(np.uint8), f"Predicted flood coverage: {flood_pct:.2f}% of patch"

demo = gr.Interface(
    fn=predict,
    inputs=gr.File(label="Upload 6-band GeoTIFF (VV_post, VH_post, VV_pre, VH_pre, log_ratio, elevation)"),
    outputs=[gr.Image(label="Predicted Flood Mask"), gr.Textbox(label="Summary")],
    title="Kosi River Flood Segmentation (SAR U-Net)",
    description=(
        "U-Net trained on Sentinel-1 SAR imagery over the Kosi river basin, Bihar, India, "
        "to segment flood extent from pre/post-flood radar backscatter. "
        "Developed by Satwik Shreshth. Upload a 6-band GeoTIFF patch (256x256 or smaller) "
        "to get a flood extent prediction."
    ),
    article=(
        "**Model:** U-Net, 6-channel input (VV_post, VH_post, VV_pre, VH_pre, log_ratio, elevation).<br>"
        "**Data:** Sentinel-1 GRD (IW, VV+VH, descending orbit), Aug-Sep 2024, Kosi basin, Bihar.<br>"
        "**Performance:** Dice 0.9593, IoU 0.9218 on held-out test data.<br>"
        "**Author:** Satwik Shreshth"
    ),
)

if __name__ == "__main__":
    demo.launch(show_api=False)