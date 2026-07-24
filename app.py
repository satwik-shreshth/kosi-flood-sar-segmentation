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

model = UNet(in_channels=6, out_channels=1)
state_dict = torch.load("best_model.pt", map_location="cpu")
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()

PATCH_SIZE = 256

def _resolve_path(tif_file):
    """gr.File may hand back a plain path string (Gradio 5.x) or an object
    with a .name attribute (older Gradio). Handle both."""
    return tif_file.name if hasattr(tif_file, "name") else tif_file

@spaces.GPU(duration=60)
def predict(tif_file):
    path = _resolve_path(tif_file)

    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0)

    if arr.shape[0] != 6:
        return None, f"Expected 6 bands, got {arr.shape[0]}"

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
    mask = (pred * 255).astype(np.uint8)
    return mask, f"Predicted flood coverage: {flood_pct:.2f}%"

demo = gr.Interface(
    fn=predict,
    inputs=gr.File(label="Upload 6-band GeoTIFF"),
    outputs=[gr.Image(label="Predicted Flood Mask"), gr.Textbox(label="Summary")],
    title="Kosi River Flood Segmentation (SAR U-Net)",
    description="U-Net trained on Sentinel-1 SAR imagery over the Kosi river basin, Bihar, India. Developed by Satwik Shreshth.",
)

demo.queue()
demo.launch()