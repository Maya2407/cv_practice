import os
import glob
import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

os.makedirs("out_det", exist_ok=True)
os.makedirs("out_seg", exist_ok=True)

print("Loading YOLOv8...")
det_model = YOLO("yolov8n.pt")

print("Loading DeepLabV3...")
weights = DeepLabV3_ResNet50_Weights.DEFAULT
seg_model = deeplabv3_resnet50(weights=weights).eval()
preprocess = weights.transforms()

rng = np.random.RandomState(42)
palette = rng.randint(0, 255, (256, 3), dtype=np.uint8)
palette[0] = [0, 0, 0]

frames = sorted(glob.glob("frames/*.png"))
print("Frames found:", len(frames))

for f in frames:
    name = os.path.basename(f)
    img_bgr = cv2.imread(f)
    if img_bgr is None:
        print("Cannot read", f)
        continue

    res = det_model(img_bgr, verbose=False)[0]
    annotated_det = res.plot()
    cv2.imwrite(os.path.join("out_det", name), annotated_det)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    batch = preprocess(pil_img).unsqueeze(0)
    with torch.no_grad():
        out = seg_model(batch)["out"][0]

    mask = out.argmax(0).byte().cpu().numpy()
    color_mask = palette[mask]

    h, w = img_rgb.shape[:2]
    color_mask = cv2.resize(color_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    overlay = cv2.addWeighted(img_rgb, 0.5, color_mask, 0.5, 0)
    cv2.imwrite(os.path.join("out_seg", name), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print("Done", name)

def make_gif(in_dir, out_path, duration=200):
    files = sorted(glob.glob(os.path.join(in_dir, "*.png")))
    if not files:
        print("No images in", in_dir)
        return
    images = [Image.open(f).convert("P", palette=Image.ADAPTIVE) for f in files]
    images[0].save(out_path, save_all=True, append_images=images[1:], duration=duration, loop=0, optimize=True)
    print("Created", out_path)

make_gif("out_det", "detection.gif")
make_gif("out_seg", "segmentation.gif")
print("ALL DONE")