from typing import Tuple
import cv2
import numpy as np
import torch

def to_tensor(img: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(img).float().permute(2,0,1) / 255.0
    return t

def normalize_img(t: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=t.dtype, device=t.device).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], dtype=t.dtype, device=t.device).view(3,1,1)
    return (t - mean) / std

def resize_square(img: np.ndarray, out_size: int) -> Tuple[np.ndarray, float, np.ndarray]:
    h, w = img.shape[:2]
    scale = out_size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img_r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x = (out_size - nw) / 2.0
    pad_y = (out_size - nh) / 2.0
    top = int(np.floor(pad_y)); bottom = int(np.ceil(pad_y))
    left = int(np.floor(pad_x)); right  = int(np.ceil(pad_x))
    img_p = cv2.copyMakeBorder(img_r, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
    img_p = img_p[:out_size, :out_size]
    pad = np.array([left, top], dtype=np.float32)
    return img_p, float(scale), pad
