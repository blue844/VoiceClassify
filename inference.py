# filename: inference_gender_audio.py
import os
import json
import numpy as np
import torch
import librosa
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ====== 配置区 ======
@dataclass
class Config:
    sample_rate: int = 16000
    target_duration_sec: float = 3.0
    use_mfcc: bool = False
    n_mels: int = 64
    n_mfcc: int = 40
    n_fft: int = 1024
    hop_length: int = 256
    fmin: int = 50
    fmax: int = 7600
    center_crop: bool = True
    feature_normalize: bool = True
    use_gpu: bool = True
    top_k: int = 2

    def device(self) -> str:
        return "cuda" if (self.use_gpu and torch.cuda.is_available()) else "cpu"


cfg = Config()

# ====== 模型定义（保持与训练时一致） ======
class ConvBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1, pool=2):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, kernel_size=k, padding=p, bias=False),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=pool),
        )

    def forward(self, x):
        return self.net(x)


class GenderCNN(torch.nn.Module):
    def __init__(self, num_classes: int = 2, in_ch: int = 1, base_ch: int = 32):
        super().__init__()
        self.features = torch.nn.Sequential(
            ConvBlock(in_ch, base_ch, k=3, p=1, pool=2),
            ConvBlock(base_ch, base_ch * 2, k=3, p=1, pool=2),
            ConvBlock(base_ch * 2, base_ch * 4, k=3, p=1, pool=2),
        )
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(base_ch * 4, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


# ====== 工具函数 ======
def load_audio_fixed(path: str, sr: int, target_len: int, center_crop: bool = True):
    y, _ = librosa.load(path, sr=sr, mono=True)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    elif len(y) > target_len:
        if center_crop:
            start = (len(y) - target_len) // 2
        else:
            start = 0
        y = y[start:start + target_len]
    return y


def extract_features(y: np.ndarray, sr: int, cfg: Config) -> np.ndarray:
    if cfg.use_mfcc:
        feat = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=cfg.n_mfcc, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
            n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax
        )
    else:
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
            n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax, power=2.0
        )
        feat = librosa.power_to_db(mel, ref=np.max)

    if cfg.feature_normalize:
        mean = feat.mean(axis=1, keepdims=True)
        std = feat.std(axis=1, keepdims=True) + 1e-6
        feat = (feat - mean) / std

    return feat.astype(np.float32)


def load_label_map(path: str) -> Dict[int, str]:
    with open(path, "r", encoding="utf-8") as f:
        maps = json.load(f)
    return {int(k): v for k, v in maps["idx2label"].items()}


def load_model(model_path: str, num_classes: int, cfg: Config) -> torch.nn.Module:
    model = GenderCNN(num_classes=num_classes)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state["model_state"])
    model.to(cfg.device())
    model.eval()
    return model


def predict(file_path: str, model: torch.nn.Module, label_map: Dict[int, str], cfg: Config):
    target_len = int(cfg.sample_rate * cfg.target_duration_sec)
    y = load_audio_fixed(file_path, cfg.sample_rate, target_len, cfg.center_crop)
    feat = extract_features(y, cfg.sample_rate, cfg)
    feat_tensor = torch.from_numpy(np.expand_dims(feat, axis=(0, 1))).to(cfg.device())

    with torch.no_grad():
        logits = model(feat_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        top_k_idx = probs.argsort()[::-1][:cfg.top_k]

    results = [(label_map[i], float(probs[i])) for i in top_k_idx]
    return results


# ====== 主入口 ======
if __name__ == "__main__":
    # 修改为你的模型文件和标签映射文件路径
    model_path = "./outputs/best_model.pt"
    label_map_path = "./outputs/label_map.json"
    audio_path = r"organized_audio/test/male\from_bytes_139.tmp.wav"  # 你要预测的音频文件
# D:\GitHub\VoiceClassify\VoiceClassify\organized_audio\test\male\from_bytes_139.tmp.wav
    label_map = load_label_map(label_map_path)
    model = load_model(model_path, num_classes=len(label_map), cfg=cfg)

    results = predict(audio_path, model, label_map, cfg)
    print(f"文件: {audio_path}")
    for label, prob in results:
        print(f"{label}: {prob:.4f}")
