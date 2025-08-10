# filename: gender_audio_classifier.py
import os
import glob
import json
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import librosa
import soundfile as sf

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


# =========================
# 配置区（无需命令行）
# =========================
@dataclass
class Config:
    # 数据目录：类别名 -> 多个目录路径
    DATA_DIRS: Dict[str, List[str]] = None

    # 音频与特征
    sample_rate: int = 16000
    target_duration_sec: float = 3.0     # 统一区间长度（秒）
    use_mfcc: bool = False               # False=用Log-Mel，True=用MFCC
    n_mels: int = 64
    n_mfcc: int = 40
    n_fft: int = 1024
    hop_length: int = 256
    fmin: int = 50
    fmax: int = 7600
    center_crop: bool = True             # 截断时是否居中裁剪
    feature_normalize: bool = True       # 特征标准化（每条样本）

    # 数据划分与加载
    val_size: float = 0.15
    random_seed: int = 42
    batch_size: int = 32
    num_workers: int = 2
    balance_classes: bool = True         # 是否按类采样平衡

    # 训练参数
    epochs: int = 25
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    use_gpu: bool = True
    grad_clip_norm: float = 5.0

    # 输出
    output_dir: str = "./outputs"
    save_best_only: bool = True

    # 推理
    top_k: int = 2

    def device(self) -> str:
        return "cuda" if (self.use_gpu and torch.cuda.is_available()) else "cpu"


# 你可以直接在这里修改成你的真实目录
cfg = Config(
    DATA_DIRS={
        "male": [r"./organized_audio/male", ],         # 示例：同类多目录
        "female": [r"./organized_audio/female"]
    },
    epochs=30,
    learning_rate=1e-3,
    use_gpu=True,
    batch_size=32,
)


# =========================
# 工具函数
# =========================
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_files(data_dirs: Dict[str, List[str]]) -> Tuple[List[str], List[int], Dict[str, int]]:
    """从多个目录收集音频文件，返回文件路径、标签索引、标签映射。"""
    paths, labels = [], []
    label2idx = {label: i for i, label in enumerate(sorted(data_dirs.keys()))}
    for label, dir_list in data_dirs.items():
        for d in dir_list:
            if not os.path.isdir(d):
                continue
            for ext in AUDIO_EXTS:
                for p in glob.glob(os.path.join(d, f"**/*{ext}"), recursive=True):
                    paths.append(p)
                    labels.append(label2idx[label])
    return paths, labels, label2idx


def load_audio_fixed(path: str, sr: int, target_len: int, center_crop: bool = True):
    """读取音频，转单声道、重采样，并补齐/截断到固定长度（样本点）。"""
    # librosa.load: always mono if mono=True
    y, file_sr = librosa.load(path, sr=sr, mono=True)
    if len(y) == 0:
        # 处理空音频
        y = np.zeros(target_len, dtype=np.float32)

    if len(y) < target_len:
        pad = target_len - len(y)
        y = np.pad(y, (0, pad), mode="constant")
    elif len(y) > target_len:
        if center_crop:
            start = (len(y) - target_len) // 2
        else:
            start = 0
        y = y[start: start + target_len]
    return y


def extract_features(
    y: np.ndarray,
    sr: int,
    cfg: Config
) -> np.ndarray:
    """提取 Log-Mel 或 MFCC 特征，并做 dB/标准化处理。"""
    if cfg.use_mfcc:
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=cfg.n_mfcc,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            fmin=cfg.fmin,
            fmax=cfg.fmax,
        )
        feat = mfcc  # [n_mfcc, T]
    else:
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            fmin=cfg.fmin,
            fmax=cfg.fmax,
            power=2.0,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        feat = log_mel  # [n_mels, T]

    if cfg.feature_normalize:
        mean = feat.mean(axis=1, keepdims=True)
        std = feat.std(axis=1, keepdims=True) + 1e-6
        feat = (feat - mean) / std
    return feat.astype(np.float32)


# =========================
# 数据集与加载器
# =========================
class GenderAudioDataset(Dataset):
    def __init__(self, files: List[str], labels: List[int], cfg: Config):
        self.files = files
        self.labels = labels
        self.cfg = cfg
        self.target_len = int(cfg.sample_rate * cfg.target_duration_sec)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        label = self.labels[idx]
        try:
            y = load_audio_fixed(path, self.cfg.sample_rate, self.target_len, self.cfg.center_crop)
            feat = extract_features(y, self.cfg.sample_rate, self.cfg)
        except Exception as e:
            # 避免坏样本导致崩溃，用全零特征代替
            feat_dim = self.cfg.n_mfcc if self.cfg.use_mfcc else self.cfg.n_mels
            T = 1 + self.target_len // self.cfg.hop_length
            feat = np.zeros((feat_dim, T), dtype=np.float32)

        # [C=1, F, T]
        feat = np.expand_dims(feat, axis=0)
        return torch.from_numpy(feat), torch.tensor(label, dtype=torch.long)


def make_dataloaders(cfg: Config):
    all_files, all_labels, label2idx = collect_files(cfg.DATA_DIRS)
    if len(all_files) == 0:
        raise RuntimeError("未找到任何音频文件，请检查 DATA_DIRS 配置。")

    X_train, X_val, y_train, y_val = train_test_split(
        all_files, all_labels, test_size=cfg.val_size, random_state=cfg.random_seed, stratify=all_labels
    )

    ds_train = GenderAudioDataset(X_train, y_train, cfg)
    ds_val = GenderAudioDataset(X_val, y_val, cfg)

    if cfg.balance_classes:
        # 计算各类别样本数，构造权重
        class_counts = np.bincount(y_train)
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = [class_weights[y] for y in y_train]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(
            ds_train, batch_size=cfg.batch_size, sampler=sampler, num_workers=cfg.num_workers, pin_memory=True
        )
    else:
        train_loader = DataLoader(
            ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True
        )

    val_loader = DataLoader(
        ds_val, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True
    )

    return train_loader, val_loader, label2idx


# =========================
# 模型
# =========================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1, pool=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool),
        )

    def forward(self, x):
        return self.net(x)


class GenderCNN(nn.Module):
    def __init__(self, num_classes: int = 2, in_ch: int = 1, base_ch: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_ch, base_ch, k=3, p=1, pool=2),
            ConvBlock(base_ch, base_ch * 2, k=3, p=1, pool=2),
            ConvBlock(base_ch * 2, base_ch * 4, k=3, p=1, pool=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_ch * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)      # [B, C, F', T']
        x = self.pool(x)          # [B, C, 1, 1]
        x = self.classifier(x)    # [B, num_classes]
        return x


# =========================
# 训练与验证
# =========================
def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=None):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for xb, yb in tqdm(loader, desc="Train", leave=False):
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for xb, yb in tqdm(loader, desc="Val  ", leave=False):
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = criterion(logits, yb)

        running_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(yb.cpu().numpy().tolist())

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc, np.array(all_preds), np.array(all_labels)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_label_map(label2idx: Dict[str, int], out_dir: str):
    idx2label = {v: k for k, v in label2idx.items()}
    with open(os.path.join(out_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({"label2idx": label2idx, "idx2label": idx2label}, f, ensure_ascii=False, indent=2)


def train_pipeline(cfg: Config):
    set_seed(cfg.random_seed)
    ensure_dir(cfg.output_dir)

    print("配置参数：")
    print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))

    train_loader, val_loader, label2idx = make_dataloaders(cfg)
    device = cfg.device()
    print(f"使用设备：{device}")

    num_classes = len(label2idx)
    model = GenderCNN(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_path = os.path.join(cfg.output_dir, "best_model.pt")

    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, grad_clip=cfg.grad_clip_norm
        )
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)

        print(f"Train: loss={tr_loss:.4f} acc={tr_acc:.4f} | Val: loss={val_loss:.4f} acc={val_acc:.4f}")

        if (not cfg.save_best_only) or (val_acc > best_val_acc):
            best_val_acc = max(best_val_acc, val_acc)
            torch.save({"model_state": model.state_dict(), "config": asdict(cfg)}, best_path)
            save_label_map(label2idx, cfg.output_dir)
            print(f"已保存最佳模型到: {best_path}")

    # 最终评估报告
    print("\n验证集分类报告：")
    print(classification_report(val_labels, val_preds, target_names=[k for k, _ in sorted(label2idx.items(), key=lambda x: x[1])]))
    print("混淆矩阵：")
    print(confusion_matrix(val_labels, val_preds))


# =========================
# 推理
# =========================
@torch.no_grad()
def predict(file_path: str, model: nn.Module, cfg: Config, label_map_path: str) -> List[Tuple[str, float]]:
    with open(label_map_path, "r", encoding="utf-8") as f:
        maps = json.load(f)
    idx2label = {int(k): v for k, v in maps["idx2label"].items()}

    device = cfg.device()
    model.eval()
    model.to(device)

    target_len = int(cfg.sample_rate * cfg.target_duration_sec)
    y = load_audio_fixed(file_path, cfg.sample_rate, target_len, cfg.center_crop)
    feat = extract_features(y, cfg.sample_rate, cfg)
    feat = torch.from_numpy(np.expand_dims(feat, axis=(0, 1))).to(device)  # [1, 1, F, T]

    logits = model(feat)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    topk_idx = probs.argsort()[::-1][: cfg.top_k]
    return [(idx2label[i], float(probs[i])) for i in topk_idx]


def load_trained_model(model_path: str, num_classes: int) -> nn.Module:
    model = GenderCNN(num_classes=num_classes)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state["model_state"])
    return model


# =========================
# 主程序入口（无需命令行）
# =========================
if __name__ == "__main__":
    # 训练
    train_pipeline(cfg)

    # 示例推理（可根据需要开启）
    # model_path = "./outputs/best_model.pt"
    # label_map_path = "./outputs/label_map.json"
    # with open(label_map_path, "r", encoding="utf-8") as f:
        # num_classes = len(json.load(f)["idx2label"])
    # model = load_trained_model(model_path, num_classes=num_classes)
    # results = predict("./some_audio.wav", model, cfg, label_map_path)
    # print("预测：", results)
