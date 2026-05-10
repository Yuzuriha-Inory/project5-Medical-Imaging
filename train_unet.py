"""
PJ-CV2 基于 U-Net 的脑部 MRI 病灶分割 - 单文件训练脚本

前置: 数据已就位 C:\\Users\\23027\\Desktop\\10\\kaggle_3m
       (110 个 TCGA_xxx 病例, 3929 张 256x256 图像 + 配对 mask)

实验设计 (3 组对比):
    1. unet_bce      U-Net + BCEWithLogitsLoss (二值交叉熵)
    2. unet_dice     U-Net + Dice Loss
    3. unet_bce_dice U-Net + BCE + Dice 联合损失 (改进版)

数据划分: 严格按病例 (case-level) 70/15/15 划分, 同源切片绝不跨集合.

用法 (Jupyter):
    exec(open("train_unet.py", encoding="utf-8").read())

用法 (终端):
    python train_unet.py
"""
# ============================================================
# 1. 配置
# ============================================================
DATA_DIR    = r"C:/Users/肖澎/Desktop/医学图像/data"
OUTPUT_DIR  = r"C:/Users/肖澎/Desktop/医学图像/outputs_cv2"
SEED        = 42
EPOCHS      = 25
BATCH_SIZE  = 16
LR          = 1e-3
WEIGHT_DECAY= 1e-4
IMG_SIZE    = 256
NUM_WORKERS = 0       # Windows + Jupyter 必须 0
USE_AMP     = True
BASE_CH     = 32      # U-Net 基础通道数 (32 -> 7.7M 参数; 64 -> 31M)

# 3 组损失函数实验
EXPERIMENTS = ["unet_bce", "unet_dice", "unet_bce_dice"]


# ============================================================
# 2. 导入 + 种子
# ============================================================
import os
import sys
import json
import time
import copy
import random
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

env_info = {
    "python_version": sys.version.split()[0],
    "platform":       platform.platform(),
    "torch_version":  torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device":    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "seed":           SEED,
    "epochs":         EPOCHS,
    "batch_size":     BATCH_SIZE,
    "img_size":       IMG_SIZE,
    "base_ch":        BASE_CH,
    "data_dir":       DATA_DIR,
}
print("=== 环境 ===")
for k, v in env_info.items():
    print(f"  {k}: {v}")
with open(os.path.join(OUTPUT_DIR, "env_info.json"), "w", encoding="utf-8") as f:
    json.dump(env_info, f, ensure_ascii=False, indent=2)


# ============================================================
# 3. 数据集 - 按病例划分, 严格无泄漏
# ============================================================
def collect_cases(data_dir):
    """返回 {case_name: [(img_path, mask_path), ...]}"""
    root = Path(data_dir)
    cases = {}
    for cd in sorted(root.iterdir()):
        if not cd.is_dir() or not cd.name.startswith("TCGA_"):
            continue
        samples = []
        for img_path in sorted(cd.glob("*.tif")):
            if "_mask" in img_path.name:
                continue
            mask_path = cd / f"{img_path.stem}_mask.tif"
            if mask_path.exists():
                samples.append((str(img_path), str(mask_path)))
        if samples:
            cases[cd.name] = samples
    return cases


def split_cases(case_names, seed=42, train_ratio=0.7, val_ratio=0.15):
    """对病例做随机划分(同源切片严格不跨集合)"""
    rng = np.random.RandomState(seed)
    case_names = list(case_names)
    perm = rng.permutation(len(case_names))
    n = len(case_names)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    train_cs = [case_names[i] for i in perm[:n_train]]
    val_cs = [case_names[i] for i in perm[n_train:n_train + n_val]]
    test_cs = [case_names[i] for i in perm[n_train + n_val:]]
    return train_cs, val_cs, test_cs


class LGGDataset(Dataset):
    """LGG MRI 二值分割数据集.
    image: (3, H, W) tensor in [0, 1] -> normalized to [-1, 1]
    mask:  (1, H, W) tensor in {0, 1}
    """
    IMG_MEAN = [0.5, 0.5, 0.5]
    IMG_STD  = [0.5, 0.5, 0.5]

    def __init__(self, samples, augment=False, img_size=256):
        self.samples = samples
        self.augment = augment
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    def _joint_transform(self, img, mask):
        """对 image 与 mask 同步施加几何增强"""
        # resize (训练 / 验证 / 测试 都需要)
        img = TF.resize(img, [self.img_size, self.img_size],
                        interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.img_size, self.img_size],
                         interpolation=TF.InterpolationMode.NEAREST)
        if self.augment:
            # horizontal flip
            if random.random() < 0.5:
                img = TF.hflip(img); mask = TF.hflip(mask)
            # vertical flip
            if random.random() < 0.5:
                img = TF.vflip(img); mask = TF.vflip(mask)
            # rotation
            angle = random.uniform(-15, 15)
            img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST, fill=0)
            # 仅图像做亮度/对比度抖动 (mask 不变)
            if random.random() < 0.5:
                img = TF.adjust_brightness(img, brightness_factor=random.uniform(0.85, 1.15))
                img = TF.adjust_contrast(img, contrast_factor=random.uniform(0.85, 1.15))
        return img, mask

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        img, mask = self._joint_transform(img, mask)

        # 转 tensor
        img_t = TF.to_tensor(img)  # [0,1]
        img_t = TF.normalize(img_t, self.IMG_MEAN, self.IMG_STD)
        mask_arr = (np.array(mask) > 0).astype(np.float32)
        mask_t = torch.from_numpy(mask_arr).unsqueeze(0)  # (1, H, W)
        return img_t, mask_t


# ============================================================
# 4. U-Net 模型
# ============================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """经典 U-Net (Ronneberger et al., 2015)
    Encoder: 4 个下采样块, 通道 base -> base*16
    Decoder: 4 个上采样块, 用 ConvTranspose2d + skip connection
    Output:  1x1 conv -> 1 通道 logits (二值分割)
    """
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)         # 256
        self.enc2 = DoubleConv(base, base * 2)       # 128
        self.enc3 = DoubleConv(base * 2, base * 4)   # 64
        self.enc4 = DoubleConv(base * 4, base * 8)   # 32
        self.bottleneck = DoubleConv(base * 8, base * 16)  # 16
        self.pool = nn.MaxPool2d(2, 2)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.head = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ============================================================
# 5. 损失函数
# ============================================================
class DiceLoss(nn.Module):
    """二值 soft Dice loss"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, target):
        prob = torch.sigmoid(logits)
        prob = prob.flatten(1)
        target = target.flatten(1)
        inter = (prob * target).sum(1)
        denom = prob.sum(1) + target.sum(1)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    """BCE + Dice 联合损失"""
    def __init__(self, smooth=1.0, w_bce=0.5, w_dice=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth)
        self.w_bce = w_bce
        self.w_dice = w_dice
    def forward(self, logits, target):
        return self.w_bce * self.bce(logits, target) + self.w_dice * self.dice(logits, target)


def get_loss(name):
    if name == "unet_bce":
        return nn.BCEWithLogitsLoss()
    if name == "unet_dice":
        return DiceLoss(smooth=1.0)
    if name == "unet_bce_dice":
        return BCEDiceLoss(smooth=1.0, w_bce=0.5, w_dice=0.5)
    raise ValueError(name)


# ============================================================
# 6. 评估指标
# ============================================================
@torch.no_grad()
def compute_seg_metrics(logits, target, threshold=0.5, smooth=1e-6):
    """逐样本计算 dice/iou/pixel_acc.
    返回 (dice_per_sample, iou_per_sample, acc_per_sample, has_pos_per_sample)
    has_pos: target 是否含正像素 (True/False), 用于区分含/不含肿瘤切片
    """
    prob = torch.sigmoid(logits)
    pred = (prob > threshold).float()
    t = target
    B = t.size(0)
    pred_f = pred.flatten(1)
    t_f = t.flatten(1)
    inter = (pred_f * t_f).sum(1)
    union = pred_f.sum(1) + t_f.sum(1) - inter
    dice = (2 * inter + smooth) / (pred_f.sum(1) + t_f.sum(1) + smooth)
    iou  = (inter + smooth) / (union + smooth)
    acc  = (pred_f == t_f).float().mean(1)
    has_pos = (t_f.sum(1) > 0)
    return dice.cpu().numpy(), iou.cpu().numpy(), acc.cpu().numpy(), has_pos.cpu().numpy()


# ============================================================
# 7. 训练 / 评估循环
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss, total_n = 0.0, 0
    for img, mask in tqdm(loader, desc="train", leave=False, ncols=80):
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(img)
                loss = criterion(logits, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(img)
            loss = criterion(logits, mask)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * img.size(0)
        total_n += img.size(0)
    return total_loss / total_n


@torch.no_grad()
def eval_loader(model, loader, criterion, device):
    """返回 (avg_loss, avg_dice_all, avg_dice_pos, avg_iou_all, avg_iou_pos, avg_acc)"""
    model.eval()
    total_loss = 0.0; total_n = 0
    all_dice, all_iou, all_acc, all_pos = [], [], [], []
    for img, mask in loader:
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        logits = model(img)
        loss = criterion(logits, mask)
        d, i, a, hp = compute_seg_metrics(logits, mask)
        all_dice.append(d); all_iou.append(i); all_acc.append(a); all_pos.append(hp)
        total_loss += loss.item() * img.size(0)
        total_n += img.size(0)
    dice = np.concatenate(all_dice)
    iou = np.concatenate(all_iou)
    acc = np.concatenate(all_acc)
    pos = np.concatenate(all_pos)
    return {
        "loss": total_loss / total_n,
        "dice_all":  float(dice.mean()),
        "dice_pos":  float(dice[pos].mean()) if pos.any() else 0.0,
        "iou_all":   float(iou.mean()),
        "iou_pos":   float(iou[pos].mean()) if pos.any() else 0.0,
        "acc":       float(acc.mean()),
        "n_pos":     int(pos.sum()),
        "n_total":   int(len(pos)),
    }


def train_model(model, train_loader, val_loader, criterion, num_epochs, lr,
                weight_decay, device, exp_name, use_amp):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None

    history = {"epoch": [], "train_loss": [],
               "val_loss": [], "val_dice_all": [], "val_dice_pos": [],
               "val_iou_all": [], "val_iou_pos": [], "val_acc": [],
               "lr": [], "time_sec": []}
    best_dice, best_state, best_epoch = -1.0, None, -1

    for ep in range(1, num_epochs + 1):
        t0 = time.time()
        tl = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        vm = eval_loader(model, val_loader, criterion, device)
        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        elapsed = time.time() - t0

        history["epoch"].append(ep)
        history["train_loss"].append(tl)
        history["val_loss"].append(vm["loss"])
        history["val_dice_all"].append(vm["dice_all"])
        history["val_dice_pos"].append(vm["dice_pos"])
        history["val_iou_all"].append(vm["iou_all"])
        history["val_iou_pos"].append(vm["iou_pos"])
        history["val_acc"].append(vm["acc"])
        history["lr"].append(cur_lr)
        history["time_sec"].append(elapsed)

        print(f"[{exp_name}] Ep {ep:02d}/{num_epochs} | "
              f"train_loss={tl:.4f} | val_loss={vm['loss']:.4f} "
              f"dice_all={vm['dice_all']:.4f} dice_pos={vm['dice_pos']:.4f} "
              f"iou_pos={vm['iou_pos']:.4f} | lr={cur_lr:.6f} | {elapsed:.1f}s")

        # 用 val_dice_pos 选最佳 (含肿瘤切片的 dice 是真正衡量分割能力的指标)
        if vm["dice_pos"] > best_dice:
            best_dice = vm["dice_pos"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = ep

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"model": model, "history": history,
            "best_val_dice_pos": best_dice, "best_epoch": best_epoch}


@torch.no_grad()
def predict_and_collect(model, loader, samples, device):
    """对 test loader 预测, 返回 per-image 指标 + per-sample 元信息"""
    model.eval()
    rows = []
    sample_idx = 0
    for img, mask in loader:
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        logits = model(img)
        d, i, a, hp = compute_seg_metrics(logits, mask)
        for k in range(img.size(0)):
            img_path, mask_path = samples[sample_idx]
            case_name = Path(img_path).parent.name
            rows.append({
                "case": case_name,
                "image": Path(img_path).name,
                "has_tumor": int(hp[k]),
                "dice": float(d[k]),
                "iou":  float(i[k]),
                "pixel_acc": float(a[k]),
            })
            sample_idx += 1
    return pd.DataFrame(rows)


# ============================================================
# 8. 绘图设置 (字体: 宋体 + Times New Roman, 五号 10.5pt)
# ============================================================
def setup_fonts():
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["Times New Roman", "SimSun"]
    rcParams["font.sans-serif"] = ["SimSun", "Times New Roman"]
    rcParams["axes.unicode_minus"] = False
    rcParams["mathtext.fontset"] = "stix"
    rcParams["font.size"] = 10.5
    rcParams["axes.titlesize"] = 10.5
    rcParams["axes.labelsize"] = 10.5
    rcParams["xtick.labelsize"] = 10.5
    rcParams["ytick.labelsize"] = 10.5
    rcParams["legend.fontsize"] = 10.5
    rcParams["savefig.dpi"] = 300
    rcParams["savefig.bbox"] = "tight"


def plot_learning_curves(histories, save_path):
    setup_fonts()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    cmap = plt.get_cmap("tab10")
    for i, (name, h) in enumerate(histories.items()):
        c = cmap(i % 10); ep = h["epoch"]
        axes[0].plot(ep, h["train_loss"], c=c, ls="-", lw=1.2, label=f"{name}-train")
        axes[0].plot(ep, h["val_loss"],   c=c, ls="--", lw=1.2, label=f"{name}-val")
        axes[1].plot(ep, h["val_dice_all"], c=c, ls="-", lw=1.2, label=f"{name}-all")
        axes[1].plot(ep, h["val_dice_pos"], c=c, ls="--", lw=1.2, label=f"{name}-pos")
        axes[2].plot(ep, h["val_iou_pos"], c=c, ls="-", lw=1.2, label=f"{name}")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].set_title("Loss curve")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Dice"); axes[1].set_title("Validation Dice")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("IoU"); axes[2].set_title("Validation IoU (positive)")
    for ax in axes:
        ax.grid(True, alpha=0.3, ls=":")
        ax.legend(loc="best", fontsize=8.5, ncol=1, framealpha=0.85)
    plt.tight_layout(); plt.savefig(save_path); plt.close(fig)


def plot_metrics_compare(metrics_list, save_path):
    setup_fonts()
    names = [m["exp_name"] for m in metrics_list]
    keys = ["test_dice_all", "test_dice_pos", "test_iou_all", "test_iou_pos"]
    labels = ["Dice (all)", "Dice (pos)", "IoU (all)", "IoU (pos)"]
    x = np.arange(len(names)); width = 0.2
    fig, ax = plt.subplots(figsize=(max(7, 2 * len(names)), 4.2))
    cmap = plt.get_cmap("Set2")
    for i, (k, l) in enumerate(zip(keys, labels)):
        vals = [m[k] for m in metrics_list]
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=l, color=cmap(i))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.set_title("Test set segmentation metrics")
    ax.grid(axis="y", alpha=0.3, ls=":"); ax.legend(loc="lower right", ncol=2)
    plt.tight_layout(); plt.savefig(save_path); plt.close(fig)


def plot_pred_examples(model, dataset, indices, save_path, device, title=""):
    """在测试集上抽几个样本可视化 image / GT / pred"""
    setup_fonts()
    model.eval()
    n = len(indices)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    with torch.no_grad():
        for row, idx in enumerate(indices):
            img, mask = dataset[idx]
            logits = model(img.unsqueeze(0).to(device))
            pred = (torch.sigmoid(logits) > 0.5).float().cpu().squeeze().numpy()
            # 反归一化用于显示
            img_disp = img.numpy().transpose(1, 2, 0) * 0.5 + 0.5
            img_disp = np.clip(img_disp, 0, 1)
            axes[row, 0].imshow(img_disp); axes[row, 0].set_title("Image" if row == 0 else "")
            axes[row, 1].imshow(mask.squeeze().numpy(), cmap="gray"); axes[row, 1].set_title("Ground truth" if row == 0 else "")
            axes[row, 2].imshow(pred, cmap="gray"); axes[row, 2].set_title("Prediction" if row == 0 else "")
            for c in range(3):
                axes[row, c].axis("off")
    if title:
        fig.suptitle(title, fontsize=11, y=0.99)
    plt.tight_layout(); plt.savefig(save_path); plt.close(fig)


# ============================================================
# 9. 数据准备
# ============================================================
print("\n=== 收集病例 ===")
all_cases = collect_cases(DATA_DIR)
case_names = sorted(all_cases.keys())
print(f"病例总数: {len(case_names)}")

train_cs, val_cs, test_cs = split_cases(case_names, seed=SEED,
                                         train_ratio=0.7, val_ratio=0.15)
print(f"划分: train {len(train_cs)} / val {len(val_cs)} / test {len(test_cs)} 病例")

train_samples = [s for c in train_cs for s in all_cases[c]]
val_samples   = [s for c in val_cs   for s in all_cases[c]]
test_samples  = [s for c in test_cs  for s in all_cases[c]]
print(f"切片数:  train {len(train_samples)} / val {len(val_samples)} / test {len(test_samples)}")

# 保存划分细节(用于报告 + 可复现)
split_rows = []
for c in train_cs: split_rows.append({"case": c, "split": "train", "n_slices": len(all_cases[c])})
for c in val_cs:   split_rows.append({"case": c, "split": "val",   "n_slices": len(all_cases[c])})
for c in test_cs:  split_rows.append({"case": c, "split": "test",  "n_slices": len(all_cases[c])})
df_split = pd.DataFrame(split_rows)
df_split.to_csv(os.path.join(OUTPUT_DIR, "case_split.csv"), index=False)

# 构造 dataset / dataloader (3 组实验复用同一份划分)
def make_loaders():
    tr_ds = LGGDataset(train_samples, augment=True,  img_size=IMG_SIZE)
    va_ds = LGGDataset(val_samples,   augment=False, img_size=IMG_SIZE)
    te_ds = LGGDataset(test_samples,  augment=False, img_size=IMG_SIZE)
    g = torch.Generator(); g.manual_seed(SEED)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                       num_workers=NUM_WORKERS, pin_memory=True, generator=g)
    va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=True)
    te_ld = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=True)
    return tr_ds, va_ds, te_ds, tr_ld, va_ld, te_ld


# ============================================================
# 10. 跑 3 组实验
# ============================================================
def run_one(exp_name):
    set_seed(SEED)
    tr_ds, va_ds, te_ds, tr_ld, va_ld, te_ld = make_loaders()
    model = UNet(in_ch=3, out_ch=1, base=BASE_CH).to(device)
    n_params = count_params(model)
    criterion = get_loss(exp_name)
    print(f"\n{'='*70}\n实验: {exp_name} | 参数: {n_params:,} | "
          f"loss: {criterion.__class__.__name__}\n{'='*70}")

    res = train_model(model, tr_ld, va_ld, criterion, EPOCHS, LR,
                      WEIGHT_DECAY, device, exp_name, USE_AMP)

    # test 评估
    test_metrics = eval_loader(res["model"], te_ld,
                                nn.BCEWithLogitsLoss(), device)
    # per-image 详细预测
    pred_df = predict_and_collect(res["model"], te_ld, test_samples, device)

    summary = {
        "exp_name": exp_name,
        "loss":  criterion.__class__.__name__,
        "n_params": n_params,
        "best_epoch": res["best_epoch"],
        "best_val_dice_pos": res["best_val_dice_pos"],
        "test_loss":     test_metrics["loss"],
        "test_dice_all": test_metrics["dice_all"],
        "test_dice_pos": test_metrics["dice_pos"],
        "test_iou_all":  test_metrics["iou_all"],
        "test_iou_pos":  test_metrics["iou_pos"],
        "test_acc":      test_metrics["acc"],
    }
    print(f"\n[{exp_name}] 测试集结果:")
    for k in ("test_dice_all", "test_dice_pos", "test_iou_all",
              "test_iou_pos", "test_acc"):
        print(f"  {k}: {summary[k]:.4f}")

    return {
        "exp_name": exp_name, "history": res["history"],
        "metrics": summary, "pred_df": pred_df,
        "model": res["model"], "test_ds": te_ds,
    }


all_results = []
for exp in EXPERIMENTS:
    r = run_one(exp)
    all_results.append(r)
print("\n=== 全部实验完成 ===")


# ============================================================
# 11. 保存 + 绘图
# ============================================================
# 1) learning_curve.csv (合并 3 组)
rows = []
for r in all_results:
    h = r["history"]; en = r["exp_name"]
    for i, ep in enumerate(h["epoch"]):
        rows.append({"exp_name": en, "epoch": ep,
                     "train_loss":   h["train_loss"][i],
                     "val_loss":     h["val_loss"][i],
                     "val_dice_all": h["val_dice_all"][i],
                     "val_dice_pos": h["val_dice_pos"][i],
                     "val_iou_all":  h["val_iou_all"][i],
                     "val_iou_pos":  h["val_iou_pos"][i],
                     "val_acc":      h["val_acc"][i],
                     "lr":           h["lr"][i],
                     "time_sec":     h["time_sec"][i]})
pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "learning_curve.csv"), index=False)

# 2) test_metrics.csv
df_metrics = pd.DataFrame([r["metrics"] for r in all_results])
df_metrics.to_csv(os.path.join(OUTPUT_DIR, "test_metrics.csv"), index=False)
print("\n=== 全部实验测试集指标 ===")
print(df_metrics.to_string(index=False))

# 3) per-image predictions for each experiment
for r in all_results:
    out = os.path.join(OUTPUT_DIR, f"predictions_{r['exp_name']}.csv")
    r["pred_df"].to_csv(out, index=False)

# 4) per-case summary (Dice 按病例平均)
for r in all_results:
    case_summary = (r["pred_df"]
                    .groupby("case")
                    .agg(n=("dice","size"), n_pos=("has_tumor","sum"),
                         dice_all=("dice","mean"),
                         iou_all=("iou","mean"),
                         pixel_acc=("pixel_acc","mean"))
                    .reset_index())
    case_summary.to_csv(
        os.path.join(OUTPUT_DIR, f"per_case_{r['exp_name']}.csv"), index=False)

# 5) 最佳模型权重保存
best = max(all_results, key=lambda r: r["metrics"]["test_dice_pos"])
torch.save(best["model"].state_dict(),
           os.path.join(OUTPUT_DIR, f"best_model_{best['exp_name']}.pth"))
print(f"\n最佳实验: {best['exp_name']} | "
      f"test_dice_pos={best['metrics']['test_dice_pos']:.4f}")

# 6) 图像
histories = {r["exp_name"]: r["history"] for r in all_results}
plot_learning_curves(histories, os.path.join(OUTPUT_DIR, "fig_learning_curves.png"))
plot_metrics_compare([r["metrics"] for r in all_results],
                     os.path.join(OUTPUT_DIR, "fig_metrics_compare.png"))

# 7) 可视化最佳模型在 test 集上的几个样本 (选含肿瘤的样本)
best_pred_df = best["pred_df"].copy()
positive_idx = best_pred_df[best_pred_df["has_tumor"] == 1].index.tolist()
sample_indices = positive_idx[:6] if len(positive_idx) >= 6 else positive_idx
if sample_indices:
    plot_pred_examples(
        best["model"], best["test_ds"], sample_indices,
        os.path.join(OUTPUT_DIR, f"fig_pred_samples_{best['exp_name']}.png"),
        device, title=f"Predictions ({best['exp_name']})")

print(f"\n所有结果保存到: {os.path.abspath(OUTPUT_DIR)}")
