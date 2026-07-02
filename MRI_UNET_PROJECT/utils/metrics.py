# ============================================================
# 评估指标、阈值搜索与预测结果整理
# ------------------------------------------------------------
# 这一板块主要负责三件事：
#   1.将模型输出的logits转成二值mask，并计算Dice / IoU / Pixel Accuracy；
#   2.在验证集上搜索合适的threshold和连通域后处理参数；
#   3.在测试集上逐张图像收集预测指标，方便后续保存CSV与做病例级分析。
#
# 说明：
#   ①训练阶段不应使用test集选择 threshold 或 min_size，threshold和后处理参数都只在验证集上确定；
#   ②test集只用于最终报告一次结果，避免评估泄漏。
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import (
    THRESHOLD_VALUES,
    POST_PROCESS_MIN_SIZES,
    POST_PROCESS_FILL_HOLES,
)

try:
    from scipy import ndimage as ndi
except Exception:
    ndi = None

def _as_bool_mask(mask):
    """将输入 mask 统一转成 bool 类型，便于后续连通域处理。"""
    return np.asarray(mask).astype(bool)


def _to_float_mask(mask):
    """将 bool mask 转成 float32"""
    return mask.astype(np.float32)


def _score_from_metrics(metrics):
    """
    验证集参数搜索时使用的综合分数。
    dice_pos 直接反映含肿瘤切片的重叠质量，权重可以设置稍高？
    iou_pos 对过分割和漏分割更敏感，因此也具有一定的权重。
    """
    return 0.7 * metrics["dice_pos"] + 0.3 * metrics["iou_pos"]


def post_process_mask(pred_mask,min_size=0,fill_holes=True):
    """
    对单张二值预测 mask 做连通域后处理。

    pred_mask: 单张预测 mask，形状通常为 (H, W)
    min_size: 小于该面积的连通区域会被删除
    fill_holes: 是否对预测区域进行填洞

    返回：
        float32 类型 mask，取值为 0 或 1
    """
    pred_mask = pred_mask.astype(bool)

    # min_size<=0 表示不删除小连通域，直接返回原始预测。
    if min_size <= 0:
        return pred_mask.astype(np.float32)

    # 若scipy不可用，跳过后处理。
    if ndi is None:
        return pred_mask.astype(np.float32)

    # 对预测区域填洞，可以让一些内部小黑洞被补上。
    if fill_holes:
        pred_mask = ndi.binary_fill_holes(pred_mask)

    labeled, num = ndi.label(pred_mask)
    cleaned = np.zeros_like(pred_mask,dtype=bool)

    # 0表示背景，从1开始遍历
    for i in range(1,num + 1):
        region = (labeled == i)

        # 面积足够大的区域保留，面积太小的区域视为零散假阳性。
        if int(region.sum()) >= min_size:
            cleaned[region] = True

    return cleaned.astype(np.float32)


@torch.no_grad()
def compute_seg_metrics(logits,target,threshold=0.5,smooth=1e-6,
                        post_process_min_size=0,post_process_fill_holes=True):
    """
    逐样本计算分割指标。

    logits: 模型原始输出，(B, 1, H, W)
    target: 真实 mask， (B, 1, H, W)
    threshold: 概率图转二值 mask 的阈值
    smooth: 防止除零的小常数
    post_process_min_size: 连通域最小面积阈值
    post_process_fill_holes: 是否填洞

    返回：
        dice: 每张图的 Dice
        iou: 每张图的 IoU
        acc: 每张图的像素准确率
        has_pos: 每张图是否含真实病灶
    """
    # 转为概率图
    prob = torch.sigmoid(logits)

    # 将概率图转成二值mask。threshold会在验证集上搜索。
    pred = (prob > threshold).float()

    # AI提供思路：可选后处理，删除小连通域，减少零散假阳性。
    if post_process_min_size > 0:
        pred_np = pred.detach().cpu().numpy()
        processed = []
        for b in range(pred_np.shape[0]):
            mask_np = pred_np[b, 0]
            pp = post_process_mask(
                mask_np,
                min_size=post_process_min_size,
                fill_holes=post_process_fill_holes,
            )
            processed.append(pp[None, ...])
        pred = torch.from_numpy(np.stack(processed,axis=0)).to(target.device).float()

    # 拉平为(B,H*W)，一次性计算每张图的Dice/IoU。
    pred_f = pred.flatten(1)
    t_f = target.flatten(1)

    #Dice,IoU,acc计算过程
    inter = (pred_f * t_f).sum(1)
    pred_sum = pred_f.sum(1)
    t_sum = t_f.sum(1)
    union = pred_sum + t_sum - inter

    dice = (2 * inter + smooth) / (pred_sum + t_sum + smooth)
    iou = (inter + smooth) / (union + smooth)

    # Pixel accuracy会被大量背景像素抬高，只能作为辅助指标。
    acc = (pred_f == t_f).float().mean(1)

    # has_pos用于区分含肿瘤切片和无肿瘤切片。
    # dice_pos / iou_pos只在has_pos=True的样本上计算，用于体现模型对阳性样本的识别情况
    has_pos = t_sum > 0

    return dice.cpu().numpy(),iou.cpu().numpy(),acc.cpu().numpy(),has_pos.cpu().numpy()

@torch.no_grad()
def eval_loader(model,loader,criterion,device,threshold=0.5,
                post_process_min_size=0,post_process_fill_holes=True):
    """
    评估模型。
    返回：
        dice_all / iou_all：全部切片平均，包括无肿瘤切片；
        dice_pos / iou_pos：只统计含肿瘤切片，更能反映病灶分割能力；
        acc：像素准确率，仅作为辅助参考；
        n_pos / n_total：阳性切片数和总切片数。
    """
    model.eval()

    total_loss = 0.0
    total_n = 0

    all_dice = []
    all_iou = []
    all_acc = []
    all_pos = []

    for img,mask in loader:
        img = img.to(device,non_blocking=True)
        mask = mask.to(device,non_blocking=True)

        logits = model(img)
        loss = criterion(logits,mask)

        d,i,a,hp = compute_seg_metrics(
            logits,
            mask,
            threshold=threshold,
            post_process_min_size=post_process_min_size,
            post_process_fill_holes=post_process_fill_holes,
        )

        all_dice.append(d)
        all_iou.append(i)
        all_acc.append(a)
        all_pos.append(hp)

        # 按batch size加权累计loss，最后除以总样本数
        total_loss += loss.item() * img.size(0)
        total_n += img.size(0)

    dice = np.concatenate(all_dice) if all_dice else np.array([0.0])
    iou = np.concatenate(all_iou) if all_iou else np.array([0.0])
    acc = np.concatenate(all_acc) if all_acc else np.array([0.0])
    pos = np.concatenate(all_pos) if all_pos else np.array([False])

    # 含肿瘤切片指标单独统计，避免被大量无肿瘤切片掩盖。
    dice_pos = dice[pos].mean() if pos.any() else 0.0
    iou_pos = iou[pos].mean() if pos.any() else 0.0

    return {
        "loss": total_loss / max(total_n, 1),
        "dice_all": float(dice.mean()),
        "dice_pos": float(dice_pos),
        "iou_all": float(iou.mean()),
        "iou_pos": float(iou_pos),
        "acc": float(acc.mean()),
        "n_pos": int(pos.sum()),
        "n_total": int(len(pos)),
    }


@torch.no_grad()
def tune_threshold(model,val_loader,criterion,device,thresholds=THRESHOLD_VALUES):
    """
    在验证集上搜索最佳 threshold。

    不同损失函数训练出的概率分布可能不同，所以固定阈值 0.5不一定最优。但若是为了比较的公平性或许还是应设同样的阈值？
    此处设置阈值搜索，姑且是为了得到每个损失函数的最佳发挥
    仅使用验证集选择 threshold，测试集不参与选择，避免评估泄漏。
    """
    best_threshold = 0.5
    best_score = -1.0
    best_metrics = None

    for th in thresholds:
        m = eval_loader(model,val_loader,criterion,device,threshold=th)
        score = _score_from_metrics(m)
        if score > best_score:
            best_score = score
            best_threshold = th
            best_metrics = m

    return best_threshold, best_metrics


@torch.no_grad()
def tune_postprocess_min_size(model,val_loader,criterion,device,threshold,
                              min_sizes=POST_PROCESS_MIN_SIZES):
    """
    参考 AI,固定最佳 threshold 后，在验证集上搜索连通域最小面积 min_size。

    min_size 太小：零散假阳性删不干净；
    min_size 太大：真实小病灶可能被误删。
    因此在验证集上选择一个折中值。
    """
    best_min_size = 0
    best_score = -1.0
    best_metrics = None

    for min_size in min_sizes:
        m = eval_loader(
            model,
            val_loader,
            criterion,
            device,
            threshold=threshold,
            post_process_min_size=min_size,
            post_process_fill_holes=POST_PROCESS_FILL_HOLES,
        )
        score = _score_from_metrics(m)
        if score > best_score:
            best_score = score
            best_min_size = min_size
            best_metrics = m

    return best_min_size,best_metrics


@torch.no_grad()
def predict_and_collect(model,loader,samples,device,threshold=0.5,
                     post_process_min_size=0):
    """
    对 DataLoader 中的样本逐张预测，并整理成 DataFrame。

    主要用于测试集结果保存：
        case: 病例名
        image: 图像文件名
        mask: mask 文件名
        has_tumor: 真实 mask 是否含肿瘤
        dice / iou / pixel_acc: 单张图像指标
    """
    model.eval()

    rows = []
    sample_idx = 0

    for img,mask in loader:
        img = img.to(device,non_blocking=True)
        mask = mask.to(device,non_blocking=True)

        logits = model(img)

        d, i, a, hp = compute_seg_metrics(
            logits,
            mask,
            threshold=threshold,
            post_process_min_size=post_process_min_size,
            post_process_fill_holes=POST_PROCESS_FILL_HOLES,
        )

        # loader不返回文件名，用 sample_idx 和 samples 对齐。
        # 测试/验证集 loader 使用 shuffle=False，顺序可靠。
        for k in range(img.size(0)):
            img_path, mask_path = samples[sample_idx]
            case_name = Path(img_path).parent.name
            rows.append({
                "case": case_name,
                "image": Path(img_path).name,
                "mask": Path(mask_path).name,
                "has_tumor": int(hp[k]),
                "dice": float(d[k]),
                "iou": float(i[k]),
                "pixel_acc": float(a[k]),
            })
            sample_idx += 1
    return pd.DataFrame(rows)


