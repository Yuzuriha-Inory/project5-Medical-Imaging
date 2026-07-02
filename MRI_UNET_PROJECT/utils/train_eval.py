# ============================================================
# 训练与验证循环
# ------------------------------------------------------------
# 这一板块：
#   1.负责单轮训练、完整训练流程、验证指标的记录和早停控制。
#   2.模型结构、损失函数、数据集划分和可视化在其他板块，避免训练流程过于臃肿。
#   3.训练过程中会同时记录 train 和 val 的 Dice / IoU，便于后续画曲线和写实验分析。
#   4.验证阶段默认使用 threshold=0.5，最终测试前的最佳 threshold 会在 metrics.py 中单独搜索。
# ============================================================

import time
import copy

import numpy as np
import torch
import torch.optim as optim
from tqdm.auto import tqdm

from config import USE_EARLY_STOPPING,PATIENCE
from utils.metrics import  eval_loader


def _build_history_dict():
    """
    创建训练日志字典。
    把所有需要保存到 csv 的字段集中放在一起，后面追加数据时不容易漏。
    """
    return {
        "epoch": [],
        "train_loss": [],
        "train_eval_loss": [],
        "train_dice_all": [],
        "train_dice_pos": [],
        "train_iou_all": [],
        "train_iou_pos": [],
        "train_acc": [],
        "val_loss": [],
        "val_dice_all": [],
        "val_dice_pos": [],
        "val_iou_all": [],
        "val_iou_pos": [],
        "val_acc": [],
        "lr": [],
        "time_sec": [],
    }

def _append_history(history,epoch,train_loss,train_metrics,val_metrics,lr,elapsed):
    """把当前 epoch 的训练和验证结果追加到 history 中。"""

    # 基础训练信息
    history["epoch"].append(epoch)
    history["train_loss"].append(train_loss)
    history["train_eval_loss"].append(train_metrics["loss"])

    # 训练集评估指标：用于观察模型有没有明显欠拟合或过拟合
    history["train_dice_all"].append(train_metrics["dice_all"])
    history["train_dice_pos"].append(train_metrics["dice_pos"])
    history["train_iou_all"].append(train_metrics["iou_all"])
    history["train_iou_pos"].append(train_metrics["iou_pos"])
    history["train_acc"].append(train_metrics["acc"])

    # 验证集评估指标：用于选最佳模型和画主要实验曲线
    history["val_loss"].append(val_metrics["loss"])
    history["val_dice_all"].append(val_metrics["dice_all"])
    history["val_dice_pos"].append(val_metrics["dice_pos"])
    history["val_iou_all"].append(val_metrics["iou_all"])
    history["val_iou_pos"].append(val_metrics["iou_pos"])
    history["val_acc"].append(val_metrics["acc"])

    # 额外记录学习率和每轮耗时，方便排查训练是否异常
    history["lr"].append(lr)
    history["time_sec"].append(elapsed)

def _print_epoch_log(exp_name,epoch,num_epochs,train_loss,train_metrics,val_metrics,lr,elapsed):
    """统一打印每轮训练日志"""
    print(
        f"[{exp_name}] Ep {epoch:02d}/{num_epochs} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_metrics['loss']:.4f} | "
        f"train_iou_pos={train_metrics['iou_pos']:.4f} | "
        f"val_dice_pos={val_metrics['dice_pos']:.4f} | "
        f"val_iou_pos={val_metrics['iou_pos']:.4f} | "
        f"lr={lr:.6f} | "
        f"{elapsed:.1f}s"
    )

def _backward_and_step(loss,model,optimizer,scaler=None):
    """
    反向传播并更新参数。
    因此单独封装AMP反向传播写法
    使 train_one_epoch() 中的主逻辑更清楚。
    """
    if scaler is not None:
        # AMP 模式下先 scale loss，再 backward。
        scaler.scale(loss).backward()

        # 梯度裁剪前先 unscale，否则裁剪到的是放大后的梯度。
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()
    else:
        # CPU或未开启 AMP 时，直接正常反向传播。
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
        optimizer.step()

def train_one_epoch(model,loader,criterion,optimizer,device,scaler):
    """
    训练一个 epoch，并返回平均训练损失。

    model: 当前训练的分割模型
    loader: 训练集 DataLoader
    criterion: 损失函数
    optimizer: 优化器
    device: cuda 或 cpu
    scaler: AMP 混合精度训练用的 GradScaler；不用 AMP 时为 None
    """

    model.train()
    total_loss = 0.0
    total_n = 0

    for img,mask in tqdm(loader,desc="train",leave=False,ncols=80):
        img = img.to(device,non_blocking=True)
        mask = mask.to(device,non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and device.type == "cuda":
            # 混合精度只在 CUDA 下启用，CPU下不使用 autocast
            with torch.amp.autocast("cuda",enabled=True):
                logits = model(img)
                loss = criterion(logits,mask)
            _backward_and_step(loss,model,optimizer,scaler=scaler)

        else:
            # 普通精度训练流程
            logits = model(img)
            loss = criterion(logits,mask)
            _backward_and_step(loss,model,optimizer,scaler=None)

        # batch size加权累计，得到按样本平均的 loss。
        batch_size = img.size(0)
        total_loss += loss.item() * batch_size
        total_n += batch_size

    return total_loss / max(total_n,1)


def train_model(model,train_loader,train_eval_loader,val_loader,criterion,num_epochs,lr,
                weight_decay,device,exp_name, use_amp):
    """
    完整训练一个模型，并返回最佳模型和训练日志。
    这里的最佳模型只根据验证集 dice_pos 选择，后续也可根据不同场景的实质需求，按照不同指标来选取。
    测试集不参与模型选择，避免把测试集信息泄漏到训练流程中。
    """

    optimizer = optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)

    # CosineAnnealingLR适合固定最大轮数训练；eta_min防止学习率降到0。
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=num_epochs,eta_min=1e-6)

    scaler = torch.amp.GradScaler("cuda") if (use_amp and device.type == "cuda") else None

    history = _build_history_dict()

    best_dice = -1.0
    best_state = None
    best_epoch = -1
    bad_epochs = 0

    for ep in range(1,num_epochs + 1):
        t0 = time.time()

        # 1.先训练一轮，只返回训练loss。
        train_loss = train_one_epoch(model,train_loader,criterion,optimizer,device,scaler)

        # 2.固定阈值0.5评估训练集和验证集，画训练过程曲线。
        train_metrics = eval_loader(model,train_eval_loader,criterion,device,threshold=0.5)
        val_metrics = eval_loader(model,val_loader,criterion,device,threshold=0.5)

        # 记录当前学习率 + scheduler.step()，使得 history 中对应的是本epoch实际使用的 lr。
        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        elapsed = time.time() - t0

        _append_history(history,ep,train_loss,train_metrics,val_metrics,cur_lr,elapsed,)

        _print_epoch_log(exp_name,ep,num_epochs,train_loss,train_metrics,val_metrics, cur_lr,elapsed,)

        # 3. 根据验证集正样本 Dice 选择最佳模型，优先看 dice_pos。
        if val_metrics["dice_pos"] > best_dice:
            best_dice = val_metrics["dice_pos"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = ep
            bad_epochs = 0
        else:
            bad_epochs += 1

        # 4. 早停：验证集连续若干轮没有提升时停止训练。
        #    注意：若config.py 中 USE_EARLY_STOPPING=False，那这段不会生效。
        if USE_EARLY_STOPPING and bad_epochs >= PATIENCE:
            print(f"[{exp_name}] Early stopping at epoch {ep}, best epoch = {best_epoch}")
            break

    # 训练结束后恢复验证集上最好的权重，保证后续测试时使用最佳模型。
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "model": model,
        "history": history,
        "best_val_dice_pos": best_dice,
        "best_epoch": best_epoch,
    }
