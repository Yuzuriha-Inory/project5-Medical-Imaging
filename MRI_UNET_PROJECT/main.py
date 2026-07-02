# ============================================================
# 主程序入口：病例划分、训练、评估与结果保存
# ------------------------------------------------------------
# 注：这一板块负责把各个模块串起来，包括数据准备、病例级划分、DataLoader构造、三组 loss 实验、阈值搜索、后处理、结果保存、绘图输出等。
# ============================================================

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader,WeightedRandomSampler

from config import *

from models.attention_unet import AttentionUNet,count_params

from utils.data_split import (
    collect_cases,
    summarize_cases,
    split_cases_stratified,
    estimate_pos_weight,
    subsample_train_cases,
)

from utils.dataset import LGGDataset

from utils.losses import get_loss

from utils.metrics import (
    tune_threshold,
    tune_postprocess_min_size,
    predict_and_collect,
    eval_loader
)

from utils.seed import set_seed,get_device,save_env_info

from utils.train_eval import train_model

from utils.visualize import (
    analyze_slice_balance,
    save_slice_balance_analysis,
    save_case_diversity_demo,
    plot_learning_curves,
    plot_iou_curve,
    plot_metrics_compare,
    plot_pred_examples,
    save_qualitative_results,
)

def _save_case_split(case_stats,all_cases,train_cases,
                     val_cases,test_cases,output_dir):
    """
    保存病例级划分结果。
    输出 csv 主要包括：
    - 每个病例属于 train / val / test 哪一部分；
    - 每个病例有多少切片；
    - 每个病例的阳性切片比例和阳性像素比例。
    """
    split_rows = []

    # 记录训练集病例
    for case_name in train_cases:
        split_rows.append({
            "case": case_name,
            "split": "train",
            # 当前病例下包含的 MRI 切片数量。
            "n_slices": len(all_cases[case_name]),
        })

    # 记录验证集病例
    for case_name in val_cases:
        split_rows.append({
            "case": case_name,
            "split": "val",
            "n_slices": len(all_cases[case_name]),
        })

    #记录测试集病例
    for case_name in test_cases:
        split_rows.append({
            "case": case_name,
            "split": "test",
            "n_slices": len(all_cases[case_name]),
        })

    df_split = pd.DataFrame(split_rows).merge(case_stats,on="case",how="left")
    df_split.to_csv(
        Path(output_dir) / "case_split.csv",
        index=False,
        encoding="utf-8",
    )

    return df_split

def _build_train_weights(train_samples):
    """
    为 WeightedRandomSampler 构造每张训练切片的采样权重。对阳性切片给一个更高权重，帮助训练阶段更频繁看到病灶样本。
    避免模型受到大量背景图干扰
    """
    train_weights = []

    for _,mask_path in train_samples:
        # 读取 mask 判断是否为阳性切片
        mask = np.array(Image.open(mask_path).convert("L")) > 0
        has_tumor = mask.sum() > 0

        # POS_SAMPLE_WEIGHT 由 config.py设置，应大于 1。
        weight = POS_SAMPLE_WEIGHT if has_tumor else 1.0
        train_weights.append(weight)

    return train_weights

def main():
    """
    主函数：按顺序完成数据准备、训练、评估和结果保存。
    """

    set_seed(SEED)
    device = get_device()

    Path(OUTPUT_DIR).mkdir(parents=True,exist_ok=True)

    # 整理本次实验的关键配置，记录和实验结果强相关的参数
    config_info = {
        "seed": SEED,
        "epochs": EPOCHS,
        "use_early_stopping": USE_EARLY_STOPPING,
        "use_attention": USE_ATTENTION,
        "patience": PATIENCE,
        "batch_size": BATCH_SIZE,
        "img_size": IMG_SIZE,
        "base_ch": BASE_CH,
        "data_dir": DATA_DIR,
        "output_dir": OUTPUT_DIR,
        "model_type": "Attention U-Net" if USE_ATTENTION else "Basic U-Net",
        "input_channels": 1,
        "train_case_fraction": TRAIN_CASE_FRACTION,
        # 各个消融实验开关
        "use_bce_pos_weight": USE_BCE_POS_WEIGHT,
        "use_weighted_sampler": USE_WEIGHTED_SAMPLER,
        "use_threshold_search": USE_THRESHOLD_SEARCH,
        "fixed_threshold": FIXED_THRESHOLD,
        "use_post_process": USE_POST_PROCESS,
        "use_loss_specific_lr": USE_LOSS_SPECIFIC_LR,
        "common_lr": COMMON_LR,
    }

    # ============================================================
    # 1，环境信息记录
    # ============================================================
    print("=== 环境 ===")
    env_info = save_env_info(OUTPUT_DIR,config_info)
    for k,v in env_info.items():
        print(f"{k}: {v}")

    # ============================================================
    # 2，收集病例并统计病例信息
    # ============================================================
    print("\n=== 收集病例 ===")
    all_cases = collect_cases(DATA_DIR)
    case_names = sorted(all_cases.keys())
    print(f"病例总数: {len(case_names)}")

    if len(case_names) == 0:
        raise RuntimeError(f"未在DATA_DIR中找到 TCGA_ 开头的病例文件夹: {DATA_DIR}")

    # 统计每个病例基本信息并保存
    case_stats = summarize_cases(all_cases)
    case_stats.to_csv(
        os.path.join(OUTPUT_DIR,"case_stats.csv"),
        index=False,
        encoding="utf-8",
    )

    # ============================================================
    # 3，病例级划分，避免数据泄漏
    # ============================================================
    train_cs,val_cs,test_cs = split_cases_stratified(
        case_stats,
        seed=SEED,
        train_ratio=0.7, # 比例可随时调整，小样本训练实验在config.py中设置
        val_ratio=0.15,
    )

    # 小样本实验只抽训练病例，验证集和测试集保持不变。
    train_cs_full = train_cs
    train_cs = subsample_train_cases(
        train_cs_full,
        case_stats,
        fraction=TRAIN_CASE_FRACTION,
        seed=SEED,
    )

    print(
        f"小样本训练比例: {TRAIN_CASE_FRACTION} | "
        f"原训练病例数: {len(train_cs_full)} | "
        f"实际训练病例数: {len(train_cs)}"
    )

    print(f"病例划分: train {len(train_cs)} / val {len(val_cs)} / test {len(test_cs)} ")

    train_samples = [s for c in train_cs for s in all_cases[c]]
    val_samples = [s for c in val_cs for s in all_cases[c]]
    test_samples = [s for c in test_cs for s in all_cases[c]]

    print(f"各个数据集切片数: train {len(train_samples)} / val {len(val_samples)} / test {len(test_samples)}")

    # ============================================================
    # 4，数据探索与报告图生成
    # ============================================================
    all_samples = train_samples + val_samples + test_samples

    balance_rows = [
        analyze_slice_balance(all_samples,split_name="all"),
        analyze_slice_balance(train_samples,split_name="train"),
        analyze_slice_balance(val_samples,split_name="val"),
        analyze_slice_balance(test_samples,split_name="test"),
    ]

    save_slice_balance_analysis(balance_rows,OUTPUT_DIR)

    # 随机展示 12 个阳性病例的 MRI + mask overlay，体现肿瘤多样性
    save_case_diversity_demo(all_samples,OUTPUT_DIR,n_cases=12,seed=SEED)

    # ============================================================
    # 5，类别权重估计与划分文件保存
    # ============================================================
    raw_train_pos_weight = estimate_pos_weight(train_samples)
    # 根据开关决定实际传给loss的 pos_weight。关闭时使用1.0，相当于不额外加权。
    if USE_BCE_POS_WEIGHT:
        train_pos_weight = raw_train_pos_weight
    else:
        train_pos_weight = 1.0
    print(f"训练集估计 pos_weight: {raw_train_pos_weight:.3f}")
    print(f"当前实际使用 pos_weight: {train_pos_weight:.3f}")

    env_info["estimated_train_pos_weight"] = train_pos_weight
    with open(os.path.join(OUTPUT_DIR,"env_info.json"),"w",encoding="utf-8") as f:
        json.dump(env_info,f,ensure_ascii=False,indent=2)

    # 保存划分细节，用于报告和复现
    _save_case_split(
        case_stats=case_stats,
        all_cases=all_cases,
        train_cases=train_cs,
        val_cases=val_cs,
        test_cases=test_cs,
        output_dir=OUTPUT_DIR,
    )

    # ============================================================
    # 6，跑多组实验
    # ============================================================
    def make_loaders():
        """
        构造训练、验证和测试 DataLoader。
        注：
        1.train_loader 使用 augment=True，并结合 WeightedRandomSampler；
        2.train_eval_loader 使用同一批训练样本，但 augment=False，用于稳定评估训练集指标；
        3.val/test 不做随机增强，并且 shuffle=False，方便后续预测结果和样本路径对齐。
        """
        tr_ds = LGGDataset(train_samples, augment=True, img_size=IMG_SIZE)
        tr_eval_ds = LGGDataset(train_samples, augment=False, img_size=IMG_SIZE)
        va_ds = LGGDataset(val_samples, augment=False, img_size=IMG_SIZE)
        te_ds = LGGDataset(test_samples, augment=False, img_size=IMG_SIZE)

        pin = device.type == "cuda"
        g = torch.Generator()
        g.manual_seed(SEED)

        # ========================================================
        # 根据开关选择训练集采样方式
        # ========================================================
        if USE_WEIGHTED_SAMPLER:
            # 启用加权采样：含肿瘤切片更容易被抽到。
            train_weights = _build_train_weights(train_samples)

            sampler = WeightedRandomSampler(
                weights=torch.DoubleTensor(train_weights),
                num_samples=len(train_weights),
                replacement=True,
                generator=g,
            )

            tr_ld = DataLoader(
                tr_ds,
                batch_size=BATCH_SIZE,
                sampler=sampler,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=pin,
            )

        else:
            tr_ld = DataLoader(
                tr_ds,
                batch_size=BATCH_SIZE,
                shuffle=True,
                generator=g,
                num_workers=NUM_WORKERS,
                pin_memory=pin,
            )

        tr_eval_ld = DataLoader(
            tr_eval_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=pin,
        )

        va_ld = DataLoader(
            va_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=pin,
        )

        te_ld = DataLoader(
            te_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=pin,
        )

        return tr_ds, tr_eval_ds, va_ds, te_ds, tr_ld, tr_eval_ld, va_ld, te_ld

    def run_one(exp_name):
        """
        运行单个实验，例如 unet_bce / unet_dice / unet_bce_dice。
        主要流程：
        1.重新固定随机种子；
        2.构造 DataLoader；
        3.初始化 U-Net / Attention U-Net；
        4.根据实验名构造对应 loss；
        5.训练模型；
        6.在验证集上搜索 threshold 和后处理 min_size；
        7.在测试集上做最终评估。
        """
        set_seed(SEED)
        tr_ds,tr_eval_ds,va_ds,te_ds,tr_ld,tr_eval_ld,va_ld,te_ld = make_loaders()

        # 每组实验都重新初始化模型，保证 loss 对比公平。
        model = AttentionUNet(in_ch=1,out_ch=1,base=BASE_CH,use_attention=USE_ATTENTION).to(device)
        n_params = count_params(model)

        # get_loss 根据 exp_name 返回 BCE / Dice / BCE-Dice。
        criterion = get_loss(exp_name,train_pos_weight,device)

        if USE_LOSS_SPECIFIC_LR:
            init_lr = LR[exp_name]
        else:
            init_lr = COMMON_LR

        model_name = "Attention U-Net" if USE_ATTENTION else "Basic U-Net"
        print(f"\n{'=' * 70}")
        print(f"实验: {exp_name} | 模型: {model_name}| 参数: {n_params:,} | loss: {criterion.__class__.__name__} | 当前初始学习率: {init_lr:.6f}")
        print(f"{'=' * 70}")

        res = train_model(
            model=model,
            train_loader=tr_ld,
            train_eval_loader=tr_eval_ld,
            val_loader=va_ld,
            criterion=criterion,
            num_epochs=EPOCHS,
            lr=init_lr,
            weight_decay=WEIGHT_DECAY,
            device=device,
            exp_name=exp_name,
            use_amp=USE_AMP,
        )

        # ============================================================
        # 根据开关决定是否搜索threshold
        # ============================================================
        # 只在验证集上搜索阈值
        if USE_THRESHOLD_SEARCH:
            best_threshold,best_val_metrics = tune_threshold(
                res["model"],
                va_ld,
                criterion,
                device,
                thresholds=THRESHOLD_VALUES,
            )
        else:
            best_threshold = FIXED_THRESHOLD
            best_val_metrics = eval_loader(
                res["model"],
                va_ld,
                criterion,
                device,
                threshold=best_threshold,
            )

        # 打印完整训练情况
        print(
            f"[{exp_name}] best_threshold={best_threshold:.2f} | "
            f"val_dice_pos={best_val_metrics['dice_pos']:.4f} "
            f"val_iou_pos={best_val_metrics['iou_pos']:.4f}"
        )

        # ============================================================
        # 根据开关决定是否启用连通域后处理
        # ============================================================

        # 固定 threshold 后，再在验证集上选择连通域后处理的最小面积。
        if USE_POST_PROCESS:
            best_min_size,best_val_metrics_pp = tune_postprocess_min_size(
                res["model"],
                va_ld,
                criterion,
                device,
                threshold=best_threshold,
                min_sizes=POST_PROCESS_MIN_SIZES,
            )
        else:
            # 关闭后处理时，min_size=0，表示不删除小连通域。
            best_min_size = 0
            best_val_metrics_pp = eval_loader(
                res["model"],
                va_ld,
                criterion,
                device,
                threshold=best_threshold,
                post_process_min_size=0,
                post_process_fill_holes=False,
            )

        # 打印最佳最小面积，并输出此时模型在 val 上的情况
        print(
            f"[{exp_name}] best_min_size={best_min_size} | "
            f"val_dice_pos_pp={best_val_metrics_pp['dice_pos']:.4f} "
            f"val_iou_pos_pp={best_val_metrics_pp['iou_pos']:.4f}"
        )

        # 测试集只在所有验证集参数确定后评估一次。
        test_metrics = eval_loader(
            res["model"],
            te_ld,
            criterion,
            device,
            threshold=best_threshold,
            post_process_min_size=best_min_size,
            post_process_fill_holes=POST_PROCESS_FILL_HOLES,
        )

        pred_df = predict_and_collect(
            res["model"],
            te_ld,
            test_samples,
            device,
            threshold=best_threshold,
            post_process_min_size=best_min_size,
        )

        summary = {
            "exp_name": exp_name,
            "loss": criterion.__class__.__name__,
            "n_params": n_params,
            "best_epoch": res["best_epoch"],
            "best_val_dice_pos": res["best_val_dice_pos"],
            "best_threshold": best_threshold,
            "best_min_size": best_min_size,
            "best_val_dice_pos_tuned": best_val_metrics["dice_pos"],
            "best_val_iou_pos_tuned": best_val_metrics["iou_pos"],
            "best_val_dice_pos_pp": best_val_metrics_pp["dice_pos"],
            "best_val_iou_pos_pp": best_val_metrics_pp["iou_pos"],
            "test_loss": test_metrics["loss"],
            "test_dice_all": test_metrics["dice_all"],
            "test_dice_pos": test_metrics["dice_pos"],
            "test_iou_all": test_metrics["iou_all"],
            "test_iou_pos": test_metrics["iou_pos"],
            "test_acc": test_metrics["acc"],
        }

        print(f"\n[{exp_name}] 测试集结果:")
        for k in ("test_dice_all","test_dice_pos","test_iou_all","test_iou_pos","test_acc"):
            print(f"  {k}: {summary[k]:.4f}")

        return {
            "exp_name": exp_name,
            "history": res["history"],
            "metrics": summary,
            "pred_df": pred_df,
            "model": res["model"],
            "test_ds": te_ds,
            "test_samples": test_samples,
        }

    all_results = []
    for exp_name in EXPERIMENTS:
        all_results.append(run_one(exp_name))

    # ============================================================
    # 7，保存结果、绘图和最佳模型
    # ============================================================
    """保存三组实验的曲线、测试指标、逐图预测结果和逐病例结果。"""

    histories = {r["exp_name"]: r["history"] for r in all_results}
    metrics_list = [r["metrics"] for r in all_results]

    # 保存每个epoch的训练曲线数据。
    hist_rows = []
    for r in all_results:
        h = pd.DataFrame(r["history"])
        h.insert(0,"exp_name",r["exp_name"])
        hist_rows.append(h)

    pd.concat(hist_rows,ignore_index=True).to_csv(
        os.path.join(OUTPUT_DIR,"learning_curve.csv"),
        index=False,
        encoding="utf-8",
    )

    # 保存测试集汇总指标。
    pd.DataFrame(metrics_list).to_csv(
        os.path.join(OUTPUT_DIR,"test_metrics.csv"),
        index=False,
        encoding="utf-8",
    )

    for r in all_results:
        exp_name = r["exp_name"]
        r["pred_df"].to_csv(
            os.path.join(OUTPUT_DIR,f"predictions_{exp_name}.csv"),  # 保存逐图预测指标
            index=False,
            encoding="utf-8",
        )

        per_case = (
            r["pred_df"]
            .groupby("case")[["dice","iou","pixel_acc"]]
            .mean()
            .reset_index()
        )
        per_case.to_csv(
            os.path.join(OUTPUT_DIR,f"per_case_{exp_name}.csv"),  # 保存逐病例预测指标
            index=False,
            encoding="utf-8",
        )

    # 输出打印结果图像
    plot_learning_curves(histories, os.path.join(OUTPUT_DIR,"fig_learning_curves.png"))
    plot_iou_curve(histories, os.path.join(OUTPUT_DIR,"iou_curve.png"))
    plot_metrics_compare(metrics_list, os.path.join(OUTPUT_DIR,"fig_metrics_compare.png"))


    #  依据Dice_pos (也可以选择其他指标，依据场景选择合适的就可以) 选择最佳实验，并保存模型和定性结果。
    best = max(all_results,key=lambda r: r["metrics"]["best_val_dice_pos_pp"])
    best_name = best["exp_name"]
    best_threshold = best["metrics"]["best_threshold"]
    best_min_size = best["metrics"]["best_min_size"]

    torch.save(
        best["model"].state_dict(),
        os.path.join(OUTPUT_DIR,f"best_model_{best_name}.pth"),
    )

    # 只挑含肿瘤的测试切片做可视化，更能观察模型对病灶的分割效果。
    pos_indices = [
        i for i, (_, mask_path) in enumerate(test_samples)
        if np.array(Image.open(mask_path).convert("L")).sum() > 0
    ]

    sample_indices = pos_indices[:10]

    plot_pred_examples(
        best["model"],
        best["test_ds"],
        sample_indices,
        os.path.join(OUTPUT_DIR,f"fig_pred_samples_{best_name}.png"),
        device,
        threshold=best_threshold,
        post_process_min_size=best_min_size,
        title=f"Predictions ({best_name},threshold={best_threshold:.2f},min_size={best_min_size})",
    )

    # 将三联图中选取的图存入文件夹中，以便确认
    save_qualitative_results(
        best["model"],
        best["test_ds"],
        sample_indices,
        os.path.join(OUTPUT_DIR,"qualitative_results"),
        device,
        threshold=best_threshold,
        post_process_min_size=best_min_size,
        max_n=10,
    )

    print("\n=== 全部实验完成 ===")
    print(f"最佳实验: {best_name}")
    print(f"结果已保存到: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()