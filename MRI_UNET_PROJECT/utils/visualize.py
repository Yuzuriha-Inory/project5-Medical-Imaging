# ============================================================
# 可视化与结果输出
# ------------------------------------------------------------
# 注：这一板块负责训练曲线、测试集指标柱状图、病例多样性图、以及分割结果示例图的保存。
# ============================================================

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

from config import POST_PROCESS_FILL_HOLES
from utils.metrics import post_process_mask

# ============================================================
# 1.设置全局绘图样式
# ============================================================
def setup_fonts():
    """
    统一设置绘图字体与保存参数。
    同时兼顾英文标题和中文说明：英文图题尽量用 Times New Roman / DejaVu；中文说明用 SimSun
    """
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["Times New Roman","SimSun","DejaVu Serif"]
    rcParams["font.sans-serif"] = ["SimSun","Times New Roman","DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False
    rcParams["mathtext.fontset"] = "stix"

    # 字号设置，可随时修改
    rcParams["font.size"] = 10.5
    rcParams["axes.titlesize"] = 10.5
    rcParams["axes.labelsize"] = 10.5
    rcParams["xtick.labelsize"] = 10.5
    rcParams["ytick.labelsize"] = 10.5
    rcParams["legend.fontsize"] = 8.5

    # 保存图像时统一使用较高分辨率
    rcParams["savefig.dpi"] = 300
    rcParams["savefig.bbox"] = "tight"
    rcParams["figure.constrained_layout.use"] = True


# ============================================================
# 2.基础显示工具
# ============================================================
def normalize_for_show(arr):
    """把灰度图归一化到 [0, 1]，仅用于显示。"""
    arr = arr.astype(np.float32)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)


def make_mask_overlay(img_arr,mask_bin,color=(1.0,0.0,0.0),alpha=0.45):
    """
    在 MRI 灰度图上叠加 mask。
    默认红色表示肿瘤区域。
    alpha : 叠加强度
    """
    img_show = normalize_for_show(img_arr)

    # 复制 3 通道，做彩色叠加。
    overlay = np.stack([img_show,img_show,img_show],axis=-1)
    color_arr = np.array(color).reshape(1,1,3)

    # 只修改 mask 区域，其他背景保持原始灰度显示。
    overlay[mask_bin] = (1 - alpha) * overlay[mask_bin] + alpha * color_arr

    return np.clip(overlay,0,1)

# ============================================================
# 3.病例多样性展示
# ============================================================
def save_case_diversity_demo(samples,output_dir,n_cases=12,seed=42):
    """
    非必需，优化改进用
    随机展示若干阳性病例的多样性。
    1.先从所有样本中筛出阳性切片；
    2.对每个病例，只保留 “肿瘤像素比例最大” 的那一张切片，用于展示；
    3.再随机抽取 n_cases 个病例，画成网格图。
    """
    setup_fonts()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)

    rows = []
    for img_path, mask_path in samples:
        case_name = Path(img_path).parent.name
        mask = np.array(Image.open(mask_path).convert("L")) > 0
        tumor_pixel_ratio = float(mask.mean())

        # 这里只保留阳性样本。
        if tumor_pixel_ratio > 0:
            rows.append({
                "case": case_name,
                "img_path": img_path,
                "mask_path": mask_path,
                "tumor_pixel_ratio": tumor_pixel_ratio,
            })

    if len(rows) == 0:
        print("警告：没有找到阳性样本，无法生成病例多样性图。")
        return None

    df = pd.DataFrame(rows)

    # 每个病例选肿瘤面积最明显的一张切片，方便展示病灶形态，同时避免同一病例重复出现很多次。
    df_case = (
        df.sort_values("tumor_pixel_ratio",ascending=False)
        .groupby("case",as_index=False)
        .first()
    )

    # 随机选择 n_cases 个病例
    rng = random.Random(seed)
    idx_list = list(df_case.index)

    # 使用固定seed打乱顺序，既有随机性，又能保证多次运行结果可复现。
    rng.shuffle(idx_list)
    idx_list = idx_list[:min(n_cases,len(idx_list))]
    df_show = df_case.loc[idx_list].reset_index(drop=True)

    # 保存本次被选中的病例和切片路径。
    df_show.to_csv(output_dir / "case_diversity_samples.csv",index=False,encoding="utf-8")

    n = len(df_show)
    cols = 4
    rows_n = int(np.ceil(n / cols))

    fig,axes = plt.subplots(rows_n,cols,figsize=(4 * cols, 4 * rows_n))
    axes = np.array(axes).reshape(-1)

    # 逐个读取选中的病例切片，并绘制MRI + mask叠加图。
    for i,(_, row) in enumerate(df_show.iterrows()):
        img = np.array(Image.open(row["img_path"]).convert("L")) #读取 MRI 灰度图
        mask = np.array(Image.open(row["mask_path"]).convert("L")) > 0 #读取对应 Mask，并转为 bool 类型

        # 将mask以半透明红色叠加到 MRI 图像上，更直观体现肿瘤位置、大小和形态。
        overlay = make_mask_overlay(img,mask,color=(1.0, 0.0, 0.0),alpha=0.45)

        axes[i].imshow(overlay) #显示叠加后图像
        axes[i].set_title(
            f"{row['case']}\nTumor pixels={row['tumor_pixel_ratio'] * 100:.2f}%", #显示病例名和肿瘤像素占比
            fontsize=8,
        )
        axes[i].axis("off")

    for j in range(i + 1,len(axes)):
        axes[j].axis("off")

    fig.suptitle("Diversity of Positive MRI Cases",fontsize=12)
    plt.savefig(output_dir / "case_diversity_12_cases.png")
    plt.close(fig)

    print(f"已保存病例多样性图: {output_dir / 'case_diversity_12_cases.png'}")
    return df_show


# ============================================================
# 4.切片类别不平衡分析（统计整体阳性比例、肿瘤像素比例等指标，便于分析不同批数据【如果有】的特征，初步判断用什么loss更稳妥）
# ============================================================
def analyze_slice_balance(samples,split_name="all"):
    """
    统计某个数据集切片中，含肿瘤 / 不含肿瘤的分布情况。

    返回内容包括：总切片数，阳性切片数，阴性切片数，阳性切片比例，阴性切片比例，肿瘤像素在全部像素中的比例
    """
    n_total = len(samples)
    n_tumor = 0
    pos_pixels = 0
    total_pixels = 0

    for _,mask_path in samples:
        mask = np.array(Image.open(mask_path).convert("L"))
        mask_bin = mask > 0

        if mask_bin.any():
            n_tumor += 1

        # 做全局像素统计
        pos_pixels += int(mask_bin.sum())
        total_pixels += int(mask_bin.size)

    n_no_tumor = n_total - n_tumor

    return {
        "split": split_name,
        "total_slices": n_total,
        "tumor_slices": n_tumor,
        "no_tumor_slices": n_no_tumor,
        "tumor_slice_ratio": n_tumor / max(n_total, 1),
        "no_tumor_slice_ratio": n_no_tumor / max(n_total, 1),
        "tumor_pixel_ratio": pos_pixels / max(total_pixels, 1),
    }


def save_slice_balance_analysis(balance_rows,output_dir):
    """
    保存上述切片不平衡分析结果。

    返回内容：
    1.一份 csv保存数据。
    2.一张柱状图展示整体与各个数据集中 tumor / no-tumor 的数量对比。
    """
    setup_fonts()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)

    df = pd.DataFrame(balance_rows)
    df.to_csv(output_dir / "slice_balance_analysis.csv",index=False,encoding="utf-8")

    print("\n=== 是否含肿瘤切片统计 ===")
    for row in balance_rows:
        print(
            f"[{row['split']}] "
            f"总切片: {row['total_slices']} | "
            f"含肿瘤: {row['tumor_slices']} ({row['tumor_slice_ratio'] * 100:.1f}%) | "
            f"无肿瘤: {row['no_tumor_slices']} ({row['no_tumor_slice_ratio'] * 100:.1f}%) | "
            f"肿瘤像素占比: {row['tumor_pixel_ratio'] * 100:.3f}%"
        )

    # 对all / train / val / test各画一个总览柱状图
    splits = df["split"].tolist()
    tumor_vals = df["tumor_slices"].tolist()
    no_tumor_vals = df["no_tumor_slices"].tolist()

    x = np.arange(len(splits))
    width = 0.35

    fig,ax = plt.subplots(figsize=(8,4.8))
    bars1 = ax.bar(x - width / 2,tumor_vals,width,label="Tumor slices")
    bars2 = ax.bar(x + width / 2,no_tumor_vals,width,label="No-tumor slices")

    # 给每个柱子标数值，可视化更直观。
    top_offset = max(df["total_slices"]) * 0.01 if len(df) > 0 else 1
    for bars in [bars1,bars2]:
        for b in bars:
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2,
                h + top_offset,
                f"{int(h)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    # 柱子的高度表示切片数量。
    ax.set_ylabel("Number of slices")
    # 整张柱状图标题
    ax.set_title("Tumor / No-tumor Slice Distribution")
    ax.legend()
    # y轴方向添加浅色虚线网格，更方便观察不同柱子之间的数量差异
    ax.grid(axis="y",alpha=0.3,linestyle=":")
    plt.savefig(output_dir / "slice_balance_analysis.png")
    plt.close(fig)

    return df


# ============================================================
# 5.训练曲线与测试集指标图
# ============================================================
def plot_learning_curves(histories,save_path):
    """
    绘制训练主曲线图。便于清楚直观地分析各个 loos 实验跑的情况。
    依次输出 4张图：
    1.Loss curve
    2.Validation Dice (all)
    3.Validation Dice (positive)
    4.Validation IoU (positive)
    """
    setup_fonts()

    fig,axes = plt.subplots(1,4,figsize=(20,4.2))
    cmap = plt.get_cmap("tab10")

    for i,(name,h) in enumerate(histories.items()):
        c = cmap(i % 10)
        ep = h["epoch"]

        # 图1：loss 曲线，同时画train / val，观察是否过拟合？
        axes[0].plot(ep,h["train_loss"],c=c,ls="-",lw=1.2,label=f"{name}-train")
        axes[0].plot(ep,h["val_loss"],c=c,ls="--",lw=1.2, label=f"{name}-val")

        # 图2：验证集Dice(all)
        axes[1].plot(ep,h["val_dice_all"],c=c,ls="-",lw=1.2,label=f"{name}-all")

        # 图3：验证集Dice(pos)
        axes[2].plot(ep,h["val_dice_pos"],c=c,ls="-",lw=1.2,label=f"{name}-pos")

        # 图4：验证集IoU(pos)
        axes[3].plot(ep,h["val_iou_pos"],c=c,ls="-",lw=1.2,label=f"{name}")

    # 各曲线加上标签，绘图
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss curve")

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].set_title("Validation Dice (all)")

    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Dice")
    axes[2].set_title("Validation Dice (positive)")

    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("IoU")
    axes[3].set_title("Validation IoU (positive)")

    for ax in axes:
        ax.grid(True,alpha=0.3,ls=":")
        ax.legend(loc="best",fontsize=8,ncol=1,framealpha=0.85)

    plt.savefig(save_path)
    plt.close(fig)


def plot_iou_curve(histories,save_path):
    """
    train 与 val 的 IoU 曲线。
    统一画正样本切片（ pos ）上的 IoU。
    """
    setup_fonts()

    fig,ax = plt.subplots(figsize=(8,4.8))

    for name,h in histories.items():
        ep = h["epoch"]
        ax.plot(ep,h["train_iou_pos"],linestyle="-",linewidth=1.2, label=f"{name}-train")
        ax.plot(ep,h["val_iou_pos"],linestyle="--",linewidth=1.2, label=f"{name}-val")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("IoU")
    ax.set_title("Train / Val IoU Curve (positive slices)")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(fontsize=8,ncol=2)
    plt.savefig(save_path)
    plt.close(fig)


def plot_metrics_compare(metrics_list,save_path):
    """
    将不同实验在测试集上的分割指标画成柱状图。
    对比指标：
    1，Dice(all)
    2，Dice(pos)
    3，IoU(all)
    4，IoU(pos)
    因为 acc 受背景影响过大，不适合作为统计实验性能的指标，所以不特意画出，相比较的话可通过训练日志
    """
    setup_fonts()

    # 取每个实验的名称，用作x轴标签。
    names = [m["exp_name"] for m in metrics_list]

    keys = ["test_dice_all","test_dice_pos","test_iou_all","test_iou_pos"]
    labels = ["Dice (all)","Dice (pos)","IoU (all)","IoU (pos)"]

    x = np.arange(len(names))
    width = 0.2

    # 由实验数量调整图像宽度。
    fig,ax = plt.subplots(figsize=(max(7,2 * len(names)),4.2))
    cmap = plt.get_cmap("Set2")

    # 依次绘制Dice(all)、Dice(pos)、IoU(all)、IoU(pos)四组柱子。
    for i,(k,label) in enumerate(zip(keys,labels)):
        vals = [m[k] for m in metrics_list]
        bars = ax.bar(x + (i - 1.5) * width,vals,width,label=label,color=cmap(i))

        # 顶部标出数值，便于读图
        for b,v in zip(bars,vals):
            ax.text(b.get_x() + b.get_width() / 2,v + 0.005,f"{v:.3f}",ha="center",va="bottom",fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names,rotation=15,ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0,1.05)
    ax.set_title("Test set segmentation metrics")
    ax.grid(axis="y",alpha=0.3,ls=":")
    ax.legend(loc="lower right",ncol=2)

    plt.savefig(save_path)
    plt.close(fig)

# ============================================================
# 6.分割结果可视化
# ============================================================
def tensor_to_display_image(img_t):
    """
    将 z-score 后的单通道 MRI tensor 转成 0~1 灰度图保存显示。
    """
    arr = img_t.detach().cpu().numpy().squeeze()
    p1,p99 = np.percentile(arr,[1,99])

    if p99 > p1:
        arr = (arr - p1) / (p99 - p1)
    else:
        # 极端情况下分位点几乎重合，则退回普通归一化。
        arr = arr - arr.min()
        arr = arr / (arr.max() + 1e-6)

    return np.clip(arr,0,1)


def apply_prediction_postprocess_from_logits(logits,threshold=0.5,post_process_min_size=0):
    """
    从模型输出的 logits 得到最终二值预测，并做与评估阶段一致的后处理。
    1.sigmoid -> 概率图
    2.概率图 -> threshold -> 二值图
    3.按需去掉很小的连通域
    """
    pred = (torch.sigmoid(logits) > threshold).float().cpu().squeeze().numpy()
    if post_process_min_size > 0:
        pred = post_process_mask(
            pred,
            min_size=post_process_min_size,
            fill_holes=POST_PROCESS_FILL_HOLES,
        )

    return pred


def plot_pred_examples(model,dataset,indices,save_path,device,threshold=0.5,
                       post_process_min_size=0,title=""):
    """保存一张总览图：image / GT / pred。"""
    setup_fonts()
    model.eval()

    n = len(indices)
    if n == 0:
        print("警告：无样本可可视化")
        return

    # 创建 n行3列 的子图。
    # 每一行对应一个样本，三列分别显示Image / Ground Truth / Prediction。
    fig,axes = plt.subplots(n,3,figsize=(9,3 * n))

    if n == 1:
        axes = axes.reshape(1,-1)

    with torch.no_grad():
        for row,idx in enumerate(indices):
            # 根据样本下标从dataset中取出一张 MRI 图像和对应 mask。
            img,mask = dataset[idx]
            logits = model(img.unsqueeze(0).to(device))

            # 二值化 + 后处理
            pred = apply_prediction_postprocess_from_logits(
                logits,
                threshold=threshold,
                post_process_min_size=post_process_min_size,
            )

            img_disp = tensor_to_display_image(img)
            mask_disp = mask.squeeze().numpy()

            # MRI 原图
            axes[row,0].imshow(img_disp,cmap="gray")
            axes[row,0].set_title("Image" if row == 0 else "")

            # 真实分割 Mask
            axes[row,1].imshow(mask_disp,cmap="gray")
            axes[row,1].set_title("Ground truth" if row == 0 else "")

            # 模型预测 Mask
            axes[row,2].imshow(pred,cmap="gray")
            axes[row,2].set_title("Prediction" if row == 0 else "")

            for c in range(3):
                axes[row,c].axis("off")

    if title:
        fig.suptitle(title,fontsize=11,y=0.99)

    plt.savefig(save_path)
    plt.close(fig)


def save_qualitative_results(model,dataset,indices,out_dir,device,threshold=0.5,
                             post_process_min_size=0,max_n=10):
    """
    保存 qualitative_results/ 下至少 10 张三联图结果。
    每张图包含：
    1，原图 Image
    2，真值 GT
    3，预测 Pred
    """
    setup_fonts()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True,exist_ok=True)
    model.eval()

    selected = indices[:max_n]
    if not selected:
        print("警告：没有可保存的 qualitative 样本。")
        return

    with torch.no_grad():
        for k,idx in enumerate(selected):
            img,mask = dataset[idx]
            logits = model(img.unsqueeze(0).to(device))
            pred = apply_prediction_postprocess_from_logits(
                logits,
                threshold=threshold,
                post_process_min_size=post_process_min_size,
            )

            img_disp = tensor_to_display_image(img)
            mask_disp = mask.squeeze().numpy()

            fig,axes = plt.subplots(1,3,figsize=(9,3))

            axes[0].imshow(img_disp,cmap="gray")
            axes[0].set_title("Image")

            axes[1].imshow(mask_disp,cmap="gray")
            axes[1].set_title("GT")

            axes[2].imshow(pred,cmap="gray")
            axes[2].set_title("Pred")

            for ax in axes:
                ax.axis("off")

            plt.savefig(out_dir / f"case_{k:02d}_triple.png")
            plt.close(fig)
