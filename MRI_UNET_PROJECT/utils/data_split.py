# ============================================================
# 数据集按病例划分,严格无泄漏 + 类别不平衡分析
# 注：
#   1.本代码负责找数据、统计病例、划分病例、估计类别权重。
#   2.所有划分都以“病例”为单位，避免同一病例的相邻切片同时进入训练集和测试集，造成数据泄露影响模型的实际泛化能力。
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from config import POS_WEIGHT_MIN, POS_WEIGHT_MAX

def _read_binary_mask(mask_path):
    """
    读取 mask 并转为布尔数组。
    非零都视为病灶区域。
    """
    return np.array(Image.open(mask_path).convert("L")) > 0

def collect_cases(data_dir):
    """
    按病例收集图像和 mask 路径（只读取、收集路径，不直接读取图像）。

    返回格式：
        {
            "TCGA_xxx": [
                (image_path, mask_path),
                ...
            ],
            ...
        }

    """
    root=Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"DATA_DIR不存在:{root}")

    cases={}

    # 数据集中每个TCGA_xxx文件夹代表一个病例。
    # 排序保证相同seed下，每次运行的病例顺序一致。
    for cd in sorted(root.iterdir()):
        if not cd.is_dir() or not cd.name.startswith("TCGA_"):
            continue

        samples=[]

        for img_path in sorted(cd.glob("*.tif")):
            # 从原图反推对应 mask 路径。
            if "_mask" in img_path.name:
                continue

            mask_path=cd/f"{img_path.stem}_mask.tif"

            # 只有image和mask成对存在时才加入样本。
            if mask_path.exists():
                samples.append((str(img_path), str(mask_path)))

        if samples:
            cases[cd.name]=samples
    return cases


def summarize_cases(cases):
    """
    统计每个病例的切片数、阳性切片数、阳性像素比例，用于病例级分层划分。
    尽量让 train/val/test 的病灶数据占比接近。
    """
    rows=[]

    for case_name, samples in cases.items():
        n_slices=len(samples)
        n_pos_slices=0
        pos_pixels=0
        total_pixels=0

        for _,mask_path in samples:
            mask=_read_binary_mask(mask_path)
            cur_pos=int(mask.sum())
            pos_pixels += cur_pos
            total_pixels += int(mask.size)
            if cur_pos > 0:
                n_pos_slices += 1

        rows.append(
            {
                "case": case_name,
                "n_slices": n_slices,
                "n_pos_slices": n_pos_slices,
                "pos_slice_ratio": n_pos_slices/max(n_slices,1),
                "pos_pixel_ratio": pos_pixels/max(total_pixels,1),
            }
        )
    return pd.DataFrame(rows)

def _safe_stratify_labels(values, n_bins=4):
    """
    求助参考AI实现函数，作用于总体病例的按整体阳性切片比例来划分 train/val/test 三个集，把连续的阳性切片比例转换成分层标签。

    如果样本太少、分桶后某些桶只有 1 个病例，返回 None，后续自动退化为普通病例级随机划分。
    """
    values=pd.Series(values).astype(float)
    if len(values) < 4:
        return None

    if values.nunique() <= 1:
        labels=(values > 0).astype(int)
    else:
        labels=pd.qcut(
            values.rank(method="first"),
            q=min(n_bins, len(values)),
            labels=False,
            duplicates="drop",
        )

    counts=pd.Series(labels).value_counts()

    # 分层划分要求每个类别至少有2个样本，否则无法分层。
    if len(counts) < 2 or counts.min() < 2:
        return None

    return labels

def split_cases(case_names, seed=42, train_ratio=0.7, val_ratio=0.15):
    """保留随机病例划分函数，作为分层划分失败时的回退方案。确保不会把同一病例的切片拆到不同集合中。"""

    #创建固定随机数生成器，以seed来确定后续随机打乱的结果
    rng=np.random.RandomState(seed)
    case_names=list(case_names)

    #生成随机排列
    perm=rng.permutation(len(case_names))

    n=len(case_names)
    n_train=int(round(n * train_ratio))
    n_val=int(round(n * val_ratio))

    #按照上面计算的各个数据集病例数量，在随机排列上进行划分
    train_cs=[case_names[i] for i in perm[:n_train]]
    val_cs=[case_names[i] for i in perm[n_train:n_train + n_val]]
    test_cs=[case_names[i] for i in perm[n_train + n_val:]]

    return train_cs, val_cs, test_cs

def split_cases_stratified(case_stats, seed=42, train_ratio=0.7, val_ratio=0.15):
    """
    按病例划分，同时近似分层，尽量保持 train/val/test 的阳性病例比例一致。
    若分层失败，仍可回到普通病例划分函数，确保不会出现数据泄露问题，保障基本功能
    """
    try:
        from sklearn.model_selection import train_test_split
    except Exception:
        print("警告：未能导入 sklearn，回退到普通病例级随机划分。")
        return split_cases(case_stats["case"].tolist(),seed,train_ratio,val_ratio)

    df=case_stats.copy().reset_index(drop=True)
    stratify_all=_safe_stratify_labels(df["pos_slice_ratio"])

    try:
        # 第一次划分：train和临时集合temp。
        train_df, temp_df=train_test_split(
            df,
            test_size=1 - train_ratio,
            random_state=seed,
            shuffle=True,
            stratify=stratify_all,
        )

        # 第二次划分：把temp再拆成val和test。
        val_size_in_temp=val_ratio / (1 - train_ratio)
        stratify_temp=_safe_stratify_labels(temp_df["pos_slice_ratio"])
        val_df, test_df=train_test_split(
            temp_df,
            test_size=1 - val_size_in_temp,
            random_state=seed,
            shuffle=True,
            stratify=stratify_temp,
        )

    except Exception as e:
        print(f"警告：分层病例划分失败，回退到普通病例级随机划分。原因: {e}")
        return split_cases(df["case"].tolist(),seed,train_ratio,val_ratio)

    return (
        train_df["case"].tolist(),
        val_df["case"].tolist(),
        test_df["case"].tolist(),
    )


def estimate_pos_weight(samples):
    """
    根据训练集 mask 自动估计 BCE Loss 的 pos_weight。

    因为背景像素远多于病灶像素，如果 BCE 不加权，模型容易偏向预测背景。尽管拥有较高的准确率，但实际分割效果会很差
    这里按训练集统计正负像素比例：
        raw_ratio = 背景像素数 / 病灶像素数
    同时使用 sqrt(raw_ratio)进行缩减，再裁剪到指定范围。
    """
    pos_pixels = 0
    total_pixels = 0
    for _,mask_path in samples:
        mask=_read_binary_mask(mask_path)
        pos_pixels += int(mask.sum())
        total_pixels += int(mask.size)

    neg_pixels = total_pixels - pos_pixels
    raw_ratio = neg_pixels / max(pos_pixels, 1)

    # 原始比例通常很大，直接使用会造成大量假阳性，因此取sqrt后再裁剪。
    pos_weight = np.sqrt(raw_ratio)
    pos_weight = float(np.clip(pos_weight,POS_WEIGHT_MIN,POS_WEIGHT_MAX))

    return pos_weight


def subsample_train_cases(train_cs, case_stats, fraction=1.0, seed=42):
    """
    从原训练病例中抽取一部分病例，用于小样本训练实验。

    注：
        1. 只对训练集病例进行抽样；
        2. 验证集和测试集保持不变；
        3. 按病例抽样，而不是按切片抽样；
        4. 尽量保留 “有阳性病例/无阳性病例 ”的比例。

    例如：
        fraction=1.0 表示使用完整训练病例；
        fraction=0.2 表示只使用原训练病例中的 20%。
        """

    if fraction >= 1.0:
        return train_cs

    rng = np.random.RandomState(seed)

    train_df = case_stats[case_stats["case"].isin(train_cs)].copy()
    train_df = train_df.reset_index(drop=True)

    # 尽量保留阳性比例分布：按是否有阳性切片分层
    train_df["has_pos_case"] = (train_df["n_pos_slices"] > 0).astype(int)

    selected = []

    for label,group in train_df.groupby("has_pos_case"):
        cases = group["case"].tolist()
        rng.shuffle(cases)

        n_keep = max(1, int(round(len(cases) * fraction)))
        selected.extend(cases[:n_keep])

    return  sorted(selected)