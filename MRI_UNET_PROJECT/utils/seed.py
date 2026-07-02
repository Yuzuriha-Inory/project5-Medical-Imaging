# ============================================================
# 随机种子与实验环境记录
# ------------------------------------------------------------
# 注：
#   1.这一板块只放实现“可复现性”和“环境记录”的工具函数，不直接启动函数也不读取数据。
#   2.main.py负责调用set_seed()、get_device()和save_env_info()。
# ============================================================


import os
import sys
import json
import random
import platform
from pathlib import Path

import numpy as np
import torch

def set_seed(seed):
    """
    固定实验中常见的随机来源。

    保证同一份代码、同一份数据、同一个 seed 下，
    病例划分、模型初始化、训练采样顺序和随机增强尽量保持一致。

    参数:
        seed: int
            随机种子。使用普通非负整数，例如 42、2026、814等。
    """

    # 统一转为 int，避免从配置文件或命令行读取时出现类型不一致的问题。
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 若使用GPU，需要固定CUDA上的随机数。
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic=True

    # 不让cuDNN根据输入尺寸自动寻找最快算法，否则不同运行之间可能选择不同算法，影响复现性。
    torch.backends.cudnn.benchmark=False

    os.environ["PYTHONHASHSEED"]=str(seed)

def get_device():
    """
    自动选择训练设备。
    返回:
        torch.device
            如果当前环境可以使用 CUDA，则返回 cuda；
            否则返回 cpu。
    """

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_env_info(output_dir,config_info):
    """
    保存实验环境信息和关键配置参数。生成文件:env_info.json
    参数:
        output_dir: str 或 Path
            实验结果输出目录。
        config_info: dict
            main.py 中整理好的关键实验配置，例如 seed、batch_size、img_size 等。
    返回:
        env_info: dict
            已经写入 json 的环境与配置字典，方便 main.py 后续继续补充信息。
    """
    # 确保输出目录存在。
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)

    # 记录环境相关信息
    env_info={
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    env_info.update(config_info)

    env_path = output_dir / "env_info.json"
    with open(env_path,"w",encoding="utf-8") as f:
        json.dump(env_info,f,ensure_ascii=False,indent=2)

    return env_info
