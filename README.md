
本仓库上传了项目的核心运行文件，包括：

- main.py：项目主程序入口；
- config.py：实验参数、数据路径、训练设置等配置文件；
- run.bat：Windows 下一键运行脚本；
- environment.yml：环境依赖配置文件；
- models文件夹：模型结构代码文件夹；
- utils文件夹：数据处理、训练、评估和可视化工具代码文件夹。

下载整理后项目结构应为：

MRI_UNET_PROJECT/
├─ main.py
├─ config.py
├─ run.bat
├─ environment.yml
├─ models/
│  └─ attention_unet.py
├─ utils/
│  ├─ data_split.py
│  ├─ dataset.py
│  ├─ losses.py
│  ├─ metrics.py
│  ├─ seed.py
│  ├─ train_eval.py
│  └─ visualize.py