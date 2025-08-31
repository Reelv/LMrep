## 前提条件

1. Python 3.10.14
2. Windows 操作系统
3. 已安装 **conda** Anaconda 或 Miniconda

## 搭建环境步骤

### 步骤 1: 创建虚拟环境

1. 打开终端（命令提示符或 Anaconda Prompt）。

2. 创建虚拟环境并指定 Python 版本：

   ```bash
   conda create --name lmvrep python=3.10.14
   ```

### 步骤 2: 激活虚拟环境

激活刚刚创建的虚拟环境：

* 运行以下命令来激活 `lmvrep` 环境：

  ```bash
  conda activate lmvrep
  ```

### 步骤 3: 安装所需的依赖包

虚拟环境激活后，使用以下命令安装 `requirements.txt` 文件中列出的依赖包：

1. 确保 `requirements.txt` 文件位于项目根目录中。
2. 运行以下命令来安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

### 步骤 4: 设置仿真环境

1. 仿真环境为CoppeliaSim_Edu_V4_1_0。
2. windows环境下可以直接下载压缩包解压后使用
### 步骤 5: 运行仿真

要启动仿真并控制无人机，按照以下步骤操作：

1. 打开终端（确保虚拟环境已经激活）。
2. 打开仿真软件并进入预设场景
2. 进入包含 `LLM_ctrl.py` 文件的目录。
3. 运行以下命令启动控制脚本：

   ```bash
   python LLM_ctrl.py
   ```

该命令会启动仿真，您可以与仿真进行交互。

### 项目目录结构概览

以下是项目目录结构：

```
项目根目录
│
├── data/
├── images/
├── sim/
├── dronecontroller.py
├── dronellm.py
├── LLM_ctrl.py
├── modelscope.json
├── requirements.txt
└── simConst.py
```

* **data/**: 存放数据文件。
* **images/**: 存放仿真图像文件。
* **sim/**: 存放仿真相关文件（包括无人机场景）。
* **dronellm.py**: 包含 AI 设置和无人机控制代码。
* **dronecontroller.py**: 包含无人机控制代码。
* **LLM\_ctrl.py**: 与仿真交互的主要控制脚本。
* **modelscope.json**: `DroneAI` 类使用的配置文件。
* **requirements.txt**: 所需的 Python 包列表。
* **simConst.py**: 仿真常量设置文件。
