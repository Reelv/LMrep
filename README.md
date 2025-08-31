好嘞，我来帮你把 `modelscope.json` 这一段也加到你的环境搭建文档里，让用户知道怎么配置。可以在“项目目录结构概览”之后单独加一节说明配置文件内容：

---

## 前提条件

1. Python 3.10.14
2. Windows 操作系统
3. 已安装 **conda**（Anaconda 或 Miniconda）

---

## 搭建环境步骤

### 步骤 1: 创建虚拟环境

```bash
conda create --name lmvrep python=3.10.14
```

---

### 步骤 2: 激活虚拟环境

```bash
conda activate lmvrep
```

---

### 步骤 3: 安装依赖

1. 确认 `requirements.txt` 文件在项目根目录下。
2. 安装依赖：

```bash
pip install -r requirements.txt
```

---

### 步骤 4: 设置仿真环境

1. 使用 CoppeliaSim\_Edu\_V4\_1\_0。
2. Windows 下直接下载压缩包解压使用即可。

---

### 步骤 5: 配置 API Key

在项目根目录下的 `modelscope.json` 中填入魔塔社区 API Key 和 OpenAI 配置，文件内容示例：

```json
{
    "GIT_KEY": "xxxxxxxxxxxxxxxxxxx",
    "OPENAI_API_KEY": "xxxxxxxxxxxxxxxxxxxxxxxxxxxx",
    "OPENAI_BASE_URL": "https://api-inference.modelscope.cn/v1/"
}
```

* `GIT_KEY`： Git 仓库 Key（可选,不影响使用）
* `OPENAI_API_KEY`：从魔塔社区获取的 API Key
* `OPENAI_BASE_URL`：魔塔 API 接口地址

---

### 步骤 6: 运行仿真

1. 打开终端并激活虚拟环境：

   ```bash
   conda activate lmvrep
   ```

2. 打开 CoppeliaSim 并加载预设场景。

3. 进入包含 `LLM_ctrl.py` 的目录。

4. 运行控制脚本：

```bash
python LLM_ctrl.py
```

---

### 项目目录结构

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

* **data/**: 存放数据文件
* **images/**: 存放仿真图像文件
* **sim/**: 仿真相关文件（无人机场景等）
* **dronellm.py**: AI 设置和无人机控制逻辑
* **dronecontroller.py**: 无人机控制模块
* **LLM\_ctrl.py**: 主控制脚本
* **modelscope.json**: 存放 API Key 和服务配置
* **requirements.txt**: Python 依赖列表
* **simConst.py**: 仿真常量配置
