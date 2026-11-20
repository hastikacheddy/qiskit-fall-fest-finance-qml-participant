# 金融量子机器学习 —— 学员 Notebook

[English](README.md) | 简体中文

QubitHub Qiskit Fall Fest 2026。我们用同一套规则，两次回答同一个真实的金融问题——*银行客户在接到电话营销后，是否会购买定期存款？*：一次用经典的逻辑回归（scikit-learn），一次用 Qiskit 构建的**变分量子分类器（VQC）**。然后进行实事求是的比较：准确率、精确率、召回率、F1、ROC-AUC、混淆矩阵、运行时间，以及两者的差距是否真的大于随机噪声。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hastikacheddy/qiskit-fall-fest-finance-qml-participant/blob/main/student_finance_qml_workshop.ipynb)

> 这是一个教学实验，**不是**量子优势的演示，也**不构成**任何投资建议。所有数字和图表都由 Notebook 实际计算得出，没有任何手工填写的结果。量子模型在普通 CPU 上进行精确模拟。

全部内容可在普通笔记本电脑或免费的 Google Colab（仅 CPU）上运行：**无需 API 密钥、无需付费服务、无需量子硬件**，也不需要任何量子计算基础。

## 工作坊流程

讲师讲解，你在 `student_finance_qml_workshop.ipynb` 中跟随操作，并亲手编写几个简短的函数。

* 所有样板代码都已提供（安装、数据下载、绘图、优化循环）。你需要完成 **8 个简短的 TODO 函数**（T1 到 T8），以及一个可选的进阶任务 S1。
* 每个 TODO 之后都有一个**检查单元格**，会立即告诉你哪里对、哪里错；**提示**可以逐级展开。
* 后面的章节会用到前面完成的 TODO。每个章节开头都有一个**关卡检查**，明确告诉你还缺什么，避免在无关单元格里遇到莫名其妙的报错。
* 如果卡住了，随时提问。讲师可能会发给你一个名为 `instructor_solutions.py` 的文件：把它放在 **Notebook 同一目录下**，输入 `use_reference("T4")`（换成你的 TODO 编号）即可加载讲师版本的函数。该文件有意不放在本仓库中。

## Google Colab（无需安装）

1. 点击上方的 **Open in Colab** 徽章。Colab 会提示该 Notebook 并非由 Google 编写——这是所有来自 GitHub 的 Notebook 都会出现的标准提示，选择 *仍然运行（Run anyway）* 即可。
2. 运行设置单元格。如果缺少 Qiskit，它会自动安装（约一分钟）。若 Colab 要求重启，请选择 *运行时 → 重启会话*，然后从头重新运行。
3. 如果讲师给了你 `instructor_solutions.py`，请通过左侧 *文件* 面板（文件夹图标）上传，使其与 Notebook 位于同一目录。

Colab 不会把你的修改保存到 GitHub，运行时被回收后所有内容都会丢失：离开前请使用 *文件 → 在云端硬盘中保存副本*（或 *下载 .ipynb*）。

## 本地 Jupyter

需要 Python 3.11 或更高版本。

```bash
python -m venv .venv
# Windows (PowerShell):   .venv\Scripts\Activate.ps1
# macOS / Linux:          source .venv/bin/activate
pip install -r requirements.txt
jupyter lab student_finance_qml_workshop.ipynb
```

Qiskit 2.x 需要 **NumPy 2 或更高版本**。`requirements.txt` 锁定了测试时使用的确切版本（Qiskit 2.6.0、NumPy 2.4.6、pandas 3.0.6、scikit-learn 1.9.1 等）。其他版本通常也能运行，但第一个代码单元格会提示版本差异，此时结果的最后几位小数可能略有不同。

如果电脑性能较弱，可在配置单元格中设置 `QUICK_MODE = True`：这是一个规模更小的实验，其结果与完整运行有意不同。如果出现 joblib 或 “insufficient system resources” 报错，请设置 `N_JOBS = 1`。

## 数据

UCI 机器学习数据库 *Bank Marketing* 数据集（`bank-additional-full.csv`，葡萄牙某银行 2008–2010 年的 41,188 条联系记录），许可证 **CC BY 4.0**，[DOI 10.24432/C5K306](https://doi.org/10.24432/C5K306)。Notebook 会将其下载一次（约 1 MB）到 `data/` 目录，校验 SHA-256 后重复使用；本仓库不附带任何数据。引用请注明：Moro、Cortez 和 Rita（2014），*A data-driven approach to predict the success of bank telemarketing*，Decision Support Systems 62, 22-31，[doi:10.1016/j.dss.2014.03.001](https://doi.org/10.1016/j.dss.2014.03.001)。

通话时长字段已被删除（它只有在通话结束后才知道，保留会泄露答案，原作者也对此作出了提醒），另外删除了 12 条完全重复的记录。

## 方法简介

* 使用固定随机种子进行一次 75/25 分层划分。测试集严格封存：预处理、PCA 和决策阈值都只在训练数据上拟合。
* 三个模型，外加一个“什么都不做”的基准：使用全部样本和特征的逻辑回归；一个**对等**的逻辑回归，它看到的样本和特征与量子线路完全相同；以及**变分量子分类器**（4 个量子比特、角度编码、24 个可训练角度，使用 COBYLA 训练，在无噪声条件下精确模拟）。
* 不确定性评估：使用 bootstrap 置信区间，并用配对 bootstrap 比较模型之间的差异。

## 验证情况

* 在加载讲师参考答案的情况下，Notebook 已在由 `requirements.txt` 构建的全新 Windows 环境（Python 3.11）中从头到尾完整运行：0 个错误，多次重复运行结果完全一致。
* 2026-11-12 也在真实的 Google Colab CPU 运行时（Python 3.13 及 Colab 自带的库版本，Qiskit 由设置单元格安装）上从头到尾运行：0 个错误，安装后约 75 秒完成计算。
* 在 Colab 上，两个逻辑回归的结果与 Windows 完全一致，但量子模型的数字略有不同：测试集 ROC-AUC 为 0.776 而非 0.770，相应地，量子模型与对等逻辑回归之间的配对比较结论也发生了变化。库版本并不能解释这一点（在 Windows 上安装 Colab 的库版本后仍能精确复现原结果）；最可能的原因是不同机器的浮点运算存在细微差异，而无导数优化器把这种差异放大成了不同的训练结果。你自己的结果也可能再有不同：只要两个逻辑回归的结果与讲师一致、只有量子模型那一行不同，就说明一切正常。
* 尚未测试：macOS、Colab 之外的 Linux、Colab 之外的 Python 3.12 及更高版本，以及真实课堂环境。

## 仓库内容

| 文件 | 说明 |
|---|---|
| `student_finance_qml_workshop.ipynb` | 含 TODO 的工作坊 Notebook |
| `requirements.txt` | Notebook 测试时使用的确切库版本 |
