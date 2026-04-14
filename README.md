# SSCBM-CGens
半监督概念瓶颈模型与概念生成（Semi-Supervised Concept Bottleneck Model with Concept Generation, SSCBM-CGens）

---

## 📦 数据集准备
### 1. CUB-200-2011 数据集
**请单独下载并按以下步骤准备数据集**：
- 从 Caltech 官方网站下载数据集压缩包 `CUB_200_2011.tgz`：  
  https://www.vision.caltech.edu/datasets/cub_200_2011/
- 将压缩包解压到项目根目录下的 `data/` 文件夹中：
  ```bash
  tar -xzf CUB_200_2011.tgz -C ./data/
  ```
- 从 Codalab 平台下载概念标注文件 `class_attr_data_10`：  
  https://worksheets.codalab.org/bundles/0x5b9d528d2101418b87212db92fea6683  
  （该标注文件源自原始概念瓶颈模型（vanilla CBM）的实现）
- 将下载好的 `class_attr_data_10` 文件放入 `CUB_200_2011/` 目录下。

### 2. CEM 包
**CEM（Concept Embedding Module，概念嵌入模块）包需要单独下载**，并手动安装：
- 运行代码前，请确保 CEM 包已成功安装到你的 Python 环境中。

---

## 📝 注意事项
- **请勿将数据集或大文件提交到仓库**（已通过 `.gitignore` 自动过滤）
- CEM 包为独立依赖项，运行模型前请务必完成安装
- 若遇到数据集访问或 CEM 安装问题，请参考对应项目的官方文档

---

## 📄 开源协议
本项目基于 [MIT 协议](sslocal://flow/file_open?url=LICENSE&flow_extra=eyJsaW5rX3R5cGUiOiJjb2RlX2ludGVycHJldGVyIn0=) 开源（如需使用，请补充 LICENSE 文件）
