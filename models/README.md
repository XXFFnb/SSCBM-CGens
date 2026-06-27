# models

存放模型定义和 D-CGFS 模块实现。

主要文件：

```text
sscbm.py          SSCBM 模型
dcgfs_modules.py D-CGFS 概念定位和门控融合模块
cgens_modules.py 历史 C-Gens/C-GFS 相关模块
cbm.py/cem.py    其他概念瓶颈模型实现
```

当前主方法主要依赖 `sscbm.py` 和 `dcgfs_modules.py`。

