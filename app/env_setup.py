"""在导入 ultralytics 之前，把它的配置目录安排到项目内部。

ultralytics 默认把设置写到用户目录（Windows 是 `%APPDATA%\\Ultralytics`）。
在权限受限的环境里，这一步会直接抛 `PermissionError: [WinError 5]`，
而且报错发生在 import 阶段，看起来像是"库装坏了"，其实只是没地方写文件。

这里统一改成项目内的 `.ultralytics/`，既能让它在受限环境里跑起来，
也避免在用户目录里留一堆看不见的缓存。
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / ".ultralytics"
MPL_DIR = PROJECT_ROOT / ".matplotlib"


def prepare() -> Path:
    """把 ultralytics 和 matplotlib 的缓存目录都安排到项目内部。

    两个库默认都往用户目录写缓存，在没有写权限的环境里会报 PermissionError。
    统一放到项目内的隐藏目录，既不污染用户目录，也保证受限环境下能跑。
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    MPL_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(CONFIG_DIR))
    os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
    return CONFIG_DIR
