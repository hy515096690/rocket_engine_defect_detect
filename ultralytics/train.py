"""在 PyCharm 里打开本文件，直接点绿色 Run（无需命令行参数）。

说明：
  • 脚本所在目录下的嵌套包 ``ultralytics/`` 会插入 ``sys.path``，与工作区「当前运行目录」无关。
  • Run/Debug Configuration 中可把 Working directory 设为任意常用目录（例如项目根或本目录）。
"""

from __future__ import annotations

import multiprocessing
import os
import sys
from pathlib import Path

# --- 保证加载的是本仓库内的 ultralytics（与 PyCharm 运行目录无关）---
_SCRIPT_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _SCRIPT_DIR / "ultralytics"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from ultralytics import YOLO


def _default_workers() -> int:
    if sys.platform == "win32":
        return 0
    return min(8, os.cpu_count() or 4)


# =============================================================================
# PyCharm 绿色运行：只改下面几项即可（不要用命令行）
# =============================================================================
MODEL_YAML = _PKG_ROOT / "cfg" / "models" / "mambaNeXt-yolo" / "MambaNeXt-YOLO-seg.yaml"
DATA_YAML = "coco8-seg.yaml"
EPOCHS = 50
IMGSZ = 640
BATCH = 8
DEVICE = "0"
WORKERS = _default_workers()
PROJECT_DIR = _SCRIPT_DIR / "runs" / "segment"
RUN_NAME = "mambanext_coco8_seg"
EXPORT_ONNX = True
PRETRAINED = False


def main() -> None:
    if not MODEL_YAML.is_file():
        raise FileNotFoundError(
            f"找不到模型配置：{MODEL_YAML}\n"
            "请确认与本 train.py 同级的目录中存在 ultralytics/cfg/models/mambaNeXt-yolo/MambaNeXt-YOLO-seg.yaml"
        )

    PROJECT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_YAML), task="segment")

    train_results = model.train(
        task="segment",
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        exist_ok=True,
        pretrained=PRETRAINED,
        plots=True,
        verbose=True,
    )

    if train_results is None:
        return

    if EXPORT_ONNX:
        try:
            model.export(format="onnx")
        except Exception as e:
            print(f"[train] ONNX 导出失败（已忽略）：{e}", file=sys.stderr)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
