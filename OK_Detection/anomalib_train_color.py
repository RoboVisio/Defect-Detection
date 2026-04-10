"""
彩色相机 OK/NG 异常检测训练脚本 —— anomalib v1+
对应 DLL 文档 5.2 彩色链路：3 通道 RGB 图，输出 ONNX 供 C++ 推理。

目录结构(自行准备)：
    datasets/color/
        normal/      # 仅放 OK 图(必需)
        abnormal/    # 可选, 放 NG 图用于验证/阈值
        masks/       # 可选, 像素级标注

用法:
    python anomalib_train_color.py
产物:
    results/color/export/weights/onnx/model.onnx  (供 DLL 加载)
"""
from pathlib import Path

from torchvision.transforms.v2 import Compose, Resize, ToDtype, Normalize
import torch

# --- 绕过 anomalib 对中文路径的 ASCII 校验 (项目目录含 "半导体缺陷检测") ---
import anomalib.data.utils.path as _ap
_ap.validate_path = lambda path, base_dir=None, should_exist=True: Path(path).resolve()
# -----------------------------------------------------------------------

from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.deploy import ExportType

ROOT = Path(__file__).parent
DATA_ROOT = ROOT / "datasets" / "color"
IMAGE_SIZE = (448, 960)   # 与 DLL 彩色预处理 Resize 尺寸保持一致, NCHW (H, W)

def main():
    transform = Compose([
        Resize(IMAGE_SIZE, antialias=True),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    datamodule = Folder(
        name="color",
        root=DATA_ROOT,
        normal_dir="normal",
        abnormal_dir=("abnormal" if (DATA_ROOT / "abnormal").exists()
                      and any((DATA_ROOT / "abnormal").iterdir()) else None),
        mask_dir=("masks" if (DATA_ROOT / "masks").exists()
                  and any((DATA_ROOT / "masks").iterdir()) else None),
        train_augmentations=transform,
        val_augmentations=transform,
        test_augmentations=transform,
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=4,
    )

    # PatchCore: 彩色纹理类 OK/NG 表现通常优于 Padim, 同样无需反向传播
    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
    )

    engine = Engine(
        max_epochs=1,
        default_root_dir=str(ROOT / "results" / "color"),
        accelerator="cpu",
        devices=1,
    )

    engine.fit(model=model, datamodule=datamodule)
    engine.test(model=model, datamodule=datamodule)

    engine.export(
        model=model,
        export_type=ExportType.ONNX,
        export_root=ROOT / "results" / "color" / "export",
        input_size=IMAGE_SIZE,
    )
    print("[OK] Color model exported to results/color/export/")

if __name__ == "__main__":
    main()
