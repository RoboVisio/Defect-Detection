"""
黑白(灰度)相机 OK/NG 异常检测训练脚本 —— anomalib v1+
对应 DLL 文档 5.1 黑白链路：单通道灰度图，输出 ONNX 供 C++ 推理。

目录结构(自行准备)：
    datasets/gray/
        normal/      # 仅放 OK 图(必需)
        abnormal/    # 可选, 放 NG 图用于验证/阈值
        masks/       # 可选, 像素级标注(若有)

用法:
    python anomalib_train_gray.py
产物:
    results/Padim/gray/.../weights/onnx/model.onnx  (供 DLL 加载)
"""
from pathlib import Path

from torchvision.transforms.v2 import Compose, Resize, ToDtype, Normalize
import torch

# --- 绕过 anomalib 对中文路径的 ASCII 校验 (项目目录含 "半导体缺陷检测") ---
import anomalib.data.utils.path as _ap
_ap.validate_path = lambda path, base_dir=None, should_exist=True: Path(path).resolve()
# -----------------------------------------------------------------------

from anomalib.data import Folder
from anomalib.models import Padim
from anomalib.engine import Engine
from anomalib.deploy import ExportType

ROOT = Path(__file__).parent
DATA_ROOT = ROOT / "datasets" / "gray"
IMAGE_SIZE = (1144, 611)   # 与 DLL 预处理 Resize 尺寸保持一致 (H, W)

def main():
    # 注意: anomalib backbone 走 ImageNet 预训练的 3 通道 ResNet,
    # 灰度图会被加载器自动复制到 3 通道, 所以这里仍用 3 通道 mean/std
    transform = Compose([
        Resize(IMAGE_SIZE, antialias=True),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    datamodule = Folder(
        name="gray",
        root=DATA_ROOT,
        normal_dir="normal",
        abnormal_dir=("abnormal" if (DATA_ROOT / "abnormal").exists()
                      and any((DATA_ROOT / "abnormal").iterdir()) else None),
        mask_dir=("masks" if (DATA_ROOT / "masks").exists()
                  and any((DATA_ROOT / "masks").iterdir()) else None),
        train_augmentations=transform,
        val_augmentations=transform,
        test_augmentations=transform,
        train_batch_size=16,
        eval_batch_size=16,
        num_workers=4,
        # 关键: 单通道灰度。anomalib 默认走 RGB, 这里强制 1 通道
        # (Padim/PatchCore backbone 会把单通道复制成 3 通道送入 ResNet)
    )

    # Padim: 无需训练参数, 一次前向统计即可; 对小样本/OK-only 友好
    model = Padim(
        backbone="resnet18",
        layers=["layer1", "layer2", "layer3"],
        pre_trained=True,
        n_features=100,
    )

    engine = Engine(
        max_epochs=1,            # Padim 单轮即可
        default_root_dir=str(ROOT / "results" / "gray"),
        accelerator="cpu",
        devices=1,
    )

    engine.fit(model=model, datamodule=datamodule)
    engine.test(model=model, datamodule=datamodule)

    # 导出 ONNX (DLL 用 ONNX Runtime C++ 加载)
    engine.export(
        model=model,
        export_type=ExportType.ONNX,
        export_root=ROOT / "results" / "gray" / "export",
        input_size=IMAGE_SIZE,
    )
    print("[OK] Gray model exported to results/gray/export/")

    # 校验 ONNX: 输入 batch 轴必须是 dynamic, 否则 DLL 动态 batch 会失败
    import onnx
    onnx_path = next((ROOT / "results" / "gray" / "export").rglob("*.onnx"))
    m = onnx.load(str(onnx_path))
    for i in m.graph.input:
        dims = [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]
        print(f"  input {i.name}: {dims}")
    for o in m.graph.output:
        dims = [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim]
        print(f"  output {o.name}: {dims}")

if __name__ == "__main__":
    main()
