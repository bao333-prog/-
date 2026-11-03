# 微波炉项目模型代码包

当前模型：MobileNetV3-Small，六类单标签分类。不是YOLO检测模型，不支持食材计数。

## 文件
- src/oven_food/：现有项目完整Python包，包含模型构建、训练、数据读取与评测代码。
- pyproject.toml：项目依赖与安装配置。
- models/TEMP_best.pt：8轮试跑权重，学习率0.002，临时验证14/17（82.35%）；不是正式模型A。
- reference/：原始试跑配置与结果，仅供查阅。其中路径属于原电脑，不能直接在其他电脑使用。

## 使用
需要Python 3.10或更新版本，并安装兼容的PyTorch、torchvision、NumPy和Pillow。原试跑使用项目专用Python 3.12及CPU版PyTorch。
解压后在目录内执行：

    python -m pip install -e .

构建六类模型：

    from oven_food.model import build_model
    model = build_model("mobilenet_v3_small", num_classes=6, pretrained=True)

加载包内临时模型（在解压根目录执行）：

    import torch
    from oven_food.model import build_model
    checkpoint = torch.load("models/TEMP_best.pt", map_location="cpu", weights_only=True)
    model = build_model(checkpoint["model_name"], num_classes=len(checkpoint["class_names"]), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(checkpoint["class_names"])

预测时必须使用检查点内的类别顺序，不能自行按文件夹顺序解释输出。
试跑预处理：保留纵横比缩放并白色补边至224×224，透明图合成白底，再按ImageNet均值和标准差归一化。原始数据读取baseline本身采用中心裁剪；本次先生成了补边副本，因此运行时未裁掉主体。

训练入口：python -m oven_food train <你的配置.json>
训练前需准备自己的manifest.csv，并修改配置内的数据与输出路径。此包未包含图片，不是完整数据集。

## 边界
代码为当前项目快照，其中部分工具与早期代理烤箱、玉米整理流程有关，不表示它们是正式训练的必需步骤。
试跑只证明流程能够完成，小样本验证成绩不代表烤箱实测或RK3506部署效果。
正式模型A应从原始预训练权重重新训练，不继续使用此临时模型。
