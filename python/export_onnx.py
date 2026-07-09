import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
import onnx
import onnxruntime as ort

class HandNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = mobilenet_v2(weights=None)
        # 修复1：正确获取输出通道数
        last_channel = backbone.classifier[1].in_features
        # 修复2：根据你的标签范围决定是否保留 Sigmoid
        backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, 42),
            # 如果标签是归一化的0-1，保留 Sigmoid；否则删除
            nn.Sigmoid()
        )
        self.model = backbone

    def forward(self, x):
        return self.model(x)


# 加载模型
device = torch.device("cpu")
model = HandNet()
model.load_state_dict(torch.load("model_a_best.pth", map_location=device))
model.to(device)
model.eval()

# 导出ONNX
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "hand_model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11,
    do_constant_folding=True,
    export_params=True,  # 确保导出参数
)

print("ONNX模型导出成功: hand_model.onnx")

# 验证ONNX模型
onnx_model = onnx.load("hand_model.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX 模型结构验证通过")

# 测试推理
session = ort.InferenceSession("hand_model.onnx")
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: dummy_input.numpy()})
print(f"输出形状: {output[0].shape}, 输出范围: [{output[0].min():.4f}, {output[0].max():.4f}]")