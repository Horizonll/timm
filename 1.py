import torch
from fvcore.nn import FlopCountAnalysis
from timm import create_model

model = create_model("deit_tiny_patch16_224")
model.eval()
inputs = torch.randn(1, 3, 224, 224)
flops = FlopCountAnalysis(model, inputs)
print(f"{flops.total() / 1e9:.3f} GFLOPs")
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M")
