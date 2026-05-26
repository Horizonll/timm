import time
import torch
from timm import create_model

torch.cuda.set_device(1)
model = create_model("dual_tiny_patch16_224")
model.eval()
model.cuda()

batch_size = 128
image_size = [224, 1024]
image_size = image_size[0]
x = torch.randn(batch_size, 3, image_size, image_size).cuda()
with torch.no_grad():
    for _ in range(10):
        _ = model(x)
total_time = 0
num_runs = 2
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model(x)
        torch.cuda.synchronize()
        end = time.perf_counter()
        total_time += end - start
throughput = (num_runs * batch_size) / total_time
peak_memory = torch.cuda.max_memory_allocated()
peak_per_sample = peak_memory / batch_size
print(f"吞吐量: {throughput:.2f} 样本/秒")
print(f"每样本峰值显存: {peak_per_sample / 1024**2:.2f} MB")
