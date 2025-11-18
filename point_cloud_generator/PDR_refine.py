import os
import torch
import h5py
import numpy as np
import sys

# ================== 自动定位到 Gaussian_human 根目录 ==================
current_file = os.path.abspath(__file__)                       # 当前脚本路径
project_root = os.path.dirname(os.path.dirname(current_file)) # Gaussian_human 目录
os.chdir(project_root)                                         # 切换工作目录
sys.path.insert(0, project_root)                               

WEIGHT_PATH = "point_cloud_generator/pointnet_ckpt_120899_best_cd.pkl"
INPUT_H5    = "data/input/.h5/partial.h5"
OUTPUT_PLY  = "data/pointcloud/refined.ply"


from point_cloud_generator.pointnet2.models.pointnet2_ssg_sem import PointNet2SemSegSSG


print(f"正在读取输入 ← {INPUT_H5}")
if not os.path.exists(INPUT_H5):
    raise FileNotFoundError(f"输入文件不存在：{INPUT_H5}\n请放入一个包含 partial_cloud 的 h5 文件")

with h5py.File(INPUT_H5, 'r') as f:
    if 'partial_cloud' not in f:
        raise KeyError("h5 文件中必须包含 'partial_cloud' 数据集")
    partial = f['partial_cloud'][:].astype(np.float32)

if partial.shape != (3072, 3):
    raise ValueError(f"输入点云必须是 3072×3，当前是 {partial.shape}")

# ================== 加载模型 ==================
print("正在加载 Refinement 模型...")
if not os.path.exists(WEIGHT_PATH):
    raise FileNotFoundError(f"权重文件不存在：{WEIGHT_PATH}")

model = PointNet2SemSegSSG(input_channels=0, num_classes=0, use_xyz=True)
ckpt = torch.load(WEIGHT_PATH, map_location='cpu')
model.load_state_dict(ckpt['model_state'] if 'model_state' in ckpt else ckpt)
model.eval()
print("模型加载完成")

print("开始精炼点云...")
with torch.no_grad():
    x = torch.from_numpy(partial).unsqueeze(0).transpose(1, 2)   # (1, 3, 3072)
    refined = model(x).transpose(1, 2).squeeze(0)                # (2048, 3)
    refined = refined.cpu().numpy()

# ================== 保存 ==================
os.makedirs(os.path.dirname(OUTPUT_PLY), exist_ok=True)

def save_ply(points, path):
    with open(path, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for p in points:
            f.write("%.6f %.6f %.6f\n" % (p[0], p[1], p[2]))

save_ply(refined, OUTPUT_PLY)
print(f"精炼完成！输出保存至 → {os.path.abspath(OUTPUT_PLY)}")
