import torch
import os
from pointnet2.models import pointnet2_with_pcld_condition as Model
from torchvision import transforms

model = Model.get_model(num_points=8192).cuda()
model.load_state_dict(torch.load("point_cloud_generator/weights/refine.pth"))
model.eval()

def infer_one_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transforms.ToTensor()(img).cuda().unsqueeze(0)
    with torch.no_grad():
        pcld = model(img)  # [1, 8192, 3]
    return pcld[0].cpu().numpy()

# 用法：在主流程中遍历 frames/ 调用此函数得到 per-frame pointcloud