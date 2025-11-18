import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
import imageio
import trimesh
from scene import Scene
from scene.deformable_gaussian_model import DeformableGaussianModel
from gaussian_renderer import render
from utils.camera_utils import loadCam
from utils.graphics_utils import getWorld2View2
from scene.colmap_loader import readColmapSceneInfo
from smplx import SMPLX

def get_smplx_model():
    path = "data/body_models/smplx/SMPLX_NEUTRAL.npz"
    if not os.path.exists(path):
        raise FileNotFoundError(f"SMPL-X model not found: {path}")
    return SMPLX(model_path=path, batch_size=1).cuda()

parser = argparse.ArgumentParser()
parser.add_argument("--source_path", type=str, required=True)
parser.add_argument("--model_path", type=str, default="output/deformable_human")
parser.add_argument("--images", type=str, default="images")
parser.add_argument("--iteration", type=int, default=30000)
parser.add_argument("--pose_path", type=str, default=None)
args = parser.parse_args()

os.makedirs(args.model_path, exist_ok=True)
img_dir = os.path.join(args.model_path, args.images)
os.makedirs(img_dir, exist_ok=True)

# Step 1: 抽帧
if len(os.listdir(img_dir)) < 10:
    reader = imageio.get_reader(args.source_path)
    for i, im in enumerate(reader):
        imageio.imwrite(os.path.join(img_dir, f"{i:05d}.png"), im)
    print(f"抽帧完成：{i+1} 帧")

# Step 2: COLMAP + 点云
scene_info = readColmapSceneInfo(args.model_path, args.images, eval=False, llffhold=8)

# Step 3: 初始化高斯（使用 COLMAP 点云）
gaussians = DeformableGaussianModel(sh_degree=3)
scene = Scene(args, gaussians, load_iteration=None, shuffle=False)
gaussians.create_from_pcd(scene_info.point_cloud, spatial_lr_scale=scene_info.nerf_normalization["radius"])

# Step 4: 训练
print("开始训练 3D Gaussian...")
bg = torch.tensor([1,1,1]).cuda()
for iteration in tqdm(range(1, args.iteration + 1)):
    gaussians.update_learning_rate(iteration)
    view = np.random.choice(scene.getTrainCameras())
    render_pkg = render(view, gaussians, args, bg)
    image = render_pkg["render"]
    gt = view.original_image.cuda()
    loss = 0.8 * (gt - image).abs().mean() + 0.2 * (1.0 - render_pkg["ssim"])
    loss.backward()
    gaussians.optimizer.step()
    gaussians.optimizer.zero_grad()

    if iteration % 5000 == 0:
        gaussians.save_ply(os.path.join(args.model_path, f"point_cloud/iteration_{iteration}.ply"))

gaussians.save_ply(os.path.join(args.model_path, "point_cloud/final.ply"))
print("训练完成")

# Step 5: 提取 Canonical Mesh
from games.mesh_splatting.scene.gaussian_mesh_model import GaussianMeshModel as GMM
mesh_path = os.path.join(args.model_path, "canonical_mesh.ply")
GMM(gaussians.get_xyz, gaussians.get_scaling, gaussians.get_rotation, 
    gaussians.get_opacity, gaussians.get_covariance).extract_mesh(path=mesh_path, resolution=512)
print("Canonical mesh 已保存")

# Step 6: 拟合 SMPL beta
smpl_model = get_smplx_model()
cano_mesh = trimesh.load(mesh_path)
vertices = torch.from_numpy(cano_mesh.vertices).cuda().float().unsqueeze(0)

betas_raw = torch.zeros(1, 10).cuda().requires_grad_()
optimizer = torch.optim.Adam([betas_raw], lr=0.05)
print("拟合 SMPL beta...")
for i in range(800):
    optimizer.zero_grad()
    betas = torch.tanh(betas_raw)
    body = smpl_model(betas=betas)
    loss = ((body.vertices - vertices) ** 2).mean()
    loss.backward()
    optimizer.step()
final_beta = torch.tanh(betas_raw).detach().cpu().numpy()
np.savez(os.path.join(args.model_path, "smpl_beta.npz"), beta=final_beta)
print("SMPL beta 已保存")

# Step 7: 绑定高斯
gaussians.bind_to_smpl_mesh(cano_mesh.vertices)
print("高斯已绑定到 SMPL 顶点")

# Step 8: 可控渲染
render_path = os.path.join(args.model_path, "render")
os.makedirs(render_path, exist_ok=True)

pose_files = [None] * 5
if args.pose_path and os.path.exists(args.pose_path):
    pose_files = sorted([f for f in os.listdir(args.pose_path) if f.endswith(".npy")])

for idx, pose_file in enumerate(pose_files):
    if pose_file:
        body_pose = np.load(os.path.join(args.pose_path, pose_file))
        body_pose = torch.from_numpy(body_pose).cuda().unsqueeze(0)  # (1,69)
    else:
        body_pose = torch.zeros(1, 69).cuda()
        body_pose[0, 3:] = torch.randn(66).cuda() * 0.6

    with torch.no_grad():
        new_body = smpl_model(betas=torch.from_numpy(final_beta).cuda(), body_pose=body_pose)
    new_vertices = new_body.vertices[0].cpu().numpy()

    gaussians.deform_with_new_mesh(new_vertices)

    for cam_idx, cam in enumerate(scene.getTrainCameras()):
        rendering = render(cam, gaussians, args, bg)["render"]
        rendering = (rendering.cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(render_path, f"pose{idx:03d}_view{cam_idx:03d}.png"), rendering)

print(f"渲染完成！结果在 {render_path}")