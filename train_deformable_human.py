# train_deformable_human.py
# Full pipeline: Video → PointCloud → 3DGS Training → Mesh → SMPL → Pose Control → Deformable Gaussian Rendering

import os
import argparse
import numpy as np
import torch
import trimesh
from tqdm import tqdm

from scene import Scene, GaussianModel
from scene.deformable_gaussian_model import DeformableGaussianModel
from utils.camera_utils import cameraList_from_cameras
from utils.graphics_utils import focal2fov
from gaussian_renderer import render, network_gui
import uuid

# === 新增模块 ===
from point_cloud_generator.pointcloudgeneration.generate_pointcloud import infer_one_image
from scripts.mesh_to_smpl import fit_smplx_from_mesh
from scripts.retarget_pose import generate_mesh_from_pose
# =================

def get_args():
    parser = argparse.ArgumentParser()
    # 输入输出
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, default="output/deformable_human", help="Output directory")
    parser.add_argument("--frames_dir", type=str, default=None)

    # 训练参数
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--densify_until_iter", type=int, default=15000)
    parser.add_argument("--sh_degree", type=int, default=3)

    # 渲染参数
    parser.add_argument("--render_pose_dir", type=str, default=None, help="Dir with new pose params (.npy)")

    return parser.parse_args()

def extract_frames(video_path, frames_dir):
    os.makedirs(frames_dir, exist_ok=True)
    os.system(f"ffmpeg -i {video_path} -vf fps=10 {frames_dir}/%06d.png -y")
    print(f"Frames extracted to {frames_dir}")

def prepare_colmap_structure(frames_dir, colmap_dir):
    images_dir = os.path.join(colmap_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    os.system(f"cp {frames_dir}/*.png {images_dir}/")
    print(f"Images copied to {images_dir} for COLMAP")

def run_colmap(colmap_dir):
    db_path = os.path.join(colmap_dir, "database.db")
    os.system(f"colmap feature_extractor --database_path {db_path} --image_path {os.path.join(colmap_dir, 'images')}")
    os.system(f"colmap exhaustive_matcher --database_path {db_path}")
    os.makedirs(os.path.join(colmap_dir, "sparse"), exist_ok=True)
    os.system(f"colmap mapper --database_path {db_path} --image_path {os.path.join(colmap_dir, 'images')} --output_path {os.path.join(colmap_dir, 'sparse')}")
    os.system(f"colmap model_converter --input_path {os.path.join(colmap_dir, 'sparse/0')} --output_path {os.path.join(colmap_dir, 'sparse/0')} --output_type TXT")
    print("COLMAP done.")

def generate_pointclouds(frames_dir, pc_dir):
    os.makedirs(pc_dir, exist_ok=True)
    img_paths = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".png")])
    for img_path in tqdm(img_paths, desc="Generating point clouds"):
        pc = infer_one_image(img_path)  # [N, 3]
        frame_id = os.path.basename(img_path).split('.')[0]
        np.save(os.path.join(pc_dir, f"{frame_id}.npy"), pc)
    print(f"Point clouds saved to {pc_dir}")

def main():
    args = get_args()
    os.makedirs(args.output, exist_ok=True)

    # === Step 1: 视频抽帧 ===
    frames_dir = args.frames_dir or os.path.join(args.output, "frames")
    if not os.path.exists(frames_dir):
        extract_frames(args.video, frames_dir)

    # === Step 2: 生成每帧点云 ===
    pc_dir = os.path.join(args.output, "pointclouds")
    if not os.path.exists(pc_dir):
        generate_pointclouds(frames_dir, pc_dir)

    # === Step 3: COLMAP SfM (简化版：跳过，假设已有 cameras.txt) ===
    colmap_dir = os.path.join(args.output, "colmap")
    if not os.path.exists(os.path.join(colmap_dir, "sparse/0/cameras.txt")):
        os.makedirs(colmap_dir, exist_ok=True)
        prepare_colmap_structure(frames_dir, colmap_dir)
        run_colmap(colmap_dir)
    else:
        print("Using existing COLMAP results.")

    # === Step 4: 构建 Scene + 初始化 GaussianModel ===
    model = DeformableGaussianModel(sh_degree=args.sh_degree)
    scene = Scene(args, model, load_iteration=None, shuffle=False)
    model.training_setup(args)

    # === Step 5: 从点云初始化高斯 ===
    # 取第一帧点云作为初始化（或融合多帧）
    first_pc_path = os.path.join(pc_dir, sorted(os.listdir(pc_dir))[0])
    init_pcd = np.load(first_pc_path)
    model.create_from_pcd(init_pcd, spatial_lr_scale=1.0)
    print(f"Initialized Gaussians from point cloud: {init_pcd.shape[0]} points")

    # === Step 6: 训练 3DGS ===
    print("Starting training...")
    for iteration in tqdm(range(1, args.iterations + 1)):
        model.update_learning_rate(iteration)
        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack[np.random.randint(0, len(viewpoint_stack))]

        render_pkg = render(viewpoint_cam, model, args, bg_color=torch.tensor([1,1,1], device="cuda"))
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = torch.abs(gt_image - image).mean()
        loss = (1.0 - args.lambda_dssim) * Ll1 + args.lambda_dssim * (1.0 - render_pkg["ssim"])

        loss.backward()
        model.optimizer.step()
        model.optimizer.zero_grad()

        if iteration % 1000 == 0:
            print(f"Iteration {iteration}, Loss: {loss.item():.6f}")

        # Densification
        if iteration < args.densify_until_iter:
            model.densification_postfix(viewspace_point_tensor, visibility_filter, radii, viewpoint_cam)
        if iteration == args.densify_until_iter:
            model.densify_and_prune(max_grad=0.005, min_opacity=0.005, extent=1.0, max_screen_size=1)

    print("Training completed.")

    # === Step 7: 提取 Canonical Mesh ===
    mesh_path = os.path.join(args.output, "canonical_mesh.ply")
    if not os.path.exists(mesh_path):
        from games.mesh_splatting.scene.gaussian_mesh_model import GaussianMeshModel
        mesh_model = GaussianMeshModel(model.get_xyz, model.get_scaling, model.get_rotation, model.get_opacity, model.get_covariance)
        mesh_model.extract_mesh(resolution=512, threshold=0.0, path=mesh_path)
        print(f"Canonical mesh saved to {mesh_path}")

    # === Step 8: Mesh → SMPL 拟合 β (只做一次) ===
    smpl_params_path = os.path.join(args.output, "smpl_params.pkl")
    if not os.path.exists(smpl_params_path):
        beta, _ = fit_smplx_from_mesh(mesh_path)  # 只取 beta
        import pickle
        with open(smpl_params_path, 'wb') as f:
            pickle.dump({"beta": beta}, f)
        print(f"SMPL beta fitted and saved to {smpl_params_path}")
    else:
        import pickle
        with open(smpl_params_path, 'rb') as f:
            beta = pickle.load(f)["beta"]

    # === Step 9: 绑定高斯到 SMPL 顶点 ===
    cano_mesh = trimesh.load(mesh_path)
    model.bind_to_smpl_mesh(cano_mesh.vertices)  # 核心绑定
    print("Gaussians bound to SMPL vertices.")

    # === Step 10: 可控姿态渲染 ===
    render_pose_dir = args.render_pose_dir or os.path.join(args.output, "render_poses")
    os.makedirs(render_pose_dir, exist_ok=True)

    # 示例：渲染 5 个新姿态（可替换为外部输入）
    pose_dir = os.path.join(args.output, "poses")
    os.makedirs(pose_dir, exist_ok=True)

    for i in range(5):
        # 生成新姿态 mesh
        new_pose_mesh_path = os.path.join(pose_dir, f"pose_{i:04d}.ply")
        if not os.path.exists(new_pose_mesh_path):
            # 这里用随机姿态，你可以替换为 AIST++/AMASS 姿态
            random_pose = np.random.randn(1, 66) * 0.5
            generate_mesh_from_pose(beta, random_pose, new_pose_mesh_path)

        # 变形高斯
        new_mesh = trimesh.load(new_pose_mesh_path)
        model.deform_with_new_mesh(new_mesh.vertices)

        # 渲染所有训练相机视角
        for idx, cam in enumerate(scene.getTrainCameras()):
            render_pkg = render(cam, model, args, bg_color=torch.tensor([1,1,1], device="cuda"))
            rendering = render_pkg["render"]
            rendering = (rendering.clamp(0,1).cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
            save_path = os.path.join(render_pose_dir, f"pose_{i:04d}_view_{idx:03d}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            from PIL import Image
            Image.fromarray(rendering).save(save_path)

        print(f"Pose {i} rendered.")

    print(f"All done. Results in {args.output}")

if __name__ == "__main__":
    main()