import torch
import numpy as np
import trimesh
from smplx import SMPL
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes

class MeshPoseEstimator:
    def __init__(self, smpl_model_path: str, device='cuda'):
        self.device = device
        self.smpl = SMPL(model_path=smpl_model_path).to(device)

    def load_mesh(self, mesh_path):
        mesh = trimesh.load(mesh_path, process=False)
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=self.device)
        faces = torch.tensor(mesh.faces, dtype=torch.int64, device=self.device)
        return Meshes(verts=[vertices], faces=[faces])

    def estimate_pose(self, mesh_path):
        mesh = self.load_mesh(mesh_path)
        sampled_points = sample_points_from_meshes(mesh, num_samples=5000)  # 采样
        betas = torch.zeros((1, 10), device=self.device, requires_grad=True)
        body_pose = torch.zeros((1, 69), device=self.device, requires_grad=True)
        global_orient = torch.zeros((1, 3), device=self.device, requires_grad=True)

        optimizer = torch.optim.Adam([betas, body_pose, global_orient], lr=0.01)

        for step in range(200):
            smpl_output = self.smpl(betas=betas, body_pose=body_pose, global_orient=global_orient)
            smpl_points = smpl_output.vertices
            loss = torch.mean((sampled_points - smpl_points[:, :5000, :]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                print(f"Step {step} | Loss: {loss.item():.6f}")

        joints = self.smpl(betas=betas, body_pose=body_pose, global_orient=global_orient).joints
        return joints.detach().cpu().numpy()
