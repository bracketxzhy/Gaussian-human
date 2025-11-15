# scene/deformable_gaussian_model.py
from scene.gaussian_model import GaussianModel
import torch
from scipy.spatial import cKDTree

class DeformableGaussianModel(GaussianModel):
    def __init__(self, sh_degree: int = 3):
        super().__init__(sh_degree)
        self.vertex_to_gaussian = None
        self.cano_xyz = None

    def bind_to_smpl_mesh(self, smpl_vertices: np.ndarray):
        tree = cKDTree(smpl_vertices)
        dist, idx = tree.query(self.get_xyz.detach().cpu().numpy())
        self.vertex_to_gaussian = torch.from_numpy(idx).long().to(self.get_xyz.device)
        self.cano_xyz = self.get_xyz.clone()
        print(f"绑定完成：{len(self.get_xyz)} 个高斯 → SMPL 顶点")

    def deform_with_new_mesh(self, new_smpl_vertices: np.ndarray):
        if self.vertex_to_gaussian is None:
            raise RuntimeError("请先调用 bind_to_smpl_mesh")
        new_pos = torch.from_numpy(new_smpl_vertices).to(self.cano_xyz.device)[self.vertex_to_gaussian]
        self._xyz = new_pos.contiguous()  # 直接替换位置（最稳）
        # 可选：同步放大尺度
        delta = new_pos - self.cano_xyz
        scale_factor = (torch.norm(delta, dim=-1, keepdim=True) * 0.5 + 1.0)
        self._scaling = self._scaling + torch.log(scale_factor).repeat(1,3)