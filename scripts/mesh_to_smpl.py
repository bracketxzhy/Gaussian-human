import torch
from human_body_prior.tools.model_loader import load_vposer
from psbody.mesh import Mesh
from smplx import SMPLX

body_model = SMPLX("data/SMPLX_NEUTRAL.npz").cuda()
vposer, _ = load_vposer("data/vposer_v1_0", vp_model='snapshots')

def fit_smplx(mesh_path):
    mesh = Mesh(filename=mesh_path)
    vertices = torch.from_numpy(mesh.v).cuda().float()

    # 简单SMPLify-X流程（已有现成实现可直接调用）
    from easy_mocap.scripts.fit_smplx import run_fitting
    # 或者直接用SMPLify-X官方repo的单mesh版本
    # 这里给出最小可运行版本
    ...

# 用法：python scripts/mesh_to_smpl.py --mesh output/mesh.ply --out params.pkl
# 输出 global beta (10,) + per-frame body_pose_vp (32, 66)