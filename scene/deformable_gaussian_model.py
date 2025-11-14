class DeformableGaussianModel(GaussianModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.orig_xyz = None          # 记录初始/正则姿态下的位置
        self.vertex_to_gaussian = None  # 每个gaussian最近的SMPL vertex id

    def bind_to_smpl_mesh(self, smpl_vertices):
        # 在第一次得到正则姿态mesh后调用一次
        from scipy.spatial import cKDTree
        tree = cKDTree(smpl_vertices)
        dist, idx = tree.query(self.get_xyz.detach().cpu().numpy(), k=1)
        self.vertex_to_gaussian = torch.from_numpy(idx).cuda()
        self.orig_xyz = self.get_xyz.clone()

    def deform_with_new_mesh(self, new_smpl_vertices):
        # 关键变形函数：在渲染新姿态时调用
        delta = torch.from_numpy(new_smpl_vertices).cuda()[self.vertex_to_gaussian] - \
                torch.from_numpy(orig_cano_vertices).cuda()[self.vertex_to_gaussian]
        self._xyz = self.orig_xyz + delta * 0.9   # 0.9是经验权重，防过变形
        # 可选：同步调节scale，让高斯球随皮肤拉伸
        scale_factor = torch.norm(delta, dim=-1, keepdim=True) + 1.0
        self._scaling = self._scaling + torch.log(scale_factor).repeat(1,3)