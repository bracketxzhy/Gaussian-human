body_model = SMPLX(...)
new_body_pose = torch.tensor(new_pose)   # (66,) or vposer latent
output = body_model(body_pose=new_body_pose, betas=beta)
new_vertices = output.vertices[0].cpu().numpy()
new_mesh = trimesh.Trimesh(new_vertices, body_model.faces)
new_mesh.export("output/new_pose_mesh.ply")