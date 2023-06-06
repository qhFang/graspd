import trimesh
import numpy as np 
import torch
from trimesh.exchange.ply import export_ply

mesh = trimesh.load(f'/apdcephfs/share_1330077/qihangfang/Data/ibs/gd_detergent_bottle_poisson_001_scaled/textured.obj', process=False)
sample = np.load(f'/apdcephfs/share_1330077/qihangfang/Data/ibs/gd_detergent_bottle_poisson_001_scaled/sample.npy', allow_pickle=True)
center = sample.item()['center']

center = torch.from_numpy(center).to('cuda').float().unsqueeze(0)
vertices = torch.from_numpy(mesh.vertices).to('cuda').float().unsqueeze(1)

distance = vertices - center

distance = distance.norm(dim=-1)
part_id = torch.argmin(distance, dim=-1).cpu().numpy()


list_colors = np.random.randint(0, 256, size=(10, 4))

vertices_colors = list_colors[part_id]
print(vertices_colors.shape)

mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_color=vertices_colors)
#mesh.export('/apdcephfs/private_qihangfang/split.ply', vertex_color=True)
mesh.visual.vertex_colors = vertices_colors


assert mesh.visual.kind == "vertex"

mesh.export("/apdcephfs/private_qihangfang/split.ply")

#with open("/apdcephfs/private_qihangfang/split.ply", "wb") as file:
#    file.write(export_ply(mesh, vertex_color=True))

