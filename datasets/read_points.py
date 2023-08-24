import open3d as o3d
import numpy as np

print("Load a ply point cloud, print it, and render it")
# ply_point_cloud = o3d.data.PLYPointCloud()
plyname = r'mesh_k_2.ply'
# 读点云
pcd = o3d.io.read_point_cloud(plyname)
print(pcd)
print(np.asarray(pcd.points))
# 点云显示
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])
