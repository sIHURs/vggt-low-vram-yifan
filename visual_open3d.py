import torch
import numpy as np
import open3d as o3d

def visualize_open3d(pred_dict, conf_threshold=50.0, show_cameras=True, max_points=500_000):
    """
    用 Open3D 显示点云与相机。
    参数：
        pred_dict: 模型输出字典，包括 world_points, world_points_conf, extrinsic, intrinsic, images 等
        conf_threshold: 置信度百分比阈值 (越高 → 过滤掉更多低置信度点)
        show_cameras: 是否显示相机坐标轴
        max_points: 最多显示多少点（太多会卡）
    """

    # 取出数据
    world_points = pred_dict["world_points"]   # (S, H, W, 3)
    conf = pred_dict["world_points_conf"]      # (S, H, W)
    images = pred_dict.get("images", None)     # (S, 3, H, W) 可选
    # extrinsics = pred_dict["extrinsic"]        # (S, 3, 4)

    B, S, H, W, _ = world_points.shape

    # --- Flatten
    points = world_points.reshape(-1, 3)
    conf_flat = conf.reshape(-1)

    # --- 置信度过滤
    threshold_val = np.percentile(conf_flat, conf_threshold)
    mask = conf_flat >= threshold_val
    points = points[mask]

    # --- 下采样（避免卡顿）
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]

    # --- 获取颜色（如果存在图像）
    if images is not None:
        colors = (images.transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)
        colors = colors[mask]
        if len(colors) > max_points:
            colors = colors[idx]
    else:
        colors = np.ones_like(points) * 128  # 灰色

    # --- 创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    # --- 创建可视化对象
    geometries = [pcd]

    # --- 相机坐标轴
    # if show_cameras:
    #     for i, ext in enumerate(extrinsics):
    #         T = np.eye(4)
    #         T[:3, :] = ext
    #         frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    #         frame.transform(T)
    #         geometries.append(frame)

    # --- 可视化
    # print(f"💡 显示点数: {len(points)}, 相机数: {len(extrinsics if show_cameras else 0}")
    o3d.visualization.draw_geometries(geometries)


if __name__ == "__main__":
    pred = torch.load("output/ampere/predictions_210_1_images_11Pig800x800.pt")

    # 只显示少量点，不显示相机
    visualize_open3d(pred, conf_threshold=80.0, show_cameras=False, max_points=200_00)
