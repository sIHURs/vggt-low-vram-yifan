# thanks Claude

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_recon(camera_poses, camera_matrices, world_points, rgb_images, 
                          frustum_size=1.0, point_subsample=1000, camera_colors=None, 
                          figsize=(12, 10), show_plot=True):
    """
    Plot 3D reconstruction results with camera frustums and point cloud.
    
    Parameters:
    -----------
    camera_poses : np.ndarray, shape (N, 3, 4)
        Camera poses in world coordinates [R|t] format
    camera_matrices : np.ndarray, shape (N, 3, 3)
        Camera intrinsic matrices K in OpenCV convention
    world_points : np.ndarray, shape (N, H, W, 3)
        Dense world-space points for each camera view
    rgb_images : np.ndarray, shape (N, 3, H, W) or (N, H, W, 3)
        Input RGB images corresponding to each camera
    frustum_size : float, default=1.0
        Size scaling factor for camera frustums
    point_subsample : int, default=1000
        Number of points to subsample from point cloud for visualization
    camera_colors : list or None
        Colors for each camera frustum. If None, uses default colormap
    figsize : tuple, default=(12, 10)
        Figure size for the plot
    show_plot : bool, default=True
        Whether to display the plot
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The 3D axes object
    """
    
    # Handle different RGB image formats
    if rgb_images.shape[1] == 3:  # (N, 3, H, W) format
        rgb_images = rgb_images.transpose(0, 2, 3, 1)  # Convert to (N, H, W, 3)
    
    N, H, W = world_points.shape[:3]
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract and plot point cloud
    valid_points = []
    colors = []
    
    for i in range(N):
        # Get valid points (assuming invalid points have all zeros or very large values)
        points_3d = world_points[i].reshape(-1, 3)
        rgb_vals = rgb_images[i].reshape(-1, 3)
        
        # Filter out invalid points (you may need to adjust this condition)
        valid_mask = np.all(np.abs(points_3d) < 1000, axis=1) & np.any(points_3d != 0, axis=1)
        
        if np.sum(valid_mask) > 0:
            valid_points.append(points_3d[valid_mask])
            colors.append(rgb_vals[valid_mask])
    
    if valid_points:
        all_points = np.vstack(valid_points)
        all_colors = np.vstack(colors)
        
        # Subsample points for visualization
        if len(all_points) > point_subsample:
            indices = np.random.choice(len(all_points), point_subsample, replace=False)
            all_points = all_points[indices]
            all_colors = all_colors[indices]
        
        # Normalize colors to [0, 1] if needed
        if all_colors.max() > 1.0:
            all_colors = all_colors / 255.0
        
        # Plot point cloud
        ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2], 
                  c=all_colors, s=1, alpha=0.6)
    
    # Set up camera colors
    if camera_colors is None:
        cmap = plt.cm.tab10
        camera_colors = [cmap(i % 10) for i in range(N)]
    
    # Plot camera frustums
    for i in range(N):
        pose = camera_poses[i]  # (3, 4) matrix [R|t]
        K = camera_matrices[i]  # (3, 3) intrinsic matrix
        
        # Extract rotation and translation
        R = pose[:, :3]  # (3, 3) rotation matrix
        t = pose[:, 3]   # (3,) translation vector
        
        # Camera center in world coordinates
        camera_center = -R.T @ t
        
        # Define image corners in pixel coordinates
        corners_2d = np.array([
            [0, 0, 1],
            [W-1, 0, 1],
            [W-1, H-1, 1],
            [0, H-1, 1]
        ]).T  # (3, 4)
        
        # Backproject to normalized camera coordinates
        K_inv = np.linalg.inv(K)
        corners_normalized = K_inv @ corners_2d  # (3, 4)
        
        # Scale by frustum size and transform to world coordinates
        corners_cam = corners_normalized * frustum_size
        corners_world = R.T @ corners_cam + camera_center[:, np.newaxis]
        
        # Create frustum vertices
        frustum_vertices = np.concatenate([
            camera_center.reshape(1, 3),
            corners_world.T
        ], 0)  # (5, 3) - camera center + 4 corners
        
        # Define frustum faces (triangular faces forming the pyramid)
        faces = [
            [0, 1, 2],  # Camera center to corner 0-1
            [0, 2, 3],  # Camera center to corner 1-2
            [0, 3, 4],  # Camera center to corner 2-3
            [0, 4, 1],  # Camera center to corner 3-0
            [1, 2, 3, 4]  # Far plane (rectangle)
        ]
        
        # Create and add frustum faces
        frustum_collection = []
        for face in faces[:-1]:  # Triangular faces
            triangle = frustum_vertices[face]
            frustum_collection.append(triangle)
        
        # Add rectangular far plane
        rectangle = frustum_vertices[faces[-1]]
        frustum_collection.append(rectangle)
        
        # Add frustum to plot
        poly3d = Poly3DCollection(frustum_collection, 
                                 facecolors=camera_colors[i], 
                                 alpha=0.3, 
                                 edgecolors='black',
                                 linewidths=0.5)
        ax.add_collection3d(poly3d)
        
        # Plot camera center
        # ax.scatter(camera_center[0], camera_center[1], camera_center[2], 
        #           c=[camera_colors[i]], s=50, marker='o')
        
        # Add camera label
        ax.text(camera_center[0], camera_center[1], camera_center[2], 
               f' {i}', fontsize=8)
    
    # Set axis properties
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D Reconstruction')
    
    # Set equal aspect ratio
    if valid_points:
        max_range = np.max(all_points.max(axis=0) - all_points.min(axis=0)) / 2.0
        mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
        mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
        mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Show plot if requested
    if show_plot:
        plt.show()
    
    return fig, ax


# Example usage function
def example_usage():
    """
    Example of how to use the plot_3d_reconstruction function.
    """
    # Create synthetic data for demonstration
    N = 5  # Number of cameras
    H, W = 480, 640  # Image dimensions
    
    # Synthetic camera poses (circular arrangement)
    camera_poses = []
    for i in range(N):
        angle = 2 * np.pi * i / N
        
        # Rotation matrix (looking towards center)
        R = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
        
        # Translation (positioned in circle)
        t = np.array([3 * np.cos(angle), 0, 3 * np.sin(angle)])
        
        pose = np.column_stack([R, t])
        camera_poses.append(pose)
    
    camera_poses = np.array(camera_poses)
    
    # Synthetic camera matrices
    focal_length = 500
    cx, cy = W // 2, H // 2
    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ])
    camera_matrices = np.tile(K[np.newaxis], (N, 1, 1))
    
    # Synthetic world points (random point cloud)
    world_points = np.random.randn(N, H, W, 3) * 2
    
    # Synthetic RGB images
    rgb_images = np.random.rand(N, H, W, 3)
    
    # Plot the reconstruction
    fig, ax = plot_3d_reconstruction(
        camera_poses=camera_poses,
        camera_matrices=camera_matrices,
        world_points=world_points,
        rgb_images=rgb_images,
        frustum_size=0.5,
        point_subsample=5000
    )
    
    return fig, ax


if __name__ == "__main__":
    # Run example
    example_usage()

