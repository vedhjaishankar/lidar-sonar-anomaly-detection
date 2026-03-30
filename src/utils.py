import numpy as np

class PointCloudTransforms:
    """Group of augmentations for point cloud data during training."""
    
    @staticmethod
    def random_rotate_z(points):
        """Randomly rotate the point cloud around the Z axis."""
        theta = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ])
        return points @ rotation_matrix

    @staticmethod
    def random_jitter(points, sigma=0.01, clip=0.05):
        """Randomly jitter points to add noise."""
        N, C = points.shape
        jitter = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
        return points + jitter
        
    @staticmethod
    def compose(transforms_list):
        """Compose multiple transforms."""
        def apply(points):
            for t in transforms_list:
                points = t(points)
            return points
        return apply

def get_train_transforms():
    """Returns a composed training transform."""
    return PointCloudTransforms.compose([
        PointCloudTransforms.random_rotate_z,
        PointCloudTransforms.random_jitter
    ])

def get_val_transforms():
    """Returns a validation transform (usually just identity or none)."""
    return None
