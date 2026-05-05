import numpy as np

class PointCloudTransforms:
    """Augmentations for point cloud data during training."""

    @staticmethod
    def random_rotate_z(points):
        """Randomly rotate the point cloud around the Z axis."""
        theta = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ])
        return (points @ rotation_matrix).astype(np.float32)

    @staticmethod
    def random_jitter(points, sigma=0.01, clip=0.05):
        """Randomly jitter points to add noise."""
        N, C = points.shape
        jitter = np.clip(sigma * np.random.randn(N, C), -clip, clip)
        return (points + jitter).astype(np.float32)

    @staticmethod
    def random_dropout(points, max_dropout_ratio=0.20):
        """
        Randomly drop up to max_dropout_ratio of points and duplicate
        others to keep array size constant.  Simulates sonar shadow zones
        and beam dropouts.
        """
        dropout_ratio = np.random.uniform(0, max_dropout_ratio)
        drop_idx = np.where(np.random.random(points.shape[0]) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            # Replace dropped points with a random surviving point
            keep_idx = np.where(np.random.random(points.shape[0]) > dropout_ratio)[0]
            if len(keep_idx) == 0:          # edge case: everything dropped
                return points.astype(np.float32)
            fill = points[np.random.choice(keep_idx, size=len(drop_idx))]
            points = points.copy()
            points[drop_idx] = fill
        return points.astype(np.float32)

    @staticmethod
    def random_scale(points, lo=0.9, hi=1.1):
        """Uniform scale jitter — simulates depth estimation uncertainty."""
        return (points * np.random.uniform(lo, hi)).astype(np.float32)

class Compose:
    """Compose multiple transforms (must be a class so Windows multiprocessing can pickle it)."""
    def __init__(self, transforms_list):
        self.transforms_list = transforms_list

    def __call__(self, points):
        for t in self.transforms_list:
            points = t(points)
        return points

def get_train_transforms():
    """Returns a composed training transform."""
    return Compose([
        PointCloudTransforms.random_rotate_z,       # rotation invariance
        PointCloudTransforms.random_jitter,         # point-level noise
        PointCloudTransforms.random_dropout,        # simulate sonar dropouts
        PointCloudTransforms.random_scale,          # depth estimation uncertainty
    ])

def get_val_transforms():
    """Returns a validation transform (identity — no augmentation at eval time)."""
    return None
