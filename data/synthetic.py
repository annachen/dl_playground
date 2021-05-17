"""Synthetic datasets."""

import numpy as np
import tensorflow as tf


def gaussian_in_nd(
    n_dims,
    n_centers,
    n_points,
    stddev,
    limit=1.0,
    splits=None,
    seed=None
):
    """Randomly sample 2D points on a plane.

    The centers are uniformly sampled in [-limit, limit] range.

    Parameters
    ----------
    n_dims : int
        Number of dimensions
    n_centers : int
        Number of clusters
    n_points : int
        Number of total points to generate
    stddev : float
    splits : [float] | None
        If not None, the list needs to sum up to 1. The ratio of
        samples going into different dataset splits
    seed : int | None

    """
    rs = np.random.RandomState(seed)

    # sample the centers
    centers = rs.uniform(
        low=-limit, high=limit, size=n_centers * n_dims
    )
    centers = centers.reshape((n_centers, n_dims))
    print("centers: ", centers)

    # roughly equally sample points from each center
    n_points_per_center = np.array(
        [n_points // n_centers] * n_centers
    )
    remainder = n_points % n_centers
    if remainder > 0:
        n_points_per_center[:remainder] += 1

    points = []
    cluster_ids = []
    for cidx, center in enumerate(centers):
        cur_n_points = n_points_per_center[cidx]
        cov = np.eye(n_dims) * stddev
        # (N, n_dims)
        cur_points = rs.multivariate_normal(
            mean=center, cov=cov, size=cur_n_points
        )
        points.extend(cur_points)
        cluster_ids.extend([cidx] * cur_n_points)

    # shuffle the dataset
    idxs = np.arange(0, n_points, 1)
    rs.shuffle(idxs)
    points = np.array(points)[idxs]
    cluster_ids = np.array(cluster_ids)[idxs]

    data = {
        'data': points.astype(np.float32),
        'label': cluster_ids,
    }

    if splits is None:
        return tf.data.Dataset.from_tensor_slices(data)

    n_samples = []
    for ratio in splits[:-1]:
        n = np.round(n_points * ratio).astype(np.int32)
        n_samples.append(n)
    n_samples.append(n_points - sum(n_samples))

    datasets = []
    start = 0
    for ns in n_samples:
        cur_data = {
            'data': data['data'][start : start + ns],
            'label': data['label'][start : start + ns],
        }
        datasets.append(tf.data.Dataset.from_tensor_slices(cur_data))

    return datasets


def vertical_or_horizontal(im_size, n_images, splits=None, seed=None):
    # creates a dataset that has three classes:
    # 1 - only vertical lines
    # 2 - only horizontal lines
    # 3 - both vertical and horizontal lines
    n_lines = im_size // 2

    rs = np.random.RandomState(seed=seed)
    ims = []
    labels = []
    for _ in range(n_images):
        n_v_lines = rs.randint(1, n_lines)
        n_h_lines = rs.randint(1, n_lines)

        v_locs = rs.choice(
            np.arange(im_size), n_v_lines, replace=False,
        )
        h_locs = rs.choice(
            np.arange(im_size), n_h_lines, replace=False,
        )

        cls_label = rs.randint(0, 3)
        im = np.zeros((im_size, im_size, 1), dtype=np.float32)
        if cls_label == 0 or cls_label == 2:
            im[:, v_locs, :] = 1.0
        if cls_label == 1 or cls_label == 2:
            im[h_locs, :, :] = 1.0

        ims.append(im)
        labels.append(cls_label)

    ims = np.array(ims)
    labels = np.array(labels)

    data = {
        'image': ims,
        'label': labels,
    }

    if splits is None:
        return tf.data.Dataset.from_tensor_slices(data)

    # num samples per split
    n_samples = []
    for ratio in splits[:-1]:
        n = np.round(n_images * ratio).astype(np.int32)
        n_samples.append(n)
    n_samples.append(n_images - sum(n_samples))

    datasets = []
    start = 0
    for ns in n_samples:
        cur_data = {
            'image': data['image'][start : start + ns],
            'label': data['label'][start : start + ns],
        }
        datasets.append(tf.data.Dataset.from_tensor_slices(cur_data))

    return datasets


def sphere_in_space(radii, n_points, limit=1.0, splits=None, seed=None):
    """Generates spheres in space.

    Each sphere is represented as a center (randomly sampled) and
    points are sampled around the center by a Gaussian distribution.

    The centers are sampled in the [-limit, limit] cubic space.

    Parameters
    ----------
    radii : [float]
        The stddev of the Gaussians. The length of the list is the
        number of spheres in the space.
    n_points : int
        Number of total points (size of dataset)
    limit : float
    splits : [float] | None
        If not None, the list needs to sum up to 1. The ratio of
        samples going into different dataset splits
    seed : int | None

    Returns
    -------
    dataset : tf.data.Dataset | [tf.data.Dataset]
        A single dataset if `splits` is None. Otherwise, a list of
        split datasets.

    """
    rs = np.random.RandomState(seed)

    n_spheres = len(radii)
    centers = rs.uniform(
        low=-limit, high=limit, size=n_spheres * 3
    )
    centers = centers.reshape((n_spheres, 3))
    print("centers: {}".format(centers))

    # roughly equally sample points from each sphere
    n_points_per_sphere = np.array(
        [n_points // n_spheres] * n_spheres
    )
    remainder = n_points % n_spheres
    if remainder > 0:
        n_points_per_sphere[:remainder] += 1

    points = []
    cluster_ids = []
    for cidx, center in enumerate(centers):
        cur_n_points = n_points_per_sphere[cidx]
        cov = np.eye(3) * radii[cidx]
        # (N, 3)
        cur_points = rs.multivariate_normal(
            mean=center, cov=cov, size=cur_n_points
        )
        points.extend(cur_points)
        cluster_ids.extend([cidx] * cur_n_points)

    idxs = np.arange(0, n_points, 1)
    rs.shuffle(idxs)
    points = np.array(points)[idxs]
    cluster_ids = np.array(cluster_ids)[idxs]

    data = {
        'point': points,
        'cluster_id': cluster_ids,
    }

    if splits is None:
        return tf.data.Dataset.from_tensor_slices(data)

    n_samples = []
    for ratio in splits[:-1]:
        n = np.round(n_points * ratio).astype(np.int32)
        n_samples.append(n)
    n_samples.append(n_points - sum(n_samples))

    datasets = []
    start = 0
    for ns in n_samples:
        cur_data = {
            'point': data['point'][start : start + ns],
            'cluster_id': data['cluster_id'][start : start + ns],
        }
        datasets.append(tf.data.Dataset.from_tensor_slices(cur_data))

    return datasets


def plane_in_space(n_planes, n_points, limit=1):
    """Randomly sample planes and points on the plane in space.

    The planes are represented as ax + by + cz + d = 0, where
    (a, b, c) is sampled as a unit normal and d is sampled between
    [-limit, limit].

    Then, points are sampled in the cubic [-limit, limit] and
    projected onto the plane(s).

    Parameters
    ----------
    n_planes : int
    n_points : int
    limit : float

    Returns
    -------
    dataset : tf.data.Dataset

    """
    normals = np.random.uniform(
        low=-limit, high=limit, size=n_planes * 3
    )
    normals = normals.reshape((n_planes, 3))
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    # (N,)
    ds = np.random.uniform(low=-limit, high=limit, size=n_planes)

    # (N, 4)
    planes = np.concatenate([normals, ds[:, np.newaxis]], axis=1)

    points = np.random.uniform(
        low=-limit, high=limit, size=n_points * 3
    )
    points = points.reshape((n_points, 3))

    n_points_per_plane = np.array(
        [n_points // n_planes] * n_planes
    )
    remainder = n_points % n_planes
    n_points_per_plane[:remainder] += 1

    projected = []
    cluster_idx = []
    point_idx = 0
    for plane_idx in range(n_planes):
        cur_n_points = n_points_per_plane[plane_idx]
        cur_points = points[point_idx : point_idx + cur_n_points]
        proj = project_points_to_plane(cur_points, planes[plane_idx])
        projected.extend(proj)
        cluster_idx.extend([plane_idx] * cur_n_points)

    idxs = np.arange(0, n_points, 1)
    np.random.shuffle(idxs)
    projected = np.array(projected)[idxs]
    cluster_idx = np.array(cluster_idx)[idxs]
    data = {
        'point': projected,
        'cluster_id': cluster_idx,
    }
    return tf.data.Dataset.from_tensor_slices(data)


def project_points_to_plane(points, plane):
    """Projects 3D points to a plane.

    Parameters
    ----------
    points : np.array, shape (N, 3)
    plane : np.array, shape (4,)
        (a, b, c, d) for the plane ax + by + cz + d = 0

    Returns
    -------
    projected : np.array, shape (N, 3)

    """
    # Let the input point be p0 (x0, y0, z0)
    # Let the projected point be p1 (x1, y1, z1)
    # Since the line going throuth p0 and p1 is parallel to the normal
    # of the plane (a, b, c), we can set x1 = x0 + at, y1 = y0 + bt,
    # and z1 = z0 + ct.
    # We also have ax1 + by1 + cz1 + d = 0. We can solve for t.
    n = plane[:3]
    # (N,)
    t_top = -np.sum(points * n[np.newaxis], axis=1) - plane[3]
    t_bottom = np.sum(n * n)
    t = t_top / t_bottom
    # (N, 3)
    nt = n[np.newaxis] * t[:, np.newaxis]
    projected = points + nt
    return projected
