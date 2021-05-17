import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np


def subplots(imgs, r, c, size_scale=1., colorbar=False, **imshow_kwargs):
    # imgs: [img] of same shapes
    img_h, img_w = imgs[0].shape[:2]
    full_w = 20.
    each_w = 20. / c
    each_h = each_w / img_w * img_h
    full_h = each_h * r
    fig, axes = plt.subplots(
        r, c, figsize=(full_w * size_scale, full_h * size_scale))
    to_draw = min(len(imgs), r * c)
    if to_draw < len(imgs):
        print("Not enough subplots for all images. Plotting the first {}".format(to_draw))

    if r == 1 or c == 1:
        for i in range(to_draw):
            im = axes[i % c].imshow(imgs[i], **imshow_kwargs)
            axes[i % c].set_xticks([])
            axes[i % c].set_yticks([])
            if colorbar:
                fig.colorbar(im, ax=axes[i % c])
    else:
        for i in range(to_draw):
            im = axes[i // c][i % c].imshow(imgs[i], **imshow_kwargs)
            axes[i // c][i % c].set_xticks([])
            axes[i // c][i % c].set_yticks([])
            if colorbar:
                fig.colorbar(im, ax=axes[i // c][i % c])

    return fig, axes


def visualize_as_clusters(code, n_clusters=4, r=1, c=1):
    """Visualize spatial codes as clusters.

    Parameters
    ----------
    code : np.array, shape (B, H, W, C)
    n_clusters : int
    r : int
        Number of rows to subplot
    c : int
        Number of columns to subplot

    """
    B, H, W, C = code.shape
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(code.reshape((-1, C)))
    labels = kmeans.labels_.reshape((B, H, W))
    _ = subplots(labels, r, c)
    return labels


def visualize_sparse_code_as_clusters(
    code,
    location,
    image_shape,
    n_clusters=4,
    r=1,
    c=1,
    mask=None
):
    """Visualize sparse spatial codes as clusters.

    Parameters
    ----------
    code : np.array, shape (B, N, C)
        The code of some sparse points
    location : np.array, shape (B, N, 2)
        The (y, x) coordinates of the sparse points
    image_shape : (int, int)
        (H, W) of the images to visualize
    n_clusters : int
    r : int
    c : int
    mask : np.array, shape (B, N) | None
        The valid points in the code and location array. There can
        be invalid points due to padding and batching.

    """
    B, N, C = code.shape
    code = code.reshape((-1, C))
    if mask is not None:
        mask = mask.flatten()
    else:
        mask = np.ones((B, N)).flatten()
    valid_code_idx = np.where(mask > 0)[0]
    # (M, C)
    valid_code = code[valid_code_idx]

    # add batch index as part of the location as well
    # (B,)
    batch_idx = np.arange(0, B, 1)
    # (B, N, 1)
    batch_idx = np.tile(
        batch_idx[:, np.newaxis, np.newaxis], [1, N, 1]
    )
    # make location integers
    location = np.round(location).astype(np.int32)
    # (B, N, 3)
    location = np.concatenate([batch_idx, location], axis=2)
    location = location.reshape((-1, 3))
    # (M, 3)
    location = location[valid_code_idx]

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(valid_code)
    # (M,)
    labels = kmeans.labels_

    H, W = image_shape
    arranged = np.zeros((B, H, W))
    arranged[location[:, 0], location[:, 1], location[:, 2]] = (
        labels + 1
    )

    _ = subplots(arranged, r, c)
    return arranged
