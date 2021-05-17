from collections import namedtuple
import cv2
import numpy as np


Detection = namedtuple('Detection', [
    'keypoints',  # (N, 2)
    'scores',  # (N,)
    'data_idx',
])

Evaluation = namedtuple('Evaluation', [
    'detection',  # Detection | dict
    'threshold',  # float
    'tp',  # (N,)
    'valid',  # (N,) opposite of ignore
])


def visualize(im, colors, detection=None, gt=None):
    im = np.copy(im[..., ::-1])
    if gt is not None:
        im = _visualize_gt(im, gt, colors)
    if detection is not None:
        im = _visualize_det(im, detection, colors)
    return im[..., ::-1]


def _visualize_gt(im, gt, colors):
    # gt: Detection
    for pidx, pt in enumerate(gt.keypoints):
        if gt.scores[pidx] < 1e-4:
            continue
        center = np.round(pt).astype(np.int32)
        center = list(map(int, center))
        im = cv2.drawMarker(
            img=im,
            position=(center[1], center[0]),
            color=colors[pidx],
            markerType=cv2.MARKER_CROSS,
            markerSize=5,
            thickness=1,
        )
    return im


def _visualize_det(im, det, colors):
    # det: Detection
    for pidx, pt in enumerate(det.keypoints):
        center = np.round(pt).astype(np.int32)
        center = list(map(int, center))
        im = cv2.drawMarker(
            img=im,
            position=(center[1], center[0]),
            color=colors[pidx],
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=5,
            thickness=1,
        )
    return im
