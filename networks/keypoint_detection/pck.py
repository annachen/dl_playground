import numpy as np

from artistcritic.networks.keypoint_detection.utils import Evaluation


def pck(detections, gts, threshold):
    """Calculate pck scores

    Parameters
    ----------
    detections : [Detection]
    gts : [Detection]
    threshold : float

    """
    raw_detections = np.array([det.keypoints for det in detections])
    kp_gts = np.array([det.keypoints for det in gts])
    mask = np.array([det.scores for det in gts])
    scores, tp = _pck(raw_detections, kp_gts, threshold, mask)

    evals = []
    for det, is_tp, valid in zip(detections, tp, mask):
        ev = Evaluation(
            detection=det._asdict(),
            threshold=threshold,
            tp=is_tp,
            valid=valid,
        )
        evals.append(ev)

    return scores, evals


def _pck(detections, gts, threshold, mask=None):
    """Returns the pck scores per keypoint.

    Parameters
    ----------
    detections : np.array, (N, P, 2)
        Detections; each is (y, x)
    gts : np.array, (N, P, 2)
        Ground truth; each is y, x
    threshold : float
        L2 distance threshold in pixels for a detection to be
        considered truth positive
    mask : np.array | None, (N, P)
        Valid keypoints

    Returns
    -------
    score : np.array, (P,)
        Scores per keypoint p averaged across samples (ignoring the
        ones with 0 mask)
    tp : np.array, (N, P)
        Whether each detection is a true positive

    """
    diff = detections - gts
    # (N, P)
    dist = np.linalg.norm(diff, axis=-1)

    # (N, P)
    is_correct = (dist < threshold)

    if mask is None:
        score = np.mean(is_correct, axis=0)
    else:
        score = np.sum(is_correct * mask, axis=0) / (np.sum(
            mask, axis=0
        ) + 1e-6)

    # (P,), (N, P)
    return score, is_correct
