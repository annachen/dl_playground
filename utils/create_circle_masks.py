import cv2
import numpy as np


patch_sizes = np.arange(3, 12, 2)

mask_dict = {}

for ps in patch_sizes:
    radius = ps // 2
    mask = np.zeros((ps, ps), dtype=np.uint8)
    mask = cv2.circle(
        img=mask,
        center=(radius, radius),
        radius=radius,
        color=1,
        thickness=-1
    )
    mask_dict[ps] = mask


np.save("circle_masks.npy", mask_dict)
