import cv2
import fire
import imageio
import glob
import os
import numpy as np
import matplotlib.cm


WNAME = 'image'
DATA_ROOT = '/home/data/anna/datasets/hand_gesture_sketches/crops'

POINT_LABELS = [
    'wrist',
    'thumb_palm',
    'thumb_palm_edge',
    'thumb_joint',
    'thumb_tip',
    'index_palm',
    'index_joint1',
    'index_joint2',
    'index_tip',
    'middle_palm',
    'middle_joint1',
    'middle_joint2',
    'middle_tip',
    'ring_palm',
    'ring_joint1',
    'ring_joint2',
    'ring_tip',
    'pinky_palm',
    'pinky_joint1',
    'pinky_joint2',
    'pinky_tip',
]

CMAP = matplotlib.cm.get_cmap('jet')
COLORS = CMAP(np.arange(21) / 20.0)[:, :3]


def run(annotation_root, data_root=DATA_ROOT):
    if not os.path.isdir(annotation_root):
        os.makedirs(annotation_root)

    im_files = glob.glob(os.path.join(data_root, '*.png'))
    #im_files.sort()

    for im_file in im_files:
        im_name = os.path.basename(im_file)
        im_name = os.path.splitext(im_name)[0]
        anno_path = os.path.join(
            annotation_root, '{}.npy'.format(im_name)
        )
        if os.path.isfile(anno_path):
            print("{} exists, skipping...".format(anno_path))
            continue

        points = annotate_image(im_file, colors=COLORS)
        occluded = annotate_occlusion(im_file, points, COLORS)

        # for each occlusion annotation, mark the closest point as
        # occluded
        is_visible = np.ones(len(points))
        for occ_pt in occluded:
            dist = np.linalg.norm(points - occ_pt, axis=-1)
            pt_idx = np.argmin(dist)
            is_visible[pt_idx] = 0

        points = np.concatenate(
            [points, is_visible[:, np.newaxis]], axis=-1
        )

        print(points)
        np.save(anno_path, points)


def annotate_image(im_path, colors, target_size=600):
    im = imageio.imread(im_path)[..., ::-1]
    ratio = float(target_size) / im.shape[0]
    im = cv2.resize(
        im, (target_size, target_size)
    )
    cv2.imshow(WNAME, im)

    ims = [im]
    points = []
    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            points.append((x, y))
            pidx = len(points) - 1

            color = np.round(colors[pidx] * 255).astype(np.uint8)
            color = map(int, color)
            im_copy = np.copy(ims[-1])
            im_copy = cv2.circle(
                img=im_copy,
                center=(x, y),
                radius=2,
                color=tuple(color),
                thickness=-1
            )
            cv2.imshow(WNAME, im_copy)

            ims.append(im_copy)

    cv2.setMouseCallback(WNAME, click)

    while len(points) < len(POINT_LABELS):
        cv2.waitKey(100)

    return np.array(points) / ratio


def annotate_occlusion(im_path, points, colors, target_size=600):
    im = imageio.imread(im_path)[..., ::-1]
    im = np.array(im)

    for pidx, p in enumerate(points):
        x, y = np.round(p).astype(np.int32)
        color = np.round(colors[pidx] * 255).astype(np.uint8)
        color = tuple(map(int, color))
        im = cv2.circle(
            img=im,
            center=(x, y),
            radius=2,
            color=tuple(color),
            thickness=-1
        )

    ratio = float(target_size) / im.shape[0]
    im = cv2.resize(
        im, (target_size, target_size)
    )
    cv2.imshow(WNAME, im)

    print("Click on the occluded points. Press q to stop")

    occluded = []
    ims = [im]
    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            occluded.append((x, y))

            im_copy = np.copy(ims[-1])
            im_copy = cv2.circle(
                img=im_copy,
                center=(x, y),
                radius=10,
                color=(0, 0, 255),
                thickness=1
            )
            cv2.imshow(WNAME, im_copy)

            ims.append(im_copy)

    cv2.setMouseCallback(WNAME, click)

    while True:
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    return np.array(occluded) / ratio


if __name__ == '__main__':
    fire.Fire(run)
