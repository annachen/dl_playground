import imageio
import tensorflow as tf
import os
import numpy as np
import cv2


RAW_ROOT = '/home/data/anna/datasets/cats_vs_dogs/PetImages'
STYLIZED_ROOT = '/home/data/anna/datasets/stylized_cats_vs_dogs'

MAX_ID = 12499


def resize(im, size, interpolation=cv2.INTER_LINEAR):
    """
    Resizes an image. Modified from https://github.com/pytorch/vision/
    blob/master/torchvision/transforms/functional_pil.py

    Parameters
    ----------
    im : np.array, shape (H, W, 3)
    size : [int, int] | (int, int) | int
        Desired output size. If size is a sequence like (h, w),
        output size will be matched to this. If size is an int,
        smaller edge of the image will be matched to this number.
        i.e, if height > width, then image will be rescaled to
        (size * height / width, size).
    interpolation : int

    """
    if isinstance(size, int):
        h, w = im.shape[:2]
        if (w <= h and w == size) or (h <= w and h == size):
            return im
        if w < h:
            ow = size
            oh = int(size * h / w)
            return cv2.resize(
                im, (ow, oh), interpolation=interpolation
            )
        else:
            oh = size
            ow = int(size * w / h)
            return cv2.resize(
                im, (ow, oh), interpolation=interpolation
            )
    else:
        return cv2.resize(im, size[::-1], interpolation=interpolation)


def center_crop(im, target_size):
    """Center crops an image.

    Modified from https://github.com/pytorch/vision/blob/master/
    torchvision/transforms/functional.py

    """
    image_height, image_width = im.shape[:2]
    crop_height, crop_width = target_size

    if crop_width > image_width or crop_height > image_height:
        left = (crop_width - image_width) // 2 \
            if crop_width > image_width else 0
        top = (crop_height - image_height) // 2 \
            if crop_height > image_height else 0
        right = (crop_width - image_width + 1) // 2 \
            if crop_width > image_width else 0
        bottom = (crop_height - image_height + 1) // 2 \
            if crop_height > image_height else 0
        im = np.pad(
            im,
            [[top, bottom], [left, right], [0, 0]]
        )
        image_height, image_width = im.shape[:2]
        if crop_width == image_width and crop_height == image_height:
            return img

    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))
    return im[
        crop_top : crop_top + crop_height,
        crop_left : crop_left + crop_width
    ]


def get_dataset(
    data_type,
    cats_only=False,
    dogs_only=False,
    split=None,
    target_size=224,
):
    assert data_type in ['raw', 'stylized', 'paired']
    folders = ['Cat', 'Dog']
    if cats_only:
        folders = ['Cat']
    elif dogs_only:
        folders = ['Dog']

    if split is None:
        ids = range(MAX_ID + 1)
    elif split == 'train':
        ids = list(range(MAX_ID + 1))[:-1000]
    elif split == 'test':
        ids = list(range(MAX_ID + 1))[-1000:]
    else:
        raise ValueError()

    def gen():
        for i in ids:
            for fidx, folder in enumerate(folders):
                if data_type in ['stylized', 'paired']:
                    name = '{}.jpg'.format(i)
                    path = os.path.join(STYLIZED_ROOT, folder, name)
                    if os.path.isfile(path):
                        im = imageio.imread(path)
                        data_stylized = {
                            'image': im[..., :3],
                            'label': fidx,
                            'id': i,
                        }
                    elif data_type == 'paired':
                        continue
                if data_type in ['raw', 'paired']:
                    name = '{}.jpg'.format(i)
                    path = os.path.join(RAW_ROOT, folder, name)
                    if os.path.isfile(path):
                        im = imageio.imread(path)
                        if target_size is not None:
                            # this mirrors how the images are
                            # processed in pytorch when generating the
                            # stylized version
                            im = resize(im, target_size)
                            im = center_crop(
                                im, (target_size, target_size)
                            )
                        data_raw = {
                            'image': im[..., :3],
                            'label': fidx,
                            'id': i,
                        }
                    elif data_type == 'paired':
                        continue

                if data_type == 'raw':
                    yield data_raw
                elif data_type == 'stylized':
                    yield data_stylized
                elif data_type == 'paired':
                    yield (data_raw, data_stylized)
                else:
                    raise ValueError()

    output_types = {
        'image': tf.uint8,
        'label': tf.int32,
        'id': tf.int32,
    }
    output_shapes = {
        'image': tf.TensorShape([None, None, 3]),
        'label': tf.TensorShape([]),
        'id': tf.TensorShape([]),
    }

    if data_type == 'paired':
        output_types = (output_types, output_types)
        output_shapes = (output_shapes, output_shapes)

    return tf.data.Dataset.from_generator(
        gen,
        output_types=output_types,
        output_shapes=output_shapes,
    )
