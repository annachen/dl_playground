from collections import namedtuple
import tensorflow as tf
import tensorflow_addons as tfa


AugmentationConfig = namedtuple('AugmentationConfig', [
    'angle_range',
    'offset_range',
    'scale_range',
    'brightness_max_delta',
    'contrast_min',
    'contrast_max',
    'hue_max_delta',
    'saturation_min',
    'saturation_max',
], defaults=[
    None,  # angle_range
    None,  # offset_range
    None,  # scale_range
    None,  # brightness_max_delta
    None,  # contrast_min
    None,  # contrast_max
    None,  # hue_max_delta
    None,  # sat_min
    None,  # sat_max
])


def augment(batch, aug_config):
    batch = random_rotate(batch, aug_config.angle_range)
    batch = random_shift(batch, aug_config.offset_range)
    batch = random_image_changes(
        batch,
        brightness_max_delta=aug_config.brightness_max_delta,
        contrast_min=aug_config.contrast_min,
        contrast_max=aug_config.contrast_max,
        hue_max_delta=aug_config.hue_max_delta,
        saturation_min=aug_config.saturation_min,
        saturation_max=aug_config.saturation_max,
    )
    return batch


def random_rotate(batch, angle_range):
    if angle_range is None:
        return batch

    # randomly rotate the batch
    B = tf.shape(batch['image'])[0]
    angles = tf.random.uniform(
        shape=(B,),
        minval=angle_range[0],
        maxval=angle_range[1],
        dtype=tf.float32,
    )
    rotated = tfa.image.rotate(
        batch['image'],
        angles,
    )

    # do the same for the keypoints
    # note that this transform is inverted
    transform = tfa.image.angles_to_projective_transforms(
        angles,
        tf.cast(tf.shape(batch['image'])[1], tf.float32),
        tf.cast(tf.shape(batch['image'])[2], tf.float32),
    )[:, :6]
    transform = tf.reshape(transform, [B, 2, 3])
    # (B, 2, 2)
    transform_mat = tf.linalg.inv(transform[:, :, :2])
    # (B, 2)
    transform_offset = transform[:, :, 2]

    # (B, N, 2)
    kps = batch['keypoints'][:, :, :2][..., ::-1]
    offset_kps = kps - transform_offset[:, tf.newaxis]
    rotated_kps = tf.matmul(
        offset_kps,
        tf.transpose(transform_mat, (0, 2, 1))
    )
    new_kps = tf.concat([
        rotated_kps[..., ::-1],
        batch['keypoints'][:, :, 2:]
    ], axis=-1)

    batch['image'] = rotated
    batch['keypoints'] = new_kps

    return batch


def random_shift(batch, offset_range):
    if offset_range is None:
        return batch

    B = tf.shape(batch['image'])[0]
    offsets = tf.random.uniform(
        shape=(B, 2),
        minval=offset_range[0],
        maxval=offset_range[1],
        dtype=tf.float32,
    )
    ims = tfa.image.translate(
        images=batch['image'],
        translations=offsets[..., ::-1],
    )
    kps = batch['keypoints'][..., :2]
    new_kps = kps + offsets[:, tf.newaxis]
    new_kps = tf.concat([
        new_kps,
        batch['keypoints'][..., 2:]
    ], axis=-1)

    batch['image'] = ims
    batch['keypoints'] = new_kps

    return batch


def random_scale(batch, scale_range):
    if scale_range is None:
        return batch

    scale = tf.random.uniform(
        minval=scale_range[0], maxval=scale_range[1]
    )
    H = tf.shape(batch['image'])[1]
    W = tf.shape(batch['image'])[2]
    target_h = tf.cast(tf.cast(H, tf.float32) * scale, tf.int32)
    target_w = tf.cast(tf.cast(W, tf.float32) * scale, tf.int32)
    ims = tf.image.resize(batch['image'], (target_h, target_w))

    padded_ims, padded_kps = pad_to(
        ims=ims,
        kps=batch['keypoints'],
        target_size=(H, W),
    )
    cropped_ims, cropped_kps = crop_to(
        ims=padded_ims,
        kps=padded_kps,
        target_size=(H, W),
    )

    batch['image'] = cropped_ims
    batch['keypoints'] = cropped_kps

    return batch


def crop_to(ims, kps, target_size):
    H = tf.shape(ims)[1]
    W = tf.shape(ims)[2]

    top_crop = tf.maximum((H - target_size[0]) // 2, 0)
    bot_crop = tf.maximum(H - target_size[0] - top_crop, 0)
    left_crop = tf.maximum((W - target_size[1]) // 2, 0)
    right_crop = tf.maximum(W - target_size[1] - left_crop, 0)

    cropped = ims[:, top_crop:-bot_crop, left_crop:-right_crop]

    offset = tf.stack([top_crop, left_crop])
    coords = kps[..., :2] - offset
    kps = tf.concat([coords, kps[..., 2:]], axis=-1)

    return cropped, kps


def pad_to(ims, kps, target_size):
    # kps: (B, P, 3)
    H = tf.shape(ims)[1]
    W = tf.shape(ims)[2]
    top_pad = tf.maximum((target_size[0] - H) // 2, 0)
    bot_pad = tf.maximum(target_size[0] - H - top_pad, 0)
    left_pad = tf.maximum((target_size[1] - W) // 2, 0)
    right_pad = tf.maximum(target_size[1] - W - left_pad, 0)

    padded = tf.pad(
        ims,
        paddings=[
            [0, 0], [top_pad, bot_pad], [left_pad, right_pad], [0, 0]
        ],
        mode='REFLECT'
    )
    # (2,)
    offset = tf.stack([top_pad, left_pad])
    coords = kps[..., :2] + offset
    kps = tf.concat([coords, kps[..., 2:]], axis=-1)
    return padded, kps


def random_image_changes(
    batch,
    brightness_max_delta=0.2,
    contrast_min=0.0,
    contrast_max=0.5,
    hue_max_delta=0.1,
    saturation_min=0.0,
    saturation_max=2.0,
):
    ims = batch['image']
    if brightness_max_delta is not None:
        ims = tf.image.random_brightness(ims, brightness_max_delta)
    if contrast_min is not None:
        ims = tf.image.random_contrast(
            ims, contrast_min, contrast_max
        )
    if hue_max_delta is not None:
        ims = tf.image.random_hue(ims, hue_max_delta)
    if saturation_min is not None:
        ims = tf.image.random_saturation(
            ims,
            saturation_min,
            saturation_max,
        )
    batch['image'] = ims

    return batch
