import numpy as np


def random_crop(img, crop_size, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()

    loc_x = random_state.randint(0, img.shape[1] - crop_size[1])
    loc_y = random_state.randint(0, img.shape[0] - crop_size[0])

    return img[loc_y:loc_y + crop_size[0], loc_x:loc_x + crop_size[1]]


def random_moving_crops(img, crop_size, movement_step, n_steps, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()

    pair_dist = np.asarray(movement_step) * (n_steps - 1)

    while True:
        loc_x = random_state.randint(0, img.shape[1] - crop_size[1])
        loc_y = random_state.randint(0, img.shape[0] - crop_size[0])
        next_loc_x = loc_x + pair_dist[1]
        next_loc_y = loc_y + pair_dist[0]
        if (next_loc_x >= 0 and next_loc_x < img.shape[1] - crop_size[1] and
                next_loc_y >= 0 or next_loc_y < img.shape[0] - crop_size[0]):
            break

    crops = []
    for i in range(n_steps):
        crop = img[
            loc_y + movement_step[0] * i : loc_y + movement_step[0] * i + crop_size[0],
            loc_x + movement_step[1] * i : loc_x + movement_step[1] * i + crop_size[1],
        ]
        crops.append(crop)
    return crops
