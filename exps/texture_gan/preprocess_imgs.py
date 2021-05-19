import fire
import imageio
import numpy as np
import imageio
import os

from dl_playground.data.my_paintings import raw_dataset, scale_and_crop


ROOT = '/home/data/anna/datasets/my_painting_patches_109'
n_crops_per_image = 12000
scales = [0.2, 0.4, 0.6, 0.8]
patch_size = 109


def run():
    for sidx, s in enumerate(raw_dataset()):
        print("processing image {}".format(sidx))
        folder = os.path.join(ROOT, str(sidx))
        os.makedirs(folder)

        for i in range(n_crops_per_image):
            scale = np.random.choice(scales)
            crop_loc = np.random.random(size=(2,))
            crop = scale_and_crop(
                sample=s,
                patch_size=patch_size,
                scale=scale,
                crop_loc=crop_loc
            )['image']
            im_path = '{}.png'.format(i)
            int_im = np.round(crop * 255).astype(np.uint8)
            imageio.imsave(os.path.join(folder, im_path), int_im)


if __name__ == '__main__':
    fire.Fire(run)
