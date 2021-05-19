import tensorflow_datasets as tfds
import tensorflow as tf
import os

from dl_playground.data import (
    mnist_superpixel, cats_vs_dogs, my_paintings, my_painting_patches,
    div2k_patches
)


DATASET_CHANNELS = {
    'mnist': 1,
    'mnist_superpixel': 1,
    'mnist_superpixel2': 1,  # added seg
    'mnist_superpixel3': 1,  # added parent and children
    'cifar10': 3,
    'cifar100': 3,
    'div2k_lr': 3,
    'div2k_hr': 3,
    'cats_vs_dogs': 3,
    'dtd': 3,
    'my_paintings': 3,
    'my_painting_patches': 3,
    'div2k_hr_patches': 3,
    'omniglot': 1,
    'fashion_mnist': 1,
}
DATASETS_ROOT = '/home/data/anna/datasets'


def load_dataset(
    dataset,
    split,
    image_only=False,
    to_float=False,
    to_grayscale=False,
    to_data=False,
    **kwargs
):
    if dataset in ['mnist', 'cifar10', 'cifar100', 'dtd', 'fashion_mnist', 'omniglot']:
        dataset = tfds.load(dataset, split=split)
        if split is None:
            train_dset = dataset['train']
            test_dset = dataset['test']
            dataset = train_dset.concatenate(test_dset)

    elif dataset == 'div2k_lr':
        if split == 'test':
            split = 'validation'
        dataset = tfds.load('div2k', split=split)
        dataset = dataset.map(lambda x: {'image': x['lr']})

    elif dataset == 'div2k_hr':
        if split == 'test':
            split = 'validation'
        dataset = tfds.load('div2k', split=split)
        dataset = dataset.map(lambda x: {'image': x['hr']})

    elif dataset.startswith('mnist_superpixel'):
        if split == 'test':
            data_path = os.path.join(
                DATASETS_ROOT, dataset, 'test'
            )
        elif split == 'train':
            data_path = os.path.join(
                DATASETS_ROOT, dataset, 'train'
            )
        dataset = mnist_superpixel.get_dataset(
            data_path,
            max_nodes=kwargs['max_nodes'],
            max_neighbors=kwargs['max_neighbors'],
        )

    elif dataset == 'cats_vs_dogs':
        dataset = cats_vs_dogs.get_dataset(
            split=split,
            **kwargs,
        )

    elif dataset == 'my_paintings':
        assert to_float is True
        assert split == 'train'
        to_float = False
        dataset = my_paintings.get_dataset(
            **kwargs
        )

    elif dataset == 'my_painting_patches':
        assert split == 'train'
        dataset = my_painting_patches.get_dataset(**kwargs)

    elif dataset == 'div2k_hr_patches':
        assert split == 'train'
        dataset = div2k_patches.get_dataset(**kwargs)

    else:
        raise ValueError('Unrecognized dataset {}'.format(dataset))

    preproc_fn = get_preproc_fn(
        image_only, to_float, to_grayscale, to_data
    )
    dataset = dataset.map(preproc_fn)
    return dataset


def get_preproc_fn(image_only, to_float, to_grayscale, to_data):
    def preprocess(sample):
        if to_float:
            sample['image'] = (
                tf.cast(sample['image'], tf.float32) / 255.0
            )

        if to_grayscale:
            sample['image'] = tf.image.rgb_to_grayscale(
                sample['image']
            )

        if to_data:
            assert not image_only
            sample['data'] = sample.pop('image')

        if image_only:
            sample = sample['image']

        return sample

    return preprocess


def split_dataset(dataset, ratios, shuffle_buffer=1000):
    """Split a dataset into several datasets

    Parameters
    ----------
    dataset : tf.data.Dataset
    ratios : [float]
        The ratio for each dataset split. Has to sum up to 1.
    shuffle_buffer : int

    Returns
    -------
    datasets : [tf.data.Dataset]
        A list of datasets, as many as the length of `ratios`

    """
    dset = dataset.shuffle(shuffle_buffer)

    accumulated = [ratios[0]]
    for i in range(1, len(ratios)):
        accumulated.append(ratios[i] + accumulated[i - 1])

    def gen():
        for sample in dset:
            d = tf.random.uniform()
            for idx, thresh in enumerate(accumulated):
                if d < thresh:
                    sample['split_id'] = idx
                    break
            yield sample

    # {key: TensorSpec}
    elem_spec = dataset._type_spec._element_spec
    output_types = {k: spec.dtype for k, spec in elem_spec.items()}
    output_types['split_id'] = tf.int32
    dset = tf.data.Dataset.from_generator(
        gen, output_types=output_types
    )

    datasets = []
    for idx in range(len(ratios)):
        # each idx would create a new dataset
        new_dset = dset.filter(
            lambda x: tf.math.equal(x['split_id'], idx)
        )

        datasets.append(new_dset)

    return datasets
