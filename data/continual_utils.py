"""Utility functions for converting datasets for continual learning."""

from collections import defaultdict
import tensorflow as tf
import numpy as np

SHUFFLE_BUFFER = 1000
EPS = 1e-5

def to_clsf_datasets(dataset, class_id_lists, relabel_classes=True):
    """Converts a classification dataset to several tasks.

    Parameters
    ----------
    dataset : tf.data.Dataset
    class_id_set : [[int]]
        Each element is a list of labels to be separated into a task.
    relabel_classes : bool
        Whether to relabel the classes in each task from 0

    Returns
    -------
    datasets : [tf.data.Dataset]
        A list of datasets, as many as the length of `class_id_lists`.
        Each dataset only contains images from the specified class
        IDs. If `relabel_classes` is True, the labels of the classes
        are converted to 0...N-1 where N is the number of classes in
        each split dataset.

    """
    datasets = []

    for id_list in class_id_lists:
        # each id_set would create a new dataset
        new_dset = dataset.filter(
            lambda x: tf.reduce_any(tf.math.equal(x['label'], id_list))
        )

        if relabel_classes:
            # convert the class ID to 0...N-1 for each task
            to_new_label = {
                cls_id : idx for idx, cls_id in enumerate(id_list)
            }
            def convert(sample):
                label = sample['label']
                for old_label, new_label in to_new_label.items():
                    label = tf.where(
                        tf.math.equal(label, old_label),
                        new_label,
                        tf.cast(label, tf.int32),
                    )
                sample['label'] = label
                return sample

            new_dset = new_dset.map(convert)

        datasets.append(new_dset)

    return datasets


def to_stream_dataset(dataset, n_classes, p=0.5, seed=0):
    """Converts a dataset to a streaming dataset with recurring classes

    At any time, the probability of the class of the next sample is a
    function of when the class was last seen. The more recent a class
    was seen, the more likely it is to show up next.

    The class that showed up at time t-n has probability p**n to
    show up at time t. p has to be < 0.5 since p + p^2 + .. equals 1 if
    p = 0.5.

    Since the series is limited (n doesn't really approach infinity),
    whatever probability is left after p + p^2 + ... + p^t is equally
    distributed among all classes.

    Parameters
    ----------
    dataset : tf.data.Dataset
        A classification dataset, each sample has key `image` and
        `label`.
    n_classes : int
        The number of classes in the dataset.

    Returns
    -------
    new_dataset : tf.data.Dataset

    """
    #dataset = dataset.shuffle(SHUFFLE_BUFFER)

    def make_dataset_gen():
        for sample in dataset:
            yield sample


    def make_stream_dataset_gen():
        # initialize the cache and the probability
        class_cache = defaultdict(list)
        class_probs = np.zeros(n_classes, dtype=np.float32)
        gone_classes = []  # keep track of classes that are all used
        dataset_gen = make_dataset_gen()

        rand_state = np.random.RandomState(seed)

        end_of_dataset = False
        while True:
            # first calculate the probabilities of the classes
            cur_total = np.sum(class_probs)
            rest = max(
                (1.0 - cur_total) / (n_classes - len(gone_classes)),
                EPS
            )
            cur_prob = class_probs + rest
            # set the classes that are gone to 0 prob
            cur_prob[gone_classes] = 0.0

            # sample
            # input needs to have the batch dimension
            sample_class = rand_state.choice(
                a=n_classes,
                p=cur_prob,
            )

            # go through the dataset until there's something of the
            # desired class
            while(
                not end_of_dataset and
                len(class_cache[sample_class]) == 0
            ):
                try:
                    sample = next(dataset_gen)
                except StopIteration:
                    end_of_dataset = True
                    break

                cur_class = sample['label'].numpy()
                class_cache[cur_class].append(sample)

            # when there's no more samples in the dataset, and we're
            # out of the desired class, set the probability of the
            # class to 0 and repeat
            if len(class_cache[sample_class]) == 0:
                class_probs[sample_class] = 0
                gone_classes.append(sample_class)

                # if all classes are gone, break
                if len(gone_classes) == n_classes:
                    break

                # else continue
                continue

            # Note that pop() pops the end of the list
            yield class_cache[sample_class].pop()

            # now update the probabilities
            class_probs *= p
            class_probs[sample_class] += p

    type_spec_dict = dataset._type_spec._element_spec
    output_types = {k: v.dtype for k, v in type_spec_dict.items()}
    output_shapes = {k: v.shape for k, v in type_spec_dict.items()}

    stream_dataset = tf.data.Dataset.from_generator(
        generator=make_stream_dataset_gen,
        output_types=output_types,
        output_shapes=output_shapes,
    )
    return stream_dataset


def to_stream_train_val_pair(train_dset, val_dset, n_classes, p=0.5):
    updated_train_dset = to_stream_dset(train_dset)
    # use a validation sample that's the same as the training sample
    # for each pair
    # TODO: I'll worry about this later - I've been mainly looking at
    # the training curves anyway.
