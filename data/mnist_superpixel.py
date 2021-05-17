import fire
import skimage.segmentation
import skimage.color
import numpy as np
import os
import tensorflow as tf
from functools import partial

from artistcritic.utils.superpixel import segmentation_to_graph
from artistcritic.utils import tfrecord


# Note: these are manually tuned for MNIST
N_SEGS = [100, 50, 17, 6]
BASE = 2.8 * 2  + 1
RADII = [
    BASE,
    BASE * np.sqrt(2),
    BASE * np.sqrt(2) * np.sqrt(3),
    BASE * np.sqrt(2) * np.sqrt(3) * np.sqrt(3),
]

FEATURE_MAP = None

MAX_NODES = [0] * len(N_SEGS)
MAX_NEIGHBORS = [0] * len(N_SEGS)
MAX_CHILDREN = [0] * len(N_SEGS)


def superpixel_graphs_generator(dataset):
    """Generates superpixel graphs for a dataset

    Parameters
    ----------
    dataset : tf.data.Dataset

    Returns
    -------
    generator : Generator
        yields (dict, [Graph], [np.array]), where `dict` is the
        corresponding original dictionary from the input dataset,
        and the np.array is the pixel-wise segmentation map

    """
    global MAX_NODES, MAX_NEIGHBORS, MAX_CHILDREN
    for sample in dataset:
        im = sample['image'].numpy()
        if im.shape[-1] == 1:
            im = skimage.color.gray2rgb(im)

        graphs = []
        segs = []
        prev_seg = None
        for level_idx, n_seg in enumerate(N_SEGS):
            # (H, W, 1)
            seg = skimage.segmentation.slic(
                image=im,
                n_segments=n_seg,
                convert2lab=True,
                start_label=1,
            )
            graph = segmentation_to_graph(seg, RADII[level_idx])
            graphs.append(graph)
            segs.append(seg)

            if level_idx > 0:
                graph = set_center_children(graph, prev_seg)
                graph = add_children(
                    graph, graphs[level_idx - 1], seg
                )
                set_parent(graphs[level_idx - 1], seg)

            prev_seg = seg

            # update the max counts
            if graph.n_nodes > MAX_NODES[level_idx]:
                MAX_NODES[level_idx] = graph.n_nodes
            if graph.max_neighbors > MAX_NEIGHBORS[level_idx]:
                MAX_NEIGHBORS[level_idx] = graph.max_neighbors
            if graph.max_children > MAX_CHILDREN[level_idx]:
                MAX_CHILDREN[level_idx] = graph.max_children

        yield (sample, graphs, segs)


def _int64_to_rgb(seg_map):
    # Converts an int64 map into 3-channel RGB so we can PNG-encode it
    r = seg_map % 256
    seg_map = seg_map // 256
    g = seg_map % 256
    seg_map = seg_map // 256
    b = seg_map % 256
    return np.concatenate([r, g, b], axis=-1).astype(np.uint8)


def _rgb_to_int64(seg_map):
    seg_map = tf.cast(seg_map, tf.int64)
    return (
        seg_map[..., 2] * (256 ** 2) +
        seg_map[..., 1] * 256 +
        seg_map[..., 0]
    )


def set_center_children(graph, prev_seg):
    """For each superpixel in the graph, set the center children from
    the previous layer.

    The center childer is the previous superpixel within which the
    current superpixel's center lies in.

    Parameters
    ----------
    graph : Graph
    prev_seg : np.array, shape (H, W, 1)

    Returns
    -------
    graph : Graph
        Note that `graph` is updated in place.

    """
    for node in graph.nodes.values():
        int_center = np.round(node._center).astype(np.int32)
        prev_label = prev_seg[int_center[0], int_center[1], 0]
        node.set_center_child(prev_label)

    return graph


def add_children(graph, prev_graph, cur_seg):
    """Adds children labels to the graph.

    A child is the previous superpixel whose center pixel is covered
    by the higher level superpixel.

    """
    for node in prev_graph.nodes.values():
        int_center = np.round(node._center).astype(np.int32)
        cur_label = cur_seg[int_center[0], int_center[1], 0]
        graph.nodes[cur_label].add_child(node._label)
        if type(node._label) != int:
            print(type(node._label))
        #print('###')
        #print(graph.nodes[cur_label]._child_labels)

    return graph


def set_parent(graph, next_seg):
    """For each superpixel in the graph, set the parent label at the
    next layer.

    The parent is the next superpixel which contains the center of
    the current superpixel.

    Parameters
    ----------
    graph : Graph
    next_seg : np.array, shape (H, W, 1)

    """
    for node in graph.nodes.values():
        int_center = np.round(node._center).astype(np.int32)
        next_label = next_seg[int_center[0], int_center[1], 0]
        node.set_parent(next_label)


def graph_list_to_features(graphs):
    """Get a TFRecord feature dictionary from a list of graphs.

    The features from each graph is prepended with a prefix and all
    added to the final dictionary at the same level.

    Parameters
    ----------
    graphs : [Graph]

    Returns
    -------
    features : dict (str -> Feature)

    """
    features = {}

    for i in range(len(graphs)):
        prefix = 'g{}_'.format(i)
        graph_feats = graphs[i].to_features()
        updated = {(prefix) + k : v for (k, v) in graph_feats.items()}
        features.update(updated)

    return features


def seg_list_to_features(segs):
    features = {}
    for i in range(len(segs)):
        key = 'seg{}'.format(i)
        features[key] = tfrecord.image_feature(_int64_to_rgb(segs[i]))
    return features


def run(output_folder, examples_per_file=1000):
    # Importing here to avoid circular import
    from artistcritic.data.datasets import load_dataset
    for split in ['train', 'test']:
        print("Running {} set".format(split))
        writer = tfrecord.ShardedTFRecordWriter(
            output_folder=os.path.join(output_folder, split),
            examples_per_file=examples_per_file,
        )
        dset = load_dataset('mnist', split)
        generator = superpixel_graphs_generator(dset)
        cnt = 0
        for sample, graphs, segs in generator:
            features = graph_list_to_features(graphs)

            seg_feats = seg_list_to_features(segs)
            features.update(seg_feats)

            features['image'] = tfrecord.image_feature(
                sample['image']
            )
            features['label'] = tfrecord.int64_feature(
                sample['label']
            )
            example = tf.train.Example(
                features=tf.train.Features(feature=features)
            )
            writer.write(example)
            cnt += 1

            if cnt % 1000 == 0:
                print("Finished {} samples".format(cnt))
                print("Current max:")
                print("max_nodes: {}".format(MAX_NODES))
                print("max_neighbors: {}".format(MAX_NEIGHBORS))
                print("max_children: {}".format(MAX_CHILDREN))

        writer.flush()
        writer.close()

        print("Finished {} samples".format(cnt))
        print("Current max:")
        print("max_nodes: {}".format(MAX_NODES))
        print("max_neighbors: {}".format(MAX_NEIGHBORS))
        print("max_children: {}".format(MAX_CHILDREN))

        info = {
            'max_nodes': MAX_NODES,
            'max_neighbors': MAX_NEIGHBORS,
            'max_children': MAX_CHILDREN,
        }
        with open(os.path.join(output_folder, 'info.yaml')) as f:
            yaml.dump(info, f)


def _generate_feature_map():
    """Lazily generates the feature map for reading TFRecord."""
    global FEATURE_MAP
    if FEATURE_MAP is not None:
        return

    n_graphs = len(N_SEGS)
    feats = {}

    for i in range(n_graphs):
        prefix = 'g{}_'.format(i)
        cur_f = {
            prefix + 'n_nodes': tf.io.FixedLenFeature([], tf.int64),
            prefix + 'neighbors': tf.io.VarLenFeature(tf.int64),
            prefix + 'n_neighbors': tf.io.VarLenFeature(tf.int64),
            prefix + 'center': tf.io.VarLenFeature(tf.float32),
            prefix + 'mass': tf.io.VarLenFeature(tf.float32),
            prefix + 'label': tf.io.VarLenFeature(tf.int64),
            prefix + 'center_child': tf.io.VarLenFeature(tf.int64),
            #prefix + 'parent': tf.io.VarLenFeature(tf.int64),
            #prefix + 'children': tf.io.VarLenFeature(tf.int64),
            #prefix + 'n_children': tf.io.VarLenFeature(tf.int64),
        }
        feats.update(cur_f)

        #seg_key = 'seg{}'.format(i)
        #feats[seg_key] = tf.io.FixedLenFeature([], tf.string)

    feats['image'] = tf.io.FixedLenFeature([], tf.string)
    feats['label'] = tf.io.FixedLenFeature([], tf.int64)

    FEATURE_MAP = feats


def _to_dense(sp_tensor):
    return tf.sparse.to_dense(sp_tensor)


def graphs_from_example(example):
    """Reads from a TFRecord example.

    Parameters
    ----------
    example : ? (an example for iterating over a TFRecordDataset)

    Returns
    -------
    ret_dict : dict
        "image": tf.Tensor, shape (H, W, C)
        "label": tf.int64
        "seg{n}": tf.Tensor, shape (H, W, 1)
        "{prefix}n_nodes"
        "{prefix}neighbors"
        "{prefix}n_neighbors"
        "{prefix}center"
        "{prefix}mass"
        "{prefix}label"
        "{prefix}center_child"
        "{prefix}parent"
        "{prefix}children"
        "{prefix}n_children"
        where {prefix} is "g<n>_" and n is from 0 to n_graphs - 1

    """
    _generate_feature_map()
    td = tf.io.parse_single_example(example, FEATURE_MAP)
    graph_dict = {}
    for i in range(len(N_SEGS)):
        prefix = 'g{}_'.format(i)
        graph = {}
        graph[prefix + 'n_nodes'] = td[prefix + 'n_nodes']
        graph[prefix + 'neighbors'] = tf.reshape(
            _to_dense(td[prefix + 'neighbors']),
            (graph[prefix + 'n_nodes'], -1)
        )
        graph[prefix + 'n_neighbors'] = _to_dense(
            td[prefix + 'n_neighbors']
        )
        graph[prefix + 'center'] = tf.reshape(
            _to_dense(td[prefix + 'center']),
            (graph[prefix + 'n_nodes'], 2)
        )
        graph[prefix + 'mass'] = _to_dense(td[prefix + 'mass'])
        graph[prefix + 'label'] = _to_dense(td[prefix + 'label'])
        graph[prefix + 'center_child'] = _to_dense(
            td[prefix + 'center_child']
        )
        #graph[prefix + 'parent'] = _to_dense(td[prefix + 'parent'])
        #graph[prefix + 'children'] = tf.reshape(
        #    _to_dense(td[prefix + 'children']),
        #    (graph[prefix + 'n_nodes'], -1)
        #)
        #graph[prefix + 'n_children'] = _to_dense(
        #    td[prefix + 'n_children']
        #)
        graph_dict.update(graph)

        #seg_key = 'seg{}'.format(i)
        #decoded = tf.io.decode_png(td[seg_key])
        #graph_dict[seg_key] = _rgb_to_int64(decoded)

    ret_dict = {
        'image': tf.io.decode_png(td['image']),
        'label': td['label'],
    }
    ret_dict.update(graph_dict)
    return ret_dict


def pad_to_max_nodes_and_neighbors(sample, max_nodes, max_neighbors):
    """Pads the sample so it can be batched.

    Parameter
    ---------
    sample : dict
        The output of `graphs_from_example`
    max_nodes : [int]
        The maximum number of nodes to consider for each layer.
    max_neighbors : [int]
        The maximum number of neighbors to consider for each layer.

    Returns
    -------
    ret_dict : dict

    """
    graph_dict = sample
    for idx in range(len(N_SEGS)):
        prefix = 'g{}_'.format(idx)

        # pad to max_nodes
        n_nodes = tf.shape(graph_dict[prefix + 'center'])[0]
        to_pad = tf.maximum(max_nodes[idx] - n_nodes, 0)
        for data_type in ['neighbors', 'center']:
            # These are 2D arrays
            key = prefix + data_type
            graph_dict[key] = tf.pad(
                graph_dict[key],
                [[0, to_pad], [0, 0]],
            )
            graph_dict[key] = graph_dict[key][:max_nodes[idx]]

        for data_type in [
            'n_neighbors', 'mass', 'label', 'center_child'
        ]:
            # These are 1D arrays
            key = prefix + data_type
            graph_dict[key] = tf.pad(
                graph_dict[key],
                [[0, to_pad]],
            )
            graph_dict[key] = graph_dict[key][:max_nodes[idx]]

        # pad to max_neighbors
        key = prefix + 'neighbors'
        n_neighbors = tf.shape(graph_dict[key])[1]
        to_pad = tf.maximum(max_neighbors[idx] - n_neighbors, 0)
        graph_dict[key] = tf.pad(
            graph_dict[key],
            [[0, 0], [0, to_pad]],
        )
        graph_dict[key] = graph_dict[key][:, :max_neighbors[idx]]

    return graph_dict


def get_dataset(dataset_path, max_nodes, max_neighbors):
    """Returns a tf Dataset that can be padded."""
    files = os.listdir(dataset_path)
    files = [os.path.join(dataset_path, f) for f in files]
    dset = tf.data.TFRecordDataset(files)
    dset = dset.map(graphs_from_example)

    _pad = partial(
        pad_to_max_nodes_and_neighbors,
        max_nodes=max_nodes,
        max_neighbors=max_neighbors
    )
    dset = dset.map(_pad)

    return dset


if __name__ == '__main__':
    fire.Fire(run)
