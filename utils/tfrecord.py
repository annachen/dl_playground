import tensorflow as tf
import os
from functools import partial
from enum import Enum
import numpy as np
import time


def bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  if type(value) == str:
    value = value.encode('utf-8')
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  """Returns a bytes_list from a list / array of string / byte."""
  values = []
  for v in value:
    if isinstance(v, type(tf.constant(0))):
      v = v.numpy() # BytesList won't unpack a string from an EagerTensor.
    if type(v) == str:
      v = v.encode('utf-8')

    values.append(v)
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
  """Returns a float_list from a list /array of float / double."""
  # from https://www.tensorflow.org/tutorials/load_data/tfrecord
  # it sounds like I should use tf.io.serialize_tensor to serialize
  # tensors into byte strings?
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  """Returns an int64_list from a list / array of bool / enum / int /
  uint.

  """
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def image_feature(image):
    """Returns a bytes_list which is the PNG-encoded image."""
    encoded = tf.io.encode_png(image)
    return bytes_feature(encoded)


def image_list_feature(images):
    codes = [tf.io.encode_png(im) for im in images]
    return bytes_list_feature(codes)


class FeatureType(Enum):
    BYTES = 1
    BYTES_LIST = 2
    FLOAT = 3
    FLOAT_LIST = 4
    INT = 5
    INT_LIST = 6
    IMAGE = 7
    IMAGE_LIST = 8

    # arrays
    FLOAT_ARRAY = 9
    INT_ARRAY = 10

    # ragged
    FLOAT_RAGGED_ARRAY = 11
    INT_RAGGED_ARRAY = 12

    # sparse
    FLOAT_SPARSE_ARRAY = 13
    INT_SPARSE_ARRAY = 14


LIST_FEATURE_TYPES = set([
    FeatureType.BYTES_LIST,
    FeatureType.FLOAT_LIST,
    FeatureType.INT_LIST,
])


ARRAY_FEATURE_TYPES = set([
    FeatureType.FLOAT_ARRAY,
    FeatureType.INT_ARRAY,
])


RAGGED_FEATURE_TYPES = set([
    FeatureType.FLOAT_RAGGED_ARRAY,
    FeatureType.INT_RAGGED_ARRAY,
])


SPARSE_FEATURE_TYPES = set([
    FeatureType.FLOAT_SPARSE_ARRAY,
    FeatureType.INT_SPARSE_ARRAY,
])


WRITE_FEATURE_FN = {
    FeatureType.BYTES: bytes_feature,
    FeatureType.BYTES_LIST: bytes_list_feature,
    FeatureType.FLOAT: float_feature,
    FeatureType.FLOAT_LIST: float_list_feature,
    FeatureType.INT: int64_feature,
    FeatureType.INT_LIST: int64_list_feature,
    FeatureType.IMAGE: image_feature,
    FeatureType.IMAGE_LIST: image_list_feature,
    FeatureType.FLOAT_ARRAY: float_list_feature,
    FeatureType.INT_ARRAY: int64_list_feature,
    FeatureType.FLOAT_RAGGED_ARRAY: float_list_feature,
    FeatureType.INT_RAGGED_ARRAY: int64_list_feature,
    FeatureType.FLOAT_SPARSE_ARRAY: float_list_feature,
    FeatureType.INT_SPARSE_ARRAY: int64_list_feature,
}


READ_FEATURE_FN = {
    FeatureType.BYTES: partial(
        tf.io.FixedLenFeature, shape=[], dtype=tf.string
    ),
    FeatureType.BYTES_LIST: partial(
        tf.io.VarLenFeature, dtype=tf.string,
    ),
    FeatureType.FLOAT: partial(
        tf.io.FixedLenFeature, shape=[], dtype=tf.float32
    ),
    FeatureType.FLOAT_LIST: partial(
        tf.io.VarLenFeature, dtype=tf.float32
    ),
    FeatureType.INT: partial(
        tf.io.FixedLenFeature, shape=[], dtype=tf.int64
    ),
    FeatureType.INT_LIST: partial(
        tf.io.VarLenFeature, dtype=tf.int64
    ),
    FeatureType.IMAGE: partial(
        tf.io.FixedLenFeature, shape=[], dtype=tf.string
    ),
    FeatureType.IMAGE_LIST: partial(
        tf.io.VarLenFeature, dtype=tf.string
    ),
    FeatureType.FLOAT_ARRAY: partial(
        tf.io.FixedLenFeature, dtype=tf.float32,
    ),
    FeatureType.INT_ARRAY: partial(
        tf.io.FixedLenFeature, dtype=tf.int64,
    ),
    FeatureType.FLOAT_RAGGED_ARRAY: partial(
        tf.io.VarLenFeature, dtype=tf.float32
    ),
    FeatureType.INT_RAGGED_ARRAY: partial(
        tf.io.VarLenFeature, dtype=tf.int64
    ),
    FeatureType.FLOAT_SPARSE_ARRAY: partial(
        tf.io.VarLenFeature, dtype=tf.float32,
    ),
    FeatureType.INT_SPARSE_ARRAY: partial(
        tf.io.VarLenFeature, dtype=tf.int64,
    )
}


class TFRecordConverter:
    """Convenient class for converting TFRecord and python dict.

    Parameters
    ----------
    feature_def : dict (str -> FeatureType)
        A dictionary defining all the features to read/write.
        The keys are the names of the features.
        The values are the types of the features.
    feature_shape : dict (str -> [int])
        For any array type features, the shape of the arrays must be
        provided (at reading time).
    ragged_rank : dict (str -> int)
        For any ragged type features, pass in the ragged rank of the
        array.

    """
    def __init__(self, feature_def, feature_shape=None, ragged_rank=None):
        self._feature_def = feature_def
        self._feature_shape = feature_shape
        self._ragged_rank = ragged_rank

    def to_example(self, data):
        # data: str -> int | float | np.array
        feats = {}
        for k, v in data.items():
            assert k in self._feature_def

            feature_type = self._feature_def[k]
            write_feat_fn = WRITE_FEATURE_FN[feature_type]

            if feature_type in ARRAY_FEATURE_TYPES:
                feats[k] = write_feat_fn(v.flatten())
                feats[k + '_shape'.format(k)] = int64_list_feature(
                    v.shape
                )

            elif feature_type in RAGGED_FEATURE_TYPES:
                rank = self._ragged_rank[k]
                t = tf.ragged.constant(v)
                # a tuple with `rank` items; each item a variable len
                # integer tensor
                nested = t.nested_row_lengths()

                # each item in the tuple will be written to its own
                # feature field
                for r in range(rank):
                    feats[k + '_nested_len_' + str(r)] = (
                        int64_list_feature(nested[r].numpy())
                    )

                feats[k] = write_feat_fn(t.flat_values.numpy())

            elif feature_type in SPARSE_FEATURE_TYPES:
                # currently only supports 2D arrays
                # v is a coo_matrix
                feats[k] = write_feat_fn(v.data)
                indices = np.stack([v.row, v.col], axis=-1)
                feats[k + '_indices'] = int64_list_feature(
                    indices.flatten()
                )
                feats[k + '_shape'] = int64_list_feature(v.shape)
                feats[k + '_len'] = int64_feature(indices.shape[0])

            else:
                feats[k] = write_feat_fn(v)

        return tf.train.Example(
            features=tf.train.Features(feature=feats)
        )

    def from_example(self, example):
        feature_map = {}
        for k, v in self._feature_def.items():
            feature_type = self._feature_def[k]
            read_feat_fn = READ_FEATURE_FN[feature_type]

            if feature_type in ARRAY_FEATURE_TYPES:
                feature_map[k] = read_feat_fn(
                    shape=self._feature_shape[k]
                )
                feature_map[k + '_shape'] = READ_FEATURE_FN[
                    FeatureType.INT_LIST
                ]()
            elif feature_type in RAGGED_FEATURE_TYPES:
                feature_map[k] = read_feat_fn()

                rank = self._ragged_rank[k]
                for r in range(rank):
                    feature_map[k + '_nested_len_' + str(r)] = (
                        READ_FEATURE_FN[FeatureType.INT_LIST]()
                    )

            elif feature_type in SPARSE_FEATURE_TYPES:
                feature_map[k] = read_feat_fn()
                feature_map[k + '_indices'] = READ_FEATURE_FN[
                    FeatureType.INT_LIST
                ]()
                feature_map[k + '_shape'] = READ_FEATURE_FN[
                    FeatureType.INT_LIST
                ]()
                feature_map[k + '_len'] = READ_FEATURE_FN[
                    FeatureType.INT
                ]()
            else:
                feature_map[k] = read_feat_fn()

        tensor_dict = tf.io.parse_single_example(example, feature_map)

        data = {}
        for k in self._feature_def.keys():
            feature_type = self._feature_def[k]

            if feature_type in LIST_FEATURE_TYPES:
                data[k] = tf.sparse.to_dense(tensor_dict[k])

            elif feature_type in RAGGED_FEATURE_TYPES:
                rank = self._ragged_rank[k]
                nested_lengths = []
                for r in range(rank):
                    nested_lengths.append(
                        tf.sparse.to_dense(
                            tensor_dict[k + '_nested_len_' + str(r)]
                        )
                    )
                    #print(nested_lengths[-1].numpy())

                values = tf.sparse.to_dense(tensor_dict[k])
                #print(values.numpy())

                data[k] = tf.RaggedTensor.from_nested_row_lengths(
                    flat_values=values,
                    nested_row_lengths=nested_lengths,
                    #validate=False,
                )

            elif feature_type in SPARSE_FEATURE_TYPES:
                dense_shape = tf.sparse.to_dense(
                    tensor_dict[k + '_shape']
                )
                n_elems = tensor_dict[k + '_len']
                indices = tf.reshape(
                    tf.sparse.to_dense(
                        tensor_dict[k + '_indices']
                    ), (n_elems, -1)
                )
                values = tf.sparse.to_dense(tensor_dict[k])
                sp = tf.sparse.SparseTensor(
                    indices=indices,
                    values=values,
                    dense_shape=dense_shape,
                )
                data[k] = sp

            elif feature_type == FeatureType.IMAGE:
                data[k] = tf.io.decode_png(tensor_dict[k])

            elif feature_type == FeatureType.IMAGE_LIST:
                tensor = tf.sparse.to_dense(tensor_dict[k])
                data[k] = tf.map_fn(
                    fn=tf.io.decode_png,
                    elems=tensor,
                    fn_output_signature=tf.TensorSpec(
                        shape=[None, None, None],
                        dtype=tf.uint8
                    )
                )

            else:
                data[k] = tensor_dict[k]

        return data


class ShardedTFRecordWriter:
    def __init__(self, output_folder, examples_per_file):
        self._output_folder = output_folder
        self._examples_per_file = examples_per_file
        self._ex_cnt = 0
        self._file_cnt = 0
        self._writer = None

        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

    def write(self, example):
        if self._writer is None:
            self.create_new_writer()
        elif self._ex_cnt == self._examples_per_file:
            self.create_new_writer()

        self._writer.write(example.SerializeToString())
        self._ex_cnt += 1

    def create_new_writer(self):
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()

        filename = os.path.join(
            self._output_folder, '{:04d}.tfr'.format(self._file_cnt)
        )
        self._writer = tf.io.TFRecordWriter(path=filename)
        self._file_cnt += 1
        self._ex_cnt = 0

    def flush(self):
        self._writer.flush()

    def close(self):
        self._writer.close()
