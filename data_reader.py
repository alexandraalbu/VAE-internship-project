from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets.base import maybe_download
import numpy
import collections
import tensorflow as tf
from tensorflow.python.framework import dtypes, random_seed
import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
import CIFAR10


flags = tf.app.flags
flags.DEFINE_string('data_dir_mnist', '../tmp/mnist', 'Data directory')
flags.DEFINE_string('data_dir_frey', '../tmp/frey_faces', 'Data directory')
flags.DEFINE_string('data_dir_svhn', '../tmp/svhn_data', 'Data directory')
FLAGS = flags.FLAGS

Datasets = collections.namedtuple('Datasets', ['train', 'test'])

# this class is a slight modification of the one available at
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py
class DataSet(object):
    def __init__(self, images, fake_data=False, one_hot=False, dtype=dtypes.float32, seed=None):
        """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
        seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed is returned
        numpy.random.seed(seed1 if seed is None else seed2)
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)
        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            self._num_examples = images.shape[0]
            if dtype == dtypes.float32:
                # Convert from [0, 255] -> [0.0, 1.0].
                images = images.astype(numpy.float32)
                images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def set_images(self, images):
        self._images = images

    @property
    def images(self):
        return self._images

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * 784
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in range(batch_size)], [
                fake_label for _ in range(batch_size)
            ]
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm0)
            self._images = self.images[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._images = self.images[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            return numpy.concatenate((images_rest_part, images_new_part), axis=0), None
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], None


def read_data_set(name):
    if name == 'mnist':
        return input_data.read_data_sets(FLAGS.data_dir_mnist), 28, 28, 1
    elif name == 'frey_faces':
        maybe_download('frey_rawface.mat', FLAGS.data_dir_frey, 'http://www.cs.nyu.edu/~roweis/data/frey_rawface.mat')
        images = sio.loadmat(FLAGS.data_dir_frey + '/frey_rawface.mat', squeeze_me=True)
        img_rows, img_cols = 28, 20
        n_pixels = img_rows * img_cols

        images = images["ff"].T.reshape((-1, img_rows, img_cols))
        train_images, test_images = train_test_split(images, test_size=0.185)

        train_images = train_images.reshape((-1, n_pixels))
        test_images = test_images.reshape((-1, n_pixels))

        train = DataSet(train_images, dtype=dtypes.float32, seed=None)
        test = DataSet(test_images, dtype=dtypes.float32, seed=None)
        return Datasets(train=train, test=test), 20, 28, 1
    elif name == 'svhn':
        maybe_download('train_32x32.mat', FLAGS.data_dir_svhn, 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat')
        train_images = sio.loadmat(FLAGS.data_dir_svhn + '/train_32x32.mat')['X']
        train_images = np.transpose(train_images, [3, 0, 1, 2])
        train_images = np.reshape(train_images, [-1, 32*32*3])

        maybe_download('test_32x32.mat', FLAGS.data_dir_svhn, 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat')
        test_images = sio.loadmat(FLAGS.data_dir_svhn + '/test_32x32.mat')['X']
        test_images = np.transpose(test_images, [3, 0, 1, 2])
        test_images = np.reshape(test_images, [-1, 32 * 32 * 3])

        train = DataSet(train_images, dtype=dtypes.float32, seed=None)
        test = DataSet(test_images, dtype=dtypes.float32, seed=None)
        return Datasets(train=train, test=test), 32, 32, 3
    elif name == 'cifar10':
        ds = CIFAR10.loadCIFAR10(8)
        train = DataSet(ds['train_set'], dtype=dtypes.float32, seed=None)
        test = DataSet(ds['test_set'], dtype=dtypes.float32, seed=None)
        return Datasets(train=train, test=test), 8, 8, 3
    elif name == 'cifar10_full':
        ds = CIFAR10.loadCIFAR10(32)
        train = DataSet(ds['train_set'], dtype=dtypes.float32, seed=None)
        test = DataSet(ds['test_set'], dtype=dtypes.float32, seed=None)
        return Datasets(train=train, test=test), 32, 32, 3
    else:
        print('No such data set')
