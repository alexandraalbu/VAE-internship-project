from init_variables import *


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def conv2d_transpose(x, w, output_shape):
    return tf.nn.conv2d_transpose(x, w, tf.stack(output_shape), strides=[1, 2, 2, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def encoder(x, dim_latent, activation, width, height, nr_channels, dim_hidden=-1):
    x_image = tf.reshape(x, [-1, height, width, nr_channels])

    w_conv1 = weight_variable('w_conv1', [3, 3, nr_channels, 32])  # !!! <32
    b_conv1 = bias_variable([32])
    h_conv1 = activation(conv2d(x_image, w_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    w_conv2 = weight_variable('w_conv2', [7, 7, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = activation(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    w_conv3 = weight_variable('w_conv3', [5, 5, 64, 128])
    b_conv3 = bias_variable([128])
    h_conv3 = activation(conv2d(h_pool2, w_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    h_flat = tf.reshape(h_pool3, [-1, width * height * 2])  # width/8 * height/8 * 128
    w_fc11 = weight_variable('w_fcl', [width * height * 2, 2 * dim_latent])
    b_fc11 = bias_variable([2 * dim_latent])
    # h_fcl1 = activation(tf.matmul(h_flat, w_fc11) + b_fc11)

    # h_fc1_drop = tf.nn.dropout(h_fcl1, keep_prob)
    #
    # w_fcl2 = weight_variable('w_fcl2', [50, 2 * dim_latent])  # 2 fully connected layers: no
    # b_fcl2 = bias_variable([2 * dim_latent])

    z = tf.matmul(h_flat, w_fc11) + b_fc11

    return z[:, :dim_latent], z[:, dim_latent:]


def decoder(z, dim_latent, activation, width, height, nr_channels, dim_hidden=-1):
    w_dec_fcl1 = weight_variable('w_dec_fcl', [dim_latent, width * height * 2])
    b_dec_fcl1 = bias_variable([width * height * 2])
    h1 = activation(tf.matmul(z, w_dec_fcl1) + b_dec_fcl1)

    # h_drop = tf.nn.dropout(h1, keep_prob)
    #
    # w_dec_fcl2 = weight_variable('w_dec_fcl2', [50, width * height * 2])
    # b_dec_fcl2 = bias_variable([height * width * 2])
    # h2 = activation(tf.matmul(h_drop, w_dec_fcl2) + b_dec_fcl2)

    h1_reshaped = tf.reshape(h1, [-1, 4, 4, 128])
    batch_size = h1_reshaped.shape[0]

    w_conv1 = weight_variable('w_dec_conv1', [5, 5, 64, 128])
    b_conv1 = bias_variable([64])
    h_conv1 = activation(conv2d_transpose(h1_reshaped, w_conv1, [batch_size, 8, 8, 64]) + b_conv1)

    w_conv2 = weight_variable('w_dec_conv2', [7, 7, 32, 64])
    b_conv2 = bias_variable([32])
    h_conv2 = activation(conv2d_transpose(h_conv1, w_conv2, [batch_size, 16, 16, 32]) + b_conv2)

    w_conv3 = weight_variable('w_dec_conv3', [7, 7, nr_channels, 32])
    b_conv3 = bias_variable([nr_channels])

    mean = conv2d_transpose(h_conv2, w_conv3, [batch_size, 32, 32, 3]) + b_conv3
    return tf.reshape(mean, [-1, height * width * nr_channels])
