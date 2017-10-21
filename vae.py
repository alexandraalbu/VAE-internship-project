import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import network
import network_conv_3
import network_normal

network_architecture = {'flat_binarized': {'encoder': network.encoder, 'decoder': network.decoder},
                        'flat_cont_logit': {'encoder': network_normal.encoder, 'decoder': network_normal.decoder},
                        'flat_cont_normal': {'encoder': network_normal.encoder, 'decoder': network_normal.decoder},
                        'conv3': {'encoder': network_conv_3.encoder, 'decoder': network_conv_3.decoder}}


class VariationalAutoencoder:
    def __init__(self, width, height, nr_channels, input_values, type, dim_latent=20, dim_hidden=500,
                 activation=tf.nn.relu, batch_size=100, learning_rate=0.001):
        self.input = input_values
        self.activation = activation
        self.dim_latent = dim_latent
        self.dim_hidden = dim_hidden
        self.dim_input = width * height * nr_channels
        self.batch_size = batch_size
        self.type = type

        self.x = tf.placeholder(tf.float32, [None, self.dim_input])

        with tf.variable_scope("vae" + str(self.dim_latent)):
            self.z_mu, self.z_log_sigma = network_architecture[type]['encoder'](
                self.x, dim_latent, activation, width, height, nr_channels, dim_hidden=self.dim_hidden)
            self.z = self.draw_sample()
            self.output = network_architecture[type]['decoder'](
                self.z, dim_latent, activation, width, height, nr_channels, dim_hidden=self.dim_hidden)

        self.reconstr_err = tf.reduce_mean(self.reconstruction_error())
        self.kl_div = tf.reduce_mean(self.kl_divergence())

        self.loss = self.reconstr_err + self.kl_div

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def draw_sample(self):
        epsilon = tf.random_normal(shape=(self.batch_size, self.dim_latent))
        return self.z_mu + tf.exp(self.z_log_sigma) * epsilon  # reparameterization trick

    def logit(self):
        return tf.log(self.x) - tf.log(1 - self.x)

    def reconstruction_error(self):
        if self.type == 'flat_binarized' or self.type == 'conv3':
            return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels=self.x), 1)
        elif self.type == 'flat_cont_logit':
            dist = tf.contrib.distributions.MultivariateNormalDiag(loc=self.output[0],
                                                                   scale_diag=tf.exp(self.output[1])+0.5)
            return -dist.log_prob(self.logit()) + tf.reduce_sum(tf.log(self.x) + tf.log(1 - self.x), 1)
        elif self.type == 'flat_cont_normal':
            dist = tf.contrib.distributions.Normal(loc=tf.nn.sigmoid(self.output[0]),
                                                   scale=tf.exp(self.output[1]) + 0.5)  # 0.1 for frey faces
            return -tf.reduce_sum(dist.log_prob(self.x), 1)

    def kl_divergence(self):
        return -0.5 * tf.reduce_sum(1 + 2 * self.z_log_sigma - tf.square(self.z_mu) - tf.exp(2 * self.z_log_sigma), 1)

    def get_output(self):
        if self.type == 'flat_binarized' or self.type == 'conv3':
            return tf.nn.sigmoid(self.output)
        else:
            return tf.nn.sigmoid(self.output[0])

    def train(self, n_epochs=200):
        num_examples = self.input.train.num_examples
        for epoch_i in range(n_epochs):
            print('\t Epoch', epoch_i)
            avg_err = avg_kl_div = 0
            for i in range(num_examples // self.batch_size):
                x, _ = self.input.train.next_batch(self.batch_size)
                if self.type == 'flat_binarized':
                    x = np.random.binomial(1, x)
                elif self.type == 'flat_cont_logit':
                    x = np.clip(x, 1e-2, 1-1e-2)
                _, err, kl = self.sess.run((self.optimizer, self.reconstr_err, self.kl_div), feed_dict={self.x: x})
                avg_err += err * self.batch_size / num_examples
                avg_kl_div += kl * self.batch_size / num_examples
            print('Reconstruction error {}, KL divergence {}'.format(avg_err, avg_kl_div))

    def plot_errors_train_test(self, n_epochs=30):
        train_num_examples = self.input.train.num_examples
        test_num_examples = self.input.test.num_examples
        train_errors = []
        test_errors = []
        for epoch_i in range(n_epochs):
            avg_error = 0
            print('\t Epoch', epoch_i)
            for i in range(train_num_examples // self.batch_size):
                x, _ = self.input.train.next_batch(self.batch_size)
                _, loss = self.sess.run((self.optimizer, self.loss), feed_dict={self.x: x})
                avg_error += loss * self.batch_size / train_num_examples
            train_errors.append(avg_error)
            print('Average loss: ', avg_error)

            avg_error = 0
            for i in range(test_num_examples // self.batch_size):
                x, _ = self.input.test.next_batch(self.batch_size)
                loss = self.sess.run(self.loss, feed_dict={self.x: x})
                avg_error += loss * self.batch_size / test_num_examples
            test_errors.append(avg_error)

        plt.plot(list(range(n_epochs)), train_errors, marker='x', color='b', linestyle='-')
        plt.plot(list(range(n_epochs)), test_errors, marker='x', color='y', linestyle='-')
