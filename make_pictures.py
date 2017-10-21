from vae import VariationalAutoencoder
import tensorflow as tf
import data_reader
import matplotlib.pyplot as plt
import argparse
import numpy as np


def generate(vae):
    grid = np.empty(grid_shape)
    for i in range(n_pictures):
        for j in range(n_pictures):
            z = np.random.normal(size=(vae.batch_size, vae.dim_latent))
            output = vae.sess.run(vae.get_output(), feed_dict={vae.z: z})
            grid[i * height: (i + 1) * height, j * width: (j + 1) * width] = output[0].reshape(image_shape)
    plt.imshow(grid, cmap=color_map)
    plt.axis('off')
    plt.savefig(result_dir + 'generate_' + data_set_name + '_' + vae.type)
    plt.close()


def multiple_reconstructions(vae):
    x_sample, _ = input_values.test.next_batch(vae.batch_size)
    z_mu, z_log_sigma = vae.sess.run([vae.z_mu, vae.z_log_sigma], feed_dict={vae.x: x_sample})
    grid = np.empty(grid_shape)
    for i in range(n_pictures):
        grid[i * height: (i + 1) * height, 0: width] = x_sample[i].reshape(image_shape)
        for j in range(1, n_pictures):
            z = np.random.normal(loc=z_mu[i], scale=np.exp(z_log_sigma[i]), size=(vae.batch_size, vae.dim_latent))
            output = vae.sess.run(vae.get_output(), feed_dict={vae.z: z})
            grid[i * height: (i + 1) * height, j * width: (j + 1) * width] = output[0].reshape(image_shape)
    plt.imshow(grid, cmap=color_map)
    plt.axis('off')
    plt.axvline(x=width, color='r')
    plt.savefig(result_dir + 'm_reconstr_' + data_set_name + '_' + vae.type)
    plt.close()


def interpolate_latent_space(vae):
    n_columns = 20
    alpha_range = np.linspace(0, 1, n_columns)
    z = np.empty((0, vae.dim_latent))
    for row in range(5):
        x_sample, _ = input_values.test.next_batch(2)
        z_mu, z_log_sigma = vae.sess.run([vae.z_mu, vae.z_log_sigma], feed_dict={vae.x: x_sample})
        for alpha in alpha_range:
            new_z_mu = z_mu[0] * alpha + z_mu[1] * (1 - alpha)
            new_z_sigma = np.exp(z_log_sigma[0]) * alpha + np.exp(z_log_sigma[1]) * (1 - alpha)
            new_z = np.random.normal(loc=new_z_mu, scale=new_z_sigma, size=(1, vae.dim_latent))
            z = np.concatenate([z, new_z])
    output = vae.sess.run(vae.get_output(), feed_dict={vae.z: z})
    for i in range(5):
        for j in range(n_columns):
            current = i * n_columns + j
            plt.subplot(5, n_columns, current+1)
            plt.imshow(output[current].reshape(image_shape), cmap=color_map)
            plt.axis('off')
    plt.show()


def restore_model(vae, checkpoint_path):
    saver = tf.train.Saver()
    saver.restore(vae.sess, checkpoint_path)
    return vae

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='mnist',
                        help='dataset name, choose from: mnist, frey_faces, cifar10, cifar10_full, svhn')
    parser.add_argument('--vae_type', default='flat_binarized', help='')
    parser.add_argument('--latent_dim', type=int, default=20, help='latent dimensions')
    parser.add_argument('--num_epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--hidden_size', type=int, default=200, help='hidden size of network layers')

    args = parser.parse_args()
    data_set_name = args.dataset
    vae_type = args.vae_type
    result_dir = './pictures/new_pictures/'

    input_values, width, height, nr_channels = data_reader.read_data_set(data_set_name)
    image_shape = height, width
    color_map = 'gray'
    n_pictures = 10
    grid_shape = (height*n_pictures, width*n_pictures)

    if data_set_name in ['svhn', 'cifar10', 'cifar10_full']:
        image_shape = height, width, 3
        color_map = None
        grid_shape = (height * n_pictures, width * n_pictures, 3)

    v = VariationalAutoencoder(width, height, nr_channels, dim_latent=args.latent_dim, input_values=input_values,
                               type=vae_type, learning_rate=args.learning_rate, batch_size=args.batch_size,
                               dim_hidden=args.hidden_size)
    model_path = '../checkpoints/' + data_set_name + '_' + vae_type + '_' + str(v.dim_latent)
    v = restore_model(v, model_path)
    generate(v)
    multiple_reconstructions(v)
    interpolate_latent_space(v)
