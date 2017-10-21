from vae import VariationalAutoencoder
import tensorflow as tf
import data_reader
import matplotlib.pyplot as plt
import argparse
from sklearn.decomposition import PCA


def scatter_plot_latent_space(z_mu, y, file_path):
    plt.scatter(z_mu[:, 0], z_mu[:, 1], c=y)
    plt.colorbar()
    plt.grid()
    plt.savefig(file_path)
    plt.close()


def plot_latent_space_pca_mean(vae, x, y):
    Z = vae.sess.run(vae.z_mu, feed_dict={vae.x: x})

    pca_2 = PCA(n_components=2)
    z_pca = pca_2.fit_transform(Z)
    scatter_plot_latent_space(z_pca, y, result_dir + 'latent_space_pca_mean')
    # actual implementation of PCA:
    # _, _, V = np.linalg.svd(Z, full_matrices=False)
    # z = Z.dot(V[:2, :].T)
    # scatter_plot_latent_space(z, y, result_dir + 'latent_space_pca_mean')


def restore_model(vae, checkpoint_path):
    saver = tf.train.Saver()
    saver.restore(vae.sess, checkpoint_path)
    return vae


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='mnist',
                        help='dataset name, choose from: mnist, frey_faces, cifar10, cifar10_full, svhn')
    parser.add_argument('--vae_type', default='flat_binarized', help='choose from: flat_binarized'
                                                                     'flat_cont_normal, flat_cont_logit, conv3')
    parser.add_argument('--num_epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--hidden_size', type=int, default=200, help='hidden size of network layers')

    args = parser.parse_args()
    data_set_name = args.dataset
    vae_type = args.vae_type
    result_dir = './pictures/new_pictures/'

    input_values, width, height, nr_channels = data_reader.read_data_set(data_set_name)

    v20 = VariationalAutoencoder(width, height, nr_channels, dim_latent=20, input_values=input_values,
                                 type=vae_type, learning_rate=args.learning_rate, batch_size=args.batch_size,
                                 dim_hidden=args.hidden_size)

    model_path = '../checkpoints/' + data_set_name + '_' + vae_type + '_20'
    v20 = restore_model(v20, model_path)
    plot_latent_space_pca_mean(v20, input_values.test.images, input_values.test.labels)

