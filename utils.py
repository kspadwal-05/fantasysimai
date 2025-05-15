import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

def mdn_loss(n_components):
    def loss(y_true, params):
        pi, mu, sigma = params
        cat = tfp.distributions.Categorical(probs=pi)
        comp = tfp.distributions.Normal(loc=mu, scale=sigma)
        gmm = tfp.distributions.MixtureSameFamily(mixture_distribution=cat, components_distribution=comp)
        return -tf.reduce_mean(gmm.log_prob(y_true))
    return loss

def mdn_sample(pi, mu, sigma, n):
    indices = np.random.choice(len(pi), size=n, p=pi)
    samples = np.random.normal(loc=mu[indices], scale=sigma[indices])
    return samples