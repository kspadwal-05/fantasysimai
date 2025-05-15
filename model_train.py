import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Must be set before importing tensorflow

import pickle
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks
import tensorflow.keras.backend as K
import numpy as np

# Load data
with open('training_data.pkl', 'rb') as f:
    X, y = pickle.load(f)

n_features = X.shape[1]
n_components = 5

# MDN loss function
def mdn_loss(n_components):
    def loss(y_true, y_pred):
        # y_pred shape: (batch_size, n_components * 3)
        # Slice outputs
        pi_logits = y_pred[:, :n_components]            # Mixture weights logits
        mu = y_pred[:, n_components:2*n_components]     # Means
        sigma_logits = y_pred[:, 2*n_components:3*n_components]  # Sigma logits

        # Softmax over pi logits for mixture weights
        pi = tf.nn.softmax(pi_logits)

        # Sigma must be positive, use softplus or exponential
        sigma = tf.math.exp(sigma_logits) + 1e-6  # add epsilon for numerical stability

        # Expand dims for broadcasting
        y_true_exp = tf.expand_dims(y_true, axis=1)  # shape (batch, 1)

        # Compute probability of y_true under each Gaussian component
        # Gaussian PDF: 1/(sqrt(2pi)*sigma) * exp(-0.5*((y - mu)/sigma)^2)
        norm = 1.0 / (sigma * tf.sqrt(2.0 * np.pi))
        exponent = tf.exp(-0.5 * tf.square((y_true_exp - mu) / sigma))
        component_pdf = norm * exponent  # shape (batch, n_components)

        # Weighted sum over components (mixture)
        weighted_pdf = pi * component_pdf  # shape (batch, n_components)
        prob = tf.reduce_sum(weighted_pdf, axis=1)  # shape (batch,)

        # Negative log likelihood
        nll = -tf.math.log(prob + 1e-8)  # add epsilon for numerical stability

        return tf.reduce_mean(nll)
    return loss

# Build model
inp = layers.Input(shape=(n_features,))
net = layers.Dense(128, activation='relu')(inp)
net = layers.Dense(128, activation='relu')(net)
out = layers.Dense(n_components * 3)(net)  # Single output layer for pi, mu, sigma

model = Model(inputs=inp, outputs=out)

model.compile(optimizer=optimizers.Adam(1e-3), loss=mdn_loss(n_components))

# TensorBoard callback
tb = callbacks.TensorBoard(log_dir='logs')

# Train model
model.fit(
    X, y,
    epochs=80,
    batch_size=64,
    validation_split=0.1,
    callbacks=[tb],
    verbose=2
)

# Save model
model.save('mdn_model.keras')
print('Model saved.')
