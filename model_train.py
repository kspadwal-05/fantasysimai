import pickle
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks
from utils import mdn_loss

# Load data
with open('training_data.pkl','rb') as f:
    X, y = pickle.load(f)

n_features = X.shape[1]
n_components = 5

# Build MDN network
inp = layers.Input(shape=(n_features,))
net = layers.Dense(128, activation='relu')(inp)
net = layers.Dense(128, activation='relu')(net)
pi = layers.Dense(n_components, activation='softmax', name='pi')(net)
mu = layers.Dense(n_components, name='mu')(net)
sigma = layers.Dense(n_components, activation='exponential', name='sigma')(net)
model = Model(inputs=inp, outputs=[pi,mu,sigma])
model.compile(optimizer=optimizers.Adam(1e-3), loss=mdn_loss(n_components))

# Train with TensorBoard logging
tb = callbacks.TensorBoard(log_dir='logs')
model.fit(X,y,epochs=80,batch_size=64,validation_split=0.1,callbacks=[tb],verbose=2)
model.save('mdn_model')
print('Model saved')