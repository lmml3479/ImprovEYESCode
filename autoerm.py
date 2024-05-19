import os
import argparse
import numpy as np
from typing import Tuple
import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv3D, Conv3DTranspose, Dropout, Input
from tensorflow.keras.layers import Flatten, ReLU, LeakyReLU, BatchNormalization, Reshape
from sklearn.cluster import KMeans
from pymatgen.io.cif import CifWriter
from pymatgen.core import Lattice, Structure
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.ext.matproj import MPRester
from m3gnet.models import M3GNet
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import _Merge
from tensorflow.keras import backend as K

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, help="Path to where the dataset is stored.")
parser.add_argument("--save_path", type=str, help="Path to where the models and loss history is stored.")
parser.add_argument("--dir", type=str, help="Path to where crystals are.")
parser.add_argument("--m3gnet_model_path", type=str, help="Path to where M3GNet model is.")
parser.add_argument("--ehull_path", type=str, help="Path to where energy above calculations will be stored.")
parser.add_argument("--mp_api_key", type=str, help="API key for materials project.")

def conv_norm(x: keras.engine.keras_tensor.KerasTensor, units: int, 
              filter: Tuple[int, int, int], stride : Tuple[int, int, int], 
              discriminator: bool = True
              ) -> keras.engine.keras_tensor.KerasTensor:
  if discriminator:
    activation_function = LeakyReLU(alpha = 0.2)
    conv = Conv3D(units, filter, strides = stride, padding = 'valid')
  else:
    activation_function = ReLU()
    conv = Conv3DTranspose(units, filter, strides = stride, padding = 'valid')
  x = conv(x)
  x = BatchNormalization()(x)
  x = activation_function(x) 
  return x

def dense_norm(x: keras.engine.keras_tensor.KerasTensor, units: int, 
               discriminator: bool) -> keras.engine.keras_tensor.KerasTensor:
  if discriminator:
    activation_function = LeakyReLU(alpha = 0.2)
  else:
    activation_function = ReLU()
  x = Dense(units)(x)
  x = BatchNormalization()(x)
  x = activation_function(x)
  return x

def define_discriminator(in_shape: Tuple[int, int, int, int] = (64, 64, 4, 1)) -> keras.engine.functional.Functional:
    tens_in = Input(shape=in_shape, name="input")
    y = Flatten()(tens_in)
    y = dense_norm(y, 1024, True) 
    y = dense_norm(y, 1024, True)
    y = dense_norm(y, 1024, True)
    y = dense_norm(y, 1024, True)
    x = conv_norm(tens_in, 32, (1,1,1), (1,1,1), True)
    x = conv_norm(x, 32, (3,3,1), (1,1,1), True)  
    x = conv_norm(x, 32, (3,3,1), (1,1,1), True)  
    x = conv_norm(x, 32, (3,3,1), (1,1,1), True)  
    x = conv_norm(x, 64, (3,3,1), (1,1,1), True) 
    x = conv_norm(x, 64, (7,7,1), (1,1,1), True)
    x = conv_norm(x, 64, (7,7,1), (1,1,1), True)
    x = conv_norm(x, 64, (7,7,1), (1,1,1), True)
    x = conv_norm(x, 64, (7,7,1), (1,1,1), True)
    z = Reshape((32, 32, 1, 1))(y)
    x = z + x
    y = dense_norm(y, 9, True) 
    x = conv_norm(x, 128, (5,5,2), (5,5,1), True)
    x = conv_norm(x, 256, (2,2,2), (2,2,2), True)  
    z = Reshape((3, 3, 1, 1))(y)
    x = z + x
    x = Flatten()(x)
    x = Dropout(0.25)(x)
    disc_out = Dense(1, activation = "sigmoid")(x)
    model = Model(inputs=tens_in, outputs=disc_out)
    opt = Adam(learning_rate = 1e-5)
    model.compile(loss = 'binary_crossentropy', optimizer = opt,metrics = ['accuracy'])
    return model

def define_generator(latent_dim: int) -> keras.engine.functional.Functional:
    n_nodes = 16 * 16 * 4
    noise_in = Input(shape=(latent_dim, ), name="noise_input")
    hull_in = Input(shape=(latent_dim, ), name="ehull_input")
    y = dense_norm(noise_in, 484, False)
    y = dense_norm(y, 484, False)
    x = dense_norm(noise_in, n_nodes, False)
    x = Reshape((16,16, 4, 1))(x)
    x = conv_norm(x, 256, (3,3,3), (1,1,1), False)
    x = conv_norm(x, 128, (3,3,3), (1,1,1), False)
    x = conv_norm(x, 128, (3,3,3), (1,1,1), False)
    z = Reshape((22, 22, 1, 1))(y)
    x = z + x
    y = dense_norm(y, 784, False)
    y = dense_norm(y, 784, False)
    y = dense_norm(y, 784, False)
    x = conv_norm(x, 128, (3,3,3), (1,1,1), False)
    x = conv_norm(x, 64, (3,3,3), (1,1,1), False) 
    x = conv_norm(x, 64, (3,3,3), (1,1,1), False) 
    z = Reshape((28, 28, 1, 1))(y)
    x = z + x
    y = dense_norm(y, 1024, False)
    y = dense_norm(y, 1024, False)
    x = conv_norm(x, 64, (3,3,3), (1,1,1), False) 
    x = conv_norm(x, 64, (3,3,3), (1,1,1), False) 
    z = Reshape((32, 32, 1, 1))(y)
    x = z + x
    y = dense_norm(y, 4096, False)
    x = conv_norm(x, 32, (2,2,2), (2,2,2), False)   
    z = Reshape((64, 64, 1, 1))(y)
    x = z + x
    outMat = Conv3D(1,(1,1,10), activation = 'sigmoid', strides = (1,1,10), padding = 'valid')(x)
    model = Model(inputs=[noise_in, hull_in], outputs=outMat)
    return model

class NoiseGenerator(object):
    def __init__(self, noise_shapes, batch_size=512, random_seed=None):
        self.noise_shapes = noise_shapes
        self.batch_size = batch_size
        self.prng = np.random.RandomState(seed=random_seed)

    def __iter__(self):
        return self

    def __next__(self, mean=0.0, std=1.0):

        def noise(shape):
            shape = (self.batch_size, shape)

            n = self.prng.randn(*shape).astype(np.float32)
            if std != 1.0:
                n *= std
            if mean != 0.0:
                n += mean
            return n

        return [noise(s) for s in self.noise_shapes]

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred, axis=-1)

class RandomWeightedAverage(_Merge):
    def build(self, input_shape):
        super(RandomWeightedAverage, self).build(input_shape)
        if len(input_shape) != 2:
            raise ValueError('A `RandomWeightedAverage` layer should be '
                             'called on a list of 2 inputs')

    def call(self, inputs, **kwargs):
        weights = K.random_uniform((K.shape(inputs[0])[0], 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

def define_gan(generator, discriminator, latent_dim):
    discriminator.trainable = False
    gen_noise, gen_hull = generator.input
    gen_output = generator.output
    gan_output = discriminator(gen_output)
    model = Model(inputs=[gen_noise, gen_hull], outputs=gan_output)
    opt = Adam(lr=1e-5)
    model.compile(loss=wasserstein_loss, optimizer=opt)
    return model

def train_discriminator(model, dataset, n_steps, n_batch, half_batch, k):
    # Unroll generator for k steps
    for i in range(k):
        [X_real, y_real] = dataset.next()
        noise, ehull = generate_latent_points(latent_dim, half_batch)
        X_fake = generator.predict([noise, ehull])
        y_fake = -np.ones((half_batch, 1))
        model.train_on_batch(X_real, y_real)
        model.train_on_batch(X_fake, y_fake)

    # Train discriminator
    [X_real, y_real] = dataset.next()
    noise, ehull = generate_latent_points(latent_dim, half_batch)
    X_fake = generator.predict([noise, ehull])
    y_fake = -np.ones((half_batch, 1))
    d_loss1 = model.train_on_batch(X_real, y_real)
    d_loss2 = model.train_on_batch(X_fake, y_fake)
    return d_loss1, d_loss2

def train(generator, discriminator, gan, dataset, latent_dim, n_epochs=100, n_batch=128, k=5):
    half_batch = int(n_batch / 2)
    for i in range(n_epochs):
        d_loss1, d_loss2 = train_discriminator(discriminator, dataset, n_steps=1, n_batch=n_batch, half_batch=half_batch, k=k)
        noise, ehull = generate_latent_points(latent_dim, n_batch)
        y_gan = np.ones((n_batch, 1))
        g_loss = gan.train_on_batch([noise, ehull], y_gan)
        print(f'Epoch {i+1}, d1={d_loss1}, d2={d_loss2}, g={g_loss}')

if __name__ == "__main__":
    args = parser.parse_args()
    latent_dim = 100
    dataset = NoiseGenerator([64*64*4, latent_dim])
    discriminator = define_discriminator()
    generator = define_generator(latent_dim)
    gan = define_gan(generator, discriminator, latent_dim)
    train(generator, discriminator, gan, dataset, latent_dim)
