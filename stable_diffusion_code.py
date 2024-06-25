# Made by @StealthyPanda
# Don't use this for nefarious purposes, I am not liable for anything! 


# This code uses tensorflow, make sure it is downloaded with GPU options.
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import json

from typing import Literal
from tensorflow import keras



#Chekcking for GPU...
if tf.config.list_physical_devices('GPU'): print('Using GPU!')
else: print('NOT Using GPU! Make sure this is not unintended...')




# This function give you a poistional embedding matrix, whose rows are each time vectors.
def positional_embedding(time_steps : int, d_embed : int, N : int = 1e+4) -> tf.Tensor:
    embs = tf.range(0, d_embed // 2, 1, dtype = tf.float32)
    embs = tf.expand_dims(embs, -1)
    embs = tf.tile(embs, (1, 2))
    embs = tf.reshape(embs, (-1))
    embs = tf.expand_dims(embs, -1)
    embs = tf.tile(embs, (1, time_steps))
    
    
    embs = embs * (2 / d_embed)
    embs = tf.pow(N, -embs)
    
    embs = embs.numpy()
    for each in range(time_steps):
        embs[:, each] *= each
    
    for each in range(d_embed):
        if each % 2: embs[each] = np.cos(embs[each])
        else: embs[each] = np.sin(embs[each])
    
    embs = tf.constant(embs)
    embs = tf.transpose(embs)
    
    return embs


# Calculating constants

# This returns a list of betas, currently it returns a simple linear beta
def betas(lower : float = 1e-4, upper : float = 0.02, steps : int = 1e3) -> tf.Tensor:
    return tf.linspace(lower, upper, int(steps))

def alphabars(betavals : tf.Tensor) -> tf.Tensor:
    alphas = 1.0 - betavals
    alphabars = tf.math.cumprod(alphas)
    return alphabars

# This takes in the inital no noise image, and outputs
def noiser(init_img : tf.Tensor, alphabarsvals : tf.Tensor, t : int) -> tuple[tf.Tensor, tf.Tensor]:
    noise = tf.random.normal(init_img.shape)
    noisey = (tf.math.sqrt(alphabarsvals[t]) * init_img) + (tf.math.sqrt(1 - alphabarsvals[t]) * noise)
    return noise, noisey

# Clamps x to any range
def clamp(x, lower : float = 0, upper : float = 1):
    minimum, maximum = tf.reduce_min(x), tf.reduce_max(x)
    x = ((x - minimum) * (upper - lower) / (maximum - minimum)) + lower
    return x


def sqrtrecipalphas(alphas : tf.Tensor) -> tf.Tensor:
    return tf.math.sqrt(1.0 / alphas)

def sqrtalphabar(alphabars : tf.Tensor) -> tf.Tensor:
    return tf.math.sqrt(alphabars)

def sqrtoneminusalphabar(alphabars : tf.Tensor) -> tf.Tensor:
    return tf.math.sqrt(1 - alphabars)

def posteriorvariance(betas : tf.Tensor, alphabars) -> tf.Tensor:
    return betas * (1 - tf.pad(alphabars[:-1], [[1, 0]], constant_values=1)) / (1 - alphabars)









#Makes the batch for a single generation
def get_batch(
        data : tf.Tensor | np.ndarray, 
        beta_min : float = 1e-2,
        beta_max : float = 0.05,
        time_steps : int = 1e3,
        batch_size : int = 128,
        nimages : int = 16,
    ) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
    
    B = betas(lower=beta_min, upper=beta_max, steps=time_steps)
    A = alphabars(B)
    steps = int(time_steps)
    
    times = np.random.randint(0, steps, size = (batch_size,))
    
    length = data.shape[0]
    
    imgs = np.tile(data[np.random.randint(0, length, size = (1,))], (batch_size // nimages, 1, 1, 1))
    for each in range(1, nimages):
        imgs = np.concatenate((
            imgs,
            np.tile(data[np.random.randint(0, length, size = (1,))], (batch_size // nimages, 1, 1, 1))
        ), axis = 0)
    
    noises = np.zeros(imgs.shape)
    noiseys = np.zeros(imgs.shape)
    for each in range(batch_size):
        noises[each], noiseys[each] = noiser(imgs[each], A, times[each])
    
    
    
    return (noiseys, times), noises, A, B


#A single block from UNet architecture.
class UBlock(keras.layers.Layer):
    
    def __init__(
            self, channels : int, direction : Literal['up' , 'down'], ksize : tuple[int, int] = (3, 3),
            activation = keras.activations.relu,
            trainable=True, name=None, dtype=None, dynamic=False, **kwargs
        ):
        
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        
        
        self.channels = channels
        self.ksize = ksize
        self.activation = activation
        
        if direction == 'down':
            self.c1 = keras.layers.Conv2D(channels, (3, 3), activation = self.activation, padding='same')
            self.final = keras.layers.Conv2D(channels, (4, 4), (2, 2), padding='same')
        elif direction == 'up':
            self.c1 = keras.layers.Conv2DTranspose(channels, ksize, activation = self.activation, padding='same')
            self.final = keras.layers.Conv2DTranspose(channels, (4, 4), (2, 2), padding='same')
        else:
            raise TypeError(f'Invalid direction {direction}!')
        
        self.c2 = keras.layers.Conv2D(channels, (3, 3), activation = self.activation, padding='same')
        
        self.time_projector = keras.layers.Dense(channels, use_bias = True, activation=self.activation)
        
        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization()
        
        
    
    def call(self, x, t):
        # x, t = inputs
        x # b h h ci
        t # b d_embed
        
        t = (self.time_projector(t)) # b ci
        t = tf.expand_dims(t, 1) # b 1 ci
        t = tf.expand_dims(t, 1) # b 1 1 ci
        
        x = (self.c1(x))
        x = self.bn1(x)
        
        t1 = tf.tile(t, (1, x.shape[1], x.shape[2], 1)) # b x.height x.width ci
        x = x + t1
        
        x = (self.c2(x))
        x = self.bn2(x)
        
        x = self.final(x)
        
        return x


#Constructs a U-Net model
def UNet(inputshape, channels : list[int], d_embed : int) -> keras.Model:
    
    x = keras.Input(inputshape)
    inputlayer = x
    
    t = keras.Input((d_embed,))
    timeinput = t
    
    t = keras.layers.Dense(d_embed, activation=keras.activations.relu)(t)
    
    x = keras.layers.Conv2D(channels[0], (3, 3), padding='same')(x)
    
    checkpoints = []
    
    for each in channels:
        x = UBlock(each, direction='down')(x, t)
        checkpoints.append(x)
    
    
    for i, each in enumerate(channels[::-1]):
        # x = keras.layers.Concatenate(-1)((x, checkpoints.pop()))
        x = x + checkpoints.pop()
        x = UBlock(each // 2, 'up')(x, t) 
    
    x = keras.layers.Conv2D(inputshape[-1], (1, 1))(x)
    
    
    model = keras.Model(inputs = [inputlayer, timeinput], outputs = x)
    return model





def train(
        model : keras.Model, dataset : tf.Tensor, timesteps : int, pos_enc : tf.Tensor,
        generations : int = 10, epochs : int = 50, verbose : bool = False,
    ):
    
    losses = []
    pointers = '⣾⣽⣻⢿⡿⣟⣯⣷'
    # pointers = "⠄⠆⠇⠋⠙⠸⠸⠰⠠⠰⠸⠙⠋⠇⠆"
    # pointers = ('⢀⠀⡀⠀⠄⠀⢂⠀⡂⠀⠅⠀⢃⠀⡃⠀⠍⠀⢋⠀⡋⠀⠍⠁⢋⠁⡋⠁⠍⠉⠋⠉⠋⠉⠉⠙⠉⠙⠉⠩⠈⢙⠈⡙⢈⠩⡀⢙⠄⡙⢂⠩⡂' +
    # '⢘⠅⡘⢃⠨⡃⢐⠍⡐⢋⠠⡋⢀⠍⡁⢋⠁⡋⠁⠍⠉⠋⠉⠋⠉⠉⠙⠉⠙⠉⠩⠈⢙⠈⡙⠈⠩⠀⢙⠀⡙⠀⠩⠀⢘⠀⡘⠀⠨⠀⢐⠀⡐⠀⠠⠀⢀⠀⡀')
    def callback(gen, i, val):
        print(f'\rGeneration {gen + 1}/{generations} {i+1}/{epochs} {val["loss"]:.4f} {pointers[i % len(pointers)]}...', end='')
    
    for each in range(generations):
        if verbose: print(f'\n\nGeneration {each + 1}/{generations}...', )
        
        n_unique_image = 128
        p = 32
        (bimgs, btimes), bnoises, A, B = get_batch(
            dataset,
            beta_min = 1e-3,
            beta_max = 0.02,
            time_steps = timesteps,
            batch_size = n_unique_image * p,
            nimages = n_unique_image
        )
        btimesvects = np.array([pos_enc[x] for x in btimes])
        
        hist = model.fit(
            (bimgs, btimesvects), bnoises,
            epochs = epochs,
            batch_size = n_unique_image,
            verbose = verbose,
            callbacks = [
                keras.callbacks.LambdaCallback(
                    on_epoch_end = lambda i, val : callback(each, i, val)
                )
            ]
        )
        
        losses += (hist.history['loss'])
    return losses, bimgs, btimes, btimesvects, bnoises, A, B


# Denoises the given image, and returns it.
def denoise(
        image : np.ndarray | tf.Tensor, model : keras.Model, t : int, betas : tf.Tensor, alphabars : tf.Tensor,
        pos_enc : tf.Tensor
    ) -> tf.Tensor:
    alphas = 1 - betas
    sqrt_one_minus_alphabars = sqrtoneminusalphabar(alphabars)
    sqrt_recip_alphas = sqrtrecipalphas(alphas)
    pvs = posteriorvariance(betas, alphabars)
    
    tvect = pos_enc[t:t+1]
    
    mean = sqrt_recip_alphas[t] * (
        image - ( (betas[t] * model.predict((image, tvect), verbose=False)) / sqrt_one_minus_alphabars[t] )
    )
    
    if t == 0: return mean
    else:
        return mean + ( tf.math.sqrt(pvs[t]) * tf.random.normal(image.shape) )


# Plots the progression of pure noise to an output image.
def sample_plot_image(time_steps : int, nimgs : int, model : keras.Model, betas : tf.Tensor, alphabars : tf.Tensor):
    img = tf.random.normal([1, 64, 64, 3])
    
    stepsize = int(time_steps/nimgs)

    fig, ax = plt.subplots(1, 10, figsize = (20, 2))

    for t in range(0, time_steps)[::-1]:
        img = denoise(img, model, t, betas, alphabars)
        img = clamp(img, -1, 1)
        print(t, stepsize, end='\r')
        if t % stepsize == 0:
            ax[nimgs - (t // stepsize) - 1].imshow(clamp(img[0]))
            plt.axis('off')
    fig.show()
