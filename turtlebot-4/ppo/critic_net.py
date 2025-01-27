import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class ResBlock(tf.keras.Model):
    def __init__(self, input_dim, output_dim, n_neurons=512):
        super(ResBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = layers.Dense(n_neurons, activation=None, kernel_initializer='he_uniform')
        # self.bn1 = layers.BatchNormalization()  
        self.activation = layers.LeakyReLU(negative_slope=0.2)

        self.fc2 = layers.Dense(output_dim, activation=None, kernel_initializer='he_uniform')
        # self.bn2 = layers.BatchNormalization()  

        if input_dim != output_dim:
            self.fc_res = layers.Dense(output_dim, activation=None, kernel_initializer='he_uniform')
        else:
            self.fc_res = None

    def call(self, x, final_nl=True):
        residual = x
        if self.fc_res:
            residual = self.fc_res(residual)
            residual = self.activation(residual)  # Применяем активацию к пути выравнивания

        x = self.fc1(x)
        # x = self.bn1(x)  
        x = self.activation(x)

        x = self.fc2(x)
        # x = self.bn2(x)  

        x += residual
        if final_nl:
            x = self.activation(x)
        return x


class ImprovedCritic(tf.keras.Model):
    def __init__(self, state_dim, n_neurons=512):
        super(ImprovedCritic, self).__init__()
        # self.bn1 = layers.BatchNormalization()  
        self.rb1 = ResBlock(state_dim, state_dim, n_neurons)
        self.rb2 = ResBlock(state_dim + state_dim, state_dim + state_dim, n_neurons)
        self.dropout = layers.Dropout(rate=0.1)  
        self.out = layers.Dense(1, activation=None, kernel_initializer='he_uniform')

    def call(self, obs, training=True):
        if isinstance(obs, np.ndarray):
            obs = tf.convert_to_tensor(obs, dtype=tf.float32)

        x0 = obs
        # x0 = self.bn1(x0)  
        x = self.rb1(x0, final_nl=True)
        x = self.rb2(tf.concat([x0, x], axis=-1), final_nl=True)

        x = self.dropout(x, training=training)  # Применяем Dropout
        output = self.out(x)
        return output
