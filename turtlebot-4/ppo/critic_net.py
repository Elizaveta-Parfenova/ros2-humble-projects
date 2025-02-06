import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import cv2
import random
import math
import matplotlib.pyplot as plt


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


def compute_deviation_from_path(current_pos, optimal_path):

    path_points = np.array(optimal_path)
    distances = np.linalg.norm(path_points - np.array(current_pos), axis=1)
    min_distance = np.min(distances)
    return min_distance


class ImprovedCritic(tf.keras.Model):
    def __init__(self, state_dim, grid_map, optimal_path, n_neurons=512):
        super(ImprovedCritic, self).__init__()
        # self.bn1 = layers.BatchNormalization()  
        # Преобразуем grid_map в тензор
        self.grid_map = (
            tf.convert_to_tensor(grid_map, dtype=tf.float32)
            if not isinstance(grid_map, tf.Tensor)
            else grid_map
        )

        # Преобразуем optimal_path в тензор и разворачиваем в одномерный вектор
        self.optimal_path = (
            tf.convert_to_tensor(optimal_path, dtype=tf.float32)
            if not isinstance(optimal_path, tf.Tensor)
            else optimal_path
        )
        self.optimal_path = tf.reshape(self.optimal_path, [-1])  # Делаем 1D
        
        self.rb1 = ResBlock(state_dim + 1, state_dim + 1, n_neurons)  # +1 для отклонения от пути
        self.rb2 = ResBlock((state_dim + 1) * 2, (state_dim + 1) * 2, n_neurons)
        self.dropout = layers.Dropout(rate=0.1)  
        self.out = layers.Dense(1, activation=None, kernel_initializer='he_uniform')

    def call(self, obs, deviation_from_path, collision_penalty=0, training=True):
        obs_with_deviation = tf.concat([obs, [deviation_from_path + collision_penalty]], axis=-1)
        
        if isinstance(obs, np.ndarray):
            obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        if isinstance(deviation_from_path, (float, int)):
            deviation_from_path = tf.convert_to_tensor([deviation_from_path], dtype=tf.float32)

        # print(deviation_from_path)
        # Объединяем состояние с отклонением
        obs_with_deviation = tf.concat([obs, deviation_from_path], axis=-1)

        # print(obs_with_deviation)

        if tf.math.reduce_any(tf.math.is_nan(obs_with_deviation)) or tf.math.reduce_any(tf.math.is_inf(obs_with_deviation)):
            print("Обнаружены некорректные значения (NaN или Inf) в данных:", obs_with_deviation)
            return tf.constant([[float('nan')]])

        if len(obs_with_deviation.shape) == 1:
            obs_with_deviation = tf.expand_dims(obs_with_deviation, axis=0)

        x0 = obs_with_deviation
        # x0 = self.bn1(x0)  
        x = self.rb1(x0, final_nl=True)
        x = self.rb2(tf.concat([x0, x], axis=-1), final_nl=True)

        x = self.dropout(x, training=training)  # Применяем Dropout
        output = self.out(x)
        return output

    def eval_value(self, state):
        # Вычисляем отклонение от оптимального пути
        state = state[0]
        current_pos = (state[0], state[1])
        deviation = compute_deviation_from_path(current_pos, self.optimal_path)
        
        # Проверяем, есть ли столкновение
        collision_penalty = 0
        if self.is_near_obstacle(current_pos):
            collision_penalty = 1e3  # Большой штраф за близость к препятствию

        return self.call(state, deviation, collision_penalty)

    def is_near_obstacle(self, point, safe_distance=2):
        x, y = int(point[0]), int(point[1])
        h, w = self.grid_map.shape

        # Проверка области вокруг точки на наличие препятствий
        x_min = max(0, x - safe_distance)
        x_max = min(w, x + safe_distance)
        y_min = max(0, y - safe_distance)
        y_max = min(h, y + safe_distance)

        area = self.grid_map[y_min:y_max, x_min:x_max]
        return np.any(area == 1)
    
