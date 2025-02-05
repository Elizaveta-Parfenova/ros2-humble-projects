import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import cv2
import random
import math
import matplotlib.pyplot as plt

def world_to_map(world_coords, resolution, origin):
    x_world, y_world = world_coords
    x_map = int((x_world - origin[0]) / resolution)
    y_map = int((y_world - origin[1]) / resolution)
    return (x_map, y_map)

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

def slam_to_grid_map(slam_map, threshold=50):
    """
    Преобразование карты SLAM в бинарную grid map.
    Args:
        slam_map (np.ndarray): Исходная карта от SLAM.
        threshold (int): Порог для определения препятствий.
    Returns:
        np.ndarray: Бинарная grid map (0 - свободно, 1 - препятствие).
    """
    grid_map = np.where(slam_map < threshold, 1, 0)  # 1 - препятствия, 0 - свободно
    num_obstacles = np.count_nonzero(grid_map == 1)

    # print(num_obstacles)
    
    # # Визуализация grid_map
    # plt.figure(figsize=(8, 8))
    # plt.imshow(grid_map, cmap='gray')
    # plt.title(f'Grid Map с порогом {threshold}')
    # plt.axis('off')
    # plt.show()
    
    return grid_map

def compute_deviation_from_path(current_pos, optimal_path):
    """
    Вычисляем отклонение от ближайшей точки оптимального пути.
    Args:
        current_pos (tuple): Текущая позиция агента (x, y).
        optimal_path (list): Список координат оптимального пути.
    Returns:
        float: Отклонение от пути.
    """
    path_points = np.array(optimal_path)
    distances = np.linalg.norm(path_points - np.array(current_pos), axis=1)
    min_distance = np.min(distances)
    return min_distance


class ImprovedCritic(tf.keras.Model):
    def __init__(self, state_dim, n_neurons=512):
        super(ImprovedCritic, self).__init__()
        # self.bn1 = layers.BatchNormalization()  
        self.rb1 = ResBlock(state_dim + 1, state_dim + 1, n_neurons)  # +1 для отклонения от пути
        self.rb2 = ResBlock((state_dim + 1) * 2, (state_dim + 1) * 2, n_neurons)
        self.dropout = layers.Dropout(rate=0.1)  
        self.out = layers.Dense(1, activation=None, kernel_initializer='he_uniform')

    def call(self, obs, deviation_from_path, training=True):
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

    def eval_value(self, state, goal):
        slam_map = cv2.imread('map.pgm', cv2.IMREAD_GRAYSCALE)
        grid_map = slam_to_grid_map(slam_map)

        map_resolution = 0.05
        map_origin = (-7.52, -8.2,)
        
        # # print(grid_map)
        # # print(state)

        state_world = state[0][:2]  # Текущая позиция в мировых координатах
        goal_world = goal  # Цель в мировых координатах

        state_pixel = world_to_map(state_world, map_resolution, map_origin)
        goal_pixel = world_to_map(goal_world, map_resolution, map_origin)

        # print(state_pixel)
        # print(goal_pixel)

        rrt_star = RRTStar(state_pixel, goal_pixel, grid_map)
        optimal_path = rrt_star.plan()
        # print(optimal_path)

        state = state[0]
        if optimal_path:
            print("Найден оптимальный путь:",)
            deviation = compute_deviation_from_path((state_pixel[0], state_pixel[1]), optimal_path)
        else:
            print("Оптимальный путь не найден")
            deviation = 1e6
        
        return self.call(state, deviation)