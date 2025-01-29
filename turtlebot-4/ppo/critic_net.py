import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from my_turtlebot_package.config import TARGET_X, TARGET_Y
import random


def convert_lidar_to_obstacles(lidar_ranges, robot_state):
        """ Преобразует данные лидара в глобальные координаты """

        if isinstance(robot_state, tf.Tensor):
            robot_state = robot_state.numpy()

        # print(lidar_ranges)
        robot_state = robot_state[0]
        robot_x, robot_y = robot_state[:2]
        angle_diff = robot_state[2]
        theta_goal = np.arctan2(TARGET_Y - robot_y, TARGET_X - robot_x)  # Угол на цель
        theta_robot = theta_goal - angle_diff  # Ориентация робота

        angle_min = 0.0
        angle_increment = 0.01749303564429283
        num_readings = len(lidar_ranges)
    
        obstacles = []
        for i, r in enumerate(lidar_ranges):
            if r > 0.12:  # Учитываем минимальный диапазон сканирования
                scan_angle = angle_min + i * angle_increment  # Угол относительно робота
                obs_x = robot_x + r * np.cos(theta_robot + scan_angle)
                obs_y = robot_y + r * np.sin(theta_robot + scan_angle)
                obstacles.append((obs_x, obs_y, 0.5))  # Радиус препятствия

        return obstacles

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

class RRTStar:
    def __init__(self, start, goal, obstacle_list, map_size, step_size=2.0, max_iter=500):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.obstacle_list = obstacle_list
        self.map_size = map_size
        self.step_size = step_size
        self.max_iter = max_iter
        self.node_list = [self.start]

    def get_random_node(self):
        if random.random() > 0.1:
            return Node(random.uniform(-self.map_size[0], self.map_size[0]),
                        random.uniform(-self.map_size[1], self.map_size[1]))

        return self.goal  # Иногда пробуем идти прямо к цели

    def get_nearest_node(self, rand_node):
        return min(self.node_list, key=lambda node: np.hypot(node.x - rand_node.x, node.y - rand_node.y))

    def is_collision_free(self, node1, node2):
        for (ox, oy, r) in self.obstacle_list:
            if self.line_circle_collision(node1, node2, ox, oy, r):
                return False
        return True

    def line_circle_collision(self, node1, node2, ox, oy, r):
        dx, dy = node2.x - node1.x, node2.y - node1.y
        a = dx**2 + dy**2
        b = 2 * (dx * (node1.x - ox) + dy * (node1.y - oy))
        c = (node1.x - ox) ** 2 + (node1.y - oy) ** 2 - r**2
        det = b**2 - 4 * a * c
        return det >= 0

    def plan(self):
        for _ in range(self.max_iter):
            rand_node = self.get_random_node()
            nearest_node = self.get_nearest_node(rand_node)

            theta = np.arctan2(rand_node.y - nearest_node.y, rand_node.x - nearest_node.x)
            new_node = Node(nearest_node.x + self.step_size * np.cos(theta), nearest_node.y + self.step_size * np.sin(theta))
            new_node.parent = nearest_node

            if self.is_collision_free(nearest_node, new_node):
                self.node_list.append(new_node)

                if np.hypot(new_node.x - self.goal.x, new_node.y - self.goal.y) < self.step_size:
                    self.goal.parent = new_node
                    self.node_list.append(self.goal)
                    return self.extract_path()

        return None  # Если не найден путь

    def extract_path(self):
        path = []
        node = self.goal
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([self.start.x, self.start.y])
        return path[::-1]  # Вернуть путь от старта к цели
    
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
        self.rb1 = ResBlock(state_dim + 1, state_dim + 1, n_neurons)  # +1 из-за `nearest_dist`
        self.rb2 = ResBlock((state_dim + 1) * 2, (state_dim + 1) * 2, n_neurons)
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
    
    def get_nearest_waypoint(self, state, path):
        """ Возвращает расстояние от текущего состояния до ближайшей точки на пути """
        if not path:
            return np.inf  # Если путь не найден
        x, y = state[:2]
        return min(np.linalg.norm(np.array([x, y]) - np.array(waypoint)) for waypoint in path)

    def evaluate_with_rrt(self, lidar_ranges, robot_state, TARGET_X, TARGET_Y):
        """ Оценка состояния с использованием RRT* и критика """
        # Преобразуем данные лидара в глобальные координаты
        # print(lidar_ranges)
        filter_obs = [obs for obs in lidar_ranges if obs < 3.5 and obs != 0.0]
        obstacles = convert_lidar_to_obstacles(filter_obs, robot_state)
        robot_state = robot_state[0]  # Преобразуем из тензора, если нужно

        # print(obstacles)
    # Строим путь с использованием RRT*
        rrt_star = RRTStar(robot_state[:2], (TARGET_X, TARGET_Y), obstacles, map_size=(30, 30))
        path = rrt_star.plan()
        print("Сгенерированный путь:", path)

        # Оцениваем расстояние до ближайшей точки пути
        nearest_dist = self.get_nearest_waypoint(robot_state, path)

        # Добавляем `nearest_dist` в состояние робота перед подачей в нейросеть
        extended_state = np.append(robot_state, nearest_dist)
    
        # Передаем новое состояние в критик
        robot_tensor = tf.convert_to_tensor(extended_state.reshape(1, -1), dtype=tf.float32)
        return self.call(robot_tensor)

    
