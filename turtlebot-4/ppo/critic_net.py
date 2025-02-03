import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from my_turtlebot_package.config import TARGET_X, TARGET_Y
import random
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree
import matplotlib.pyplot as plt


def convert_lidar_to_obstacles(lidar_ranges, robot_state, current_yaw, eps=0.2, min_samples=5):
    """ Преобразует данные лидара в глобальные координаты с кластеризацией """
    if isinstance(robot_state, tf.Tensor):
        robot_state = robot_state.numpy()
    
    robot_state = robot_state[0]
    robot_x, robot_y = robot_state[:2]
    # angle_diff = robot_state[2]  

    # theta_goal = np.arctan2(TARGET_Y - robot_y, TARGET_X - robot_x)
    # robot_theta = theta_goal - angle_diff  

    angle_min = 0.0
    angle_increment = 0.01749303564429283
    num_readings = len(lidar_ranges)
    
    obstacle_points = []
    
    for i, r in enumerate(lidar_ranges):
        if r > 0.12:  # Фильтрация по минимальному расстоянию
            scan_angle = angle_min + i * angle_increment  
            global_angle = current_yaw + scan_angle  

            obs_x = robot_x + r * np.cos(global_angle)
            obs_y = robot_y + r * np.sin(global_angle)
            obstacle_points.append([obs_x, obs_y])

    if not obstacle_points:
        return []
    
    # Кластеризация препятствий
    obstacle_points = np.array(obstacle_points)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(obstacle_points)

    unique_obstacles = []
    for cluster_label in set(clustering.labels_):
        if cluster_label == -1:
            continue  # Игнорируем шумовые точки

        cluster_points = obstacle_points[clustering.labels_ == cluster_label]
        cluster_center = np.mean(cluster_points, axis=0)  # Средняя точка кластера
        unique_obstacles.append((cluster_center[0], cluster_center[1], 0.3))

    return unique_obstacles

class Node:
    def __init__(self, x, y, parent=None, cost=0):
        self.x = x
        self.y = y
        self.parent = parent
        self.cost = cost

class RRTStar:
    def __init__(self, start, goal, obstacle_list, map_size, step_size=0.5, goal_radius=0.5, max_iter=5000):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.obstacle_list = obstacle_list
        self.map_size = map_size
        self.step_size = step_size
        self.goal_radius = goal_radius
        self.max_iter = max_iter
        self.nodes = [self.start]
        self.kd_tree = KDTree([[self.start.x, self.start.y]])

    def get_random_node(self):
        if np.random.rand() < 0.1:
            return Node(self.goal.x, self.goal.y)
        return Node(np.random.uniform(-self.map_size[0], self.map_size[0]),
                    np.random.uniform(-self.map_size[1], self.map_size[1]))

    def get_nearest_node(self, rand_node):
        distances, idx = self.kd_tree.query([rand_node.x, rand_node.y])
        return self.nodes[idx]

    def is_collision_free(self, node1, node2):
        for obstacle in self.obstacle_list:
            if isinstance(obstacle, tuple):  # Обычное круглое препятствие
                ox, oy, r = obstacle
                if self.line_circle_collision(node1, node2, ox, oy, r):
                    return False
            elif isinstance(obstacle, list):  # Прямолинейное препятствие (стена)
                wall_start, wall_end = obstacle
                if self.line_line_collision(node1, node2, wall_start, wall_end):
                    return False
        return True

    def line_circle_collision(self, node1, node2, ox, oy, r):
        dx, dy = node2.x - node1.x, node2.y - node1.y
        a = dx**2 + dy**2
        b = 2 * (dx * (node1.x - ox) + dy * (node1.y - oy))
        c = (node1.x - ox) ** 2 + (node1.y - oy) ** 2 - r**2
        det = b**2 - 4 * a * c
        return det > 0 

    def line_line_collision(node1, node2, wall_start, wall_end):
        """ Проверка пересечения двух отрезков """
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
        A = (node1.x, node1.y)
        B = (node2.x, node2.y)
        C = wall_start
        D = wall_end

        return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))

    def steer(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        distance = np.hypot(dx, dy)
        if distance > self.step_size:
            ratio = self.step_size / distance
            new_x = from_node.x + ratio * dx
            new_y = from_node.y + ratio * dy
        else:
            new_x, new_y = to_node.x, to_node.y
        return Node(new_x, new_y, parent=from_node, cost=from_node.cost + distance)

    def rewire(self, new_node):
        neighbors_idx = self.kd_tree.query_ball_point([new_node.x, new_node.y], self.step_size * 2)
        for idx in neighbors_idx:
            neighbor_node = self.nodes[idx]
            potential_cost = new_node.cost + np.hypot(new_node.x - neighbor_node.x, new_node.y - neighbor_node.y)
            if potential_cost < neighbor_node.cost and self.is_collision_free(new_node, neighbor_node):
                neighbor_node.parent = new_node
                neighbor_node.cost = potential_cost

    def plan(self):
        for _ in range(self.max_iter):
            rand_node = self.get_random_node()
            nearest_node = self.get_nearest_node(rand_node)
            new_node = self.steer(nearest_node, rand_node)

            if self.is_collision_free(nearest_node, new_node):
                self.nodes.append(new_node)
                self.rewire(new_node)
                self.kd_tree = KDTree([[node.x, node.y] for node in self.nodes])

                if np.hypot(new_node.x - self.goal.x, new_node.y - self.goal.y) < self.goal_radius:
                    return self.extract_path(new_node)
        return None
    
    # def interpolate_path(self, path, resolution=0.1):
    #     interpolated_path = []
    #     for i in range(len(path) - 1):
    #         start = np.array(path[i])
    #         end = np.array(path[i + 1])
    #         dist = np.linalg.norm(end - start)
    #         num_points = int(dist / resolution)
    #         for t in np.linspace(0, 1, num_points):
    #             interpolated_point = start + t * (end - start)
    #             interpolated_path.append(interpolated_point.tolist())
    #     interpolated_path.append(path[-1])  # Добавляем последнюю точку
    #     return interpolated_path

    def extract_path(self, node):
        path = []
        while node:
            path.append([node.x, node.y])
            node = node.parent
        return path[::-1]

    def visualize(rrt_star, path):
        plt.figure(figsize=(10, 10))
        plt.xlim(-rrt_star.map_size[0], rrt_star.map_size[0])
        plt.ylim(-rrt_star.map_size[1], rrt_star.map_size[1])
    
        # Рисуем препятствия
        for (ox, oy, r) in rrt_star.obstacle_list:
            circle = plt.Circle((ox, oy), r, color='red', fill=True)
            plt.gca().add_patch(circle)

        # Рисуем узлы
        for node in rrt_star.nodes:
            if node.parent:
                plt.plot([node.x, node.parent.x], [node.y, node.parent.y], color='blue', linewidth=0.5)

        # Рисуем путь
        if path:
            path_x, path_y = zip(*path)
            plt.plot(path_x, path_y, color='green', linewidth=2, label='Planned Path')
        else:
            print("Path is None")
    
        plt.scatter(rrt_star.start.x, rrt_star.start.y, color='black', label='Start')
        plt.scatter(rrt_star.goal.x, rrt_star.goal.y, color='yellow', label='Goal')
    
        plt.legend()
        plt.show()
    
    
    
class ResBlock(tf.keras.Model):
    def __init__(self, input_dim, output_dim, n_neurons=512):
        super(ResBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = layers.Dense(n_neurons, activation=None, kernel_initializer='he_uniform')
        self.bn1 = layers.BatchNormalization()  
        self.activation = layers.LeakyReLU(negative_slope=0.2)

        self.fc2 = layers.Dense(output_dim, activation=None, kernel_initializer='he_uniform')
        self.bn2 = layers.BatchNormalization()  

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
        x = self.bn1(x)  
        x = self.activation(x)

        x = self.fc2(x)
        x = self.bn2(x)  

        x += residual
        if final_nl:
            x = self.activation(x)
        return x


class ImprovedCritic(tf.keras.Model):
    def __init__(self, state_dim, n_neurons=512):
        super(ImprovedCritic, self).__init__()
        self.bn_input = layers.BatchNormalization()
        self.rb1 = ResBlock(state_dim + 1, state_dim + 1, n_neurons)
        self.rb2 = ResBlock((state_dim + 1) * 2, (state_dim + 1) * 2, n_neurons)
        self.dropout = layers.Dropout(rate=0.1)
        self.out = layers.Dense(1, activation=None, kernel_initializer='he_uniform')
    
    def call(self, obs, training=True):
        obs = tf.convert_to_tensor(obs, dtype=tf.float32) if isinstance(obs, np.ndarray) else obs
        # x0 = obs
        x0 = self.bn_input(obs)
        x = self.rb1(x0)
        x = self.rb2(tf.concat([x0, x], axis=-1), training=training)
        x = self.dropout(x, training=training)
        output = self.out(x)
        return output
    
    def get_nearest_waypoint(self, state, path):
        min_dist = float('inf')
        for waypoint in path[1:]:
            if np.allclose(state[:2], waypoint):
                print("Robot is exactly on the waypoint!")
            dist = np.hypot(state[0] - waypoint[0], state[1] - waypoint[1])
            # print(f"Robot: {state[:2]}, Waypoint: {waypoint}, Distance: {dist}")
            if dist < min_dist:
                min_dist = dist
            # print('Nearest_dist during iteration: ', min_dist)
        # print('Final nearest_dist:', min_dist)
        return min_dist
    
    def evaluate_with_rrt(self, current_yaw, lidar_ranges, robot_state, TARGET_X, TARGET_Y):
        obstacles = convert_lidar_to_obstacles(lidar_ranges, robot_state, current_yaw)
        robot_state = robot_state[0]
        # print(robot_state)
        # print(obstacles)
        rrt_star = RRTStar(robot_state[:2], (TARGET_X, TARGET_Y), obstacles, map_size=(15, 15))
        path = rrt_star.plan()
        # print(path)
        rrt_star.visualize(path)
        if not path:
            print('Path is None ')
            nearest_dist = 10.0
            # path = rrt_star.interpolate_path(path, resolution=0.1)  # Интерполяция пути
        else:
            nearest_dist = self.get_nearest_waypoint(robot_state, path) 
        # print(f"Interpolated path: {path}")
        # print(f"Nearest distance: {nearest_dist}")
        # print(nearest_dist)
        extended_state = np.append(robot_state, nearest_dist)
        robot_tensor = tf.convert_to_tensor(extended_state.reshape(1, -1), dtype=tf.float32)
        return self.call(robot_tensor)
    
