import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import cv2
import random
import math
import matplotlib.pyplot as plt


class Node:
    def __init__(self, point):
        self.point = point
        self.parent = None
        self.cost = 0

class RRTStar:
    def __init__(self, start, goal, grid_map, max_iter=1000, step_size=5, goal_sample_rate=0.2):
        self.start = Node(start)
        # print(start)
        self.goal = Node(goal)
        self.grid_map = grid_map
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.node_list = [self.start]

    def get_random_point(self):
        if random.random() < self.goal_sample_rate:
            return self.goal.point
        else:
            h, w = self.grid_map.shape
            return (random.randint(0, w - 1), random.randint(0, h - 1))

    def nearest(self, random_point):
        return min(self.node_list, key=lambda node: np.linalg.norm(np.array(node.point) - np.array(random_point)))

    def steer(self, from_point, to_point):
        direction = np.array(to_point) - np.array(from_point)
        length = np.linalg.norm(direction)

        if length == 0:
            return from_point
    
        direction = direction / length if length > 0 else direction
        new_point = tuple(np.array(from_point) + self.step_size * direction)
        return tuple(map(int, new_point))

    def is_collision(self, point):
        # print(point)
        x, y = int(point[0]), int(point[1])
        if x < 0 or y < 0 or x >= self.grid_map.shape[1] or y >= self.grid_map.shape[0]:
            return True
        return self.grid_map[x, y] == 1

    def get_near_nodes(self, new_node):
        radius = self.step_size * 5
        return [node for node in self.node_list if np.linalg.norm(np.array(node.point) - np.array(new_node.point)) < radius]

    def rewire(self, new_node, near_nodes):
        for near_node in near_nodes:
            potential_cost = new_node.cost + np.linalg.norm(np.array(new_node.point) - np.array(near_node.point))
            if potential_cost < near_node.cost:
                near_node.parent = new_node
                near_node.cost = potential_cost

    def plan(self):
        for _ in range(self.max_iter):
            random_point = self.get_random_point()
            # print(random_point)
            nearest_node = self.nearest(random_point)
            # print(nearest_node)
            new_point = self.steer(nearest_node.point, random_point)
            # print(new_point)

            if not self.is_collision(new_point):
                # print(self.is_collision(new_point))
                new_node = Node(new_point)
                new_node.parent = nearest_node
                new_node.cost = nearest_node.cost + np.linalg.norm(np.array(nearest_node.point) - np.array(new_point))

                near_nodes = self.get_near_nodes(new_node)
                self.rewire(new_node, near_nodes)

                self.node_list.append(new_node)

                if np.linalg.norm(np.array(new_node.point) - np.array(self.goal.point)) < self.step_size:
                    self.goal.parent = new_node
                    self.goal.cost = new_node.cost
                    return self.extract_path()

        return None

    def extract_path(self):
        path = []
        node = self.goal
        while node:
            path.append(node.point)
            node = node.parent
        return path[::-1]
    
