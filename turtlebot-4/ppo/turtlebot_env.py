import gym
from gym import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Image
import math
from std_srvs.srv import Empty
from cv_bridge import CvBridge
import cv2
import collections


class TurtleBotEnv(Node, gym.Env):
    def __init__(self):
        super().__init__('turtlebot_env')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.subscription_laser = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.subscription_camera = self.create_subscription(Image, '/camera/image_raw', self.camera_callback, 10)
        
        self.bridge = CvBridge()
        self.camera_obstacle_detected = False
        
        self.target_x = -2.0
        self.target_y = -6.0
        self.goal = [self.target_x, self.target_y]
        
        self.x_range = [-10,10]
        self.y_range = [-10,10]

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.obstacles = []
        self.prev_distance = None
        self.past_distance = 0
        self.max_steps = 5000
        self.steps = 0 
        
        self.action_space = spaces.Discrete(3)  
        self.observation_space = spaces.Box(low=np.array([-10.0, -10.0, -np.pi, 0.0]), 
                                            high=np.array([10.0, 10.0, np.pi, 12.0]), 
                                            shape=(4,), dtype=np.float32)
        
        self.timer = self.create_timer(0.1, self._timer_callback)

    def _timer_callback(self):
        pass 

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        orientation_q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (orientation_q.w * orientation_q.z + orientation_q.x * orientation_q.y)
        cosy_cosp = 1.0 - 2.0 * (orientation_q.y * orientation_q.y + orientation_q.z * orientation_q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    # def scan_callback(self, msg):
        # self.obstacles = [r if not math.isinf(r) and not math.isnan(r) and msg.range_min < r < msg.range_max else msg.range_max for r in msg.ranges]
    
    def scan_callback(self, msg):
        self.obstacles = [r if not math.isinf(r) and not math.isnan(r) and msg.range_min < r < msg.range_max 
                      else msg.range_max for r in msg.ranges]

        if not self.obstacles:
            self.obstacles = []  

    def camera_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if cv_image is not None:
                self.camera_obstacle_detected = self.process_camera_image(cv_image)
            else:
                self.camera_obstacle_detected = False
            # print(self.camera_obstacle_detected)
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")
            self.camera_obstacle_detected = False

    def process_camera_image(self, cv_image):
   
        # Преобразование изображения в формат для обработки
        pixel_values = cv_image.reshape((-1, 3))  # Преобразование в список пикселей
        pixel_values = np.float32(pixel_values)

        # Применение K-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = 2  # Количество кластеров (фон и препятствие)
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Преобразование обратно в изображение
        centers = np.uint8(centers)
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(cv_image.shape)

        # Анализ кластеров
        obstacle_detected = np.count_nonzero(labels == 1) > 210000 # Если пикселей кластеров > чего-то, то препятствие обнаружено
        # print(obstacle_detected)
        return obstacle_detected
    
    def step(self, action):
        cmd_msg = Twist()
        if action == 0:
            cmd_msg.angular.z = 0.5  
        elif action == 1:
            cmd_msg.linear.x = 0.2  
        elif action == 2:
            cmd_msg.angular.z = -0.5  
        
        rclpy.spin_once(self, timeout_sec=0.1) 
        self.publisher_.publish(cmd_msg)
    
        self.steps += 1
        
        # print(self.obstacles)
        distance = math.sqrt((self.target_x - self.current_x) ** 2 + (self.target_y - self.current_y) ** 2)
        angle_to_goal = math.atan2(self.target_y - self.current_y, self.target_x - self.current_x)
        angle_diff = (angle_to_goal - self.current_yaw + np.pi) % (2 * np.pi) - np.pi

        min_obstacle_dist = min(self.obstacles) if self.obstacles else float('inf')
        obstacle_detected = (min_obstacle_dist < 0.5 and self.camera_obstacle_detected)
        state = np.array([self.current_x, self.current_y, angle_diff, min_obstacle_dist])

        distance_rate = (self.past_distance - distance)
        # print(min_obstacle_dist)
        reward = 500.0 * distance_rate
        self.past_distance = distance
        # if self.prev_distance is not None:
        #     reward += 10 if distance < self.prev_distance else -10
        # self.prev_distance = distance

        done = False
        if obstacle_detected:
            reward -= 100
            done = False
        elif distance < 0.2:
            reward += 120 
            done = True
        elif self.steps >= self.max_steps:
            reward -= 150
            done = True
        else:
            done = False
        
        # reward = reward / 100.0 
        # print(state)

        return state, reward, done, {}

    def reset(self):
    # Остановить движение
        cmd_msg = Twist()
        cmd_msg.linear.x = 0.0  
        cmd_msg.angular.z = 0.0
        self.publisher_.publish(cmd_msg)  
        rclpy.spin_once(self, timeout_sec=0.1) 
    
        # Использовать reset_simulation для физического сброса в Gazebo
        client = self.create_client(Empty, '/reset_simulation')
        request = Empty.Request()
        if client.wait_for_service(timeout_sec=1.0):
            client.call_async(request)
        else:
            self.get_logger().warn('Gazebo reset service not available!')

    # Сбросить внутренние переменные
        self.current_x = -2.0
        self.current_y = -0.5
        self.current_yaw = 0.0
        self.steps = 0
        self.prev_distance = None
        # self.obstacles = []
        self.camera_obstacle_detected = False
        return np.array([self.current_x, self.current_y, 0.0, 0.0])  
 
    def render(self, mode='human'):
        pass

    def close(self):
        pass
    
    def seed(self, seed=None):
        np.random.seed(seed)
