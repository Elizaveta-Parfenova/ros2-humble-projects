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
        
        self.target_x = 5.0
        self.target_y = 1.0
        
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.obstacles = []
        self.prev_distance = None
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

    def scan_callback(self, msg):
        self.obstacles = [r if not math.isinf(r) and not math.isnan(r) and msg.range_min < r < msg.range_max 
                      else msg.range_max for r in msg.ranges]

        if not self.obstacles:
            self.obstacles = [msg.range_max]  
    
    def camera_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            # Преобразование в оттенки серого
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            # Применение пороговой бинаризации к изображению.
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            # Подсчет белых пикселей
            non_zero_pixels = cv2.countNonZero(thresh)
            self.camera_obstacle_detected = non_zero_pixels > 1000

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")
            self.camera_obstacle_detected = False

 
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
        
        distance = math.sqrt((self.target_x - self.current_x) ** 2 + (self.target_y - self.current_y) ** 2)
        angle_to_goal = math.atan2(self.target_y - self.current_y, self.target_x - self.current_x)
        angle_diff = (angle_to_goal - self.current_yaw + np.pi) % (2 * np.pi) - np.pi

        min_obstacle_dist = min(self.obstacles) if self.obstacles else float('inf')
        obstacle_detected = (min_obstacle_dist < 0.2 and self.camera_obstacle_detected)
        state = np.array([self.current_x, self.current_y, angle_diff, min_obstacle_dist])

        # print(min_obstacle_dist)
        reward = -distance
        if self.prev_distance is not None:
            reward += 10 if distance < self.prev_distance else -10
        self.prev_distance = distance

        done = False
        if obstacle_detected:
            reward -= 50
            done = False
        elif distance < 0.2:
            reward += 1000
            done = True
        elif self.steps >= self.max_steps:
            reward -= 50
            done = True
        else:
            done = False
        
        reward = reward / 1000.0 
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
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.steps = 0
        self.prev_distance = None
        self.obstacles = []
        self.camera_obstacle_detected = False
        return np.array([self.current_x, self.current_y, 0.0, 0.0])  
 

    def render(self, mode='human'):
        pass

    def close(self):
        pass
    
    def seed(self, seed=None):
        np.random.seed(seed)
