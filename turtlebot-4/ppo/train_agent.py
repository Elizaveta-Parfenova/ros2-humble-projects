import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import rclpy
from my_turtlebot_package.turtlebot_env import TurtleBotEnv
from my_turtlebot_package.actor_net import ImprovedActor
from my_turtlebot_package.critic_net import ImprovedCritic
from my_turtlebot_package.config import TARGET_X, TARGET_Y
from my_turtlebot_package.rrt_star import RRTStar
import cv2



def slam_to_grid_map(slam_map, threshold=50):
    """
    Преобразование карты SLAM в бинарную grid map.
    Args:def map_to_world(map_coords, resolution, origin):
    x_map, y_map = map_coords
    x_world = x_map * resolution + origin[0]
    y_world = y_map * resolution + origin[1]
    return (x_world, y_world)
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

def world_to_map(world_coords, resolution, origin):
    x_world, y_world = world_coords
    x_map = int((x_world - origin[0]) / resolution)
    y_map = int((y_world - origin[1]) / resolution)
    return (x_map, y_map)

def map_to_world(map_coords, resolution, origin):
    x_map, y_map = map_coords
    x_world = x_map * resolution + origin[0]
    y_world = y_map * resolution + origin[1]
    return (x_world, y_world)

slam_map = cv2.imread('map.pgm', cv2.IMREAD_GRAYSCALE)
grid_map = slam_to_grid_map(slam_map)

def path(state, goal, map_resolution = 0.05, map_origin = (-7.52, -8.2,)):
    
    state_world = state # Текущая позиция в мировых координатах
    goal_world = goal  # Цель в мировых координатах

    state_pixel = world_to_map(state_world, map_resolution, map_origin)
    goal_pixel = world_to_map(goal_world, map_resolution, map_origin)

    # print(state_pixel)
    # print(goal_pixel)

    rrt_star = RRTStar(state_pixel, goal_pixel, grid_map)
    optimal_path = rrt_star.plan()

    # optimal_path = [map_to_world(p, map_resolution, map_origin) for p in optimal_path_grid]

    # print(optimal_path)
    return optimal_path

# --- Класс агента PPO ---
class PPOAgent:
    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.state_pose = [-2.0, -0.5]
        self.goal = np.array(env.goal, dtype=np.float32)
        # print(self.goal)
 
        self.x_range = np.array(env.x_range, dtype=np.float32)  # Диапазон X как массив NumPy
        self.y_range = np.array(env.y_range, dtype=np.float32)  # Диапазон Y как массив NumPy

        # self.obstacles = np.array(env.obstacles, dtype=np.float32)
        # print(self.obstacles)

        # Коэффициенты
        self.gamma = 0.99  # коэффициент дисконтирования
        self.epsilon = 0.2  # параметр клиппинга
        self.actor_lr = 0.0003
        self.critic_lr = 0.0003
        self.gaelam = 0.95

        self.optimal_path = path(self.state_pose, self.goal)

        # Модели
        self.actor = ImprovedActor(self.state_dim, self.action_dim)
        self.critic = ImprovedCritic(self.state_dim, grid_map=grid_map, optimal_path=self.optimal_path)

        # Оптимизаторы
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.critic_lr)

    # Выбор действия и его вероятность
    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        prob = self.actor(state)
        prob = np.squeeze(prob.numpy())
        prob = np.nan_to_num(prob, nan=1.0/self.action_dim)
        prob /= np.sum(prob)
        action = np.random.choice(self.action_dim, p=prob)
        # if not np.isclose(np.sum(prob), 1.0):
        #     print("Warning: Probabilities do not sum to 1!", prob)
        return action, prob

    # Вычисление преимущесвт и возврата
    def compute_advantages(self, rewards, values, dones):
        # print('Rewrds: ', rewards)
        # print('Values:', values) angle_diff
        # print('Next values:', next_value)
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        last_gae = 0
        next_value = values[-1]
        for t in reversed(range(len(rewards))):
            # Обработка последнего шага
            if t == len(rewards) - 1:
                next_value = values[-1]
                next_done = dones[t]  
            else:
                # Обработка остальных шагов
                next_value = values[t + 1]
                next_done = dones[t + 1]

            # Вычисление ошибки предсказания
            delta = rewards[t] + self.gamma * next_value * (1 - next_done) - values[t]
            # if np.isnan(delta):
            #     print(f"NaN in delta: rewards[{t}]={rewards[t]}, next_value={next_value}, values[{t}]={values[t]}")
            advantages[t] = last_gae = delta + self.gamma * self.gaelam * (1 - next_done) * last_gae
        # print('Advanteges:', advantages)
    # Возвраты для обновления критика
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        returns = advantages + values[:-1]  
        # print ('Returns:', returns)
        return advantages, returns
    
    # Обновление политик
    def update(self, states, actions, advantages, returns, old_probs):
        # states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        old_probs = tf.convert_to_tensor(old_probs, dtype=tf.float32)

        old_probs = tf.reduce_sum(old_probs * tf.one_hot(actions, depth=self.action_dim), axis=1)
        #print(old_probs)

        with tf.GradientTape() as tape:
            # Получаем вероятности действий от актора
            prob = self.actor(states)
            # Вероятности выбранных действий
            chosen_probs = tf.reduce_sum(prob * tf.one_hot(actions, depth=self.action_dim), axis=1)

             # Отношение текущих и старых вероятностей
            prob_ratio = chosen_probs / old_probs
            # Клиппинг
            clipped_prob_ratio = tf.clip_by_value(prob_ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
            
            # Вычисление функции потерь surrogate (PPO loss)
            surrogate_loss = tf.minimum(prob_ratio * advantages, clipped_prob_ratio * advantages)
            # Финальный actor loss (усреднённый отрицательный surrogate loss)
            actor_loss = -tf.reduce_mean(surrogate_loss)
            
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        with tf.GradientTape() as tape:
            # Получаем значения из критика
            values = tf.squeeze(self.critic.eval_value(states))
            # print(values)
            # Рассчитываем потерю критика
            critic_loss = tf.keras.losses.Huber()(returns, values)

        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        # clipped_gradients = [tf.clip_by_norm(g, 1.0) for g in critic_grads]
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

    def train(self, max_episodes=500, batch_size=32):
        all_rewards = []
        
        for episode in range(max_episodes):
            state = np.reshape(self.env.reset(), [1, self.state_dim])
            episode_reward = 0
            done = False

            states, actions, rewards, dones, probs, values = [], [], [], [], [], []

            while not done:
                action, prob = self.get_action(state)
                # print('Action:', action)
                # print('Prob:', prob)
                next_state, reward, done, _ = self.env.step(action)
                # print(next_state)
                # print(reward)
                # print(done)
                next_state = np.reshape(next_state, [1, self.state_dim])
                # print(next_state)
                # print(self.goal)
                # print(self.obstacles)
                value = self.critic.eval_value(state)[0, 0]
                print(value)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                probs.append(prob)
                values.append(value)

                state = next_state
                episode_reward += reward
                # print(episode_reward)
                
                # if len(states) >= batch_size:
            next_value = self.critic.eval_value(next_state)[0, 0]
            values.append(next_value)
            # print(values)
            advantages, returns = self.compute_advantages(rewards, values, dones)
            # print(advantages)
            # print(returns)
            self.update(np.vstack(states), actions, advantages, returns, probs)
            
            all_rewards.append(episode_reward)
            print(f'Episode {episode + 1}, Reward: {episode_reward}')

        self.actor.save('ppo_turtlebot_actor')
        self.critic.save('ppo_turtlebot_critic')

def main(args=None):
    rclpy.init(args=args)
    env = TurtleBotEnv()
    agent = PPOAgent(env)
    agent.train()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
