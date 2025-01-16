import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import rclpy
from my_turtlebot_package.turtlebot_env import TurtleBotEnv

class PPOAgent:
    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        # Коэффициент дисконтирования
        self.gamma = 0.99 
        # Параметр для клиппинга
        self.epsilon = 0.2 
        # Скорость обучения
        self.actor_lr = 0.0003
        self.critic_lr = 0.001
        self.gaelam = 0.95

        # Создаем модели и оптимизаторы
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.critic_lr)

    # Модель актора
    def build_actor(self):
        return tf.keras.Sequential([
            layers.Input(shape=(self.state_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.action_dim, activation='softmax')
        ])

    # Модель критика
    def build_critic(self):
        return tf.keras.Sequential([
            layers.Input(shape=(self.state_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(1)
        ])

    # Выбор действия и его вероятность
    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        prob = self.actor(state)
        prob = np.squeeze(prob.numpy())
        prob = np.nan_to_num(prob, nan=1.0/self.action_dim)
        prob /= np.sum(prob)
        action = np.random.choice(self.action_dim, p=prob)
        return action, prob

    # Вычисление преимущесвт и возврата
    def compute_advantages(self, rewards, values, dones):
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
            advantages[t] = last_gae = delta + self.gamma * self.gaelam * (1 - next_done) * last_gae

    # Возвраты для обновления критика
        returns = advantages + values[:-1]  
        return advantages, returns
    
    # Обновление политик
    def update(self, states, actions, advantages, returns, old_probs):
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        old_probs = tf.convert_to_tensor(old_probs, dtype=tf.float32)

        old_probs = tf.reduce_sum(old_probs * tf.one_hot(actions, depth=self.action_dim), axis=1)

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
            values = tf.squeeze(self.critic(states))
            # Рассчитываем потерю критика
            critic_loss = tf.reduce_mean(tf.square(returns - values))

        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
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
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_dim])
                value = self.critic(state)[0, 0]

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                probs.append(prob)
                values.append(value)

                state = next_state
                episode_reward += reward
                
                # if len(states) >= batch_size:
            next_value = self.critic(next_state)[0, 0]
            values.append(next_value)

            advantages, returns = self.compute_advantages(rewards, values, dones)
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
