import tensorflow as tf
import numpy as np
import gym
import tqdm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow_probability import distributions

class PolicyNetwork(Model):
    def __init__(self, obs_dims, n_actions):
        super(PolicyNetwork, self).__init__()
        self.d1 = Dense(128, activation='relu', input_shape=(obs_dims, ))
        self.d2 = Dense(128, activation='relu')
        self.pi = Dense(n_actions, activation='softmax')

    def __call__(self, obs, training=None):
        x = self.d1(obs)
        x = self.d2(x)
        x = self.pi(x)
        return x


class ReplayBuffer():
    def __init__(self):
        self.obsMem = []
        self.nextobsMem = []
        self.actionMem = []
        self.rewardMem = []
        self.doneMem = []

    def push(self, obs, obs_, action, reward, done):
        self.obsMem.append(obs)
        self.nextobsMem.append(obs_)
        self.actionMem.append(action)
        self.rewardMem.append(reward)
        self.doneMem.append(done)
    
    def get_returns(self, gamma):
        G = np.zeros(len(self.rewardMem), dtype=np.float32)
        g = 0.
        for i in reversed(range(len(self.rewardMem))):
            g = self.rewardMem[i] + gamma*g
            
            G[i] = g 

            if done:
                g = 0.
                G[i] = g 
                
        G = (G - np.mean(G)) / np.std(G)
        return G

            
    def pull(self):
        return (
            np.array(self.obsMem,dtype=np.float32),
            np.array(self.nextobsMem, dtype=np.float32),
            np.array(self.actionMem, dtype=np.int64),
            np.array(self.rewardMem, dtype=np.float32),
            np.array(self.doneMem, dtype=np.float32)
        )
    
    def reset(self):
        self.__init__()

class Agent():
    def __init__(self, learning_rate, obs_dims, n_actions, gamma):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.obs_dims = obs_dims
        self.n_actions = n_actions

        self.Replay = ReplayBuffer()
        self.Policy = PolicyNetwork(self.obs_dims, self.n_actions)
        self.optimizer = Adam(learning_rate=self.learning_rate)
    
    @tf.function
    def get_action(self, obs):
        obs = tf.reshape(obs, (-1, self.obs_dims))
        probs = self.Policy(obs)

        probs = distributions.Categorical(probs=probs)
        action = probs.sample()
        return action

    #@tf.function
    def train(self):
        obs , obs_, action, reward, done = self.Replay.pull()
        returns = self.Replay.get_returns(self.gamma)
        with tf.GradientTape() as tape:
            loss = 0
            for idx, (g, s) in enumerate(zip(returns, obs)):
                s = tf.reshape(s, (-1, self.obs_dims))

                probs = self.Policy(s)
                action_probs = distributions.Categorical(probs = probs)
                logprob = action_probs.log_prob(action[idx])
                loss += -tf.squeeze(logprob)*g
       
        policyGradient = tape.gradient(loss, self.Policy.trainable_variables)
        self.optimizer.apply_gradients(zip(policyGradient, self.Policy.trainable_variables))

if __name__ == "__main__":
    env = gym.make("CartPole-v0")

    n_actions = env.action_space.n
    obs_dims = env.reset().shape[0]
    gamma = 1
    learning_rate=0.0001

    BATCH_SIZE = 1000
    NUM_UPDATES = 500

    agent = Agent(learning_rate=learning_rate, obs_dims=obs_dims, n_actions=n_actions, gamma=gamma)

    scores = []
    avg_scores = []

    with tqdm.trange(NUM_UPDATES) as t:
        for update in t:
            
            score = 0
            s = env.reset()
            done = False
            scores = []
            while True:
                a = agent.get_action(s).numpy()[0]
                s_, r, done, info = env.step(a)
                
                score  += r

                r = (-195 if done else r)
                agent.Replay.push(s, s_, a, r, done)
                s = s_
                if done:
                    scores.append(score)

                    score = 0
                    s = env.reset()
                    done = False
                    t.set_description(
                            f"size = {len(agent.Replay.rewardMem)}"
                        )
                    if len(agent.Replay.rewardMem) > BATCH_SIZE:
                        break

            agent.train()
            agent.Replay.reset()
            t.set_postfix(
                            avg_score = np.mean(scores),
                            
                        )
            avg_score = np.mean(scores)
            avg_scores.append(avg_score)
            