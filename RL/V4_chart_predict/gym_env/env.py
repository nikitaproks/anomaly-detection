import gym
from gym import spaces
import numpy as np

class CustomEnv(gym.Env):
    def __init__(self, data, episode_length, obs_size):
        self.env = Env(data, obs_size)
        self.episode_length = episode_length
        self.observation_space = spaces.Box(low=-1, high=1, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    def reset(self, force=False):
        self.env.reset(force)
        obs = self.env.get_observation()
        return obs

    def step(self, action):
        reward = self.env.get_reward(action)
        self.env.move()
        obs = self.env.get_observation()
        info = {}
        if self.env.pos %self.episode_length == 0:
            done = True
        else:
            done = False
            
        return obs, reward, done, info
    
    def move(self):
        self.env.move()
            
class Env:
    def __init__(self, data, obs_size):
        self.normalizer = Normalizer()
        self.data_norm = self.normalizer.normalize(data)
        self.data = data
        
        self.lookback = obs_size - 1
        self.pos = self.lookback
        self.len = len(self.data)
        
    def get_reward(self, action):
        
        next_value = self.data[self.pos+1].item()
        current_value = self.data[self.pos].item()
        action = self.normalizer.denormalize(action).item()
        predicted_value = action
    
        difference = abs(next_value - predicted_value)
          
        if difference <= 3:
            reward  = 1 - difference/3
        elif difference <= 10:
            reward = 0
        elif difference <= 20:
            reward = 1-difference/10
        else:
            reward = -1
             
        #print(self.pos,"\t", "{:.2f}".format(action),"\t", "{:.2f}".format(next_value),"\t", "{:.2f}".format(reward))
        return reward
    
    def get_observation(self):
        obs = self.data_norm[self.pos-self.lookback:self.pos+1]
        return obs
    
    def reset(self, force):
        if force:
            self.pos = self.lookback
        else:
            pass
                
    def move(self):
        if self.is_end():
            self.pos = self.lookback
        else:
            self.pos +=1
    
    def is_end(self):
        if self.len-2 == self.pos:
            end = True
        else:
            end = False
        return end
    
class Normalizer:
    def __init__(self, data = None):
        if data != None:
            self.original_max = max([abs(val) for val in data])
    def normalize(self, data):
        self.original_max = max([abs(val) for val in data])
        normalized_data = [float(val)/self.original_max  for val in data]
        normalized_data = np.array(normalized_data)
        return normalized_data
    
    def denormalize(self, normalized_data):
        data = [float(val)*self.original_max  for val in normalized_data]
        data = np.array(data)
        return data
