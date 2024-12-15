import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_baselines3 import DQN,PPO
import datetime
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from env import MySim_D

if __name__=="__main__":
    
    # DQN,A2C,HER,PPO,QR-DQN,TRPO,Maskable PPO can use
    env=Monitor(MySim_D())
    model =DQN(policy="MlpPolicy", env=env,)
    model.learn(total_timesteps=40000)
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    model.save(f'test_{timestamp}')
    episode_lengths = env.get_episode_lengths()
    episode_rewards = env.get_episode_rewards()

    for i in range(10):
      obs, info = env.unwrapped.reset()
      terminated = False
      truncated = False
      print(f"Starting Episode {i+1}")
      while not (terminated or truncated):
          action, _ = model.predict(observation=obs)
          obs, reward, terminated, truncated, info = env.unwrapped.step(action)

      path_xdata = env.unwrapped.path_xdata
      path_ydata = env.unwrapped.path_ydata
      # Create a figure and axis for plotting
      fig, ax = env.unwrapped.render(mode='human')
      # Line object to update in the animation
      line, = ax.plot([], [], marker='o', color='blue', label='Path')

      # Update function for animation
      def update(frame):
          line.set_data(path_ydata[:frame+1], path_xdata[:frame+1])
          return line,

      # Create the animation
      ani = animation.FuncAnimation(fig, update, frames=len(path_xdata), interval=100, blit=False)

      # Display the animation
      plt.show()
      print(f"Path for Episode {i+1}: X - {path_xdata}, Y - {path_ydata}")
