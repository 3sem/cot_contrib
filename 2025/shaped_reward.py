import compiler_gym
from compiler_gym.envs import LlvmEnv
from compiler_gym.wrappers import TimeLimit
from compiler_gym.wrappers import CompilerEnvWrapper
from compiler_gym.envs import CompilerEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from compile_cbench import prepare_baselines
import numpy as np
import random

DFL_BENCH = "cbench-v1/bitcount"

class BaselineRuntimeWrapper(CompilerEnvWrapper):
    def __init__(self, env: LlvmEnv, baseline_runtime=None):
        super().__init__(env)
        calc_bl = prepare_baselines(DFL_BENCH.split('/')[-1])
        self._baseline_runtime = calc_bl[0]
        self._baseline_size = calc_bl[1]
        print(f"Initial env with Oz baselines: {self._baseline_runtime} 's and {self._baseline_size} bytes")

    def calc_baselines(self):
        self.env.reset()
        self._baseline_size = self.env.observation["Runtime"]
        self._baseline_size = self.env.observation["TextSizeBytes"]
        return self._baseline_size

    def reset(self, *args, **kwargs):
        _obs = super().reset(*args, **kwargs)
        calc_bl = prepare_baselines(DFL_BENCH.split('/')[-1])
        self._baseline_runtime = calc_bl[0]#kwargs.get("Runtime", self.env.observation["Runtime"])
        self._baseline_size = calc_bl[1]#kwargs.get("TextSizeBytes", self.env.observation["TextSizeBytes"])
        return _obs


def penaltized_reward_function(env, baseline_runtime=0., runtime=0., 
                                
                               penalty_factor=0.5) -> float:

    text_file_size_oz = env.observation["TextSizeOz"]
    text_file_size_bytes = env.observation["TextSizeBytes"]
    # Calculate size improvement
    size_improvement = 1 - text_file_size_bytes/text_file_size_oz
    # Calculate runtime factor
    if runtime <= baseline_runtime:
        runtime_factor = 1.0
    else:
        runtime_factor = max(0, 1 - (runtime - baseline_runtime) / baseline_runtime)

    
    # Calculate reward
    reward = size_improvement * runtime_factor

    # Apply penalty if runtime exceeds RuntimeOz
    if runtime > baseline_runtime:
        penalty = -penalty_factor * (runtime - baseline_runtime) / baseline_runtime
        reward += penalty

    return reward


# Function to create and wrap the environment
def make_env(reward_func=penaltized_reward_function,
             max_episode_steps=200, 
             benchmark=DFL_BENCH,
             observation_space="Autophase"):
    
    env = compiler_gym.make("llvm-v0", benchmark=benchmark, observation_space=observation_space)
    
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    # Override the default step() method to use our custom reward function
    env = BaselineRuntimeWrapper(env)
    
    original_step = env.step
    env.reset()
    #rt = env.baseline_runtime()
    def custom_step(action):
        #print(f"Action: {action}")
        
        observation, _, done, info = original_step(int(action))
        reward = reward_func(env, env._baseline_runtime[0], env.observation['Runtime'][0], 0.5)
        
        return observation, reward, done, info

    env.step = custom_step
    return env

# Create vectorized environment
vec_env = DummyVecEnv([make_env for _ in range(4)])  # Use parallel environments

# Create the PPO agent with a custom policy network
policy_kwargs = dict(
    net_arch=[dict(pi=[128, 128], vf=[128, 128])],
)

model = PPO("MlpPolicy", vec_env, verbose=1, policy_kwargs=policy_kwargs,
            n_steps=1024, batch_size=64, n_epochs=10, learning_rate=3e-4)

# Train the agent
model.learn(total_timesteps=100000)

# Evaluate the trained agent
env = make_env()
autophase_init_obs = env.reset()

done = False
total_reward = 0
obs = env.reset()

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    
print(f"Total reward: {total_reward}")

# Get the final optimized IR
optimized_ir = env.ir

# Print some statistics
print(f"Benchmark: {env.benchmark}")
print(f"Final bytes count: {env.observation['TextSizeBytes']}; \
      Profit: {(1 - env.observation['TextSizeBytes'] / env.observation['TextSizeOz'])*100 }%")

print(f"Final runtime: {env.observation['Runtime'][0]}; Profit: \
      {(1 - env.observation['Runtime'][0]/env._baseline_runtime) * 100}%")

# Close the environment
env.close()
