import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import defaultdict
import random

class ConstrainedActorCritic(nn.Module):
    """Improved Actor-Critic with Lagrangian constraints using PPO-style updates."""
    
    def __init__(self, state_dim, action_dim, params):
        super().__init__()
        self.params = params
        
        # Policy Network (Actor)
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, params['hidden_size']),
            nn.ReLU(),
            nn.Linear(params['hidden_size'], params['hidden_size']),
            nn.ReLU(),
            nn.Linear(params['hidden_size'], action_dim)
        )
        
        # Value Network (Critic)
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, params['hidden_size']),
            nn.ReLU(),
            nn.Linear(params['hidden_size'], params['hidden_size']),
            nn.ReLU(),
            nn.Linear(params['hidden_size'], 1)
        )
        
        # Constraint Network
        self.constraint_net = nn.Sequential(
            nn.Linear(state_dim, params['hidden_size']),
            nn.ReLU(),
            nn.Linear(params['hidden_size'], 1)
        )
        
        # Lagrangian multiplier
        self.register_buffer('lagrangian_mult', torch.tensor(1.0))
        self.constraint_threshold = params.get('constraint_threshold', 0.1)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, state):
        """Forward pass for all networks"""
        state = torch.FloatTensor(state).flatten()
        action_probs = F.softmax(self.policy_net(state), dim=-1)
        state_value = self.value_net(state)
        constraint_value = self.constraint_net(state)
        return action_probs, state_value, constraint_value

class RuntimeConstrainedAgent:
    def __init__(self, state_dim, action_dim, params):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ConstrainedActorCritic(state_dim, action_dim, params).to(self.device)
        self.params = params
        
        # Optimization
        self.optimizer = optim.Adam([
            {'params': self.policy.policy_net.parameters()},
            {'params': self.policy.value_net.parameters()},
            {'params': self.policy.constraint_net.parameters()}
        ], lr=params.get('learning_rate', 3e-4))
        
        # Hyperparameters
        self.gamma = params.get('gamma', 0.99)
        self.lam = params.get('lam', 0.95)
        self.eps_clip = params.get('eps_clip', 0.2)
        self.lagrangian_lr = params.get('lagrangian_lr', 0.01)
        self.max_lagrangian_mult = params.get('max_lagrangian_mult', 10.0)
        self.entropy_coef = params.get('entropy_coef', 0.01)
        self.max_grad_norm = params.get('max_grad_norm', 0.5)
        
        # Buffers
        self.buffer = defaultdict(list)
        self.baseline_runtime = None

    def select_action(self, state, exploration_rate=0.0):
        state = torch.FloatTensor(state).flatten().to(self.device)
        probs, value, constraint_val = self.policy(state)
        
        # Exploration
        if random.random() < exploration_rate:
            action = torch.randint(0, len(probs), (1,))
        else:
            dist = Categorical(probs)
            action = dist.sample()
        
        # Store transition
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['log_probs'].append(Categorical(probs).log_prob(action))
        self.buffer['values'].append(value)
        self.buffer['constraint_vals'].append(constraint_val)
        
        return action.item()

    def update_constraint(self, current_runtime):
        if self.baseline_runtime is None:
            self.baseline_runtime = current_runtime
            return 0.0
        
        runtime_diff = abs(current_runtime - self.baseline_runtime) / (self.baseline_runtime + 1e-5)
        self.buffer['constraints'].append(runtime_diff)
        return runtime_diff

    def compute_gae(self, rewards, values, masks):
        deltas = rewards + self.gamma * masks * values[1:] - values[:-1]
        advantages = torch.zeros_like(rewards)
        advantage = 0
        for t in reversed(range(len(rewards)-1)):
            advantage = deltas[t] + self.gamma * self.lam * masks[t] * advantage
            advantages[t] = advantage
        return advantages

    def update(self):
        if len(self.buffer['states']) == 0:
            return self._empty_stats()
            
        # Prepare batch data
        batch = {k: torch.stack(v).to(self.device) for k, v in self.buffer.items()}
        
        # Compute returns and advantages
        with torch.no_grad():
            batch['returns'] = self._compute_returns(batch['rewards'], batch['masks'])
            batch['advantages'] = self.compute_gae(
                batch['rewards'], 
                batch['values'], 
                batch['masks']
            )
            batch['advantages'] = (batch['advantages'] - batch['advantages'].mean()) / \
                                (batch['advantages'].std() + 1e-8)
            
            # Constraint returns
            batch['constraint_returns'] = self._compute_constraint_returns(
                batch['constraints'],
                batch['masks']
            )

        # Get current policy probs
        current_probs = F.softmax(self.policy.policy_net(batch['states']), dim=-1)
        current_log_probs = Categorical(current_probs).log_prob(batch['actions'])
        
        # PPO-clipped policy loss
        ratios = torch.exp(current_log_probs - batch['log_probs'].detach())
        clip_adv = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * batch['advantages']
        policy_loss = -torch.min(ratios * batch['advantages'], clip_adv).mean()
        
        # Value loss with clipping
        current_values = self.policy.value_net(batch['states']).squeeze()
        v_loss_unclipped = F.mse_loss(current_values, batch['returns'])
        v_loss_clipped = F.mse_loss(
            current_values, 
            batch['values'] + torch.clamp(current_values-batch['values'], -0.5, 0.5)
        )
        value_loss = torch.max(v_loss_unclipped, v_loss_clipped)
        
        # Constraint handling
        constraint_violations = F.relu(batch['constraints'] - self.policy.constraint_threshold)
        constraint_loss = constraint_violations.mean()
        
        # Entropy bonus
        entropy = Categorical(current_probs).entropy().mean()
        
        # Total loss
        total_loss = (policy_loss + 
                    0.5 * value_loss + 
                    self.policy.lagrangian_mult * constraint_loss -
                    self.entropy_coef * entropy)
        
        # Update networks
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Update Lagrangian multiplier
        with torch.no_grad():
            lagrangian_grad = (constraint_loss - self.policy.constraint_threshold)
            self.policy.lagrangian_mult += self.lagrangian_lr * lagrangian_grad
            self.policy.lagrangian_mult.clamp_(min=0, max=self.max_lagrangian_mult)
        
        # Clear buffers
        self._clear_buffers()
        
        return {
            'loss/total': total_loss.item(),
            'loss/policy': policy_loss.item(),
            'loss/value': value_loss.item(),
            'loss/constraint': constraint_loss.item(),
            'entropy': entropy.item(),
            'lagrangian': self.policy.lagrangian_mult.item(),
            'constraint_violation': constraint_violations.mean().item()
        }

    def _compute_returns(self, rewards, masks):
        returns = torch.zeros_like(rewards)
        R = 0
        for t in reversed(range(len(rewards))):
            R = rewards[t] + self.gamma * masks[t] * R
            returns[t] = R
        return returns

    def _compute_constraint_returns(self, constraints, masks):
        returns = torch.zeros_like(constraints)
        R = 0
        for t in reversed(range(len(constraints))):
            R = constraints[t] + 0.9 * masks[t] * R  # Different gamma for constraints
            returns[t] = R
        return returns

    def _clear_buffers(self):
        self.buffer = defaultdict(list)

    def _empty_stats(self):
        return {k: 0.0 for k in [
            'loss/total', 'loss/policy', 'loss/value', 
            'loss/constraint', 'entropy', 'lagrangian',
            'constraint_violation'
        ]}


def TrainActorCritic(env, PARAMS=FLAGS, reward_estimator=const_factor_threshold, 
                    reward_if_list_func=lambda a: np.mean(a), exp_name='gsm'):
    # Initialize agent with proper dimensions
    state_dim = len(env.observation_space.sample()[0])  # Assuming state[0] is the observation
    action_size = len(env.action_spaces[0].names)
    
    # Use white list if provided
    if FLAGS.get("actions_white_list"):
        action_size = len(FLAGS["actions_white_list"])
    
    # Initialize agent with updated architecture
    agent = RuntimeConstrainedAgent(
        state_dim=state_dim,
        action_dim=action_size,
        params={
            **PARAMS,
            'hidden_size': PARAMS.get('hidden_size', 64),
            'constraint_threshold': PARAMS.get('runtime_threshold', 0.1)
        }
    )
    
    # Tracking variables
    results_ep = []
    max_ep_reward = -float("inf")
    avg_reward = MovingExponentialAverage(0.95)
    avg_loss = MovingExponentialAverage(0.95)
    avg_constraint_violation = MovingExponentialAverage(0.95)
    ts = time.time()
    
    # Get baseline runtime
    baseline_state = env.reset()
    baseline_runtime = reward_if_list_func(baseline_state[2])
    
    for episode in range(1, PARAMS['episodes'] + 1):
        state = env.reset()
        ep_reward = 0
        prev_size = state[1]
        prev_runtime = reward_if_list_func(state[2])
        
        action_log = []
        pat = 0
        best_ep_logs = {'size': 0., 'runtime': 0, 'sequence': ['None']}
        print(f"Trying to beat runtime {baseline_runtime}...")
        
        while True:
            # Select action (handles white listing internally)
            action = agent.select_action(
                state[0],
                exploration_rate=PARAMS['exploration']
            )
            
            action_name = env.action_spaces[0].names[action]
            action_log.append(action_name)
            
            # Environment step
            next_state, reward, done, _ = env.step(action)
            
            # Calculate metrics
            current_runtime = reward_if_list_func(next_state[2])
            reward = reward_estimator(
                baseline_m=env.hetero_os_baselines[0],
                measured_m=next_state[1],
                baseline_n=baseline_runtime,
                measured_n=current_runtime,
                prev_m=prev_size,
                prev_n=prev_runtime,
                m_norm=prev_size
            )
            
            # Update constraint
            runtime_diff = agent.update_constraint(current_runtime)
            
            # Store transition
            agent.buffer['rewards'].append(reward)
            agent.buffer['masks'].append(0 if done else 1)
            
            # Early stopping
            if reward <= 0.:
                pat += 1
            if FLAGS.get("patience", 10) <= pat:
                done = True
                
            # Track best results
            size_reward = next_state[1]
            size_baseline = env.hetero_os_baselines[0]
            
            size_gain = -(size_reward - size_baseline) * 100 / size_baseline
            runtime_diff_pct = (current_runtime - baseline_runtime) * 100 / baseline_runtime
            
            if size_reward < size_baseline and runtime_diff_pct <= 0.001:
                if size_gain > best_ep_logs['size']:
                    best_ep_logs.update({
                        'size': size_gain,
                        'runtime_diff': runtime_diff_pct,
                        'sequence': action_log.copy()
                    })
            
            # Update state tracking
            prev_size = next_state[1]
            prev_runtime = current_runtime
            state = next_state
            ep_reward += reward
            
            if done:
                # Log best results
                print(f"--\nBest results of episode {episode}:")
                if best_ep_logs['size'] > 0.:
                    print_results(best_ep_logs)
                
                results_ep.append({
                    'time': time.time() - ts,
                    'num': episode,
                    'size_gain': best_ep_logs['size'],
                    'runtime_diff': best_ep_logs.get('runtime_diff', 'N/A'),
                    'sequence': best_ep_logs['sequence']
                })
                break
        
        # Update policy
        update_stats = agent.update()
        
        # Update statistics
        max_ep_reward = max(max_ep_reward, ep_reward)
        avg_reward.next(ep_reward)
        avg_loss.next(update_stats['loss/total'])
        avg_constraint_violation.next(update_stats['constraint_violation'])
        
        # Logging
        if episode == 1 or episode % FLAGS['log_interval'] == 0 or episode == FLAGS['episodes']:
            log_progress(
                episode, 
                ep_reward, 
                avg_reward.value, 
                update_stats,
                FLAGS["logger"]
            )
            
        # Periodic saving
        if episode % 50 == 0 or episode == FLAGS['episodes']:
            save_results(exp_name, ts, results_ep, episode, ep_reward, avg_reward.value, update_stats)
    
    # Final save
    save_list_of_dicts_to_json(
        data=results_ep, 
        filename=f"runtime_constrained_{int(time.time())}.json"
    )
    
    return avg_reward.value

# Helper functions
def print_results(best_ep_logs):
    print(f"  Size reduction: {best_ep_logs['size']:.2f}%")
    print(f"  Runtime difference: {best_ep_logs.get('runtime_diff', 'N/A')}%")
    print(f"  Action sequence: {best_ep_logs['sequence']}")
    display_progress_advanced(
        best_ep_logs['size'],
        filled_char='█',
        empty_char=' ',
        prefix='Size   : ',
        suffix=f" reduction",
        color=Colors.BLUE
    )
    display_progress_advanced(
        -best_ep_logs.get('runtime_diff'),
        filled_char='█',
        empty_char=' ',
        prefix='Runtime: ',
        suffix=f" reduction",
        color=Colors.RED
    )

def log_progress(episode, ep_reward, avg_reward, update_stats, logger):
    logger.save_and_print(
        f"Episode {episode}\t"
        f"Last reward: {ep_reward:.2f}\t"
        f"Avg reward: {avg_reward:.2f}\t"
        f"Constraint viol: {update_stats['constraint_violation']:.4f}\t"
        f"Lambda: {update_stats['lagrangian']:.4f}",
        mode=LogMode.SHORT
    )

def save_results(exp_name, ts, results_ep, episode, ep_reward, avg_reward, update_stats):
    # Save raw results
    with open(f"outputs/{exp_name}_{ts}.txt", "w") as f:
        f.write('\n'.join([f"{entry['num']} {entry['size_gain']}" for entry in results_ep]))
    
    # Calculate and save statistics
    means, succ_rates = calculate_means_and_success_rate(f"outputs/{exp_name}_{ts}.txt", separ=50)
    
    # Save model info
    with open(f"outputs/{exp_name}_{ts}_model.txt", "w") as f:
        f.write(f"Episode {episode}\n"
                f"Success rates {succ_rates}\n"
                f"Last reward: {ep_reward:.2f}\n"
                f"Avg reward: {avg_reward:.2f}\n"
                f"Constraint viol: {update_stats['constraint_violation']:.4f}\n"
                f"Lambda: {update_stats['lagrangian']:.4f}")
    
    # Save means
    with open(f"outputs/{exp_name}_{ts}_means.txt", "w") as f:
        f.write('\n'.join([str(m) for m in means]))
