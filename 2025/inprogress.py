from common import *
import torch
import time
import json
import random
import numpy as np
from collections import namedtuple

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class ConstrainedPolicy(nn.Module):
    """Actor-Critic policy with runtime difference constraints."""
    
    def __init__(self, sz, PARAMS):
        super().__init__()
        # Network architecture
        self.affine1 = nn.Linear((PARAMS['episode_len'] - 1) * sz, PARAMS['hidden_size'])
        self.affine2 = nn.Linear(PARAMS['hidden_size'], PARAMS['hidden_size'])
        self.affine3 = nn.Linear(PARAMS['hidden_size'], PARAMS['hidden_size'])
        self.affine4 = nn.Linear(PARAMS['hidden_size'], PARAMS['hidden_size'])
        
        self.action_head = nn.Linear(PARAMS['hidden_size'], sz)
        self.value_head = nn.Linear(PARAMS['hidden_size'], 1)
        
        # Constraint network
        self.constraint_head = nn.Linear(PARAMS['hidden_size'], 1)
        
        # Lagrangian multiplier
        self.register_parameter('lagrangian_mult', nn.Parameter(torch.tensor(0.0)))
        
        # Buffers
        self.saved_actions = []
        self.rewards = []
        self.saved_constraints = []
        self.constraint_threshold = PARAMS.get('runtime_threshold', 0.1)  # 10% threshold by default
        self.baseline_runtime = None

    def forward(self, x):
        """Forward pass for actor, critic and constraint networks"""
        x = F.relu(self.affine1(x))
        x = x.add(F.relu(self.affine2(x)))
        x = x.add(F.relu(self.affine3(x)))
        x = x.add(F.relu(self.affine4(x)))
        
        action_prob = F.softmax(self.action_head(x), dim=-1)
        state_value = self.value_head(x)
        constraint_value = self.constraint_head(x)
        
        return action_prob, state_value, constraint_value

class RuntimeConstrainedPolicy:

    def select_action(self, state, exploration_rate=0.0, white_list=None):
        state = torch.from_numpy(state.flatten()).float()
        probs, value, _ = self.policy(state)
        
        m = Categorical(probs)
        while True:
            if random.random() < exploration_rate:
                action = torch.tensor(random.randrange(0, len(probs)))
            else:
                action = m.sample()
            if white_list and action not in white_list:
                continue
            break
            
        self.policy.saved_actions.append(SavedAction(m.log_prob(action), value))
        return action.item()

    def update_constraint(self, current_runtime, baseline_runtime):
        """Calculate and store runtime difference constraint"""
        if baseline_runtime == 0:  # Avoid division by zero
            runtime_diff = 0
        else:
            runtime_diff = abs(current_runtime - baseline_runtime) / baseline_runtime
        
        self.policy.saved_constraints.append(runtime_diff)
        return runtime_diff

    def __init__(self, policy, PARAMS):
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=PARAMS.get('learning_rate', 0.01))

        self.lagrangian_lr = PARAMS.get('lagrangian_lr', 0.01)
        self.lagrangian_momentum = 0
        self.max_lagrangian_mult = 100  # Or another appropriate upper bound
        
        # Normalization buffers
        self.returns_mean = 0
        self.returns_std = 1
        self.returns_alpha = PARAMS.get('mean_smoothing', 0.9)
        self.gamma = 0.99


    def update(self):
        """Performs a full constrained policy update with empty tensor handling"""
        
        # Early return if no data
        if len(self.policy.saved_actions) == 0:
            return {
                'total_loss': 0.0,
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'constraint_violation': 0.0,
                'lagrangian_value': self.policy.lagrangian_mult.item(),
                'avg_runtime_diff': 0.0,
                'max_violation': 0.0
            }

        # Convert buffers to tensors and move to device
        device = next(self.policy.parameters()).device
        rewards = torch.tensor(self.policy.rewards, dtype=torch.float32, device=device)
        constraints = torch.tensor(self.policy.saved_constraints, dtype=torch.float32, device=device)
        
        # 1. Compute returns with discounting
        returns = []
        discounted_return = 0
        for r in reversed(rewards):
            discounted_return = r + self.gamma * discounted_return
            returns.insert(0, discounted_return)
        returns = torch.stack(returns)
        
        # 2. Normalize advantages
        values = torch.stack([v for (_, v) in self.policy.saved_actions]).squeeze()
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 3. Calculate constraint violations with safe empty tensor handling
        constraint_violations = F.relu(constraints - self.policy.constraint_threshold)
        
        # Safe calculation of violation metrics
        if constraint_violations.numel() == 0:
            avg_violation = torch.tensor(0.0, device=device)
            max_violation = torch.tensor(0.0, device=device)
        else:
            avg_violation = constraint_violations.mean()
            max_violation = constraint_violations.max()
        
        # 4. Compute losses
        policy_loss = []
        value_loss = []
        
        for (log_prob, value), adv, R in zip(self.policy.saved_actions, advantages, returns):
            policy_loss.append(-log_prob * adv)
            value_loss.append(F.mse_loss(value, R.unsqueeze(0)))
        
        # Handle empty loss lists
        policy_loss_t = torch.stack(policy_loss).mean() if policy_loss else torch.tensor(0.0, device=device)
        value_loss_t = torch.stack(value_loss).mean() if value_loss else torch.tensor(0.0, device=device)
        
        # 5. Combine losses
        constraint_loss_t = constraint_violations.mean() if constraint_violations.numel() > 0 else torch.tensor(0.0, device=device)
        total_loss = policy_loss_t + value_loss_t + self.policy.lagrangian_mult.detach() * constraint_loss_t
        
        # 6. Update networks
        if policy_loss:  # Only update if we have data
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
        
        # 7. Dual update
        with torch.no_grad():
            if constraint_violations.numel() > 0:
                lagrangian_grad = avg_violation
                new_lagrangian = self.policy.lagrangian_mult + self.lagrangian_lr * lagrangian_grad
                self.policy.lagrangian_mult.copy_(torch.clamp(new_lagrangian, min=0, max=self.max_lagrangian_mult))
            elif self.policy.lagrangian_mult > 0:
                # Exponential decay when no violations
                self.policy.lagrangian_mult *= 0.99
        
        # 8. Clear buffers
        self.policy.saved_actions = []
        self.policy.rewards = []
        self.policy.saved_constraints = []
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss_t.item(),
            'value_loss': value_loss_t.item(),
            'constraint_violation': avg_violation.item(),
            'lagrangian_value': self.policy.lagrangian_mult.item(),
            'avg_runtime_diff': constraints.mean().item() if constraints.numel() > 0 else 0.0,
            'max_violation': max_violation.item()
        }

    def update_canonical(self):
        """Performs a full constrained policy update with improved Lagrangian handling"""
        
        # Early return if no data
        if len(self.policy.saved_actions) == 0:
            return {
                'total_loss': 0.0,
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'constraint_violation': 0.0,
                'lagrangian_value': self.policy.lagrangian_mult.item(),
                'avg_runtime_diff': 0.0
            }

        # Convert buffers to tensors and move to device
        device = next(self.policy.parameters()).device
        rewards = torch.tensor(self.policy.rewards, dtype=torch.float32).to(device)
        constraints = torch.tensor(self.policy.saved_constraints, dtype=torch.float32).to(device)
        
        # 1. Compute returns with discounting if needed
        returns = torch.cumsum(rewards.flip(0), 0).flip(0)
        
        # 2. Normalize returns with running statistics
        self.returns_mean = (self.returns_alpha * self.returns_mean + 
                            (1 - self.returns_alpha) * returns.mean())
        self.returns_std = (self.returns_alpha * self.returns_std + 
                        (1 - self.returns_alpha) * returns.std())
        returns = (returns - self.returns_mean) / (self.returns_std + 1e-8)
        
        # 3. Calculate constraint violations with adaptive threshold
        constraint_violations = F.relu(constraints - self.policy.constraint_threshold)
        avg_violation = constraint_violations.mean() if constraint_violations.numel() > 0 else torch.tensor(0.0)
        max_violation = constraint_violations.max() if constraint_violations.numel() > 0 else torch.tensor(0.0)
        
        # 4. Compute losses with separate policy and constraint terms
        policy_loss = []
        value_loss = []
        constraint_loss = []
        
        for (log_prob, value), R, cv in zip(self.policy.saved_actions, returns, constraint_violations):
            advantage = R - value.item()
            
            # Policy gradient term
            policy_loss.append(-log_prob * advantage)
            
            # Value function term
            value_loss.append(F.mse_loss(value, R.unsqueeze(0)))
            
            # Constraint term (track separately)
            constraint_loss.append(cv)
        
        # Stack losses
        policy_loss_t = torch.stack(policy_loss).mean() if policy_loss else torch.tensor(0.0)
        value_loss_t = torch.stack(value_loss).mean() if value_loss else torch.tensor(0.0)
        constraint_loss_t = torch.stack(constraint_loss).mean() if constraint_loss else torch.tensor(0.0)
        
        # 5. Combine losses with Lagrangian multiplier
        total_loss = policy_loss_t + value_loss_t + self.policy.lagrangian_mult * constraint_loss_t
        
        # 6. Update networks if we have data
        if policy_loss and value_loss:
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)  # Gradient clipping
            self.optimizer.step()
        
        # 7. Improved dual update with adaptive learning and momentum
        with torch.no_grad():
            # Adaptive learning rate based on violation severity
            adaptive_lr = self.lagrangian_lr * (1 + 0.5 * (max_violation / (self.policy.constraint_threshold + 1e-8)))
            
            # Update with momentum (smoother changes)
            self.lagrangian_momentum = 0.9 * self.lagrangian_momentum + adaptive_lr * avg_violation
            self.policy.lagrangian_mult += self.lagrangian_momentum
            
            # Clamp with exponential decay toward zero when no violations
            if avg_violation < 1e-4:
                self.policy.lagrangian_mult *= 0.99
            
            self.policy.lagrangian_mult.clamp_(min=0, max=self.max_lagrangian_mult)  # Reasonable upper bound
        
        # 8. Clear buffers
        self.policy.saved_actions = []
        self.policy.rewards = []
        self.policy.saved_constraints = []
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss_t.item(),
            'value_loss': value_loss_t.item(),
            'constraint_violation': avg_violation.item(),
            'lagrangian_value': self.policy.lagrangian_mult.item(),
            'avg_runtime_diff': constraints.mean().item(),
            'max_violation': max_violation.item()
        }

def TrainActorCritic(env, PARAMS=FLAGS, reward_estimator=const_factor_threshold, reward_if_list_func=lambda a: np.mean(a), exp_name='gsm'):
    action_size = len(env.action_spaces[0].names)
    try:
        action_size = len(FLAGS["actions_white_list"])
    except:
        pass
        
    policy = ConstrainedPolicy(action_size, PARAMS=PARAMS)
    agent = RuntimeConstrainedPolicy(policy, PARAMS)
    
    results_ep = list()
    max_ep_reward = -float("inf")
    avg_reward = MovingExponentialAverage(0.95)
    avg_loss = MovingExponentialAverage(0.95)
    avg_constraint_violation = MovingExponentialAverage(0.95)
    ts = time.time()
    
    # Get baseline runtime
    baseline_state = env.reset()

    baseline_runtime = reward_if_list_func(baseline_state[2])
    policy.baseline_runtime = baseline_runtime
    
    for episode in range(1, PARAMS['episodes'] + 1):
        state = env.reset()
        ep_reward = 0
        prev_size = state[1]
        
        prev_runtime = reward_if_list_func(state[2])
         

        action_log = list()
        pat = 0
        best_ep_logs = {'size':0., 'runtime':0, 'sequence':['None']}
        print(f"Trying to beat runtime {baseline_runtime}...")
        while True:
            action = agent.select_action(
                state[0], 
                exploration_rate=PARAMS['exploration'],
                white_list=FLAGS.get("actions_white_list", None)
            )

            action_log.append(env.action_spaces[0].names[action])
            state, reward, done, _ = env.step(action)
            
            # Calculate current runtime and constraint
            current_runtime = reward_if_list_func(state[2])
            #runtime_diff = agent.update_constraint(current_runtime, baseline_runtime)
            
            # Calculate reward
            reward = reward_estimator(
                baseline_m=env.hetero_os_baselines[0], #size
                measured_m=state[1], # size
                baseline_n=baseline_runtime,
                measured_n=current_runtime,
                prev_m=prev_size,
                prev_n=prev_runtime,
                m_norm=prev_size
            )
            # 'Early stopping' for episode with sequence of non-positive (negative) rewards
            if reward <= 0.:
                pat += 1
            if FLAGS.get("patience", 10) <= pat:
                done = True
                
            # Track best results
            size_reward = state[1]
            __size_baseline = env.hetero_os_baselines[0]
            
            prev_size = state[1]
            prev_runtime = current_runtime
            
            

            size_gain = -(size_reward - __size_baseline) * 100/__size_baseline
            runtime_diff_pct = (current_runtime - baseline_runtime) * 100/baseline_runtime
            
            if size_reward < __size_baseline and runtime_diff_pct <= 0.001:
                if size_gain > best_ep_logs['size']:
                    best_ep_logs['size'] = size_gain
                    best_ep_logs['runtime_diff'] = runtime_diff_pct
                    best_ep_logs['sequence'] = action_log.copy()
            
            agent.policy.rewards.append(reward)
            ep_reward += reward
            
            if done:
                print(f"--\nBest results of episode {episode}:")
                print(f"  Size reduction: {best_ep_logs['size']:.2f}%")
                if best_ep_logs['size'] > 0.:
                    print(f"  Runtime difference: {best_ep_logs.get('runtime_diff', 'N/A')}%")
                    print(f"  Action sequence: {best_ep_logs['sequence']}")
                    display_progress_advanced(
                        best_ep_logs['size'],
                        filled_char='█',
                        empty_char=' ',
                        prefix='Size   : ',
                        suffix=f" reduction",
                        color = Colors.BLUE
                    )

                    display_progress_advanced(
                        -best_ep_logs.get('runtime_diff'),
                        filled_char='█',
                        empty_char=' ',
                        prefix='Runtime: ',
                        suffix=f" reduction",
                        color = Colors.RED
                    )
                    #print('\n')
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
        avg_loss.next(update_stats['total_loss'])
        avg_constraint_violation.next(update_stats['constraint_violation'])
        
        # Logging
        if episode == 1 or episode % FLAGS['log_interval'] == 0 or episode == FLAGS['episodes']:
            FLAGS["logger"].save_and_print(
                f"Episode {episode}\t"
                f"Last reward: {ep_reward:.2f}\t"
                f"Avg reward: {avg_reward.value:.2f}\t"
                f"Runtime diff: {update_stats['avg_runtime_diff']:.4f}\t"
                f"Constraint viol: {update_stats['constraint_violation']:.4f}\t"
                f"Lambda: {update_stats['lagrangian_value']:.4f}",
                mode=LogMode.SHORT
            )

        if episode % 50 == 0 or episode == FLAGS['episodes']:
            with open(f"outputs/{exp_name}_{ts}.txt", "w+") as f:
                f.write('\n'.join([str(entry['num']) + ' ' + str(entry['size_gain']) for entry in results_ep]))
            means, succ_rates = calculate_means_and_success_rate(f"outputs/{exp_name}_{ts}.txt", separ=50)
            with open(f"outputs/{exp_name}_{ts}_model.txt", "w+") as f:
                f.write(f"Episode {episode}\n"
                        f"Success rates {succ_rates}\n"
                        f"Last reward: {ep_reward:.2f}\n"
                        f"Avg reward: {avg_reward.value:.2f}\n"
                        f"Runtime diff: {update_stats['avg_runtime_diff']:.4f}\n"
                        f"Constraint viol: {update_stats['constraint_violation']:.4f}\n"
                        f"Lambda: {update_stats['lagrangian_value']:.4f}")
            print("Means:", means, "\nSuccess rates:", succ_rates)
            with open(f"outputs/{exp_name}_{ts}_means.txt", "w+") as f:
                f.write('\n'.join([str(m) for m in means]))

    # Save results
    save_list_of_dicts_to_json(
        data=results_ep, 
        filename=f"runtime_constrained_{int(time.time())}.json"
    )
    
    return avg_reward.value
