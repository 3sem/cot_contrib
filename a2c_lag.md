Here's the mathematical formulation of the key components in your code, particularly focusing on the **loss function with Lagrangian multiplier** for runtime degradation constraints:

---

### **1. Policy Network (Actor-Critic)**
The neural network outputs:
- **Action probabilities** (Actor):  
  \[
  \pi(a|s) = \text{softmax}(W_A \cdot h_4), \quad h_i = \text{ReLU}(W_i \cdot h_{i-1} + b_i)
  \]
  where \( h_0 = \text{flatten}(s) \), and \( W_A \) is the actor head.

- **State value** (Critic):  
  \[
  V(s) = W_C \cdot h_4 + b_C
  \]
  where \( W_C \) is the critic head.

---

### **2. Loss Components**
#### **(a) Policy Loss (Actor)**
\[
\mathcal{L}_{\text{policy}} = -\sum_t \log \pi(a_t|s_t) \cdot A_t
\]
where \( A_t = R_t - V(s_t) \) is the **advantage** (reward-to-go \( R_t \) minus critic's value estimate).

#### **(b) Value Loss (Critic)**
Uses **smooth L1 loss** (Huber loss) between predicted and actual returns:
\[
\mathcal{L}_{\text{value}} = \sum_t \text{smooth\_L1}(V(s_t), R_t)
\]

#### **(c) Constraint Loss (Lagrangian)**
Penalizes runtime degradation violations:
\[
\mathcal{L}_{\text{constraint}} = \lambda \cdot \max\left(0, \frac{\Delta T}{T_{\text{prev}}} - \tau\right)
\]
where:
- \( \lambda \): Lagrangian multiplier (learned).
- \( \Delta T = T_{\text{new}} - T_{\text{prev}} \): Runtime change.
- \( \tau \): Constraint threshold (e.g., 0.1 for 10% degradation).

---

### **3. Total Loss**
\[
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{policy}} + \mathcal{L}_{\text{value}} + \mathcal{L}_{\text{constraint}}
\]

---

### **4. Lagrangian Multiplier Update**
Gradient ascent on the multiplier to enforce constraints:
\[
\lambda \leftarrow \lambda + \alpha_\lambda \cdot \left(\frac{\Delta T}{T_{\text{prev}}} - \tau\right)^+
\]
where \( \alpha_\lambda \) is the learning rate for \( \lambda \), and \( (\cdot)^+ = \max(0, \cdot) \).

---

### **5. Optimization Steps**
1. **Forward pass**: Compute \( \pi(a|s) \) and \( V(s) \).
2. **Rollout**: Collect trajectories with runtime changes \( \Delta T \).
3. **Loss computation**: Calculate \( \mathcal{L}_{\text{total}} \).
4. **Backward pass**: Update policy/value weights via gradient descent:
   \[
   \theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}_{\text{total}}
   \]
5. **Multiplier update**: Adjust \( \lambda \) via gradient ascent.

---

### **Key Equations Summary**
| Component           | Formula                                                                 |
|---------------------|-------------------------------------------------------------------------|
| **Policy Loss**     | \( -\sum_t \log \pi(a_t|s_t) \cdot (R_t - V(s_t)) \)                   |
| **Value Loss**      | \( \sum_t \text{smooth\_L1}(V(s_t), R_t) \)                             |
| **Constraint Loss** | \( \lambda \cdot \max\left(0, \frac{\Delta T}{T_{\text{prev}}} - \tau\right) \) |
| **Total Loss**      | \( \mathcal{L}_{\text{policy}} + \mathcal{L}_{\text{value}} + \mathcal{L}_{\text{constraint}} \) |
| **Multiplier Update** | \( \lambda \leftarrow \lambda + \alpha_\lambda \cdot \left(\frac{\Delta T}{T_{\text{prev}}} - \tau\right)^+ \) |

---

### **Pseudocode for `finish_episode`**
```python
def finish_episode(model, optimizer):
    # Compute returns R_t
    R = 0
    returns = []
    for r in reversed(model.rewards):
        R += r
        returns.insert(0, R)
    
    # Normalize returns
    returns = (returns - μ) / (σ + ε)  # μ, σ: moving averages
    
    # Constraint violation
    degradation = relu(ΔT / T_prev - τ)  # τ: threshold
    constraint_violation = mean(degradation)
    
    # Losses
    policy_loss = -sum(log_prob * (R_t - V(s_t)))
    value_loss = sum(smooth_L1(V(s_t), R_t))
    constraint_loss = λ * constraint_violation
    
    # Total loss
    loss = policy_loss + value_loss + constraint_loss
    
    # Update
    loss.backward()
    optimizer.step()
    λ += α_λ * constraint_violation  # Project λ ≥ 0
```

This formulation ensures the RL agent optimizes for performance while respecting runtime degradation constraints.
