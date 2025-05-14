Formulating compiler optimization sequence search as a Mixed Integer Linear Programming (MILP) problem involves modeling the selection and ordering of optimizations as discrete decisions while capturing their effects on performance (or other metrics) via linear constraints. Here's a structured approach:

---

### **Key Components of the MILP Formulation**

1. **Decision Variables**:
   - **Binary variables** \( x_{o,p} \): 
     - \( x_{o,p} = 1 \) if optimization \( o \) is placed at position \( p \) in the sequence.
     - \( x_{o,p} = 0 \) otherwise.
   - **Continuous variables** \( t_p \): 
     - Represents the "state" of the program (e.g., execution time, code size) after applying the optimization at position \( p \).

2. **Objective Function**:
   - Minimize the final program metric (e.g., execution time \( t_{N} \), where \( N \) is the sequence length):
     \[
     \text{minimize} \quad t_{N}
     \]
   - Alternatively, maximize a performance metric (e.g., speedup).

3. **Constraints**:
   - **Sequence Constraints**:
     - Each position \( p \) has at most one optimization:
       \[
       \sum_{o \in O} x_{o,p} \leq 1 \quad \forall p \in \{1, \dots, N\}
       \]
     - Each optimization \( o \) is applied at most once:
       \[
       \sum_{p=1}^{N} x_{o,p} \leq 1 \quad \forall o \in O
       \]
   - **Dependency Constraints** (if optimizations have prerequisites):
     - If optimization \( o_2 \) requires \( o_1 \), enforce:
       \[
       \sum_{p=1}^{k} x_{o_1,p} \geq x_{o_2,k+1} \quad \forall k \in \{1, \dots, N-1\}
       \]
   - **State Transition Constraints**:
     - Model the effect of optimization \( o \) at position \( p \) on the program state \( t_p \):
       \[
       t_p = t_{p-1} + \sum_{o \in O} c_{o} \cdot x_{o,p} \quad \forall p \in \{1, \dots, N\}
       \]
       where \( c_o \) is the estimated impact of \( o \) on the metric (e.g., reduction in execution time).
   - **Resource Constraints** (optional):
     - Limit total compilation time or memory usage:
       \[
       \sum_{o,p} w_o \cdot x_{o,p} \leq W_{\text{max}}
       \]
       where \( w_o \) is the cost of applying \( o \).

4. **Additional Considerations**:
   - **Nonlinear Effects**: If optimizations interact nonlinearly, piecewise linear approximations or auxiliary variables may be needed.
   - **Bounds**: Define bounds for \( t_p \) (e.g., \( t_p \geq 0 \)).

---

### **Example MILP Formulation**
Let \( O = \{o_1, \dots, o_M\} \) be the set of optimizations and \( N \) the sequence length.

**Variables**:
- \( x_{o,p} \in \{0,1\} \) for all \( o \in O, p \in \{1, \dots, N\} \).
- \( t_p \in \mathbb{R}^+ \) for all \( p \in \{1, \dots, N\} \).

**Objective**:
\[
\text{minimize} \quad t_N
\]

**Constraints**:
1. **Sequence validity**:
   \[
   \sum_{o \in O} x_{o,p} \leq 1 \quad \forall p
   \]
2. **No duplicate optimizations**:
   \[
   \sum_{p=1}^N x_{o,p} \leq 1 \quad \forall o
   \]
3. **State transition**:
   \[
   t_p = t_{p-1} - \sum_{o \in O} \Delta t_o \cdot x_{o,p} \quad \forall p \geq 1
   \]
   (where \( \Delta t_o \) is the time reduction from \( o \), and \( t_0 \) is the baseline time).
4. **Dependencies** (if \( o_2 \) requires \( o_1 \)):
   \[
   \sum_{p=1}^k x_{o_1,p} \geq x_{o_2,k+1} \quad \forall k \in \{1, \dots, N-1\}
   \]

---

### **Challenges and Extensions**
1. **Large Search Space**: For long sequences, use symmetry-breaking constraints or heuristic bounds.
2. **Interaction Effects**: Model pairwise interactions with terms like \( x_{o_1,p} \cdot x_{o_2,p+1} \) (requires linearization).
3. **Dynamic Programming**: Alternatively, frame as a shortest-path problem (e.g., with a state graph).

This formulation allows solvers like CPLEX or Gurobi to search for optimal sequences while respecting dependencies and resource limits. For practicality, the model can be simplified by focusing on a subset of critical optimizations.
