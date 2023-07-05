# "An Improved Distributed Sampling PPO Algorithm Based on Beta Policy for Continuous Global Path Planning Scheme" source code

### Paper Abstract:

​	Traditional path planning is mainly utilized for path planning in discrete action space, which results in incomplete ship navigation power propulsion strategies during the path search process. Moreover, reinforcement learning experiences low success rates due to its unbalanced sample collection and unreasonable design of reward function. In this paper, an environment framework is designed, which is constructed using the Box2D physics engine and employs a reward function, with the distance between the agent and arrival point as the main, and the potential field superimposed by boundary control, obstacles, and arrival point as the supplement. We also employ the state-of-the-art PPO (Proximal Policy Optimization) algorithm as a baseline for global path planning to address the issue of incomplete ship navigation power propulsion strategy. Additionally, a Beta policy-based distributed sample collection PPO algorithm is proposed to overcome the problem of unbalanced sample collection in path planning by dividing sub-regions to achieve distributed sample collection. The experimental results show the following: (1) The distributed sample collection training policy exhibits stronger robustness in the PPO algorithm; (2) The introduced Beta policy for action sampling results in a higher path planning success rate and reward accumulation than the Gaussian policy at the same training time; (3) When planning a path of the same length, the proposed Beta policy-based distributed sample collection PPO algorithm generates a smoother path than traditional path planning algorithms, such as A*, IDA*, and Dijkstra.

### Paper Link: 

[An Improved Distributed Sampling PPO Algorithm Based on Beta Policy for Continuous Global Path Planning Scheme](https://www.mdpi.com/1424-8220/23/13/6101#metrics)

### Errata:

​	In **Table 3 Hyperparameters**, due to text editing error, the parameter is same with the Table2. It is now corrected as follows:

| Parameter                     |                  | Value       |
| ----------------------------- | ---------------- | ----------- |
| Actor training episode        | $E_{\pi}$        | $600$       |
| Actor learning rate           | $lr_\pi$         | $1e-4$      |
| Actor learning rate decay     | $Step{LR}_\pi$   | $[310,410]$ |
| Actor gradient clipping (L2)  | $cn_\pi$         | $1$         |
| Critic training Episode       | $E_v$            | $600$       |
| Critic learning rate          | $lr_v$           | $3e-4$      |
| Critic learning rate decay    | $Step{LR}_v$     | $[300,400]$ |
| Critic gradient clipping (L2) | $gn_v$           | $20$        |
| Discount factor               | $\gamma$         | $0.98$      |
| Clip ratio                    | $0.1$            | $0.1$       |
| Episode training step         | $train_{iter}$   | $20$        |
| GAE lambda                    | $\lambda$        | $0.97$      |
| Adapt KL target               | $KL_{tg}$        | $0.01$      |
| Batch Size                    | $batch_{iter}$   | $128$       |
| Max Step                      | $max_{timestep}$ | $512$       |


