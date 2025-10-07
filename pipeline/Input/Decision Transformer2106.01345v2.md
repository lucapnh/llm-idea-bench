# Title: Decision Transformer: Reinforcement Learning via Sequence Modeling

## Keywords

offline reinforcement learning; transformers; sequence modeling; return-conditioned policies; long-term credit assignment

## TL;DR

A GPT-style transformer is trained to autoregress over trajectories conditioned on a desired return, directly outputting actions without dynamic programming. This “Decision Transformer” matches or surpasses strong offline RL baselines on Atari and D4RL, and excels when rewards are sparse or credit assignment is long-horizon.

## Abstract

This paper reframes reinforcement learning (RL) as conditional sequence modeling: instead of fitting value functions or computing policy gradients, a causally masked transformer is trained to predict actions in trajectories interleaving return-to-go, state, and action tokens. At test time, the model is prompted with a target return and recent context, then generates the next action autoregressively; the target return is decremented by observed rewards as the episode unfolds. The architecture uses modality-specific embeddings, a learned timestep (episodic) encoding, and a GPT backbone; context windows cover the last $K$ steps (3K tokens) and training minimizes action prediction loss. This sidesteps bootstrapped TD backups and their instability, leveraging self-attention for direct credit assignment over long horizons.

Empirically, Decision Transformer is competitive with or superior to state-of-the-art offline RL methods. On 1% DQN-replay Atari, it matches or beats REM, QR-DQN, and behavior cloning, and is comparable to Conservative Q-Learning in 3/4 games under gamer-normalized scoring. On D4RL locomotion (HalfCheetah, Hopper, Walker) and a sparse-reward Reacher, it achieves the highest or near-highest normalized returns across Medium, Medium-Replay, and Medium-Expert settings. The model also learns effective policies in a long-horizon Key-to-Door task from random-walk data and remains robust under delayed-reward (sparse) variants where TD methods collapse. Finally, longer context windows significantly improve Atari performance, underscoring the benefits of sequence modeling over single-step conditioning. Together, these results support sequence modeling as a simple, scalable paradigm for offline RL, with particular advantages in sparse-reward and long-horizon regimes.
