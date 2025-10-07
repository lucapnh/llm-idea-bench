# Title: ReAct: Synergizing Reasoning and Acting in Language Models

## Keywords

reasoning-and-acting, chain-of-thought, decision-making agents, retrieval-augmented LM, few-shot prompting

## TL;DR

ReAct prompts language models to interleave free-form reasoning with task-specific actions, yielding more grounded, interpretable trajectories and better performance across QA, fact verification, and interactive environments. It reduces hallucinations versus pure Chain-of-Thought and outperforms act-only and learned agents on ALFWorld and WebShop.

## Abstract

Large language models (LLMs) are typically studied either for *reasoning* (e.g., chain-of-thought prompting) or for *acting* (planning/execution in an environment). This paper introduces **ReAct**, a prompting paradigm where LLMs generate \*\*interleaved reasoning traces (“Thoughts”) and environment **actions**, enabling each to inform the other: reasoning guides what to do next; actions fetch external observations to ground subsequent reasoning. Operationally, ReAct augments the agent’s action space with language “thought” steps and alternates thought–action–observation loops, instantiated with few-shot prompts over PaLM and GPT-3.

The authors evaluate on four benchmarks: HotpotQA (multi-hop QA), FEVER (fact verification), ALFWorld (text-based household tasks), and WebShop (goal-driven online shopping). On knowledge-intensive tasks, ReAct interacts with a minimalist Wikipedia API (search, lookup, finish) to retrieve just-in-time evidence. It matches or improves over strong CoT and act-only baselines and, when combined with self-consistent CoT, achieves the best prompting results (e.g., FEVER 64.6% with CoT-SC→ReAct; HotpotQA 35.1% with ReAct→CoT-SC), illustrating complementary strengths of internal reasoning and external grounding.  ReAct also exhibits markedly fewer hallucinations than CoT (0% vs. 56% among analyzed failures), though it can incur more reasoning-loop errors when retrieval is uninformative, highlighting a factuality–flexibility trade-off.

On decision-making, ReAct’s sparse thoughts improve long-horizon control: on ALFWorld it reaches 71% success (best trial), surpassing act-only prompting (45%) and the BUTLER imitation agent (37%); on WebShop it improves success rate by 10 absolute points over prior IL/IL+RL systems (40.0% vs. ≤30.1%).  Finetuning smaller models on \~3k ReAct trajectories further closes the gap, with finetuned ReAct beating larger prompt-only baselines.  Overall, ReAct offers a simple, general, and interpretable way to couple reasoning and acting, reducing hallucination, enhancing robustness, and providing transparent decision traces across diverse tasks.
