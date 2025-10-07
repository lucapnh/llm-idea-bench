# Title: Self-Consistency Improves Chain-of-Thought Reasoning in Language Models

## Keywords

chain-of-thought prompting; self-consistency; reasoning in LLMs; decoding strategies; few-shot learning

## TL;DR

Instead of trusting a single reasoning path, sample many chains of thought and pick the answer most supported by them. This simple “self-consistency” decoding step yields large accuracy gains on arithmetic and commonsense reasoning benchmarks across multiple large language models.&#x20;

## Abstract

This paper proposes **self-consistency**, a lightweight decoding strategy that strengthens chain-of-thought (CoT) prompting for complex reasoning tasks. Rather than relying on greedy decoding to produce a single rationale and answer, the method (i) samples a diverse set of reasoning paths from a language model, and then (ii) **marginalizes over these rationales** by selecting the final answer that appears most consistently across samples. The approach exploits the intuition that correct solutions to multi-step problems can be reached via multiple, diverse lines of reasoning, whereas incorrect solutions are less likely to agree. Self-consistency is **model- and task-agnostic**, requires **no additional training, supervision, or auxiliary verifiers**, and functions as a “self-ensemble” over one model’s generations. Empirically, across arithmetic and commonsense benchmarks—including GSM8K, SVAMP, AQuA, StrategyQA, and ARC-Challenge—the method delivers substantial absolute accuracy gains (e.g., **+17.9% on GSM8K** with PaLM-540B and GPT-3 variants) and often sets new state of the art within the prompted (non-finetuned) regime. The gains increase with model scale and with the number of sampled reasoning paths; majority voting over answers performs on par with probability-weighted aggregation, highlighting robustness to calibration imperfections. Beyond headline results, the paper shows self-consistency (a) outperforms sample-and-rank, beam search, and prompt-ensemble baselines; (b) improves performance even when CoT would otherwise hurt relative to standard prompting; and (c) correlates consistency with accuracy, offering a pragmatic confidence signal. The main trade-off is **inference cost** due to multiple samples, though performance often saturates with tens of paths. Overall, self-consistency offers a simple, widely applicable upgrade to CoT prompting that reliably boosts reasoning without additional training.&#x20;
