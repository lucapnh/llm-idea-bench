# Title: Tree of Thoughts: Deliberate Problem Solving with Large Language Models

## Keywords

large language models; tree search; deliberate reasoning; chain-of-thought; heuristic evaluation; planning

## TL;DR

The paper proposes “Tree of Thoughts” (ToT), an inference framework that lets LLMs explore, evaluate, and backtrack across multiple reasoning paths, yielding large gains on tasks that require search and planning—e.g., boosting GPT-4’s success on the Game of 24 from 4% (CoT) to 74%.

## Abstract

Language models typically decode left-to-right, committing to one continuation at a time; this makes them brittle on problems that benefit from exploration, lookahead, or backtracking. The authors introduce Tree of Thoughts (ToT), a general inference framework that elevates the unit of reasoning from tokens to “thoughts” (coherent textual steps) and frames problem solving as search over a tree of partial solutions. At each node, the model (i) **generates** candidate thoughts, (ii) **evaluates** states via self-assessment (scalar “value” or comparative “vote”), and (iii) **searches** using algorithms such as breadth-first search (BFS) or depth-first search (DFS), enabling lookahead and backtracking. This modular recipe contrasts with standard chain-of-thought and self-consistency, which lack local branching and global planning. Empirically, ToT improves GPT-4 on three tasks designed to require search: Game of 24 (74% vs. 4% CoT), Creative Writing (higher GPT-4 judged coherence), and Mini Crosswords (word-level accuracy \~60%, solving 4/20 puzzles), with ablations analyzing breadth, pruning, and backtracking. While ToT increases compute and cost relative to CoT/IO, its components are plug-and-play and expose performance–cost trade-offs. Conceptually, ToT augments the “System 1” decoding of LLMs with a “System 2” style planning process that is expressed in language and guided by the model’s own evaluative heuristics, pointing toward more capable and interpretable decision-making with LLMs.

**Key details & findings:** The paper formalizes thought decomposition, two thought generators (i.i.d. CoT sampling vs. sequential “propose” prompting), and two evaluators (independent valuation vs. cross-state voting), and instantiates ToT with BFS/DFS; e.g., BFS with b=5 and sure/maybe/impossible scoring drives the 24-game gains. Costs rise (e.g., \~5.5k completion tokens per 24-game case), but improve success beyond best-of-100 CoT; limitations include resource overhead and the need for better pruning heuristics in DFS.
