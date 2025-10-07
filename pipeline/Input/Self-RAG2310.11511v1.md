# Title: SELF-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection

## Keywords

retrieval-augmented generation (RAG); reflection tokens; controllable decoding; adaptive retrieval; factuality & citation accuracy; large language models

## TL;DR

SELF-RAG trains one LM to decide **when** to retrieve, **how** to use evidence, and **how well** its outputs are supported—via reflection tokens—yielding sizable gains in factuality, controllability, and citation precision across QA, reasoning, and long-form settings.

## Abstract

Large language models (LLMs) often hallucinate when they rely solely on parametric knowledge. Retrieval-augmented generation helps, but typical pipelines always prepend a fixed number of passages and never revisit output quality, which can introduce irrelevant context and weak attributions. SELF-RAG proposes an end-to-end alternative that teaches a single LM to **retrieve on demand**, **generate with evidence**, and **critique its own segments** using special reflection tokens. The model predicts (i) `Retrieve` to trigger retrieval when helpful, and three critique tokens per segment: `ISREL` (passage relevance), `ISSUP` (degree of support), and `ISUSE` (perceived utility). These tokens both supervise learning and control decoding, enabling soft or hard constraints and segment-level beam re-ranking to balance evidence support vs. fluency at test time. 

Training proceeds in two stages: a critic model is distilled from GPT-4 judgments to label training data with reflection tokens; then the generator LM is trained with an expanded vocabulary to emit both task tokens and reflections. Retrieval text is masked from the loss, and the final system no longer needs the critic at inference. The training set spans \~150k instruction-following and knowledge-intensive examples; Contriever-MS MARCO provides passages. 

Across six benchmarks—including PopQA, TriviaQA, PubHealth, ARC-Challenge, biographies (FactScore), and ASQA—SELF-RAG (7B/13B) achieves the best non-proprietary results and improves citation precision/recall for long-form answers; it even surpasses ChatGPT in citation precision on ASQA. Ablations confirm that both on-demand retrieval and critique-guided decoding are essential, and inference weights allow practitioners to trade off MAUVE fluency against evidence support. Overall, SELF-RAG delivers a controllable, verifiable generator that raises factual accuracy without sacrificing versatility.
