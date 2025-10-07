# Title: Toolformer: Language Models Can Teach Themselves to Use Tools&#x20;

## Keywords

self-supervised tool use; API-augmented language modeling; zero-shot generalization; retrieval & QA; programmatic reasoning; multilingual translation&#x20;

## TL;DR

Toolformer shows that a language model can *teach itself* to call external tools via simple APIs—deciding what to call, when, and how—using only a few demonstrations and a self-supervised filtering objective. The resulting GPT-J–based model achieves large zero-shot gains on factual, mathematical, temporal, and QA tasks without hurting core language-modeling ability.

## Abstract

Large language models excel at instruction following yet still falter on basic capabilities such as exact arithmetic, factual lookup, or temporal awareness. Toolformer addresses this gap by enabling a model to learn when and how to invoke external tools—calculator, question answering, Wikipedia search, machine translation, and a calendar—through a lightweight, self-supervised procedure. Starting from a handful of human-written demonstrations per API, the base LM proposes many candidate API calls within raw pretraining text, executes them, and retains only those that *reduce the model’s next-token loss*; the model is then finetuned on the original text interleaved with these helpful calls and results. At inference, generation proceeds normally until the model emits a special “→” token, at which point the corresponding API is queried and its result is inserted before decoding resumes. This keeps inputs/outputs purely textual, preserves generality, and requires no task-specific supervision.

Empirically, a 6.7B-parameter GPT-J Toolformer delivers strong zero-shot improvements: on LAMA subsets (e.g., T-REx) it surpasses GPT-J baselines and even larger models by leveraging the QA tool; on math word problems it more than doubles accuracy by invoking the calculator; on open-domain QA it beats same-size baselines using Wikipedia search (though still trails GPT-3 175B); on temporal queries, it gains substantially on DATESET via the calendar; and multilingual QA benefits from automatic question translation. Crucially, language-model perplexity remains unchanged when API calls are disabled, indicating no loss of base modeling ability. Scaling studies suggest effective tool use emerges around \~775M parameters. Limitations include lack of multi-step tool chaining, non-interactive search, prompt sensitivity, and sample inefficiency for some tools—pointing to promising directions for iterative bootstrapping and interactive APIs.
