# Title: AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation

## Keywords

multi-agent systems; conversable agents; conversation programming; tool use; RAG; evaluation

## TL;DR

AutoGen is an open-source framework for building LLM applications as conversations among specialized agents—LLM-, tool-, and human-backed—that coordinate via unified interfaces and flexible control. It demonstrates strong results across math, retrieval-augmented QA/code, embodied decision-making, and safeguarded coding, while reducing developer effort. &#x20;

## Abstract

The paper introduces **AutoGen**, a general framework for constructing large-language-model (LLM) applications through *multi-agent conversations*. Two core ideas underpin the design. First, **conversable agents**: modular, customizable components that can be backed by LLMs, humans, tools, or any combination thereof, and that interact through a unified send/receive/generate-reply interface. This design supports role specialization (e.g., writer, executor, validator) and enables human-in-the-loop operation as needed. Second, **conversation programming**: developers specify application behavior by defining agents and steering their interactions and control flow using both natural language (prompts, role instructions) and code (termination conditions, tool execution policies, custom auto-reply functions). Together, these abstractions enable static or dynamic topologies, including group chats managed by an orchestrator that selects the next speaker and broadcasts messages.  &#x20;

Empirically, AutoGen is instantiated in six applications: autonomous and human-in-the-loop math problem solving; retrieval-augmented chat for QA and code; text-world decision making (ALFWorld) with a commonsense “grounding” agent; multi-agent coding with a safeguard; dynamic group chat; and conversational chess with a rules-checking board agent. Across benchmarks, the framework yields competitive or improved performance—for example, superior success rates on challenging MATH problems versus strong baselines; higher QA F1/recall via an interactive retrieval loop; gains on ALFWorld by adding a grounding agent; and better unsafe-code detection in the safeguarded multi-agent coding setup—while often reducing orchestration code. &#x20;

The authors position AutoGen as a reusable infrastructure for rapid experimentation with multi-agent designs, emphasizing modularity, programmability, and human oversight. They also discuss limitations and future directions, including designing cost-effective agent topologies, building more capable tool-using agents, and addressing safety (e.g., autonomy, execution risks, bias) as multi-agent workflows scale. &#x20;
