# Title: GNNExplainer: Generating Explanations for Graph Neural Networks

## Keywords

graph neural networks, explainability, mutual information, subgraph masking, feature attribution, prototypes

## TL;DR

GNNExplainer is a model-agnostic method that reveals *why* a trained GNN made a prediction by extracting a compact subgraph and small set of node features that maximally preserve the model’s output. It outperforms gradient and attention baselines on synthetic and real datasets, improving explanation accuracy by up to 43%.

## Abstract

Graph Neural Networks (GNNs) achieve state-of-the-art performance on node, link, and graph prediction tasks but remain difficult to interpret because they couple relational structure with rich node features. GNNExplainer addresses this gap with a general, post-hoc approach that explains any GNN’s prediction by identifying (i) a small, connected subgraph within the model’s computation graph and (ii) a sparse subset of node feature dimensions that jointly account for the output. Formally, explanations are optimized to maximize the mutual information between the prediction and a distribution over subgraphs and features. The method realizes this objective via a continuous mask on edges (mean-field Bernoulli relaxation) and a learnable feature mask with a reparameterization trick, combined with regularizers that favor discrete, succinct explanations. Beyond single-instance rationales, GNNExplainer aggregates aligned instance-level explanations to derive class-level prototypes that highlight recurring structural motifs. Experiments on synthetic benchmarks (BA-Shapes, BA-Community, Tree-Cycles, Tree-Grid) and real datasets (MUTAG molecules, Reddit threads) show that the method recovers ground-truth motifs (e.g., houses, cycles, grids) and domain-relevant structures (e.g., NO₂ groups, aromatic rings; star-like reply patterns) while outperforming gradient saliency and attention-based proxies by up to 43% in explanation accuracy. The approach is architecture-agnostic and extends naturally to link and graph classification without retraining, offering a practical tool for debugging, trust, and scientific insight in relational learning.
