# RC-evaluation-analysis

This repository contains the code for the following paper:

Elisa Bassignana, Rob van der Goot, and Barbara Plank. 2024. What’s wrong with your model? A Quantitative Analysis of Relation Classification. In Proceedings of the 12th Joint Conference on Lexical and Computational Semantics (*SEM 2024), Mexico City, Mexico, 2024.

In order to reproduce the experiments in the paper
- download the CrossRE dataset from [HERE](https://github.com/mainlp/CrossRE)
- install the dependency packages using the command `pip install -r requirements.txt`
- run the command `./run.sh`

## Cite
If you use the code from this repository please include the following reference:
```
@inproceedings{bassignana-etal-2024-what,
title = {What’s wrong with your model? {A} {Q}uantitative {A}nalysis of {R}elation {C}lassification},
abstract = "With the aim of improving the process of developing new state-of-the-art (SOTA) architectures, we propose to enrich the procedure with a preliminary quantitative analysis: First, explore weaknesses by analyzing the hard cases where the existing model fails, and then target the improvement based on those. Interpretable evaluation has received little attention for structured prediction tasks. Therefore we propose the first in-depth analysis suite for Relation Classification (RC), and show its effectiveness through a case study. We propose a set of potentially influential attributes to focus on (e.g., entity distance, sentence length). Then, we bucket our datasets based on these attributes, and weight the importance of them through correlations. This allows us to identify highly challenging scenarios for the RC model. By exploiting the findings of our analysis, with a simple, but carefully targeted adjustment to our architecture, we effectively improve the performance over the baseline by >3 Micro-F1.",
author = "Bassignana, Elisa and van der Goot, Rob and Plank, Barbara",
booktitle = "Proceedings of the 12th Joint Conference on Lexical and Computational Semantics (*SEM 2024)",
publisher = "Association for Computational Linguistics",
address = "Mexico City, Mexico",
year = "2024",
language = "English"
}
```
