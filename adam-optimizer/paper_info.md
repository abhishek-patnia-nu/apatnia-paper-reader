# Adam: A Method for Stochastic Optimization

- **Authors:** Diederik P. Kingma (University of Amsterdam, OpenAI), Jimmy Lei Ba (University of Toronto)
- **arXiv:** [1412.6980](https://arxiv.org/abs/1412.6980)
- **Published:** ICLR 2015 (submitted Dec 2014, last revised Jan 2017)
- **Topics:** Stochastic optimization, adaptive learning rates, moment estimation, deep learning

## Abstract

Introduces Adam (Adaptive Moment Estimation), an optimizer that computes per-parameter adaptive learning rates from running averages of the first moment (mean) and second moment (uncentered variance) of the gradient. Combines AdaGrad's strength on sparse gradients with RMSProp's strength on non-stationary objectives. Includes bias correction for the moment estimates and a convergence proof with O(√T) regret. Also introduces AdaMax, a variant based on the infinity norm.

## Paper Structure

| Section | Topic | Status |
|---------|-------|--------|
| 1 | Introduction | done |
| 2 | Algorithm | |
| 2.1 | Adam's Update Rule | |
| 3 | Initialization Bias Correction | |
| 4 | Convergence Analysis | |
| 5 | Related Work (RMSProp, AdaGrad) | |
| 6 | Experiments | |
| 6.1 | Logistic Regression | |
| 6.2 | Multi-Layer Neural Networks | |
| 6.3 | Convolutional Neural Networks | |
| 6.4 | Bias-Correction Term | |
| 7 | Extensions (AdaMax, Temporal Averaging) | |
| 8 | Conclusion | |
| App | Convergence Proof | |
