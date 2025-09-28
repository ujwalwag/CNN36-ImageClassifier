PyTorch project for **36-class alphanumeric character recognition** (**A–Z + 0–9**) on **28×28** grayscale images (**100,800 samples; ~2,800/class**).  
Implements clean data loaders, preprocessing & EDA, ≥3 visualizations, a modular CNN (≤10 hidden layers), optimizer comparisons, ≥2 training improvements, and full evaluation (accuracy/loss curves, confusion matrix, precision/recall/F1).  
**Target**: test accuracy **> 85%**.

---

## ✨ Highlights
- Dataset ready via `torchvision.datasets.ImageFolder` (folder name = label)
- Reproducible splits (train/val/test)
- Character-safe augmentations (no flips that change semantics)
- Modular CNN with `torchinfo` summaries
- Optimizer sweep: SGD / Adam / AdamW
- Learning-rate scheduling, early stopping, dropout, L2
- Plots: train/val curves, confusion matrix, misclassified samples
- Metrics: accuracy, precision, recall, F1 (macro & weighted)
