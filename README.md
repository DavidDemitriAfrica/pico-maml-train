# 🚀 Pico MAML Train

Pico MAML Train is a fork of **Pico Train**, extended with meta‑learning capabilities for language model pretraining. It retains all the original lightweight design and rich checkpointing features of Pico Train, and adds **Model‑Agnostic Meta‑Learning (MAML)** using **Subset Masked Language Modeling Tasks (SMLMT)** to help you pretrain transformers to rapidly adapt to downstream tasks with just a few gradient steps. This repository is written for the final project in partial completion of the MPhil in Advanced Computer Science at the University of Cambridge.

1. **Sample a small support set** of K token positions and mask them.
2. **Run N inner-loop SGD steps** on the support set, updating only a lightweight classification head.
3. **Evaluate adaptation** on a disjoint query set of tokens to compute an outer-loop meta-loss, which updates the shared model backbone.
4. **Optionally blend** this meta-loss with the standard autoregressive objective according to a configurable hybrid ratio.

**Example:** Given the sentence:

> “The cat sat on the mat.”

* **Support set masks:** “\_ cat \_ on \_ mat.”
* **Inner-loop:** Update classifier head to recover “The” and “sat.”
* **Query set mask:** “The \_ sat \_ the \_.”
* **Outer-loop:** Use performance on “sat” and “mat” to shape backbone representations.

This pretraining paradigm ideally encourages the model to learn representations that are not only fluent but also **few-shot adaptable**, improving sample efficiency and downstream performance on tasks like masked token classification, infilling, and prompt-based learning.


## 🎯 What’s New in This Fork?

* **MAML‑style Pretraining**: Inner‑loop/outer‑loop optimization that alternates between task‑specific adaptation steps and shared backbone updates.
* **SMLMT Hybrid Training**: Balance between autoregressive language modeling and N‑way K‑shot meta‑learning via a configurable `hybrid_ratio`.
* **Classification Head**: A lightweight token‑level classification head for inner‑loop support/query tasks, fully configurable (hidden dims, layers, init).
* **Granular Logging of Meta‑Metrics**: Track support/query accuracies, inner‑loop gradient norms, meta/AR step counts, and head‐parameter statistics alongside standard LM metrics.
* **Seamless Resume & Checkpointing**: Carry over both outer and inner optimizer states, learning rate schedulers, and head classifier weights when resuming from checkpoints.

---

## 🔑 Key Features

1. **Lightning Fabric Integration**

   * Distributed, multi‑node, multi‑GPU training support with minimal code changes.
   * Built‑in logging hooks for both TensorBoard and Weights & Biases.

2. **MAML Inner‑Loop**

   * Configurable number of inner steps (`inner_steps`), learning rate (`inner_lr`), way (`n_way`), and shot (`k_shot`).
   * Freezes backbone during adaptation and only updates the classification head in support phase.

3. **Hybrid AR + Meta Learning**

   * At each batch, stochastically choose between autoregressive LM loss and meta‑learning loss based on `hybrid_ratio`.

4. **Comprehensive Checkpointing**

   * Saves outer model + optimizer + scheduler, inner optimizer state, and meta‑learning histories.
   * Hooks for capturing activations and gradients for interpretability (compatible with **pico‑analyze**).

5. **Config‑Driven**

   * All hyperparameters, model sizes, data paths, and training settings defined in YAML config.
   * Example configs for pure LM, pure MAML, and hybrid SMLMT training included.

---

## 🛠️ Training Philosophy

We believe in:

* **Reproducible Meta‑Learning**: Identical architectures and data orders across runs, isolating the effect of inner‑loop adaptation.
* **Flexible Task Definitions**: Easily swap between AR, few‑shot classification, or any custom inner‐loop objective.
* **Rich Learning Dynamics**: Log everything from loss curves to layer‑wise gradient norms to support in‑depth analysis.

---

## 📦 Resources

* **Pre‑tokenized Dolma Dataset** on Hugging Face
* **Pico‑Analyze** for post‑hoc interpretability of checkpoints
* **Example Configs** under `configs/` for SMLMT pre-training at various sizes

---

## 🏃 Quick Start

1. **Clone the Fork**

   <!-- ```bash
   git clone https://github.com/DavidDemitriAfrica/pico-maml-train
   cd pico-maml-train
   ``` -->

2. **Configure Environment**

   ```bash
   export HF_TOKEN=<your_hf_token>
   export WANDB_API_KEY=<your_wandb_key>
   ```

3. **Install Dependencies**

   ```bash
   source setup.sh
   ```

   This script installs dependencies, sets up a virtual environment, and verifies CUDA/Distributed setup.

4. **Run Training**

   ```bash
   poetry run train --config_path configs/{insert_your_config_here}.yaml
   ```
}
5. **Inspect Logs & Checkpoints**

   * Checkpoints: `runs/<run_name>/checkpoints/step_<N>.pt`
   * W\&B: View support/query accuracies, inner‑loop grad norms, AR loss, meta loss, and more.

---

## 📁 Repository Structure (additions to original)

```
├── src/
│   │── config/
│   │   └── smlmt_config.py  # Config for SMLMT style training
│   ├── model/
│   │   └── pico_decoder.py  # LLAMA‑style Transformer
│   ├── training/
│   │   ├── trainer.py       # Main Trainer with MAML/SMLMT loops
│   │   └── utils/           # Config, dataloader, logging, LR scheduler
│   └── checkpointing/       # Save/load model, optimizer, dynamics hooks
```

---

## 🔍 Analysis & Interpretability

Leverage our companion tool [**pico-analyze**](https://github.com/pico-lm/pico-analyze) to:

* Compute CKA, PWCCA between layers
* Track gradient sparsity and neuron activation distributions
* Compare learning trajectories across different inner-loop hyperparameters

---

<!-- ## 📜 License & Citation

Pico MAML Train is released under the **Apache 2.0** license (see [`LICENSE`](LICENSE)).

If you use this framework in your research, please cite:

```bibtex
@software{africa2025pico_maml_train,
  author = {Africa, David Demitri and Martinez, Richard Diehl},
  title = {Pico MAML Train: A Meta‑Learning Extension for Language Model Pretraining},
  year = {2025},
  url = {https://github.com/pico-lm/pico-maml-train}
} -->
```

**Happy meta‑learning!**
