# Pico-MAML: Learning Dynamics Research on Model-Agnostic Meta-Learning

Pico is a framework designed to facilitate research into language model learning dynamics through a comprehensive suite of small to medium-scale models (1M-1B parameters). Built on a LLAMA-style architecture, Pico emphasizes simplicity, modularity, and research accessibility. We fork Pico to implement Subset Masked Language Modeling Tasks (SMLMT) an unsupervised approach to create meta-training tasks from unlabeled data. We then implement Model-Agnostic Meta-Learning (MAML) to train the actual model.

## 🚀 Quick Start

1. **Clone Project**
```bash
git clone https://github.com/rdiehlmartinez/pico.git && cd pico
```

2. **Configure Environment**
Create `.env` file:
```bash
export HF_TOKEN=your_huggingface_token
export WANDB_API_KEY=your_wandb_key
```

3. **Setup Dependencies**
```bash
source setup.sh
```
### Exploring the Codebase

The core implementation is organized into these key files and packages:

- **`src/model/pico.py`**: The heart of Pico
  - LLAMA-style transformer implementation
  - Attention mechanism with KV-cache
  - RoPE positional embeddings
  - Documentation references for each component

- **`src/training/trainer.py`**: Training pipeline
  - Distributed training setup
  - Checkpoint management
  - Logging configuration

- **`src/training/smlmt.py`**: SMLMT helper class
  - Task generation
  - Task sampling
  - Injecting your own sentences

- **`src/config`**: Model configuration
  - Hyperparameter definitions
  - Model architecture settings
  - Training parameters

- **`src/checkpointing`**: Checkpointing and State Management
  - Training state persistence (model, optimizer, scheduler)
  - Learning dynamics tracking (activations, weights, gradients)
  - Evaluation results storage
  - Automatically store huggingface-compatible version of model for down-stream use 

### Common Starting Points

1. **Using Pre-trained Models**
```python
from transformers import AutoModelForCausalLM

# Load a specific model size
model = AutoModelForCausalLM.from_pretrained("pico-lm/[...]")
```

2. **Training Your Own Suite**
```bash
# Create a config yaml file, e.g. `my_config.yaml`
# You can follow the provided demo template in configs/demo.yaml
# If no config file is provided the default config values are used
poetry run train --config_path my_config.yaml
```

## 📚 References

Our implementation draws inspiration from and builds upon:
- [LLAMA](https://arxiv.org/abs/2302.13971)
- [RoPE](https://arxiv.org/abs/2104.09864)
- [SwiGLU](https://arxiv.org/abs/2002.05202)
- [MAML](https://arxiv.org/abs/1703.03400)
- [SMLMT](https://arxiv.org/abs/2009.08445)
- [SMLMT w/ Prototypical Networks](https://aclanthology.org/2021.metanlp-1.8/)


## 🤝 Contributing

We welcome contributions in:
- New features and improvements
- Documentation and tutorials
- Bug fixes and testing
- Research findings and analysis


## 📝 License

Apache 2.0 License

## 📫 Contact

- GitHub: [DavidDemitriAfrica/pico-maml](https://github.com/DavidDemitriAfrica/pico-maml)
- Author: [David Demitri Africa](https://daviddemitriafrica.github.io/)

## Citation

If you use Pico-MAML in your research, please cite:

```bibtex
@software{picomaml2025,
    author = {Africa, David Demitri},
    title = {Pico-MAML: Learning Dynamics Research on Model-Agnostic Meta-Learning},
    year = {2025},
}
```