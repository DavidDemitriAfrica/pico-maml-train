"""
Pico Language Model Trainer

This Trainer implements a minimalistic end-to-end training pipeline of the Pico language model with
distributed training support via Lightning Fabric. It provides a modular and configurable training
pipeline with the features:

    - Configuration Management: YAML-based configuration for all aspects of training
    - Distributed Training: Multi-GPU support via Lightning Fabric
    - Checkpointing: Regular model saving and training state recovery
    - Evaluation: Periodic model evaluation on validation datasets
    - Logging: Comprehensive metric tracking and experiment monitoring
    - Optimization: Support for gradient accumulation, clipping, and LR scheduling
"""

import logging
import os
import platform
from typing import Any, Dict

import lightning as L
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from lightning.fabric.utilities.rank_zero import rank_zero_only

from src.checkpointing import (
    compute_learning_dynamics_states,
    load_checkpoint,
    save_checkpoint,
    save_evaluation_results,
    save_learning_dynamics_states,
)
from src.evaluation import run_evaluation
from src.training.utils import (
    initialize_configuration,
    initialize_dataloader,
    initialize_dataset,
    initialize_fabric,
    initialize_hf_checkpointing,
    initialize_logging,
    initialize_lr_scheduler,
    initialize_model,
    initialize_run_dir,
    initialize_tokenizer,
    initialize_wandb,
)
from src.training.utils.logging import pretty_print_yaml_config


class Trainer:
    def __init__(self, config_path: str):
        """
        Initializes the Trainer class. This Trainer class implements a `train` method, which is the
        main entry point for training the Pico model. Before calling `train`, the Trainer class
        initializes the following:

            - Configuration loading and validation
            - Model, optimizer, and dataset setup
            - Logging and experiment tracking setup
            - Checkpoint management

        Args:
            config_path (str): Path to the YAML configuration file containing any overrides.
        """

        ########################################################
        #
        # Basic Initialization of Configs, Fabric, Model, Optimizer, etc.
        #
        ########################################################

        torch.autograd.set_detect_anomaly(True)

        # Setup Config
        self.configs = initialize_configuration(config_path)

        # ——— Reset RoPE’s one‐time frequency table so it's rebuilt with our max_seq_len ———
        from src.model.pico_decoder import RoPE

        RoPE._freqs_cis_tensor = None

        # Setup Run Directory (i.e. where we store checkpoints, logs, etc.)
        initialize_run_dir(checkpointing_config=self.configs["checkpointing"])

        # Setup Logger
        if self.configs["monitoring"].save_to_wandb:
            wandb_logger = initialize_wandb(
                monitoring_config=self.configs["monitoring"],
                checkpointing_config=self.configs["checkpointing"],
            )
        else:
            wandb_logger = None

        # Setup Fabric
        self.fabric = initialize_fabric(
            training_config=self.configs["training"],
            wandb_logger=wandb_logger,
        )
        L.seed_everything(42, verbose=False)

        # Set up logging
        self.logger = initialize_logging(
            monitoring_config=self.configs["monitoring"],
            checkpointing_config=self.configs["checkpointing"],
            fabric=self.fabric,
        )

        self.tokenizer = initialize_tokenizer(
            self.configs["data"], self.configs["model"]
        )

        # finally stash the id for use in masking
        self.mask_id = self.tokenizer.mask_token_id

        # Setup Model
        self.model = initialize_model(model_config=self.configs["model"])

        ########################################################
        #
        # SMLMT flags used during training
        #
        ########################################################

        self.should_smlmt = (
            self.configs["smlmt"].enabled and self.configs["smlmt"].hybrid_ratio > 0.0
        )
        if self.should_smlmt:
            self.smlmt_hybrid_ratio = self.configs["smlmt"].hybrid_ratio
            self.smlmt_min_token_freq = self.configs["smlmt"].min_token_freq
            self.smlmt_max_token_freq = self.configs["smlmt"].max_token_freq
            self.smlmt_inner_steps = self.configs["smlmt"].inner_steps
            self.smlmt_inner_lr = self.configs["smlmt"].inner_lr
            self.smlmt_support_size = self.configs["smlmt"].support_size
            head_cfg = self.configs["smlmt"].classifier_head
            layers = []
            in_dim = self.model.config.d_model
            for _ in range(head_cfg.num_layers - 1):
                layers += [
                    nn.Linear(in_dim, head_cfg.hidden_dim),
                    nn.ReLU(),
                ]
                in_dim = head_cfg.hidden_dim
            # final projection to vocab-size (for token prediction)
            layers.append(nn.Linear(in_dim, self.model.config.vocab_size))
            self.model.classifier_head = nn.Sequential(*layers)
            if head_cfg.init_method == "xavier":
                for p in self.model.classifier_head.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
            # counters for logging
            self.smlmt_step_count = 0
            self.ar_step_count = 0
            self.inner_acc_history = []
            self.query_acc_history = []

            # param split
            self.head_params = list(self.model.classifier_head.parameters())
            self.backbone_params = [
                p
                for n, p in self.model.named_parameters()
                if not n.startswith("classifier_head")
            ]
            self.outer_params = self.backbone_params + self.head_params
            self.inner_optimizer = torch.optim.SGD(
                self.head_params, lr=self.smlmt_inner_lr
            )
        else:
            self.outer_params = list(self.model.parameters())

        raw_outer_opt = torch.optim.AdamW(
            self.outer_params, lr=self.configs["training"].optimization.lr
        )

        self.lr_scheduler = initialize_lr_scheduler(
            self.configs["training"], raw_outer_opt
        )

        # 3) DeepSpeed only supports ONE optimizer at setup time.
        #    Always wrap model + outer optimizer only.
        #    Move the tokenizer to initialize before the model is wrapped.

        self.model, self.outer_optimizer = self.fabric.setup(self.model, raw_outer_opt)

        # Setup HuggingFace Checkpointing
        if self.configs["checkpointing"].save_to_hf:
            initialize_hf_checkpointing(
                checkpointing_config=self.configs["checkpointing"], fabric=self.fabric
            )

        ########################################################
        #
        # Boilerplate to deal with loading/resuming from checkpoints
        #
        ########################################################

        self.should_load_checkpoint = self.configs["checkpointing"].training.auto_resume

        # Possibly load a checkpoint
        if self.should_load_checkpoint:
            resume_checkpoint = load_checkpoint(
                checkpointing_config=self.configs["checkpointing"],
                checkpoint_step="latest",
                fabric=self.fabric,
                model=self.model,
                optimizer=self.outer_optimizer,
                lr_scheduler=self.lr_scheduler,
            )

            if resume_checkpoint:
                (
                    self.model,
                    self.outer_optimizer,
                    self.lr_scheduler,
                    self.initial_batch_step,
                    inner_opt_state,
                ) = resume_checkpoint
                if self.should_smlmt and inner_opt_state is not None:
                    self.inner_optimizer.load_state_dict(inner_opt_state)
            else:
                self.initial_batch_step = 0
        else:
            self.initial_batch_step = 0

        ########################################################
        #
        # Initialization of Dataset & DataLoader (possibly fast-forwarding to correct batch)
        #
        ########################################################

        self.train_dataset, fast_forward_steps = initialize_dataset(
            data_config=self.configs["data"],
            fabric=self.fabric,
            initial_batch_step=self.initial_batch_step,
            return_fast_forward_steps=True,
        )

        self.train_dataloader = initialize_dataloader(
            data_config=self.configs["data"],
            training_config=self.configs["training"],
            fabric=self.fabric,
            dataset=self.train_dataset,
        )
        self.train_dataloader = self.fabric.setup_dataloaders(
            self.train_dataloader, use_distributed_sampler=True
        )

        # NOTE: We may need to fast-forward the iterator to the correct step so that we can
        # continue from the correct batch of data we would have seen had training not
        # previously stopped.
        train_iterator = iter(self.train_dataloader)
        if fast_forward_steps > 0:
            fast_forward_sub_steps = (
                fast_forward_steps
                * self.configs["training"].optimization.gradient_accumulation_steps
            )
            for _ in range(fast_forward_sub_steps):
                next(train_iterator)

        self.train_iterator = train_iterator

        # NOTE: Sychronizing processes after fast-forwarding iterator
        self.fabric.barrier()

        ########################################################
        #
        # Helper flags used during training for checkpointing and evaluation
        #
        ########################################################

        # Helper flag to determine if we should evaluate the model
        self.should_evaluate = (
            self.configs["evaluation"].metrics is not None
            and len(self.configs["evaluation"].metrics) > 0
        )

        self.should_compute_learning_dynamics = (
            self.configs["checkpointing"].learning_dynamics.layer_suffixes is not None
            and len(self.configs["checkpointing"].learning_dynamics.layer_suffixes) > 0
        )

        if self.should_compute_learning_dynamics:
            if self.configs["checkpointing"].learning_dynamics.eval_data is not None:
                self.learning_dynamics_eval_dataset = load_dataset(
                    self.configs["checkpointing"].learning_dynamics.eval_data,
                    split="val",
                )
            else:
                self.learning_dynamics_eval_dataset = None

    def train(self) -> None:
        """Execute the main training pipeline.

        This method orchestrates the complete training process by:
        1. Creating an initial checkpoint to save the starting state and evaluate the model as a
            baseline
        2. Running the main training loop via `_training_loop`
        3. Handling final checkpointing and evaluation

        The training progress is tracked through checkpoints and evaluations
        at intervals specified in the configuration.
        """

        ########################################################
        #
        # Initial Checkpointing and Evaluation
        #
        ########################################################

        # Save Initial Checkpoint -- If the checkpoint already exists, this performs a no-op
        save_checkpoint(
            configs=self.configs,
            checkpoint_step=self.initial_batch_step,
            fabric=self.fabric,
            model=self.model,
            optimizer=self.outer_optimizer,
            lr_scheduler=self.lr_scheduler,
            tokenizer=self.tokenizer,
            inner_optimizer=self.inner_optimizer if self.should_smlmt else None,
        )

        # Save Initial Evaluation Results
        if self.should_evaluate:
            if self.initial_batch_step == 0:
                evaluation_results = run_evaluation(
                    evaluation_config=self.configs["evaluation"],
                    checkpointing_config=self.configs["checkpointing"],
                    fabric=self.fabric,
                    model=self.model,
                )
                self._log_evaluation_results(
                    evaluation_results, self.initial_batch_step
                )
                save_evaluation_results(
                    checkpointing_config=self.configs["checkpointing"],
                    fabric=self.fabric,
                    evaluation_results=evaluation_results,
                    checkpoint_step=self.initial_batch_step,
                )
            else:
                # NOTE: If the run crashed while evaluating, we need to restart the evaluation
                eval_results_path = os.path.join(
                    self.configs["checkpointing"].evaluation.eval_results_dir,
                    f"step_{self.initial_batch_step}.json",
                )
                if not os.path.exists(eval_results_path):
                    evaluation_results = run_evaluation(
                        evaluation_config=self.configs["evaluation"],
                        checkpointing_config=self.configs["checkpointing"],
                        fabric=self.fabric,
                        model=self.model,
                    )
                    self._log_evaluation_results(
                        evaluation_results, self.initial_batch_step
                    )
                    save_evaluation_results(
                        checkpointing_config=self.configs["checkpointing"],
                        fabric=self.fabric,
                        evaluation_results=evaluation_results,
                        checkpoint_step=self.initial_batch_step,
                    )

        ########################################################
        #
        # Main Training Loop (see `_training_loop` for details)
        #
        ########################################################

        if self.initial_batch_step < self.configs["training"].max_steps:
            self._log_training_configuration()
            final_step = self._training_loop()
        else:
            final_step = self.initial_batch_step

        ########################################################
        #
        # Final Checkpointing and Evaluation
        #
        ########################################################

        # Save Learning Dynamics States
        if self.should_compute_learning_dynamics:
            if self.learning_dynamics_eval_dataset is not None:
                self.log(f"Step {final_step} -- 📈 Saving Learning Dynamics")
                learning_dynamics_val_states = compute_learning_dynamics_states(
                    checkpointing_config=self.configs["checkpointing"],
                    fabric=self.fabric,
                    model=self.model,
                    dataset=self.learning_dynamics_eval_dataset,
                    compute_gradients=True,
                )
                save_learning_dynamics_states(
                    checkpointing_config=self.configs["checkpointing"],
                    fabric=self.fabric,
                    learning_dynamics_states=learning_dynamics_val_states,
                    checkpoint_step=final_step,
                    prefix="val",
                )

        # Handle checkpointing and final evaluation
        save_every = self.configs["checkpointing"].save_every_n_steps
        if save_every > 0 and final_step % save_every != 0:
            self.log(f"Step {final_step} -- 💾 Saving Final Checkpoint")
            save_checkpoint(
                configs=self.configs,
                checkpoint_step=final_step,
                fabric=self.fabric,
                model=self.model,
                optimizer=self.outer_optimizer,
                lr_scheduler=self.lr_scheduler,
                tokenizer=self.tokenizer,
                inner_optimizer=self.inner_optimizer if self.should_smlmt else None,
            )

            # Final evaluation
            if self.should_evaluate:
                evaluation_results = run_evaluation(
                    evaluation_config=self.configs["evaluation"],
                    checkpointing_config=self.configs["checkpointing"],
                    fabric=self.fabric,
                    model=self.model,
                )
                self._log_evaluation_results(evaluation_results, final_step)
                save_evaluation_results(
                    checkpointing_config=self.configs["checkpointing"],
                    checkpoint_step=final_step,
                    fabric=self.fabric,
                    evaluation_results=evaluation_results,
                )

        self.log(f"🎉 Training complete! Final step: {final_step}")

        if final_step < self.configs["training"].max_steps:
            self.log(
                f"\t Note: Training stopped before max steps ({self.configs['training'].max_steps})",
                level=logging.WARNING,
            )

        # Cleanup distributed training
        self.fabric.barrier()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()

            del self.train_dataloader  # NOTE: shutting down worker nodes

        self.fabric.barrier()

    def _training_loop(self) -> int:
        """Execute the main training loop.

        This method orchestrates the core training loop and includes the following features:
            - Gradient accumulation
            - Gradient clipping
            - Periodic model evaluation and checkpointing
            - Learning Dynamics Checkpointing
            - Learning rate scheduling
            - Logging of training metrics including loss and learning rate
            - Handling of infinite/NaN losses

        Returns:
            int: The final step count reached during training.
                NOTE: A complete training run should match the configured max_steps.
        """
        # Setup training loop variables
        batch_step = self.initial_batch_step
        # Guard against zero checkpoint interval
        save_every = self.configs["checkpointing"].save_every_n_steps
        # Remove all stale grads
        self.outer_optimizer.zero_grad()
        # NOTE: these are used to compute the average loss over a training interval.
        # This is more accurate than using the loss at the end of the interval.
        interval_loss = torch.tensor(0.0, device=self.fabric.device)
        interval_steps = torch.tensor(0, device=self.fabric.device)
        interval_inf_or_nan_count = torch.tensor(0, device=self.fabric.device)

        if self.should_compute_learning_dynamics:
            # NOTE: we basically re-construct the full batch here so that we can compute learning dynamics
            training_batch = {"input_ids": []}

        # NOTE: determine what sub-batch we should start from
        initial_sub_batch_step = (
            batch_step
            * self.configs["training"].optimization.gradient_accumulation_steps
        )

        ###############################################################
        #
        # Core loop starts here
        # NOTE: the ratio between sub_batch_step and batch_step
        # is the configured number of gradient_accumulation_steps
        # i.e. with 32 configured gradient accumulation steps,
        # there are 32 sub_batch_steps for each batch_step
        #
        ###############################################################

        for sub_batch_step, sub_batch in enumerate(
            self.train_iterator, start=initial_sub_batch_step
        ):
            # NOTE: We want to store the entire training batch whenever we are computing learning dynamics
            # and we are at a checkpointing step.
            should_store_training_batch = self.should_compute_learning_dynamics and (
                batch_step % self.configs["checkpointing"].save_every_n_steps == 0
            )

            ################################################################################
            # Forward Pass (with mutually‐exclusive SMLMT vs autoregressive branches)
            ################################################################################
            _input_ids = torch.tensor(sub_batch["input_ids"], device=self.fabric.device)
            max_len = self.configs["model"].max_seq_len
            if _input_ids.size(1) > max_len:
                _input_ids = _input_ids[:, -max_len:]
            # 1) learning‐dynamics gather (unchanged)
            if should_store_training_batch:
                gathered = self.fabric.all_gather(_input_ids)
                if self.fabric.world_size > 1:
                    gathered = gathered.reshape(-1, *gathered.shape[2:])
                training_batch["input_ids"].extend(gathered.tolist())

            # 2) choose branch *synchronously* across all ranks
            rand_val = torch.rand(1, device=self.fabric.device)
            rand_val = self.fabric.broadcast(rand_val, src=0)
            do_meta = self.should_smlmt and (rand_val.item() < self.smlmt_hybrid_ratio)
            if do_meta:
                # 1) snapshot head
                orig_head = {
                    k: v.clone()
                    for k, v in self.model.classifier_head.state_dict().items()
                }
                B, L = _input_ids.size()

                # 2) sample two mask‐positions per sequence: support & query
                k = self.smlmt_support_size
                pos_sup = torch.randint(
                    1, L - 1, (B, k), device=_input_ids.device
                )  # (B,k)
                pos_q = torch.randint(1, L - 1, (B,), device=_input_ids.device)

                support_ids = _input_ids.clone()
                mask_id = self.tokenizer.mask_token_id
                if mask_id is None:
                    # e.g. use pad_token_id or eos_token_id
                    mask_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

                support_labels = support_ids[
                    torch.arange(B).unsqueeze(1), pos_sup
                ].clone()  # (B,k)
                support_ids[torch.arange(B).unsqueeze(1), pos_sup] = mask_id

                query_ids = _input_ids.clone()
                query_labels = query_ids[torch.arange(B), pos_q].clone()
                query_ids[torch.arange(B), pos_q] = mask_id

                for inner_step in range(self.smlmt_inner_steps):
                    # zero grad on _inner_ optimizer up front
                    # self.inner_optimizer.zero_grad()
                    # freeze transformer weights
                    hidden_sup, _ = self.model(support_ids, return_hidden=True)
                    # make sure hidden_sup is the same dtype as the head’s weights:
                    head_dtype = next(self.model.classifier_head.parameters()).dtype
                    hidden_sup = hidden_sup.to(head_dtype)
                    logits_sup = self.model.classifier_head(hidden_sup)
                    logits_sup = logits_sup[torch.arange(B).unsqueeze(1), pos_sup, :]
                    # flatten to (B*k, V) and (B*k,) for cross‐entropy:
                    B, k, V = logits_sup.shape
                    logits_sup = logits_sup.view(B * k, V)  # (B*k, V)
                    support_labels = support_labels.view(B * k)  # (B*k,)

                    # 1) pick out the unique support classes (size = K ≤ B*k)
                    unique_labels = torch.unique(support_labels)  # (K,)

                    # 2) slice logits to just those K columns → (B*k, K)
                    small_logits = logits_sup[:, unique_labels]

                    # 3) remap original labels into 0…K-1 for the small softmax
                    new_labels = torch.bucketize(support_labels, unique_labels)

                    # 4) compute loss & accuracy on the K-way problem
                    loss_sup = F.cross_entropy(small_logits, new_labels)

                    with torch.no_grad():
                        preds_sup = small_logits.argmax(dim=-1)
                        acc_sup = (preds_sup == new_labels).float().mean()

                    # record support accuracy, but only keep it for checkpoint summaries
                    self.inner_acc_history.append(acc_sup.item())
                    # still log
                    self.fabric.log("inner/acc_sup", acc_sup.item(), step=batch_step)

                    self.fabric.backward(loss_sup, model=self.model)
                    # self.inner_optimizer.step()
                    with torch.no_grad():
                        for p in self.model.classifier_head.parameters():
                            if p.grad is not None:
                                p.data -= self.smlmt_inner_lr * p.grad
                    for p in self.model.classifier_head.parameters():
                        p.grad = None

                # 5) compute query loss on the *adapted* head

                hidden_q, _ = self.model(query_ids, return_hidden=True)
                hidden_q = hidden_q.to(head_dtype)
                logits_q = self.model.classifier_head(hidden_q)[
                    torch.arange(B), pos_q, :
                ]
                # ── compute & log query accuracy ────────────────────────────────────────
                with torch.no_grad():
                    preds_q = logits_q.argmax(dim=-1)  # (B,)
                    acc_q = (preds_q == query_labels).float().mean()
                self.query_acc_history.append(acc_q.item())
                self.fabric.log("inner/acc_q", acc_q.item(), step=batch_step)
                loss = F.cross_entropy(logits_q, query_labels)

            else:
                # --- Autoregressive LM branch ---
                input_ids = _input_ids[:, :-1]
                labels = _input_ids[:, 1:]
                logits, _ = self.model(input_ids)  # (B, T, V)
                B2, T2, V = logits.shape
                loss = F.cross_entropy(
                    logits.view(B2 * T2, V), labels.contiguous().view(-1)
                )
            ########################################################
            #
            # Gradient accumulation
            #
            ########################################################

            should_accumulate_gradients = (sub_batch_step + 1) % self.configs[
                "training"
            ].optimization.gradient_accumulation_steps != 0

            self.fabric.backward(
                loss
                / self.configs["training"].optimization.gradient_accumulation_steps,
                model=self.model,
            )

            if do_meta:
                with torch.no_grad():
                    self.model.classifier_head.load_state_dict(orig_head)

            if torch.isnan(loss) or torch.isinf(loss):
                interval_inf_or_nan_count += 1
            else:
                interval_loss += loss.item()
                interval_steps += 1

            ########################################################
            #
            # Logging
            #
            ########################################################

            # if save_every > 0 and batch_step % save_every == 0:
            #    self._log_training_metrics(
            #        interval_loss=interval_loss,
            #        interval_steps=interval_steps,
            #        interval_inf_or_nan_count=interval_inf_or_nan_count,
            #        batch_step=batch_step,
            #    )
            #    interval_loss = torch.tensor(0.0, device=self.fabric.device)
            #    interval_steps = torch.tensor(0, device=self.fabric.device)
            #    interval_inf_or_nan_count = torch.tensor(0, device=self.fabric.device)

            ########################################################
            #
            # Learning Dynamics Checkpointing
            #
            ########################################################

            # if save_every > 0 and batch_step % save_every == 0:
            #    if self.should_compute_learning_dynamics:
            #        self.log(f"Step {batch_step} -- 📈 Saving Learning Dynamics")

            # Training Batch Learning Dynamics
            #        training_batch_dataset = Dataset.from_dict(training_batch)

            #        learning_dynamics_train_states = compute_learning_dynamics_states(
            #            checkpointing_config=self.configs["checkpointing"],
            #            fabric=self.fabric,
            #            model=self.model,
            #            dataset=training_batch_dataset,
            #            compute_gradients=True,
            #        )

            #        save_learning_dynamics_states(
            #            checkpointing_config=self.configs["checkpointing"],
            #            checkpoint_step=batch_step,
            #            prefix="train",
            #            fabric=self.fabric,
            #            learning_dynamics_states=learning_dynamics_train_states,
            #            learning_dynamics_dataset=training_batch_dataset,
            #            tokenizer=self.tokenizer,
            #        )
            #        training_batch = {
            #            "input_ids": []
            #        }  # Resetting training_batch for next training batch

            # Validation Data Learning Dynamics
            #        if self.learning_dynamics_eval_dataset is not None:
            #            learning_dynamics_val_states = compute_learning_dynamics_states(
            #                checkpointing_config=self.configs["checkpointing"],
            #                fabric=self.fabric,
            #                model=self.model,
            #                dataset=self.learning_dynamics_eval_dataset,
            #                compute_gradients=True,
            #            )
            #            save_learning_dynamics_states(
            #                checkpointing_config=self.configs["checkpointing"],
            #                checkpoint_step=batch_step,
            #                prefix="val",
            #                fabric=self.fabric,
            #                learning_dynamics_states=learning_dynamics_val_states,
            #            )

            ########################################################
            #
            # Optimization step
            #
            ########################################################

            # only step once per full (accumulation) batch
            if not should_accumulate_gradients:
                self.outer_optimizer.step()
                self.lr_scheduler.step()
                self.outer_optimizer.zero_grad()
                batch_step += 1
                if do_meta:
                    self.smlmt_step_count += 1
                else:
                    self.ar_step_count += 1
                # ——— Logging ———
                if (
                    batch_step % self.configs["monitoring"].logging.log_every_n_steps
                    == 0
                ):
                    self._log_training_metrics(
                        interval_loss=interval_loss,
                        interval_steps=interval_steps,
                        interval_inf_or_nan_count=interval_inf_or_nan_count,
                        batch_step=batch_step,
                    )
                    interval_loss = torch.tensor(0.0, device=self.fabric.device)
                    interval_steps = torch.tensor(0, device=self.fabric.device)
                    interval_inf_or_nan_count = torch.tensor(
                        0, device=self.fabric.device
                    )

                # ——— Learning Dynamics Checkpointing ———
                if (
                    save_every > 0
                    and batch_step % save_every == 0
                    and self.should_compute_learning_dynamics
                ):
                    self.log(f"Step {batch_step} -- 📈 Saving Learning Dynamics")
                    # (reuse your existing LD code here)

                # ——— Training Checkpointing & Evaluation ———
                if save_every > 0 and batch_step % save_every == 0:
                    self.log(f"Step {batch_step} -- 💾 Saving Checkpoint")
                    avg_acc = (
                        sum(self.inner_acc_history) / len(self.inner_acc_history)
                        if self.inner_acc_history
                        else 0.0
                    )
                    self.log(
                        f"SMLMT steps={self.smlmt_step_count}, AR steps={self.ar_step_count}, "
                        f"avg_support_acc={avg_acc:.4f}"
                    )
                    save_checkpoint(
                        configs=self.configs,
                        checkpoint_step=batch_step,
                        fabric=self.fabric,
                        model=self.model,
                        optimizer=self.outer_optimizer,
                        lr_scheduler=self.lr_scheduler,
                        tokenizer=self.tokenizer,
                        inner_optimizer=self.inner_optimizer
                        if self.should_smlmt
                        else None,
                    )
                    if self.should_evaluate:
                        evaluation_results = run_evaluation(
                            evaluation_config=self.configs["evaluation"],
                            checkpointing_config=self.configs["checkpointing"],
                            fabric=self.fabric,
                            model=self.model,
                        )
                        if evaluation_results is not None:
                            self._log_evaluation_results(evaluation_results, batch_step)
                            save_evaluation_results(
                                checkpointing_config=self.configs["checkpointing"],
                                fabric=self.fabric,
                                evaluation_results=evaluation_results,
                                checkpoint_step=batch_step,
                            )

            ########################################################
            #
            # Training Checkpointing and evaluation
            #
            ########################################################

            # if save_every > 0 and batch_step % save_every == 0:
            #    self.log(f"Step {batch_step} -- 💾 Saving Checkpoint")
            #    save_checkpoint(
            #        configs=self.configs,
            #        checkpoint_step=batch_step,
            #        fabric=self.fabric,
            #        model=self.model,
            #        optimizer=self.outer_optimizer,
            #        lr_scheduler=self.lr_scheduler,
            #        tokenizer=self.tokenizer,
            #        inner_optimizer=self.inner_optimizer if self.should_smlmt else None,
            #    )

            #    if self.should_evaluate:
            #        evaluation_results = run_evaluation(
            #            evaluation_config=self.configs["evaluation"],
            #            checkpointing_config=self.configs["checkpointing"],
            #            fabric=self.fabric,
            #            model=self.model,
            #        )
            #        if evaluation_results is not None:
            #            self._log_evaluation_results(evaluation_results, batch_step)
            #            save_evaluation_results(
            #                checkpointing_config=self.configs["checkpointing"],
            #                fabric=self.fabric,
            #                evaluation_results=evaluation_results,
            #                checkpoint_step=batch_step,
            #            )

            # Break if we've reached training steps
            # if batch_step >= self.configs["training"].max_steps:
            #    break

        return batch_step

    ########################################################
    #
    # Trainer Logging Functinalities
    #
    ########################################################

    def _log_training_metrics(
        self,
        interval_loss: torch.Tensor,
        interval_steps: torch.Tensor,
        interval_inf_or_nan_count: torch.Tensor,
        batch_step: int,
    ):
        """
        Gathers together the training metrics computed across all processes in distributed training
        and logs them in a tree-style format.
        """
        # ---- aggregate scalars ----
        total_loss = self.fabric.all_reduce(interval_loss, reduce_op="sum").item()
        total_inf_nan = self.fabric.all_reduce(
            interval_inf_or_nan_count, reduce_op="sum"
        ).item()
        total_steps = self.fabric.all_reduce(interval_steps, reduce_op="sum").item()
        avg_loss = total_loss / total_steps if total_steps > 0 else float("inf")
        lr = self.lr_scheduler.get_last_lr()[0]

        # ---- log to tensorboard/wandb ----
        self.fabric.log("train/loss", avg_loss, step=batch_step)
        self.fabric.log("trainer/inf_or_nan_count", total_inf_nan, step=batch_step)
        self.fabric.log("trainer/learning_rate", lr, step=batch_step)

        # ---- optionally log head stats ----
        if self.configs["smlmt"].enabled:
            # gather all head parameters
            head_params = list(self.model.classifier_head.parameters())
            # weight vector
            all_w = torch.cat([p.detach().view(-1) for p in head_params])
            w_mean, w_std = all_w.mean().item(), all_w.std().item()
            # gradient norm
            grads = [p.grad for p in head_params if p.grad is not None]
            grad_norm = (
                torch.norm(torch.stack([g.norm() for g in grads])).item()
                if grads
                else 0.0
            )

            self.fabric.log("head/weight_mean", w_mean, step=batch_step)
            self.fabric.log("head/weight_std", w_std, step=batch_step)
            self.fabric.log("head/grad_norm", grad_norm, step=batch_step)

        # ---- console output ----
        self.log(f"Step {batch_step} -- 🔄 Training Metrics")
        self.log(f"├── Loss:           {avg_loss:.4f}")
        self.log(f"├── Learning Rate:  {lr:.2e}")
        self.log(f"└── Inf/NaN count:  {total_inf_nan}")

        if self.configs["smlmt"].enabled:
            self.log("    └── Classifier Head:")
            self.log(f"         ├── weight_mean: {w_mean:.4f}")
            self.log(f"         ├── weight_std:  {w_std:.4f}")
            self.log(f"         └── grad_norm:   {grad_norm:.4f}")

            # —— Meta/AR mix & avg‐inner/query accuracy ——
            meta_steps = getattr(self, "smlmt_step_count", 0)
            ar_steps = getattr(self, "ar_step_count", 0)
            avg_sup = (
                sum(self.inner_acc_history) / len(self.inner_acc_history)
                if self.inner_acc_history
                else 0.0
            )
            avg_q = (
                sum(self.query_acc_history) / len(self.query_acc_history)
                if self.query_acc_history
                else 0.0
            )
            mix_msg = (
                f"    └─ SMLMT mix: meta_steps={meta_steps}, "
                f"ar_steps={ar_steps}, avg_sup_acc={avg_sup:.4f}, "
                f"avg_query_acc={avg_q:.4f}"
            )
            self.log(mix_msg)

            # push these to wandb as well
            self.fabric.log("meta/meta_steps", meta_steps, step=batch_step)
            self.fabric.log("meta/ar_steps", ar_steps, step=batch_step)
            self.fabric.log("meta/avg_sup_acc", avg_sup, step=batch_step)
            self.fabric.log("meta/avg_q_acc", avg_q, step=batch_step)

            if grad_norm == 0.0:
                self.log(
                    "⚠️ Head gradients zero—inner loop may not be updating the head",
                    level=logging.WARNING,
                )
            if avg_sup == 0.0:
                self.log(
                    "⚠️ Support accuracy is zero—support signal may be too weak",
                    level=logging.WARNING,
                )
            if avg_q == 0.0:
                self.log(
                    "⚠️ Query accuracy is zero—head is not adapting",
                    level=logging.WARNING,
                )

    def _log_evaluation_results(
        self, evaluation_results: Dict[str, Any], batch_step: int
    ):
        """Log model evaluation metrics to experiment tracking system and console."""
        self.log(f"Step {batch_step} -- 📊 Evaluation Results")
        for i, (metric, result) in enumerate(evaluation_results.items()):
            prefix = "└──" if i == len(evaluation_results) - 1 else "├──"
            self.log(f"{prefix} {metric}: {result}")
            self.fabric.log(f"eval/{metric}", result, step=batch_step)

    def _log_training_configuration(self):
        """
        Log training configuration details as well as runtime information about the hardware,
        software, and batch settings.

        This function is called at the beginning of the training loop to provide a summary of the
        training configuration.
        """

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        global_batch_size = self.configs["data"].dataloader.batch_size
        per_device_batch_size = self.train_dataloader.batch_size
        gradient_accumulation_steps = self.configs[
            "training"
        ].optimization.gradient_accumulation_steps

        device_type = ""
        fabric_device = str(self.fabric.device)
        if torch.cuda.is_available() and "cuda" in fabric_device:
            device_type = torch.cuda.get_device_name(self.fabric.device)
        elif torch.backends.mps.is_available() and "mps" in fabric_device:
            device_type = "MPS (Apple Silicon)"
        else:
            device_type = "CPU"

        training_config_path = os.path.join(
            self.configs["checkpointing"].runs_dir,
            self.configs["checkpointing"].run_name,
            "training_config.yaml",
        )
        if os.path.exists(training_config_path):
            self.log("=" * 50)
            self.log("✨ Training Configuration")
            self.log("=" * 50)
            training_config = yaml.safe_load(open(training_config_path, "r"))
            pretty_print_yaml_config(self.logger, training_config)

        self.log("=" * 50)
        self.log("⛭ Runtime Summary:")
        self.log("=" * 50)
        self.log(f"Starting from step: {self.initial_batch_step}")

        self.log("Model Setup:")
        self.log(f"└─ Total Parameters: {total_params:,}")
        self.log(f"└─ Trainable Parameters: {trainable_params:,}")

        self.log("Distributed Setup:")
        self.log(f"└─ Number of Devices: {self.fabric.world_size}")
        self.log(f"└─ Device Type: {device_type}")
        self.log(
            f"└─ Available Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            if torch.cuda.is_available()
            else f"└─ Available Memory: {psutil.virtual_memory().total / 1e9:.2f} GB"
        )

        self.log("Software Setup:")
        self.log(f"└─ Python Version: {platform.python_version()}")
        self.log(f"└─ PyTorch Version: {torch.__version__}")
        self.log(
            f"└─ CUDA Version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}"
        )
        self.log(f"└─ Operating System: {platform.system()} {platform.release()}")

        self.log("Batch Size Configuration:")
        self.log(f"└─ Global Batch Size: {global_batch_size}")
        self.log(f"└─ Per Device Batch Size: {per_device_batch_size}")
        self.log(f"└─ Gradient Accumulation Steps: {gradient_accumulation_steps}")
        self.log("=" * 50)

    @rank_zero_only
    def log(self, msg: str, level: int = logging.INFO) -> None:
        """NOTE: Log messages only from rank zero process."""
        self.logger.log(level, msg)
