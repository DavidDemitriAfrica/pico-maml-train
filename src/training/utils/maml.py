import torch
from collections import OrderedDict
import torch.nn as nn
from typing import Dict, Tuple
import torch.nn.functional as F


def clone_classifier_params(model: nn.Module) -> OrderedDict:
    """
    Extract the classifier's parameters (weight, bias), clone them,
    and mark them as requires_grad=True for the MAML inner loop.
    """
    fast_params = OrderedDict()
    # We assume model.classifier is an nn.Linear
    for name, p in model.classifier.named_parameters():
        fast_params[name] = p.clone().detach().requires_grad_(True)
    return fast_params


def forward_classifier_with_params(
    model: nn.Module, hidden: torch.Tensor, params: OrderedDict
) -> torch.Tensor:
    """
    Forward pass for the classifier using the "fast" parameters in `params`
    instead of the model.classifier’s actual (meta) parameters.
    """
    W = params["weight"]  # shape [num_classes, d_model]
    b = params["bias"]  # shape [num_classes]
    # hidden is [batch_size, d_model]
    logits = hidden.matmul(W.t()) + b
    return logits


def maml_inner_update(
    model: nn.Module,
    fast_params: OrderedDict,
    support_inputs: Dict[str, torch.Tensor],
    support_labels: torch.Tensor,
    inner_lr: float,
    create_graph: bool = True,
) -> Tuple[OrderedDict, torch.Tensor]:
    """
    Perform ONE step of MAML's 'inner' update on the classifier's parameters.

    Returns:
      - updated fast_params (OrderedDict)
      - the support loss (torch.Tensor)
    """
    # Forward pass through the model to get hidden states
    # Here we do a normal forward to get a hidden representation
    # (because we are only MAML-ing the final classifier).
    # We'll assume your Pico forward can do: model(..., return_hidden=True)
    # and returns (logits, hidden, cached_key_values).
    # We'll ignore the standard logits, use `hidden` to feed the classifier:
    with torch.set_grad_enabled(True):
        _, support_hidden, _ = model(
            support_inputs["input_ids"],
            attention_mask=support_inputs["attention_mask"],
            return_hidden=True,
        )
        # Suppose we just take the mean-pooled hidden states:
        support_repr = support_hidden.mean(dim=1)
        # Now do classifier forward with fast_params
        support_preds = forward_classifier_with_params(model, support_repr, fast_params)
        support_loss = F.cross_entropy(support_preds, support_labels)

        # Manually compute grads wrt the *fast_params* only
        grads = torch.autograd.grad(
            support_loss,
            list(fast_params.values()),
            create_graph=create_graph,  # True if we want second-order grads
        )

    # Update each fast-param by gradient descent
    updated_params = OrderedDict()
    for (key, param), grad in zip(fast_params.items(), grads):
        updated_params[key] = param - inner_lr * grad

    return updated_params, support_loss
