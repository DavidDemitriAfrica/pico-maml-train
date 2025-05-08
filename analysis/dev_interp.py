import torch
from devinterp.optim import SGLD
from devinterp.slt.sampler import LLCEstimator, sample
from devinterp.utils import default_nbeta
from torch.utils.data import DataLoader


def run_devinterp_llc(
    model: torch.nn.Module,
    dataset,
    batch_size: int = 32,
    device: str = "cpu",
    sgld_lr: float = 1e-4,
    num_steps: int = 1000,
):
    """
    Runs the Timaeus devinterp LLCEstimator on `dataset` using SGLD.
    Returns the full estimator results dict (including 'llc/mean').
    """
    model = model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # create the LLCEstimator (auto‐computes nbeta from your loader)
    llc_est = LLCEstimator(model=model, nbeta=default_nbeta(loader))

    # set up SGLD optimizer over the model’s parameters
    optimizer = SGLD(model.parameters(), lr=sgld_lr)

    # run the sampler: this will call your callback internally
    sample(
        model=model,
        data_loader=loader,
        optimizer=optimizer,
        num_steps=num_steps,
        callbacks=[llc_est],
    )

    # grab the results
    results = llc_est.get_results()  # e.g. {'llc/mean': 0.123, ...}
    return results
