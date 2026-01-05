import torch

def squashing_fn(error, alpha, beta):
    """Compute bounded reward function."""
    return 1.0 / (torch.exp(alpha * error) + beta + torch.exp(-alpha * error))

def collapse_obs_dict(obs_dict, obs_order) -> torch.Tensor:
    """Stack observations in given order."""
    obs_tensors = [obs_dict[obs_name] for obs_name in obs_order]
    obs_tensors = torch.cat(obs_tensors, dim=-1)
    return obs_tensors