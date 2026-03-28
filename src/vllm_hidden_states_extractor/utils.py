import torch


def reshape_hidden_states_for_kv_cache(
    hidden_states: torch.Tensor, head_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    # hidden_states shape: [batch_size, hidden_size * num_hidden_states]
    # e.g. hidden_states = torch.cat([h_1, h_2, ..., h_n], dim=1)
    # where h_i is a hidden state of shape [batch_size, hidden_size]

    # Assuming num_hidden_states is a multiple of 2 for now

    batch_size = hidden_states.shape[0]
    split_size = hidden_states.shape[1] // 2
    key, value = torch.split(hidden_states, [split_size, split_size], dim=1)
    # key/value shape: [batch_size, hidden_size * num_hidden_states / 2]

    key = key.view(batch_size, -1, head_size)
    value = value.view(batch_size, -1, head_size)
    return key, value


def reshape_hidden_states_from_kv_cache(
    kv: torch.Tensor, num_hidden_states: int
) -> torch.Tensor:
    # kv shape: [2, batch_size, hidden_size / head_size * num_hidden_states / 2, head_size]
    kv = kv.flatten(2)
    # kv shape: [2, batch_size, hidden_size * num_hidden_states / 2]

    hidden_states = torch.cat([kv[0], kv[1]], dim=1)
    # hidden_states shape: [batch_size, hidden_size * num_hidden_states]

    split_size = hidden_states.shape[1] // num_hidden_states
    hidden_states = hidden_states.split(split_size, dim=1)

    return torch.stack(hidden_states, dim=0)
