import pyro
import pyro.distributions as dist
import torch
from pyro import poutine
from pyro.infer import Trace_ELBO, SVI
from pyro.optim import Adam
from torch import Tensor
from tqdm import trange

cache = {}


def state_to_name(state: Tensor) -> str:
    state_idx = state.item()
    return {0: "pizza", 1: "steak frites"}.get(state_idx, "Unknown state")


def action_to_name(action: Tensor) -> str:
    action_idx = action.item()
    return {0: 'italian', 1: 'french'}.get(action_idx, "Unknown action")


def action_prior_weights(state) -> Tensor:
    return torch.zeros(2)


def is_final_state(state: Tensor, action: Tensor) -> Tensor:
    pass


def transition(state: Tensor, action: Tensor) -> Tensor:
    return action


def expected_util(state: Tensor, action: Tensor, **kwargs) -> Tensor:
    name_of_state = state_to_name(state)
    name_of_action = action_to_name(action)
    if name_of_state not in cache['expected_util']:
        cache['expected_util'][name_of_state] = {}
    if name_of_action not in cache['expected_util'][name_of_state]:
        optim_constructor = kwargs.get('optim_constructor', Adam)
        optim_args = kwargs.get('optim_args', [])
        optim_kwargs = kwargs.get('optim_kwargs', {})
        optim = optim_constructor(*optim_args, **optim_kwargs)
        loss_constructor = kwargs.get('loss_constructor', Trace_ELBO)
        loss_args = kwargs.get('loss_args', [])
        loss_kwargs = kwargs.get('loss_kwargs', {})
        opt_steps = kwargs.get('opt_steps', 1_000)
        alpha = kwargs.get('alpha', 1.0)
        loss = loss_constructor(*loss_args, **loss_kwargs)
        svi = SVI(expected_util_model, expected_util_guide, optim, loss)
        for _ in trange(opt_steps):
            with poutine.block():
                svi.step(state, action)

        cache['expected_util'][name_of_state][name_of_action] = 0.0
    return cache['expected_util'][name_of_state][name_of_action]


def expected_util_model(state, action):
    pass


def expected_util_guide(state, action):
    pass


def action_dist(state, **kwargs):
    if 'action_dist' not in cache:
        cache['action_dist'] = {}
    name_of_state = state_to_name(state)
    if name_of_state not in cache['action_dist']:
        optim_constructor = kwargs.get('optim_constructor', Adam)
        optim_args = kwargs.get('optim_args', [])
        optim_kwargs = kwargs.get('optim_kwargs', {})
        optim = optim_constructor(*optim_args, **optim_kwargs)
        loss_constructor = kwargs.get('loss_constructor', Trace_ELBO)
        loss_args = kwargs.get('loss_args', [])
        loss_kwargs = kwargs.get('loss_kwargs', {})
        opt_steps = kwargs.get('opt_steps', 1_000)
        alpha = kwargs.get('alpha', 1.0)
        loss = loss_constructor(*loss_args, **loss_kwargs)
        svi = SVI(next_action_model, next_action_guide, optim, loss)
        cache['action_dist'][name_of_state] = action_prior_weights(state)
        for _ in trange(opt_steps):
            with poutine.block():
                svi.step(state, alpha=alpha)
            cache['action_dist'][name_of_state] = pyro.param(f"next_action_{name_of_state}_preferences")
    return cache['action_dist'][name_of_state]


def next_action_model(state, alpha=1.0) -> Tensor:
    next_action_dist = dist.Categorical(logits=action_dist(state))
    next_action = pyro.sample("next_action", next_action_dist)
    util = expected_util(state, next_action)
    pyro.factor("action_factor", alpha * util)
    return next_action


def next_action_guide(state) -> Tensor:
    name_of_state = state_to_name(state)
    preferences = pyro.param(f"next_action_{name_of_state}_preferences", action_dist(state))
    next_action = pyro.sample("next_action", dist.Categorical(logits=preferences))
    return next_action
