from itertools import chain
from typing import Optional, Tuple, List

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer import Trace_ELBO, SVI
from pyro.optim import Adam
import torch
from torch import tensor, Tensor
from tqdm import trange
import qqn.initial_exploration.gridworld as gw

cache = {}


def normalize_probs(probs: Tensor) -> Tensor:
    return probs / probs.sum()


def logits_to_probs(logits: Tensor) -> Tensor:
    return (logits - logits.max()).exp()


# state: [(Time) (Pos X) (Pos Y)]
___ = ' '
DN = 'DN'
DS = 'DS'
V = 'V'
N = 'N'

grid_raw = [
    ['#', '#', '#', '#', V, '#'],
    ['#', '#', '#', ___, ___, ___],
    ['#', '#', DN, ___, '#', ___],
    ['#', '#', '#', ___, '#', ___],
    ['#', '#', '#', ___, ___, ___],
    ['#', '#', '#', ___, '#', N],
    [___, ___, ___, ___, '#', '#'],
    [DS, '#', '#', ___, '#', '#']
]

gw_t = gw.as_tensor(grid_raw)


def state_value(state):
    """
    Rates the value of a state
    :param state: a tensor
    :return: a tensor
    """
    pos = state[1:]
    if (pos == tensor([0, 7])).all():
        return tensor(1.)  # Donut S
    elif (pos == tensor([2, 2])).all():
        return tensor(1.)
    elif (pos == tensor([4, 0])).all():
        return tensor(3.)
    elif (pos == tensor([5, 5])).all():
        return tensor(2.)
    return tensor(0.)


def action_value(state, action):
    """
    Rates the value of an action in a certain state
    :param state: a tensor
    :param action: a tensor
    :return: a tensor
    """
    time_left = state[0]
    assert time_left >= 1, "Action does not have a value with less than 1 time."
    next_state = transition(state, action)
    primary_value = state_value(next_state)
    new_time_left = next_state[0]
    if new_time_left <= 0:
        return primary_value
    pol = policy(next_state)
    secondary_value = policy_value(pol, next_state)
    return primary_value + secondary_value


def policy_value(pol, state):
    return policy_value_exact(pol, state)


def policy_value_exact(pol: Tensor, state):
    action_probs = normalize_probs(logits_to_probs(pol))
    action_values = tensor(
        [action_value(state, tensor(0)), action_value(state, tensor(1)),
         action_value(state, tensor(2)), action_value(state, tensor(3))])  # TODO vectorize?
    pv = (action_values * action_probs)
    pv = pv[~torch.any(pv.isnan())]
    if pv.size(dim=0) <= 0:
        return tensor(0.)
    return pv.mean()


def policy_value_sampling(pol, state, time_left):
    pass


def policy_estimation_sampling_model(pol, state, time_left):
    action = pyro.sample("pe_action", pol)
    return action_value(state, action)


def policy(state):
    """
    Returns the logits (unnormalized) for a categorical distribution over all possible actions.
    :param state: a tensor
    :return: logits
    """
    return action_prior(state)

    if 'policy' not in cache:
        cache['policy'] = {}
    state_idx = state.tolist()
    if state_idx not in cache['policy']:
        svi = SVI(policy_model, policy_guide, Adam({"lr": 0.025}), Trace_ELBO())
        cache['policy'][state_idx] = pyro.param(f"p_preferences_{state_idx}", action_prior(state))
        for _ in trange(100):
            svi.step(state)
            cache['policy'][state_idx] = pyro.param(f"p_preferences_{state_idx}")
    return cache['policy'][state_idx]


def policy_model(state):
    action_idx = pyro.sample("p_action_idx", dist.Categorical(logits=action_prior(state)))
    with poutine.block():
        au = action_value(state, action_idx)
    pyro.factor("p_factor", 10. * au)
    return action_idx


def policy_guide(state, time_left):
    preferences = pyro.param(f"p_preferences_{state.item()}_{time_left}")
    action = pyro.sample("p_action_idx", dist.Categorical(logits=preferences))
    return action


# Action 0: Nord
# Action 1: East
# Action 2: South
# Action 3: West

def transition(state, action_idx):
    t = tensor([0, 0])
    s = torch.clone(state)
    if action_idx == 0:  # North
        t = tensor([0, -1])  # Go one up

    elif action_idx == 1:  # East
        t = tensor([1, 0])  # Go one right

    elif action_idx == 2:  # South
        t = tensor([0, 1])  # Go one down

    elif action_idx == 3:  # West
        t = tensor([-1, 0])  # Go one left

    s[0] -= 1
    if state_value(s) > 0:
        s[0] = 0
    s[1:] += t
    return s


def action_prior(state):
    a = gw.allowed_actions(gw_t, state).float()
    a[a == 0] = float('-inf')
    a[a == 1] = 0.
    return a


def simulate_random(init_state):
    next = init_state
    trace = []
    x, y = next[1].item(), next[2].item()
    trace.append(((x, y), None))
    while next[0] > 0:
        p = policy(next)
        a = dist.Categorical(logits=p).sample()
        next = transition(p, a)
        x, y = next[1].item(), next[2].item()
        trace.append(((x, y), a.item()))
    x, y = next[1].item(), next[2].item()
    trace.append(((x, y), None))
    return trace


def tracer(init):
    if init[0] > 0:
        la = gw.allowed_actions(gw_t, init)
        next_states = [transition(init, tensor(a)) for a, t in enumerate(la) if t]
        prospect_traces = chain(*[tracer(next) for next in next_states])
        return [[init] + trace for trace in prospect_traces]
    return [[init]]

def simulate_all(state: Tensor):
    t, x, y = state.tolist()
    unfinished_traces: List[List[Tuple[int, Tuple[int, int], Optional[int]]]] = [[(t, (x, y), None)]]
    finished_traces: List[List[Tuple[int, Tuple[int, int], Optional[int]]]] = []
    while unfinished_traces:
        curr_trace = unfinished_traces.pop()
        t, (x, y), a = curr_trace[-1]
        if t <= 0:
            finished_traces.append(curr_trace)
            continue
        s = tensor([t, x, y])
        for a, t in enumerate(gw.allowed_actions(gw_t, s)):
            if t:
                trace = curr_trace.copy()
                next_state = transition(s, tensor(a))
                t, x, y = next_state.tolist()
                trace.append((t, (x, y), a))
                unfinished_traces.append(trace)
    return finished_traces


def simulate(init_state):
    curr_trace = []
    all_traces = []
    x, y = init_state[1].item(), init_state[2].item()
    curr_trace.append(((x, y), -1))
    if init_state[0] <= 0:
        all_traces.append(curr_trace)
    else:
        for a, t in enumerate(gw.allowed_actions(gw_t, init_state)):
            if t:
                all_traces += simulate_action(init_state, tensor(a), curr_trace.copy())
    return all_traces


def simulate_action(state, action, curr_trace):
    next_state = transition(state, action)
    x, y = next_state[1].item(), next_state[2].item()
    curr_trace.append(((x, y), action.item()))
    all_traces = []
    if next_state[0] <= 0:
        all_traces.append(curr_trace)
    else:
        for a, t in enumerate(gw.allowed_actions(gw_t, state)):
            if t:
                all_traces += simulate_action(next_state, tensor(a), curr_trace)
    return all_traces


def test():
    for x in range(6):
        for y in range(8):
            if grid_raw[y][x] != '#':
                val = state_value(tensor([1, x, y]))
                if val != 0:
                    print("x:", x, "y:", y, val)

    print('#' * 80)

    for x in range(6):
        for y in range(8):
            if grid_raw[y][x] != '#':
                s = tensor([1, x, y])
                for a, t in enumerate(gw.allowed_actions(gw_t, s)):
                    if t:
                        val = action_value(s, tensor(a))
                        if val != 0:
                            print("x:", x, "y:", y, "a:", a, val)

    print('#' * 80)

    for x in range(6):
        for y in range(8):
            if grid_raw[y][x] != '#':
                s = tensor([2, x, y])
                for a, t in enumerate(gw.allowed_actions(gw_t, s)):
                    if t:
                        val = action_value(s, tensor(a))
                        if val != 0:
                            print("x:", x, "y:", y, "a:", a, val)

    print('#' * 80)

    t = simulate_all(tensor([1, 0, 6]))
    print(t)
    t = simulate_all(tensor([2, 1, 6]))
    print(t)

    print('#' * 80)

    # for x in range(6):
    #     for y in range(8):
    #         if grid_raw[y][x] == ' ':
    #             for spirit in range(1, 5):
    #                 s = tensor([spirit, x, y])
    #                 t = simulate(s)
    #                 (x_t, y_t), _ = t[-1]
    #                 val = state_value(tensor([0, x_t, y_t]))
    #                 if val != 0:
    #                     print("x:", x, "y:", y, "s:", spirit, "t:", t, val)


torch.manual_seed(0)
pyro.set_rng_seed(0)
pyro.clear_param_store()
gw.display(grid_raw)
test()

