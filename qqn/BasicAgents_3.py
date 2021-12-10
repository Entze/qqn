import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer import Trace_ELBO, SVI
from pyro.optim import Adam
import torch
from torch import tensor, Tensor
from tqdm import trange


def normalize_probs(probs: Tensor) -> Tensor:
    return probs / probs.sum()


def logits_to_probs(logits: Tensor) -> Tensor:
    return (logits - logits.max()).exp()


def state_value(state):
    """
    Rates the value of a state
    :param state: a tensor
    :return: a tensor
    """
    return torch.where(state == 3, 1, 0).float()


def action_value(state, action, time_left):
    """
    Rates the value of an action in a certain state
    :param state: a tensor
    :param action: a tensor
    :param time_left: a number
    :return: a tensor
    """
    assert time_left >= 1, "Action does not have a value with less than 1 time."
    next_state = transition(state, action)
    new_time_left = time_left - 1
    primary_value = state_value(next_state)
    if new_time_left == 0:
        return primary_value
    pol = policy(next_state, new_time_left)
    secondary_value = policy_value(pol, next_state, new_time_left)
    return primary_value + secondary_value


def policy_value(pol, state, time_left):
    return policy_estimation_exact(pol, state, time_left)


def policy_estimation_exact(pol: Tensor, state, time_left):
    action_probs = normalize_probs(logits_to_probs(pol))
    action_values = tensor(
        [action_value(state, tensor(0), time_left), action_value(state, tensor(1), time_left), action_value(state, tensor(2), time_left)])  # TODO vectorize?
    return (action_values * action_probs).mean()


def policy_estimation_sampling(pol, state, time_left):
    pass


def policy_estimation_sampling_model(pol, state, time_left):
    action = pyro.sample("pe_action", pol)
    return action_value(state, action, time_left)


def policy(state, time_left):
    """
    Returns the logits (unnormalized) for a categorical distribution over all possible actions.
    :param state: a tensor
    :return: logits
    """
    svi = SVI(policy_model, policy_guide, Adam({"lr": 0.5}), Trace_ELBO())
    for _ in trange(100):
        # pol_before = normalize_probs(
        #     logits_to_probs(pyro.param(f"p_preferences_{state.item()}_{time_left}", torch.zeros(2))))
        svi.step(state, time_left)
        # pol_after = normalize_probs(logits_to_probs(pyro.param(f"p_preferences_{state.item()}_{time_left}")))
        # pol_diff = pol_after - pol_before
        # print("Adjusted policy by: {:.3f},{:.3f}. \nFrom {:.3f},{:.3f}\nTo {:.3f},{:.3f}".format(pol_diff[0].item(),
        #                                                                                         pol_diff[1].item(),
        #                                                                                         pol_before[0].item(),
        #                                                                                         pol_before[1].item(),
        #                                                                                         pol_after[0].item(),
        #                                                                                         pol_after[1].item()))

    return pyro.param(f"p_preferences_{state.item()}_{time_left}")


def policy_model(state, time_left):
    action_idx = pyro.sample("p_action_idx", dist.Categorical(logits=torch.zeros(3)))
    with poutine.block():
        au = action_value(state, action_idx, time_left)
    pyro.factor("p_factor", au)
    return action_idx


def policy_guide(state, time_left):
    preferences = pyro.param(f"p_preferences_{state.item()}_{time_left}", torch.zeros(3))
    action = pyro.sample("p_action_idx", dist.Categorical(logits=preferences))
    return action


# State 0: Value(0)
# State 1: Value(1)

# Action 0: Stay
# Action 1: Move to the right

def transition(state, action_idx):
    step = action_idx - 1
    return torch.minimum(torch.maximum(state + step, tensor(-1)), tensor(1))


def test():
    for state in range(-1, 2):
        print("state_value:", state, ", ", state_value(tensor(state)))
    print('#' * 80)
    for state in range(-1, 2):
        for action in range(2):
            au = action_value(tensor(state), tensor(action), 1)
            print()
            print("action_value:", state, ",", action, ",", au)
    print('#' * 80)
    for state in range(-1, 2):
        pol = normalize_probs(logits_to_probs(policy(tensor(state), 1)))
        print()
        print("policy:", state, ",", pol)


torch.manual_seed(0)
pyro.set_rng_seed(0)
pyro.clear_param_store()
# test()
p = policy(tensor(0), 3)
print(normalize_probs(logits_to_probs(p)))
