import contextlib
import datetime
import time
from collections import defaultdict
from pprint import pprint

from torch.distributions.utils import logits_to_probs
from tqdm import trange

from qqn.library.action import action_estimate_type
from qqn.library.cacher import Cacher
from qqn.library.estimator_messenger import SamplingEstimatingAgentMessenger
from qqn.library.max_collapse_messenger import MaxCollapseMessenger
from qqn.library.policy import policy_eff, policy_posterior_eff
from qqn.library.simulate import simulate_by_sampling
from qqn.library.softmaxagent_messenger import softmax_agent
from qqn.library.weighted_rate_messenger import WeightedRateMessenger


def test_context(initial_state, traces=100, progressbar=True):
    trajectory_length = defaultdict(int)
    for _ in trange(traces, disable=not progressbar):
        trace = simulate_by_sampling(initial_state)
        trajectory_length[len(trace) - 2] += 1
    pprint(dict(trajectory_length))
    print(simulate_by_sampling(initial_state))
    print(policy_eff(initial_state))
    print(logits_to_probs(policy_posterior_eff(initial_state).float()))


def test(initial_state,
         nr_of_actions,
         transition,
         state_value,
         min_estimation_value,
         max_estimation_value,
         alpha=10.,
         action_islegal=contextlib.nullcontext(),
         state_isfinal=contextlib.nullcontext(),
         traces=100,
         progressbar=True):
    starttime = time.monotonic()
    print('#' * 80)
    print("Argmax:")
    with nr_of_actions, transition, state_value, action_islegal, state_isfinal, Cacher(types=[action_estimate_type]):
        test_context(initial_state, traces, progressbar)

    print('#' * 80)
    print("Weighted Argmax:")
    with nr_of_actions, transition, state_value, action_islegal, state_isfinal, WeightedRateMessenger(
            alpha=alpha), Cacher(types=[action_estimate_type]):
        test_context(initial_state, traces, progressbar)

    print('#' * 80)
    print("Argmax with max collapse:")
    with nr_of_actions, transition, state_value, action_islegal, state_isfinal, MaxCollapseMessenger(), Cacher(
            types=[action_estimate_type]):
        test_context(initial_state, traces, progressbar)

    print('#' * 80)
    print("Weighted Argmax with max collapse:")
    with nr_of_actions, transition, state_value, action_islegal, state_isfinal, MaxCollapseMessenger(), WeightedRateMessenger(
            alpha=alpha), Cacher(types=[action_estimate_type]):
        test_context(initial_state, traces, progressbar)

    print('#' * 80)
    print("Weighted Softmax:")
    with nr_of_actions, transition, state_value, action_islegal, state_isfinal, WeightedRateMessenger(
            alpha=alpha), softmax_agent(), Cacher(types=[action_estimate_type]):
        test_context(initial_state, traces, progressbar)

    print('#' * 80)
    print("Weighted Softmax with max collapse:")
    with nr_of_actions, transition, state_value, action_islegal, state_isfinal, WeightedRateMessenger(
            alpha=alpha), softmax_agent(), MaxCollapseMessenger(), Cacher(types=[action_estimate_type]):
        test_context(initial_state, traces, progressbar)

    print('#' * 80)
    print("Softmax with estimation sampling:")
    with nr_of_actions, state_value, transition, action_islegal, state_isfinal, softmax_agent(), SamplingEstimatingAgentMessenger(
            min_estimation_value=min_estimation_value,
            max_estimation_value=max_estimation_value,
            nr_of_bins=(max_estimation_value - min_estimation_value) * 10 + 1,
            optimization_steps=50), \
            Cacher(types=[action_estimate_type]):
        test_context(initial_state, traces, progressbar)

    print('#' * 80)
    print("Weighted Softmax with estimation sampling:")
    with nr_of_actions, state_value, transition, action_islegal, state_isfinal, WeightedRateMessenger(
            alpha=alpha), softmax_agent(), SamplingEstimatingAgentMessenger(
        min_estimation_value=min_estimation_value,
        max_estimation_value=max_estimation_value,
        nr_of_bins=(max_estimation_value - min_estimation_value) * 10 + 1,
        optimization_steps=50), \
            Cacher(types=[action_estimate_type]):
        test_context(initial_state, traces, progressbar)

    print('#' * 80)
    print("Softmax with estimation sampling and max collapse:")
    with nr_of_actions, state_value, transition, action_islegal, state_isfinal, MaxCollapseMessenger(), softmax_agent(), SamplingEstimatingAgentMessenger(
            min_estimation_value=min_estimation_value,
            max_estimation_value=max_estimation_value,
            nr_of_bins=(max_estimation_value - min_estimation_value) * 10 + 1,
            optimization_steps=50), \
            Cacher(types=[action_estimate_type]):
        test_context(initial_state, traces, progressbar)

    print('#' * 80)
    print("Weighted Softmax with estimation sampling and max collapse:")
    with nr_of_actions, state_value, transition, action_islegal, state_isfinal, MaxCollapseMessenger(), WeightedRateMessenger(
            alpha=alpha), softmax_agent(), SamplingEstimatingAgentMessenger(
        min_estimation_value=min_estimation_value,
        max_estimation_value=max_estimation_value,
        nr_of_bins=(max_estimation_value - min_estimation_value) * 10 + 1,
        optimization_steps=50), \
            Cacher(types=[action_estimate_type]):
        test_context(initial_state, traces, progressbar)
    print('#' * 80)
    print('#' * 80)
    print('#' * 80)
    print(f"Finished in {datetime.timedelta(seconds=time.monotonic() - starttime)}")
