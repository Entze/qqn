import contextlib
import datetime
import time
from collections import defaultdict
from pprint import pprint

from torch.distributions.utils import logits_to_probs
from tqdm import trange

from qqn.library.action import action_estimate_type
from qqn.library.argmaxagent_messenger import argmax_agent
from qqn.library.cacher import Cacher
from qqn.library.estimator_messenger import sampling_estimating_agent_gen, svi_estimating_agent_gen
from qqn.library.max_collapse_messenger import max_collapse_agent
from qqn.library.policy import policy_eff, policy_posterior_eff
from qqn.library.simulate import simulate_by_sampling, simulate_eff
from qqn.library.softmaxagent_messenger import softmax_agent
from qqn.library.weighted_rate_messenger import weighted_rate_agent_gen


def test_context(initial_state, traces=100, progressbar=True):
    trajectory_length = defaultdict(int)
    for _ in trange(traces, disable=not progressbar):
        trace = simulate_by_sampling(initial_state)
        trajectory_length[len(trace) - 2] += 1
    pprint(dict(trajectory_length))
    print(simulate_eff(initial_state))
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
         update_belief=contextlib.nullcontext(),
         traces=100,
         progressbar=True):
    starttime = time.monotonic()
    with nr_of_actions, transition, state_value, action_islegal, state_isfinal, update_belief:

        # ARGMAX vs. SOFTMAX
        # WEIGHTED vs. UNWEIGHTED
        # MEAN vs. MAX
        # ONESHOT vs. SAMPLING vs. VI

        for agent_name, agent in (('Argmax', argmax_agent), ('Softmax', softmax_agent)):
            for weighted_name, weighted in (
                    ('Unweighted', contextlib.nullcontext), ('Weighted', weighted_rate_agent_gen(alpha=alpha))):
                for collapse_name, collapse in (('Mean', contextlib.nullcontext), ('Max', max_collapse_agent)):
                    for inference_name, infer in (('Oneshot', contextlib.nullcontext),
                                                  ('Sampling', sampling_estimating_agent_gen(
                                                      min_estimation_value=min_estimation_value,
                                                      max_estimation_value=max_estimation_value,
                                                      nr_of_bins=100,
                                                      optimization_steps=50
                                                  )),
                                                  ('VI', svi_estimating_agent_gen(
                                                      min_estimation_value=min_estimation_value,
                                                      max_estimation_value=max_estimation_value,
                                                      optimization_steps=10,
                                                      optim_args=dict(lr=0.25)
                                                  ))):
                        print('#' * 80)
                        print(f"{agent_name} {weighted_name} {collapse_name} {inference_name}:")
                        with agent(), weighted(), collapse(), infer(), Cacher(types=[action_estimate_type]):
                            test_context(initial_state, traces, progressbar)

    print('#' * 80)
    print('#' * 80)
    print('#' * 80)
    print(f"Finished in {datetime.timedelta(seconds=time.monotonic() - starttime)}")
