from pyro.poutine.runtime import effectful

from qqn.library.action import action_generate_eff, action_estimate_eff, action_islegal_eff
from qqn.library.common import nothing
from qqn.library.policy import policy_eff
from qqn.library.state import state_isfinal_eff
from qqn.library.transition import transition_eff


def simulate_by_sampling(initial_state, *args, **kwargs):
    curr_state = initial_state
    trace = [(curr_state, None)]
    while not state_isfinal_eff(curr_state, *args, **kwargs):
        action = policy_eff(curr_state)
        assert action_islegal_eff(action, curr_state)
        curr_state = transition_eff(curr_state, action)
        trace.append((curr_state, action))
    trace.append((curr_state, None))
    return trace


def simulate_by_enumeration(initial_state, *args, **kwargs):
    unfinished_traces = [(initial_state, None)]
    finished_traces = []
    while unfinished_traces:
        curr_state, last_action = unfinished_traces.pop()
        options = action_generate_eff(curr_state, *args, **kwargs)
        estimations = [action_estimate_eff(curr_state, option) for option in options]
        pass
    return finished_traces


simulate_type = 'simulate'
_simulate_eff = effectful(simulate_by_sampling, type=simulate_type)


def simulate_eff(initial_state, *args, **kwargs):
    req_args = (initial_state,)
    return _simulate_eff(*req_args, *args, **kwargs)
