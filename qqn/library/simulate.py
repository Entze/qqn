from pyro.poutine.runtime import effectful

from qqn.library.common import nothing
from qqn.library.option import option_generator_eff, option_estimator_eff, option_rater_eff, option_selector_eff, \
    option_map_estimator_eff
from qqn.library.policy import policy_eff
from qqn.library.state import state_isfinal_eff
from qqn.library.transition import transition_eff


def simulate_by_sampling(initial_state):
    curr_state = initial_state
    trace = [(curr_state, None)]
    while not state_isfinal_eff(curr_state):
        action = policy_eff(curr_state)
        curr_state = transition_eff(curr_state, action)
        trace.append((curr_state, action))
    trace.append((curr_state, None))
    return trace


def simulate_by_enumeration(initial_state):
    unfinished_traces = [(initial_state, None)]
    finished_traces = []
    while unfinished_traces:
        curr_state, last_action = unfinished_traces.pop()
        options = option_generator_eff(curr_state)
        estimations = [option_estimator_eff(curr_state, option) for option in options]
        pass
    return finished_traces


simulate_type = 'simulate'
_simulate_eff = effectful(nothing, type=simulate_type)


def simulate_eff(initial_state):
    args = (initial_state,)
    return _simulate_eff(*args)
