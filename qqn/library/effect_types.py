from frozenlist import FrozenList

from qqn.library.action import all_actions_type, nr_of_actions_type, action_islegal_type, action_prior_type, \
    action_generate_type, action_estimate_type, action_heuristic_type, \
    action_map_estimate_type, action_rate_type, action_select_type, action_collapse_type
from qqn.library.policy import policy_type, policy_posterior_type, policy_distribution_type
from qqn.library.simulate import simulate_type
from qqn.library.state import state_key_type, state_resource_type, resource_depleted_type, state_isfinal_type, \
    state_value_type
from qqn.library.transition import transition_type

qqn_types = FrozenList([all_actions_type, nr_of_actions_type, action_islegal_type, action_prior_type,
                        action_generate_type, action_estimate_type, action_heuristic_type, action_map_estimate_type,
                        action_rate_type, action_select_type, action_collapse_type,
                        policy_type, policy_posterior_type, policy_distribution_type,
                        simulate_type,
                        state_key_type, state_resource_type, resource_depleted_type, state_isfinal_type,
                        state_value_type,
                        transition_type
                        ])

qqn_types.freeze()
