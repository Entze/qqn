from frozenlist import FrozenList

from qqn.library.action import all_actions_type, nr_of_actions_type, action_islegal_type, action_prior_type
from qqn.library.option import option_generator_type, option_estimator_type, option_heuristic_type, \
    option_map_estimator_type, option_rater_type, option_selector_type, option_collapser_type
from qqn.library.policy import policy_type, policy_posterior_type, policy_distribution_type
from qqn.library.simulate import simulate_type
from qqn.library.state import state_key_type, state_resource_type, resource_depleted_type, state_isfinal_type, \
    state_value_type
from qqn.library.transition import transition_type

qqn_types = FrozenList([all_actions_type, nr_of_actions_type, action_islegal_type, action_prior_type,
                        option_generator_type, option_estimator_type, option_heuristic_type, option_map_estimator_type,
                        option_rater_type, option_selector_type, option_collapser_type,
                        policy_type, policy_posterior_type, policy_distribution_type,
                        simulate_type,
                        state_key_type, state_resource_type, resource_depleted_type, state_isfinal_type,
                        state_value_type,
                        transition_type
                        ])

qqn_types.freeze()
