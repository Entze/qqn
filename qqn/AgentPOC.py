states = [1, 2, 3]
actions = [1, 2]
alpha = 1


def transition(s, a):
    return {
        (0, 1): 1,
        (1, 1): 2,
        (1, 2): 3,
        (2, 1): 3,
    }.get((s, a), None)


weights = {
    (0, 1): 1.,
    (1, 1): .25,
    (1, 2): .75,
    (2, 1): 1.
}


def reward(s, a):
    return {
        (0, 1): 0,
        (1, 1): 5,
        (1, 2): 1,
        (2, 1): 3
    }.get((s, a), 0)


def is_final_state(s, a):
    return transition(s, a) is None


def expected_util(s, a):
    util = reward(s, a)

    if is_final_state(s, a):
        print("final_state:", s, util)
        return util

    next_state = transition(s, a)
    print("state:", next_state)

    future_rewards = []
    future_weights = []

    for act in actions:
        print("act:", act)
        next_util = expected_util(next_state, act)
        next_weight = weights.get((next_state, act), 0)
        future_rewards.append(next_util * next_weight)
        future_weights.append(next_weight)
        print(f"util: {next_util}, weight: {next_weight}, rew: {future_rewards[-1]}")
    all_rewards = sum(future_rewards)
    all_weights = sum(future_weights)
    if all_weights == 0:
        return util
    return util + all_rewards / all_weights


print(expected_util(0, 1))
