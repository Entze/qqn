import random

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


class Infer:
    def __init__(self, fn, samples=100):
        self.fn = fn
        self.results = [fn() for i in range(samples)]

    @property
    def expectation(self):
        return sum(self.results) / len(self.results)

    def sample(self):
        return self.fn()


def state_to_action(s):
    return [act for act in actions if transition(s, act) is not None]


def next_action(s):
    draw = random.randint(0, len(actions) - 1)
    drawed_action = actions[draw]
    eu = expected_util(s, drawed_action)
    return drawed_action


def expected_util(s, a):
    util = reward(s, a)

    if is_final_state(s, a):
        print("final_state:", s, util)
        return util

    next_state = transition(s, a)
    print("state:", next_state)

    future_rewards = []
    future_weights = []

    next_act = test_agent.sample()

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


def make_agent(s):
    return Infer(lambda: next_action(s))


test_agent = make_agent(0)

print(make_agent(0).sample())
