import pyro.distributions as dist

# total number of apples (known by speaker and listener)
from torch import tensor

from qqn.initial_exploration.webppl import viz

total_apples = 3

# red apple base rate
base_rate_red = 0.8

binom_dist = dist.Binomial(total_count=total_apples, probs=tensor(base_rate_red))


# state = how many apples of 'total_apples' are red?
def state_pior():
    return binom_dist.sample()


samples = [state_pior() for _ in range(100)]

viz(binom_dist)


#### TODO