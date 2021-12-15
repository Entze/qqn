import pyro
import pyro.distributions


def weather(p_cloudy):
    is_cloudy = pyro.sample("is_cloudy", pyro.distributions.Bernoulli(p_cloudy))
    if is_cloudy:
        loc, scale = 55.0, 10.0
    else:
        loc, scale = 75.0, 15.0
    temperature = pyro.sample("temp", pyro.distributions.Normal(loc, scale))
    return is_cloudy, temperature



print(weather(0.5))

