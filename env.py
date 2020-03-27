# Contextual Bandit
# Environment in which the agent acts
# Action space: [0, 1000]
# State space: [0, 500]
# Reward: Bimodal distribution centered at (state) and (state*2)

import numpy as np
import math

stdev = 20

def query(s_t, a_t):
    normal_thing = 1/(stdev * math.sqrt(2 * math.pi))
    return (normal_thing * math.pow(math.e, -math.pow(a_t-s_t,2)/(2*math.pow(stdev,2)))
          + normal_thing * math.pow(math.e, -math.pow(a_t-2*s_t,2)/(2*math.pow(stdev,2))))

