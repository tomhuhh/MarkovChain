from os import stat
import numpy as np
import pandas as pd

# Model dimensions (PAR, MIM, MIP)
# PAR = Parity number; 1 to 10
# MIM = Months after calving or months in milk; 1 to 20
# MIP = Months in pregnancy; 0 to 9, 0 for nonpregnant, and 1 to 9 for pregnant
MAX_PAR = 10
MAX_MIM = 20
MAX_MIP = 9

states = []
for par in range(1, MAX_PAR+1):
    for mim in range(1, MAX_MIM+1):
        for mip in range(0, MAX_MIP+1):
            # Exclude impossible states: cannot be pregnant for more months than months in milk
            if mip == 0:
                states.append((par, mim, mip))
            elif mip + 2 <= mim:
                states.append((par, mim, mip))
state_idx = {s: i for i, s in enumerate(states)}
n_states = len(states)

print(n_states, "states generated")