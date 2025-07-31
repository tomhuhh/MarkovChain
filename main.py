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

# Fixed transition probabilities
# Parity-specific culling rates (parities 1–10)
annual_cull = [0, 0.208, 0.289, 0.386] + [0.525]*7  # Index 0 unused for parities 1–10

# Convert to monthly
monthly_cull = [0] + [1 - (1 - x) ** (1/12) for x in annual_cull[1:]]
# monthly_cull[1] is for parity 1, [2] for parity 2, etc.

# Pregnancy and abortion (monthly)
monthly_preg = 0.35 / 12 
monthly_abort = 0.065 / 9

T = np.zeros((n_states, n_states))

for i, (par, mim, mip) in enumerate(states):
    # Culling: parity capped at 4+ for culling rates
    cull_p = monthly_cull[min(par, 4)]
    
    # === 1. Nonpregnant cows ===
    if mip == 0:
        # (a) Stay not pregnant
        next_mim = mim + 1 if mim < MAX_MIM else 1
        next_state_stillopen = (par, next_mim, 0)
        T[i, state_idx.get(next_state_stillopen, i)] += (1 - cull_p) * (1 - monthly_preg)
        
        # (b) Become pregnant
        next_state_preg = (par, next_mim, 1)
        T[i, state_idx.get(next_state_preg, i)] += (1 - cull_p) * monthly_preg

        # (c) Culling: always transitions to fresh heifer state (1,1,0)
        T[i, state_idx[(1, 1, 0)]] += cull_p

    # === 2. Pregnant cows (not yet calving) ===
    elif mip < MAX_MIP:
        next_mim = mim + 1 if mim < MAX_MIM else 1
        # (a) Progress pregnancy (not aborting)
        next_state_preg = (par, next_mim, mip + 1)
        T[i, state_idx.get(next_state_preg, i)] += (1 - cull_p) * (1 - monthly_abort)
        
        # (b) Abort: return to open (not pregnant)
        next_state_abort = (par, next_mim, 0)
        T[i, state_idx.get(next_state_abort, i)] += (1 - cull_p) * monthly_abort

        # (c) Culling: to fresh heifer
        T[i, state_idx[(1, 1, 0)]] += cull_p

    # === 3. Pregnant cows at month 9 (calving) ===
    else:  # mip == MAX_MIP (about to calve)
        # (a) Calving: move to next parity, MIM=1, MIP=0
        next_par = min(par + 1, MAX_PAR)
        next_state_fresh = (next_par, 1, 0)
        T[i, state_idx.get(next_state_fresh, i)] += (1 - cull_p) * (1 - monthly_abort)
        
        # (b) Abort in 9th month: next lactation, open
        next_mim = 1 if mim < MAX_MIM else 1
        next_state_abort = (par, next_mim, 0)
        T[i, state_idx.get(next_state_abort, i)] += (1 - cull_p) * monthly_abort
        
        # (c) Culling: to fresh heifer
        T[i, state_idx[(1, 1, 0)]] += cull_p

# Optional: Check that each row sums to (almost) 1.0
row_sums = T.sum(axis=1)
assert np.allclose(row_sums, 1), "Some transitions don't sum to 1!"