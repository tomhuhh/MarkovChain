from os import stat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Model dimensions (PAR, MIM, MIP)
# PAR = Parity number; 1 to 10
# MIM = Months after calving or months in milk; 1 to 33
# MIP = Months in pregnancy; 0 to 9, 0 for nonpregnant, and 1 to 9 for pregnant
MAX_PAR = 10
MAX_MIM = 33
MAX_MIP = 9

states = []
for par in range(1, MAX_PAR+1):
    for mim in range(1, MAX_MIM+1):
        for mip in range(-1, MAX_MIP+1):  # mip = -1 for "do not breed", 0 for open, 1-9 for pregnant
            if mip == -1:
                states.append((par, mim, mip))
            if mip == 0:
                states.append((par, mim, mip))
            # Exclude impossible states: cannot be pregnant for more months than months in milk
            elif mip + 2 <= mim:
                states.append((par, mim, mip))
state_idx = {s: i for i, s in enumerate(states)}
n_states = len(states)

# Fixed transition probabilities
# Parity-specific culling rates (parities 1–10)
annual_cull = [0, 0.083, 0.114, 0.141] + [0.200]*7  # Index 0 unused for parities 1–10

# Convert to monthly
monthly_cull = [0] + [1 - (1 - x) ** (1/12) for x in annual_cull[1:]]
# monthly_cull[1] is for parity 1, [2] for parity 2, etc.

# Pregnancy and abortion (monthly)
monthly_preg = 0.6 / 12 
monthly_abort = 0.065 / 9

# User-defined milk threshold
Milk_Threshold_to_Cull = 22.7  # Example threshold

def milk_yield(par, mim):
    """Calculate milk yield based on Wood's lactation curve."""
    # Wood's base parameters
    a_base = 19.9
    b_base = 24.7
    c_base = 33.76
    # Adjust parameters based on parity
    parity_adj = {
        1: {"a": -4.18, "b": -0.37, "c": -9.31},
        2: {"a": 2.16, "b": -1.20, "c": 2.66},
        3: {"a": 2.02, "b": 1.57, "c": 6.65},  # 3 = parity 3 and beyond
    }

    # Use parity 3 adjustments for parity >= 3
    p = par if par in [1, 2] else 3
    adj = parity_adj[p]
    a = a_base + adj["a"]
    b = (b_base + adj["b"])*(np.e-2)
    c = (c_base + adj["c"])*(np.e-4)
    DIM = mim * 30  # Convert months in milk to days
    return a * (DIM ** b) * np.exp(-c * DIM)

Last_MIM_to_Breed = 10  # Example: last month in milk to breed

# Transition matrix initialization
# T[i, j] = Probability of transitioning from state i to state j
# n_states x n_states matrix
T = np.zeros((n_states, n_states))

for i, (par, mim, mip) in enumerate(states):
    # Culling: parity capped at 4+ for culling rates
    cull_p = monthly_cull[min(par, 4)]
    
    # === 1. Nonpregnant cows, normal ===
    if mip == 0:
        if mim <= Last_MIM_to_Breed:
            # Normal transitions (can breed)
            if mim < MAX_MIM:
                next_mim = mim + 1
                next_state_stillopen = (par, next_mim, 0)
                T[i, state_idx.get(next_state_stillopen, i)] += (1 - cull_p) * (1 - monthly_preg)
                next_state_preg = (par, next_mim, 1)
                T[i, state_idx.get(next_state_preg, i)] += (1 - cull_p) * monthly_preg
            else:
                T[i, state_idx[(1, 1, 0)]] += (1 - cull_p)
            T[i, state_idx[(1, 1, 0)]] += cull_p
        else:
            # After last MIM to breed, check milk
            if milk_yield(par, mim) < Milk_Threshold_to_Cull:
                # Cull for low milk
                T[i, state_idx[(1, 1, 0)]] = 1.0
            else:
                # Transition to "do not breed" state
                next_mim = mim + 1 if mim < MAX_MIM else MAX_MIM
                next_state_dnb = (par, next_mim, -1)
                T[i, state_idx.get(next_state_dnb, i)] = 1.0

    # === 2. Nonpregnant, do not breed ===
    elif mip == -1:
        if milk_yield(par, mim) < Milk_Threshold_to_Cull:
            T[i, state_idx[(1, 1, 0)]] = 1.0
        else:
            next_mim = mim + 1 if mim < MAX_MIM else MAX_MIM
            next_state_dnb = (par, next_mim, -1)
            T[i, state_idx.get(next_state_dnb, i)] = 1.0

    # === 3. Pregnant cows (not yet calving) ===
    elif mip < MAX_MIP:
        next_mim = mim + 1 if mim < MAX_MIM else MAX_MIM
        # (a) Progress pregnancy (not aborting)
        next_state_preg = (par, next_mim, mip + 1)
        T[i, state_idx.get(next_state_preg, i)] += (1 - cull_p) * (1 - monthly_abort)
        
        # (b) Abort: return to open (not pregnant)
        if mim < MAX_MIM:
            next_mim = mim + 1
            next_state_abort = (par, next_mim, 0)
            T[i, state_idx.get(next_state_abort, i)] += (1 - cull_p) * monthly_abort
        else:
            # At maximum MIM: if not calved and not pregnant, cull the cow
            T[i, state_idx[(1, 1, 0)]] += monthly_abort * (1 - cull_p)

        # (c) Culling: to fresh heifer
        T[i, state_idx[(1, 1, 0)]] += cull_p

    # === 4. Pregnant cows at month 9 (calving) ===
    else:  # mip == MAX_MIP (about to calve)
        # (a) Calving: move to next parity, MIM=1, MIP=0
        next_par = min(par + 1, MAX_PAR)
        next_state_fresh = (next_par, 1, 0)
        T[i, state_idx.get(next_state_fresh, i)] += (1 - cull_p) * (1 - monthly_abort)
        
        # (b) Abort in 9th month: the current lactation, open
        next_mim = min(mim + 1, MAX_MIM)
        next_state_abort = (par, next_mim, 0)
        T[i, state_idx.get(next_state_abort, i)] += (1 - cull_p) * monthly_abort
        
        # (c) Culling: to fresh heifer
        T[i, state_idx[(1, 1, 0)]] += cull_p

# Check that each row sums to (almost) 1.0
row_sums = T.sum(axis=1)
assert np.allclose(row_sums, 1), "Some transitions don't sum to 1!"

# Initial state vector
# Start with all cows in PAR=1, MIM=1, MIP=0 (fresh heifers)
herd = np.zeros(n_states)
start_state = (1, 1, 0)
herd[state_idx[start_state]] = 1000

# Simulate herd evolution over a specified number of steps
n_steps = 200  # months
herd_evolution = [herd.copy()]
for step in range(n_steps):
    herd = herd @ T
    herd_evolution.append(herd.copy())
herd_evolution = np.array(herd_evolution)

# Extract list of parities from states
parities = sorted(set(par for par, _, _ in states))
n_steps = herd_evolution.shape[0]

# Plot number of cows in each parity group over time
for par in parities:
    par_indices = [i for i, (p, _, _) in enumerate(states) if p == par]
    par_counts = herd_evolution[:, par_indices].sum(axis=1)
    plt.plot(par_counts, label=f'Parity {par}')

plt.xlabel('Month')
plt.ylabel('Number of Cows')
plt.title('Cows by Parity Over Time')
plt.legend()
plt.tight_layout()
plt.show()