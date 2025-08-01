import numpy as np
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
    b = (b_base + adj["b"])/100
    c = (c_base + adj["c"])/10000
    DIM = max(mim * 30, 1)  # Convert months in milk to days; Avoid DIM=0
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
                next_mim = MAX_MIM
                next_state_stillopen = (par, next_mim, 0)
                T[i, state_idx.get(next_state_stillopen, i)] += (1 - cull_p) * (1 - monthly_preg)
                next_state_preg = (par, next_mim, 1)
                T[i, state_idx.get(next_state_preg, i)] += (1 - cull_p) * monthly_preg
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
            next_mim = MAX_MIM
            next_state_abort = (par, next_mim, 0)
            T[i, state_idx.get(next_state_abort, i)] += (1 - cull_p) * monthly_abort

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
herd[state_idx[start_state]] = 1

# Simulate herd evolution over a specified number of steps
n_steps = 150  # months
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

# Economic parameters
milk_price = 0.35  # $/kg
feed_intake = 22   # kg/cow/day
feed_price = 0.2   # $/kg
repro_cost = 20    # $/cow/month
replacement_cost = 1300  # $/cow
salvage_value_per_kg = 0.84  # $/kg body weight
body_weight = 650  # kg/cow
calf_value = 100   # $/calf
discount_rate = 0.06  # annual
monthly_discount = (1 + discount_rate) ** (-1/12)

# Methane-reducing additive-related constants
additive_cost_per_g = 0.33  # $/g (Pupo et al., 2025)
additive_dose_mg_per_kg_feed = 80  # mg per kg feed intake
milk_fat_pct = 3.8  # %
milk_protein_pct = 3.2  # %
ndf_pct = 33  # %
methane_intensity_reduction = 0.32  # 32%
feed_reduction = 0.09     # 9%
milk_reduction = 0.05     # 5%

n_months = herd_evolution[1:].shape[0]

def methane_production(feed_intake, ndf_pct, milk_fat_pct, body_weight):
    # Returns methane production in g/day (Niu et al., 2018)
    methane_g_per_day = (
        -126
        + 11.3 * feed_intake
        + 2.30 * ndf_pct
        + 28.8 * milk_fat_pct
        + 0.148 * body_weight
    )
    return methane_g_per_day

def ecm(milk_kg, fat_pct, protein_pct):
    # Returns ECM in kg (DRMS, 2014)
    return milk_kg * (0.327 + 12.95 * (fat_pct / 100) + 7.2 * (protein_pct / 100))

def methane_intensity(methane_g_per_day, ecm_kg):
    # Returns methane intensity in g/kg ECM
    if ecm_kg <= 0:
        return 0.0
    return methane_g_per_day / (ecm_kg * 1000)  # Convert to kg/kg

def methane_kg_per_day_add(methane_intensity, methane_intensity_reduction, ecm_kg):
    """Calculate methane production with additive."""
    return methane_intensity * (1 - methane_intensity_reduction) * ecm_kg

def monthly_econ_emis_summary(with_additive=False):
    npv_over_time_additive = []
    npv_over_time_noadditive = []
    methane_add_over_time = []
    methane_noadd_over_time = []
    discount_factor_add = 1.0
    discount_factor_noadd = 1.0

    for month in range(n_months):
        herd_state = herd_evolution[month]
        # --- With additive ---
        monthly_methane_add = 0.0
        milk_income_add = 0.0
        feed_cost_add = 0.0
        repro_total_cost_add = 0.0
        cull_cost_add = 0.0
        replacement_total_cost_add = 0.0
        calf_income_add = 0.0
        additive_cost = 0.0

        # --- Without additive ---
        monthly_methane_noadd = 0.0
        milk_income_noadd = 0.0
        feed_cost_noadd = 0.0
        repro_total_cost_noadd = 0.0
        cull_cost_noadd = 0.0
        replacement_total_cost_noadd = 0.0
        calf_income_noadd = 0.0

        for idx, (par, mim, mip) in enumerate(states):
            n_cows = herd_state[idx]
            if n_cows == 0:
                continue

            # --- Without additive ---
            milk_noadd = milk_yield(par, mim) # kg/day
            methane_g_per_day_noadd = methane_production(
                    feed_intake, ndf_pct, milk_fat_pct, body_weight
                )
            ecm_kg_noadd = ecm(milk_noadd, milk_fat_pct, milk_protein_pct)
            methane_intensity_noadd = methane_intensity(methane_g_per_day_noadd, ecm_kg_noadd)
            monthly_methane_noadd += methane_g_per_day_noadd * 30 * n_cows / 1000 # kg/month

            milk_income_noadd += n_cows * milk_noadd * 30 * milk_price
            feed_cost_noadd += n_cows * feed_intake * 30 * feed_price
            if mip == 0 and mim <= Last_MIM_to_Breed:
                repro_total_cost_noadd += n_cows * repro_cost
            if mip == MAX_MIP:
                calf_income_noadd += n_cows * calf_value

            if with_additive:
                # --- With additive ---
                feed_intake_add = feed_intake * (1 - feed_reduction)
                milk_add = milk_yield(par, mim) * (1 - milk_reduction)
                # ECM (kg/day)
                ecm_kg_add = ecm(milk_add, milk_fat_pct, milk_protein_pct)
                # Methane with additive reduction (kg/day)
                methane_kg_per_day_add_final = methane_kg_per_day_add(
                    methane_intensity_noadd, methane_intensity_reduction, ecm_kg_add
                )
                monthly_methane_add += methane_kg_per_day_add_final * 30 * n_cows

                milk_income_add += n_cows * milk_add * 30 * milk_price
                feed_cost_add += n_cows * feed_intake_add * 30 * feed_price
                # Additive cost
                additive_g_per_cow_per_month = (additive_dose_mg_per_kg_feed / 1000) * feed_intake_add * 30 / 1000
                additive_cost += n_cows * additive_g_per_cow_per_month * additive_cost_per_g
                if mip == 0 and mim <= Last_MIM_to_Breed:
                    repro_total_cost_add += n_cows * repro_cost
                if mip == MAX_MIP:
                    calf_income_add += n_cows * calf_value

        # Store monthly total methane production
        methane_add_over_time.append(monthly_methane_add)
        methane_noadd_over_time.append(monthly_methane_noadd)
        
        # Culling and replacement (estimate from herd loss)
        if month > 0:
            prev_herd = herd_evolution[month-1].sum()
            curr_herd = herd_state.sum()
            n_culled = max(prev_herd - curr_herd, 0)
            if with_additive:
                cull_cost_add += n_culled * (-salvage_value_per_kg * body_weight)
                replacement_total_cost_add += n_culled * replacement_cost
            cull_cost_noadd += n_culled * (-salvage_value_per_kg * body_weight)
            replacement_total_cost_noadd += n_culled * replacement_cost

        # --- Net cash flow with additive ---
        if with_additive:
            net_cash_flow_add = (
                milk_income_add
                - feed_cost_add
                - repro_total_cost_add
                - additive_cost
                + calf_income_add
                + cull_cost_add
                - replacement_total_cost_add
            )
            monthly_npv_add = net_cash_flow_add * discount_factor_add
            npv_over_time_additive.append(monthly_npv_add)
            discount_factor_add *= monthly_discount

        # --- Net cash flow without additive ---
        net_cash_flow_noadd = (
            milk_income_noadd
            - feed_cost_noadd
            - repro_total_cost_noadd
            + calf_income_noadd
            + cull_cost_noadd
            - replacement_total_cost_noadd
        )
        monthly_npv_noadd = net_cash_flow_noadd * discount_factor_noadd
        npv_over_time_noadditive.append(monthly_npv_noadd)
        discount_factor_noadd *= monthly_discount

    return {
        "npv_over_time_additive": npv_over_time_additive,
        "npv_over_time_noadditive": npv_over_time_noadditive,
        "methane_add_over_time": methane_add_over_time,
        "methane_noadd_over_time": methane_noadd_over_time
    }

npv_over_time_additive = monthly_econ_emis_summary(with_additive=True)["npv_over_time_additive"]
npv_over_time_noadditive = monthly_econ_emis_summary(with_additive=False)["npv_over_time_noadditive"]

# Plot both on the same graph
plt.figure()
plt.plot(npv_over_time_additive, label="With Additive")
plt.plot(npv_over_time_noadditive, label="Without Additive")
plt.xlabel('Month')
plt.ylabel('Discounted Net Present Value ($)')
plt.title('Monthly Discounted Net Present Value (NPV; $): With vs Without Additive')
plt.legend()
plt.tight_layout()
plt.show()

avg_monthly_npv_add = np.mean(npv_over_time_additive)
avg_monthly_npv_noadd = np.mean(npv_over_time_noadditive)

print(f"Average monthly discounted NPV (with additive): ${avg_monthly_npv_add:,.2f}")
print(f"Average monthly discounted NPV (without additive): ${avg_monthly_npv_noadd:,.2f}")