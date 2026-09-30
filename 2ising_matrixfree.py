import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigsh
import time


def get_site(state, i):
    """
    Extracts the 2-bit state at site i.
    Uses bitwise AND with 3 (binary 0b11) to isolate the 2 bits.
    """
    return (state >> (2 * i)) & 3

def see_state(state, L):
    """
    Prints a string that shows the state in braket form.
    """
    state_string = ""
    for i in range(L):
        state_string += str(get_site(state, i))
    return state_string

def check_valid_state(state, L):
    for i in range(L):
        if i == L-1:
            # Check the boundary condition
            value_at_i = get_site(state, 0)
            value_at_j = get_site(state, L-1) # PBC
            if (value_at_i == 3) or (value_at_j == 3):
                return False
            elif (value_at_i == 0) and (value_at_j == 1):
                return False
            elif (value_at_i == 1) and (value_at_j == 0):
                return False
        else:
            value_at_i = get_site(state, i)
            value_at_j = get_site(state, i+1) # j=i+1
            if (value_at_i == 3):
                return False
            elif (value_at_i == 0) and (value_at_j == 1):
                return False
            elif (value_at_i == 1) and (value_at_j == 0):
                return False
    return True

def set_site(state, i, val):
    """
    Updates the 2-bit state at site i to 'val' (0, 1, or 2).
    First clears the existing 2 bits at site i, then inserts the new value.
    """
    clear_mask = ~(3 << (2 * i))
    return (state & clear_mask) | (val << (2 * i))

def global_flip(state, L):
    """
    Flips fermions and keep anyon unchanged.
    """
    for i in range(L):
        # sigma -> unflipped
        if get_site(state, i) == 0:
            state = set_site(state, i, 1) # 1 -> psi
        elif get_site(state, i) == 1:
            state = set_site(state, i, 0) # psi -> 1
    return state

def translate_right(state, L):
    """
    Circularly shifts the lattice right by 1 site (site i -> site i+1).
    Site L-1 wraps around to become Site 0.
    """
    boundary_site = (state >> (2 * (L - 1))) & 3
    state = (state << 2) | boundary_site
    lattice_mask = (1 << (2 * L)) - 1
    return state & lattice_mask

def translate_left(state, L):
    """
    Circularly shifts the lattice left by 1 site (site i -> site i-1).
    Site 0 wraps around to become Site L-1.
    """
    boundary_site = state & 3
    state = (state >> 2) | (boundary_site << (2 * (L - 1)))
    lattice_mask = (1 << (2 * L)) - 1
    return state & lattice_mask

def build_basis_and_lookup(valid_states, L):
    """
    Creates a lookup table for all valid states and identifies unique representatives.
    
    Returns:
    - lookup_table: A dictionary where keys are raw states and values are 
                    tuples (rep, m, p) such that applying translate_right m times 
                    and global_flip p times to the raw state yields the rep.
    - representatives: A sorted list of the unique canonical representatives.
    """
    lookup_table = {}
    representatives = []
    
    for state in valid_states:
        if state in lookup_table:
            continue
        
        # Find rep
        orbit_min = state
        current_search = state
        for m in range(L): # momentum
            if current_search < orbit_min: # translated
                orbit_min = current_search
            
            flipped_search = global_flip(current_search, L)
            if flipped_search < orbit_min: # flipped
                orbit_min = flipped_search

            current_search = translate_right(current_search, L)
            
        rep = orbit_min
        representatives.append(rep)
        
        # Look-up table
        current_raw = rep
        for m in range(L):
            m_to_raw = m % L
            if current_raw not in lookup_table:
                lookup_table[current_raw] = (rep, m_to_raw, 0) # Unflipped
                
            flipped_raw = global_flip(current_raw, L)
            if flipped_raw not in lookup_table:
                lookup_table[flipped_raw] = (rep, m_to_raw, 1) # Flipped
                
            current_raw = translate_right(current_raw, L)

    representatives.sort()

    return lookup_table, representatives

def get_rep_to_index(representatives):
    return {rep: idx for idx, rep in enumerate(representatives)}

def get_orbit_size(representatives):
    """
    Returns a dictionary in which key is matrix index
    and value is the number of distinct states in the orbit
    """
    orbit_sizes = {}
    for idx, rep in enumerate(representatives):
        distinct_states = {rep}

        state_temp = rep
        for m in range(L):
            state_temp = translate_right(state_temp, L) # T only
            if state_temp not in distinct_states:
                distinct_states.add(state_temp)

        state_temp = global_flip(rep, L)
        for m in range(L):
            state_temp = translate_right(state_temp, L) # UT
            if state_temp not in distinct_states:
                distinct_states.add(state_temp)

        orbit_sizes[idx] = len(distinct_states)
        # print([see_state(state, L) for state in list(distinct_states)]) # Show the orbits
    return orbit_sizes

def apply_local_H(state, i, L, r, theta):
    """
    Applies H_i = H^{dw}_i + H^{flip}_i to the state at site i.
    Returns a list of tuples: (generated_raw_state, coefficient)
    """
    # Use modulo L to enforce periodic boundary conditions
    left_val = get_site(state, (i - 1) % L)
    mid_val = get_site(state, i)
    right_val = get_site(state, (i + 1) % L)
    
    generated_terms = []
    
    # -------------------------------------------------------------
    # 1. H^{dw} (Diagonal terms - state does not change)[cite: 2]
    # -------------------------------------------------------------
    diag_coeff = 0.0
    if left_val < 2 and mid_val < 2 and right_val == 2:     # |mu mu sigma>[cite: 2]
        diag_coeff = r
    elif left_val == 2 and mid_val < 2 and right_val < 2:   # |sigma mu mu>[cite: 2]
        diag_coeff = r
    elif left_val < 2 and mid_val == 2 and right_val == 2:  # |mu sigma sigma>[cite: 2]
        diag_coeff = r
    elif left_val == 2 and mid_val == 2 and right_val < 2:  # |sigma sigma mu>[cite: 2]
        diag_coeff = r
    # All other configurations (|mu mu mu>, |mu sigma nu>, |sigma mu sigma>, |sigma sigma sigma>) yield 0[cite: 2]
    
    if diag_coeff != 0:
        generated_terms.append((state, diag_coeff))

    # -------------------------------------------------------------
    # 2. H^{flip} (Off-diagonal terms - middle site changes)[cite: 3]
    # -------------------------------------------------------------
    if left_val < 2 and mid_val < 2 and right_val < 2:      # |mu mu mu> -> |mu sigma mu>[cite: 3]
        new_state = set_site(state, i, 2)
        generated_terms.append((new_state, np.cos(theta)))
        
    elif left_val < 2 and mid_val < 2 and right_val == 2:   # |mu mu sigma> -> |mu sigma sigma>[cite: 3]
        new_state = set_site(state, i, 2)
        generated_terms.append((new_state, np.sin(theta)))
        
    elif left_val < 2 and mid_val == 2 and right_val < 2:   # |mu sigma nu>[cite: 3]
        if left_val == right_val: # delta_{mu, nu} check[cite: 3]
            new_state = set_site(state, i, left_val)
            generated_terms.append((new_state, np.cos(theta)))
            
    elif left_val == 2 and mid_val < 2 and right_val < 2:   # |sigma mu mu> -> |sigma sigma mu>[cite: 3]
        new_state = set_site(state, i, 2)
        generated_terms.append((new_state, np.sin(theta)))
        
    elif left_val < 2 and mid_val == 2 and right_val == 2:  # |mu sigma sigma> -> |mu mu sigma>[cite: 3]
        new_state = set_site(state, i, left_val)
        generated_terms.append((new_state, np.sin(theta)))
        
    elif left_val == 2 and mid_val < 2 and right_val == 2:  # |sigma mu sigma> -> |sigma sigma sigma>[cite: 3]
        new_state = set_site(state, i, 2)
        generated_terms.append((new_state, np.cos(theta) / np.sqrt(2)))
        
    elif left_val == 2 and mid_val == 2 and right_val < 2:  # |sigma sigma mu> -> |sigma mu mu>[cite: 3]
        new_state = set_site(state, i, right_val)
        generated_terms.append((new_state, np.sin(theta)))
        
    elif left_val == 2 and mid_val == 2 and right_val == 2: # |sigma sigma sigma> -> |sigma 1 sigma> + |sigma psi sigma>[cite: 3]
        coeff = np.cos(theta) / np.sqrt(2)
        state_1 = set_site(state, i, 0) # 0 is 1 (I)
        state_psi = set_site(state, i, 1) # 1 is psi
        generated_terms.append((state_1, coeff))
        generated_terms.append((state_psi, coeff))
        
    return generated_terms

def filter_sector_basis(representatives, L, mom, charge):
    sector_reps = []
    for rep in representatives:
        phase_sum = 0.0
        for p in [0, 1]:
            # Apply flip p times
            test_state = rep if p == 0 else global_flip(rep, L)
            for m in range(L):
                if test_state == rep:
                    phase = np.exp(-1j * mom * m * 2 * np.pi / L) * (charge ** p)
                    phase_sum += phase
                test_state = translate_right(test_state, L)

        if np.abs(phase_sum) > 1e-8:
            sector_reps.append(rep)
            
    return sector_reps

def get_matvec(L, r, theta, sector_reps, lookup_table, rep_to_index, orbit_sizes, mom, charge):
    dim = len(sector_reps)
    
    def matvec(v):
        v_out = np.zeros_like(v, dtype=np.complex128)
        
        for a, rep_a in enumerate(sector_reps):
            if v[a] == 0:
                continue

            for i in range(L):
                generated_terms = apply_local_H(rep_a, i, L, r, theta)
                for raw_b, raw_coeff in generated_terms:
                    rep_b, m, p = lookup_table[raw_b]

                    if rep_b not in rep_to_index:
                        continue

                    b = rep_to_index[rep_b]
                    phase = np.exp(-1j * mom * m * 2 * np.pi / L) * (charge ** p)
                    weight = np.sqrt(orbit_sizes[a] / orbit_sizes[b]) * phase
                    v_out[b] += (-1) * raw_coeff * weight * v[a]
                    
        return v_out
        
    return LinearOperator((dim, dim), matvec=matvec, dtype=np.complex128)

L = 14
r = 20
theta = np.pi / 4
k = 5
spectrum = {}

# Generate valid states and group them into orbits
t0 = time.perf_counter()
valid_states = []
for state in range(4**L):
    if check_valid_state(state, L):
        valid_states.append(state)
lookup_table, representatives = build_basis_and_lookup(valid_states, L)
print("Finished generation of valid state and grouping into orbits.")
print("Time took: ", time.perf_counter() - t0)

# Create a fast lookup for the vector index of each representative
t0 = time.perf_counter()
rep_to_index = get_rep_to_index(representatives)
orbit_sizes = get_orbit_size(representatives)
print("Finished preparing look up dictionary for all representatives.")
print("Time took: ", time.perf_counter() - t0)

# Construct and diagonalize the block hamiltonians
total_time = 0.0
for mom in range(L):
    for charge in [1, -1]:
        t0 = time.perf_counter()

        # Performance check
        sector_reps = filter_sector_basis(representatives, L, mom, charge)
        dim = len(sector_reps) # Check if exists for L=3
        if dim == 0:
            continue
        
        sector_rep_to_index = {rep: idx for idx, rep in enumerate(sector_reps)}
        H_op = get_matvec(L, r, theta, sector_reps, lookup_table, sector_rep_to_index, orbit_sizes, mom, charge)

        if dim == 1:
            test_vec = np.array([1.0], dtype=np.complex128) # Check why 1.0 is used
            spectrum[(mom, charge)] = H_op.matvec(test_vec)[0].real
        else:
            if (np.shape(H_op)[0] -1) <= k: # Check if there is way around this, N-1 <= k
                continue
            eigenvalues, _ = eigsh(H_op, k=k, which='SA')
            spectrum[(mom, charge)] = eigenvalues

        t = time.perf_counter() - t0
        total_time += t

        print(f"mom={mom}, charge={charge} | Dim: {dim} | E_0: {spectrum[(mom, charge)]} | Time: {t:2f}s")

print("Total runtime = ", total_time, "s")