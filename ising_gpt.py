#!/usr/bin/env python3
"""Exact diagonalization of Roychowdhury--Wang's Ising fusion category chain.

Implements Eqs. (7)--(8), (12)--(15), and (18)--(21) of the supplied
September 9, 2026 manuscript, with kappa=+1 and periodic boundary conditions.
Dependencies: Python >=3.10, NumPy, SciPy. No other packages are required.

Conventions
-----------
The stored tuple contains the horizontal fusion labels x_i, not the a_i.
I=0, PSI=1, SIGMA=2. T is RIGHT translation, i.e. numpy.roll(state,+1).
For k=2*pi*m/L and a translation orbit of length R, our normalized ket is

    |a;k> = R**(-1/2) sum_{t=0}^{R-1} exp(+i*k*t) T**t |a>.

Thus T|a;k> = exp(-i*k)|a;k>. If b=T**s|target> is the representative,
the reduced matrix element carries exp(+i*k*s)*sqrt(R_a/R_b). These
signs and normalization are consistent; neither is a correction to the
original posted matrix formula. Changing the translation eigenvalue convention
to exp(+i*k) requires changing BOTH phase signs and relabels m <-> -m.

The real correction to momentum selection is exact integer arithmetic:
    (m * R) % L == 0.
The original local Hamiltonian amplitudes and global minus sign are retained.

This is sparse ED with an explicitly enumerated constrained basis. Its Hilbert
space still grows exponentially, D_L=1+(1+sqrt(2))**L+(1-sqrt(2))**L;
this implementation makes no claim of practical L=20 performance. CFT scaling
and spontaneous degeneracies refer to finite-size extrapolation, not exact
degeneracy of generic short rings.

Examples
--------
python ising_fusion_ed.py --L 6 --r 0 --theta 0.7853981633974483 --nev 8 --validate
python ising_fusion_ed.py --L 6 --r 20 --nev 8 --output spectrum.json
python ising_fusion_ed.py --L 6 --r -20 --nev 8
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Iterator

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, eye, issparse
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt # added

I, PSI, SIGMA = 0, 1, 2
State = tuple[int, ...]


def _integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not {value!r}")
    return int(value)


def _parameters(r: float, theta: float) -> tuple[float, float]:
    for name, value in (("r", r), ("theta", theta)):
        if not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite real number")
    return float(r), float(theta)


def hilbert_dimension(L: int) -> int:
    """Exact trace(A**L), without cancellation or floating-point powers.

    A=[[1,0,1],[0,1,1],[1,1,1]]. For S_L=(1+sqrt(2))**L+
    (1-sqrt(2))**L, S_0=S_1=2 and S_L=2*S_{L-1}+S_{L-2}.
    """
    L = _integer(L, "L")
    if L < 0:
        raise ValueError("L must be nonnegative")
    if L == 0:
        return 3
    previous, current = 2, 2
    for _ in range(2, L + 1):
        previous, current = current, 2 * current + previous
    return 1 + current


def is_valid_state(state) -> bool:
    """Exactly the paper's periodic x-label basis, including the closing bond.

    Same-grade neighbours require a_i=I and therefore equal x labels. Different
    grades uniquely require a_i=SIGMA. SIGMA--SIGMA uses a_i=I as required by
    Eq. (8); its alternative a_i=PSI is deliberately excluded. Hence only
    adjacent I--PSI and PSI--I are forbidden. No fixed total-charge projection
    or extra fusion-tree multiplicity is imposed in this periodic Hilbert space.
    """
    if len(state) == 0:
        return False
    if any(x not in (I, PSI, SIGMA) for x in state):
        return False
    return all(not ((a == I and b == PSI) or (a == PSI and b == I))
               for a, b in zip(state, tuple(state[1:]) + (state[0],)))


def translate(state: State, steps: int = 1) -> State:
    """Right translation T**steps, identical to tuple(np.roll(state,steps))."""
    steps = _integer(steps, "steps") % len(state)
    return state[-steps:] + state[:-steps] if steps else state


def find_representative_and_shift(state) -> tuple[State, int]:
    """Return lexicographic representative and RIGHT shifts needed to reach it."""
    state = tuple(state)
    if not state:
        raise ValueError("state must not be empty")
    orbit = [translate(state, s) for s in range(len(state))]
    representative = min(orbit)
    return representative, orbit.index(representative)


@dataclass(frozen=True)
class Orbit:
    rep: State
    period: int

    @property
    def R(self) -> int:
        return self.period


class Basis:
    """Enumerate valid states once, with all translation-orbit lookups cached.

    max_states is a preallocation guard, not a memory/performance guarantee.
    Set it to None only after estimating your available resources.
    """

    def __init__(self, L: int, max_states: int | None = 2_000_000):
        self.L = _integer(L, "L")
        if self.L < 3:
            raise ValueError("L must be >=3 so each three-site term has distinct sites")
        dimension = hilbert_dimension(self.L)
        if max_states is not None:
            max_states = _integer(max_states, "max_states")
            if max_states < 1:
                raise ValueError("max_states must be positive or None")
            if dimension > max_states:
                raise MemoryError(
                    f"L={L} has {dimension:,} valid states, exceeding max_states="
                    f"{max_states:,}. Explicit ED is exponential; increase the guard "
                    "only with sufficient memory, or use a more scalable method."
                )

        # Adjacency pruning avoids enumerating all 3**L ternary strings.
        allowed = {I: (I, SIGMA), PSI: (PSI, SIGMA), SIGMA: (I, PSI, SIGMA)}
        states: list[State] = []

        def extend(prefix: list[int]) -> None:
            if len(prefix) == self.L:
                if prefix[0] in allowed[prefix[-1]]:  # periodic closing bond
                    states.append(tuple(prefix))
                return
            for label in allowed[prefix[-1]]:
                prefix.append(label)
                extend(prefix)
                prefix.pop()

        for first in (I, PSI, SIGMA):
            extend([first])
        if len(states) != dimension:
            raise RuntimeError("Basis enumeration failed the exact dimension identity")
        self.states = tuple(states)
        self.state_to_index = {state: i for i, state in enumerate(self.states)}
        self.lookup: dict[State, tuple[int, int]] = {}
        orbits: list[Orbit] = []
        # Enumeration is lexicographic, so the first unseen member is the rep.
        for representative in self.states:
            if representative in self.lookup:
                continue
            orbit_states = [representative]
            shifted = translate(representative)
            while shifted != representative:
                orbit_states.append(shifted)
                shifted = translate(shifted)
            period = len(orbit_states)
            orbit_index = len(orbits)
            orbits.append(Orbit(representative, period))
            for t, member in enumerate(orbit_states):
                self.lookup[member] = (orbit_index, (-t) % period)
        self.orbits = tuple(orbits)
        self._sector_cache: dict[int, tuple[Orbit, ...]] = {}

    def __len__(self) -> int:
        return len(self.states)

    def momentum(self, m: int) -> int:
        return _integer(m, "m") % self.L

    def sector(self, m: int) -> tuple[Orbit, ...]:
        m = self.momentum(m)
        if m not in self._sector_cache:
            # Numerical fix: NEVER test a floating-point modulus against zero.
            self._sector_cache[m] = tuple(
                orbit for orbit in self.orbits if (m * orbit.period) % self.L == 0
            )
        return self._sector_cache[m]


def generate_representatives(L: int, m: int, *, basis: Basis | None = None):
    """Convenience replacement; second argument is INTEGER m, not floating k.

    Supply an existing basis to avoid repeated enumeration across momenta.
    """
    basis = Basis(L) if basis is None else basis
    if basis.L != L:
        raise ValueError("L and basis.L disagree")
    return [{"rep": orbit.rep, "R": orbit.period} for orbit in basis.sector(m)]


def _local_action_valid(state: State, r: float, c: float, s: float):
    """Apply H=-sum_i(H_dw+H_flip) to a known valid state.

    Duplicate outputs are intentional; sparse COO assembly sums them exactly
    as distinct local terms in the Hamiltonian. No post-hoc symmetrization.
    """
    L = len(state)
    diagonal = 0.0
    for j in range(L):
        left, mid, right = state[(j - 1) % L], state[j], state[(j + 1) % L]
        lm, mm, rm = left != SIGMA, mid != SIGMA, right != SIGMA
        # Eq. (14): the four nonzero cases have opposite endpoint grades.
        # This is a next-nearest-grade interaction, not a nearest-neighbour
        # domain-wall count. Its overall Hamiltonian contribution is -r.
        if lm != rm:
            diagonal -= r

        transitions: tuple[tuple[int, float], ...] = ()
        if lm and mm and rm:                    # |mu mu mu>
            transitions = ((SIGMA, c),)
        elif lm and mm and not rm:              # |mu mu sigma>
            transitions = ((SIGMA, s),)
        elif lm and not mm and rm:              # |mu sigma nu>
            if left == right:
                transitions = ((left, c),)       # Kronecker delta_mu,nu
        elif not lm and mm and rm:              # |sigma mu mu>
            transitions = ((SIGMA, s),)
        elif lm and not mm and not rm:          # |mu sigma sigma>
            transitions = ((left, s),)
        elif not lm and mm and not rm:          # |sigma mu sigma>
            transitions = ((SIGMA, c / math.sqrt(2.0)),)
        elif not lm and not mm and rm:          # |sigma sigma mu>
            transitions = ((right, s),)
        elif not lm and not mm and not rm:      # |sigma sigma sigma>
            amplitude = c / math.sqrt(2.0)
            transitions = ((I, amplitude), (PSI, amplitude))

        # Eq. (15) already contains the F-symbol and sqrt(2) factors; do not
        # multiply them by another F matrix, anyon dimension, or extra sign.
        for new_mid, amplitude in transitions:
            if amplitude != 0.0:
                target = state[:j] + (new_mid,) + state[j + 1:]
                yield target, -amplitude         # Eq. (12) minus sign
    if diagonal != 0.0:
        yield state, diagonal


def local_action(state, r: float, theta: float) -> Iterator[tuple[State, float]]:
    """Yield (target_state, coefficient) for H|state>, including diagonal terms."""
    state = tuple(state)
    if len(state) < 3 or not is_valid_state(state):
        raise ValueError("state must be a valid periodic fusion-label state of L>=3")
    r, theta = _parameters(r, theta)
    yield from _local_action_valid(state, r, math.cos(theta), math.sin(theta))


def build_hamiltonian_block(basis: Basis, m: int, r: float, theta: float) -> csr_matrix:
    """Momentum block with columns as input kets and rows as output kets."""
    m = basis.momentum(m)
    r, theta = _parameters(r, theta)
    sector = basis.sector(m)
    indices = {orbit.rep: i for i, orbit in enumerate(sector)}
    k = 2.0 * math.pi * m / basis.L
    c, s = math.cos(theta), math.sin(theta)
    rows, columns, data = [], [], []
    for col, source_orbit in enumerate(sector):
        for target, coefficient in _local_action_valid(source_orbit.rep, r, c, s):
            orbit_index, shift = basis.lookup[target]
            target_orbit = basis.orbits[orbit_index]
            row = indices.get(target_orbit.rep)
            if row is None:
                # This entire orbit has zero projector in this m sector.
                continue
            factor = math.sqrt(source_orbit.period / target_orbit.period)
            phase = np.exp(1j * k * shift)
            rows.append(row)
            columns.append(col)
            data.append(coefficient * factor * phase)
    result = coo_matrix((data, (rows, columns)), shape=(len(sector), len(sector)),
                        dtype=np.complex128).tocsr()
    result.eliminate_zeros()
    return result


def full_hamiltonian(basis: Basis, r: float, theta: float) -> csr_matrix:
    """Unprojected real Hamiltonian, useful for independent projection checks."""
    r, theta = _parameters(r, theta)
    c, s = math.cos(theta), math.sin(theta)
    rows, columns, data = [], [], []
    for col, state in enumerate(basis.states):
        for target, coefficient in _local_action_valid(state, r, c, s):
            rows.append(basis.state_to_index[target])
            columns.append(col)
            data.append(coefficient)
    result = coo_matrix((data, (rows, columns)), shape=(len(basis), len(basis)),
                        dtype=np.float64).tocsr()
    result.eliminate_zeros()
    return result


def explicit_projector(basis: Basis, m: int) -> csr_matrix:
    """Columns are normalized |a;k> in the full basis; for small-system checks.

    Checks Q^dagger H Q by an explicit orbit sum rather than reusing the
    representative matrix-element formula. Q is an isometry, not square.
    """
    m = basis.momentum(m)
    k = 2.0 * math.pi * m / basis.L
    sector = basis.sector(m)
    rows, columns, data = [], [], []
    for col, orbit in enumerate(sector):
        for t in range(orbit.period):
            rows.append(basis.state_to_index[translate(orbit.rep, t)])
            columns.append(col)
            data.append(np.exp(1j * k * t) / math.sqrt(orbit.period))
    return coo_matrix((data, (rows, columns)), shape=(len(basis), len(sector)),
                      dtype=np.complex128).tocsr()


def translation_operator(basis: Basis) -> csr_matrix:
    rows = [basis.state_to_index[translate(state)] for state in basis.states]
    return coo_matrix((np.ones(len(basis)), (rows, np.arange(len(basis)))),
                      shape=(len(basis), len(basis))).tocsr()


def symmetry_action(state: State, label: str) -> Iterator[tuple[State, float]]:
    """Full-basis action of U(psi) or U(sigma), Eqs. (18)--(21).

    Eq. (20) uses I=0, PSI=1 in its exponent. Each old sigma domain becomes
    a new uniform mu' domain. Its sign is (-1)**((mu_left+mu_right)*mu').
    """
    state = tuple(state)
    if len(state) < 3 or not is_valid_state(state):
        raise ValueError("symmetry_action requires a valid state of L>=3")
    if label == "psi":
        yield tuple(1 - x if x != SIGMA else SIGMA for x in state), 1.0
        return
    if label != "sigma":
        raise ValueError("label must be 'psi' or 'sigma'")
    L = len(state)
    if len(set(state)) == 1:                     # Eq. (21), special n=0 case
        if state[0] == SIGMA:
            yield (I,) * L, 1.0
            yield (PSI,) * L, 1.0
        else:
            yield (SIGMA,) * L, 1.0
        return
    domains = []
    for j in range(L):
        if state[j] == SIGMA and state[(j - 1) % L] != SIGMA:
            sites = []
            end = j
            while state[end] == SIGMA:
                sites.append(end)
                end = (end + 1) % L
            domains.append((sites, state[(j - 1) % L] + state[end]))
    amplitude = 2.0 ** (-len(domains) / 2.0)
    for values in product((I, PSI), repeat=len(domains)):
        target = [SIGMA] * L
        exponent = 0
        for value, (sites, boundary_sum) in zip(values, domains):
            exponent += boundary_sum * value
            for site in sites:
                target[site] = value
        yield tuple(target), amplitude * (-1 if exponent % 2 else 1)


def symmetry_operator(basis: Basis, label: str, max_nnz: int = 10_000_000) -> csr_matrix:
    """Explicit categorical symmetry for SMALL validation/charge calculations."""
    rows, columns, data = [], [], []
    for col, state in enumerate(basis.states):
        for target, coefficient in symmetry_action(state, label):
            if len(data) >= max_nnz:
                raise MemoryError("Explicit symmetry matrix exceeded max_nnz guard")
            rows.append(basis.state_to_index[target])
            columns.append(col)
            data.append(coefficient)
    return coo_matrix((data, (rows, columns)), shape=(len(basis), len(basis)),
                      dtype=np.float64).tocsr()


def _max_abs(matrix) -> float:
    if issparse(matrix):
        matrix = matrix.tocsr()
        return float(np.max(np.abs(matrix.data))) if matrix.nnz else 0.0
    array = np.asarray(matrix)
    return float(np.max(np.abs(array))) if array.size else 0.0


def lowest_eigenpairs(H, nev: int = 5, *, return_residuals: bool = False,
                      tol: float = 1e-11, maxiter: int | None = None,
                      seed: int = 1729, dense_cutoff: int = 128,
                      max_dense_dimension: int = 4096):
    """Return SORTED lowest (eigenvalues,eigenvectors[,residual_norms]).

    Complex scipy eigsh delegates to eigs, which cannot use k>=N-1 on sparse
    matrices. We explicitly fall back to dense eigh in that range. A generic
    deterministic random start avoids confining Lanczos to the symmetry sector
    of an all-ones seed. Near/exact degeneracies should still be verified with
    a dense calculation for small N or resolved symmetry sectors at large N.
    """
    nev = _integer(nev, "nev")
    if nev < 1:
        raise ValueError("nev must be positive")
    dense_cutoff = _integer(dense_cutoff, "dense_cutoff")
    if dense_cutoff < 0:
        raise ValueError("dense_cutoff must be nonnegative")
    max_dense_dimension = _integer(max_dense_dimension, "max_dense_dimension")
    if max_dense_dimension < 1:
        raise ValueError("max_dense_dimension must be positive")
    if not isinstance(tol, Real) or not math.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")
    if maxiter is not None and _integer(maxiter, "maxiter") < 1:
        raise ValueError("maxiter must be positive or None")
    seed = _integer(seed, "seed")
    H = csr_matrix(H)
    if H.shape[0] != H.shape[1]:
        raise ValueError("H must be square")
    if np.any(~np.isfinite(H.data)):
        raise ValueError("H contains nonfinite matrix elements")
    error = _max_abs(H - H.getH())
    if error > 1e-10 * max(1.0, _max_abs(H)):
        raise ValueError(f"H is not Hermitian (maximum discrepancy {error:.3g})")
    n = H.shape[0]
    count = min(nev, n)
    if n == 0:
        evals = np.empty(0)
        evecs = np.empty((0, 0), dtype=H.dtype)
    elif n <= dense_cutoff or count >= n - 1:
        # Prevent an unexpectedly large dense allocation when nev is large.
        # At the default bound one complex dense array already occupies 256 MiB,
        # and the eigensolver needs additional arrays/workspace.
        if n > max_dense_dimension:
            raise MemoryError(
                f"Dense diagonalization of dimension {n} exceeds the guard "
                f"{max_dense_dimension}; request fewer eigenvalues or explicitly "
                "raise max_dense_dimension with sufficient memory."
            )
        evals, evecs = np.linalg.eigh(H.toarray())
        evals, evecs = evals[:count], evecs[:, :count]
    else:
        rng = np.random.default_rng(seed)
        v0 = rng.normal(size=n)
        if np.iscomplexobj(H.data):
            v0 = v0 + 1j * rng.normal(size=n)
        v0 /= np.linalg.norm(v0)
        # Allow adequate Krylov space; failure propagates instead of silently
        # returning incomplete unconverged Ritz values.
        ncv = min(n, max(2 * count + 1, 32))
        evals, evecs = eigsh(H, k=count, which="SA", v0=v0, tol=tol,
                             maxiter=maxiter, ncv=ncv)
        order = np.argsort(evals)
        evals, evecs = evals[order], evecs[:, order]
    if return_residuals:
        residuals = np.linalg.norm(H @ evecs - evecs * evals[None, :], axis=0)
        return evals, evecs, residuals
    return evals, evecs


def solve_spectrum(basis: Basis | int, r: float, theta: float,
                   nev: int = 5) -> dict[int, np.ndarray]:
    """Return {integer_m: sorted_eigenvalues}; k=2*pi*m/basis.L.

    Integer keys avoid floating-point key ambiguities. Pass a Basis object to
    reuse enumeration and orbit data for a parameter scan.
    """
    if isinstance(basis, Integral):
        basis = Basis(basis)
    result = {}
    for m in range(basis.L):
        block = build_hamiltonian_block(basis, m, r, theta)
        result[m] = lowest_eigenpairs(block, nev=nev)[0]
    return result


def validate_model(basis: Basis, r: float, theta: float,
                   tolerance: float = 1e-9, max_dimension: int = 2000) -> dict:
    """Small-system structural validation, including categorical symmetry.

    This checks the reduced-orbit formula independently via explicit projection
    and verifies the complete union of all block spectra against full ED. It
    does not by itself establish thermodynamic CFT convergence.
    """
    if len(basis) > max_dimension:
        raise ValueError("Validation uses full dense ED; use L<=8 for this check")
    H = full_hamiltonian(basis, r, theta)
    T = translation_operator(basis)
    ident = eye(len(basis), format="csr")
    upsi = symmetry_operator(basis, "psi")
    usigma = symmetry_operator(basis, "sigma")
    errors = {
        "H_hermiticity": _max_abs(H - H.getH()),
        "translation_commutator": _max_abs(T @ H - H @ T),
        "Upsi_hermiticity": _max_abs(upsi - upsi.getH()),
        "Usigma_hermiticity": _max_abs(usigma - usigma.getH()),
        "Upsi_squared_minus_I": _max_abs(upsi @ upsi - ident),
        "Upsi_Usigma_minus_Usigma": _max_abs(upsi @ usigma - usigma),
        "Usigma_Upsi_minus_Usigma": _max_abs(usigma @ upsi - usigma),
        "Usigma_squared_minus_I_minus_Upsi": _max_abs(usigma @ usigma - ident - upsi),
        "Upsi_H_commutator": _max_abs(upsi @ H - H @ upsi),
        "Usigma_H_commutator": _max_abs(usigma @ H - H @ usigma),
    }
    all_evals = []
    dimensions = []
    for m in range(basis.L):
        Q = explicit_projector(basis, m)
        block = build_hamiltonian_block(basis, m, r, theta)
        dimensions.append(block.shape[0])
        errors[f"m{m}_isometry"] = _max_abs(Q.getH() @ Q - eye(Q.shape[1]))
        errors[f"m{m}_projected_H"] = _max_abs(Q.getH() @ H @ Q - block)
        phase = np.exp(-2j * math.pi * m / basis.L)
        errors[f"m{m}_translation_eigenvalue"] = _max_abs(T @ Q - phase * Q)
        errors[f"m{m}_H_hermiticity"] = _max_abs(block - block.getH())
        all_evals.extend(np.linalg.eigvalsh(block.toarray()))
    if sum(dimensions) != len(basis):
        raise AssertionError("Sum of momentum dimensions is not the full dimension")
    errors["full_vs_momentum_spectrum"] = _max_abs(
        np.sort(all_evals) - np.linalg.eigvalsh(H.toarray())
    )
    failed = {key: value for key, value in errors.items() if value > tolerance}
    if failed:
        raise AssertionError(f"Validation failed (absolute tolerance {tolerance}): {failed}")
    return {"passed": True, "dimension": len(basis), "sector_dimensions": dimensions,
            "max_error": max(errors.values()), "errors": errors}


def run_simulation(L=12, r=0, theta=math.pi/4, nev=12):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--L", type=int, default=L)
    parser.add_argument("--r", type=float, default=r)
    parser.add_argument("--theta", type=float, default=theta,
                        help="angle in radians (default pi/4)")
    parser.add_argument("--nev", type=int, default=nev)
    parser.add_argument("--max-states", type=int, default=50_000_000)
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    parser.add_argument("--validate", action="store_true", help="full small-system checks, L<=8")
    args = parser.parse_args()
    r, theta = _parameters(args.r, args.theta)
    basis = Basis(args.L, max_states=args.max_states)
    result = {"L": basis.L, "r": r, "theta": theta, "dimension": len(basis),
              "translation": "right", "translation_eigenvalue": "exp(-i*k)",
              "momentum_formula": "k=2*pi*m/L", "sectors": []}
    print(f"L={basis.L}, dimension={len(basis)}, r={r:.12g}, theta={theta:.12g}")
    print("T is right translation; T eigenvalue exp(-ik), k=2*pi*m/L")

    # store data
    energy = []

    for m in range(basis.L):
        block = build_hamiltonian_block(basis, m, r, theta)
        evals, _, residuals = lowest_eigenpairs(block, nev=args.nev, return_residuals=True)
        result["sectors"].append({"m": m, "k": 2 * math.pi * m / basis.L,
                                  "dimension": block.shape[0], "eigenvalues": evals.tolist(),
                                  "residual_norms": residuals.tolist()})
        printed = " ".join(f"{energy:.12f}" for energy in evals)
        print(f"m={m:2d}  dim={block.shape[0]:7d}  E: {printed}")
        energy.append(evals)
    if args.validate:
        result["validation"] = validate_model(basis, r, theta)
        print(f"Validation passed; maximum absolute discrepancy "
              f"{result['validation']['max_error']:.3e}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")

    # plot the results
    plt.figure(figsize=(5, 7))

    # the axes
    energy = np.array(energy)
    ks = np.arange(L)

    for idx, k in enumerate(ks):
        # Scatter plot all eigenvalues for each k sector
        plt.scatter([k] * len(energy[idx]), energy[idx] - np.min(energy), color='blue', zorder=3, marker='x') # reset reference level

    # plt.ylim([-0.001, 0.13]) # r = -20, theta = pi/4
    # plt.ylim([-0.000001, 0.000032]) # r = 20, theta = pi/4/
    # plt.ylim([-0.01, 2.5]) # r = 0, theta = pi/4
    plt.xlabel(r'Momentum $k$')
    plt.ylabel('Energy $E$')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    

run_simulation(L=3, r=20.0, theta=np.pi/4, nev=4)