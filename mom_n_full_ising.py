#!/usr/bin/env python3
"""
Exact diagonalization of Ising fusion category chain.

Implements Eqs. (7)--(8), (12)--(15), and (18)--(21) of the September 9, 2026 manuscript,
with kappa=+1, periodic boundary conditions, and C_Ising symmetry block classification.
Dependencies: Python >=3.10, NumPy, SciPy, Matplotlib.

This is sparse ED block diagonalized a/c momentum sector, full C_ising symmetry. PBC is assumed.

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
import matplotlib
matplotlib.use('Agg') # Required for headless cluster execution
import matplotlib.pyplot as plt

I, PSI, SIGMA = 0, 1, 2
State = tuple[int, ...]
INV_SQRT2 = 1.0 / math.sqrt(2.0)


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
    """Exact trace(A**L), without cancellation or floating-point powers."""
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
    """Check periodic fusion label basis validity."""
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
    representative = state
    shift_to_representative = 0
    for shift in range(1, len(state)):
        candidate = state[-shift:] + state[:-shift]
        if candidate < representative:
            representative = candidate
            shift_to_representative = shift
    return representative, shift_to_representative


def _pack_state(state: State) -> int:
    """Pack labels into two-bit digits for compact dictionary keys."""
    packed = 0
    for label in state:
        packed = (packed << 2) | label
    return packed


@dataclass(frozen=True)
class Orbit:
    rep: State
    period: int

    @property
    def R(self) -> int:
        return self.period


class Basis:
    """Enumerate valid states once, with all translation-orbit lookups cached."""

    def __init__(self, L: int, max_states: int | None = 2_000_000, *,
                 retain_states: bool = True):
        self.L = _integer(L, "L")
        if self.L < 3:
            raise ValueError("L must be >=3 so each three-site term has distinct sites")
        dimension = hilbert_dimension(self.L)
        self._dimension = dimension
        if not isinstance(retain_states, bool):
            raise TypeError("retain_states must be a boolean")
        if max_states is not None:
            max_states = _integer(max_states, "max_states")
            if max_states < 1:
                raise ValueError("max_states must be positive or None")
            if dimension > max_states:
                raise MemoryError(
                    f"L={L} has {dimension:,} valid states, exceeding max_states="
                    f"{max_states:,}."
                )

        allowed = {I: (I, SIGMA), PSI: (PSI, SIGMA), SIGMA: (I, PSI, SIGMA)}
        states: list[State] | None = [] if retain_states else None
        valid_state_count = 0

        def extend(prefix: list[int]) -> None:
            nonlocal valid_state_count
            if len(prefix) == self.L:
                if prefix[0] in allowed[prefix[-1]]:
                    state = tuple(prefix)
                    valid_state_count += 1
                    if states is not None:
                        states.append(state)
                return
            for label in allowed[prefix[-1]]:
                prefix.append(label)
                extend(prefix)
                prefix.pop()

        for first in (I, PSI, SIGMA):
            extend([first])
        if valid_state_count != dimension:
            raise RuntimeError("Basis enumeration failed exact dimension identity")

        self.states = tuple(states) if states is not None else None
        self.state_to_index = ({state: i for i, state in enumerate(self.states)}
                               if self.states is not None else None)
        self.lookup: dict[int, tuple[int, int]] = {}
        orbits: list[Orbit] = []
        iterator = self.states if self.states is not None else self._iterate_states(allowed)
        for representative in iterator:
            packed_rep = _pack_state(representative)
            if packed_rep in self.lookup:
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
                self.lookup[_pack_state(member)] = (orbit_index, (-t) % period)
        self.orbits = tuple(orbits)
        self._sector_cache: dict[int, tuple[Orbit, ...]] = {}

    def _iterate_states(self, allowed: dict[int, tuple[int, ...]]) -> Iterator[State]:
        def extend(prefix: list[int]) -> Iterator[State]:
            if len(prefix) == self.L:
                if prefix[0] in allowed[prefix[-1]]:
                    yield tuple(prefix)
                return
            for label in allowed[prefix[-1]]:
                prefix.append(label)
                yield from extend(prefix)
                prefix.pop()
        for first in (I, PSI, SIGMA):
            yield from extend([first])

    def require_explicit_states(self) -> None:
        if self.states is None or self.state_to_index is None:
            raise MemoryError(
                "This Basis was created with retain_states=False to save memory. "
                "Rebuild with retain_states=True for explicit full-basis operations."
            )

    def locate(self, state: State) -> tuple[int, int]:
        return self.lookup[_pack_state(state)]

    def __len__(self) -> int:
        return self._dimension

    def momentum(self, m: int) -> int:
        return _integer(m, "m") % self.L

    def sector(self, m: int) -> tuple[Orbit, ...]:
        m = self.momentum(m)
        if m not in self._sector_cache:
            self._sector_cache[m] = tuple(
                orbit for orbit in self.orbits if (m * orbit.period) % self.L == 0
            )
        return self._sector_cache[m]


def generate_representatives(L: int, m: int, *, basis: Basis | None = None):
    basis = Basis(L) if basis is None else basis
    if basis.L != L:
        raise ValueError("L and basis.L disagree")
    return [{"rep": orbit.rep, "R": orbit.period} for orbit in basis.sector(m)]


def _local_action_valid(state: State, r: float, c: float, s: float):
    L = len(state)
    diagonal = 0.0
    for j in range(L):
        left, mid, right = state[(j - 1) % L], state[j], state[(j + 1) % L]
        lm, mm, rm = left != SIGMA, mid != SIGMA, right != SIGMA
        if lm != rm:
            diagonal -= r

        transitions: tuple[tuple[int, float], ...] = ()
        if lm and mm and rm:
            transitions = ((SIGMA, c),)
        elif lm and mm and not rm:
            transitions = ((SIGMA, s),)
        elif lm and not mm and rm:
            if left == right:
                transitions = ((left, c),)
        elif not lm and mm and rm:
            transitions = ((SIGMA, s),)
        elif lm and not mm and not rm:
            transitions = ((left, s),)
        elif not lm and mm and not rm:
            transitions = ((SIGMA, c * INV_SQRT2),)
        elif not lm and not mm and rm:
            transitions = ((right, s),)
        elif not lm and not mm and not rm:
            amplitude = c * INV_SQRT2
            transitions = ((I, amplitude), (PSI, amplitude))

        for new_mid, amplitude in transitions:
            if amplitude != 0.0:
                target = state[:j] + (new_mid,) + state[j + 1:]
                yield target, -amplitude
    if diagonal != 0.0:
        yield state, diagonal


def local_action(state, r: float, theta: float) -> Iterator[tuple[State, float]]:
    state = tuple(state)
    if len(state) < 3 or not is_valid_state(state):
        raise ValueError("state must be a valid periodic fusion-label state of L>=3")
    r, theta = _parameters(r, theta)
    yield from _local_action_valid(state, r, math.cos(theta), math.sin(theta))


def build_hamiltonian_block(basis: Basis, m: int, r: float, theta: float) -> csr_matrix:
    """Momentum block Hamiltonian matrix."""
    m = basis.momentum(m)
    r, theta = _parameters(r, theta)
    sector = basis.sector(m)
    indices = {orbit.rep: i for i, orbit in enumerate(sector)}
    k = 2.0 * math.pi * m / basis.L
    phase_by_shift = np.exp(1j * k * np.arange(basis.L))
    c, s = math.cos(theta), math.sin(theta)
    rows, columns, data = [], [], []
    append_row = rows.append
    append_col = columns.append
    append_data = data.append
    for col, source_orbit in enumerate(sector):
        for target, coefficient in _local_action_valid(source_orbit.rep, r, c, s):
            orbit_index, shift = basis.locate(target)
            target_orbit = basis.orbits[orbit_index]
            row = indices.get(target_orbit.rep)
            if row is None:
                continue
            factor = math.sqrt(source_orbit.period / target_orbit.period)
            append_row(row)
            append_col(col)
            append_data(coefficient * factor * phase_by_shift[shift])
    result = coo_matrix((data, (rows, columns)), shape=(len(sector), len(sector)),
                        dtype=np.complex128).tocsr()
    result.eliminate_zeros()
    return result


def symmetry_action(state: State, label: str) -> Iterator[tuple[State, float]]:
    """Full-basis action of U(psi) or U(sigma), Eqs. (18)--(21)."""
    state = tuple(state)
    if len(state) < 3 or not is_valid_state(state):
        raise ValueError("symmetry_action requires a valid state of L>=3")
    if label == "psi":
        yield tuple(1 - x if x != SIGMA else SIGMA for x in state), 1.0
        return
    if label != "sigma":
        raise ValueError("label must be 'psi' or 'sigma'")
    L = len(state)
    if len(set(state)) == 1:
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


def build_symmetry_block(basis: Basis, m: int, label: str) -> csr_matrix:
    """Build U(psi) or U(sigma) symmetry operator projected into momentum block m."""
    m = basis.momentum(m)
    sector = basis.sector(m)
    indices = {orbit.rep: i for i, orbit in enumerate(sector)}
    k = 2.0 * math.pi * m / basis.L
    phase_by_shift = np.exp(1j * k * np.arange(basis.L))
    rows, columns, data = [], [], []
    append_row = rows.append
    append_col = columns.append
    append_data = data.append
    for col, source_orbit in enumerate(sector):
        for target, coefficient in symmetry_action(source_orbit.rep, label):
            orbit_index, shift = basis.locate(target)
            target_orbit = basis.orbits[orbit_index]
            row = indices.get(target_orbit.rep)
            if row is None:
                continue
            factor = math.sqrt(source_orbit.period / target_orbit.period)
            append_row(row)
            append_col(col)
            append_data(coefficient * factor * phase_by_shift[shift])
    result = coo_matrix((data, (rows, columns)), shape=(len(sector), len(sector)),
                        dtype=np.complex128).tocsr()
    result.eliminate_zeros()
    return result


def full_hamiltonian(basis: Basis, r: float, theta: float) -> csr_matrix:
    basis.require_explicit_states()
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
    basis.require_explicit_states()
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
    basis.require_explicit_states()
    rows = [basis.state_to_index[translate(state)] for state in basis.states]
    return coo_matrix((np.ones(len(basis)), (rows, np.arange(len(basis)))),
                      shape=(len(basis), len(basis))).tocsr()


def symmetry_operator(basis: Basis, label: str, max_nnz: int = 10_000_000) -> csr_matrix:
    basis.require_explicit_states()
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
    nev = _integer(nev, "nev")
    if nev < 1:
        raise ValueError("nev must be positive")
    dense_cutoff = _integer(dense_cutoff, "dense_cutoff")
    max_dense_dimension = _integer(max_dense_dimension, "max_dense_dimension")
    if not isinstance(tol, Real) or not math.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")
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
        if n > max_dense_dimension:
            raise MemoryError(f"Dense diagonalization dimension {n} exceeds guard {max_dense_dimension}")
        evals, evecs = np.linalg.eigh(H.toarray())
        evals, evecs = evals[:count], evecs[:, :count]
    else:
        rng = np.random.default_rng(seed)
        v0 = rng.normal(size=n)
        if np.iscomplexobj(H.data):
            v0 = v0 + 1j * rng.normal(size=n)
        v0 /= np.linalg.norm(v0)
        ncv = min(n, max(2 * count + 1, 32))
        evals, evecs = eigsh(H, k=count, which="SA", v0=v0, tol=tol,
                             maxiter=maxiter, ncv=ncv)
        order = np.argsort(evals)
        evals, evecs = evals[order], evecs[:, order]
    if return_residuals:
        residuals = np.linalg.norm(H @ evecs - evecs * evals[None, :], axis=0)
        return evals, evecs, residuals
    return evals, evecs


def identify_irrep(lam_psi: float, lam_sigma: float) -> int:
    """Classify C_Ising irrep charge tuple (lambda_1, lambda_psi, lambda_sigma).

    Returns irrep index:
      0: (1,  1, +sqrt(2)) [Identity / Trivial irrep]
      1: (1,  1, -sqrt(2)) [Non-trivial irrep 1]
      2: (1, -1, 0)        [Non-trivial irrep 2]
    """
    if lam_psi > 0.0:
        return 0 if lam_sigma > 0.0 else 1
    return 2


def solve_sector_with_symmetries(basis: Basis, m: int, r: float, theta: float, nev: int = 5):
    """Diagonalize Hamiltonian block and classify C_Ising symmetry irreps."""
    H_block = build_hamiltonian_block(basis, m, r, theta)
    U_psi = build_symmetry_block(basis, m, "psi")
    U_sigma = build_symmetry_block(basis, m, "sigma")

    # Combine commuting operators to split energy-degenerate subspaces into exact irrep kets
    H_diag = H_block + 1e-6 * U_psi + 1.5e-6 * U_sigma
    _, evecs = lowest_eigenpairs(H_diag, nev=nev)

    evals = []
    l_psi = []
    l_sigma = []
    irreps = []
    residuals = []

    for i in range(evecs.shape[1]):
        v = evecs[:, i]
        E_val = float(np.real(np.vdot(v, H_block @ v)))
        lp_val = float(np.real(np.vdot(v, U_psi @ v)))
        ls_val = float(np.real(np.vdot(v, U_sigma @ v)))
        res_val = float(np.linalg.norm(H_block @ v - E_val * v))
        irr = identify_irrep(lp_val, ls_val)

        evals.append(E_val)
        l_psi.append(lp_val)
        l_sigma.append(ls_val)
        irreps.append(irr)
        residuals.append(res_val)

    order = np.argsort(evals)
    return (np.array(evals)[order], np.array(l_psi)[order],
            np.array(l_sigma)[order], np.array(irreps)[order],
            np.array(residuals)[order], H_block.shape[0])


def validate_model(basis: Basis, r: float, theta: float,
                   tolerance: float = 1e-9, max_dimension: int = 2000) -> dict:
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
        upsi_b = build_symmetry_block(basis, m, "psi")
        usigma_b = build_symmetry_block(basis, m, "sigma")
        dimensions.append(block.shape[0])
        errors[f"m{m}_isometry"] = _max_abs(Q.getH() @ Q - eye(Q.shape[1]))
        errors[f"m{m}_projected_H"] = _max_abs(Q.getH() @ H @ Q - block)
        errors[f"m{m}_projected_Upsi"] = _max_abs(Q.getH() @ upsi @ Q - upsi_b)
        errors[f"m{m}_projected_Usigma"] = _max_abs(Q.getH() @ usigma @ Q - usigma_b)
        errors[f"m{m}_Upsi_H_comm"] = _max_abs(upsi_b @ block - block @ upsi_b)
        errors[f"m{m}_Usigma_H_comm"] = _max_abs(usigma_b @ block - block @ usigma_b)
        phase = np.exp(-2j * math.pi * m / basis.L)
        errors[f"m{m}_translation_eigenvalue"] = _max_abs(T @ Q - phase * Q)
        errors[f"m{m}_H_hermiticity"] = _max_abs(block - block.getH())
        all_evals.extend(np.linalg.eigvalsh(block.toarray()))
    if sum(dimensions) != len(basis):
        raise AssertionError("Sum of momentum dimensions is not full dimension")
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
    parser.add_argument("--theta", type=float, default=theta, help="angle in radians (default pi/4)")
    parser.add_argument("--nev", type=int, default=nev)
    parser.add_argument("--max-states", type=int, default=50_000_000)
    parser.add_argument("--retain-states", action="store_true",
                        help="keep the full state list and index map in memory")
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    parser.add_argument("--validate", action="store_true", help="full small-system checks, L<=8")
    args, _ = parser.parse_known_args()

    r, theta = _parameters(args.r, args.theta)
    basis = Basis(args.L, max_states=args.max_states,
                  retain_states=args.retain_states or args.validate)
    result = {"L": basis.L, "r": r, "theta": theta, "dimension": len(basis),
              "translation": "right", "translation_eigenvalue": "exp(-i*k)",
              "momentum_formula": "k=2*pi*m/L", "sectors": []}
    print(f"L={basis.L}, dimension={len(basis)}, r={r:.12g}, theta={theta:.12g}")
    print("T is right translation; T eigenvalue exp(-ik), k=2*pi*m/L")

    sector_data = []

    for m in range(basis.L):
        evals, l_psi, l_sigma, irreps, residuals, dim = solve_sector_with_symmetries(
            basis, m, r, theta, nev=args.nev
        )
        result["sectors"].append({
            "m": m,
            "k": 2 * math.pi * m / basis.L,
            "dimension": dim,
            "eigenvalues": evals.tolist(),
            "lambda_psi": l_psi.tolist(),
            "lambda_sigma": l_sigma.tolist(),
            "irrep_indices": irreps.tolist(),
            "residual_norms": residuals.tolist()
        })
        printed = " ".join(f"{energy:.8f}(irr{irr})" for energy, irr in zip(evals, irreps))
        print(f"m={m:2d}  dim={dim:7d}  E: {printed}")
        sector_data.append((evals, l_psi, l_sigma, irreps))

    if args.validate:
        result["validation"] = validate_model(basis, r, theta)
        print(f"Validation passed; maximum absolute discrepancy "
              f"{result['validation']['max_error']:.3e}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")

    # Global ground state energy reference
    E_ground = min(np.min(evals) for evals, _, _, _ in sector_data)

    # Plot low-lying spectrum color-coded by C_Ising irrep
    plt.figure(figsize=(6, 7))

    irrep_info = {
        0: {"color": "#d62728", "marker": "o", "label": r"$\vec{\lambda} = (1, 1, \sqrt{2})$"},
        1: {"color": "#1f77b4", "marker": "s", "label": r"$\vec{\lambda} = (1, 1, -\sqrt{2})$"},
        2: {"color": "#2ca02c", "marker": "^", "label": r"$\vec{\lambda} = (1, -1, 0)$"},
    }

    for irr in (0, 1, 2):
        ks_plot = []
        energies_plot = []
        for m, (evals, _, _, irreps) in enumerate(sector_data):
            k_val = 2 * math.pi * m / basis.L
            for ev, irr_id in zip(evals, irreps):
                if irr_id == irr:
                    ks_plot.append(k_val)
                    energies_plot.append(ev - E_ground)
        if ks_plot:
            info = irrep_info[irr]
            plt.scatter(
                ks_plot, energies_plot,
                color=info["color"],
                marker=info["marker"],
                label=info["label"],
                s=40,
                alpha=0.85,
                zorder=3
            )

    plt.xlabel(r'Momentum $k = 2\pi m / L$')
    plt.ylabel(r'Energy $E - E_0$')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title=r"$\mathcal{C}_{\text{Ising}}$ Irrep Charges $\vec{\lambda}$", loc='best')
    plt.tight_layout()

    # Save data
    fig_filename = f"spectrum_L{args.L}_r{args.r}_theta{args.theta:.2f}.png"
    plt.savefig(fig_filename, dpi=300)
    print(f"Saved plot to {fig_filename}")
    plt.close()


if __name__ == "__main__":
    run_simulation(L=3, r=20.0, theta=np.pi/4, nev=4)