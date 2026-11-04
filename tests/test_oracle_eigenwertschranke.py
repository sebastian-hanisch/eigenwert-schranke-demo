"""Orakel (ORACLE-PLAYBOOK §5): `scipy` rechnet die Eigenwerte, die Zuordnung und die Rangkorrelation unabhängig nach; die Aufzählung ist das Orakel für die Gültigkeit der Schranken.
Ohne scipy (nur requirements-dev.txt) werden die Tests übersprungen."""

import numpy as np
import pytest

import eig_bound as B
import eig_constants as C
import eig_evaluation as ev
import eig_instance as I
import eig_landscape as L
import eig_spectral as S
from eig_assignment import hungarian

linalg = pytest.importorskip("scipy.linalg")
scipy_opt = pytest.importorskip("scipy.optimize")
stats = pytest.importorskip("scipy.stats")


@pytest.mark.parametrize("family", C.FAMILIES)
def test_eigenvalues_equal_scipy_for_every_family_and_size(family):
    for n in C.SIZES:
        inst = I.make_instance(family, n, 4)
        for M in (inst.F.astype(float), inst.D.astype(float)):
            assert S.eigenvalues(M) == pytest.approx(linalg.eigh(M, eigvals_only=True), abs=1e-9)
        V = S.complement_basis(n)
        assert S.eigenvalues(V.T @ inst.F @ V) == pytest.approx(linalg.eigh(V.T @ inst.F.astype(float) @ V, eigvals_only=True), abs=1e-9)


def test_the_bounds_never_exceed_the_enumerated_optimum_on_many_instances():
    for family in C.FAMILIES:
        for seed in range(100000, 100008):
            inst = I.make_instance(family, 8, seed)
            sp = S.spectral_bounds(inst.F, inst.D)
            opt = L.landscape(inst).opt
            assert sp.eb <= opt + ev.TOL and sp.pb <= opt + ev.TOL


def test_the_hungarian_method_equals_scipy_on_the_gilmore_lawler_matrices():
    for family in C.FAMILIES:
        for n in C.SIZES:
            inst = I.make_instance(family, n, 2)
            L_ = B.lower_matrix(inst.F, inst.D)
            r, c = scipy_opt.linear_sum_assignment(L_)
            assert hungarian(L_)[1] == int(L_[r, c].sum())


def test_spearman_equals_scipy():
    rng = np.random.default_rng(1)
    for _ in range(10):
        a, b = rng.random(40), rng.random(40)
        assert ev.spearman(a, b) == pytest.approx(float(stats.spearmanr(a, b).statistic), abs=1e-12)
    a, b = rng.integers(0, 5, 60), rng.integers(0, 4, 60)
    assert ev.spearman(a, b) == pytest.approx(float(stats.spearmanr(a, b).statistic), abs=1e-12)
