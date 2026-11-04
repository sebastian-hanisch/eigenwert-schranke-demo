"""Randfälle (DEMO-PLAYBOOK §1b): kleinste Instanzen, Nullfluss, gleiche Flüsse, ungültige Eingaben."""

import math

import numpy as np
import pytest

import eig_constants as C
import eig_evaluation as ev
import eig_instance as I
import eig_landscape as L
import eig_spectral as S
from test_landscape import _inst


def test_zero_flow_gives_zero_bounds_and_undefined_gaps():
    inst = _inst(np.zeros((4, 4)))
    sp = S.spectral_bounds(inst.F, inst.D)
    ls = L.landscape(inst)
    assert sp.eb == pytest.approx(0.0, abs=1e-12) and sp.pb == pytest.approx(0.0, abs=1e-12) and ls.opt == 0 and math.isnan(ev.rel(ls.opt, sp.pb))


def test_every_family_and_size_runs_through_the_measurement():
    for fam in C.FAMILIES:
        for n in (6, 12):
            r = ev.sweep(fam, n, seeds=(1,))[0]
            assert r["family"] == fam and r["n"] == n and r["valid"] and r["eb_negative"] and r["pb_ge_eb"] and r["gap_pb"] >= 0


def test_seed_extremes_and_rejections():
    for seed in (0, C.SEED_MAX):
        inst = I.make_instance("sparse", 6, seed)
        assert S.spectral_bounds(inst.F, inst.D).pb == S.spectral_bounds(inst.F, inst.D).pb
    with pytest.raises(ValueError):
        I.make_instance("uniform", 7, 1)
    with pytest.raises(ValueError):
        I.make_instance("gaussian", 6, 1)


def test_the_eigenvalue_bound_is_negative_in_every_measured_family_and_size():
    for fam in C.FAMILIES:
        for n in C.SIZES:
            inst = I.make_instance(fam, n, 100000)
            sp = S.spectral_bounds(inst.F, inst.D)
            assert sp.eb < 0 and sp.pb >= sp.eb


def test_verdict_helpers_survive_a_single_family_and_a_single_size():
    rows = ev.sweep("star", 6, seeds=(1, 2))
    assert ev.verdict_gaps(6, {"star": rows})[0] == "success" and "Stern" in ev.verdict_winner(6, {"star": rows})[1]
    big = {12: {"star": ev.sweep("star", 12, seeds=(1, 2))}}
    assert "nicht gleichmäßig" in ev.verdict_large(big)[1]
    counts = {6: 2}
    corr = {6: ev.order_correlations(rows)}
    assert ev.verdict_order(corr, counts)[0] == "warning"
