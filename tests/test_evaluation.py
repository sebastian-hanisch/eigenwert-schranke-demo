"""Zeilen für die Vergleiche, Zusammenfassungen, Statistik und Instanzdaten: Handwerte und Eigenschaften."""

import math

import numpy as np
import pytest

import eig_bound as B
import eig_constants as C
import eig_evaluation as ev
import eig_instance as I
import eig_landscape as L
import eig_spectral as S


def _F(values, n):
    F = np.zeros((n, n), dtype=np.int16)
    k = 0
    for a in range(n):
        for b in range(a + 1, n):
            F[a, b] = F[b, a] = values[k]
            k += 1
    return F


def test_flow_statistics_of_the_hand_example_by_hand(hand):
    """Paarflüsse 5, 0, 1: Mittel 2, Varianz (9 + 4 + 1)/3, fd = 100 · √(14/3)/2 = 108,0; Dichte 2/3."""
    assert ev.flow_dominance(hand.F) == pytest.approx(100 * math.sqrt(14 / 3) / 2) and round(ev.flow_dominance(hand.F), 1) == 108.0 and ev.density(hand.F) == pytest.approx(2 / 3)
    assert ev.flow_dominance(_F([2, 2, 2], 3)) == 0.0 and math.isnan(ev.flow_dominance(np.zeros((4, 4), dtype=np.int16))) and ev.density(np.zeros((4, 4), dtype=np.int16)) == 0.0


def test_ranks_and_spearman_by_hand():
    assert ev.ranks([10, 20, 20, 30]).tolist() == [1.0, 2.5, 2.5, 4.0]
    assert ev.spearman([1, 2, 3, 4, 5], [2, 1, 4, 3, 5]) == pytest.approx(0.8)             # d = (−1, 1, −1, 1, 0), Σd² = 4: 1 − 6 · 4 / 120
    assert ev.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0) and math.isnan(ev.spearman([1, 1, 1], [1, 2, 3])) and math.isnan(ev.spearman([1], [2])) and math.isnan(ev.spearman([1, float("nan"), 3], [1, 2, 3]))
    with pytest.raises(ValueError):
        ev.spearman([1, 2], [1, 2, 3])


def test_spearman_band_quartiles_and_helpers():
    assert ev.spearman_band(120) == pytest.approx(1.96 * 1.06 / math.sqrt(117)) and round(ev.spearman_band(120), 2) == 0.19 and round(ev.spearman_band(60), 2) == 0.28 and math.isnan(ev.spearman_band(3))
    assert ev.quartiles([1, 2, 3, 4, 5]) == (2.0, 3.0, 4.0) and ev.median_of([{"v": 1}, {"v": 5}, {"v": 2}], "v") == 2.0
    assert ev.de(-1.5, 1) == "−1,5" and ev.pct(0.266, 1) == "26,6 %" and ev.sgn(0.44) == "+0,44" and ev.sgn(-0.13) == "−0,13" and ev.fam("star") == "Stern"
    assert ev.num(27.0) == "27" and ev.num(26.5) == "26,5" and ev.num(-77.14) == "−77,1" and ev.rel(10, 8) == pytest.approx(0.2) and math.isnan(ev.rel(0, 1))


def test_the_hand_data_match_the_readme():
    h = ev.hand_data()
    assert [x["opt"] for x in h] == [6, 10, 6] and [x["glb"] for x in h] == [6.0, 10.0, 5.0] and [round(x["spectral"].pb, 4) for x in h] == [5.2966, 9.8846, 5.5858] and [round(x["spectral"].eb, 4) for x in h] == [-12.0644, -17.3044, -8.5765]


def test_exact_row_by_hand_values_and_equals_the_parts(land6):
    inst, ls = land6
    r = ev.exact_row("netlist", 6, 35)
    sp = S.spectral_bounds(inst.F, inst.D)
    assert r["ref"] == ls.opt and r["eb"] == pytest.approx(sp.eb) and r["pb"] == pytest.approx(sp.pb) and r["glb"] == B.gilmore_lawler(inst.F, inst.D).glb and r["exact"] and r["valid"]
    assert r["gap_pb"] == pytest.approx((ls.opt - sp.pb) / ls.opt) and r["eb_negative"] == (sp.eb < 0) and r["pb_ge_eb"] and r["pb_wins"] == (sp.pb > r["glb"])


def test_large_row_uses_the_best_found_solution_as_reference():
    r = ev.large_row("uniform", 12, 100000)
    inst = I.make_instance("uniform", 12, 100000)
    ub, _ = B.best_found(inst.F, inst.D, 100000)
    assert r["ref"] == ub and not r["exact"] and r["valid"] and r["eb_negative"] and r["pb_ge_eb"] and r["gap_pb"] < r["gap_eb"]


def test_sweep_dispatches_on_the_size():
    assert [r["seed"] for r in ev.sweep("cluster", 6, seeds=(1, 2))] == [1, 2] and not ev.sweep("cluster", 12, seeds=(1,))[0]["exact"]
    assert len(ev.sweep("star", 12)) == len(C.LARGE_SEEDS) == 10 and len(C.SWEEP_SEEDS) == 20


def test_family_summary_by_hand():
    rows = [{"gap_eb": 2.0 + i, "gap_pb": g, "gap_glb": 0.1, "eb_negative": True, "pb_ge_eb": True, "pb_wins": w, "valid": True, "dens": 0.5} for i, (g, w) in enumerate(((0.1, True), (0.2, False), (0.3, True)))]
    s = ev.family_summary(rows)
    assert s["count"] == 3 and s["gap_pb"] == pytest.approx(0.2) and s["gap_eb"] == 3.0 and s["pb_min"] == 0.1 and s["pb_max"] == 0.3 and s["pb_wins"] == 2 and s["eb_negative"] == 3 and s["pb_ge_eb"] == 3 and s["valid"]
    rows[1]["valid"] = False
    assert not ev.family_summary(rows)["valid"]


def test_order_and_density_correlations_by_hand():
    rows = [{"gap_eb": a, "gap_pb": b, "gap_glb": c, "dens": d} for a, b, c, d in ((1, 1, 4, 0.1), (2, 2, 3, 0.2), (3, 3, 2, 0.3), (4, 4, 1, 0.4))]
    o = ev.order_correlations(rows)
    assert o["EB~PB"] == pytest.approx(1.0) and o["PB~GLB"] == pytest.approx(-1.0) and o["EB~GLB"] == pytest.approx(-1.0)
    d = ev.density_correlations(rows)
    assert d["PB"] == pytest.approx(1.0) and d["GLB"] == pytest.approx(-1.0)


def test_pooled_keeps_the_family_order():
    assert [r["x"] for r in ev.pooled({"star": [{"x": 1}], "netlist": [{"x": 2}, {"x": 3}]})] == [2, 3, 1]


def test_instance_data_exact_and_large(land6):
    s = ev.Settings("netlist", 6, 35)
    v = ev.instance_data(s, land6[1])
    assert v["exact"] and v["ref"] == land6[1].opt and v["perms"] == 720 and sorted(v["best"]) == list(range(6)) and v["spectral"].eb < 0
    big = ev.instance_data(ev.Settings("uniform", 12, 35))
    assert not big["exact"] and big["perms"] == math.factorial(12) and sorted(big["best"]) == list(range(12)) and big["gap_pb"] < big["gap_glb"]
    assert big["ref"] == ev.large_row("uniform", 12, 35)["ref"]


def test_landscape_of_uses_the_settings():
    assert ev.landscape_of(ev.Settings("cluster", 6, 3)).opt == L.landscape(I.make_instance("cluster", 6, 3)).opt
