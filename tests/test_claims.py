"""Jede Zahl aus README.md, den Hilfetexten der Presets und den App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Alles ist deterministisch (SplitMix64, ganzzahlige Kosten); die Eigenwerte kommen aus LAPACK und sind in der letzten Stelle plattformabhängig: die Zahlen unten sind auf die Rundung der README genau, nie an einer Rundungsgrenze.
Die Vergleiche laufen über je 20 feste Instanzen (n = 6, 8; Seeds 100000 bis 100019) und je 10 (n = 12, 16, 20; Seeds 100000 bis 100009), zusammen rund zwei Minuten."""

import math

import pytest

import eig_constants as C
import eig_evaluation as ev


@pytest.fixture(scope="module")
def data():
    return {n: {f: ev.sweep(f, n) for f in C.FAMILIES} for n in (6, 8, 12, 16, 20)}


def _p(x, d=1):
    return round(100 * x, d)


def test_the_hand_examples_of_the_readme():
    """README: Eigenwertschranke / projizierte Schranke / Gilmore–Lawler / Optimum: drei Bauteile auf einer Zeile −12,06 / 5,30 / 6 / 6; Raster 2×2 −17,30 / 9,88 / 10 / 10; vier Bauteile auf einer Zeile −8,58 / 5,59 / 5 / 6."""
    h = ev.hand_data()
    assert [round(x["spectral"].eb, 2) for x in h] == [-12.06, -17.30, -8.58] and [round(x["spectral"].pb, 2) for x in h] == [5.30, 9.88, 5.59]
    assert [x["glb"] for x in h] == [6.0, 10.0, 5.0] and [x["opt"] for x in h] == [6, 10, 6]
    assert [(round(x["spectral"].const, 2), round(x["spectral"].linear, 2), round(x["spectral"].quad, 2)) for x in h][0] == (-5.33, 10.0, 0.63)


def test_the_standard_instance_of_the_readme():
    """README: Netzliste, n = 8, Seed 35 (Instanz aus Stück 1 und 3): Optimum 31, Gilmore–Lawler 27 (Lücke 12,9 %); Eigenwertschranke −77,1; projizierte Schranke 13,7 (Lücke 55,9 %); Terme −43,8, 82,5, −25,1."""
    s = ev.Settings()
    v = ev.instance_data(s, ev.landscape_of(s))
    sp = v["spectral"]
    assert (v["ref"], v["glb"]) == (31, 27.0) and round(sp.eb, 1) == -77.1 and round(sp.pb, 1) == 13.7 and (_p(v["gap_glb"]), _p(v["gap_pb"])) == (12.9, 55.9)
    assert (round(sp.const, 1), round(sp.linear, 1), round(sp.quad, 1)) == (-43.8, 82.5, -25.1)


def test_the_eigenvalue_bound_is_negative_everywhere_and_the_projected_one_is_never_below(data):
    """README: in allen 420 Instanzen (120 + 120 + 60 + 60 + 60) ist die Eigenwertschranke kleiner als null, die projizierte liegt in allen 420 mindestens bei ihr, alle Schranken sind gültig; Lücke der Eigenwertschranke (Median über alle Familien) n = 6 / 8 / 12 / 16 / 20: 279 / 317 / 341 / 332 / 330 %."""
    rows = {n: ev.pooled(data[n]) for n in data}
    assert sum(len(r) for r in rows.values()) == 420 and all(r["eb_negative"] and r["pb_ge_eb"] and r["valid"] for rr in rows.values() for r in rr)
    assert [round(100 * ev.median_of(rows[n], "gap_eb")) for n in (6, 8, 12, 16, 20)] == [279, 317, 341, 332, 330]


def test_the_gaps_for_six_and_eight_components(data):
    """README: Lücke zum aufgezählten Optimum, Median in % (Netzliste, gleichverteilt, dünn, Cluster, Stern, schief): n = 6: PB 13,1 / 6,7 / 41,2 / 14,4 / 15,4 / 36,7, GLB 7,3 / 5,2 / 10,3 / 17,0 / 7,7 / 3,4;
    n = 8: PB 20,1 / 8,5 / 46,7 / 9,4 / 12,9 / 74,1, GLB 9,2 / 7,9 / 6,9 / 9,7 / 13,3 / 6,1; über alle 120 Instanzen PB 15,4 / 14,7, GLB 7,3 / 8,9 %."""
    s6 = {f: ev.family_summary(data[6][f]) for f in C.FAMILIES}
    s8 = {f: ev.family_summary(data[8][f]) for f in C.FAMILIES}
    assert [_p(s6[f]["gap_pb"]) for f in C.FAMILIES] == [13.1, 6.7, 41.2, 14.4, 15.4, 36.7] and [_p(s6[f]["gap_glb"]) for f in C.FAMILIES] == [7.3, 5.2, 10.3, 17.0, 7.7, 3.4]
    assert [_p(s8[f]["gap_pb"]) for f in C.FAMILIES] == [20.1, 8.5, 46.7, 9.4, 12.9, 74.1] and [_p(s8[f]["gap_glb"]) for f in C.FAMILIES] == [9.2, 7.9, 6.9, 9.7, 13.3, 6.1]
    assert [(_p(s6[f]["pb_min"]), _p(s6[f]["pb_max"])) for f in C.FAMILIES] == [(3.3, 34.2), (1.8, 16.5), (15.5, 138.9), (11.6, 17.1), (7.2, 28.9), (14.3, 80.7)]
    assert [(_p(s8[f]["pb_min"]), _p(s8[f]["pb_max"])) for f in C.FAMILIES] == [(10.9, 40.1), (3.2, 11.2), (25.4, 105.0), (7.5, 10.6), (9.2, 25.0), (27.7, 144.3)]
    assert [(_p(ev.median_of(ev.pooled(data[n]), "gap_pb")), _p(ev.median_of(ev.pooled(data[n]), "gap_glb"))) for n in (6, 8)] == [(15.4, 7.3), (14.7, 8.9)]


def test_who_wins_per_family(data):
    """README: Zahl der Instanzen, in denen die projizierte Schranke größer ist als die Gilmore–Lawler-Schranke (Netzliste, gleichverteilt, dünn, Cluster, Stern, schief): n = 6: 0 / 4 / 0 / 20 / 0 / 0; n = 8: 0 / 9 / 0 / 14 / 11 / 0;
    n = 12: 0 / 10 / 0 / 10 / 6 / 0; n = 16: 0 / 10 / 0 / 10 / 9 / 0; n = 20: 0 / 10 / 1 / 10 / 8 / 0; insgesamt 24 / 34 / 26 / 29 / 29."""
    wins = {n: [ev.family_summary(data[n][f])["pb_wins"] for f in C.FAMILIES] for n in data}
    assert wins == {6: [0, 4, 0, 20, 0, 0], 8: [0, 9, 0, 14, 11, 0], 12: [0, 10, 0, 10, 6, 0], 16: [0, 10, 0, 10, 9, 0], 20: [0, 10, 1, 10, 8, 0]}
    assert [sum(wins[n]) for n in (6, 8, 12, 16, 20)] == [24, 34, 26, 29, 29]


def test_the_gaps_beyond_enumeration(data):
    """README: Lücke gegen die beste gefundene Lösung (obere Grenze), Median in %: PB n = 12: 27,4 / 8,1 / 41,2 / 7,6 / 13,9 / 78,9; n = 16: 36,8 / 7,9 / 36,8 / 6,2 / 15,1 / 83,4; n = 20: 43,7 / 8,0 / 38,5 / 9,3 / 16,6 / 82,4;
    über alle Familien PB 17,4 / 23,0 / 25,9 %, GLB 12,3 / 16,0 / 19,8 % (n = 12 / 16 / 20)."""
    pb = {n: [_p(ev.family_summary(data[n][f])["gap_pb"]) for f in C.FAMILIES] for n in (12, 16, 20)}
    assert pb == {12: [27.4, 8.1, 41.2, 7.6, 13.9, 78.9], 16: [36.8, 7.9, 36.8, 6.2, 15.1, 83.4], 20: [43.7, 8.0, 38.5, 9.3, 16.6, 82.4]}
    assert [_p(ev.median_of(ev.pooled(data[n]), "gap_pb")) for n in (12, 16, 20)] == [17.4, 23.0, 25.9] and [_p(ev.median_of(ev.pooled(data[n]), "gap_glb")) for n in (12, 16, 20)] == [12.3, 16.0, 19.8]
    assert all(not r["exact"] for n in (12, 16, 20) for r in ev.pooled(data[n]))
    assert [_p(ev.family_summary(data[n]["uniform"])["gap_glb"]) for n in (12, 16, 20)] == [10.2, 12.9, 15.1]          # bei gleichverteilten Flüssen wächst die Gilmore–Lawler-Lücke, die PB-Lücke (8,1 / 7,9 / 8,0) nicht


def test_the_ordering_of_instances_by_the_bounds(data):
    """README: Rangkorrelation der Lücken (Spearman, über alle Familien; 120 Instanzen ±0,19, 60 Instanzen ±0,28), n = 6 / 8 / 12 / 16 / 20: PB~GLB −0,01 / −0,28 / +0,41 / +0,44 / +0,71; EB~GLB −0,29 / −0,20 / +0,56 / +0,56 / +0,71; EB~PB +0,84 / +0,90 / +0,91 / +0,87 / +0,94."""
    o = {n: ev.order_correlations(ev.pooled(data[n])) for n in data}
    assert [round(o[n]["PB~GLB"], 2) for n in (6, 8, 12, 16, 20)] == [-0.01, -0.28, 0.41, 0.44, 0.71] and [round(o[n]["EB~GLB"], 2) for n in (6, 8, 12, 16, 20)] == [-0.29, -0.20, 0.56, 0.56, 0.71]
    assert [round(o[n]["EB~PB"], 2) for n in (6, 8, 12, 16, 20)] == [0.84, 0.90, 0.91, 0.87, 0.94]
    kind, text = ev.verdict_order({n: o[n] for n in (6, 8)}, {6: 120, 8: 120})
    assert kind == "info" and "n = 8" in text and "gegensinnig" in text


def test_the_density_correlations(data):
    """README: Rangkorrelation der Lücke mit der Dichte, n = 6 / 8 / 12 / 16 / 20: PB −0,78 / −0,83 / −0,70 / −0,70 / −0,77; EB −0,85 / −0,88 / −0,72 / −0,69 / −0,72; GLB +0,07 / +0,05 / −0,68 / −0,71 / −0,78."""
    d = {n: ev.density_correlations(ev.pooled(data[n])) for n in data}
    assert [round(d[n]["PB"], 2) for n in (6, 8, 12, 16, 20)] == [-0.78, -0.83, -0.70, -0.70, -0.77] and [round(d[n]["EB"], 2) for n in (6, 8, 12, 16, 20)] == [-0.85, -0.88, -0.72, -0.69, -0.72]
    assert [round(d[n]["GLB"], 2) for n in (6, 8, 12, 16, 20)] == [0.07, 0.05, -0.68, -0.71, -0.78]
    kind, text = ev.verdict_density({n: d[n] for n in (12, 16, 20)}, {n: 60 for n in (12, 16, 20)})
    assert kind == "info" and "Bei dichterem Fluss ist die Lücke der projizierten Schranke kleiner" in text


def test_the_verdicts_of_the_readme_hold(data):
    kind, text = ev.verdict_gaps(8, data[8])
    assert kind == "success" and "über 120 Instanzen" in text and "in 120 von 120 Instanzen unter 0" in text and "projizierte Schranke **14,7 %**" in text and "Gilmore–Lawler **8,9 %**" in text
    kind, text = ev.verdict_winner(12, data[12])
    assert "in 26 von 60 Instanzen bei n = 12" in text and "Immer besser ist die projizierte Schranke bei Gleichverteilt, Zwei Cluster." in text and "Nie besser ist sie bei Netzliste, Dünn besetzt, Wenige starke Flüsse." in text
    kind, text = ev.verdict_large({n: data[n] for n in (12, 16, 20)})
    assert "n = 12: PB 17,4 %, GLB 12,3 % (PB besser in 26 von 60)" in text and "n = 20: PB 25,9 %, GLB 19,8 % (PB besser in 29 von 60)" in text and "wächst mit n" in text


def test_the_counts_in_the_texts():
    """Texte: n! für n = 6, 8, 9, 12 (720, 40 320, 362 880, 479 001 600); 120 = 6 · 20; 60 = 6 · 10; 420 = 120 + 120 + 3 · 60; 200 Abstiege."""
    assert [math.factorial(n) for n in (6, 8, 9, 12)] == [720, 40320, 362880, 479001600] and C.n_text(math.factorial(12)) == "479 001 600"
    assert len(C.FAMILIES) * len(C.SWEEP_SEEDS) == 120 and len(C.FAMILIES) * len(C.LARGE_SEEDS) == 60 and 2 * 120 + 3 * 60 == 420 and C.STARTS == 200
