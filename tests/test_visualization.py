"""Abbildungen: sie entstehen für jede Eingabe, Achsen sind gesperrt, Beschriftungen und Daten stimmen mit den Werten überein; PDF-Export."""

import eig_constants as C
import eig_evaluation as ev
import eig_instance as I
import eig_visualization as V
from eig_pdf_export import build_pdf


def _locked(fig):
    return fig.layout.xaxis.fixedrange and fig.layout.yaxis.fixedrange


def test_cost_bar_has_the_four_values_in_order(land6):
    s = ev.Settings("netlist", 6, 35)
    v = ev.instance_data(s, land6[1])
    fig = V.build_cost_bar(v)
    xs = list(fig.data[0].x)
    assert xs == [v["spectral"].eb, v["spectral"].pb, v["glb"], v["ref"]] and list(fig.data[0].y)[3] == "Optimum" and _locked(fig)
    big = ev.instance_data(ev.Settings("uniform", 12, 35))
    assert list(V.build_cost_bar(big).data[0].y)[3] == "beste gefundene Lösung"


def test_spectra_terms_and_projected_charts(land6):
    v = ev.instance_data(ev.Settings("netlist", 6, 35), land6[1])
    sp = v["spectral"]
    spectra = V.build_spectra(sp)
    assert len(spectra.data) == 3 and list(spectra.data[0].y) == list(sp.eig_F) and list(spectra.data[1].y) == list(sp.eig_D[::-1]) and list(spectra.data[2].y) == list(sp.products) and _locked(spectra)
    terms = V.build_terms(sp)
    assert list(terms.data[0].y) == [sp.const, sp.linear, sp.quad, sp.pb] and _locked(terms)
    proj = V.build_projected_spectra(sp)
    assert len(proj.data) == 2 and len(proj.data[0].x) == 5 and _locked(proj)


def test_board_chart_for_every_family_and_size(hand):
    for fam in C.FAMILIES:
        for n in (6, 9, 12, 20):
            inst = I.make_instance(fam, n, 3)
            assert len(V.build_board(inst, list(range(n)), fam).data[-1].x) == n
    assert len(V.build_board(hand, [0, 1, 2], "Hand").data[-1].x) == 3


def _by(n=6, seeds=(1, 2, 3)):
    return {f: ev.sweep(f, n, seeds) for f in C.FAMILIES}


def test_gap_winner_order_and_large_charts():
    by = _by()
    bars = V.build_gap_bars(6, by, ev.median_of)
    assert [t.name for t in bars.data] == ["Gilmore–Lawler", "projizierte Schranke (PB)"] and len(bars.data[0].x) == 6 and _locked(bars)
    win = V.build_winner({6: by, 8: by})
    assert [t.name for t in win.data] == ["n = 6", "n = 8"] and all(0 <= y <= 100 for t in win.data for y in t.y) and _locked(win)
    order = V.build_order({6: ev.order_correlations(ev.pooled(by))})
    assert [t.name for t in order.data] == ["n = 6"] and list(order.data[0].x) == ["EB~GLB", "PB~GLB", "EB~PB"] and _locked(order)
    big = {12: {f: ev.sweep(f, 12, (1, 2)) for f in C.FAMILIES}, 16: {f: ev.sweep(f, 16, (1, 2)) for f in C.FAMILIES}}
    lf = V.build_large(big, ev.median_of)
    assert len(lf.data) == 4 and [t.name for t in lf.data if t.showlegend] == ["n = 12", "n = 16"] and _locked(lf)


def test_charts_skip_families_without_rows():
    by = _by()
    by["star"] = []
    assert len(V.build_gap_bars(6, by, ev.median_of).data[0].x) == 5 and len(V.build_winner({6: by}).data[0].x) == 5


def test_the_pdf_export_returns_a_pdf_for_an_exact_and_a_large_instance(land6):
    s = ev.Settings("netlist", 6, 35)
    assert build_pdf(s, ev.instance_data(s, land6[1]))[:5] == b"%PDF-"
    s2 = ev.Settings("uniform", 20, 35)
    data = build_pdf(s2, ev.instance_data(s2))
    assert data[:5] == b"%PDF-" and len(data) > 1500
