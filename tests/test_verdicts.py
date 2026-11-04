"""Verdikte (reine Funktionen in eig_evaluation.py) an handgesetzten Werten: Art, Schwellen und jede Zahl im Text (DEMO-PLAYBOOK §1b)."""

import eig_evaluation as ev


def _row(family, gap_pb=0.1, gap_glb=0.1, wins=False, valid=True, neg=True, ge=True, gap_eb=2.0, dens=0.5):
    return {"family": family, "gap_pb": gap_pb, "gap_glb": gap_glb, "gap_eb": gap_eb, "pb_wins": wins, "valid": valid, "eb_negative": neg, "pb_ge_eb": ge, "dens": dens}


def _by(**rows):
    return {f: r for f, r in rows.items()}


def test_eb_and_pb_verdicts_quote_the_computed_values(land6):
    s = ev.Settings("netlist", 6, 35)
    v = ev.instance_data(s, land6[1])
    sp = v["spectral"]
    kind, text = ev.verdict_eb(s, v)
    assert kind == "warning" and "Familie **Netzliste**, 6 Bauteile, Raster 2×3, Seed 35" in text and f"**{ev.num(sp.eb)}**" in text and "unter 0" in text and f"aufgezählte Optimum ist {v['ref']}" in text
    kind, text = ev.verdict_pb(s, v)
    assert f"**{ev.num(sp.pb)}**" in text and f"konstant {ev.num(sp.const)}, linear {ev.num(sp.linear)}, quadratisch {ev.num(sp.quad)}" in text and f"**{ev.pct(v['gap_pb'], 1)}**" in text
    assert kind == ("success" if sp.pb > v["glb"] + ev.TOL else "info")


def test_eb_verdict_for_a_positive_bound_and_pb_verdict_for_a_better_bound():
    s = ev.Settings("netlist", 6, 35)
    h = ev.hand_data()[2]
    v = {"inst": h["inst"], "spectral": h["spectral"], "glb": h["glb"], "exact": True, "ref": h["opt"], "perms": 24, "gap_pb": 0.1, "gap_glb": 0.2, "gap_eb": 2.4}
    kind, text = ev.verdict_pb(s, v)
    assert kind == "success" and "die bessere von beiden" in text
    pos = dict(v, spectral=type("S", (), {"eb": 3.0, "pb": 5.0, "const": 0.0, "linear": 0.0, "quad": 5.0})())
    kind2, text2 = ev.verdict_eb(s, pos)
    assert kind2 == "info" and "liegt über 0" in text2


def test_large_instance_verdicts_name_the_best_found_solution():
    s = ev.Settings("uniform", 12, 35)
    v = ev.instance_data(s)
    assert "beste gefundene Lösung (beste von 200 Abstiegen)" in ev.verdict_eb(s, v)[1] and "höchstens" in ev.verdict_pb(s, v)[1]


def test_gaps_verdict_counts_negative_and_ordered_and_checks_validity():
    by = _by(netlist=[_row("netlist", 0.20, 0.10, gap_eb=2.0), _row("netlist", 0.30, 0.10, gap_eb=3.0)], uniform=[_row("uniform", 0.10, 0.10, wins=True), _row("uniform", 0.08, 0.10, wins=True)])
    kind, text = ev.verdict_gaps(8, by)
    assert kind == "success" and "bei n = 8" in text and "über 4 Instanzen" in text and "unter 0 (sie liegt in 4 von 4 Instanzen" not in text and "in 4 von 4 Instanzen unter 0" in text
    assert "projizierte Schranke **15,0 %**" in text and "Gilmore–Lawler **10,0 %**" in text and "Netzliste PB 25,0 %, GLB 10,0 %" in text and "in 4 von 4 Instanzen mindestens bei der Eigenwertschranke" in text and "(gültig)" in text
    by["netlist"][0] = _row("netlist", valid=False)
    kind2, text2 = ev.verdict_gaps(8, by)
    assert kind2 == "error" and "ACHTUNG" in text2


def test_winner_verdict_names_always_never_and_the_count():
    by = _by(uniform=[_row("uniform", wins=True)] * 3, sparse=[_row("sparse", wins=False)] * 3, star=[_row("star", wins=True), _row("star", wins=False), _row("star", wins=True)])
    kind, text = ev.verdict_winner(8, by)
    assert kind == "info" and "in 5 von 9 Instanzen bei n = 8" in text and "Gleichverteilt 3 von 3" in text and "Stern 2 von 3" in text and "Immer besser ist die projizierte Schranke bei Gleichverteilt." in text
    assert "Nie besser ist sie bei Dünn besetzt." in text and "Keine der beiden Schranken gewinnt überall." in text


def test_winner_verdict_without_always_or_never():
    kind, text = ev.verdict_winner(6, _by(star=[_row("star", wins=True), _row("star", wins=False)]))
    assert "In keiner Familie ist die projizierte Schranke in jeder Instanz besser." in text and "In jeder Familie ist sie mindestens einmal besser." in text


def test_order_verdict_not_predictive_and_clear():
    corr = {6: {"EB~GLB": -0.29, "PB~GLB": -0.01, "EB~PB": 0.84}, 8: {"EB~GLB": -0.20, "PB~GLB": -0.15, "EB~PB": 0.9}}
    kind, text = ev.verdict_order(corr, {6: 120, 8: 120})
    assert kind == "warning" and "n = 6: PB~GLB −0,01, EB~GLB −0,29 (120 Instanzen, Unsicherheit grob ±0,19)" in text and "n = 8: PB~GLB −0,15" in text and "außerhalb" not in text and "verschieden" in text
    clear = ev.verdict_order({20: {"EB~GLB": 0.7, "PB~GLB": 0.71, "EB~PB": 0.9}}, {20: 60})
    assert clear[0] == "info" and "n = 20" in clear[1] and "gleichsinnig" in clear[1]
    anti = ev.verdict_order({6: {"EB~GLB": 0.0, "PB~GLB": -0.6, "EB~PB": 0.0}}, {6: 120})
    mixed = ev.verdict_order({6: {"EB~GLB": 0.0, "PB~GLB": -0.6, "EB~PB": 0.0}, 12: {"EB~GLB": 0.0, "PB~GLB": 0.6, "EB~PB": 0.0}}, {6: 120, 12: 60})
    assert "gegensinnig" in anti[1] and "uneinheitlich" in mixed[1]


def test_large_verdict_reports_medians_wins_and_growth():
    by = {12: _by(netlist=[_row("netlist", 0.30, 0.10, wins=False)] * 2, uniform=[_row("uniform", 0.08, 0.10, wins=True)] * 2),
          16: _by(netlist=[_row("netlist", 0.36, 0.15, wins=False)] * 2, uniform=[_row("uniform", 0.08, 0.13, wins=True)] * 2)}
    kind, text = ev.verdict_large(by)
    assert kind == "info" and "n = 12: PB 19,0 %, GLB 10,0 % (PB besser in 2 von 4)" in text and "n = 16: PB 22,0 %, GLB 14,0 % (PB besser in 2 von 4)" in text and "wächst mit n" in text and "nicht gleichmäßig" not in text
    flat = {12: by[12], 16: _by(netlist=[_row("netlist", 0.3, 0.05)])}
    assert "nicht gleichmäßig" in ev.verdict_large(flat)[1]


def test_density_verdict_negative_and_uneinheitlich_and_none():
    neg = ev.verdict_density({12: {"EB": -0.7, "PB": -0.7, "GLB": -0.68}, 16: {"EB": -0.7, "PB": -0.7, "GLB": -0.71}}, {12: 60, 16: 60})
    assert neg[0] == "info" and "n = 12: PB −0,70, GLB −0,68 (60 Instanzen, Unsicherheit grob ±0,28)" in neg[1] and "Bei dichterem Fluss ist die Lücke der projizierten Schranke kleiner" in neg[1] and "n = 12, n = 16" in neg[1]
    mixed = ev.verdict_density({6: {"EB": 0, "PB": 0.5, "GLB": 0}, 12: {"EB": 0, "PB": -0.6, "GLB": 0}}, {6: 120, 12: 60})
    none = ev.verdict_density({6: {"EB": 0, "PB": 0.05, "GLB": 0}}, {6: 120})
    assert "uneinheitlich" in mixed[1] and "in keiner Größe außerhalb der Unsicherheit" in none[1]


def test_core_verdict_for_exact_and_large(land6):
    s = ev.Settings("netlist", 6, 35)
    v = ev.instance_data(s, land6[1])
    text = ev.verdict_core(s, v)[1]
    assert f"Eigenwertschranke {ev.num(v['spectral'].eb)}" in text and f"projizierte Schranke {ev.num(v['spectral'].pb)}" in text and "Optimum" in text and "Experiment 3 bis 5" in text
    big = ev.instance_data(ev.Settings("uniform", 12, 35))
    assert "bester gefundener Lösung" in ev.verdict_core(ev.Settings("uniform", 12, 35), big)[1]
