"""Pfade der App, die der Rauchtest nicht erreicht: die Experimente im Seiteninhalt, PDF, jede Familie, große Instanzen, die Rechenknöpfe (DEMO-PLAYBOOK §1d)."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

import eig_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")
KIND = {"family_select": "selectbox", "n_select": "select_slider", "seed_input": "number_input"}


def _app(**state) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    for key, value in state.items():
        getattr(at.sidebar, KIND[key])(key=key).set_value(value)
    if state:
        at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return " ".join(str(e.value) for e in list(at.success) + list(at.info) + list(at.warning) + list(at.error) + list(at.markdown))


def test_default_page_shows_the_metrics_and_the_verdicts():
    at = _app()
    values = {m.label: m.value for m in at.metric}
    assert set(values) == {"Eigenwertschranke (EB)", "Projizierte Schranke (PB)", "Gilmore–Lawler", "Optimum", "Lücke der PB"}
    assert values["Optimum"] == "31" and values["Gilmore–Lawler"] == "27" and values["Eigenwertschranke (EB)"].startswith("−") and values["Lücke der PB"].endswith(" %")
    text = _texts(at)
    for part in ("Familie **Netzliste**, 8 Bauteile, Raster 2×4, Seed 35", "liegt unter 0", "Welche Schranke die bessere ist"):
        assert part in text, part


def test_the_experiments_the_hand_examples_the_limits_table_and_the_formulation_are_on_the_page():
    at = _app()
    headings = [s.value for s in at.subheader]
    assert [h.split(":")[0] for h in headings if h.startswith("🔬")] == [f"🔬 Experiment {i}" for i in (1, 2, 3, 5)] and "🚧 Wo die Annahmen enden" in headings
    assert [e.label for e in at.expander] == ["So funktioniert es", "Drei Handbeispiele zum Nachrechnen", "📐 Mathematische Formulierung"]
    assert not at.error and len(at.get("plotly_chart")) == 5


def test_every_family_renders_and_a_large_instance_replaces_the_optimum():
    for fam in C.FAMILIES:
        at = _app(family_select=fam, n_select=6)
        assert f"Familie **{C.FAMILY_NAMES[fam]}**" in _texts(at) and len(at.get("plotly_chart")) == 5
    at = _app(n_select=12)
    values = {m.label: m.value for m in at.metric}
    assert "Beste gefundene Lösung" in values and "Optimum" not in values and "Lücke der PB (höchstens)" in values


def test_nine_waits_for_the_button():
    at9 = _app(n_select=9)
    assert any("362 880 Anordnungen" in str(i.value) for i in at9.info) and any(b.label == "⏱️ n = 9 rechnen" for b in at9.button)
    assert not at9.metric and "Familie **" not in _texts(at9)


def test_the_pdf_button_makes_a_download():
    at = _app(n_select=6)
    at.button(key="pdf_make").click().run()
    assert not at.exception
    assert any(d.proto.label == "⬇️ PDF herunterladen" for d in at.get("download_button"))


def test_the_comparison_for_six_components_fills_experiments_three_and_four():
    at = _app(n_select=6)
    assert any(b.label == "⏱️ Vergleich für n = 6 rechnen" for b in at.button)
    at.button(key="sweep_button_6").click().run()
    assert not at.exception
    headings = [s.value for s in at.subheader]
    assert [h.split(":")[0] for h in headings if h.startswith("🔬")] == [f"🔬 Experiment {i}" for i in range(1, 6)]
    text = _texts(at)
    for part in ("Lücke zum aufgezählten Optimum bei n = 6", "Die projizierte Schranke ist größer als die Gilmore–Lawler-Schranke in", "Rangkorrelation der Lücken, jeweils über alle Familien: n = 6"):
        assert part in text, part
    assert len(at.get("plotly_chart")) == 5 + 3 and any("Weitere Größen" in str(c.value) for c in at.caption)


def test_the_size_choice_of_the_comparison_switches_the_button():
    at = _app()
    at.radio(key="sweep_n").set_value(8).run()
    assert any(b.label == "⏱️ Vergleich für n = 8 rechnen" for b in at.button) and not at.exception


def test_the_large_instances_button_fills_experiment_five():
    at = _app(n_select=6)
    assert any(b.label == "⏱️ Große Instanzen rechnen (n = 12, 16, 20)" for b in at.button)
    at.button(key="large_button").click().run()
    assert not at.exception
    text = _texts(at)
    assert "Schranken gegen die beste gefundene Lösung" in text and "Rangkorrelation (Spearman) der Lücke mit der Dichte des Flusses: n = 12" in text and len(at.get("plotly_chart")) == 5 + 1
