"""Eigenwertschranke und Projektion - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 4 der Reihe „Anordnen: relaxieren und runden“: statt der Zuordnung (Stück 3) das Spektrum. Die Eigenwertschranke paart die Eigenwerte der Fluss- und der Abstandsmatrix; die projizierte Schranke rechnet zusätzlich im Raum senkrecht zum Einsvektor.
Gemessen wird, wie weit beide unter dem Optimum liegen, ob sie die Gilmore–Lawler-Schranke schlagen und ob sie Instanzen gleich ordnen. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import eig_constants as C
import eig_spectral as S
from eig_evaluation import (
    Settings,
    de,
    density_correlations,
    family_summary,
    fam,
    hand_data,
    instance_data,
    landscape_of,
    median_of,
    num,
    order_correlations,
    pct,
    pooled,
    spearman_band,
    sweep,
    verdict_core,
    verdict_density,
    verdict_eb,
    verdict_gaps,
    verdict_large,
    verdict_order,
    verdict_pb,
    verdict_winner,
)
from eig_layout import dunkles_design, kopf_css
from eig_pdf_export import build_pdf
from eig_presets import bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sidebar_presets, sync_query_params
from eig_visualization import build_board, build_cost_bar, build_gap_bars, build_large, build_order, build_projected_spectra, build_spectra, build_terms, build_winner

st.set_page_config(page_title="Eigenwertschranke und Projektion – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner="Rechne alle Anordnungen …")
def _landscape(family, n, seed):
    return landscape_of(Settings(family, n, seed))


@st.cache_data(show_spinner=False)
def _sweep(family, n):
    return sweep(family, n)


def verdict(pair):
    kind, text = pair
    getattr(st, kind)(text)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in eig_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🎼 Eigenwertschranke und Projektion")
st.markdown(
    """
**Was sagt das Spektrum, und was die Projektion?** Dieselbe Aufgabe wie in Stück 1 bis 3: `n` Bauteile auf `n` Plätze, Kosten gleich der Summe aus **Fluss mal Abstand** über alle Paare. Statt einer Zuordnung (Gilmore–Lawler, Stück 3) nutzt die **Eigenwertschranke**
nur die Eigenwerte der Fluss- und der Abstandsmatrix: Permutationsmatrizen sind orthogonal, und für orthogonale Matrizen lässt sich die Summe nach unten durch gepaarte Eigenwerte abschätzen. Die **projizierte Schranke** rechnet zusätzlich in dem Raum, der zum
Einsvektor senkrecht steht, und schätzt den linearen Teil getrennt ab. Beide Schranken kosten nur zwei bis drei Eigenwertzerlegungen. Gemessen wird, wie weit sie unter dem Optimum liegen und ob sie die Gilmore–Lawler-Schranke schlagen.
Die sechs Familien der Flussmatrix sind **eigene, einfache Erfindungen** (aus Stück 2), keine echten Platinen; die projizierte Schranke ist hier eine **eigene vereinfachte Herleitung**, nicht die Originalfassung der Literatur.
Stück 4 der Reihe „Anordnen: relaxieren und runden“.

Weiter unten: **fünf Experimente** (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert es", expanded=True):
    st.markdown(
        """
1. **Eigenwertschranke (EB):** die Eigenwerte von `F` (aufsteigend) und `D` (absteigend) werden paarweise multipliziert und summiert, halbiert. Weil `F` nichtnegativ ist, hat sie einen großen positiven Eigenwert, der mit dem kleinsten Eigenwert von `D` gepaart wird: die Schranke ist oft **negativ**.
2. **Projektion:** jede Permutationsmatrix ist `X = e eᵀ/n + V Y Vᵀ` (`V`: Basis senkrecht zum Einsvektor `e`). Das zerlegt die Kosten **exakt** in einen konstanten, einen linearen und einen quadratischen Teil; der lineare wird durch Sortieren, der quadratische durch gepaarte Eigenwerte der projizierten Matrizen nach unten abgeschätzt.
3. **Die Lücke:** (Optimum − Schranke) / Optimum. Bei Instanzen, die sich nicht aufzählen lassen, steht die beste gefundene Lösung (beste von 200 steilsten Abstiegen) an der Stelle des Optimums; die Lücke ist dann **höchstens** so groß.
4. **Zum Vergleich** wird in jeder Instanz auch die Gilmore–Lawler-Schranke gerechnet (Stück 3).
5. **Feste Instanzen:** Vergleiche laufen über je 20 feste Seeds (n = 6, 8) oder 10 (n = 12, 16, 20), nie über einen einzelnen.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Die Instanz (Experiment 1 und 2)**")
    family = st.selectbox("Familie der Flussmatrix", C.FAMILIES, key="family_select", format_func=lambda f: C.FAMILY_NAMES[f], help="Wie der Fluss zwischen den Bauteilen verteilt ist. Die Beschreibung steht unter dem Regler.")
    st.caption(C.FAMILY_HELP[family])
    n = st.select_slider("Bauteile n", options=C.SIZES, key="n_select", help="Bis n = 9 wird alles aufgezählt (n = 9 auf Knopfdruck, rund eine Viertelminute); ab n = 12 gibt es nur die Schranken und die beste von 200 Abstiegen.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Instanz", width="stretch", on_click=randomize_seed)

sync_query_params({"family_select": family, "n_select": int(n), "seed_input": int(seed)})
settings = Settings(family, int(n), int(seed))

gated = settings.n == 9 and not st.session_state.get("n9_ok")
if gated:
    st.info("Bei n = 9 sind es 362 880 Anordnungen; die vollständige Rechnung dauert rund eine Viertelminute (auf Streamlit Cloud eher länger). Mit dem Knopf starten:")
    if st.button("⏱️ n = 9 rechnen", key="n9_button"):
        st.session_state["n9_ok"] = True
        st.rerun()
else:
    ls = _landscape(settings.family, settings.n, settings.seed) if settings.n in C.EXACT_SIZES else None
    v = instance_data(settings, ls)
    sp = v["spectral"]
    ref_label = "Optimum" if v["exact"] else "Beste gefundene Lösung"

    # --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

    st.markdown("## 🎯 Ergebnis")
    m1, m2, m3 = st.columns(3)
    m4, m5 = st.columns(2)
    m1.metric("Eigenwertschranke (EB)", num(sp.eb), help="Gepaarte Eigenwerte von Fluss und Abstand. Negativ heißt: schwächer als die triviale Schranke 0.")
    m2.metric("Projizierte Schranke (PB)", num(sp.pb), help="Vereinfachte Fassung: konstanter, linearer und quadratischer Teil getrennt abgeschätzt.")
    m3.metric("Gilmore–Lawler", num(v["glb"]), help="Die Zuordnungsschranke aus Stück 3, zum Vergleich.")
    m4.metric(ref_label, str(v["ref"]), help="Die kleinsten Kosten aller n! Anordnungen (aufgezählt, bewiesen)." if v["exact"] else "Die beste von 200 steilsten Abstiegen aus Zufallsstarts: eine obere Schranke des Optimums, kein Beweis.")
    m5.metric("Lücke der PB" if v["exact"] else "Lücke der PB (höchstens)", pct(v["gap_pb"], 1), delta=f"Gilmore–Lawler {pct(v['gap_glb'], 1)}", delta_color="off", help="(Vergleichswert − Schranke) / Vergleichswert.")
    st.plotly_chart(build_cost_bar(v), width="stretch", key="cost_chart")

    if st.button("📄 PDF des aktuellen Ergebnisses erstellen", key="pdf_make"):
        st.session_state["pdf_made"] = (settings, build_pdf(settings, v))
    made = st.session_state.get("pdf_made")
    if made and made[0] == settings:
        st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="eigenwertschranke.pdf", mime="application/pdf", key="pdf_download")

    # --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

    st.markdown("## ❓ Kernfrage: Was sagt das Spektrum, und was die Projektion?")
    verdict(verdict_core(settings, v))

    st.markdown("---")

    # --- Experiment 1 ---------------------------------------------------------------------------------------------------------------------------

    st.subheader("🔬 Experiment 1: Die Eigenwertschranke")
    st.caption("Links die gepaarten Eigenwerte von Fluss (aufsteigend) und Abstand (absteigend), rechts die Summanden der Schranke: rot, wo das Produkt negativ ist. Darunter drei Handbeispiele zum Nachrechnen.")
    st.plotly_chart(build_spectra(sp), width="stretch", key="spectra_chart")
    verdict(verdict_eb(settings, v))
    with st.expander("Drei Handbeispiele zum Nachrechnen"):
        for h in hand_data():
            hi, hs = h["inst"], h["spectral"]
            st.markdown(f"**{h['name']}** (Plätze auf einem {hi.rows}×{hi.cols}-Raster): Eigenwertschranke **{num(hs.eb)}**, projizierte Schranke **{num(hs.pb)}**, Gilmore–Lawler **{num(h['glb'])}**, Optimum **{h['opt']}**.")
            st.dataframe(pd.DataFrame({"Eigenwerte von F": [de(x, 3) for x in hs.eig_F], "Eigenwerte von D": [de(x, 3) for x in hs.eig_D]}, index=[f"λ{i + 1}" for i in range(hi.n)]), width="stretch")

    st.markdown("---")

    # --- Experiment 2 ---------------------------------------------------------------------------------------------------------------------------

    st.subheader("🔬 Experiment 2: Die Projektion")
    st.caption("Links die drei Terme der projizierten Schranke und ihre Summe, rechts die Eigenwerte der projizierten Matrizen. Unten die Probe: die Zerlegung ist eine **Gleichung**, sie stimmt für jede Anordnung.")
    p1, p2 = st.columns(2)
    with p1:
        st.plotly_chart(build_terms(sp), width="stretch", key="terms_chart")
    with p2:
        st.plotly_chart(build_projected_spectra(sp), width="stretch", key="projected_chart")
    verdict(verdict_pb(settings, v))
    left, right = S.decomposition_terms(v["inst"].F, v["inst"].D, v["best"])
    st.markdown(f"**Die Zerlegung nachgerechnet** an der Anordnung unten (Optimum bzw. beste gefundene Lösung): `tr(F X D Xᵀ)` = **{de(left, 3)}**, Summe der drei Terme der Zerlegung (mit der Anordnung selbst im linearen und quadratischen Teil) = **{de(right, 3)}**.")
    st.plotly_chart(build_board(v["inst"], v["best"], f"{ref_label} (Kosten {v['ref']})"), width="stretch", key="board_best")

    st.markdown("---")

# --- Experimente 3 und 4: Vergleiche mit aufgezähltem Optimum ----------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Wie weit liegen die Schranken unter dem Optimum?")
st.caption("Je Familie 20 feste Instanzen (Seeds 100000 bis 100019), alle Anordnungen aufgezählt. Die Größe wählen und die Rechnung starten (n = 6: Sekunden, n = 8: rund zwei Minuten).")
sweep_n = st.radio("Größe des Vergleichs", C.SWEEP_SIZES, key="sweep_n", horizontal=True, format_func=lambda x: f"n = {x}")
done = st.session_state.setdefault("sweeps_done", {})
if not done.get(sweep_n):
    wait = "wenige Sekunden" if sweep_n == 6 else "rund zwei Minuten (auf Streamlit Cloud eher länger)"
    st.info(f"Der Vergleich für n = {sweep_n} rechnet 6 Reihen zu je 20 Instanzen und dauert {wait}. Mit dem Knopf starten:")
    if st.button(f"⏱️ Vergleich für n = {sweep_n} rechnen", key=f"sweep_button_{sweep_n}"):
        bar = st.progress(0.0, text="Rechne die Familien …")
        for i, f in enumerate(C.FAMILIES):
            _sweep(f, sweep_n)
            bar.progress((i + 1) / len(C.FAMILIES), text=f"{C.FAMILY_NAMES[f]} fertig")
        done[sweep_n] = True
        st.rerun()
else:
    by = {f: _sweep(f, sweep_n) for f in C.FAMILIES}
    st.plotly_chart(build_gap_bars(sweep_n, by, median_of), width="stretch", key="gap_chart")
    table = []
    for f in C.FAMILIES:
        s = family_summary(by[f])
        table.append({"Familie": fam(f), "Lücke EB (Median)": pct(s["gap_eb"]), "EB unter 0": f"{s['eb_negative']} von {s['count']}", "Lücke PB (Median)": pct(s["gap_pb"], 1), "PB mindestens EB": f"{s['pb_ge_eb']} von {s['count']}",
                      "Lücke Gilmore–Lawler (Median)": pct(s["gap_glb"], 1)})
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
    verdict(verdict_gaps(sweep_n, by))

    st.markdown("---")

    st.subheader("🔬 Experiment 4: Welche Schranke ist die bessere, und ordnen sie gleich?")
    st.caption("Zahl der Instanzen, in denen die projizierte Schranke größer ist als die Gilmore–Lawler-Schranke (alle gerechneten Größen), und die Rangkorrelation der Lücken der Schranken untereinander (Spearman, über alle Familien; die Unsicherheit steht im Text).")
    by_all = {m: {f: _sweep(f, m) for f in C.FAMILIES} for m in C.SWEEP_SIZES if done.get(m)}
    if st.session_state.get("large_ok"):
        by_all.update({m: {f: _sweep(f, m) for f in C.FAMILIES} for m in C.LARGE_SIZES})
    st.plotly_chart(build_winner(by_all), width="stretch", key="winner_chart")
    verdict(verdict_winner(sweep_n, by))
    corr_by_n = {m: order_correlations(pooled(by_all[m])) for m in by_all}
    counts = {m: len(pooled(by_all[m])) for m in by_all}
    st.plotly_chart(build_order(corr_by_n), width="stretch", key="order_chart")
    st.dataframe(pd.DataFrame([{"Größe": f"n = {m}", "Instanzen": counts[m], "Unsicherheit grob ±": de(spearman_band(counts[m]), 2), **{k: de(r, 2) for k, r in corr_by_n[m].items()}} for m in corr_by_n]), hide_index=True, width="stretch")
    verdict(verdict_order(corr_by_n, counts))
    if len(by_all) < len(C.SWEEP_SIZES) + len(C.LARGE_SIZES):
        st.caption("Weitere Größen: oben die andere Größe wählen und rechnen, und unten die großen Instanzen (Experiment 5).")

st.markdown("---")

# --- Experiment 5: große Instanzen --------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Jenseits der Aufzählung")
st.caption("Für n = 12, 16 und 20 gibt es keine Aufzählung: die Schranken gegen die beste gefundene Lösung (beste von 200 steilsten Abstiegen), je Familie zehn feste Instanzen. Die Lücken sind **obere Grenzen** der wahren Lücke.")
if not st.session_state.get("large_ok"):
    st.info("Die Rechnung für n = 12, 16 und 20 dauert rund eine halbe Minute (auf Streamlit Cloud eher länger). Mit dem Knopf starten:")
    if st.button("⏱️ Große Instanzen rechnen (n = 12, 16, 20)", key="large_button"):
        bar = st.progress(0.0, text="Rechne die großen Instanzen …")
        todo = [(f, m) for m in C.LARGE_SIZES for f in C.FAMILIES]
        for i, (f, m) in enumerate(todo):
            _sweep(f, m)
            bar.progress((i + 1) / len(todo), text=f"n = {m}, {C.FAMILY_NAMES[f]} fertig")
        st.session_state["large_ok"] = True
        st.rerun()
else:
    by_n = {m: {f: _sweep(f, m) for f in C.FAMILIES} for m in C.LARGE_SIZES}
    st.plotly_chart(build_large(by_n, median_of), width="stretch", key="large_chart")
    table = []
    for m in C.LARGE_SIZES:
        for f in C.FAMILIES:
            s = family_summary(by_n[m][f])
            table.append({"n": m, "Familie": fam(f), "Lücke PB höchstens (Median)": pct(s["gap_pb"], 1), "Lücke Gilmore–Lawler höchstens (Median)": pct(s["gap_glb"], 1), "PB besser als Gilmore–Lawler": f"{s['pb_wins']} von {s['count']}"})
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
    verdict(verdict_large(by_n))
    dens_by_n = {m: density_correlations(pooled(by_n[m])) for m in C.LARGE_SIZES}
    dens_counts = {m: len(pooled(by_n[m])) for m in C.LARGE_SIZES}
    st.markdown("**Hängt die Lücke am Fluss?** Rangkorrelation der Lücke mit der Dichte des Flusses, je Größe über alle 60 Instanzen:")
    st.dataframe(pd.DataFrame([{"Größe": f"n = {m}", "Instanzen": dens_counts[m], "Unsicherheit grob ±": de(spearman_band(dens_counts[m]), 2), **{k: de(r, 2) for k, r in dens_by_n[m].items()}} for m in C.LARGE_SIZES]), hide_index=True, width="stretch")
    verdict(verdict_density(dens_by_n, dens_counts))

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die projizierte Schranke ist eine vereinfachte Eigenherleitung** | Sie folgt der Idee der projizierten Eigenwertschranke, ist aber nicht ihre Originalfassung; die Literatur kennt stärkere Fassungen (etwa mit Verschiebungen der Matrizen). Alle Befunde gelten für diese Fassung. | – |
| **Die sechs Familien sind erfunden** | Echte Platinen oder QAPLIB-Instanzen haben andere Strukturen; „bei dichtem Fluss ist die projizierte Schranke besser“ gilt nur für diese Familien. | – |
| **Das Optimum ist nur bis n = 9 bekannt** | Ab n = 12 steht die beste von 200 steilsten Abstiegen an seiner Stelle; die Lücken sind dann obere Grenzen der wahren Lücke. | Stück 8 (Branch and Bound) |
| **Nichtnegative Flüsse und Abstände** | Gerade weil beide Matrizen nichtnegativ sind, hat jede einen großen positiven Eigenwert, und die gepaarte Summe der Eigenwertschranke wird negativ. Bei anderen Matrizen (etwa mit Vorzeichenwechsel) kann sie brauchbar sein. | – |
| **Wenige Instanzen** | 20 Instanzen je Familie (n = 6, 8) und 10 (n ≥ 12); die Rangkorrelationen schwanken um grob ±0,19 bzw. ±0,28. | – |
| **Nur drei Schranken** | Linearisierung (Stück 5), doppeltstochastische Relaxation (Stück 6) und die semidefinite Schranke (Stück 7) fehlen. | Stück 5 bis 7 |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Die Aufgabe (wie in Stück 1).** Symmetrischer Fluss $F$, symmetrischer Abstand $D$ der Plätze, beide mit Diagonale 0; für die Permutationsmatrix $X$ ($X_{i,p_i} = 1$) sind die Kosten $\mathrm{cost}(p) = \tfrac12 \operatorname{tr}(F X D X^\top)$.

**Eigenwertschranke.** $X$ ist orthogonal. Für symmetrische $F$, $D$ und orthogonales $X$ gilt $\operatorname{tr}(F X D X^\top) \ge \sum_i \lambda_i(F)\,\lambda_{n+1-i}(D)$ (Eigenwerte von $F$ aufsteigend, von $D$ absteigend gepaart). Also
$$\mathrm{EB} = \tfrac12 \sum_i \lambda_i(F)\,\lambda_{n+1-i}(D).$$
Sie braucht nur zwei Eigenwertzerlegungen. Weil $F$ nichtnegativ ist, ist der größte Eigenwert positiv und wird mit dem kleinsten von $D$ (meist negativ) gepaart: die Schranke ist oft negativ.

**Projektion.** Sei $e$ der Einsvektor und $V \in \mathbb{R}^{n \times (n-1)}$ mit orthonormalen Spalten, $V^\top e = 0$. Jede Permutationsmatrix lässt sich schreiben als $X = \tfrac1n e e^\top + V Y V^\top$ mit orthogonalem $Y \in \mathbb{R}^{(n-1)\times(n-1)}$. Mit $u = Fe$, $w = De$, $s_F = e^\top F e$, $s_D = e^\top D e$, $\hat F = V^\top F V$, $\hat D = V^\top D V$ gilt **exakt**
$$\operatorname{tr}(F X D X^\top) = -\frac{s_F s_D}{n^2} + \frac{2}{n}\, u^\top X w + \operatorname{tr}(\hat F\, Y\, \hat D\, Y^\top).$$
Der mittlere Term ist linear in $X$: sein kleinster Wert über Permutationen ist das Skalarprodukt von $u$ (aufsteigend) mit $w$ (absteigend) sortiert. Der letzte Term wird wie bei der Eigenwertschranke durch gepaarte Eigenwerte von $\hat F$ und $\hat D$ abgeschätzt. Die Summe der drei Mindestwerte, halbiert, ist die **projizierte Schranke** (vereinfachte Fassung).

**Gültigkeit.** Jede Anordnung kostet mindestens so viel wie die Summe der Mindestwerte der Terme; die Demo prüft das in jeder Vergleichsinstanz gegen das Optimum bzw. die beste gefundene Lösung und rechnet die Zerlegung an Permutationen nach.

**Lücke.** $(\mathrm{opt} - B)/\mathrm{opt}$ für eine Schranke $B$; ohne bekanntes Optimum steht die beste gefundene Lösung $U \ge \mathrm{opt}$ an dessen Stelle, und wegen $B/U \le B/\mathrm{opt}$ ist $(U - B)/U \ge (\mathrm{opt} - B)/\mathrm{opt}$ eine **obere Grenze** der wahren Lücke.

**Rangkorrelation.** Spearman: Pearson-Korrelation der Ränge (Gleichstände mit mittlerem Rang), Unsicherheit grob $1{,}96 \cdot 1{,}06/\sqrt{N-3}$ (Näherung, nur eine Größenordnung).

**Literatur.** Finke, Burkard, Rendl (1987) für die Eigenwertschranke, Hadley, Rendl, Wolkowicz (1992) für die projizierte Eigenwertschranke, Gilmore (1962) und Lawler (1963) für die Zuordnungsschranke: Angaben aus der Fachliteratur übernommen, hier **nicht nachgelesen**.
Die Herleitung der projizierten Schranke oben ist eine **eigene** Rechnung; die Formeln werden durch Tests (Aufzählung, Zerlegung an Permutationen, `scipy` für die Eigenwerte) geprüft. Die Zahlen dieser Demo sind eigene Rechnungen an eigenen, erfundenen Familien.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
