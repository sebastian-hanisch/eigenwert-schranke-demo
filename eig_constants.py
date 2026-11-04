"""Konstanten: Größen, Raster, Familien der Flussmatrix, feste Reihen der Experimente, Farben und Presets."""

EXACT_SIZES = (6, 8, 9)                               # alle n! Anordnungen werden aufgezählt: das Optimum ist bekannt; n = 9 nur auf Knopfdruck
LARGE_SIZES = (12, 16, 20)                            # zu groß zum Aufzählen: die beste gefundene Lösung ersetzt das Optimum
SIZES = EXACT_SIZES + LARGE_SIZES
DEFAULT_N = 8
SWEEP_SIZES = (6, 8)                                  # Vergleiche mit aufgezähltem Optimum
GRID = {6: (2, 3), 8: (2, 4), 9: (3, 3), 12: (3, 4), 16: (4, 4), 20: (4, 5)}    # Plätze als Raster (Zeilen, Spalten), Abstand = Manhattan
SEED_MAX, DEFAULT_SEED = 999999, 35
SWEEP_SEEDS = tuple(range(100000, 100020))            # feste Instanzen: nie ein einzelner Seed als Befund
LARGE_SEEDS = SWEEP_SEEDS[:10]                        # große Instanzen: zehn feste Instanzen je Familie
STARTS = 200                                          # Zahl der steilsten Abstiege aus Zufallsstarts für die beste gefundene Lösung

FAMILIES = ("netlist", "uniform", "sparse", "cluster", "star", "skewed")
DEFAULT_FAMILY = "netlist"
FAMILY_NAMES = {"netlist": "Netzliste", "uniform": "Gleichverteilt", "sparse": "Dünn besetzt", "cluster": "Zwei Cluster", "star": "Stern", "skewed": "Wenige starke Flüsse"}
FAMILY_HELP = {
    "netlist": "Netzliste aus Dreiergruppen (IC, Kondensator, Widerstand), jedes Paar in einem Netz zählt Fluss 1: die Familie aus Stück 1.",
    "uniform": "Jedes Paar von Bauteilen hat einen zufälligen Fluss von 0 bis 9, gleichverteilt: dichte Matrix ohne Struktur.",
    "sparse": "Nur drei von zehn Paaren haben Fluss (1 bis 9), die übrigen keinen: dünn besetzt ohne weitere Struktur.",
    "cluster": "Zwei Gruppen von Bauteilen: innerhalb einer Gruppe starker Fluss (8 bis 10), zwischen den Gruppen schwacher (0 bis 2).",
    "star": "Ein Bauteil (die Nabe) hat starken Fluss (1 bis 9) zu allen anderen; zwischen den übrigen höchstens 1.",
    "skewed": "Die Flüsse sind schief verteilt: viele Paare ohne oder mit kleinem Fluss, wenige mit sehr großem (bis 60).",
}

# Netzliste (wie in Stück 1 und 2)
N_NETS_PER_COMPONENT = 1.5
NET_SIZES = (2, 3, 4)
NET_WEIGHTS = (5, 3, 2)
LOCAL_NET_PERCENT = 70
GROUP = 3
# Dünn besetzt, Cluster, Stern, schief (wie in Stück 2)
SPARSE_PERCENT = 30
UNIFORM_MAX = 9
CLUSTER_STRONG, CLUSTER_WEAK_MAX = 8, 2
SKEW_VALUES = (0, 1, 2, 4, 8, 16, 32, 60)
SKEW_WEIGHTS = (45, 22, 13, 8, 6, 3, 2, 1)           # ganzzahlige Gewichte (Summe 100), schweres Ende

HAND_FLOWS = ((0, 1, 5), (1, 2, 1))                   # Handbeispiel 1: drei Bauteile auf einer Zeile, Fluss A–B 5, B–C 1
HAND_GRID = (1, 3)
HAND_GAP_FLOWS = ((0, 2, 1), (0, 3, 1), (1, 2, 1), (1, 3, 1))   # Handbeispiel 3: vier Bauteile auf einer Zeile, A und B gegen C und D
HAND_GAP_GRID = (1, 4)
HAND_GRID_FLOWS = ((0, 1, 3), (0, 2, 1), (1, 3, 2), (2, 3, 4))  # Handbeispiel 2 (aus Stück 1): Raster 2×2
HAND_GRID_GRID = (2, 2)

COLORS = {"best": "#2F6B65", "local": "#c0504d", "other": "rgba(150,150,150,0.6)", "line": "#4c78a8", "ok": "#2F6B65", "warn": "#b08a2e", "bad": "#b34a4a",
          "family": {"netlist": "#4c78a8", "uniform": "#9c755f", "sparse": "#b08a2e", "cluster": "#2F6B65", "star": "#8e6bb0", "skewed": "#c0504d"}}

_BASE = {"family": DEFAULT_FAMILY, "n": DEFAULT_N, "seed": DEFAULT_SEED}
PRESETS = {
    "Standardfall: Netzliste, acht Bauteile": dict(_BASE),
    "Gleichverteilte Flüsse": dict(_BASE, family="uniform"),
    "Dünn besetzt": dict(_BASE, family="sparse"),
    "Zwei Cluster": dict(_BASE, family="cluster"),
    "Stern: eine Nabe": dict(_BASE, family="star"),
    "Zwölf Bauteile: ohne Aufzählung": dict(_BASE, n=12),
    "Zwanzig Bauteile": dict(_BASE, family="uniform", n=20),
    "Sechs Bauteile": dict(_BASE, n=6),
}
PRESET_HELP = {
    "Standardfall: Netzliste, acht Bauteile": "Die Familie aus Stück 1 mit acht Bauteilen auf dem 2×4-Raster: 40 320 Anordnungen, alle werden gerechnet; die Schranke wird gegen das aufgezählte Optimum gemessen.",
    "Gleichverteilte Flüsse": "Jedes Paar hat einen zufälligen Fluss von 0 bis 9: die dichte Matrix ohne Struktur.",
    "Dünn besetzt": "Nur ein Teil der Paare hat überhaupt Fluss: viele Nullen in der Matrix.",
    "Zwei Cluster": "Zwei Gruppen mit starkem Fluss innerhalb und schwachem zwischen den Gruppen.",
    "Stern: eine Nabe": "Ein Bauteil hängt mit allen anderen stark zusammen, die übrigen kaum untereinander.",
    "Zwölf Bauteile: ohne Aufzählung": "Zwölf Bauteile auf einem 3×4-Raster: 479 001 600 Anordnungen, nicht aufzählbar. Die Schranke ist trotzdem sofort da; verglichen wird mit der besten gefundenen Lösung.",
    "Zwanzig Bauteile": "Zwanzig Bauteile auf einem 4×5-Raster, gleichverteilte Flüsse: weit jenseits jeder Aufzählung.",
    "Sechs Bauteile": "Sechs Bauteile auf dem 2×3-Raster: 720 Anordnungen, die Rechnung dauert Sekundenbruchteile.",
}


def n_text(n: int) -> str:
    """Anzahl mit Leerzeichen als Tausendertrenner: 362880 → „362 880“."""
    return f"{n:,}".replace(",", " ")


assert set(PRESETS) == set(PRESET_HELP) and set(FAMILY_NAMES) == set(FAMILIES) == set(FAMILY_HELP)
assert all(p["n"] in SIZES and p["family"] in FAMILIES and 0 <= p["seed"] <= SEED_MAX for p in PRESETS.values())
assert all(r * c == n for n, (r, c) in GRID.items()) and sum(SKEW_WEIGHTS) == 100 and set(GRID) == set(SIZES)
