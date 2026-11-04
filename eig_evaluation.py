"""Kennzahlen, Vergleiche über Instanzen und Verdikte. Jedes Verdikt ist eine reine Funktion (Art, Text); die App ruft st.<Art>(Text).

Alle Aussagen gelten für die gerechneten Instanzen, die sechs erfundenen Familien (siehe README) und die **vereinfachte** projizierte Schranke dieser Demo, nicht für echte Platinen, QAPLIB oder die Originalfassung der Literatur; Zeiten stehen nie in Verdikten.
Lücke bei aufgezähltem Optimum: (Optimum − Schranke) / Optimum; bei großen Instanzen ersetzt die beste gefundene Lösung das Optimum und die Lücke ist eine **obere** Grenze der wahren Lücke."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import permutations

import numpy as np

import eig_bound as B
import eig_constants as C
import eig_cost as K
import eig_landscape as L
import eig_spectral as S
from eig_instance import hand_instance, make_instance

TOL = 1e-7      # Rechengenauigkeit der Eigenwertschranken beim Vergleich mit dem Optimum


@dataclass(frozen=True)
class Settings:
    family: str = C.DEFAULT_FAMILY
    n: int = C.DEFAULT_N
    seed: int = C.DEFAULT_SEED


# ----------------------------------------------------------------------------- Formatierung

def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def pct(x: float, digits: int = 0) -> str:
    return de(100.0 * x, digits) + " %"


def sgn(x: float, digits: int = 2) -> str:
    """Mit Vorzeichen: +0,44 / −0,13."""
    return ("+" if x >= 0 else "") + de(x, digits)


def fam(name: str) -> str:
    return C.FAMILY_NAMES.get(name, name)


def num(x: float) -> str:
    """Ganze Zahl ohne Nachkommastelle, sonst eine Stelle: 27 / 27,5 / −12,1."""
    return str(int(round(x))) if abs(x - round(x)) < 1e-9 else de(x, 1)


def rel(a: float, b: float) -> float:
    """(a − b) / a; NaN, wenn a = 0."""
    return (a - b) / a if a else float("nan")


# ----------------------------------------------------------------------------- Kennzahlen der Flussmatrix und Statistik (wie in Stück 3)

def pair_values(F) -> np.ndarray:
    return np.asarray(F)[np.triu_indices(len(F), 1)].astype(float)


def flow_dominance(F) -> float:
    v = pair_values(F)
    return float(100.0 * v.std() / v.mean()) if v.size and v.mean() > 0 else float("nan")


def density(F) -> float:
    v = pair_values(F)
    return float((v > 0).mean()) if v.size else float("nan")


def ranks(x) -> np.ndarray:
    """Ränge 1..N, Gleichstände erhalten den mittleren Rang."""
    x = np.asarray(x, dtype=float)
    order = x.argsort(kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(1, len(x) + 1)
    for u in np.unique(x):
        m = x == u
        r[m] = r[m].mean()
    return r


def spearman(a, b) -> float:
    """Rangkorrelation (Pearson-Korrelation der Ränge); NaN, wenn eine Seite konstant ist, Werte fehlen oder weniger als zwei vorliegen."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError("gleich lange Reihen")
    if len(a) < 2 or np.isnan(a).any() or np.isnan(b).any():
        return float("nan")
    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def spearman_band(count: int) -> float:
    """Grobe 95-%-Unsicherheit einer Rangkorrelation bei `count` Instanzen: 1,96 · 1,06 / √(count − 3) (nur eine Größenordnung)."""
    return 1.96 * 1.06 / math.sqrt(count - 3) if count > 3 else float("nan")


def quartiles(x) -> tuple:
    x = np.asarray(x, dtype=float)
    return float(np.percentile(x, 25)), float(np.median(x)), float(np.percentile(x, 75))


def median_of(rows: list, key: str) -> float:
    return float(np.median([r[key] for r in rows]))


# ----------------------------------------------------------------------------- Zeilen für die Vergleiche

def landscape_of(s: Settings) -> L.Landscape:
    return L.landscape(make_instance(s.family, s.n, s.seed))


def _row(family: str, n: int, seed: int, ref: int, exact: bool, F, D) -> dict:
    sp = S.spectral_bounds(F, D)
    glb = B.gilmore_lawler(F, D).glb
    return {"family": family, "n": n, "seed": seed, "ref": ref, "eb": sp.eb, "pb": sp.pb, "glb": glb, "gap_eb": rel(ref, sp.eb), "gap_pb": rel(ref, sp.pb), "gap_glb": rel(ref, glb),
            "eb_negative": sp.eb < 0, "pb_ge_eb": sp.pb >= sp.eb - TOL, "pb_wins": sp.pb > glb + TOL, "valid": max(sp.eb, sp.pb) <= ref + TOL and glb <= ref + 1e-9, "dens": density(F), "fd": flow_dominance(F), "exact": exact}


def exact_row(family: str, n: int, seed: int) -> dict:
    """Eine Instanz mit aufgezähltem Optimum."""
    inst = make_instance(family, n, seed)
    return _row(family, n, seed, L.landscape(inst).opt, True, inst.F, inst.D)


def large_row(family: str, n: int, seed: int, starts: int = C.STARTS) -> dict:
    """Eine große Instanz: die beste gefundene Lösung (obere Schranke) ersetzt das Optimum."""
    inst = make_instance(family, n, seed)
    ub, _ = B.best_found(inst.F, inst.D, seed, starts)
    return _row(family, n, seed, ub, False, inst.F, inst.D)


def sweep(family: str, n: int, seeds=None) -> list:
    """Die festen Instanzen einer Familie: bei n ≤ 9 mit aufgezähltem Optimum, sonst mit der besten gefundenen Lösung."""
    if n in C.EXACT_SIZES:
        return [exact_row(family, n, s) for s in (seeds if seeds is not None else C.SWEEP_SEEDS)]
    return [large_row(family, n, s) for s in (seeds if seeds is not None else C.LARGE_SEEDS)]


def pooled(by_family: dict) -> list:
    return [r for f in C.FAMILIES if f in by_family for r in by_family[f]]


def family_summary(rows: list) -> dict:
    """Mediane der drei Lücken und Zählungen einer Familie."""
    return {"count": len(rows), "gap_eb": median_of(rows, "gap_eb"), "gap_pb": median_of(rows, "gap_pb"), "gap_glb": median_of(rows, "gap_glb"),
            "pb_min": min(r["gap_pb"] for r in rows), "pb_max": max(r["gap_pb"] for r in rows), "eb_negative": sum(1 for r in rows if r["eb_negative"]), "pb_ge_eb": sum(1 for r in rows if r["pb_ge_eb"]),
            "pb_wins": sum(1 for r in rows if r["pb_wins"]), "valid": all(r["valid"] for r in rows), "dens": median_of(rows, "dens")}


def order_correlations(rows: list) -> dict:
    """Rangkorrelation der Lücken der Schranken untereinander: {"EB~GLB", "PB~GLB", "EB~PB"}."""
    g = lambda k: [r[k] for r in rows]
    return {"EB~GLB": spearman(g("gap_eb"), g("gap_glb")), "PB~GLB": spearman(g("gap_pb"), g("gap_glb")), "EB~PB": spearman(g("gap_eb"), g("gap_pb"))}


def density_correlations(rows: list) -> dict:
    """Rangkorrelation der Lücke jeder Schranke mit der Dichte des Flusses: {"EB", "PB", "GLB"}."""
    d = [r["dens"] for r in rows]
    return {"EB": spearman(d, [r["gap_eb"] for r in rows]), "PB": spearman(d, [r["gap_pb"] for r in rows]), "GLB": spearman(d, [r["gap_glb"] for r in rows])}


# ----------------------------------------------------------------------------- Experiment 1 und 2: eine Instanz

def instance_data(s: Settings, ls: L.Landscape | None = None) -> dict:
    """Spektrale Schranken, Gilmore–Lawler-Schranke und Vergleichswert einer Instanz (n ≤ 9: aufgezähltes Optimum, sonst beste von 200 Abstiegen)."""
    inst = make_instance(s.family, s.n, s.seed)
    sp = S.spectral_bounds(inst.F, inst.D)
    glb = B.gilmore_lawler(inst.F, inst.D).glb
    out = {"inst": inst, "spectral": sp, "glb": glb, "exact": s.n in C.EXACT_SIZES, "stats": {"fd": flow_dominance(inst.F), "dens": density(inst.F)}}
    if out["exact"]:
        ls = ls if ls is not None else L.landscape(inst)
        out.update(ref=ls.opt, best=[int(x) for x in ls.optima[0]], perms=ls.perms)
    else:
        ub, p = B.best_found(inst.F, inst.D, s.seed)
        out.update(ref=ub, best=[int(x) for x in p], perms=math.factorial(s.n))
    out.update(gap_eb=rel(out["ref"], sp.eb), gap_pb=rel(out["ref"], sp.pb), gap_glb=rel(out["ref"], glb))
    return out


def hand_data() -> list:
    """Die drei Handbeispiele der README: (Instanz, Schranken, Optimum)."""
    out = []
    for name, flows, (rows, cols) in (("Drei Bauteile auf einer Zeile", C.HAND_FLOWS, C.HAND_GRID), ("Vier Bauteile auf dem 2×2-Raster", C.HAND_GRID_FLOWS, C.HAND_GRID_GRID),
                                      ("Vier Bauteile auf einer Zeile (mit Lücke)", C.HAND_GAP_FLOWS, C.HAND_GAP_GRID)):
        inst = hand_instance(flows, rows, cols)
        out.append({"name": name, "inst": inst, "spectral": S.spectral_bounds(inst.F, inst.D), "glb": B.gilmore_lawler(inst.F, inst.D).glb,
                    "opt": min(K.cost(p, inst.F, inst.D) for p in permutations(range(inst.n)))})
    return out


# ----------------------------------------------------------------------------- Verdikte

def _head(s: Settings, v: dict) -> str:
    inst = v["inst"]
    return f"Familie **{fam(s.family)}**, {s.n} Bauteile, Raster {inst.rows}×{inst.cols}, Seed {s.seed}: "


def _refname(v: dict) -> str:
    return "das aufgezählte Optimum" if v["exact"] else f"die beste gefundene Lösung (beste von {C.STARTS} Abstiegen)"


def verdict_eb(s: Settings, v: dict):
    sp = v["spectral"]
    negative = sp.eb < 0
    tail = (f"Die Eigenwertschranke liegt unter 0 und ist damit schwächer als die triviale Schranke 0 (Kosten sind nie negativ): der große positive Eigenwert des nichtnegativen Flusses wird mit dem kleinsten Eigenwert des Abstands gepaart und zieht die Summe nach unten. "
            if negative else "Die Eigenwertschranke liegt über 0. ")
    return ("warning" if negative else "info"), (_head(s, v) + f"Die Eigenwertschranke ist **{num(sp.eb)}**; {_refname(v)} ist {v['ref']}, die Gilmore–Lawler-Schranke {num(v['glb'])}. " + tail +
                                               f"Der Rechenaufwand: zwei Eigenwertzerlegungen.")


def verdict_pb(s: Settings, v: dict):
    sp = v["spectral"]
    better = sp.pb > v["glb"] + TOL
    return ("success" if better else "info"), (_head(s, v) + f"Die projizierte Schranke ist **{num(sp.pb)}** (Terme: konstant {num(sp.const)}, linear {num(sp.linear)}, quadratisch {num(sp.quad)}), gegen die Eigenwertschranke {num(sp.eb)}, die Gilmore–Lawler-Schranke {num(v['glb'])} "
                                              f"und {_refname(v)} {v['ref']}. Lücke {'' if v['exact'] else 'höchstens '}**{pct(v['gap_pb'], 1)}** (Gilmore–Lawler {pct(v['gap_glb'], 1)}). "
                                              f"{'Hier ist die projizierte Schranke die bessere von beiden.' if better else 'Hier ist die Gilmore–Lawler-Schranke mindestens so gut wie die projizierte.'}")


def verdict_gaps(n: int, by_family: dict):
    summ = {f: family_summary(r) for f, r in by_family.items() if r}
    rows = pooled(by_family)
    total = len(rows)
    valid = all(x["valid"] for x in summ.values())
    neg = sum(x["eb_negative"] for x in summ.values())
    ge = sum(x["pb_ge_eb"] for x in summ.values())
    parts = "; ".join(f"{fam(f)} PB {pct(x['gap_pb'], 1)}, GLB {pct(x['gap_glb'], 1)}" for f, x in summ.items())
    return ("success" if valid else "error"), (f"Lücke zum aufgezählten Optimum bei n = {n}, Median über {total} Instanzen: Eigenwertschranke {pct(median_of(rows, 'gap_eb'))} (sie liegt in {neg} von {total} Instanzen unter 0), projizierte Schranke **{pct(median_of(rows, 'gap_pb'), 1)}**, "
                                               f"Gilmore–Lawler **{pct(median_of(rows, 'gap_glb'), 1)}**. Je Familie: {parts}. Die projizierte Schranke liegt in {ge} von {total} Instanzen mindestens bei der Eigenwertschranke. "
                                               f"{'Alle Schranken lagen in jeder Instanz höchstens beim Optimum (gültig).' if valid else 'ACHTUNG: in mindestens einer Instanz lag eine Schranke über dem Optimum.'}")


def verdict_winner(n: int, by_family: dict):
    summ = {f: family_summary(r) for f, r in by_family.items() if r}
    total = sum(x["count"] for x in summ.values())
    wins = sum(x["pb_wins"] for x in summ.values())
    always = [f for f, x in summ.items() if x["pb_wins"] == x["count"]]
    never = [f for f, x in summ.items() if x["pb_wins"] == 0]
    parts = "; ".join(f"{fam(f)} {x['pb_wins']} von {x['count']}" for f, x in summ.items())
    tail = ("Immer besser ist die projizierte Schranke bei " + ", ".join(fam(f) for f in always) + ". " if always else "In keiner Familie ist die projizierte Schranke in jeder Instanz besser. ")
    tail += ("Nie besser ist sie bei " + ", ".join(fam(f) for f in never) + "." if never else "In jeder Familie ist sie mindestens einmal besser.")
    return "info", f"Die projizierte Schranke ist größer als die Gilmore–Lawler-Schranke in {wins} von {total} Instanzen bei n = {n}: {parts}. {tail} Keine der beiden Schranken gewinnt überall."


def verdict_order(corr_by_n: dict, counts: dict):
    """corr_by_n: {n: order_correlations(rows)}; counts: {n: Zahl der Instanzen}. Fragt, ob die Schranken Instanzen gleich ordnen (Rangkorrelation der Lücken)."""
    sizes = sorted(corr_by_n)
    band = {n: spearman_band(counts[n]) for n in sizes}
    parts = "; ".join(f"n = {n}: PB~GLB {sgn(corr_by_n[n]['PB~GLB'])}, EB~GLB {sgn(corr_by_n[n]['EB~GLB'])} ({counts[n]} Instanzen, Unsicherheit grob ±{de(band[n], 2)})" for n in sizes)
    clear = [n for n in sizes if abs(corr_by_n[n]["PB~GLB"]) > band[n]]
    if not clear:
        return "warning", f"Rangkorrelation der Lücken, jeweils über alle Familien: {parts}. Die Lücke der projizierten Schranke sagt die der Gilmore–Lawler-Schranke in dieser Größe nicht voraus: die Schranken ordnen die Instanzen verschieden."
    pos = [n for n in clear if corr_by_n[n]["PB~GLB"] > 0]
    claim = "Sie ordnen die Instanzen gleichsinnig" if len(pos) == len(clear) else ("Sie ordnen die Instanzen gegensinnig" if not pos else "Das Vorzeichen ist uneinheitlich")
    return "info", f"Rangkorrelation der Lücken, jeweils über alle Familien: {parts}. Außerhalb der Unsicherheit liegt PB~GLB bei {', '.join(f'n = {n}' for n in clear)}: {claim}."


def verdict_large(by_n: dict):
    """by_n: {n: {Familie: Zeilen}} der großen Instanzen."""
    sizes = sorted(by_n)
    med = {n: {k: median_of(pooled(by_n[n]), k) for k in ("gap_pb", "gap_glb")} for n in sizes}
    wins = {n: sum(1 for r in pooled(by_n[n]) if r["pb_wins"]) for n in sizes}
    totals = {n: len(pooled(by_n[n])) for n in sizes}
    parts = "; ".join(f"n = {n}: PB {pct(med[n]['gap_pb'], 1)}, GLB {pct(med[n]['gap_glb'], 1)} (PB besser in {wins[n]} von {totals[n]})" for n in sizes)
    grows = len(sizes) > 1 and all(med[sizes[i + 1]]["gap_glb"] > med[sizes[i]]["gap_glb"] for i in range(len(sizes) - 1))
    claim = "Die Lücke der Gilmore–Lawler-Schranke wächst mit n" if grows else "Die Lücke der Gilmore–Lawler-Schranke wächst in dieser Messung nicht gleichmäßig mit n"
    return "info", (f"Schranken gegen die beste gefundene Lösung (beste von {C.STARTS} Abstiegen), Median der Lücke über alle Familien: {parts}. {claim}. Die Zahlen sind obere Grenzen der wahren Lücke: das Optimum ist unbekannt.")


def verdict_density(corr_by_n: dict, counts: dict):
    """corr_by_n: {n: density_correlations(rows)}: Rangkorrelation der Lücke jeder Schranke mit der Dichte."""
    sizes = sorted(corr_by_n)
    parts = "; ".join(f"n = {n}: PB {sgn(corr_by_n[n]['PB'])}, GLB {sgn(corr_by_n[n]['GLB'])} ({counts[n]} Instanzen, Unsicherheit grob ±{de(spearman_band(counts[n]), 2)})" for n in sizes)
    clear_pb = [n for n in sizes if abs(corr_by_n[n]["PB"]) > spearman_band(counts[n])]
    neg = [n for n in clear_pb if corr_by_n[n]["PB"] < 0]
    if clear_pb and len(neg) == len(clear_pb):
        tail = f"Bei dichterem Fluss ist die Lücke der projizierten Schranke kleiner (außerhalb der Unsicherheit bei {', '.join(f'n = {n}' for n in clear_pb)}); über alle Familien gepoolt."
    elif clear_pb:
        tail = f"Das Vorzeichen bei der projizierten Schranke ist uneinheitlich (außerhalb der Unsicherheit bei {', '.join(f'n = {n}' for n in clear_pb)})."
    else:
        tail = "Für die projizierte Schranke liegt die Rangkorrelation in keiner Größe außerhalb der Unsicherheit."
    return "info", f"Rangkorrelation (Spearman) der Lücke mit der Dichte des Flusses: {parts}. {tail}"


def verdict_core(s: Settings, v: dict):
    return "info", (f"An einer Instanz sieht man drei Zahlen: Eigenwertschranke {num(v['spectral'].eb)}, projizierte Schranke {num(v['spectral'].pb)}, Gilmore–Lawler {num(v['glb'])}, bei {'Optimum' if v['exact'] else 'bester gefundener Lösung'} {v['ref']}. "
                    f"Welche Schranke die bessere ist, hängt an der Familie und an n (Experiment 3 bis 5): keine gewinnt überall.")
