# Eigenwertschranke und Projektion – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-eigenwert-schranke-demo.streamlit.app/)** -->

Stück 4 von 15 der **Konzepte-Reihe „Anordnen: relaxieren und runden“** für die Website „Sebastian Hanisch – Operations Research und Machine Learning“ (die Konzepte-Seite der Reihe gibt es noch nicht). Stück 3, [Die Gilmore–Lawler-Schranke](https://github.com/sebastian-hanisch/gilmore-lawler-demo), beweist eine untere Schranke mit einer linearen **Zuordnung**. Dieses Stück macht dieselbe Aufgabe mit einem anderen Vehikel, dem **Spektrum**: Eigenwerte von Flussmatrix und Abstandsmatrix. Kante der Reihe: *Kontrast*.

## Warum dieses Problem

Permutationsmatrizen sind orthogonale Matrizen, und für orthogonale Matrizen lässt sich die Kostensumme durch gepaarte Eigenwerte nach unten abschätzen: die **Eigenwertschranke** (EB), ganz ohne Zuordnung. Die **projizierte Schranke** (PB) rechnet zusätzlich in dem Raum, der zum Einsvektor senkrecht steht, und schätzt einen linearen Teil getrennt ab. Jede der beiden Schranken braucht zwei Eigenwertzerlegungen. Die Kernfrage: **Was sagt das Spektrum, und was die Projektion, und schlagen sie die Zuordnungsschranke aus Stück 3?**

Die Antwort dieser Messung: Die Eigenwertschranke ist an diesen Instanzen **wertlos** (in allen 420 gerechneten kleiner als null), die Projektion macht sie nützlich, und ob sie besser ist als die Gilmore–Lawler-Schranke, **hängt an der Familie** und an n: bei Cluster-Flüssen (und gleichverteilten ab n = 12) ist die projizierte Schranke meist besser, bei Netzliste und wenigen starken Flüssen in keiner Instanz, bei dünn besetzten Flüssen nur in einem Einzelfall. Das sind **Befunde an sechs erfundenen Familien und an einer vereinfachten, selbst hergeleiteten projizierten Schranke**, keine Sätze.

## Modell

- **Aufgabe** (wie in Stück 1 bis 3): `n` Bauteile auf `n` Plätze, Kosten gleich der Summe über alle Paare von Fluss mal Abstand, `cost(p) = ½ · tr(F X D Xᵀ)` mit der Permutationsmatrix `X`. Kern (Kosten, Aufzählung, Zufall, sechs Familien der Flussmatrix, Gilmore–Lawler-Schranke mit eigener Ungarischer Methode, beste gefundene Lösung) als Kopie aus Stück 3 (`eig_cost.py`, `eig_landscape.py`, `eig_rng.py`, `eig_instance.py`, `eig_assignment.py`, `eig_bound.py`; Tests vergleichen die Instanzen mit Stück 1 bis 3, wenn die Schwester-Repos nebenan liegen). Die Familien sind **eigene, einfache Erfindungen**, keine echten Platinen und keine QAPLIB-Instanzen; Plätze liegen auf einem Raster (2×3, 2×4, 3×3, 3×4, 4×4, 4×5, Manhattan-Abstand).
- **Eigenwertschranke** (`eig_spectral.py`): für orthogonales `X` und symmetrische `F`, `D` gilt `tr(F X D Xᵀ) ≥ Σ_i λ_i(F) · λ_{n+1−i}(D)` (`F` aufsteigend, `D` absteigend gepaart); EB ist die Hälfte dieser Summe. Weil `F` und `D` die Diagonale 0 haben (Spur 0), ist die gepaarte Summe nicht positiv, unabhängig vom Vorzeichen der Einträge (entgegengesetzt sortierte Folgen mit Summe 0, Chebyshev-Summenungleichung); bei nichtnegativen Kosten ist die Schranke damit schwächer als die triviale Schranke 0. Der große positive Eigenwert von `F` wird dabei mit dem kleinsten von `D` gepaart und zieht die Summe nach unten. Stärkere Fassungen aus der Literatur verschieben die Matrizen vorher (hier nicht umgesetzt, nicht nachgelesen).
- **Projizierte Schranke, vereinfachte Fassung:** jede Permutationsmatrix ist `X = e eᵀ/n + V Y Vᵀ` (`V`: Basis senkrecht zum Einsvektor `e`, `Y` orthogonal). Mit `u = F e`, `w = D e`, `s_F = eᵀ F e`, `s_D = eᵀ D e`, `F̂ = Vᵀ F V`, `D̂ = Vᵀ D V` gilt **exakt** `tr(F X D Xᵀ) = −s_F s_D/n² + (2/n) · uᵀ X w + tr(F̂ Y D̂ Yᵀ)`. Der mittlere Term ist linear in `X` (kleinster Wert: `u` aufsteigend gegen `w` absteigend), der letzte wird wie bei der EB durch gepaarte Eigenwerte von `F̂` und `D̂` abgeschätzt; PB ist die Hälfte der Summe der drei Mindestwerte. Das ist eine **eigene Herleitung** der Idee der projizierten Eigenwertschranke (Hadley, Rendl, Wolkowicz 1992), nicht ihre Originalfassung; die Zerlegung wird im Test an Permutationen nachgerechnet, die Gültigkeit an jeder Vergleichsinstanz.
- **Vergleichswert und Lücke:** bis `n = 9` das aufgezählte Optimum, Lücke `(Optimum − Schranke) / Optimum`. Ab `n = 12` ersetzt die **beste von 200 steilsten Abstiegen** das Optimum; die Lücke ist dann eine **obere Grenze** der wahren Lücke.
- **Experimente:** (1) die Eigenwertschranke (Spektren, gepaart, Summanden); (2) die Projektion (drei Terme, projizierte Spektren, Probe der Zerlegung); (3) die Lücken der Schranken je Familie über 20 feste Instanzen; (4) welche Schranke die bessere ist und ob sie Instanzen gleich ordnen; (5) jenseits der Aufzählung (`n` = 12, 16, 20). Formeln im 📐-Expander der App.

## Methodik

Experiment 1 und 2 rechnen **eine** Instanz (n = 6 oder 8 sofort, n = 9 auf Knopfdruck, rund eine Viertelminute; n = 12, 16, 20 sofort). Experiment 3 und 4 rechnen auf Knopfdruck **feste Instanzen**: je Familie die Seeds 100000 bis 100019, bei n = 6 in Sekunden, bei n = 8 rund vier Minuten (Zeiten hängen vom Rechner ab und werden nicht getestet); Experiment 5 je Familie die Seeds 100000 bis 100009 für n = 12, 16 und 20 (Sekunden). Alle Aussagen laufen über feste Seeds, nie über einen einzelnen. Eigenwerte kommen aus `numpy.linalg.eigvalsh`.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` (deterministisch bis auf die letzte Stelle der Eigenwerte; die Zahlen sind auf die genannte Rundung genau). Reihenfolge der Familien: Netzliste, gleichverteilt, dünn besetzt, zwei Cluster, Stern, wenige starke Flüsse.

| Frage | Befund |
|---|---|
| Handbeispiele (EB / PB / Gilmore–Lawler / Optimum) | Drei Bauteile auf einer Zeile (Fluss A–B 5, B–C 1): −12,06 / 5,30 / 6 / 6 (Terme: −5,33 konstant, 10,00 linear, 0,63 quadratisch); Raster 2×2: −17,30 / 9,88 / 10 / 10; vier Bauteile auf einer Zeile (Fluss A–C, A–D, B–C, B–D je 1): −8,58 / **5,59** / 5 / 6, hier ist die PB größer als die Gilmore–Lawler-Schranke |
| Standardinstanz (Netzliste, n = 8, Seed 35) | Optimum 31; Gilmore–Lawler 27 (Lücke 12,9 %); Eigenwertschranke −77,1; PB 13,7 (Lücke 55,9 %; Terme −43,8 / 82,5 / −25,1) |
| Eigenwertschranke | in allen **420** Instanzen (120 bei n = 6, 120 bei n = 8, je 60 bei n = 12, 16, 20) kleiner als null; Lücke im Median 279 / 317 / 341 / 332 / 330 % (n = 6 / 8 / 12 / 16 / 20); die PB liegt in allen 420 mindestens bei ihr; alle Schranken waren in jeder Instanz höchstens beim Optimum bzw. der besten gefundenen Lösung |
| Lücke zum aufgezählten Optimum, n = 6 (Median in %), PB | 13,1 / 6,7 / 41,2 / 14,4 / 15,4 / 36,7 (Spannen 1,8 bis 138,9); Gilmore–Lawler 7,3 / 5,2 / 10,3 / 17,0 / 7,7 / 3,4; über alle 120 Instanzen PB **15,4 %**, GLB **7,3 %** |
| n = 8, PB | 20,1 / 8,5 / 46,7 / 9,4 / 12,9 / 74,1 (Spannen 3,2 bis 144,3); Gilmore–Lawler 9,2 / 7,9 / 6,9 / 9,7 / 13,3 / 6,1; über alle 120 Instanzen PB **14,7 %**, GLB **8,9 %** |
| PB größer als Gilmore–Lawler (Zahl der Instanzen je Familie) | n = 6: 0 / 4 / 0 / 20 / 0 / 0; n = 8: 0 / 9 / 0 / 14 / 11 / 0; n = 12: 0 / 10 / 0 / 10 / 6 / 0; n = 16: 0 / 10 / 0 / 10 / 9 / 0; n = 20: 0 / 10 / 1 / 10 / 8 / 0; insgesamt 24 / 34 / 26 / 29 / 29 Instanzen (von 120 bzw. 60): **keine Schranke gewinnt überall** |
| Lücke gegen die beste gefundene Lösung (obere Grenze), PB, Median in % | n = 12: 27,4 / 8,1 / 41,2 / 7,6 / 13,9 / 78,9; n = 16: 36,8 / 7,9 / 36,8 / 6,2 / 15,1 / 83,4; n = 20: 43,7 / 8,0 / 38,5 / 9,3 / 16,6 / 82,4; über alle Familien PB 17,4 / 23,0 / 25,9 %, Gilmore–Lawler 12,3 / 16,0 / 19,8 %: bei gleichverteilten Flüssen bleibt die PB-Lücke bei rund 8 %, die Gilmore–Lawler-Lücke wächst von 10,2 auf 15,1 % |
| Rangkorrelation der Lücken untereinander (Spearman, über alle Familien; 120 Instanzen ±0,19, 60 Instanzen ±0,28), n = 6 / 8 / 12 / 16 / 20 | PB~GLB −0,01 / −0,28 / +0,41 / +0,44 / +0,71; EB~GLB −0,29 / −0,20 / +0,56 / +0,56 / +0,71; EB~PB +0,84 / +0,90 / +0,91 / +0,87 / +0,94: bei kleinen Instanzen sagt die Lücke der einen Schranke die der anderen nicht voraus (n = 6) oder nur gegensinnig (n = 8) |
| Rangkorrelation der Lücke mit der Dichte des Flusses, n = 6 / 8 / 12 / 16 / 20 | PB −0,78 / −0,83 / −0,70 / −0,70 / −0,77; EB −0,85 / −0,88 / −0,72 / −0,69 / −0,72; Gilmore–Lawler +0,07 / +0,05 / −0,68 / −0,71 / −0,78: dichterer Fluss, engere projizierte Schranke (über alle Familien gepoolt) |
| Orakel | Eigenwerte gleich `scipy.linalg.eigh` (auch die der projizierten Matrizen); Zuordnung gleich `scipy.optimize.linear_sum_assignment`; Rangkorrelation gleich `scipy.stats.spearmanr`; EB und PB nie über dem aufgezählten Optimum (an 48 weiteren Instanzen) |

**Was die Zahlen nicht sagen:** Die Familien sind erfunden; eine andere Wahl der Parameter kann die Rangfolge der Schranken ändern. Die PB ist eine **vereinfachte Eigenherleitung**: die Originalliteratur kennt stärkere Fassungen (etwa mit Verschiebungen der Matrizen, die die Eigenwertschranke verbessern); ob die Befunde für sie gelten, ist nicht untersucht. Die Lücken ab n = 12 sind obere Grenzen. Dass die PB bei dichtem Fluss besser ist, ist hier über alle Familien gepoolt gemessen; in einer Prüfung innerhalb mehrerer Familien (Netzliste, dünn besetzt, Cluster) zeigte sich dieselbe Richtung, in anderen kaum (nicht Teil der Tests).

## Ehrliche Grenzen

- **Vereinfachte projizierte Schranke:** eigene Herleitung, nicht die Originalfassung; keine Reduktionen oder Verschiebungen, keine Verstärkung (Anstreicher–Brixius), keine Schnittebenen.
- **Erfundene Familien:** keine echten Platinen, keine QAPLIB-Instanzen.
- **Diagonale 0 (Spur 0):** sie macht die gepaarte Summe der Eigenwertschranke nicht positiv, unabhängig vom Vorzeichen der Einträge; bei nichtnegativen Kosten ist die Schranke damit schwächer als die triviale Schranke 0. Stärkere Fassungen der Literatur verschieben die Matrizen vorher (hier nicht umgesetzt).
- **Optimum nur bis n = 9:** ab n = 12 steht die beste von 200 Abstiegen an seiner Stelle; n = 9 nur als Einzelinstanz, nicht im Vergleich.
- **Wenige Instanzen:** 20 je Familie (n = 6, 8), 10 (n ≥ 12); die Rangkorrelationen schwanken um grob ±0,19 bzw. ±0,28.
- **Nur drei Schranken:** Linearisierung (Stück 5), doppeltstochastische Relaxation (Stück 6) und semidefinite Schranke (Stück 7) fehlen.
- **Quellen:** Finke, Burkard, Rendl (1987), Hadley, Rendl, Wolkowicz (1992), Gilmore (1962), Lawler (1963) sind aus der Fachliteratur übernommen und **nicht nachgelesen**; die Formeln sind eigene Rechnungen und werden durch Tests (Aufzählung, Zerlegung, `scipy`) geprüft.

## Verifikation

- **Tests** (`pytest tests/`): **237 Tests**, rund vier Minuten (Windows, Python 3.14). Handwerte (die drei Handbeispiele mit Eigenwerten und Termen von Hand, Kosten, Zuordnung, Statistik), Eigenschaften (die Zerlegung ist eine Gleichung für jede Permutation, die projizierte Schranke hängt nicht von der Wahl der Basis ab, Spur- und Normidentitäten der Eigenwerte, EB ≤ PB ≤ Optimum für jede Familie, Invarianz unter Umnummerierung und Skalierung), Orakel (Aufzählung für die Gültigkeit; `scipy` für Eigenwerte, Zuordnung und Rangkorrelation, nur mit `requirements-dev.txt`), Randfälle (zwei Bauteile, Nullfluss, ungültige Eingaben), Verdikte an handgesetzten Werten, Abbildungen, PDF, Rauchtest und Pfade der App (jede Familie, große Instanzen, Knopf `n = 9`, beide Vergleiche), Layout, Footer und Laufzeit-Importe. `test_claims.py` rechnet jede Zahl dieser README nach (rund zwei Minuten).
- **Kopie:** Tests vergleichen die Familien mit denen von Stück 1 bis 3, sofern die Schwester-Repos nebenan liegen.
- **Nicht getestet:** Zeiten; kein Mutationstest; Linux-CI: grün (235 Tests bestanden, 2 übersprungen, Python 3.12; letzter Lauf auf `main`).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: fünf Experimente, Grenzen, Formulierung |
| `eig_rng.py` | SplitMix64 (Kopie aus Stück 1) |
| `eig_instance.py` | Instanzen: sechs Familien der Flussmatrix, Raster, Handbeispiele |
| `eig_cost.py` | Kosten, Tausch-`Δ` (Kopie aus Stück 2) |
| `eig_landscape.py` | Aufzählung für das Optimum (Kopie aus Stück 2) |
| `eig_assignment.py` | Ungarische Methode (Kopie aus Stück 3) |
| `eig_bound.py` | Gilmore–Lawler-Schranke, Runden, Abstiege und beste gefundene Lösung (Kopie aus Stück 3) |
| `eig_spectral.py` | Eigenwerte, Basis senkrecht zum Einsvektor, Eigenwertschranke, projizierte Schranke, Zerlegung |
| `eig_evaluation.py` | Kennzahlen der Flüsse, Zeilen für die Vergleiche, Statistik, Verdikte |
| `eig_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `eig_pdf_export.py` | PDF-Export des Ergebnisses |
| `eig_constants.py` | Größen, Familien, feste Reihen, Farben, Presets |
| `eig_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Button |
| `eig_layout.py` | Fester Kopf (Stylesheet) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Spektralschranken, Zuordnung, Schranke, Kosten, Instanzen, Landschaft, Auswertung, Randfälle, Verdikte, Abbildungen, Presets, Layout, AppTest, Orakel, Zahlen der README, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Originalfassung der projizierten Eigenwertschranke, Verschiebungen der Matrizen, Anstreicher–Brixius:** Primärlektüre fehlt; die vereinfachte Fassung steht für sich.
- **Stärkere Schranken:** Linearisierung und Dual Ascent (Stück 5), doppeltstochastische Relaxation (Stück 6), semidefinit (Stück 7), Branch and Bound (Stück 8).
- **n = 9 im Vergleich und n = 24:** zu langsam für den Vergleich bzw. nicht gerechnet; n = 9 nur als Einzelinstanz.
- **Asymmetrische Flüsse und Abstände.**

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python-Version in den App-Einstellungen prüfen (lokal Python 3.14, CI Python 3.12); der Code verwendet nur Merkmale, die in Python 3.12 laufen. <!-- lint-ok: dezimalpunkt -->

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
