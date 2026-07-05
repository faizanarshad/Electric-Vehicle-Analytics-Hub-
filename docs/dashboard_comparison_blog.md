# Streamlit vs. Dash vs. Gradio: Building Three EV Analytics Dashboards From the Same Dataset

We took one dataset — 478 electric vehicle models from 59 brands (2025 spec sheet data) — and built the same analysis three times, once in each of the big Python dashboard frameworks: **Streamlit**, **Dash**, and **Gradio**. All three ship in this project ([src/dashboards](../src/dashboards)) and all three run locally with one command. This post compares how they actually look, feel, and perform side by side, based on running all three at once and clicking through every tab.

---

## TL;DR

| | Streamlit | Dash | Gradio |
|---|---|---|---|
| **Best for** | Business-style reporting, filterable views | Deep ML analysis, polished custom UI | Quick AI/NL-query demos |
| **Visual polish** | Clean, corporate | Best — custom gradient theme, animated cards | Good, but noticeably plainer (default Gradio chrome) |
| **Chart render speed** | Slow (~2–3s lag per tab, visible blank boxes first) | Fast, near-instant | Fast, but several charts are broken (see below) |
| **Interactivity depth** | Sidebar filters (brand/segment/range) apply globally | Per-tab controls (cluster slider, explorer filters) | Per-tab, but some tabs need a manual "Generate" click |
| **Bugs found** | Confusing default filter (see below) | None observed | Broken correlation matrix & distribution histograms in "Analytics" tab |
| **Setup weight** | Medium (`streamlit`, `plotly`, `seaborn`, `matplotlib`) | Medium (`dash`, `scikit-learn`, `scipy`) | Heaviest — version conflicts out of the box (see below) |

---

## 1. Streamlit — "Electric Vehicle Analytics Hub"

**Port 8501** · [ev_streamlit_dashboard.py](../src/dashboards/ev_streamlit_dashboard.py)

Streamlit's version is the most traditional — a persistent left sidebar ("Dashboard Controls") with brand/segment multi-selects and a range slider, and a tabbed main area: **Overview, Brands, Performance, Battery & Range, Charging, Segments, Trends, Explorer**.

**What's good:**
- The sidebar filters are global — pick brands and segments once and every tab respects them. This is the only one of the three dashboards with a genuinely shared filter state across all pages.
- Layout is the cleanest and most "readable as a report" — good for sharing with non-technical stakeholders.
- Widest breadth of pages (8 vs. 5 for the other two).

**What's rough:**
- **The default filter is misleading.** On first load, only 5 brands (Abarth, Aiways, Alfa, Alpine, Audi — alphabetically first) are pre-selected, so the Overview page shows "Total Vehicles: 38" instead of the full 478. A first-time visitor could easily think the dataset is tiny unless they notice the sidebar and clear the filter.
- **Visible chart pop-in.** Every tab shows two empty white placeholder boxes for roughly 2–3 seconds before Plotly charts render — noticeably slower than Dash's near-instant load with the same data and chart types.
- Streamlit prints deprecation warnings in the console (`use_container_width` → `width`) — cosmetic today, but will break on a future Streamlit release if not updated.

---

## 2. Dash — "Advanced EV Analytics Hub"

**Port 8050** · [ev_dash_dashboard.py](../src/dashboards/ev_dash_dashboard.py)

Dash's version is the most visually ambitious: a purple gradient hero header, pill-style tab buttons, and five tabs — **Market Overview, ML Insights, Predictive Analytics, Advanced Analytics, Interactive Explorer**.

**What's good:**
- By far the most "production dashboard" look — custom CSS, card shadows, consistent purple/white theme.
- **This is the only dashboard with real machine learning built in**: K-means clustering (adjustable K=2–6) with a live radar chart comparing cluster characteristics, plus a linear regression range predictor with an R² score, a scatter plot showing where your prediction lands against real vehicles, and a 3D prediction surface.
- The "Advanced Analytics" tab has a full feature-correlation heatmap (range, speed, battery, efficiency, acceleration) and four distribution histograms — all rendered correctly and fast.
- The "Interactive Explorer" tab combines 2D filtering with a rotatable 3D scatter (battery × range × speed) — nice for spotting outliers.

**What's rough:**
- No cross-tab filter memory — each tab's filters (Explorer's brand/segment/range/speed sliders) are local to that tab only.
- The dev-mode Dash debug menu ("Plotly Cloud / Errors / Callbacks / Server") is visible at the bottom of the screen — fine for local dev, but would need `debug=False` before sharing with anyone else.

---

## 3. Gradio — "EV Analytics Hub"

**Port 7861** · [ev_gradio_dashboard.py](../src/dashboards/ev_gradio_dashboard.py)

Gradio's version leans into the "AI tool" framing, with five tabs: **Market Overview, AI Assistant, Predictions, Analytics, Explorer**.

**What's good:**
- The only dashboard with a natural-language input box ("Ask me anything about EVs") — the seed of a chat-style interface.
- Fastest to interact with for simple actions (sliders + button clicks feel instant).
- Its range-prediction number (357 km at 75 kWh / 200 Wh/km) matched Dash's prediction exactly — reassuring that the underlying regression logic is consistent across dashboards.

**What's rough — this is the dashboard with real bugs:**
- **The "AI Assistant" isn't actually AI.** Asking "Which brand has the longest average range?" returns a generic canned paragraph ("The EV market is rapidly evolving...") instead of a real, data-grounded answer. It's keyword-matched template text, not a model reasoning over the dataset — worth knowing before you promote it as "AI-powered" in a demo.
- **The Analytics tab is broken.** After clicking "Generate Analytics," the Correlation Matrix, Range Distribution, and Speed Distribution charts render as empty axes with no data plotted at all (only the Segment Analysis box plot works). This looks like a Plotly figure/trace bug specific to this file — the same correlation matrix works fine in Dash.
- **The Prediction Surface (3D) never rendered** in our test, staying blank after clicking "Predict Range," while the same feature works in Dash.
- **Dependency fragility**: fresh installs of `gradio>=4.0.0` pull the latest `huggingface_hub`, which no longer exports `HfFolder` — the app crashes on import until you pin `huggingface_hub<0.26`. None of the other two dashboards had an installation problem like this.

---

## Data insights surfaced (consistent across all three)

Pulled from the full unfiltered dataset (478 vehicles / 59 brands), confirmed in Dash and Streamlit's brand charts:

- **Market leader:** Mercedes-Benz, 18.1% share, followed by Audi (12.1%) and Porsche (11.2%).
- **Averages:** 393 km range, 74.0 kWh battery, 185 km/h top speed.
- **Strong correlations:** battery capacity ↔ range (r ≈ 0.88), top speed ↔ range (r ≈ 0.73); acceleration time is *negatively* correlated with all three (faster cars tend to have more range, speed, and battery — i.e., performance and range scale together rather than trade off).
- **Clustering (K=4)** cleanly separates the market into a "mainstream" cluster (176 cars, ~429 km / 186 km/h), a "budget/short-range" cluster (148 cars, ~292 km), a "premium/long-range" cluster (127 cars, ~495 km / 226 km/h), and a small 27-car "sport" outlier cluster.

---

## Verdict

- Want something to **hand to a business stakeholder** for exploring the market with filters? **Streamlit** — just fix the misleading default filter first.
- Want the **most capable and polished analysis tool**, including real ML? **Dash** — no bugs found, fastest rendering, and the only one with working 3D/clustering/regression visuals.
- Want a **demo-friendly AI framing**? **Gradio** has the right shape (chat box, simple sliders) but needs real engineering before the "AI Assistant" and "Analytics" tabs can be trusted — right now they're the least reliable of the three.

*Screenshots for each tab discussed above were captured during this review and are available in the project conversation history; static reference charts also live in [results/figures](../results/figures).*
