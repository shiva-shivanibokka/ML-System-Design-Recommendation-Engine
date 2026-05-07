"""
Gradio Demo UI — ML System Design: Real-Time Recommendation Engine.

Four tabs:
  1. Recommendations    — Enter user ID, get top-10 movies + click any to register feedback
  2. Bandit Dashboard   — Live Thompson Sampling state: α/β, CTR, traffic split, winner
  3. System Monitoring  — Staleness signals: CTR trend, catalog coverage, PSI score
  4. Architecture       — System design diagram and component explanations

Run locally:
    python gradio_app/app.py

Or via Docker:
    docker-compose up gradio
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

import gradio as gr
import httpx
import plotly.graph_objects as go
import numpy as np

GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 10.0


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def api_recommend(
    user_id: int, top_n: int = 10, model_override: Optional[str] = None
) -> dict:
    payload = {"user_id": user_id, "top_n": top_n}
    if model_override and model_override != "auto (bandit)":
        payload["model_override"] = model_override
    try:
        r = httpx.post(
            f"{GATEWAY_URL}/recommend", json=payload, timeout=REQUEST_TIMEOUT
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def api_click(
    request_id: str, user_id: int, item_id: int, model_used: str, rank: int
) -> dict:
    try:
        r = httpx.post(
            f"{GATEWAY_URL}/feedback/click",
            json={
                "request_id": request_id,
                "user_id": user_id,
                "item_id": item_id,
                "model_used": model_used,
                "rank_shown": rank,
            },
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def api_bandit_state() -> dict:
    try:
        r = httpx.get(f"{GATEWAY_URL}/bandit/state", timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def api_monitoring() -> dict:
    try:
        r = httpx.get(f"{GATEWAY_URL}/monitoring/health", timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def api_health() -> dict:
    try:
        r = httpx.get(f"{GATEWAY_URL}/health", timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Tab 1: Recommendations
# ---------------------------------------------------------------------------

_last_request: dict = {}


def get_recommendations(user_id: int, top_n: int, model_choice: str):
    global _last_request
    result = api_recommend(user_id, top_n, model_choice)
    if "error" in result:
        return f"Error: {result['error']}", "", gr.update(choices=[])

    _last_request = result
    recs = result.get("recommendations", [])
    model_used = result.get("model_used", "?")
    is_cold = result.get("is_cold_start", False)
    latency = result.get("latency_ms", {})

    # Format recommendations table
    rows = []
    for i, rec in enumerate(recs, 1):
        rows.append(
            f"{i}. **{rec['title']}** ({rec['genre']}) "
            f"— score: {rec['score']:.4f}" + (" 🆕" if rec.get("is_fresh") else "")
        )

    status = (
        f"**Model used:** `{model_used}` | "
        f"**Cold-start:** {'Yes' if is_cold else 'No'} | "
        f"**Total latency:** {latency.get('total', 0):.1f}ms\n\n"
        f"**Stage breakdown:** "
        f"cache={latency.get('cache_check', 0):.1f}ms | "
        f"retrieval={latency.get('candidate_generation', 0):.1f}ms | "
        f"ranking={latency.get('ranking', 0):.1f}ms | "
        f"post-rank={latency.get('post_ranking', 0):.1f}ms"
    )

    rec_text = "\n".join(rows) if rows else "No recommendations returned."
    item_choices = [
        f"{i + 1}. {rec['title']} (id={rec['item_id']})" for i, rec in enumerate(recs)
    ]

    return status, rec_text, gr.update(choices=item_choices, value=None)


def register_click(selection: str):
    global _last_request
    if not _last_request or not selection:
        return "No active request or nothing selected."
    try:
        rank = int(selection.split(".")[0]) - 1
        recs = _last_request.get("recommendations", [])
        if rank >= len(recs):
            return "Invalid selection."
        rec = recs[rank]
        result = api_click(
            request_id=_last_request["request_id"],
            user_id=_last_request["user_id"],
            item_id=rec["item_id"],
            model_used=_last_request["model_used"],
            rank=rank + 1,
        )
        bandit = result.get("bandit_state", {})
        ctrs = bandit.get("ctrs", {})
        ctr_str = " | ".join(f"{m}: {v:.4f}" for m, v in ctrs.items())
        return f"Click registered for **{rec['title']}**. Updated CTRs: {ctr_str}"
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Tab 2: Bandit Dashboard
# ---------------------------------------------------------------------------


def bandit_dashboard():
    state = api_bandit_state()
    if "error" in state:
        return f"Error: {state['error']}", None, None

    models = list(state.get("alphas", {}).keys())
    alphas = [state["alphas"].get(m, 1) for m in models]
    betas = [state["betas"].get(m, 1) for m in models]
    ctrs = state.get("ctrs", {})
    mean_ctrs = state.get("mean_ctrs", {})
    traffic = state.get("traffic_split", {})
    winner = state.get("winner")
    pulls = state.get("total_pulls", {})
    rewards = state.get("total_rewards", {})

    summary = (
        f"**Winner declared:** `{winner or 'None yet'}`\n\n"
        f"| Model | α | β | Pulls | Rewards | Empirical CTR | Bayesian CTR | Traffic % |\n"
        f"|---|---|---|---|---|---|---|---|\n"
    )
    for m in models:
        summary += (
            f"| `{m}` | {alphas[models.index(m)]:.1f} | {betas[models.index(m)]:.1f} | "
            f"{pulls.get(m, 0)} | {rewards.get(m, 0)} | "
            f"{ctrs.get(m, 0):.4f} | {mean_ctrs.get(m, 0):.4f} | "
            f"{traffic.get(m, 0) * 100:.1f}% |\n"
        )

    # Beta distribution visualization
    x = np.linspace(0, 1, 300)
    fig = go.Figure()
    colors = ["#4285F4", "#EA4335", "#FBBC04", "#34A853"]
    for i, m in enumerate(models):
        a, b = alphas[i], betas[i]
        from scipy.stats import beta as beta_dist

        y = beta_dist.pdf(x, a, b)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=f"{m} Beta({a:.0f}, {b:.0f})",
                fill="tozeroy",
                opacity=0.4,
                line=dict(color=colors[i % len(colors)]),
            )
        )
    fig.update_layout(
        title="Thompson Sampling: Beta Distributions per Model",
        xaxis_title="θ (estimated CTR)",
        yaxis_title="Probability Density",
        template="plotly_dark",
        height=350,
    )

    # Traffic split pie
    fig2 = go.Figure(
        go.Pie(
            labels=list(traffic.keys()),
            values=list(traffic.values()),
            hole=0.4,
            marker_colors=colors[: len(traffic)],
        )
    )
    fig2.update_layout(
        title="Effective Traffic Split",
        template="plotly_dark",
        height=300,
    )

    return summary, fig, fig2


# ---------------------------------------------------------------------------
# Tab 3: Monitoring
# ---------------------------------------------------------------------------


def monitoring_dashboard():
    health = api_monitoring()
    sys_health = api_health()

    if "error" in health:
        return f"Error fetching monitoring data: {health['error']}", None

    ctrs = health.get("ctrs", {})
    coverage = health.get("catalog_coverage", 0)
    psi = health.get("psi_scores", {})
    any_alert = health.get("any_alert", False)

    alert_str = "🔴 STALENESS ALERT ACTIVE" if any_alert else "🟢 All signals healthy"
    models_ok = sys_health.get("models", {})

    summary = f"""
**System Health:** {alert_str}

**Service Status:**
- NCF Model loaded: {"✅" if models_ok.get("ncf") else "❌"}
- SVD Model loaded: {"✅" if models_ok.get("svd") else "❌"}
- FAISS Index loaded: {"✅" if models_ok.get("faiss") else "❌"}
- Redis connected: {"✅" if sys_health.get("redis") else "❌"}
- PostgreSQL connected: {"✅" if sys_health.get("postgres") else "❌"}
- Total items in catalog: {sys_health.get("n_total_items", 0):,}

**Rolling 7-day CTR per model:**
{chr(10).join(f"  - {m}: {v:.4f}" for m, v in ctrs.items())}

**Catalog Coverage (24h):** {coverage:.2%}
  {"⚠️ COVERAGE COLLAPSE" if coverage < 0.1 else "✅ Healthy"}

**PSI (Score Distribution Drift):**
{chr(10).join(f"  - {m}: {v:.4f} {'⚠️ DRIFT' if v > 0.2 else '✅'}" for m, v in psi.items())}
"""

    # Gauge chart for coverage
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=coverage * 100,
            title={"text": "Catalog Coverage (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "green" if coverage >= 0.1 else "red"},
                "steps": [
                    {"range": [0, 10], "color": "#ff4444"},
                    {"range": [10, 50], "color": "#ffaa00"},
                    {"range": [50, 100], "color": "#44cc44"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 10,
                },
            },
        )
    )
    fig.update_layout(template="plotly_dark", height=300)
    return summary, fig


# ---------------------------------------------------------------------------
# Tab 4: Architecture
# ---------------------------------------------------------------------------

ARCHITECTURE_MD = """
## System Architecture

```
User Request (user_id)
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│                     STAGE 0: Redis Cache Check                     │
│  cache hit? → return in <5ms    cache miss? → continue pipeline    │
└─────────────────────────────────┬─────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────┐
│              STAGE 1: Candidate Generation (FAISS ANN)             │
│  Is cold user? → popularity fallback                               │
│  Warm user → NCF GMF embedding → FAISS IVF+PQ query               │
│  Output: top-500 candidate items  |  Latency budget: 20ms          │
└─────────────────────────────────┬─────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────┐
│           STAGE 2: Feature Fetch (Redis Online Store / Feast)       │
│  User features + item features from Redis (<5ms)                   │
│  Same features used in training (eliminates training-serving skew) │
│  Latency budget: 10ms                                              │
└─────────────────────────────────┬─────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────┐
│          STAGE 3: Ranking — Thompson Sampling Bandit Router         │
│  Thompson Sampling selects: SVD or NCF arm                         │
│  NCF: full NeuMF forward pass over 500 candidates (~50ms)          │
│  SVD: dot product over 500 item factors (~2ms)                     │
│  Auto-converges to winner arm over time (no fixed 50/50 split)     │
│  Latency budget: 50ms                                              │
└─────────────────────────────────┬─────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────┐
│              STAGE 4: Post-Ranking                                  │
│  1. Freshness boost (recently interacted items get +10% score)     │
│  2. MMR diversity re-ranking (λ=0.7, penalizes same-genre repeats) │
│  3. Genre hard cap (max 3 items from same genre)                   │
│  Latency budget: 10ms                                              │
└─────────────────────────────────┬─────────────────────────────────┘
                                   │
                                   ▼
               Top-10 recommendations returned

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FEEDBACK LOOP:
  Click event → POST /feedback/click
  → Updates Thompson Sampling bandit (α/β)
  → Logs to PostgreSQL
  → Kafka consumer writes to Feast offline store
  → Nightly: Airflow DAG retrains → validates → promotes if better

STALENESS DETECTION (background thread, every 6h):
  - Rolling 7-day CTR per model (alert < 80% of baseline)
  - Catalog coverage (alert < 10% of catalog recommended)
  - PSI score distribution shift (alert > 0.2)

TOTAL P99 LATENCY TARGET: < 100ms (Netflix/YouTube standard)
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Two-stage retrieval (FAISS + NCF) | Can't run NCF over 3,700 items per request in <100ms; FAISS narrows to 500 in 20ms |
| NeuMF over plain NCF | GMF path captures linear CF signal; MLP captures non-linear — combined outperforms either alone |
| Thompson Sampling over fixed A/B | Stops sending traffic to the worse model automatically; statistically principled winner detection |
| MMR diversity | Pure relevance → same genre dominates; MMR injects diversity without sacrificing relevance |
| Cold-start routing | New users → popularity; new items → content similarity — prevents empty results |
| PSI for drift detection | Standard metric in credit risk and ML monitoring; catches distribution shift before CTR drops |
| Feast offline/online stores | Guarantees training and serving use the same features — eliminates training-serving skew |
"""


# ---------------------------------------------------------------------------
# Gradio App
# ---------------------------------------------------------------------------


def build_app():
    with gr.Blocks(
        title="RecSys — ML System Design: Real-Time Recommendation Engine",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("# ML System Design: Real-Time Recommendation Engine")
        gr.Markdown(
            "Netflix-style two-stage recommendation system: "
            "FAISS candidate generation → NeuMF/SVD ranking → "
            "MMR diversity → Thompson Sampling bandit A/B testing."
        )

        with gr.Tabs():
            # ---- Tab 1: Recommendations ----
            with gr.TabItem("Recommendations"):
                with gr.Row():
                    with gr.Column(scale=1):
                        user_id_input = gr.Number(
                            label="User ID (MovieLens 1M: 1-6040)",
                            value=1,
                            precision=0,
                        )
                        top_n_input = gr.Slider(
                            label="Top N recommendations",
                            minimum=1,
                            maximum=20,
                            value=10,
                            step=1,
                        )
                        model_choice = gr.Dropdown(
                            label="Model override",
                            choices=["auto (bandit)", "svd", "ncf"],
                            value="auto (bandit)",
                        )
                        rec_btn = gr.Button("Get Recommendations", variant="primary")

                    with gr.Column(scale=2):
                        status_out = gr.Markdown(label="Status")
                        rec_out = gr.Markdown(label="Recommendations")
                        click_selector = gr.Radio(
                            label="Click on a recommendation (registers feedback):",
                            choices=[],
                        )
                        click_btn = gr.Button("Register Click")
                        click_status = gr.Markdown()

                rec_btn.click(
                    fn=get_recommendations,
                    inputs=[user_id_input, top_n_input, model_choice],
                    outputs=[status_out, rec_out, click_selector],
                )
                click_btn.click(
                    fn=register_click,
                    inputs=[click_selector],
                    outputs=[click_status],
                )

            # ---- Tab 2: Bandit Dashboard ----
            with gr.TabItem("Bandit Dashboard"):
                gr.Markdown(
                    "### Thompson Sampling Multi-Armed Bandit\n"
                    "Real-time Beta distribution per model arm. "
                    "Traffic automatically shifts toward the winning arm."
                )
                refresh_bandit_btn = gr.Button("Refresh", variant="secondary")
                bandit_summary = gr.Markdown()
                with gr.Row():
                    beta_plot = gr.Plot(label="Beta Distributions")
                    traffic_plot = gr.Plot(label="Traffic Split")

                refresh_bandit_btn.click(
                    fn=bandit_dashboard,
                    outputs=[bandit_summary, beta_plot, traffic_plot],
                )
                demo.load(
                    fn=bandit_dashboard,
                    outputs=[bandit_summary, beta_plot, traffic_plot],
                )

            # ---- Tab 3: Monitoring ----
            with gr.TabItem("System Monitoring"):
                gr.Markdown(
                    "### Staleness Detection\n"
                    "Three signals: Rolling CTR, Catalog Coverage, PSI score distribution drift."
                )
                refresh_mon_btn = gr.Button("Refresh", variant="secondary")
                mon_summary = gr.Markdown()
                coverage_gauge = gr.Plot(label="Catalog Coverage Gauge")

                refresh_mon_btn.click(
                    fn=monitoring_dashboard,
                    outputs=[mon_summary, coverage_gauge],
                )
                demo.load(
                    fn=monitoring_dashboard, outputs=[mon_summary, coverage_gauge]
                )

            # ---- Tab 4: Architecture ----
            with gr.TabItem("System Architecture"):
                gr.Markdown(ARCHITECTURE_MD)

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
