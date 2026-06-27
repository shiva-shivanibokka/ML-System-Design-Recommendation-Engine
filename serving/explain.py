"""
Recommendation Explanation Service.

Generates per-item natural language reasons for recommendations.

Template mode: always available, deterministic, zero latency overhead.
Gemini mode: richer personalized explanations via google-generativeai.
             Requires GEMINI_API_KEY env var (free tier at aistudio.google.com).
             Falls back to templates on any API error.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List


def explain_recommendations(
    user_id: int,
    recommendations: List[Dict[str, Any]],
    model_used: str,
    is_cold_start: bool,
) -> List[Dict[str, Any]]:
    """
    Add a 'reason' field to each recommendation dict.
    Uses Gemini Flash if GEMINI_API_KEY is set, otherwise templates.
    Never raises — falls back to templates on any failure.
    """
    if os.getenv("GEMINI_API_KEY"):
        try:
            return _gemini_explain(user_id, recommendations, model_used, is_cold_start)
        except Exception:
            pass
    return _template_explain(recommendations, model_used, is_cold_start)


def _template_explain(
    recs: List[Dict[str, Any]],
    model_used: str,
    is_cold_start: bool,
) -> List[Dict[str, Any]]:
    explained = []
    for rec in recs:
        genre = rec.get("genre", "Unknown")
        score = rec.get("score", 0.0)
        is_fresh = rec.get("is_fresh", False)

        if is_cold_start:
            reason = f"A top-rated {genre} title popular among new users"
        elif model_used == "ncf":
            reason = (
                f"Neural collaborative filtering found strong alignment between "
                f"your interaction history and this {genre} title "
                f"(relevance score: {score:.3f})"
            )
        elif model_used == "svd":
            reason = (
                f"Users with a similar taste profile loved this {genre} title "
                f"(matrix factorization score: {score:.3f})"
            )
        else:
            reason = f"Highly rated {genre} title (score: {score:.3f})"

        if is_fresh:
            reason += " · Currently trending"

        explained.append({**rec, "reason": reason})
    return explained


def _gemini_explain(
    user_id: int,
    recs: List[Dict[str, Any]],
    model_used: str,
    is_cold_start: bool,
) -> List[Dict[str, Any]]:
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash")

    items_summary = "\n".join(
        f"- {r['title']} (genre={r['genre']}, score={r.get('score', 0):.3f}, "
        f"trending={r.get('is_fresh', False)})"
        for r in recs[:10]
    )
    prompt = (
        f"A recommendation engine (model={model_used}, cold_start={is_cold_start}) "
        f"recommended these movies to user {user_id}:\n{items_summary}\n\n"
        f"Write a concise 1-sentence reason for EACH recommendation explaining WHY "
        f"the engine selected it (reference genre, relevance, or user taste patterns). "
        f"Return ONLY a JSON array — no markdown, no extra text:\n"
        f'[{{"title": "...", "reason": "..."}}]'
    )

    response = model.generate_content(prompt)
    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:].strip()

    reasons = json.loads(text)
    reason_map = {r["title"]: r["reason"] for r in reasons}

    return [
        {**rec, "reason": reason_map.get(rec["title"], f"Recommended for you ({rec.get('genre', '')})")}
        for rec in recs
    ]
