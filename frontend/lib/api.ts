const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface Recommendation {
  item_id: number;
  title: string;
  genre: string;
  score: number;
  is_fresh: boolean;
  rank: number;
}

export interface RecommendResponse {
  request_id: string;
  user_id: number;
  recommendations: Recommendation[];
  model_used: string;
  is_cold_start: boolean;
  latency_ms: Record<string, number>;
}

export interface BanditState {
  alphas: Record<string, number>;
  betas: Record<string, number>;
  ctrs: Record<string, number>;
  mean_ctrs: Record<string, number>;
  uncertainty?: Record<string, number>;
  traffic_split: Record<string, number>;
  winner: string | null;
  total_pulls: Record<string, number>;
  total_rewards: Record<string, number>;
}

// Matches the backend /monitoring/health payload (boolean flags, not strings).
export interface MonitoringHealth {
  ctrs: Record<string, number>;
  ctr_alerts: Record<string, boolean>;
  catalog_coverage: number;
  coverage_alert: boolean;
  psi_scores: Record<string, number>;
  psi_alerts: Record<string, boolean>;
  any_alert: boolean;
}

export interface SystemHealth {
  status: string;
  models: Record<string, boolean>;
  redis: boolean;
  postgres: boolean;
  n_total_items: number;
}

export interface EmbeddingPoint {
  item_id: number;
  x: number;
  y: number;
  genre: string;
  title: string;
  popularity: number;
}

export interface EmbeddingMap {
  points: EmbeddingPoint[];
  genres: string[];
  count: number;
}

export interface ModelMetrics {
  dataset: string;
  n_users: number;
  n_items: number;
  eval_protocol: string;
  models: Record<
    string,
    { name: string; family: string; hr_at_10: number; ndcg_at_10: number; note: string }
  >;
}

async function req<T>(path: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  recommend(userId: number, topN: number, modelOverride?: string) {
    const body: Record<string, unknown> = { user_id: userId, top_n: topN };
    if (modelOverride && modelOverride !== "auto") body.model_override = modelOverride;
    return req<RecommendResponse>("/recommend", { method: "POST", body: JSON.stringify(body) });
  },

  click(requestId: string, userId: number, itemId: number, modelUsed: string, rank: number) {
    return req<{ status: string; bandit_state: BanditState }>("/feedback/click", {
      method: "POST",
      body: JSON.stringify({
        request_id: requestId,
        user_id: userId,
        item_id: itemId,
        model_used: modelUsed,
        rank_shown: rank,
      }),
    });
  },

  banditState: () => req<BanditState>("/bandit/state"),
  monitoringHealth: () => req<MonitoringHealth>("/monitoring/health"),
  health: () => req<SystemHealth>("/health"),
  embeddingMap: () => req<EmbeddingMap>("/catalog/embedding_map"),
  modelMetrics: () => req<ModelMetrics>("/model/metrics"),
};

// Client-side mirror of serving/explain.py template mode, so every card can show
// a "why" that is consistent with what's on screen (no extra pipeline round-trip).
export function reasonFor(rec: Recommendation, modelUsed: string, isColdStart: boolean): string {
  const genre = rec.genre?.split("|")[0]?.trim() || "this";
  let reason: string;
  if (isColdStart) {
    reason = `A top-rated ${genre} title that's popular with new users like this one.`;
  } else if (modelUsed === "ncf") {
    reason = `NeuMF found a strong non-linear match between this user's history and this ${genre} title (relevance ${rec.score.toFixed(3)}).`;
  } else if (modelUsed === "svd") {
    reason = `Users with a similar taste profile rated this ${genre} title highly (matrix-factorization score ${rec.score.toFixed(3)}).`;
  } else {
    reason = `Highly rated ${genre} title for this user (score ${rec.score.toFixed(3)}).`;
  }
  if (rec.is_fresh) reason += " It also got a freshness boost for trending recently.";
  return reason;
}

// Warm-tuned genre palette used by cards and the embedding galaxy.
export const GENRE_HUE: Record<string, string> = {
  Action: "#EA5A3D",
  Adventure: "#E7A33A",
  Animation: "#37C4A4",
  Children: "#6FBF73",
  Comedy: "#F0C34B",
  Crime: "#C1584B",
  Documentary: "#8A8A8A",
  Drama: "#5B9BD5",
  Fantasy: "#B57EDC",
  "Film-Noir": "#6B6B6B",
  Horror: "#9B5DE5",
  Musical: "#E28FC0",
  Mystery: "#7E8CE0",
  Romance: "#E27D9E",
  "Sci-Fi": "#3FB4C9",
  Thriller: "#D98736",
  War: "#A08159",
  Western: "#C9A227",
  Unknown: "#8A8A8A",
};

export function genreHue(genre: string): string {
  return GENRE_HUE[genre?.split("|")[0]?.trim()] ?? "#8A8A8A";
}
