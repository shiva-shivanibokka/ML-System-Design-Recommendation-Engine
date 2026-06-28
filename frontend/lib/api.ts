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
  traffic_split: Record<string, number>;
  winner: string | null;
  total_pulls: Record<string, number>;
  total_rewards: Record<string, number>;
}

export interface MonitoringHealth {
  ctrs: Record<string, number>;
  catalog_coverage: number;
  psi_scores: Record<string, number>;
  any_alert: boolean;
  alerts: string[];
}

export interface SystemHealth {
  status: string;
  models: Record<string, boolean>;
  redis: boolean;
  postgres: boolean;
  n_total_items: number;
}

export interface LatencyReport {
  stage_latencies: Record<string, number>;
  budget_ms: Record<string, number>;
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
      body: JSON.stringify({ request_id: requestId, user_id: userId, item_id: itemId, model_used: modelUsed, rank_shown: rank }),
    });
  },

  banditState: () => req<BanditState>("/bandit/state"),
  monitoringHealth: () => req<MonitoringHealth>("/monitoring/health"),
  monitoringLatency: () => req<LatencyReport>("/monitoring/latency"),
  health: () => req<SystemHealth>("/health"),
};
