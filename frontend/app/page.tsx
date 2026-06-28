"use client";

import { useState } from "react";
import { api, type RecommendResponse, type Recommendation } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Zap, Loader2, TrendingUp, Clock, AlertCircle, MousePointerClick } from "lucide-react";
import { cn } from "@/lib/utils";

const GENRE_COLORS: Record<string, string> = {
  Action: "bg-red-500/20 text-red-400 border-red-500/30",
  Comedy: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
  Drama: "bg-blue-500/20 text-blue-400 border-blue-500/30",
  Romance: "bg-pink-500/20 text-pink-400 border-pink-500/30",
  Thriller: "bg-orange-500/20 text-orange-400 border-orange-500/30",
  "Sci-Fi": "bg-cyan-500/20 text-cyan-400 border-cyan-500/30",
  Horror: "bg-purple-500/20 text-purple-400 border-purple-500/30",
  Animation: "bg-green-500/20 text-green-400 border-green-500/30",
  Adventure: "bg-teal-500/20 text-teal-400 border-teal-500/30",
  Fantasy: "bg-violet-500/20 text-violet-400 border-violet-500/30",
  Documentary: "bg-slate-500/20 text-slate-400 border-slate-500/30",
  Crime: "bg-rose-500/20 text-rose-400 border-rose-500/30",
  Musical: "bg-lime-500/20 text-lime-400 border-lime-500/30",
  War: "bg-stone-500/20 text-stone-400 border-stone-500/30",
  Western: "bg-amber-500/20 text-amber-400 border-amber-500/30",
};

function genreClass(genre: string) {
  return GENRE_COLORS[genre] ?? "bg-slate-500/20 text-slate-400 border-slate-500/30";
}

const MODELS = [
  { value: "auto", label: "Auto (Bandit)" },
  { value: "ncf", label: "NeuMF" },
  { value: "svd", label: "SVD" },
];

function RecommendationCard({
  rec,
  onClickFeedback,
  clicked,
}: {
  rec: Recommendation;
  onClickFeedback: () => void;
  clicked: boolean;
}) {
  return (
    <Card className="group relative overflow-hidden transition-all hover:border-primary/40 hover:shadow-lg hover:shadow-primary/5">
      <CardContent className="p-5">
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2">
            <span className="flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-full bg-primary/10 text-xs font-bold text-primary">
              {rec.rank}
            </span>
            {rec.is_fresh && (
              <Badge className="gap-1 border-amber-500/30 bg-amber-500/20 text-amber-400 text-xs">
                <TrendingUp className="h-3 w-3" />
                Trending
              </Badge>
            )}
          </div>
          <span className="text-xs font-mono text-muted-foreground">{rec.score.toFixed(4)}</span>
        </div>

        <h3 className="mt-3 font-semibold text-foreground leading-tight line-clamp-2">{rec.title}</h3>

        <div className="mt-2 flex flex-wrap gap-1.5">
          {rec.genre.split("|").map((g) => (
            <span
              key={g}
              className={cn("inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-medium", genreClass(g.trim()))}
            >
              {g.trim()}
            </span>
          ))}
        </div>

        <div className="mt-3">
          <div className="h-1 w-full overflow-hidden rounded-full bg-secondary">
            <div
              className="h-full rounded-full bg-primary/60 transition-all"
              style={{ width: `${Math.min(rec.score * 100, 100)}%` }}
            />
          </div>
        </div>

        <Button
          size="sm"
          variant={clicked ? "secondary" : "ghost"}
          className={cn(
            "mt-3 w-full gap-1.5 text-xs",
            clicked ? "text-emerald-400" : "text-muted-foreground hover:text-foreground"
          )}
          onClick={onClickFeedback}
          disabled={clicked}
        >
          <MousePointerClick className="h-3 w-3" />
          {clicked ? "Feedback registered" : "Register click"}
        </Button>
      </CardContent>
    </Card>
  );
}

export default function RecommendationsPage() {
  const [userId, setUserId] = useState<string>("1");
  const [topN, setTopN] = useState(10);
  const [model, setModel] = useState("auto");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<RecommendResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [clicked, setClicked] = useState<Set<number>>(new Set());

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const id = parseInt(userId);
    if (isNaN(id) || id < 1) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setClicked(new Set());
    try {
      const data = await api.recommend(id, topN, model === "auto" ? undefined : model);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  async function handleClick(rec: Recommendation) {
    if (!result || clicked.has(rec.item_id)) return;
    setClicked((s) => new Set([...s, rec.item_id]));
    try {
      await api.click(result.request_id, result.user_id, rec.item_id, result.model_used, rec.rank);
    } catch {
      // feedback best-effort
    }
  }

  const stageLabels: Record<string, string> = {
    cache_check: "Cache",
    candidate_generation: "Retrieval",
    feature_fetch: "Features",
    ranking: "Ranking",
    post_ranking: "Post-rank",
    total: "Total",
  };

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-2xl font-bold tracking-tight">Recommendations</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Two-stage FAISS retrieval → NeuMF / SVD ranking → MMR diversity
        </p>
      </div>

      <div className="grid grid-cols-1 gap-8 lg:grid-cols-[320px_1fr]">
        {/* Controls */}
        <div className="space-y-6">
          <Card>
            <CardHeader className="pb-4">
              <CardTitle className="text-sm font-medium">Query Parameters</CardTitle>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-5">
                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-muted-foreground">User ID (1 – 6040)</label>
                  <Input
                    type="number"
                    min={1}
                    max={6040}
                    value={userId}
                    onChange={(e) => setUserId(e.target.value)}
                    placeholder="e.g. 42"
                  />
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <label className="text-xs font-medium text-muted-foreground">Top N</label>
                    <span className="text-xs font-mono text-primary">{topN}</span>
                  </div>
                  <Slider min={5} max={20} step={1} value={[topN]} onValueChange={([v]) => setTopN(v)} />
                </div>

                <div className="space-y-1.5">
                  <label className="text-xs font-medium text-muted-foreground">Model</label>
                  <div className="flex gap-1.5">
                    {MODELS.map((m) => (
                      <button
                        key={m.value}
                        type="button"
                        onClick={() => setModel(m.value)}
                        className={cn(
                          "flex-1 rounded-md border px-2 py-1.5 text-xs font-medium transition-colors",
                          model === m.value
                            ? "border-primary bg-primary/10 text-primary"
                            : "border-border text-muted-foreground hover:border-primary/40 hover:text-foreground"
                        )}
                      >
                        {m.label}
                      </button>
                    ))}
                  </div>
                </div>

                <Button type="submit" className="w-full gap-2" disabled={loading}>
                  {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Zap className="h-4 w-4" />}
                  {loading ? "Fetching…" : "Get Recommendations"}
                </Button>
              </form>
            </CardContent>
          </Card>

          {result && (
            <Card>
              <CardContent className="p-5 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">Model used</span>
                  <Badge variant="outline" className="font-mono text-xs">{result.model_used.toUpperCase()}</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">Cold start</span>
                  <Badge variant={result.is_cold_start ? "warning" : "success"}>
                    {result.is_cold_start ? "Yes" : "No"}
                  </Badge>
                </div>
                <div className="pt-1 space-y-1.5">
                  {Object.entries(result.latency_ms)
                    .filter(([k]) => k in stageLabels)
                    .map(([k, v]) => (
                      <div key={k} className="flex items-center gap-2">
                        <span className="w-20 text-xs text-muted-foreground flex-shrink-0">{stageLabels[k]}</span>
                        <div className="flex-1 h-1 rounded-full bg-secondary overflow-hidden">
                          <div
                            className={cn("h-full rounded-full", k === "total" ? "bg-primary" : "bg-primary/40")}
                            style={{ width: `${Math.min((v / (result.latency_ms.total || 100)) * 100, 100)}%` }}
                          />
                        </div>
                        <span className="w-14 text-right text-xs font-mono text-muted-foreground">{v.toFixed(1)}ms</span>
                      </div>
                    ))}
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Results */}
        <div>
          {error && (
            <Alert variant="destructive" className="mb-6">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {!result && !loading && !error && (
            <div className="flex h-80 items-center justify-center rounded-xl border border-dashed border-border">
              <div className="text-center">
                <Zap className="mx-auto h-8 w-8 text-muted-foreground/40" />
                <p className="mt-3 text-sm text-muted-foreground">Enter a user ID and click Get Recommendations</p>
              </div>
            </div>
          )}

          {loading && (
            <div className="flex h-80 items-center justify-center">
              <div className="flex items-center gap-3 text-muted-foreground">
                <Loader2 className="h-5 w-5 animate-spin" />
                <span className="text-sm">Running 5-stage pipeline…</span>
              </div>
            </div>
          )}

          {result && !loading && (
            <>
              <div className="mb-4 flex items-center justify-between">
                <p className="text-sm text-muted-foreground">
                  <span className="font-medium text-foreground">{result.recommendations.length}</span> recommendations
                  for user <span className="font-medium text-foreground">{result.user_id}</span>
                  {" · "}
                  <span className="font-mono text-primary">{result.latency_ms.total?.toFixed(0)}ms</span>
                </p>
                <div className="flex items-center gap-1 text-xs text-muted-foreground">
                  <Clock className="h-3 w-3" />
                  p99 budget: 100ms
                </div>
              </div>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
                {result.recommendations.map((rec) => (
                  <RecommendationCard
                    key={rec.item_id}
                    rec={rec}
                    onClickFeedback={() => handleClick(rec)}
                    clicked={clicked.has(rec.item_id)}
                  />
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
