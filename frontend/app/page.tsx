"use client";

import { useState, useEffect } from "react";
import {
  api,
  reasonFor,
  genreHue,
  type RecommendResponse,
  type Recommendation,
} from "@/lib/api";
import { Info } from "@/components/info";
import { Pipeline } from "@/components/pipeline";
import { cn } from "@/lib/utils";

const MODELS = [
  { value: "auto", label: "Auto", info: "bandit" as const },
  { value: "ncf", label: "NeuMF", info: "neumf" as const },
  { value: "svd", label: "SVD", info: "svd" as const },
];

const SAMPLE_USERS = [1, 42, 314, 1729, 4040];

function RecCard({
  rec,
  rank,
  modelUsed,
  isCold,
  clicked,
  onClick,
}: {
  rec: Recommendation;
  rank: number;
  modelUsed: string;
  isCold: boolean;
  clicked: boolean;
  onClick: () => void;
}) {
  const [open, setOpen] = useState(false);
  const hue = genreHue(rec.genre);

  return (
    <div
      className="animate-fade-up group relative overflow-hidden rounded-xl border border-line bg-panel transition-colors hover:border-signal/40"
      style={{ animationDelay: `${Math.min(rank * 30, 300)}ms` }}
    >
      {/* rank ribbon */}
      <div
        className="absolute left-0 top-0 h-full w-[3px]"
        style={{ background: hue }}
      />
      <div className="p-4 pl-5">
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-center gap-2.5">
            <span className="font-mono text-lg font-bold tabular-nums text-muted-foreground">
              {String(rank).padStart(2, "0")}
            </span>
            <h3 className="text-[15px] font-semibold leading-tight text-foreground">
              {rec.title}
            </h3>
          </div>
          {rec.is_fresh && (
            <span className="flex shrink-0 items-center gap-1 rounded-full border border-signal/30 bg-signal-soft px-2 py-0.5 font-mono text-[10px] uppercase tracking-wide text-signal">
              trending
            </span>
          )}
        </div>

        <div className="mt-2.5 flex items-center gap-2">
          <span
            className="rounded-md px-2 py-0.5 text-[11px] font-medium"
            style={{ background: `${hue}22`, color: hue }}
          >
            {rec.genre.split("|")[0].trim()}
          </span>
          <div className="flex-1" />
          <span className="font-mono text-[11px] text-muted-foreground">score</span>
          <span className="font-mono text-[13px] font-semibold text-foreground">
            {rec.score.toFixed(3)}
          </span>
        </div>

        <div className="mt-2 h-1 overflow-hidden rounded-full bg-ink">
          <div
            className="h-full rounded-full transition-all duration-700"
            style={{ width: `${Math.min(rec.score * 100, 100)}%`, background: hue }}
          />
        </div>

        {/* why + click */}
        <div className="mt-3 flex items-center gap-2">
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            className={cn(
              "rounded-md border px-2 py-1 font-mono text-[11px] transition-colors",
              open
                ? "border-signal/40 bg-signal/10 text-signal"
                : "border-line text-muted-foreground hover:text-foreground"
            )}
          >
            why?
          </button>
          <button
            type="button"
            onClick={onClick}
            disabled={clicked}
            className={cn(
              "ml-auto flex items-center gap-1.5 rounded-md px-2.5 py-1 text-[12px] font-medium transition-colors",
              clicked
                ? "cursor-default text-live"
                : "text-muted-foreground hover:bg-accent hover:text-foreground"
            )}
          >
            {clicked ? "✓ click logged" : "Register click"}
          </button>
        </div>

        {open && (
          <p className="animate-fade-up mt-2.5 border-t border-line pt-2.5 text-[12.5px] leading-relaxed text-foreground/80">
            {reasonFor(rec, modelUsed, isCold)}
          </p>
        )}
      </div>
    </div>
  );
}

export default function RecommendPage() {
  const [userId, setUserId] = useState("1");
  const [topN, setTopN] = useState(12);
  const [model, setModel] = useState("auto");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<RecommendResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [clicked, setClicked] = useState<Set<number>>(new Set());
  const [hydrated, setHydrated] = useState(false);

  // Restore the last query + results so they survive tab switches and refreshes.
  useEffect(() => {
    try {
      const raw = localStorage.getItem("recsys:last");
      if (raw) {
        const s = JSON.parse(raw);
        if (typeof s.userId === "string") setUserId(s.userId);
        if (typeof s.topN === "number") setTopN(s.topN);
        if (typeof s.model === "string") setModel(s.model);
        if (s.result) setResult(s.result);
        if (Array.isArray(s.clicked)) setClicked(new Set(s.clicked));
      }
    } catch {
      /* ignore malformed cache */
    }
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    try {
      localStorage.setItem(
        "recsys:last",
        JSON.stringify({ userId, topN, model, result, clicked: [...clicked] })
      );
    } catch {
      /* ignore quota / private-mode errors */
    }
  }, [userId, topN, model, result, clicked, hydrated]);

  async function run(e?: React.FormEvent) {
    e?.preventDefault();
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

  async function handleClick(rec: Recommendation, rank: number) {
    if (!result || clicked.has(rec.item_id)) return;
    setClicked((s) => new Set([...s, rec.item_id]));
    try {
      await api.click(result.request_id, result.user_id, rec.item_id, result.model_used, rank);
    } catch {
      /* best-effort */
    }
  }

  return (
    <div className="space-y-6">
      {/* Hero */}
      <section className="pt-2">
        <p className="font-mono text-xs font-semibold uppercase tracking-[0.28em] text-signal">
          Real-time serving
        </p>
        <h1 className="mt-3 font-display text-5xl font-extrabold leading-[0.98] tracking-tight text-foreground sm:text-7xl lg:text-[5.5rem]">
          What should this user{" "}
          <span className="bg-gradient-to-r from-signal via-pink to-cyan bg-clip-text text-transparent">
            watch next?
          </span>
        </h1>
        <p className="mt-6 max-w-6xl text-lg leading-relaxed text-muted-foreground sm:text-xl">
          Every request runs a five-stage pipeline —{" "}
          <span className="text-foreground">FAISS retrieval</span>
          <Info k="faiss" className="mx-0.5 align-middle" /> →{" "}
          <span className="text-foreground">bandit-chosen ranking</span>
          <Info k="bandit" className="mx-0.5 align-middle" /> →{" "}
          <span className="text-foreground">MMR re-ranking</span>
          <Info k="mmr" className="mx-0.5 align-middle" /> — targeting a p99 under 100ms.
        </p>
      </section>

      {/* Query bar */}
      <form
        onSubmit={run}
        className="rounded-xl border border-line bg-panel p-4 sm:p-5"
      >
        <div className="flex flex-col gap-4 lg:flex-row lg:items-end">
          {/* user id */}
          <div className="lg:w-52">
            <label className="mb-1.5 block font-mono text-[13px] uppercase tracking-wide text-muted-foreground">
              User ID · 1–6040
            </label>
            <input
              type="number"
              min={1}
              max={6040}
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              className="h-10 w-full rounded-md border border-line bg-ink px-3 font-mono text-sm text-foreground outline-none transition-colors focus:border-signal/60"
            />
            <div className="mt-1.5 flex flex-wrap gap-1">
              {SAMPLE_USERS.map((u) => (
                <button
                  key={u}
                  type="button"
                  onClick={() => setUserId(String(u))}
                  className="rounded-md border border-line px-2 py-1 font-mono text-[12px] text-muted-foreground transition-colors hover:border-signal/40 hover:text-signal"
                >
                  {u}
                </button>
              ))}
            </div>
          </div>

          {/* top N */}
          <div className="lg:w-48">
            <label className="mb-1.5 flex items-center justify-between font-mono text-[13px] uppercase tracking-wide text-muted-foreground">
              <span>Top N</span>
              <span className="text-signal">{topN}</span>
            </label>
            <input
              type="range"
              min={5}
              max={20}
              step={1}
              value={topN}
              onChange={(e) => setTopN(parseInt(e.target.value))}
              className="h-10 w-full accent-signal"
            />
          </div>

          {/* model */}
          <div className="flex-1">
            <label className="mb-1.5 flex items-center gap-1 font-mono text-[13px] uppercase tracking-wide text-muted-foreground">
              Model <Info k="bandit" />
            </label>
            <div className="flex gap-1.5">
              {MODELS.map((m) => (
                <button
                  key={m.value}
                  type="button"
                  onClick={() => setModel(m.value)}
                  className={cn(
                    "flex h-10 flex-1 items-center justify-center gap-1 rounded-md border text-[13px] font-medium transition-colors",
                    model === m.value
                      ? "border-signal/60 bg-signal/12 text-signal"
                      : "border-line text-muted-foreground hover:text-foreground"
                  )}
                >
                  {m.label}
                  {model === m.value && <Info k={m.info} />}
                </button>
              ))}
            </div>
          </div>

          {/* run */}
          <button
            type="submit"
            disabled={loading}
            className="flex h-10 items-center justify-center gap-2 rounded-md bg-signal px-6 font-display text-sm font-semibold text-primary-foreground transition-opacity hover:opacity-90 disabled:opacity-60 lg:w-40"
          >
            {loading ? (
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-primary-foreground/40 border-t-primary-foreground" />
            ) : (
              "Run pipeline ▸"
            )}
          </button>
        </div>
      </form>

      {/* Pipeline viz */}
      <Pipeline latency={result?.latency_ms} active={loading} />

      {error && (
        <div className="rounded-lg border border-alert/40 bg-alert-soft px-4 py-3 text-sm text-alert">
          {error.includes("503") || error.toLowerCase().includes("fetch")
            ? "The API is waking from sleep (free tier). Give it ~30s and run again."
            : error}
        </div>
      )}

      {/* Result meta */}
      {result && !loading && (
        <div className="flex flex-wrap items-center gap-x-6 gap-y-2 rounded-lg border border-line bg-panel/50 px-4 py-3 text-sm">
          <span className="text-muted-foreground">
            <span className="font-semibold text-foreground">{result.recommendations.length}</span> picks
            for user <span className="font-semibold text-foreground">{result.user_id}</span>
          </span>
          <span className="flex items-center gap-1.5 text-muted-foreground">
            served by
            <span className="rounded bg-signal/12 px-1.5 py-0.5 font-mono text-xs font-semibold uppercase text-signal">
              {result.model_used}
            </span>
            <Info k={result.model_used === "ncf" ? "neumf" : "svd"} />
          </span>
          <span className="flex items-center gap-1.5 text-muted-foreground">
            cold start
            <span
              className={cn(
                "rounded px-1.5 py-0.5 font-mono text-xs font-semibold",
                result.is_cold_start ? "bg-signal-soft text-signal" : "bg-live-soft text-live"
              )}
            >
              {result.is_cold_start ? "YES" : "NO"}
            </span>
            <Info k="coldstart" />
          </span>
          <span className="ml-auto flex items-center gap-1.5 text-muted-foreground">
            <span className="font-mono text-xs">click a title to teach the bandit</span>
            <Info k="registerclick" />
          </span>
        </div>
      )}

      {/* Results grid */}
      {!result && !loading && !error && (
        <div className="flex h-64 flex-col items-center justify-center rounded-xl border border-dashed border-line text-center">
          <p className="font-display text-lg text-foreground">Pick a user, run the pipeline.</p>
          <p className="mt-1 text-sm text-muted-foreground">
            Try one of the sample IDs above, or type any user from 1–6040.
          </p>
        </div>
      )}

      {result && !loading && (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {result.recommendations.map((rec, i) => (
            <RecCard
              key={rec.item_id}
              rec={rec}
              rank={i + 1}
              modelUsed={result.model_used}
              isCold={result.is_cold_start}
              clicked={clicked.has(rec.item_id)}
              onClick={() => handleClick(rec, i + 1)}
            />
          ))}
        </div>
      )}
    </div>
  );
}
