"use client";

import { useCallback, useEffect, useState } from "react";
import { api, type BanditState } from "@/lib/api";
import { Info } from "@/components/info";
import { cn } from "@/lib/utils";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";

// Beta(α, β) PDF via a Lanczos log-gamma, so we can draw the belief curves.
function lgamma(x: number): number {
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  x -= 1;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
    1.5056327351493116e-7,
  ];
  const t = x + 7.5;
  let a = c[0];
  for (let i = 1; i < 9; i++) a += c[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}
function betaPDF(x: number, alpha: number, beta: number): number {
  if (x <= 0 || x >= 1) return 0;
  const lp =
    (alpha - 1) * Math.log(x) +
    (beta - 1) * Math.log(1 - x) -
    (lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta));
  return Math.exp(lp);
}

const MODEL_COLORS: Record<string, string> = { svd: "#FFB020", ncf: "#22D3EE" };
const modelColor = (m: string) => MODEL_COLORS[m] ?? "#8A8A8A";

export default function BanditPage() {
  const [state, setState] = useState<BanditState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [resetting, setResetting] = useState(false);

  const load = useCallback(async () => {
    try {
      setState(await api.banditState());
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load bandit state");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
    const t = setInterval(load, 4000); // live poll while watching
    return () => clearInterval(t);
  }, [load]);

  async function reset() {
    setResetting(true);
    try {
      await fetch(`${process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}/admin/bandit/reset`, {
        method: "POST",
      });
      await load();
    } catch {
      /* ignore */
    } finally {
      setResetting(false);
    }
  }

  const models = state ? Object.keys(state.alphas) : [];
  const chartData =
    state &&
    Array.from({ length: 160 }, (_, i) => {
      const x = (i + 1) / 161;
      const row: Record<string, number> = { x };
      for (const m of models) row[m] = betaPDF(x, state.alphas[m] ?? 1, state.betas[m] ?? 1);
      return row;
    });

  return (
    <div className="space-y-6">
      <section className="flex items-end justify-between pt-2">
        <div>
          <p className="font-mono text-xs font-semibold uppercase tracking-[0.28em] text-signal">
            Online experimentation
          </p>
          <h1 className="mt-3 font-display text-5xl font-extrabold leading-[0.98] tracking-tight text-foreground sm:text-6xl lg:text-7xl">
            Which model wins?{" "}
            <span className="bg-gradient-to-r from-signal via-pink to-cyan bg-clip-text text-transparent">
              Let traffic decide.
            </span>
          </h1>
          <p className="mt-5 max-w-5xl text-lg text-muted-foreground sm:text-xl">
            A Thompson-Sampling bandit
            <Info k="bandit" className="mx-0.5 align-middle" /> holds a belief about each model&apos;s
            click rate and routes each request to whichever looks best right now. Click titles on the
            Recommend page and watch these beliefs sharpen.
          </p>
        </div>
        <button
          onClick={reset}
          disabled={resetting}
          className="rounded-md border border-line px-3 py-1.5 font-mono text-xs text-muted-foreground transition-colors hover:border-alert/50 hover:text-alert"
        >
          {resetting ? "…" : "↺ reset"}
        </button>
      </section>

      {error && (
        <div className="rounded-lg border border-alert/40 bg-alert-soft px-4 py-3 text-sm text-alert">
          {error.includes("503") || error.toLowerCase().includes("fetch")
            ? "The API is waking from sleep (free tier). Give it ~30s."
            : error}
        </div>
      )}

      {state && (
        <>
          {/* Winner / status */}
          <div
            className={cn(
              "rounded-xl border px-4 py-3",
              state.winner ? "border-signal/40 bg-signal-soft" : "border-line bg-panel/60"
            )}
          >
            {state.winner ? (
              <p className="text-sm">
                <span className="font-mono font-semibold uppercase text-signal">
                  🏆 {state.winner} declared winner
                </span>
                <span className="text-muted-foreground"> — 95% credible the better model.</span>
              </p>
            ) : (
              <p className="text-sm text-muted-foreground">
                <span className="font-mono uppercase text-foreground">exploring</span> — not enough
                evidence yet to declare a winner. Beliefs update as clicks arrive.
              </p>
            )}
          </div>

          <div className="grid grid-cols-1 gap-5 lg:grid-cols-[1.5fr_1fr]">
            {/* Belief curves */}
            <div className="rounded-xl border border-line bg-panel p-5">
              <div className="mb-1 flex items-center gap-1.5">
                <h2 className="font-mono text-[11px] uppercase tracking-[0.16em] text-muted-foreground">
                  Belief curves · Beta(α, β)
                </h2>
                <Info k="alphabeta" />
              </div>
              <p className="mb-3 text-xs text-muted-foreground">
                Taller, narrower = more confident. The curves start identical and separate as
                evidence accumulates.
              </p>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData ?? []} margin={{ top: 8, right: 8, bottom: 4, left: -18 }}>
                    <CartesianGrid stroke="#2C2550" strokeDasharray="2 4" vertical={false} />
                    <XAxis
                      dataKey="x"
                      type="number"
                      domain={[0, 1]}
                      tick={{ fontSize: 10, fill: "#9B93C4", fontFamily: "monospace" }}
                      tickFormatter={(v) => v.toFixed(1)}
                      stroke="#2C2550"
                    />
                    <YAxis
                      tick={{ fontSize: 10, fill: "#9B93C4", fontFamily: "monospace" }}
                      stroke="#2C2550"
                    />
                    <Tooltip
                      contentStyle={{
                        background: "#1E1838",
                        border: "1px solid #2C2550",
                        borderRadius: 10,
                        fontSize: 12,
                        fontFamily: "monospace",
                        color: "#F4F1FF",
                        boxShadow: "0 10px 30px -8px rgba(0,0,0,0.6)",
                      }}
                      labelFormatter={(v) => `p(click) = ${Number(v).toFixed(2)}`}
                      formatter={(val: number, name: string) => [val.toFixed(2), name.toUpperCase()]}
                    />
                    {models.map((m) => (
                      <Line
                        key={m}
                        type="monotone"
                        dataKey={m}
                        stroke={modelColor(m)}
                        strokeWidth={2}
                        dot={false}
                        isAnimationActive={false}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Traffic split */}
            <div className="rounded-xl border border-line bg-panel p-5">
              <div className="mb-4 flex items-center gap-1.5">
                <h2 className="font-mono text-[11px] uppercase tracking-[0.16em] text-muted-foreground">
                  Traffic split
                </h2>
                <Info k="trafficsplit" />
              </div>
              <div className="space-y-4">
                {models.map((m) => {
                  const pct = (state.traffic_split?.[m] ?? 0) * 100;
                  return (
                    <div key={m}>
                      <div className="flex items-center justify-between text-sm">
                        <span className="flex items-center gap-2 font-mono uppercase">
                          <span className="h-2.5 w-2.5 rounded-sm" style={{ background: modelColor(m) }} />
                          {m}
                        </span>
                        <span className="font-mono font-semibold text-foreground">{pct.toFixed(0)}%</span>
                      </div>
                      <div className="mt-1.5 h-2.5 overflow-hidden rounded-full bg-ink">
                        <div
                          className="h-full rounded-full transition-all duration-500"
                          style={{ width: `${pct}%`, background: modelColor(m) }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
              <p className="mt-4 text-xs text-muted-foreground">
                Shifts automatically — no fixed 50/50. Winning models earn more traffic.
              </p>
            </div>
          </div>

          {/* Per-model stat cards */}
          <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
            {models.map((m) => (
              <div key={m} className="rounded-xl border border-line bg-panel p-5">
                <div className="mb-4 flex items-center justify-between">
                  <span className="flex items-center gap-2 font-display text-lg font-bold">
                    <span className="h-3 w-3 rounded-sm" style={{ background: modelColor(m) }} />
                    {m.toUpperCase()}
                  </span>
                  <span className="font-mono text-xs text-muted-foreground">
                    {m === "ncf" ? "neural CF" : "matrix factorization"}
                  </span>
                </div>
                <div className="grid grid-cols-3 gap-3">
                  {[
                    ["α wins", (state.alphas[m] ?? 0).toFixed(0)],
                    ["β losses", (state.betas[m] ?? 0).toFixed(0)],
                    ["pulls", (state.total_pulls[m] ?? 0).toString()],
                    ["est. CTR", `${((state.mean_ctrs[m] ?? 0) * 100).toFixed(1)}%`],
                    ["rewards", (state.total_rewards[m] ?? 0).toString()],
                    ["± uncert.", (state.uncertainty?.[m] ?? 0).toFixed(3)],
                  ].map(([label, val]) => (
                    <div key={label} className="rounded-lg border border-line bg-ink/50 p-2.5">
                      <p className="font-mono text-[10px] uppercase tracking-wide text-muted-foreground">
                        {label}
                      </p>
                      <p className="mt-0.5 font-mono text-lg font-semibold text-foreground">{val}</p>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      {loading && !state && (
        <div className="h-64 animate-pulse rounded-xl border border-line bg-panel" />
      )}
    </div>
  );
}
