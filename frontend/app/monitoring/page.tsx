"use client";

import { useCallback, useEffect, useState } from "react";
import { api, type MonitoringHealth, type SystemHealth } from "@/lib/api";
import { Info } from "@/components/info";
import { cn } from "@/lib/utils";

function Dot({ ok }: { ok: boolean }) {
  return (
    <span className={cn("h-2 w-2 rounded-full", ok ? "bg-live" : "bg-alert")} />
  );
}

function Panel({
  title,
  info,
  children,
}: {
  title: string;
  info?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-xl border border-line bg-panel p-5">
      <div className="mb-4 flex items-center gap-1.5">
        <h2 className="font-mono text-[11px] uppercase tracking-[0.16em] text-muted-foreground">
          {title}
        </h2>
        {info}
      </div>
      {children}
    </div>
  );
}

// Build human-readable alert lines from the backend's boolean flags.
function deriveAlerts(h: MonitoringHealth): string[] {
  const out: string[] = [];
  if (h.coverage_alert) {
    out.push(
      `Catalog coverage is ${(h.catalog_coverage * 100).toFixed(1)}% — below the 10% floor. Recommendations are concentrating on too few titles.`
    );
  }
  Object.entries(h.ctr_alerts ?? {}).forEach(([m, a]) => {
    if (a) out.push(`${m.toUpperCase()} click-through rate dropped below its healthy band.`);
  });
  Object.entries(h.psi_alerts ?? {}).forEach(([m, a]) => {
    if (a) out.push(`${m.toUpperCase()} score distribution drifted (PSI over 0.2) — model may be going stale.`);
  });
  return out;
}

export default function MonitoringPage() {
  const [health, setHealth] = useState<MonitoringHealth | null>(null);
  const [sys, setSys] = useState<SystemHealth | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [h, s] = await Promise.all([api.monitoringHealth(), api.health()]);
      setHealth(h);
      setSys(s);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load monitoring");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const coverage = health?.catalog_coverage ?? 0;
  const covPct = coverage * 100;
  const alerts = health ? deriveAlerts(health) : [];

  return (
    <div className="space-y-6">
      <section className="flex items-end justify-between pt-2">
        <div>
          <p className="font-mono text-xs font-semibold uppercase tracking-[0.28em] text-signal">
            Model health
          </p>
          <h1 className="mt-3 font-display text-5xl font-extrabold leading-[0.98] tracking-tight text-foreground sm:text-6xl lg:text-7xl">
            Is the model still{" "}
            <span className="bg-gradient-to-r from-live via-signal to-pink bg-clip-text text-transparent">
              healthy?
            </span>
          </h1>
          <p className="mt-5 max-w-6xl text-lg text-muted-foreground sm:text-xl">
            Three staleness signals a real team watches: click-through rate, catalog
            coverage, and score drift. When any crosses a threshold, it&apos;s time to retrain.
          </p>
        </div>
        <button
          onClick={load}
          disabled={loading}
          className="rounded-md border border-line px-3 py-1.5 font-mono text-xs text-muted-foreground transition-colors hover:border-signal/40 hover:text-signal"
        >
          {loading ? "…" : "↻ refresh"}
        </button>
      </section>

      {error && (
        <div className="rounded-lg border border-alert/40 bg-alert-soft px-4 py-3 text-sm text-alert">
          {error.includes("503") || error.toLowerCase().includes("fetch")
            ? "The API is waking from sleep (free tier). Give it ~30s and refresh."
            : error}
        </div>
      )}

      {/* Alert / all-clear banner */}
      {health && (
        <div
          className={cn(
            "rounded-xl border px-4 py-3",
            alerts.length ? "border-signal/40 bg-signal-soft" : "border-live/30 bg-live-soft"
          )}
        >
          {alerts.length ? (
            <div className="space-y-1">
              <p className="font-mono text-xs font-semibold uppercase tracking-wide text-signal">
                ⚠ {alerts.length} staleness signal{alerts.length > 1 ? "s" : ""} active
              </p>
              {alerts.map((a, i) => (
                <p key={i} className="text-sm text-foreground/80">
                  {a}
                </p>
              ))}
              <p className="pt-1 text-xs text-muted-foreground">
                (Expected on a fresh deploy — few clicks logged yet, so coverage is naturally low.)
              </p>
            </div>
          ) : (
            <p className="font-mono text-xs font-semibold uppercase tracking-wide text-live">
              ✓ all signals within thresholds
            </p>
          )}
        </div>
      )}

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
        {/* Service health */}
        <Panel title="Service health">
          {sys ? (
            <div className="space-y-2.5 text-sm">
              {[
                ["NeuMF model", sys.models?.ncf],
                ["SVD model", sys.models?.svd],
                ["FAISS index", sys.models?.faiss],
                ["Redis online store", sys.redis],
                ["PostgreSQL", sys.postgres],
              ].map(([label, ok]) => (
                <div key={label as string} className="flex items-center gap-2.5">
                  <Dot ok={!!ok} />
                  <span className={cn(!ok && "text-muted-foreground")}>{label as string}</span>
                </div>
              ))}
              <div className="mt-3 flex items-center justify-between border-t border-line pt-3 font-mono text-xs">
                <span className="text-muted-foreground">catalog</span>
                <span className="text-foreground">
                  {(sys.n_total_items ?? 0).toLocaleString()} items
                </span>
              </div>
            </div>
          ) : (
            <Skeleton rows={5} />
          )}
        </Panel>

        {/* Coverage */}
        <Panel title="Catalog coverage · 24h" info={<Info k="coverage" />}>
          {health ? (
            <div>
              <div className="flex items-end gap-2">
                <span
                  className={cn(
                    "font-mono text-5xl font-bold",
                    coverage < 0.1 ? "text-alert" : coverage < 0.3 ? "text-signal" : "text-live"
                  )}
                >
                  {covPct.toFixed(1)}
                </span>
                <span className="mb-1.5 text-lg text-muted-foreground">%</span>
              </div>
              <p className="mt-1 text-xs text-muted-foreground">of catalog recommended</p>
              <div className="mt-4 h-2 overflow-hidden rounded-full bg-ink">
                <div
                  className={cn(
                    "h-full rounded-full transition-all",
                    coverage < 0.1 ? "bg-alert" : coverage < 0.3 ? "bg-signal" : "bg-live"
                  )}
                  style={{ width: `${Math.max(Math.min(covPct, 100), 1)}%` }}
                />
              </div>
              <div className="mt-1.5 flex justify-between font-mono text-[10px] text-muted-foreground">
                <span>0</span>
                <span className="text-signal">floor 10%</span>
                <span>100</span>
              </div>
            </div>
          ) : (
            <Skeleton rows={3} />
          )}
        </Panel>

        {/* CTR by model */}
        <Panel title="CTR by model · 7d" info={<Info k="ctr" />}>
          {health ? (
            <div className="space-y-3">
              {Object.entries(health.ctrs ?? {}).map(([m, ctr]) => (
                <div key={m}>
                  <div className="flex items-center justify-between text-sm">
                    <span className="font-mono uppercase text-muted-foreground">{m}</span>
                    <span className="font-mono font-semibold text-foreground">
                      {(ctr * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-ink">
                    <div
                      className="h-full rounded-full bg-signal transition-all"
                      style={{ width: `${Math.min(ctr * 100 * 3, 100)}%` }}
                    />
                  </div>
                </div>
              ))}
              <p className="pt-1 text-xs text-muted-foreground">
                Simulated via the &quot;Register click&quot; buttons on the Recommend page.
              </p>
            </div>
          ) : (
            <Skeleton rows={3} />
          )}
        </Panel>

        {/* PSI drift */}
        <Panel title="Score distribution drift · PSI" info={<Info k="psi" />}>
          {health ? (
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              {Object.entries(health.psi_scores ?? {}).map(([m, psi]) => {
                const level = psi > 0.2 ? "alert" : psi > 0.1 ? "signal" : "live";
                const pct = Math.min(psi * 400, 100);
                return (
                  <div key={m}>
                    <div className="flex items-center justify-between text-sm">
                      <span className="font-mono uppercase text-muted-foreground">{m}</span>
                      <span
                        className={cn(
                          "font-mono font-semibold",
                          level === "alert"
                            ? "text-alert"
                            : level === "signal"
                              ? "text-signal"
                              : "text-live"
                        )}
                      >
                        {psi.toFixed(3)}
                      </span>
                    </div>
                    <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-ink">
                      <div
                        className={cn(
                          "h-full rounded-full transition-all",
                          level === "alert" ? "bg-alert" : level === "signal" ? "bg-signal" : "bg-live"
                        )}
                        style={{ width: `${Math.max(pct, 2)}%` }}
                      />
                    </div>
                  </div>
                );
              })}
              <p className="col-span-full text-xs text-muted-foreground">
                Alert at 0.2. Full time-series histograms live in Grafana / Prometheus.
              </p>
            </div>
          ) : (
            <Skeleton rows={2} />
          )}
        </Panel>

        {/* Latency budget reference */}
        <Panel title="Latency budget" info={<Info k="latency" />}>
          <div className="space-y-2 font-mono text-xs">
            {[
              ["Cache", 5],
              ["FAISS retrieval", 20],
              ["Feature fetch", 10],
              ["Ranking", 50],
              ["Re-rank", 10],
              ["Total p99", 100],
            ].map(([label, ms], i, arr) => (
              <div
                key={label as string}
                className={cn(
                  "flex items-center justify-between",
                  i === arr.length - 1 && "border-t border-line pt-2 text-foreground"
                )}
              >
                <span className={i === arr.length - 1 ? "text-foreground" : "text-muted-foreground"}>
                  {label as string}
                </span>
                <span className={i === arr.length - 1 ? "text-signal" : "text-foreground"}>
                  {ms as number}ms
                </span>
              </div>
            ))}
            <p className="pt-1 text-[11px] text-muted-foreground">
              Live per-request latency lights up the pipeline on the Recommend page.
            </p>
          </div>
        </Panel>
      </div>
    </div>
  );
}

function Skeleton({ rows }: { rows: number }) {
  return (
    <div className="space-y-2">
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="h-5 animate-pulse rounded bg-muted" />
      ))}
    </div>
  );
}
