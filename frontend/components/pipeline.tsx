"use client";

import { cn } from "@/lib/utils";
import { Info } from "@/components/info";

const STAGES: { key: string; label: string; budget: number; sub: string }[] = [
  { key: "cache_check", label: "Cache", budget: 5, sub: "Redis lookup" },
  { key: "candidate_generation", label: "Retrieval", budget: 20, sub: "FAISS ANN" },
  { key: "feature_fetch", label: "Features", budget: 10, sub: "Feast store" },
  { key: "ranking", label: "Ranking", budget: 50, sub: "NeuMF / SVD" },
  { key: "post_ranking", label: "Re-rank", budget: 10, sub: "MMR + caps" },
];

export function Pipeline({
  latency,
  active = false,
}: {
  latency?: Record<string, number>;
  active?: boolean;
}) {
  const total = latency?.total ?? 0;
  const has = !!latency;

  return (
    <div className="rounded-xl border border-line bg-panel/60 p-4 sm:p-5">
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
            Serving pipeline
          </span>
          <Info k="latency" />
        </div>
        <div className="flex items-center gap-2 font-mono text-[11px]">
          <span className="text-muted-foreground">p99 budget 100ms</span>
          {has && (
            <span
              className={cn(
                "rounded px-1.5 py-0.5 font-semibold",
                total <= 100 ? "bg-live-soft text-live" : "bg-alert-soft text-alert"
              )}
            >
              {total.toFixed(0)}ms
            </span>
          )}
        </div>
      </div>

      <div className="flex items-stretch gap-1.5 overflow-x-auto pb-1">
        {STAGES.map((s, i) => {
          const ms = latency?.[s.key] ?? 0;
          const over = has && ms > s.budget;
          const fill = Math.min((ms / s.budget) * 100, 100);
          return (
            <div key={s.key} className="flex min-w-[104px] flex-1 items-center gap-1.5">
              <div
                className={cn(
                  "relative flex-1 rounded-lg border p-2.5 transition-colors",
                  has
                    ? over
                      ? "border-alert/40 bg-alert-soft"
                      : "border-live/30 bg-live-soft"
                    : "border-line bg-raised/40"
                )}
              >
                <div className="flex items-baseline justify-between">
                  <span className="text-xs font-semibold text-foreground">{s.label}</span>
                  <span
                    className={cn(
                      "font-mono text-[11px]",
                      !has ? "text-muted-foreground" : over ? "text-alert" : "text-live"
                    )}
                  >
                    {has ? `${ms.toFixed(1)}` : "—"}
                  </span>
                </div>
                <span className="block font-mono text-[9px] uppercase tracking-wide text-muted-foreground">
                  {s.sub}
                </span>
                <div className="mt-2 h-1 overflow-hidden rounded-full bg-ink/60">
                  <div
                    className={cn(
                      "h-full rounded-full transition-all duration-500",
                      !has ? "bg-line" : over ? "bg-alert" : "bg-live"
                    )}
                    style={{ width: has ? `${Math.max(fill, 4)}%` : "0%" }}
                  />
                </div>
              </div>
              {i < STAGES.length - 1 && (
                <div className="relative hidden h-px w-3 overflow-visible bg-line sm:block">
                  {active && (
                    <span className="animate-flow absolute -top-[3px] h-[7px] w-[7px] rounded-full bg-signal shadow-[0_0_8px_2px] shadow-signal/50" />
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
