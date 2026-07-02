"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  api,
  genreHue,
  type EmbeddingMap,
  type EmbeddingPoint,
  type ModelMetrics,
} from "@/lib/api";
import { Info } from "@/components/info";
import { cn } from "@/lib/utils";

export default function CatalogPage() {
  const [map, setMap] = useState<EmbeddingMap | null>(null);
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [active, setActive] = useState<Set<string>>(new Set());
  const [userId, setUserId] = useState("");
  const [highlight, setHighlight] = useState<Set<number>>(new Set());
  const [highlightUser, setHighlightUser] = useState<string | null>(null);

  useEffect(() => {
    api.embeddingMap().then(setMap).catch((e) => setError(String(e)));
    api.modelMetrics().then(setMetrics).catch(() => {});
  }, []);

  const toggleGenre = (g: string) =>
    setActive((s) => {
      const n = new Set(s);
      if (n.has(g)) n.delete(g);
      else n.add(g);
      return n;
    });

  async function overlayUser() {
    const id = parseInt(userId);
    if (isNaN(id)) return;
    try {
      const r = await api.recommend(id, 12);
      setHighlight(new Set(r.recommendations.map((x) => x.item_id)));
      setHighlightUser(String(id));
    } catch {
      /* ignore */
    }
  }

  return (
    <div className="space-y-7">
      <section className="pt-2">
        <p className="font-mono text-xs font-semibold uppercase tracking-[0.28em] text-signal">
          The catalog, as the model sees it
        </p>
        <h1 className="mt-3 font-display text-5xl font-extrabold leading-[0.98] tracking-tight text-foreground sm:text-7xl lg:text-[5.5rem]">
          Every movie is a{" "}
          <span className="bg-gradient-to-r from-cyan via-signal to-pink bg-clip-text text-transparent">
            point in space.
          </span>
        </h1>
        <p className="mt-6 flex max-w-5xl flex-wrap items-center gap-x-1.5 text-lg leading-relaxed text-muted-foreground sm:text-xl">
          A 2D t-SNE projection <Info k="embedding" /> of all{" "}
          {map ? map.count.toLocaleString() : "3,533"} NeuMF item vectors. Nearby movies are ones the
          model treats as similar — retrieval literally searches this neighborhood.
        </p>
      </section>

      {error && (
        <div className="rounded-lg border border-alert/40 bg-alert-soft px-4 py-3 text-sm text-alert">
          Couldn&apos;t load the embedding map (the API may be waking — retry in ~30s).
        </div>
      )}

      {/* Control strip */}
      <div className="flex flex-col gap-3 rounded-2xl border border-line bg-panel p-4 lg:flex-row lg:items-center">
        <div className="flex shrink-0 items-center gap-2">
          <span className="font-mono text-[11px] uppercase tracking-wide text-muted-foreground">
            Overlay picks
          </span>
          <input
            type="number"
            placeholder="user ID"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && overlayUser()}
            className="h-9 w-28 rounded-lg border border-line bg-ink px-3 font-mono text-sm outline-none focus:border-signal"
          />
          <button
            onClick={overlayUser}
            className="h-9 rounded-lg bg-signal px-3 text-sm font-semibold text-white transition-opacity hover:opacity-90"
          >
            Show
          </button>
          {highlightUser && (
            <button
              onClick={() => {
                setHighlight(new Set());
                setHighlightUser(null);
              }}
              className="font-mono text-[11px] text-muted-foreground hover:text-foreground"
            >
              clear ✕
            </button>
          )}
        </div>

        <div className="h-px w-full bg-line lg:h-8 lg:w-px" />

        {/* Genre legend / filter */}
        <div className="flex flex-1 flex-wrap items-center gap-1.5">
          {(map?.genres ?? []).map((g) => {
            const on = active.size === 0 || active.has(g);
            return (
              <button
                key={g}
                onClick={() => toggleGenre(g)}
                className={cn(
                  "flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] font-medium transition-all",
                  on
                    ? "border-line bg-raised opacity-100"
                    : "border-transparent bg-muted opacity-40"
                )}
                style={on ? { borderColor: `${genreHue(g)}66` } : undefined}
              >
                <span className="h-2.5 w-2.5 rounded-full" style={{ background: genreHue(g) }} />
                {g}
              </button>
            );
          })}
          {active.size > 0 && (
            <button
              onClick={() => setActive(new Set())}
              className="ml-1 font-mono text-[11px] text-signal hover:underline"
            >
              reset
            </button>
          )}
        </div>
      </div>

      {/* Big galaxy */}
      <div className="overflow-hidden rounded-2xl border border-line bg-[radial-gradient(circle_at_50%_35%,#1b1542,#0a0715)] p-2">
        <Galaxy points={map?.points ?? []} active={active} highlight={highlight} />
      </div>

      {highlightUser && (
        <p className="-mt-4 text-center font-mono text-xs text-muted-foreground">
          <span className="text-signal">◆</span> {highlight.size} recommendations for user{" "}
          {highlightUser} highlighted above
        </p>
      )}

      {/* Model card */}
      <section>
        <div className="mb-4 flex items-baseline gap-2">
          <h2 className="font-display text-2xl font-bold text-foreground">Model card</h2>
          <span className="font-mono text-xs text-muted-foreground">· offline evaluation</span>
        </div>
        <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
          {metrics &&
            Object.entries(metrics.models).map(([key, m]) => (
              <div key={key} className="rounded-2xl border border-line bg-panel p-6">
                <div className="flex items-center justify-between">
                  <span className="font-display text-xl font-bold text-foreground">{m.name}</span>
                  <span className="font-mono text-xs text-muted-foreground">{m.family}</span>
                </div>
                <div className="mt-5 grid grid-cols-2 gap-4">
                  <Metric label="HR@10" value={m.hr_at_10} infoKey="hr" />
                  <Metric label="NDCG@10" value={m.ndcg_at_10} infoKey="ndcg" />
                </div>
                <p className="mt-4 text-sm leading-relaxed text-muted-foreground">{m.note}</p>
              </div>
            ))}
        </div>
        {metrics && (
          <p className="mt-4 font-mono text-xs text-muted-foreground">
            {metrics.dataset} · {metrics.n_users.toLocaleString()} users ·{" "}
            {metrics.n_items.toLocaleString()} items · {metrics.eval_protocol}
          </p>
        )}
      </section>
    </div>
  );
}

function Metric({
  label,
  value,
  infoKey,
}: {
  label: string;
  value: number;
  infoKey: "hr" | "ndcg";
}) {
  return (
    <div className="rounded-xl border border-line bg-raised p-4">
      <p className="flex items-center gap-1 font-mono text-[11px] uppercase tracking-wide text-muted-foreground">
        {label} <Info k={infoKey} />
      </p>
      <p className="mt-1 font-mono text-3xl font-extrabold text-signal">{value.toFixed(4)}</p>
    </div>
  );
}

/** Canvas scatter of the embedding projection — smooth at thousands of points. */
function Galaxy({
  points,
  active,
  highlight,
}: {
  points: EmbeddingPoint[];
  active: Set<string>;
  highlight: Set<number>;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<{ p: EmbeddingPoint; sx: number; sy: number } | null>(null);
  const [size, setSize] = useState({ w: 900, h: 680 });

  useEffect(() => {
    if (!wrapRef.current) return;
    const el = wrapRef.current;
    const ro = new ResizeObserver(() =>
      setSize({ w: el.clientWidth, h: Math.max(Math.min(el.clientWidth * 0.62, 760), 460) })
    );
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const project = useCallback(
    (p: EmbeddingPoint) => {
      const pad = 28;
      return {
        sx: pad + ((p.x + 1) / 2) * (size.w - 2 * pad),
        sy: pad + ((p.y + 1) / 2) * (size.h - 2 * pad),
      };
    },
    [size]
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = size.w * dpr;
    canvas.height = size.h * dpr;
    const ctx = canvas.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size.w, size.h);

    const shown = (g: string) => active.size === 0 || active.has(g);

    // Additive blending makes the stars glow and dense regions brighten.
    ctx.globalCompositeOperation = "lighter";
    for (const p of points) {
      const { sx, sy } = project(p);
      const on = shown(p.genre);
      ctx.beginPath();
      ctx.arc(sx, sy, on ? 3 : 1.6, 0, Math.PI * 2);
      ctx.fillStyle = genreHue(p.genre);
      ctx.globalAlpha = on ? 0.9 : 0.12;
      ctx.fill();
    }
    ctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = 1;

    if (highlight.size) {
      for (const p of points) {
        if (!highlight.has(p.item_id)) continue;
        const { sx, sy } = project(p);
        ctx.beginPath();
        ctx.arc(sx, sy, 7.5, 0, Math.PI * 2);
        ctx.shadowBlur = 16;
        ctx.shadowColor = "#8B5CF6";
        ctx.fillStyle = "#C4B5FD";
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.lineWidth = 2.5;
        ctx.strokeStyle = "#FFFFFF";
        ctx.stroke();
      }
    }

    if (hover) {
      ctx.beginPath();
      ctx.arc(hover.sx, hover.sy, 9, 0, Math.PI * 2);
      ctx.strokeStyle = "#F4F1FF";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }, [points, active, highlight, hover, project, size]);

  function onMove(e: React.MouseEvent) {
    const rect = canvasRef.current!.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    let best: { p: EmbeddingPoint; sx: number; sy: number } | null = null;
    let bestD = 120;
    for (const p of points) {
      if (active.size > 0 && !active.has(p.genre)) continue;
      const { sx, sy } = project(p);
      const d = (sx - mx) ** 2 + (sy - my) ** 2;
      if (d < bestD) {
        bestD = d;
        best = { p, sx, sy };
      }
    }
    setHover(best);
  }

  return (
    <div ref={wrapRef} className="relative w-full" style={{ height: size.h }}>
      <canvas
        ref={canvasRef}
        style={{ width: size.w, height: size.h }}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
        className="rounded-xl"
      />
      {hover && (
        <div
          className="pointer-events-none absolute z-10 max-w-[220px] -translate-x-1/2 rounded-lg border border-signal/40 bg-raised px-3 py-2 shadow-[0_10px_30px_-6px_rgba(0,0,0,0.6)]"
          style={{
            left: Math.min(Math.max(hover.sx, 110), size.w - 110),
            top: hover.sy + 14,
          }}
        >
          <p className="text-[13px] font-semibold leading-tight text-foreground">{hover.p.title}</p>
          <p className="mt-0.5 flex items-center gap-1.5 font-mono text-[11px] text-muted-foreground">
            <span className="h-2 w-2 rounded-full" style={{ background: genreHue(hover.p.genre) }} />
            {hover.p.genre}
          </p>
        </div>
      )}
      {!points.length && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="h-8 w-8 animate-spin rounded-full border-2 border-line border-t-signal" />
        </div>
      )}
    </div>
  );
}
