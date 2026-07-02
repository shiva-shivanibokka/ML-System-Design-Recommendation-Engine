"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
  const [active, setActive] = useState<Set<string>>(new Set()); // genre filter (empty = all)
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
    <div className="space-y-6">
      <section className="pt-2">
        <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-signal">
          The catalog, as the model sees it
        </p>
        <h1 className="mt-2 font-display text-3xl font-bold tracking-tight text-foreground">
          Every movie is a point in embedding space.
        </h1>
        <p className="mt-2 max-w-2xl text-[15px] text-muted-foreground">
          A 2D t-SNE projection
          <Info k="embedding" className="mx-0.5 align-middle" /> of all{" "}
          {map ? map.count.toLocaleString() : "3,533"} NeuMF item vectors. Nearby movies are ones the
          model treats as similar — retrieval literally searches this neighborhood.
        </p>
      </section>

      {error && (
        <div className="rounded-lg border border-alert/40 bg-alert-soft px-4 py-3 text-sm text-alert">
          Couldn&apos;t load the embedding map (the API may be waking — retry in ~30s).
        </div>
      )}

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[1fr_300px]">
        {/* Galaxy */}
        <div className="rounded-xl border border-line bg-panel p-3">
          <Galaxy points={map?.points ?? []} active={active} highlight={highlight} />
        </div>

        {/* Controls */}
        <div className="space-y-5">
          <div className="rounded-xl border border-line bg-panel p-4">
            <h2 className="mb-2 font-mono text-[11px] uppercase tracking-[0.16em] text-muted-foreground">
              Overlay a user&apos;s picks
            </h2>
            <div className="flex gap-2">
              <input
                type="number"
                placeholder="user ID"
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                className="h-9 w-full rounded-md border border-line bg-ink px-3 font-mono text-sm outline-none focus:border-signal/60"
              />
              <button
                onClick={overlayUser}
                className="h-9 shrink-0 rounded-md bg-signal px-3 text-sm font-semibold text-primary-foreground hover:opacity-90"
              >
                Show
              </button>
            </div>
            {highlightUser && (
              <p className="mt-2 flex items-center justify-between text-xs text-muted-foreground">
                <span>
                  <span className="text-signal">◆</span> {highlight.size} picks for user {highlightUser}
                </span>
                <button
                  onClick={() => {
                    setHighlight(new Set());
                    setHighlightUser(null);
                  }}
                  className="text-muted-foreground hover:text-foreground"
                >
                  clear
                </button>
              </p>
            )}
            <p className="mt-2 text-xs text-muted-foreground">
              See how one user&apos;s recommendations cluster in the space.
            </p>
          </div>

          {/* Genre legend / filter */}
          <div className="rounded-xl border border-line bg-panel p-4">
            <h2 className="mb-3 font-mono text-[11px] uppercase tracking-[0.16em] text-muted-foreground">
              Genres {active.size > 0 && `· ${active.size} shown`}
            </h2>
            <div className="flex flex-wrap gap-1.5">
              {(map?.genres ?? []).map((g) => {
                const on = active.size === 0 || active.has(g);
                return (
                  <button
                    key={g}
                    onClick={() => toggleGenre(g)}
                    className={cn(
                      "flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] transition-opacity",
                      on ? "border-line opacity-100" : "border-line opacity-35"
                    )}
                  >
                    <span className="h-2 w-2 rounded-full" style={{ background: genreHue(g) }} />
                    {g}
                  </button>
                );
              })}
            </div>
            {active.size > 0 && (
              <button
                onClick={() => setActive(new Set())}
                className="mt-2 font-mono text-[11px] text-muted-foreground hover:text-signal"
              >
                reset filter
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Model card */}
      <section>
        <div className="mb-3 flex items-center gap-1.5">
          <h2 className="font-display text-xl font-bold text-foreground">Model card</h2>
          <span className="font-mono text-xs text-muted-foreground">· offline evaluation</span>
        </div>
        <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
          {metrics &&
            Object.entries(metrics.models).map(([key, m]) => (
              <div key={key} className="rounded-xl border border-line bg-panel p-5">
                <div className="flex items-center justify-between">
                  <span className="font-display text-lg font-bold text-foreground">{m.name}</span>
                  <span className="font-mono text-xs text-muted-foreground">{m.family}</span>
                </div>
                <div className="mt-4 grid grid-cols-2 gap-3">
                  <Metric label="HR@10" value={m.hr_at_10} infoKey="hr" />
                  <Metric label="NDCG@10" value={m.ndcg_at_10} infoKey="ndcg" />
                </div>
                <p className="mt-3 text-xs leading-relaxed text-muted-foreground">{m.note}</p>
              </div>
            ))}
        </div>
        {metrics && (
          <p className="mt-3 font-mono text-xs text-muted-foreground">
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
    <div className="rounded-lg border border-line bg-ink/50 p-3">
      <p className="flex items-center gap-1 font-mono text-[10px] uppercase tracking-wide text-muted-foreground">
        {label} <Info k={infoKey} />
      </p>
      <p className="mt-1 font-mono text-2xl font-bold text-signal">{value.toFixed(4)}</p>
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
  const [size, setSize] = useState({ w: 800, h: 520 });

  // Track container width
  useEffect(() => {
    if (!wrapRef.current) return;
    const el = wrapRef.current;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: 520 }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const project = useCallback(
    (p: EmbeddingPoint) => {
      const pad = 24;
      const sx = pad + ((p.x + 1) / 2) * (size.w - 2 * pad);
      const sy = pad + ((p.y + 1) / 2) * (size.h - 2 * pad);
      return { sx, sy };
    },
    [size]
  );

  // Draw
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = size.w * dpr;
    canvas.height = size.h * dpr;
    const ctx = canvas.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size.w, size.h);

    const filtered = (g: string) => active.size === 0 || active.has(g);

    // base points
    for (const p of points) {
      const { sx, sy } = project(p);
      const on = filtered(p.genre);
      ctx.beginPath();
      ctx.arc(sx, sy, on ? 2.1 : 1.2, 0, Math.PI * 2);
      ctx.fillStyle = genreHue(p.genre);
      ctx.globalAlpha = on ? 0.7 : 0.08;
      ctx.fill();
    }
    ctx.globalAlpha = 1;

    // highlighted (a user's recommendations)
    if (highlight.size) {
      for (const p of points) {
        if (!highlight.has(p.item_id)) continue;
        const { sx, sy } = project(p);
        ctx.beginPath();
        ctx.arc(sx, sy, 6, 0, Math.PI * 2);
        ctx.fillStyle = "#E7A33A";
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#0E0C0B";
        ctx.stroke();
      }
    }

    // hover ring
    if (hover) {
      ctx.beginPath();
      ctx.arc(hover.sx, hover.sy, 7, 0, Math.PI * 2);
      ctx.strokeStyle = "#EFE6D8";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }, [points, active, highlight, hover, project, size]);

  // Hover hit-test
  function onMove(e: React.MouseEvent) {
    const rect = canvasRef.current!.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    let best: { p: EmbeddingPoint; sx: number; sy: number } | null = null;
    let bestD = 100;
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
    <div ref={wrapRef} className="relative w-full" style={{ height: 520 }}>
      <canvas
        ref={canvasRef}
        style={{ width: size.w, height: size.h }}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
        className="rounded-lg"
      />
      {hover && (
        <div
          className="pointer-events-none absolute z-10 max-w-[200px] -translate-x-1/2 rounded-md border border-line bg-popover px-2.5 py-1.5 shadow-xl"
          style={{
            left: Math.min(Math.max(hover.sx, 90), size.w - 90),
            top: hover.sy + 12,
          }}
        >
          <p className="text-xs font-semibold leading-tight text-foreground">{hover.p.title}</p>
          <p className="mt-0.5 flex items-center gap-1.5 font-mono text-[10px] text-muted-foreground">
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
