"use client";

import { useCallback, useEffect, useState } from "react";
import { api, type BanditState } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { RefreshCw, Trophy, Loader2, AlertCircle } from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
} from "recharts";

// --- Beta distribution PDF (Lanczos approximation) ---
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
  const logPDF = (alpha - 1) * Math.log(x) + (beta - 1) * Math.log(1 - x) - (lgamma(alpha) + lgamma(beta) - lgamma(alpha + beta));
  return Math.exp(logPDF);
}

function buildBetaChartData(models: string[], alphas: Record<string, number>, betas: Record<string, number>) {
  const points = 200;
  return Array.from({ length: points }, (_, i) => {
    const x = (i + 1) / (points + 1);
    const entry: Record<string, number> = { x };
    for (const m of models) {
      entry[m] = betaPDF(x, alphas[m] ?? 1, betas[m] ?? 1);
    }
    return entry;
  });
}

const MODEL_COLORS: Record<string, string> = {
  svd: "#8b5cf6",
  ncf: "#06b6d4",
};

function colorFor(model: string, i: number) {
  return MODEL_COLORS[model] ?? ["#f59e0b", "#10b981"][i % 2];
}

export default function BanditPage() {
  const [state, setState] = useState<BanditState | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setState(await api.banditState());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load bandit state");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const models = state ? Object.keys(state.alphas) : [];
  const betaData = state ? buildBetaChartData(models, state.alphas, state.betas) : [];
  const trafficData = state
    ? Object.entries(state.traffic_split).map(([name, value]) => ({ name: name.toUpperCase(), value: Math.round(value * 100) }))
    : [];

  return (
    <div className="p-8">
      <div className="mb-8 flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Bandit A/B Dashboard</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Thompson Sampling multi-armed bandit — auto-converges traffic to the winning model arm
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={load} disabled={loading} className="gap-2">
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
          Refresh
        </Button>
      </div>

      {error && (
        <Alert variant="destructive" className="mb-6">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {state && (
        <>
          {/* Stat row */}
          <div className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-3">
            <Card>
              <CardContent className="p-5">
                <div className="flex items-center justify-between">
                  <p className="text-xs text-muted-foreground uppercase tracking-wider">Winner</p>
                  <Trophy className="h-4 w-4 text-amber-400" />
                </div>
                <p className="mt-2 text-2xl font-bold">
                  {state.winner ? (
                    <span className="text-amber-400">{state.winner.toUpperCase()}</span>
                  ) : (
                    <span className="text-muted-foreground text-xl">Exploring…</span>
                  )}
                </p>
                <p className="mt-1 text-xs text-muted-foreground">95% confidence threshold</p>
              </CardContent>
            </Card>

            {models.map((m, i) => (
              <Card key={m}>
                <CardContent className="p-5">
                  <div className="flex items-center justify-between">
                    <p className="text-xs text-muted-foreground uppercase tracking-wider">{m.toUpperCase()}</p>
                    <div className="h-2 w-2 rounded-full" style={{ background: colorFor(m, i) }} />
                  </div>
                  <p className="mt-2 text-2xl font-bold font-mono" style={{ color: colorFor(m, i) }}>
                    {((state.ctrs[m] ?? 0) * 100).toFixed(2)}%
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {state.total_pulls[m] ?? 0} pulls · α={state.alphas[m]?.toFixed(1)} β={state.betas[m]?.toFixed(1)}
                  </p>
                </CardContent>
              </Card>
            ))}
          </div>

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1fr_360px]">
            {/* Beta distribution chart */}
            <Card>
              <CardHeader>
                <CardTitle className="text-sm font-medium">Beta Distributions — P(θ) per arm</CardTitle>
                <CardDescription className="text-xs">
                  Higher α concentrates mass toward higher θ (CTR). Traffic routes to the arm with the higher sampled θ.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={betaData} margin={{ left: -10, right: 10 }}>
                    <XAxis
                      dataKey="x"
                      type="number"
                      domain={[0, 1]}
                      tickFormatter={(v: number) => v.toFixed(1)}
                      tick={{ fontSize: 11, fill: "hsl(215 16% 57%)" }}
                      label={{ value: "θ (estimated CTR)", position: "insideBottom", offset: -2, fontSize: 11, fill: "hsl(215 16% 57%)" }}
                    />
                    <YAxis tick={{ fontSize: 11, fill: "hsl(215 16% 57%)" }} width={30} />
                    <Tooltip
                      formatter={(v: number, name: string) => [v.toFixed(3), name.toUpperCase()]}
                      labelFormatter={(v: number) => `θ = ${v.toFixed(3)}`}
                      contentStyle={{ background: "hsl(224 71% 6%)", border: "1px solid hsl(216 34% 17%)", borderRadius: 6, fontSize: 12 }}
                    />
                    <Legend formatter={(v: string) => v.toUpperCase()} wrapperStyle={{ fontSize: 12 }} />
                    {models.map((m, i) => (
                      <Line
                        key={m}
                        type="monotone"
                        dataKey={m}
                        dot={false}
                        strokeWidth={2}
                        stroke={colorFor(m, i)}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <div className="space-y-6">
              {/* Traffic split pie */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm font-medium">Traffic Split</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={180}>
                    <PieChart>
                      <Pie data={trafficData} cx="50%" cy="50%" innerRadius={50} outerRadius={75} dataKey="value" paddingAngle={3}>
                        {trafficData.map((entry, i) => (
                          <Cell key={entry.name} fill={colorFor(models[i], i)} />
                        ))}
                      </Pie>
                      <Tooltip formatter={(v: number) => [`${v}%`]} contentStyle={{ background: "hsl(224 71% 6%)", border: "1px solid hsl(216 34% 17%)", borderRadius: 6, fontSize: 12 }} />
                      <Legend formatter={(v: string) => v} wrapperStyle={{ fontSize: 12 }} />
                    </PieChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* Stats table */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm font-medium">Model Stats</CardTitle>
                </CardHeader>
                <CardContent className="p-0">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b border-border">
                        <th className="px-4 py-2 text-left text-muted-foreground font-medium">Model</th>
                        <th className="px-4 py-2 text-right text-muted-foreground font-medium">Pulls</th>
                        <th className="px-4 py-2 text-right text-muted-foreground font-medium">Bayesian CTR</th>
                      </tr>
                    </thead>
                    <tbody>
                      {models.map((m, i) => (
                        <tr key={m} className="border-b border-border last:border-0">
                          <td className="px-4 py-2.5 font-medium" style={{ color: colorFor(m, i) }}>
                            {m.toUpperCase()}
                            {state.winner === m && <Badge className="ml-2 border-amber-500/30 bg-amber-500/20 text-amber-400 text-xs py-0">winner</Badge>}
                          </td>
                          <td className="px-4 py-2.5 text-right font-mono text-muted-foreground">{state.total_pulls[m] ?? 0}</td>
                          <td className="px-4 py-2.5 text-right font-mono" style={{ color: colorFor(m, i) }}>
                            {((state.mean_ctrs[m] ?? 0) * 100).toFixed(2)}%
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </CardContent>
              </Card>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
