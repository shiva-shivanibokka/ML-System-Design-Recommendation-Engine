"use client";

import { useCallback, useEffect, useState } from "react";
import { api, type MonitoringHealth, type SystemHealth, type LatencyReport } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { RefreshCw, Loader2, AlertTriangle, CheckCircle2, XCircle, Database, Server, Brain, Activity } from "lucide-react";
import { cn } from "@/lib/utils";

function StatusDot({ ok, label }: { ok: boolean; label: string }) {
  return (
    <div className="flex items-center gap-2">
      {ok ? <CheckCircle2 className="h-4 w-4 text-emerald-400" /> : <XCircle className="h-4 w-4 text-red-400" />}
      <span className={cn("text-sm", ok ? "text-foreground" : "text-red-400")}>{label}</span>
    </div>
  );
}

function PsiBar({ value, label }: { value: number; label: string }) {
  const pct = Math.min(value * 500, 100); // PSI alert at 0.2 → 100%
  const color = value > 0.2 ? "bg-red-500" : value > 0.1 ? "bg-amber-500" : "bg-emerald-500";
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted-foreground">{label}</span>
        <div className="flex items-center gap-2">
          <span className={cn("text-xs font-mono", value > 0.2 ? "text-red-400" : value > 0.1 ? "text-amber-400" : "text-emerald-400")}>
            {value.toFixed(4)}
          </span>
          {value > 0.2 ? (
            <Badge variant="warning" className="text-xs py-0">DRIFT</Badge>
          ) : (
            <Badge variant="success" className="text-xs py-0">OK</Badge>
          )}
        </div>
      </div>
      <div className="h-1.5 w-full rounded-full bg-secondary overflow-hidden">
        <div className={cn("h-full rounded-full transition-all", color)} style={{ width: `${pct}%` }} />
      </div>
      <p className="text-xs text-muted-foreground">threshold 0.2</p>
    </div>
  );
}

const STAGE_LABELS: Record<string, string> = {
  cache_check: "Cache check",
  candidate_generation: "FAISS retrieval",
  feature_fetch: "Feast features",
  ranking: "Model ranking",
  post_ranking: "MMR post-rank",
  total: "Total pipeline",
};

const STAGE_BUDGETS: Record<string, number> = {
  cache_check: 5,
  candidate_generation: 20,
  feature_fetch: 10,
  ranking: 50,
  post_ranking: 10,
  total: 100,
};

export default function MonitoringPage() {
  const [health, setHealth] = useState<MonitoringHealth | null>(null);
  const [sysHealth, setSysHealth] = useState<SystemHealth | null>(null);
  const [latency, setLatency] = useState<LatencyReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [h, s, l] = await Promise.all([api.monitoringHealth(), api.health(), api.monitoringLatency()]);
      setHealth(h);
      setSysHealth(s);
      setLatency(l);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load monitoring data");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const coverage = health?.catalog_coverage ?? 0;
  const coveragePct = coverage * 100;
  const coverageColor = coverage < 0.1 ? "text-red-400" : coverage < 0.3 ? "text-amber-400" : "text-emerald-400";

  return (
    <div className="p-8">
      <div className="mb-8 flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">System Monitoring</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Staleness detection: CTR trend · catalog coverage · PSI distribution drift
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={load} disabled={loading} className="gap-2">
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCw className="h-4 w-4" />}
          Refresh
        </Button>
      </div>

      {error && (
        <Alert variant="destructive" className="mb-6">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {health?.any_alert && (
        <Alert variant="warning" className="mb-6">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Staleness Alert Active</AlertTitle>
          <AlertDescription className="mt-1 space-y-0.5">
            {health.alerts.map((a, i) => <p key={i}>{a}</p>)}
          </AlertDescription>
        </Alert>
      )}

      {!loading && !error && !health?.any_alert && (
        <Alert variant="success" className="mb-6">
          <CheckCircle2 className="h-4 w-4" />
          <AlertTitle>All signals healthy</AlertTitle>
          <AlertDescription>CTR, catalog coverage, and PSI drift are within thresholds.</AlertDescription>
        </Alert>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2 xl:grid-cols-3">
        {/* Service health */}
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Server className="h-4 w-4 text-primary" />
              Service Health
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {sysHealth ? (
              <>
                <StatusDot ok={sysHealth.models?.ncf ?? false} label="NeuMF model" />
                <StatusDot ok={sysHealth.models?.svd ?? false} label="SVD model" />
                <StatusDot ok={sysHealth.models?.faiss ?? false} label="FAISS index" />
                <Separator />
                <StatusDot ok={sysHealth.redis ?? false} label="Redis online store" />
                <StatusDot ok={sysHealth.postgres ?? false} label="PostgreSQL" />
                <Separator />
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Catalog size</span>
                  <span className="text-sm font-mono text-foreground">{(sysHealth.n_total_items ?? 0).toLocaleString()} items</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Status</span>
                  <Badge variant={sysHealth.status === "ok" ? "success" : "warning"}>
                    {sysHealth.status?.toUpperCase()}
                  </Badge>
                </div>
              </>
            ) : (
              <div className="space-y-2">
                {[...Array(5)].map((_, i) => <div key={i} className="h-5 rounded bg-muted animate-pulse" />)}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Catalog coverage */}
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Brain className="h-4 w-4 text-primary" />
              Catalog Coverage — 24h
            </CardTitle>
          </CardHeader>
          <CardContent>
            {health ? (
              <div className="space-y-4">
                <div className="text-center">
                  <span className={cn("text-5xl font-bold font-mono", coverageColor)}>
                    {coveragePct.toFixed(1)}%
                  </span>
                  <p className="mt-1 text-xs text-muted-foreground">of catalog recommended today</p>
                </div>
                <Progress
                  value={Math.min(coveragePct, 100)}
                  className={cn("h-2", coverage < 0.1 ? "[&>div]:bg-red-500" : coverage < 0.3 ? "[&>div]:bg-amber-500" : "[&>div]:bg-emerald-500")}
                />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>0%</span>
                  <span className="text-amber-400">alert &lt; 10%</span>
                  <span>100%</span>
                </div>
                <div className="space-y-1.5 pt-1">
                  {Object.entries(health.ctrs).map(([model, ctr]) => (
                    <div key={model} className="flex items-center justify-between">
                      <span className="text-xs text-muted-foreground">{model.toUpperCase()} 7-day CTR</span>
                      <span className="text-xs font-mono text-foreground">{(ctr * 100).toFixed(2)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="h-16 rounded bg-muted animate-pulse mx-auto w-24" />
                <div className="h-2 rounded bg-muted animate-pulse" />
              </div>
            )}
          </CardContent>
        </Card>

        {/* PSI drift */}
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Activity className="h-4 w-4 text-primary" />
              PSI Score Drift
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-5">
            {health ? (
              Object.entries(health.psi_scores).length > 0 ? (
                Object.entries(health.psi_scores).map(([model, psi]) => (
                  <PsiBar key={model} value={psi} label={`${model.toUpperCase()} score distribution`} />
                ))
              ) : (
                <p className="text-sm text-muted-foreground">No PSI data yet — needs at least 2 scoring windows.</p>
              )
            ) : (
              <div className="space-y-3">
                {[...Array(2)].map((_, i) => <div key={i} className="h-8 rounded bg-muted animate-pulse" />)}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Pipeline latency */}
        <Card className="lg:col-span-2 xl:col-span-3">
          <CardHeader>
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Database className="h-4 w-4 text-primary" />
              Pipeline Latency — Last Request (vs budget)
            </CardTitle>
          </CardHeader>
          <CardContent>
            {latency ? (
              <div className="space-y-3">
                {Object.entries(STAGE_LABELS).map(([key, label]) => {
                  const actual = latency.stage_latencies?.[key] ?? 0;
                  const budget = STAGE_BUDGETS[key] ?? 100;
                  const pct = Math.min((actual / budget) * 100, 100);
                  const over = actual > budget;
                  return (
                    <div key={key} className="space-y-1">
                      <div className="flex items-center justify-between text-xs">
                        <span className={cn("font-medium", key === "total" ? "text-foreground" : "text-muted-foreground")}>
                          {label}
                        </span>
                        <div className="flex items-center gap-3">
                          <span className="text-muted-foreground">budget {budget}ms</span>
                          <span className={cn("font-mono font-medium", over ? "text-red-400" : "text-emerald-400")}>
                            {actual.toFixed(1)}ms
                          </span>
                          {over && <Badge variant="warning" className="text-xs py-0">over</Badge>}
                        </div>
                      </div>
                      <div className="h-1.5 w-full rounded-full bg-secondary overflow-hidden">
                        <div
                          className={cn("h-full rounded-full transition-all", key === "total" ? (over ? "bg-red-500" : "bg-primary") : (over ? "bg-red-500/70" : "bg-primary/50"))}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="space-y-3">
                {[...Array(6)].map((_, i) => <div key={i} className="h-6 rounded bg-muted animate-pulse" />)}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
