"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import { api, type SystemHealth } from "@/lib/api";

const links = [
  { href: "/", label: "Recommend" },
  { href: "/bandit", label: "Bandit A/B" },
  { href: "/monitoring", label: "Monitoring" },
  { href: "/catalog", label: "Catalog" },
];

export function Nav() {
  const path = usePathname();
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [down, setDown] = useState(false);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setDown(true));
  }, []);

  const live = !!health && health.status === "ok";

  return (
    <header className="sticky top-0 z-40 border-b border-line bg-ink/85 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-[1440px] items-center gap-6 px-5 sm:px-8">
        {/* Wordmark */}
        <Link href="/" className="group flex items-center gap-2.5">
          <span className="relative flex h-2.5 w-2.5">
            <span
              className={cn(
                "absolute inline-flex h-full w-full rounded-full",
                live ? "animate-ping bg-live/60" : "bg-transparent"
              )}
            />
            <span
              className={cn(
                "relative inline-flex h-2.5 w-2.5 rounded-full",
                live ? "bg-live" : down ? "bg-alert" : "bg-signal"
              )}
            />
          </span>
          <span className="font-display text-[15px] font-bold tracking-tight text-foreground">
            RECSYS
            <span className="text-signal">.</span>
          </span>
          <span className="hidden font-mono text-[10px] uppercase tracking-[0.2em] text-muted-foreground sm:inline">
            engine
          </span>
        </Link>

        {/* Nav */}
        <nav className="flex items-center gap-1">
          {links.map(({ href, label }) => {
            const activeLink = href === "/" ? path === "/" : path.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                className={cn(
                  "rounded-md px-2.5 py-1.5 text-[13px] font-medium transition-colors sm:px-3",
                  activeLink
                    ? "bg-signal/12 text-signal"
                    : "text-muted-foreground hover:text-foreground"
                )}
              >
                {label}
              </Link>
            );
          })}
        </nav>

        {/* Live status */}
        <div className="ml-auto flex items-center gap-3">
          <div className="hidden items-center gap-2 font-mono text-[11px] md:flex">
            {live ? (
              <>
                <span className="flex items-center gap-1.5 text-live">
                  <span className="h-1.5 w-1.5 rounded-full bg-live" /> LIVE
                </span>
                <span className="text-line">/</span>
                <span className="text-muted-foreground">
                  {(health?.n_total_items ?? 0).toLocaleString()} items
                </span>
              </>
            ) : down ? (
              <span className="text-alert">API waking… retry shortly</span>
            ) : (
              <span className="text-muted-foreground">connecting…</span>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}
