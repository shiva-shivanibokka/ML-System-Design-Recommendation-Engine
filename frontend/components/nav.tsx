"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import { api, type SystemHealth } from "@/lib/api";

const links = [
  { href: "/", label: "Recommend", color: "#8B5CF6", fg: "#ffffff" },
  { href: "/bandit", label: "Bandit A/B", color: "#22D3EE", fg: "#06171c" },
  { href: "/monitoring", label: "Monitoring", color: "#FF5C6C", fg: "#ffffff" },
  { href: "/catalog", label: "Catalog", color: "#A3E635", fg: "#14200a" },
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
    <header className="sticky top-0 z-40 border-b border-line/60 bg-background/70 backdrop-blur-xl">
      <div className="mx-auto flex h-16 max-w-[1480px] items-center gap-4 px-5 sm:gap-6 sm:px-8">
        {/* Wordmark */}
        <Link href="/" className="flex items-center gap-2.5">
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
          <span className="font-display text-base font-extrabold tracking-tight text-foreground">
            RECSYS
          </span>
        </Link>

        {/* Colored pill nav */}
        <nav className="flex items-center gap-1.5 sm:gap-2">
          {links.map(({ href, label, color, fg }) => {
            const activeLink = href === "/" ? path === "/" : path.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                style={
                  activeLink
                    ? { background: color, color: fg, boxShadow: `0 0 18px ${color}55` }
                    : { color, background: `${color}14`, border: `1px solid ${color}38` }
                }
                className="rounded-full px-3 py-1.5 text-[12.5px] font-bold transition-all hover:brightness-110 sm:px-4 sm:text-[13px]"
              >
                {label}
              </Link>
            );
          })}
        </nav>

        {/* Live status */}
        <div className="ml-auto hidden items-center gap-2 font-mono text-[11px] md:flex">
          {live ? (
            <>
              <span className="flex items-center gap-1.5 text-live">
                <span className="h-1.5 w-1.5 rounded-full bg-live shadow-[0_0_8px] shadow-live" /> LIVE
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
    </header>
  );
}
