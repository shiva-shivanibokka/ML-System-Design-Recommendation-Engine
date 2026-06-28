"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { BarChart3, Activity, Cpu, Zap } from "lucide-react";

const links = [
  { href: "/", label: "Recommendations", icon: Zap },
  { href: "/bandit", label: "Bandit A/B", icon: BarChart3 },
  { href: "/monitoring", label: "Monitoring", icon: Activity },
];

export function Nav() {
  const path = usePathname();
  return (
    <aside className="fixed inset-y-0 left-0 z-10 flex w-60 flex-col border-r border-border bg-background">
      <div className="flex h-16 items-center gap-2 border-b border-border px-6">
        <Cpu className="h-5 w-5 text-primary" />
        <span className="font-semibold text-sm tracking-tight">RecSys Engine</span>
      </div>

      <nav className="flex-1 space-y-1 px-3 py-4">
        {links.map(({ href, label, icon: Icon }) => (
          <Link
            key={href}
            href={href}
            className={cn(
              "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
              path === href
                ? "bg-primary/10 text-primary"
                : "text-muted-foreground hover:bg-accent hover:text-foreground"
            )}
          >
            <Icon className="h-4 w-4 flex-shrink-0" />
            {label}
          </Link>
        ))}
      </nav>

      <div className="border-t border-border px-6 py-4">
        <p className="text-xs text-muted-foreground">MovieLens 1M · 3,706 items</p>
        <p className="text-xs text-muted-foreground">6,040 users · NeuMF + SVD</p>
      </div>
    </aside>
  );
}
