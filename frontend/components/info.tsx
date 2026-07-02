"use client";

import { useState, useRef, useEffect } from "react";
import { GLOSSARY, type GlossaryKey } from "@/lib/glossary";
import { cn } from "@/lib/utils";

/**
 * A "?" affordance that reveals a plain-language definition on hover or tap.
 * Pulls copy from lib/glossary so every term reads the same everywhere.
 */
export function Info({ k, className }: { k: GlossaryKey; className?: string }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLSpanElement>(null);
  const entry = GLOSSARY[k];

  useEffect(() => {
    if (!open) return;
    function onDoc(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  if (!entry) return null;

  return (
    <span ref={ref} className={cn("relative inline-flex", className)}>
      <button
        type="button"
        aria-label={`What is ${entry.term}?`}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onClick={(e) => {
          e.preventDefault();
          setOpen((v) => !v);
        }}
        className={cn(
          "flex h-4 w-4 items-center justify-center rounded-full border border-line text-[10px] font-semibold leading-none transition-colors",
          open
            ? "border-signal/60 bg-signal/15 text-signal"
            : "text-muted-foreground hover:border-signal/50 hover:text-signal"
        )}
      >
        ?
      </button>
      {open && (
        <span
          role="tooltip"
          className="animate-fade-up absolute left-1/2 top-6 z-50 w-64 -translate-x-1/2 rounded-lg border border-line bg-popover p-3 text-left shadow-xl shadow-black/40"
        >
          <span className="mb-1 block font-mono text-[11px] font-semibold uppercase tracking-wide text-signal">
            {entry.term}
          </span>
          <span className="block text-xs leading-relaxed text-foreground/85">{entry.body}</span>
        </span>
      )}
    </span>
  );
}
