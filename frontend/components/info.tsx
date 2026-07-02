"use client";

import { useState, useRef, useEffect, useLayoutEffect } from "react";
import { createPortal } from "react-dom";
import { GLOSSARY, type GlossaryKey } from "@/lib/glossary";
import { cn } from "@/lib/utils";

const WIDTH = 272;

/**
 * A "?" affordance that reveals a plain-language definition on hover or tap.
 * The panel renders in a portal with fixed, viewport-clamped coordinates, so it
 * never gets clipped by an overflow-hidden card and never spills off-screen.
 */
export function Info({ k, className }: { k: GlossaryKey; className?: string }) {
  const [open, setOpen] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [pos, setPos] = useState<{ top: number; left: number; above: boolean }>({
    top: 0,
    left: 0,
    above: false,
  });
  const btnRef = useRef<HTMLButtonElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const entry = GLOSSARY[k];

  useEffect(() => setMounted(true), []);

  const place = () => {
    const b = btnRef.current?.getBoundingClientRect();
    if (!b) return;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const left = Math.min(Math.max(b.left + b.width / 2 - WIDTH / 2, 8), vw - WIDTH - 8);
    const above = b.bottom + 140 > vh; // flip up if not enough room below
    const top = above ? b.top - 8 : b.bottom + 8;
    setPos({ top, left, above });
  };

  useLayoutEffect(() => {
    if (open) place();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onScroll = () => setOpen(false);
    const onDoc = (e: MouseEvent) => {
      if (
        btnRef.current?.contains(e.target as Node) ||
        panelRef.current?.contains(e.target as Node)
      )
        return;
      setOpen(false);
    };
    window.addEventListener("scroll", onScroll, true);
    document.addEventListener("mousedown", onDoc);
    return () => {
      window.removeEventListener("scroll", onScroll, true);
      document.removeEventListener("mousedown", onDoc);
    };
  }, [open]);

  if (!entry) return null;

  return (
    <>
      <button
        ref={btnRef}
        type="button"
        aria-label={`What is ${entry.term}?`}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onClick={(e) => {
          e.preventDefault();
          setOpen((v) => !v);
        }}
        className={cn(
          "inline-flex h-4 w-4 items-center justify-center rounded-full border text-[10px] font-bold leading-none transition-colors",
          open
            ? "border-signal bg-signal text-white"
            : "border-signal/40 text-signal/70 hover:border-signal hover:text-signal",
          className
        )}
      >
        ?
      </button>
      {mounted &&
        open &&
        createPortal(
          <div
            ref={panelRef}
            role="tooltip"
            style={{
              position: "fixed",
              top: pos.top,
              left: pos.left,
              width: WIDTH,
              transform: pos.above ? "translateY(-100%)" : undefined,
            }}
            className="z-[100] rounded-xl border border-signal/40 bg-raised p-3.5 shadow-[0_12px_44px_-6px_rgba(139,92,246,0.45)]"
          >
            <span className="mb-1 block font-mono text-[11px] font-bold uppercase tracking-wide text-signal">
              {entry.term}
            </span>
            <span className="block text-[12.5px] leading-relaxed text-foreground/90">
              {entry.body}
            </span>
          </div>,
          document.body
        )}
    </>
  );
}
