import { Pulse, WarningCircle } from "@phosphor-icons/react";
import { lazy, Suspense, useEffect, type ReactNode } from "react";

import type { Status } from "./api";
import { EventFeed } from "./components/EventFeed";
import { RunControls } from "./components/RunControls";
import { SiteTable } from "./components/SiteTable";
import { useConsole, type Phase } from "./store";
import { STATUS_COLOR, STATUS_LABEL } from "./status";

// three.js is most of the bundle: load the globe after the panels are on screen.
const Globe = lazy(() => import("./components/Globe").then((m) => ({ default: m.Globe })));

const PHASE_LABEL: Record<Phase, string> = {
  loading: "Connecting",
  idle: "Idle",
  connecting: "Starting run",
  live: "Live",
  finished: "Run finished",
  error: "Error",
};

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="border-t border-zinc-800/80 px-5 py-4 first:border-t-0">
      <h2 className="mb-3 text-[11px] font-medium uppercase tracking-wider text-zinc-500">
        {title}
      </h2>
      {children}
    </section>
  );
}

function TopBar() {
  const phase = useConsole((s) => s.phase);
  const run = useConsole((s) => s.run);
  const tick = useConsole((s) => s.tick);
  const progress = run ? Math.max(0, (tick + 1) / run.horizon) : 0;
  return (
    <header className="relative flex items-center justify-between gap-4 border-b border-zinc-800/80 px-5 py-3">
      <div className="flex items-baseline gap-3">
        <h1 className="text-base font-semibold tracking-tight text-zinc-50">SentinelOps</h1>
        <span className="hidden text-xs text-zinc-500 sm:inline">anycast DNS operations</span>
      </div>
      <div className="flex items-center gap-4 text-xs">
        {run && (
          <span className="hidden font-mono text-zinc-400 md:inline">
            {run.family} · seed {run.seed} · tick {Math.max(tick, 0)}/{run.horizon}
          </span>
        )}
        <span className="inline-flex items-center gap-1.5 rounded-full border border-zinc-800 px-2.5 py-1 text-zinc-300">
          <Pulse size={14} className={phase === "live" ? "text-emerald-400" : "text-zinc-500"} />
          {PHASE_LABEL[phase]}
        </span>
      </div>
      <div
        className="absolute inset-x-0 -bottom-px h-px origin-left bg-emerald-400/70 transition-transform duration-300"
        style={{ transform: `scaleX(${progress})` }}
        aria-hidden
      />
    </header>
  );
}

function Legend() {
  const statuses: Status[] = ["ok", "watch", "alert"];
  return (
    <ul className="flex gap-4 text-xs text-zinc-400">
      {statuses.map((status) => (
        <li key={status} className="flex items-center gap-1.5">
          <span className="size-2 rounded-full" style={{ background: STATUS_COLOR[status] }} />
          {STATUS_LABEL[status]}
        </li>
      ))}
    </ul>
  );
}

function GlobeOverlay() {
  const phase = useConsole((s) => s.phase);
  const sites = useConsole((s) => s.sites);
  const result = useConsole((s) => s.result);
  const alerting = Object.values(sites).filter((s) => s.status === "alert").length;
  let headline: string;
  if (phase === "idle" || phase === "loading") headline = "Start a run to stream live telemetry";
  else if (phase === "connecting") headline = "Starting the simulation";
  else if (alerting) headline = `${alerting} site${alerting > 1 ? "s" : ""} alerting`;
  else headline = "All sites within normal range";
  return (
    <div className="pointer-events-none absolute inset-x-0 bottom-0 flex flex-wrap items-end justify-between gap-3 p-5">
      <div>
        <p className="text-sm text-zinc-200">{headline}</p>
        {result && (
          <p className="mt-1 font-mono text-xs text-zinc-400">
            legitimate queries failed: {(result.legit_failure_share * 100).toFixed(2)}%
          </p>
        )}
      </div>
      <Legend />
    </div>
  );
}

function ErrorBanner() {
  const error = useConsole((s) => s.error);
  if (!error) return null;
  return (
    <div
      role="alert"
      className="mx-5 mt-4 flex gap-2 rounded-md border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-xs text-rose-200"
    >
      <WarningCircle size={16} className="mt-px shrink-0" />
      {error}
    </div>
  );
}

function SkeletonRows() {
  return (
    <div className="space-y-3" aria-hidden>
      {Array.from({ length: 6 }, (_, i) => (
        <div key={i} className="h-8 animate-pulse rounded bg-zinc-900" />
      ))}
    </div>
  );
}

export function App() {
  const load = useConsole((s) => s.load);
  const ready = useConsole((s) => s.topology !== null);
  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="flex min-h-[100dvh] flex-col">
      <TopBar />
      <main className="grid flex-1 grid-cols-1 lg:grid-cols-[minmax(0,1fr)_420px]">
        <div className="relative min-h-[420px] overflow-hidden bg-[radial-gradient(ellipse_at_center,#18181b_0%,#09090b_70%)]">
          {ready && (
            <Suspense fallback={null}>
              <Globe />
            </Suspense>
          )}
          <GlobeOverlay />
        </div>
        <aside className="border-t border-zinc-800/80 lg:border-t-0 lg:border-l">
          <ErrorBanner />
          <Section title="Run">
            <RunControls />
          </Section>
          <Section title="Sites">{ready ? <SiteTable /> : <SkeletonRows />}</Section>
          <Section title="Agent activity">
            <EventFeed />
          </Section>
        </aside>
      </main>
    </div>
  );
}
