import type { SiteState, Status } from "../api";
import { useConsole } from "../store";
import { STATUS_COLOR, STATUS_LABEL } from "../status";

const DASH = "—";

const number = (value: number | null, digits = 0) =>
  value === null ? DASH : value.toLocaleString("en-CA", { maximumFractionDigits: digits });
const percent = (value: number | null) => (value === null ? DASH : `${(value * 100).toFixed(1)}%`);

const RANK: Record<Status, number> = { alert: 0, watch: 1, ok: 2, warmup: 3 };

// Scores sit near 0.45 when healthy; the bar spans 0.35-0.75 so the threshold is visible.
const span = (value: number) => Math.min(1, Math.max(0, (value - 0.35) / 0.4));

function ScoreBar({ state, threshold }: { state: SiteState | undefined; threshold: number }) {
  const score = state?.score ?? null;
  return (
    <div
      className="relative h-1.5 w-16 rounded-full bg-zinc-800"
      title={score === null ? "no score yet" : `score ${score.toFixed(3)}`}
    >
      {score !== null && state && (
        <div
          className="absolute inset-y-0 left-0 rounded-full transition-[width] duration-300"
          style={{ width: `${span(score) * 100}%`, background: STATUS_COLOR[state.status] }}
        />
      )}
      <div
        className="absolute -inset-y-0.5 w-px bg-zinc-400"
        style={{ left: `${span(threshold) * 100}%` }}
        aria-hidden
      />
    </div>
  );
}

/** Live telemetry of every site, worst first. */
export function SiteTable() {
  const topology = useConsole((s) => s.topology);
  const sites = useConsole((s) => s.sites);
  const threshold = useConsole((s) => s.options?.alert_threshold ?? 0.55);
  const rows = [...(topology?.sites ?? [])].sort(
    (a, b) =>
      RANK[sites[a.id]?.status ?? "warmup"] - RANK[sites[b.id]?.status ?? "warmup"] ||
      a.name.localeCompare(b.name),
  );

  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="text-left text-[11px] uppercase tracking-wider text-zinc-500">
          <th className="pb-2 font-medium">Site</th>
          <th className="pb-2 text-right font-medium">QPS</th>
          <th className="pb-2 text-right font-medium">p95</th>
          <th className="pb-2 text-right font-medium">Fail</th>
          <th className="pb-2 pl-3 font-medium">Score</th>
        </tr>
      </thead>
      <tbody className="divide-y divide-zinc-800/70">
        {rows.map((site) => {
          const state = sites[site.id];
          const status = state?.status;
          const pulsing = status === "alert" || status === "watch";
          return (
            <tr key={site.id}>
              <td className="py-2">
                <div className="flex items-center gap-2.5">
                  <span
                    className={`size-2 shrink-0 rounded-full ${pulsing ? "breathe" : ""}`}
                    style={{ background: status ? STATUS_COLOR[status] : "#3f3f46" }}
                  />
                  <div className="leading-tight">
                    <div className="text-zinc-100">{site.name}</div>
                    <div className="text-[11px] text-zinc-500">
                      {status ? STATUS_LABEL[status] : site.id}
                      {state && !state.announced ? " · withdrawn" : ""}
                    </div>
                  </div>
                </div>
              </td>
              <td className="py-2 text-right font-mono tabular-nums">
                {number(state?.qps ?? null)}
              </td>
              <td className="py-2 text-right font-mono tabular-nums">
                {state?.p95_latency_ms == null ? DASH : `${number(state.p95_latency_ms, 1)} ms`}
              </td>
              <td className="py-2 text-right font-mono tabular-nums">
                {percent(state?.fail_ratio ?? null)}
              </td>
              <td className="py-2 pl-3">
                <ScoreBar state={state} threshold={threshold} />
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
