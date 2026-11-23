import type { EventRecord } from "../api";
import { useConsole } from "../store";

function describe(event: EventRecord): string {
  if (event.event === "action") {
    return [event.kind, event.target, event.status].filter((part) => part != null).join(" ");
  }
  if (event.event === "probe_started") return `probe ${event.kind} on ${event.target}`;
  if (event.event === "probe_result") return `${event.kind} on ${event.target}: ${event.outcome}`;
  return event.event;
}

/** The agent's own actions and probes, newest first. */
export function EventFeed() {
  const events = useConsole((s) => s.events);
  const agent = useConsole((s) => s.run?.agent);
  if (!events.length) {
    return (
      <p className="text-xs leading-relaxed text-zinc-500">
        {agent === "noop"
          ? "No agent is acting in this run, so faults play out untouched."
          : "Actions and probes appear here as the agent takes them."}
      </p>
    );
  }
  return (
    <ol className="space-y-1.5 font-mono text-xs">
      {events.map((event, i) => (
        <li key={`${event.tick}-${i}`} className="flex gap-3">
          <span className="w-12 shrink-0 text-right tabular-nums text-zinc-500">{event.tick}</span>
          <span className="text-zinc-300">{describe(event)}</span>
        </li>
      ))}
    </ol>
  );
}
