import { Play, Stop } from "@phosphor-icons/react";
import { useState, type FormEvent } from "react";

import { useConsole } from "../store";

const SPEEDS = [10, 20, 50, 100];

const field =
  "w-full rounded-md border border-zinc-800 bg-zinc-900 px-2.5 py-1.5 text-sm text-zinc-100 " +
  "outline-none transition-colors focus:border-zinc-500";

const LABELS: Record<string, string> = {
  path_loss: "path loss",
  site_outage: "site outage",
  benign_surge: "benign surge",
  noop: "no agent",
  rules: "rule baseline",
};

const label = (value: string) => LABELS[value] ?? value;

/** Pick a scenario and start or stop its live stream. */
export function RunControls() {
  const options = useConsole((s) => s.options);
  const phase = useConsole((s) => s.phase);
  const start = useConsole((s) => s.start);
  const stop = useConsole((s) => s.stop);
  const [family, setFamily] = useState("crash");
  const [agent, setAgent] = useState("noop");
  const [seed, setSeed] = useState("100");
  const [speed, setSpeed] = useState(20);

  // Plain digits only: "1e30" is an integer to JavaScript but not to the server.
  const seedValid = /^\d{1,9}$/.test(seed);
  const seedNumber = Number(seed);
  const running = phase === "live" || phase === "connecting";

  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (running) stop();
    else if (seedValid) start({ family, agent, seed: seedNumber, speed });
  };

  return (
    <form onSubmit={submit} className="grid grid-cols-2 gap-x-3 gap-y-3">
      <label className="flex flex-col gap-1.5 text-xs text-zinc-400">
        Scenario
        <select className={field} value={family} onChange={(e) => setFamily(e.target.value)}>
          {(options?.families ?? []).map((f) => (
            <option key={f} value={f}>
              {label(f)}
            </option>
          ))}
        </select>
      </label>
      <label className="flex flex-col gap-1.5 text-xs text-zinc-400">
        Seed
        <input
          className={`${field} font-mono ${seedValid ? "" : "border-rose-500/70"}`}
          inputMode="numeric"
          value={seed}
          onChange={(e) => setSeed(e.target.value.trim())}
          aria-invalid={!seedValid}
          aria-describedby="seed-help"
        />
        <span id="seed-help" className={seedValid ? "text-zinc-600" : "text-rose-400"}>
          {seedValid ? "Same seed, same incident" : "Whole number, 0 or more"}
        </span>
      </label>
      <label className="flex flex-col gap-1.5 text-xs text-zinc-400">
        Agent
        <select className={field} value={agent} onChange={(e) => setAgent(e.target.value)}>
          {(options?.agents ?? []).map((a) => (
            <option key={a} value={a}>
              {label(a)}
            </option>
          ))}
        </select>
      </label>
      <label className="flex flex-col gap-1.5 text-xs text-zinc-400">
        Speed
        <select
          className={`${field} font-mono`}
          value={speed}
          onChange={(e) => setSpeed(Number(e.target.value))}
        >
          {SPEEDS.map((s) => (
            <option key={s} value={s}>
              {s} ticks/s
            </option>
          ))}
        </select>
      </label>
      <button
        type="submit"
        disabled={!options || (!running && !seedValid)}
        className={
          "col-span-2 mt-1 inline-flex items-center justify-center gap-2 rounded-md px-3 py-2 " +
          "text-sm font-medium transition-[transform,background-color] duration-200 " +
          "active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-40 " +
          (running
            ? "bg-zinc-800 text-zinc-100 hover:bg-zinc-700"
            : "bg-emerald-400 text-zinc-950 hover:bg-emerald-300")
        }
      >
        {running ? <Stop size={16} weight="fill" /> : <Play size={16} weight="fill" />}
        {running ? "Stop run" : "Start run"}
      </button>
    </form>
  );
}
