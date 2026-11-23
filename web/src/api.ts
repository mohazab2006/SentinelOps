// Messages of the console API (src/sentinelops/api/schemas.py). Keep the two in step.

export type Status = "warmup" | "ok" | "watch" | "alert";

export interface SiteInfo {
  id: string;
  name: string;
  lat: number;
  lon: number;
  servers: string[];
}

export interface RegionInfo {
  id: string;
  name: string;
  lat: number;
  lon: number;
}

export interface TopologyInfo {
  sites: SiteInfo[];
  regions: RegionInfo[];
}

export interface Options {
  families: string[];
  agents: string[];
  horizon: number;
  alert_threshold: number;
}

export interface SiteState {
  qps: number | null;
  p95_latency_ms: number | null;
  loss: number | null;
  fail_ratio: number | null;
  announced: boolean;
  weight: number;
  score: number | null;
  status: Status;
}

export type EventRecord = { tick: number; event: string } & Record<
  string,
  string | number | boolean | null
>;

export interface RunStarted {
  type: "run_started";
  family: string;
  seed: number;
  agent: string;
  horizon: number;
  alert_threshold: number;
}

export interface TickFrame {
  type: "tick";
  tick: number;
  sites: Record<string, SiteState>;
  events: EventRecord[];
}

export interface RunFinished {
  type: "run_finished";
  tick: number;
  legit_failure_share: number;
}

export interface ErrorMessage {
  type: "error";
  message: string;
}

export type LiveMessage = RunStarted | TickFrame | RunFinished | ErrorMessage;

export interface RunRequest {
  family: string;
  seed: number;
  agent: string;
  speed: number;
}

export async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`${path} answered ${response.status}`);
  }
  return (await response.json()) as T;
}

export function liveUrl(request: RunRequest): string {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  const query = new URLSearchParams({
    family: request.family,
    seed: String(request.seed),
    agent: request.agent,
    speed: String(request.speed),
  });
  return `${scheme}://${window.location.host}/ws/live?${query}`;
}
