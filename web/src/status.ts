import type { Status } from "./api";

/** One colour per site status, shared by the globe and the panels. */
export const STATUS_COLOR: Record<Status, string> = {
  warmup: "#71717a", // zinc-500: no verdict yet
  ok: "#34d399", // emerald-400
  watch: "#fbbf24", // amber-400
  alert: "#fb7185", // rose-400
};

export const STATUS_LABEL: Record<Status, string> = {
  warmup: "Warming up",
  ok: "Healthy",
  watch: "Watch",
  alert: "Alert",
};

export const IDLE_COLOR = "#52525b"; // zinc-600: no run streaming
