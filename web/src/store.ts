import { create } from "zustand";

import {
  fetchJson,
  liveUrl,
  type EventRecord,
  type LiveMessage,
  type Options,
  type RunFinished,
  type RunRequest,
  type RunStarted,
  type SiteState,
  type TopologyInfo,
} from "./api";

/** Where the live run stands. */
export type Phase = "loading" | "idle" | "connecting" | "live" | "finished" | "error";

const EVENT_HISTORY = 40;

interface ConsoleState {
  phase: Phase;
  error: string | null;
  topology: TopologyInfo | null;
  options: Options | null;
  run: RunStarted | null;
  tick: number;
  sites: Record<string, SiteState>;
  events: EventRecord[];
  result: RunFinished | null;
  load: () => Promise<void>;
  start: (request: RunRequest) => void;
  stop: () => void;
}

let socket: WebSocket | null = null;

export const useConsole = create<ConsoleState>()((set, get) => ({
  phase: "loading",
  error: null,
  topology: null,
  options: null,
  run: null,
  tick: -1,
  sites: {},
  events: [],
  result: null,

  load: async () => {
    try {
      const [topology, options] = await Promise.all([
        fetchJson<TopologyInfo>("/api/topology"),
        fetchJson<Options>("/api/options"),
      ]);
      set({ topology, options, phase: "idle", error: null });
    } catch (error) {
      set({
        phase: "error",
        error: `Cannot reach the console API (${String(error)}). Is \`sentinelops serve\` running?`,
      });
    }
  },

  start: (request) => {
    get().stop();
    set({ phase: "connecting", error: null, run: null, tick: -1, sites: {}, events: [], result: null });
    const ws = new WebSocket(liveUrl(request));
    socket = ws;
    ws.onmessage = (raw) => {
      if (socket !== ws) return; // a newer run replaced this one
      const message = JSON.parse(raw.data as string) as LiveMessage;
      switch (message.type) {
        case "run_started":
          set({ phase: "live", run: message });
          break;
        case "tick":
          set((state) => ({
            tick: message.tick,
            sites: message.sites,
            events: message.events.length
              ? [...message.events.reverse(), ...state.events].slice(0, EVENT_HISTORY)
              : state.events,
          }));
          break;
        case "run_finished":
          set({ phase: "finished", result: message });
          break;
        case "error":
          set({ phase: "error", error: message.message });
          break;
      }
    };
    ws.onerror = () => {
      if (socket === ws) set({ phase: "error", error: "The live connection failed." });
    };
    ws.onclose = () => {
      if (socket !== ws) return;
      socket = null;
      if (get().phase === "live" || get().phase === "connecting") {
        set({ phase: "error", error: "The server closed the live run early." });
      }
    };
  },

  stop: () => {
    const ws = socket;
    socket = null;
    ws?.close();
    if (get().phase === "live" || get().phase === "connecting") set({ phase: "idle" });
  },
}));
