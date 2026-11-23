import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import GlobeGL, { type GlobeMethods } from "react-globe.gl";
import { MeshPhongMaterial } from "three";
import { feature } from "topojson-client";
import countries110m from "world-atlas/countries-110m.json";
import type { Topology } from "topojson-specification";
import type { FeatureCollection } from "geojson";

import type { SiteInfo, Status } from "../api";
import { useConsole } from "../store";
import { IDLE_COLOR, STATUS_COLOR } from "../status";

const world = countries110m as unknown as Topology;
const LAND = (feature(world, world.objects.countries) as FeatureCollection).features;

interface SitePoint extends SiteInfo {
  status: Status | null;
}

const color = (p: SitePoint) => (p.status ? STATUS_COLOR[p.status] : IDLE_COLOR);

function useSize(ref: React.RefObject<HTMLDivElement | null>): { width: number; height: number } {
  const [size, setSize] = useState({ width: 0, height: 0 });
  useLayoutEffect(() => {
    const element = ref.current;
    if (!element) return;
    const measure = element.getBoundingClientRect(); // first size, before any resize
    setSize({ width: Math.round(measure.width), height: Math.round(measure.height) });
    const observer = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setSize({ width: Math.round(width), height: Math.round(height) });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, [ref]);
  return size;
}

/** A site's name tag, made once per site and recoloured as its status changes. */
function tagFor(cache: Map<string, HTMLDivElement>, point: SitePoint): HTMLDivElement {
  let element = cache.get(point.id);
  if (!element) {
    element = document.createElement("div");
    element.textContent = point.name;
    element.style.cssText =
      "transform: translate(-50%, -140%); padding: 2px 8px; border-radius: 999px; " +
      "font: 500 11px 'Geist Variable', sans-serif; color: #fafafa; pointer-events: none; " +
      "background: rgba(9, 9, 11, 0.82); border: 1px solid;";
    cache.set(point.id, element);
  }
  element.style.borderColor = color(point);
  return element;
}

/** The network on a globe: one point per site, coloured by its live status. */
export function Globe() {
  const container = useRef<HTMLDivElement>(null);
  const globe = useRef<GlobeMethods | undefined>(undefined);
  const { width, height } = useSize(container);
  const topology = useConsole((s) => s.topology);
  const sites = useConsole((s) => s.sites);

  const material = useMemo(
    () => new MeshPhongMaterial({ color: "#18181b", emissive: "#09090b", shininess: 4 }),
    [],
  );
  useEffect(() => () => material.dispose(), [material]);

  const points: SitePoint[] = useMemo(
    () => (topology?.sites ?? []).map((site) => ({ ...site, status: sites[site.id]?.status ?? null })),
    [topology, sites],
  );
  const alerts = useMemo(() => points.filter((p) => p.status === "alert"), [points]);
  // Only sites that need attention get a name tag: nearby sites would overlap otherwise.
  // Rebuilt only when which sites are flagged (or their status) changes, not every tick.
  const flaggedKey = points
    .filter((p) => p.status === "alert" || p.status === "watch")
    .map((p) => `${p.id}:${p.status}`)
    .join(",");
  const flagged = useMemo(() => {
    const ids = new Set(flaggedKey.split(",").map((entry) => entry.split(":")[0]));
    return points.filter((p) => ids.has(p.id));
  }, [flaggedKey]); // not on `points`: it is a new array every tick
  const tags = useRef(new Map<string, HTMLDivElement>());

  return (
    <div ref={container} className="absolute inset-0" aria-label="Globe of anycast sites">
      {width > 0 && (
        <GlobeGL
          ref={globe}
          width={width}
          height={height}
          backgroundColor="rgba(0,0,0,0)"
          onGlobeReady={() => globe.current?.pointOfView({ lat: 38, lng: -32, altitude: 2.1 }, 0)}
          globeMaterial={material}
          showAtmosphere
          atmosphereColor="#3f3f46"
          atmosphereAltitude={0.12}
          hexPolygonsData={LAND}
          hexPolygonResolution={3}
          hexPolygonMargin={0.35}
          hexPolygonColor={() => "rgba(161, 161, 170, 0.22)"}
          pointsData={points}
          pointLat="lat"
          pointLng="lon"
          pointColor={(p) => color(p as SitePoint)}
          pointAltitude={(p) => ((p as SitePoint).status === "alert" ? 0.09 : 0.03)}
          pointRadius={0.55}
          pointsTransitionDuration={300}
          pointLabel={(p) => (p as SitePoint).name}
          htmlElementsData={flagged}
          htmlLat="lat"
          htmlLng="lon"
          htmlAltitude={0.1}
          htmlElement={(p) => tagFor(tags.current, p as SitePoint)}
          ringsData={alerts}
          ringLat="lat"
          ringLng="lon"
          ringColor={() => (t: number) => `rgba(251, 113, 133, ${1 - t})`}
          ringMaxRadius={5}
          ringPropagationSpeed={2.5}
          ringRepeatPeriod={900}
        />
      )}
    </div>
  );
}
