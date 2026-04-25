import type { PipelineResult } from "../domain/types";

const RESULTS_URL = `${import.meta.env.BASE_URL}results.json`;

let cached: Promise<PipelineResult> | null = null;

export function loadResults(): Promise<PipelineResult> {
  if (cached) return cached;
  cached = fetch(RESULTS_URL, { cache: "force-cache" })
    .then((r) => {
      if (!r.ok) throw new Error(`results.json fetch failed: ${r.status}`);
      return r.json() as Promise<PipelineResult>;
    })
    .catch((e) => {
      cached = null;
      throw e;
    });
  return cached;
}
