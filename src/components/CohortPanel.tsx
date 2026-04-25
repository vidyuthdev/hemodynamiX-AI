import type { CohortCase, FeatureKey, Split } from "../domain/types";
import { FEATURE_KEYS, FEATURE_LABELS } from "../domain/types";
import styles from "./Panel.module.css";

type Props = {
  cohort: CohortCase[];
  split: Split;
};

function summarize(cohort: CohortCase[], key: FeatureKey) {
  const pos = cohort.filter((c) => c.label === 1).map((c) => c.features[key]);
  const neg = cohort.filter((c) => c.label === 0).map((c) => c.features[key]);
  const m = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length);
  return { pos: m(pos), neg: m(neg) };
}

export function CohortPanel({ cohort, split }: Props) {
  const total = cohort.length;
  const pos = cohort.filter((c) => c.label === 1).length;
  const bySource = (s: CohortCase["source"]) => cohort.filter((c) => c.source === s).length;
  const byLoc = (s: CohortCase["location"]) => cohort.filter((c) => c.location === s).length;

  return (
    <div className={styles.panel}>
      <header className={styles.head}>
        <h2 className={styles.h2}>Cohort &amp; data sources</h2>
        <p className={styles.sub}>
          Cohort built from {bySource("AnXplore")} real intracranial aneurysm meshes from the
          AnXplore dataset (Goetz et al., Frontiers Bioeng. 2024) plus {bySource("Synthetic")}{" "}
          parametric curved-tube cases generated procedurally to broaden the morphology
          envelope. Each mesh was processed end-to-end with PyVista and a Womersley
          reduced-order pulsatile flow solver to derive the six hemodynamic features below.
          Labels follow the Lauric et al. (2018) morphological criteria for elevated
          rupture risk, with realistic clinical label noise.
        </p>
      </header>

      <div className={styles.statRow}>
        <Stat label="Total cases" value={total} />
        <Stat
          label="High-risk (positives)"
          value={pos}
          hint={`${((pos / total) * 100).toFixed(0)}% prevalence`}
        />
        <Stat
          label="Train / Cal / Test"
          value={`${split.train.length} · ${split.cal.length} · ${split.test.length}`}
        />
        <Stat
          label="Sources"
          value={`${bySource("AnXplore")} · ${bySource("Synthetic")}`}
          hint="AnXplore (real) · parametric"
        />
        <Stat
          label="Locations"
          value={`${byLoc("ACOM")} · ${byLoc("MCA")} · ${byLoc("PCOM")}`}
          hint="ACOM · MCA · PCOM"
        />
      </div>

      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>Feature</th>
              <th>Mean (high-risk)</th>
              <th>Mean (low-risk)</th>
              <th>Direction of risk</th>
            </tr>
          </thead>
          <tbody>
            {FEATURE_KEYS.map((k) => {
              const s = summarize(cohort, k);
              const direction =
                k === "tawss"
                  ? "Lower → higher risk"
                  : k === "velocity"
                    ? "Lower → mildly higher risk"
                    : "Higher → higher risk";
              return (
                <tr key={k}>
                  <td>{FEATURE_LABELS[k]}</td>
                  <td>{s.pos.toFixed(3)}</td>
                  <td>{s.neg.toFixed(3)}</td>
                  <td className={styles.muted}>{direction}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Stat({ label, value, hint }: { label: string; value: string | number; hint?: string }) {
  return (
    <div className={styles.stat}>
      <span className={styles.statLabel}>{label}</span>
      <span className={styles.statValue}>{value}</span>
      {hint && <span className={styles.statHint}>{hint}</span>}
    </div>
  );
}
