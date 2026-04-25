import type { ResolutionResult, SubgroupResult } from "../domain/types";
import styles from "./Panel.module.css";

type Props = {
  byLocation: SubgroupResult[];
  byResolution: ResolutionResult[];
};

export function EquityPanel({ byLocation, byResolution }: Props) {
  const w = 360;
  const h = 200;
  const padL = 38;
  const padB = 28;
  const padT = 12;
  const xs = byResolution.map((r) => r.noiseLevel);
  const xMin = 0;
  const xMax = Math.max(...xs);
  const yMin = 0.5;
  const yMax = 1;
  const x = (v: number) => padL + ((v - xMin) / (xMax - xMin)) * (w - padL - 8);
  const y = (v: number) => padT + (1 - (v - yMin) / (yMax - yMin)) * (h - padT - padB);
  const path = byResolution
    .map((p, i) => `${i === 0 ? "M" : "L"}${x(p.noiseLevel).toFixed(1)} ${y(p.auroc).toFixed(1)}`)
    .join(" ");

  return (
    <div className={styles.panel}>
      <header className={styles.head}>
        <h2 className={styles.h2}>Equity &amp; robustness audit</h2>
        <p className={styles.sub}>
          Disaggregated AUROC by aneurysm location and a resolution-degradation stress test.
          The Milestone-3 finding holds: the cohorts most likely to benefit from earlier
          screening are exactly the ones where input quality drops, so we explicitly track
          AUROC vs. input noise as a fairness metric, not a footnote.
        </p>
      </header>

      <div className={styles.row + " " + styles.rowSplit}>
        <div>
          <h3 className={styles.h2} style={{ fontSize: 14 }}>
            Subgroup performance by location
          </h3>
          <div className={styles.tableWrap} style={{ marginTop: 8 }}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>Location</th>
                  <th>n</th>
                  <th>Positives</th>
                  <th>AUROC</th>
                </tr>
              </thead>
              <tbody>
                {byLocation.map((s) => (
                  <tr key={s.subgroup}>
                    <td>{s.subgroup}</td>
                    <td>{s.n}</td>
                    <td>{s.positives}</td>
                    <td
                      style={{
                        color: s.auroc >= 0.85 ? "#5eead4" : s.auroc >= 0.78 ? "#fbbf24" : "#f87171",
                      }}
                    >
                      {s.auroc.toFixed(3)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div>
          <h3 className={styles.h2} style={{ fontSize: 14 }}>
            AUROC vs. simulated low-resolution input noise
          </h3>
          <svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label="Resolution stress test" className={styles.chart}>
            <line x1={padL} y1={y(yMin)} x2={x(xMax)} y2={y(yMin)} className={styles.axis} />
            <line x1={padL} y1={y(yMax)} x2={padL} y2={y(yMin)} className={styles.axis} />
            <line x1={padL} y1={y(0.85)} x2={x(xMax)} y2={y(0.85)} className={styles.targetLine} />
            <path d={path} className={styles.linePath} />
            {byResolution.map((p, i) => (
              <circle
                key={i}
                cx={x(p.noiseLevel)}
                cy={y(p.auroc)}
                r={3.5}
                className={styles.dot}
              />
            ))}
            {byResolution.map((p) => (
              <text
                key={`xt-${p.noiseLevel}`}
                x={x(p.noiseLevel)}
                y={h - 8}
                className={styles.axisLabel}
              >
                {p.noiseLevel.toFixed(2)}
              </text>
            ))}
            {[0.5, 0.7, 0.85, 1].map((v) => (
              <text key={v} x={6} y={y(v) + 3} className={styles.axisLabel}>
                {v.toFixed(2)}
              </text>
            ))}
            <text x={padL + 6} y={padT + 12} className={styles.axisCaption}>
              ↑ AUROC (target ≥ 0.85)
            </text>
            <text x={x(xMax) - 110} y={y(yMin) - 6} className={styles.axisCaption}>
              σ noise (std-units)
            </text>
          </svg>
        </div>
      </div>

      <div className={styles.callout}>
        Mitigations on deck (matches Milestone 3 next steps): resolution-robust training
        augmentation, automatic uncertainty inflation when input quality is low, and an
        ultrasound-derived geometry pathway so screening doesn&rsquo;t depend on full
        CT/MRI access.
      </div>
    </div>
  );
}
