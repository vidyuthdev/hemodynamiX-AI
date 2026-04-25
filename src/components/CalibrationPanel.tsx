import type { ConformalSummary, ReliabilityBin } from "../domain/types";
import styles from "./Panel.module.css";

type Props = {
  reliability: ReliabilityBin[];
  conformal: ConformalSummary;
};

export function CalibrationPanel({ reliability, conformal }: Props) {
  const w = 360;
  const h = 220;
  const padL = 36;
  const padB = 28;
  const padT = 12;

  const x = (v: number) => padL + v * (w - padL - 8);
  const y = (v: number) => padT + (1 - v) * (h - padT - padB);
  const r = (b: ReliabilityBin) => 4 + Math.sqrt(b.count) * 2;

  return (
    <div className={styles.panel}>
      <header className={styles.head}>
        <h2 className={styles.h2}>Calibration &amp; uncertainty</h2>
        <p className={styles.sub}>
          Reliability diagram (predicted probability vs. empirical rupture rate per bin) and a
          split-conformal wrapper that produces empirically valid prediction intervals at
          α = {conformal.alpha.toFixed(2)} (target coverage ≥ {(1 - conformal.alpha).toFixed(2)}).
        </p>
      </header>

      <div className={styles.row + " " + styles.rowSplit}>
        <div>
          <svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label="Reliability diagram" className={styles.chart}>
            <line x1={padL} y1={y(0)} x2={x(1)} y2={y(0)} className={styles.axis} />
            <line x1={padL} y1={y(1)} x2={padL} y2={y(0)} className={styles.axis} />
            <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} className={styles.diag} />
            {reliability.map((b, i) =>
              b.count === 0 ? null : (
                <circle
                  key={i}
                  cx={x(b.pMean)}
                  cy={y(b.yMean)}
                  r={r(b)}
                  className={styles.dot}
                />
              ),
            )}
            {[0, 0.25, 0.5, 0.75, 1].map((v) => (
              <text key={v} x={x(v)} y={h - 8} className={styles.axisLabel}>
                {v.toFixed(2)}
              </text>
            ))}
            {[0, 0.25, 0.5, 0.75, 1].map((v) => (
              <text key={v} x={6} y={y(v) + 3} className={styles.axisLabel}>
                {v.toFixed(2)}
              </text>
            ))}
            <text x={padL + 6} y={padT + 12} className={styles.axisCaption}>
              ↑ empirical rate
            </text>
            <text x={x(1) - 80} y={y(0) - 6} className={styles.axisCaption}>
              predicted →
            </text>
          </svg>
        </div>

        <div className={styles.statRow} style={{ alignContent: "start" }}>
          <Stat label="Conformal q (radius)" value={conformal.q.toFixed(3)} hint="Quantile of nonconformity scores on calibration set" />
          <Stat
            label="Empirical coverage"
            value={(conformal.empiricalCoverage * 100).toFixed(1) + "%"}
            hint={`Target ≥ ${((1 - conformal.alpha) * 100).toFixed(0)}%`}
          />
          <Stat
            label="Mean prediction set width"
            value={conformal.intervalWidthMean.toFixed(3)}
            hint="1 = decisive, 2 = abstain"
          />
          <Stat
            label="Abstain rate"
            value={(conformal.abstainRate * 100).toFixed(1) + "%"}
            hint="Cases where model returns {0,1} (defers to clinician)"
          />
        </div>
      </div>

      <div className={styles.callout}>
        Why this matters: clinicians repeatedly told us they cannot act on a probability without
        knowing how much to trust it. Conformal intervals give a distribution-free coverage
        guarantee — &ldquo;if the data behaves like calibration, the truth is in this interval ≥
        {" "}
        {((1 - conformal.alpha) * 100).toFixed(0)}% of the time.&rdquo;
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
