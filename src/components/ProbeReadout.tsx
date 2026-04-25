import type { WallCell } from "../domain/types";
import styles from "./ProbeReadout.module.css";

type Props = {
  cell: WallCell | null;
  streamIndex: number;
  nCols: number;
};

export function ProbeReadout({ cell, streamIndex, nCols }: Props) {
  const s = (streamIndex + 0.5) / nCols;
  return (
    <div className={styles.wrap}>
      <h3 className={styles.title}>Selected wall patch</h3>
      {cell ? (
        <dl className={styles.dl}>
          <div>
            <dt>Streamwise s</dt>
            <dd>{s.toFixed(3)}</dd>
          </div>
          <div>
            <dt>Circumferential band</dt>
            <dd>{cell.circumferential.toFixed(3)}</dd>
          </div>
          <div>
            <dt>Risk posterior mean μ</dt>
            <dd>{cell.riskMean.toFixed(3)}</dd>
          </div>
          <div>
            <dt>Uncertainty σ</dt>
            <dd>{cell.riskStd.toFixed(3)}</dd>
          </div>
          <div className={styles.full}>
            <dt>Clinician-facing read</dt>
            <dd>
              {cell.riskStd > 0.22
                ? "High epistemic spread here—treat the risk color as provisional until mesh / acquisition constraints are reviewed."
                : cell.riskMean > 0.62
                  ? "Flow disturbance signature is elevated; correlate with anatomy and symptoms before escalation."
                  : "Relatively stable hemodynamics at this patch; continue holistic assessment."}
            </dd>
          </div>
        </dl>
      ) : (
        <p className={styles.empty}>Select a wall cell on the map to populate this panel.</p>
      )}
    </div>
  );
}
