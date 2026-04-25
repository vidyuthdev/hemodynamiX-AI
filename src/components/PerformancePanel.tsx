import type { ModelEvaluation, ModelName } from "../domain/types";
import styles from "./Panel.module.css";

type Props = {
  models: ModelEvaluation[];
  bestModel: ModelName;
};

const TARGETS = [
  { key: "auroc", label: "AUROC ≥ 0.85", check: (m: ModelEvaluation) => m.auroc >= 0.85 },
  { key: "ece", label: "ECE < 0.08", check: (m: ModelEvaluation) => m.ece < 0.08 },
  { key: "sens", label: "Sensitivity ≥ 0.80", check: (m: ModelEvaluation) => m.sensitivity >= 0.8 },
  { key: "spec", label: "Specificity ≥ 0.80", check: (m: ModelEvaluation) => m.specificity >= 0.8 },
];

export function PerformancePanel({ models, bestModel }: Props) {
  const best = models.find((m) => m.name === bestModel) ?? models[0];

  return (
    <div className={styles.panel}>
      <header className={styles.head}>
        <h2 className={styles.h2}>Model performance</h2>
        <p className={styles.sub}>
          Three architectures trained on the same 6-feature hemodynamic vector with a 70/15/15
          stratified split. Probabilistic scores are evaluated on the held-out test set; the
          decision threshold for sensitivity / specificity is chosen by maximizing Youden&rsquo;s
          J on validation-equivalent data.
        </p>
      </header>

      <div className={styles.statRow}>
        <Stat label="Best model" value={best.name} hint={`AUROC ${best.auroc.toFixed(3)}`} />
        <Stat label="F1 (positive)" value={best.f1Pos.toFixed(3)} />
        <Stat label="Sensitivity" value={best.sensitivity.toFixed(3)} />
        <Stat label="Specificity" value={best.specificity.toFixed(3)} />
        <Stat label="ECE" value={best.ece.toFixed(3)} />
        <Stat label="Brier" value={best.brier.toFixed(3)} />
      </div>

      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>Model</th>
              <th>AUROC</th>
              <th>F1+</th>
              <th>Sens</th>
              <th>Spec</th>
              <th>ECE</th>
              <th>Brier</th>
              <th>Acc</th>
            </tr>
          </thead>
          <tbody>
            {models.map((m) => (
              <tr key={m.name}>
                <td>
                  {m.name === bestModel && <span className={styles.muted}>★ </span>}
                  {m.name}
                </td>
                <td>{m.auroc.toFixed(3)}</td>
                <td>{m.f1Pos.toFixed(3)}</td>
                <td>{m.sensitivity.toFixed(3)}</td>
                <td>{m.specificity.toFixed(3)}</td>
                <td>{m.ece.toFixed(3)}</td>
                <td>{m.brier.toFixed(3)}</td>
                <td>{m.accuracy.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className={styles.callout}>
        Pre-registered targets:{" "}
        {TARGETS.map((t, i) => (
          <span key={t.key}>
            <strong style={{ color: t.check(best) ? "#5eead4" : "#f87171" }}>
              {t.check(best) ? "✓" : "✗"} {t.label}
            </strong>
            {i < TARGETS.length - 1 ? " · " : ""}
          </span>
        ))}
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
