import { useMemo, useState } from "react";
import type {
  CaseAttribution,
  CohortCase,
  FeatureImportance,
} from "../domain/types";
import { FEATURE_LABELS } from "../domain/types";
import styles from "./Panel.module.css";

type Props = {
  importance: FeatureImportance[];
  attributions: CaseAttribution[];
  testCases: CohortCase[];
};

export function ExplainabilityPanel({ importance, attributions, testCases }: Props) {
  const [selectedId, setSelectedId] = useState<string>(
    attributions[0]?.caseId ?? "",
  );

  const selectedAttr = useMemo(
    () => attributions.find((a) => a.caseId === selectedId) ?? attributions[0],
    [attributions, selectedId],
  );
  const selectedCase = useMemo(
    () => testCases.find((c) => c.id === selectedId),
    [testCases, selectedId],
  );

  const maxAbsAttr = Math.max(
    1e-6,
    ...(selectedAttr?.contributions.map((c) => Math.abs(c.delta)) ?? [0]),
  );
  const maxImp = Math.max(1e-6, ...importance.map((i) => Math.abs(i.delta)));

  return (
    <div className={styles.panel}>
      <header className={styles.head}>
        <h2 className={styles.h2}>Explainability — SHAP feature attribution</h2>
        <p className={styles.sub}>
          Global view: mean |SHAP value| per feature across all test-set cases.
          Per-case view: TreeSHAP (or model-agnostic Permutation explainer for the MLP)
          attribution showing how each hemodynamic feature pushes the predicted risk above
          or below the cohort baseline rate.
        </p>
      </header>

      <div className={styles.row + " " + styles.rowSplit}>
        <div>
          <h3 className={styles.h2} style={{ fontSize: 14 }}>
            Global importance (mean |SHAP value|)
          </h3>
          <div className={styles.barWrap} style={{ marginTop: 8 }}>
            {importance.map((f) => (
              <div key={f.feature} className={styles.bar}>
                <span className={styles.muted}>{FEATURE_LABELS[f.feature]}</span>
                <div className={styles.barTrack}>
                  <div
                    className={styles.barFill}
                    style={{ width: `${(Math.abs(f.delta) / maxImp) * 100}%` }}
                  />
                </div>
                <span className={styles.barValue}>
                  {f.delta >= 0 ? "+" : ""}
                  {f.delta.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h3 className={styles.h2} style={{ fontSize: 14 }}>
            Per-case attribution
          </h3>
          <select
            className={styles.select}
            value={selectedId}
            onChange={(e) => setSelectedId(e.target.value)}
          >
            {attributions.map((a) => {
              const c = testCases.find((t) => t.id === a.caseId);
              const lbl = c?.label === 1 ? "high-risk" : "low-risk";
              const loc = c?.location ?? "";
              return (
                <option key={a.caseId} value={a.caseId}>
                  {a.caseId} · {loc} · {lbl} · p={a.prediction.toFixed(2)}
                </option>
              );
            })}
          </select>
          {selectedAttr && (
            <>
              <div className={styles.statRow} style={{ marginTop: 8 }}>
                <Stat label="True label" value={selectedCase?.label === 1 ? "High-risk" : "Low-risk"} />
                <Stat label="Predicted risk" value={selectedAttr.prediction.toFixed(3)} />
                <Stat label="Cohort baseline" value={selectedAttr.baseline.toFixed(3)} />
              </div>
              <div className={styles.barWrap} style={{ marginTop: 12 }}>
                {selectedAttr.contributions.map((c) => (
                  <div key={c.feature} className={styles.bar}>
                    <span className={styles.muted}>{FEATURE_LABELS[c.feature]}</span>
                    <div className={styles.barTrack}>
                      <div
                        className={c.delta >= 0 ? styles.barFill : styles.barFillNeg}
                        style={{ width: `${(Math.abs(c.delta) / maxAbsAttr) * 100}%` }}
                      />
                    </div>
                    <span className={styles.barValue}>
                      {c.delta >= 0 ? "+" : ""}
                      {c.delta.toFixed(3)}
                    </span>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </div>

      <div className={styles.callout}>
        Attributions are computed by Python: TreeSHAP for the tree-based winners
        (XGBoost / Random Forest) and SHAP&rsquo;s Permutation explainer for the MLP. Sign
        convention: positive bars push the prediction toward high-risk, negative bars
        push it toward low-risk, and the sum equals (prediction − cohort baseline).
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className={styles.stat}>
      <span className={styles.statLabel}>{label}</span>
      <span className={styles.statValue}>{value}</span>
    </div>
  );
}
