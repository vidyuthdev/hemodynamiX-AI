import { useState } from "react";
import type { ClinicianActionKind, DecisionLogEntry } from "../domain/types";
import styles from "./ClinicianConsole.module.css";

const ACTIONS: { kind: ClinicianActionKind; label: string }[] = [
  { kind: "follow_up_imaging", label: "Schedule follow-up imaging" },
  { kind: "intervention_referral", label: "Refer for intervention consult" },
  { kind: "watchful_waiting", label: "Watchful waiting + counseling" },
  { kind: "second_read", label: "Request second read / physics QA" },
];

function nowLabel() {
  return new Date().toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

type Props = {
  log: DecisionLogEntry[];
  onAdd: (entry: DecisionLogEntry) => void;
};

export function ClinicianConsole({ log, onAdd }: Props) {
  const [note, setNote] = useState("");

  function add(kind: ClinicianActionKind) {
    onAdd({
      id: crypto.randomUUID(),
      at: nowLabel(),
      kind,
      note: note.trim() || undefined,
    });
    setNote("");
  }

  return (
    <div className={styles.wrap}>
      <h3 className={styles.title}>Human-in-the-loop decisions</h3>
      <p className={styles.lead}>
        HemodynamiX augments judgment; it does not replace it. Log the next step you would take
        after reviewing flow features and uncertainty.
      </p>
      <div className={styles.actions}>
        {ACTIONS.map((a) => (
          <button key={a.kind} type="button" className={styles.btn} onClick={() => add(a.kind)}>
            {a.label}
          </button>
        ))}
      </div>
      <label className={styles.label}>
        Optional note (e.g., clinical context)
        <textarea
          className={styles.ta}
          rows={2}
          value={note}
          onChange={(e) => setNote(e.target.value)}
          placeholder="Patient-specific reasoning stays with the care team…"
        />
      </label>
      <div className={styles.log}>
        <span className={styles.logTitle}>Decision log (this session)</span>
        {log.length === 0 ? (
          <p className={styles.empty}>No actions logged yet.</p>
        ) : (
          <ul className={styles.ul}>
            {log.map((e) => (
              <li key={e.id} className={styles.li}>
                <span className={styles.time}>{e.at}</span>
                <span className={styles.kind}>{labelFor(e.kind)}</span>
                {e.note && <span className={styles.note}>{e.note}</span>}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

function labelFor(k: ClinicianActionKind) {
  const m = ACTIONS.find((a) => a.kind === k);
  return m?.label ?? k;
}
