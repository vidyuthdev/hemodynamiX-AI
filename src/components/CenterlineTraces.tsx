import { useMemo } from "react";
import type { CenterlineSample } from "../domain/types";
import styles from "./CenterlineTraces.module.css";

type Props = {
  samples: CenterlineSample[];
  markerS: number | null;
};

type SeriesKey = "velocityMag" | "wss" | "osi" | "vorticity";

const SERIES: { key: SeriesKey; label: string; unit: string }[] = [
  { key: "velocityMag", label: "Velocity magnitude", unit: "m/s (norm.)" },
  { key: "wss", label: "Wall shear stress", unit: "Pa" },
  { key: "osi", label: "Oscillatory shear index", unit: "—" },
  { key: "vorticity", label: "Vorticity", unit: "1/s" },
];

function extent(values: number[]) {
  let lo = values[0];
  let hi = values[0];
  for (const v of values) {
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  if (hi - lo < 1e-9) return { lo: lo - 1, hi: hi + 1 };
  return { lo, hi };
}

export function CenterlineTraces({ samples, markerS }: Props) {
  const w = 520;
  const h = 46;
  const padL = 34;
  const padR = 8;
  const padY = 6;

  const paths = useMemo(() => {
    return SERIES.map(({ key }) => {
      const ys = samples.map((p) => p[key]);
      const { lo, hi } = extent(ys);
      const d = samples
        .map((p, i) => {
          const x = padL + p.s * (w - padL - padR);
          const y = padY + (1 - (p[key] - lo) / (hi - lo)) * (h - padY * 2);
          return `${i === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`;
        })
        .join(" ");
      return { key, d, lo, hi };
    });
  }, [samples]);

  const markerX =
    markerS == null ? null : padL + markerS * (w - padL - padR);

  return (
    <div className={styles.wrap}>
      <div className={styles.head}>
        <h2 className={styles.title}>Physics-grounded centerline traces</h2>
        <p className={styles.sub}>
          CFD-derived scalars along normalized arclength s. These are the features HemodynamiX
          uses to localize micro-disturbances before they become catastrophic events.
        </p>
      </div>
      <div className={styles.stack}>
        {SERIES.map((spec, idx) => {
          const { d, lo, hi } = paths[idx];
          return (
            <div key={spec.key} className={styles.row}>
              <div className={styles.labels}>
                <span className={styles.lab}>{spec.label}</span>
                <span className={styles.unit}>{spec.unit}</span>
              </div>
              <svg
                className={styles.svg}
                viewBox={`0 0 ${w} ${h}`}
                preserveAspectRatio="none"
                aria-hidden
              >
                <line
                  x1={padL}
                  x2={w - padR}
                  y1={h - padY}
                  y2={h - padY}
                  className={styles.axis}
                />
                <path d={d} className={styles.path} />
                {markerX != null && (
                  <line
                    x1={markerX}
                    x2={markerX}
                    y1={padY}
                    y2={h - padY}
                    className={styles.marker}
                  />
                )}
              </svg>
              <div className={styles.scale}>
                <span>{lo.toFixed(2)}</span>
                <span>{hi.toFixed(2)}</span>
              </div>
            </div>
          );
        })}
      </div>
      <div className={styles.sAxis}>
        <span>s = 0 (inlet)</span>
        <span>s = 1 (outlet)</span>
      </div>
    </div>
  );
}
