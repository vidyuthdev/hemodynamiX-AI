import { useMemo } from "react";
import type { WallCell } from "../domain/types";
import styles from "./WallRiskHeatmap.module.css";

type Props = {
  grid: WallCell[][];
  selected: { ci: number; si: number } | null;
  onSelect: (ci: number, si: number) => void;
};

function cellPaint(mean: number, std: number) {
  const hue = 168 - mean * 132;
  const sat = 42 + std * 48;
  const light = 22 + (1 - mean) * 16;
  return `hsl(${hue} ${sat}% ${light}%)`;
}

export function WallRiskHeatmap({ grid, selected, onSelect }: Props) {
  const nr = grid.length;
  const nc = grid[0]?.length ?? 0;

  const legend = useMemo(() => {
    return [
      { t: "Lower risk", c: cellPaint(0.12, 0.08) },
      { t: "Higher risk", c: cellPaint(0.88, 0.1) },
    ];
  }, []);

  return (
    <div className={styles.wrap}>
      <div className={styles.head}>
        <div>
          <h2 className={styles.title}>Uncertainty-aware wall risk map</h2>
          <p className={styles.sub}>
            Unwrapped wall surface: streamwise (horizontal) × circumferential bands (vertical).
            Saturation encodes hemodynamic uncertainty (higher σ → more saturated).
          </p>
        </div>
        <div className={styles.legend}>
          {legend.map((x) => (
            <span key={x.t} className={styles.legendItem}>
              <span className={styles.swatch} style={{ background: x.c }} />
              {x.t}
            </span>
          ))}
        </div>
      </div>
      <div
        className={styles.grid}
        style={{
          gridTemplateColumns: `repeat(${nc}, minmax(0, 1fr))`,
          aspectRatio: `${nc} / ${nr}`,
        }}
        role="img"
        aria-label="Wall hemodynamic risk heatmap"
      >
        {grid.map((row, si) =>
          row.map((cell, ci) => {
            const isSel = selected?.ci === ci && selected?.si === si;
            return (
              <button
                key={`${si}-${ci}`}
                type="button"
                className={`${styles.cell} ${isSel ? styles.cellSel : ""}`}
                style={{
                  backgroundColor: cellPaint(cell.riskMean, cell.riskStd),
                  opacity: 0.55 + cell.riskMean * 0.42,
                }}
                title={`μ=${cell.riskMean.toFixed(2)} σ=${cell.riskStd.toFixed(2)}`}
                onClick={() => onSelect(ci, si)}
              />
            );
          }),
        )}
      </div>
      <p className={styles.foot}>
        Click a wall patch to align centerline traces and the quantitative readout below.
      </p>
    </div>
  );
}
