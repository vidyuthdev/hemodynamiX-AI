import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  BrainCircuit,
  ClipboardList,
  Database,
  GitFork,
  LineChart,
  Scale,
  Shield,
  Waves,
} from "lucide-react";
import type {
  DecisionLogEntry,
  PipelineResult,
  WallCell,
} from "../domain/types";
import { loadResults } from "../data/results";
import { CenterlineTraces } from "./CenterlineTraces";
import { ClinicianConsole } from "./ClinicianConsole";
import { CohortPanel } from "./CohortPanel";
import { CalibrationPanel } from "./CalibrationPanel";
import { EquityPanel } from "./EquityPanel";
import { ExplainabilityPanel } from "./ExplainabilityPanel";
import { PerformancePanel } from "./PerformancePanel";
import { ProbeReadout } from "./ProbeReadout";
import { WallRiskHeatmap } from "./WallRiskHeatmap";
import styles from "./HemodynamiXWorkbench.module.css";

type TabId =
  | "cohort"
  | "cfd"
  | "performance"
  | "calibration"
  | "explain"
  | "equity"
  | "decisions";

const TABS: { id: TabId; label: string; icon: typeof Database }[] = [
  { id: "cohort", label: "Cohort", icon: Database },
  { id: "cfd", label: "CFD inspector", icon: Waves },
  { id: "performance", label: "Model performance", icon: LineChart },
  { id: "calibration", label: "Calibration & UQ", icon: Shield },
  { id: "explain", label: "Explainability", icon: GitFork },
  { id: "equity", label: "Equity audit", icon: Scale },
  { id: "decisions", label: "Decisions", icon: ClipboardList },
];

function defaultSelection(grid: WallCell[][]) {
  let bestCi = 0;
  let bestSi = 0;
  let best = -1;
  for (let si = 0; si < grid.length; si++) {
    for (let ci = 0; ci < grid[si].length; ci++) {
      const v = grid[si][ci].riskMean;
      if (v > best) {
        best = v;
        bestCi = ci;
        bestSi = si;
      }
    }
  }
  return { ci: bestCi, si: bestSi };
}

export function HemodynamiXWorkbench() {
  const [tab, setTab] = useState<TabId>("cohort");
  const [log, setLog] = useState<DecisionLogEntry[]>([]);
  const [data, setData] = useState<PipelineResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sel, setSel] = useState<{ ci: number; si: number }>({ ci: 0, si: 0 });

  useEffect(() => {
    let alive = true;
    loadResults()
      .then((d) => {
        if (!alive) return;
        setData(d);
        if (d.focusCase.grid.length > 0) {
          setSel(defaultSelection(d.focusCase.grid));
        }
      })
      .catch((e) => alive && setError(String(e)));
    return () => {
      alive = false;
    };
  }, []);

  const testCases = useMemo(() => {
    if (!data) return [];
    return data.split.test.map((i) => data.cohort[i]);
  }, [data]);

  if (error) {
    return (
      <div className={styles.page}>
        <header className={styles.top}>
          <p className={styles.brandName}>HemodynamiX AI</p>
        </header>
        <main className={styles.body}>
          <div className={styles.card}>
            <h2 className={styles.h2}>Pipeline output not found</h2>
            <p className={styles.pSm}>
              Could not load <code>/results.json</code>. Run the pipeline first:
            </p>
            <pre className={styles.pre}>
              python -m pipeline.run --n-real 100 --n-synth 240
            </pre>
            <p className={styles.pSm}>Reason: {error}</p>
          </div>
        </main>
      </div>
    );
  }

  if (!data) {
    return (
      <div className={styles.page}>
        <header className={styles.top}>
          <div className={styles.brand}>
            <span className={styles.brandIcon} aria-hidden>
              <BrainCircuit size={22} strokeWidth={1.75} />
            </span>
            <div>
              <p className={styles.brandName}>HemodynamiX AI</p>
              <p className={styles.brandTag}>Loading pipeline output...</p>
            </div>
          </div>
        </header>
      </div>
    );
  }

  const focus = data.focusCase;
  const selectedCell = focus.grid[sel.si]?.[sel.ci] ?? null;
  const nCols = focus.grid[0]?.length ?? 1;
  const markerS = (sel.ci + 0.5) / nCols;

  const bestModel = data.models.find((m) => m.name === data.bestModel) ?? data.models[0];

  return (
    <div className={styles.page}>
      <header className={styles.top}>
        <div className={styles.brand}>
          <span className={styles.brandIcon} aria-hidden>
            <BrainCircuit size={22} strokeWidth={1.75} />
          </span>
          <div>
            <p className={styles.brandName}>HemodynamiX AI</p>
            <p className={styles.brandTag}>
              CFD-informed, uncertainty-aware vascular risk screening
            </p>
          </div>
        </div>
        <div className={styles.meta}>
          <span className={styles.pill}>
            <Activity size={14} strokeWidth={2} aria-hidden />
            Pipeline: {data.cohort.length} cases ({data.config.nRealCases} real
            AnXplore + {data.config.nSyntheticCases} parametric) · best{" "}
            {data.bestModel} · AUROC {bestModel.auroc.toFixed(3)}
          </span>
          <span className={styles.disclaimer}>{focus.modalityNote}</span>
        </div>
      </header>

      <nav className={styles.tabs} aria-label="Workbench sections">
        {TABS.map((t) => {
          const Icon = t.icon;
          const active = t.id === tab;
          return (
            <button
              key={t.id}
              type="button"
              className={`${styles.tab} ${active ? styles.tabActive : ""}`}
              aria-current={active ? "page" : undefined}
              onClick={() => setTab(t.id)}
            >
              <Icon size={15} strokeWidth={2} />
              {t.label}
            </button>
          );
        })}
      </nav>

      <main className={styles.body}>
        {tab === "cohort" && (
          <div className={styles.card}>
            <CohortPanel cohort={data.cohort} split={data.split} />
          </div>
        )}

        {tab === "cfd" && (
          <div className={styles.cfdGrid}>
            {focus.cfd3dImage ? (
              <div className={`${styles.card} ${styles.cfd3dCard}`}>
                <div className={styles.cfd3dHeader}>
                  <h2 className={styles.h2}>
                    Steady 3-D Navier–Stokes — {focus.id}
                  </h2>
                  <span className={styles.solverBadge}>FEM3D</span>
                </div>
                <figure className={styles.cfd3dFigure}>
                  <img
                    src={`/${focus.cfd3dImage}`}
                    alt={`Wall shear stress and streamlines, case ${focus.id}`}
                    className={styles.cfd3dImage}
                    loading="lazy"
                  />
                  <figcaption className={styles.cfd3dCaption}>
                    Aneurysm wall coloured by computed wall shear stress (Pa);
                    streamlines seeded at the inlet plane and coloured by speed
                    (m/s). Off-screen PyVista render at 1920×1440.
                  </figcaption>
                </figure>
                {focus.cfd3dSummary && (
                  <dl className={styles.cfd3dStats}>
                    <div>
                      <dt>Solver</dt>
                      <dd>{focus.cfd3dSummary.solver}</dd>
                    </div>
                    <div>
                      <dt>Tetrahedra</dt>
                      <dd>{focus.cfd3dSummary.elements.toLocaleString()}</dd>
                    </div>
                    <div>
                      <dt>Velocity DOFs</dt>
                      <dd>
                        {focus.cfd3dSummary.velocityDOFs.toLocaleString()}
                      </dd>
                    </div>
                    <div>
                      <dt>Wall facets</dt>
                      <dd>{focus.cfd3dSummary.wallFacets.toLocaleString()}</dd>
                    </div>
                    <div>
                      <dt>TAWSS</dt>
                      <dd>{focus.cfd3dSummary.tawssPa.toFixed(2)} Pa</dd>
                    </div>
                    <div>
                      <dt>WSS p95</dt>
                      <dd>{focus.cfd3dSummary.wssP95Pa.toFixed(2)} Pa</dd>
                    </div>
                    <div>
                      <dt>|u|max</dt>
                      <dd>{focus.cfd3dSummary.uMaxMs.toFixed(2)} m/s</dd>
                    </div>
                    <div>
                      <dt>Inlet ⌀</dt>
                      <dd>
                        {(2 * focus.cfd3dSummary.inletRadiusMm).toFixed(2)} mm
                      </dd>
                    </div>
                    <div>
                      <dt>Solve+render</dt>
                      <dd>{focus.cfd3dSummary.secondsPerCase.toFixed(0)} s</dd>
                    </div>
                  </dl>
                )}
              </div>
            ) : null}

            <div className={styles.card}>
              <h2 className={styles.h2}>What you are looking at</h2>
              <p className={styles.pSm}>
                Case {focus.id} ({focus.source}, {focus.location}) -{" "}
                {focus.label === 1 ? "high-risk morphology" : "low-risk morphology"}
              </p>
              <ul className={styles.ul}>
                {focus.narrativeBullets.map((b, i) => (
                  <li key={i} className={styles.li}>
                    {b}
                  </li>
                ))}
              </ul>
              <p className={styles.pSm} style={{ marginTop: 12 }}>
                {focus.uncertaintySummary}
              </p>
            </div>
            {focus.grid.length > 0 && (
              <div className={styles.card}>
                <WallRiskHeatmap
                  grid={focus.grid}
                  selected={sel}
                  onSelect={(ci, si) => setSel({ ci, si })}
                />
              </div>
            )}
            {focus.centerline.length > 0 && (
              <div className={styles.card}>
                <CenterlineTraces samples={focus.centerline} markerS={markerS} />
              </div>
            )}
            {focus.grid.length > 0 && (
              <div className={`${styles.card} ${styles.cardMuted}`}>
                <ProbeReadout cell={selectedCell} streamIndex={sel.ci} nCols={nCols} />
              </div>
            )}
          </div>
        )}

        {tab === "performance" && (
          <div className={styles.card}>
            <PerformancePanel models={data.models} bestModel={data.bestModel} />
          </div>
        )}

        {tab === "calibration" && (
          <div className={styles.card}>
            <CalibrationPanel
              reliability={data.reliability}
              conformal={data.conformal}
            />
          </div>
        )}

        {tab === "explain" && (
          <div className={styles.card}>
            <ExplainabilityPanel
              importance={data.importance}
              attributions={data.caseAttributions}
              testCases={testCases}
            />
          </div>
        )}

        {tab === "equity" && (
          <div className={styles.card}>
            <EquityPanel
              byLocation={data.byLocation}
              byResolution={data.byResolution}
            />
          </div>
        )}

        {tab === "decisions" && (
          <div className={styles.card}>
            <ClinicianConsole log={log} onAdd={(e) => setLog((p) => [e, ...p])} />
          </div>
        )}
      </main>
    </div>
  );
}
