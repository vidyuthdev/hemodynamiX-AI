export type WallCell = {
  streamwise: number;
  circumferential: number;
  riskMean: number;
  riskStd: number;
};

export type CenterlineSample = {
  s: number;
  velocityMag: number;
  wss: number;
  osi: number;
  vorticity: number;
};

export type Morphology = {
  aspectRatio: number;
  sizeRatio: number;
  sphericity: number;
  undulationIndex: number;
  bulgeAmplitude: number;
  tortuosity: number;
  centerlineLength: number;
  surfaceArea: number;
  volume: number;
};

export type Cfd3DSummary = {
  solver: string;
  elements: number;
  velocityDOFs: number;
  wallFacets: number;
  tawssPa: number;
  wssP95Pa: number;
  uMaxMs: number;
  inletRadiusMm: number;
  renderPath: string;
  secondsPerCase: number;
};

export type SolverKind = "womersley" | "fem3d";

export type FocusCase = {
  id: string;
  source: CaseSource;
  location: AneurysmLocation;
  label: 0 | 1;
  solver?: SolverKind;
  morphology: Morphology;
  narrativeBullets: string[];
  uncertaintySummary: string;
  modalityNote: string;
  grid: WallCell[][];
  centerline: CenterlineSample[];
  radiusProfile: number[];
  cfd3dImage?: string | null;
  cfd3dSummary?: Cfd3DSummary | null;
};

export type ClinicianActionKind =
  | "follow_up_imaging"
  | "intervention_referral"
  | "watchful_waiting"
  | "second_read";

export type DecisionLogEntry = {
  id: string;
  at: string;
  kind: ClinicianActionKind;
  note?: string;
};

export type AneurysmLocation = "ACOM" | "MCA" | "PCOM";
export type CaseSource = "AnXplore" | "Synthetic";

export const FEATURE_KEYS = [
  "tawss",
  "osi",
  "rrt",
  "vorticity",
  "velocity",
  "pressure",
] as const;

export type FeatureKey = (typeof FEATURE_KEYS)[number];

export const FEATURE_LABELS: Record<FeatureKey, string> = {
  tawss: "TAWSS (Pa)",
  osi: "OSI (-)",
  rrt: "RRT (1/Pa)",
  vorticity: "Vorticity (1/s)",
  velocity: "Velocity (m/s)",
  pressure: "Pressure (mmHg)",
};

export type CohortCase = {
  id: string;
  source: CaseSource;
  location: AneurysmLocation;
  features: Record<FeatureKey, number>;
  label: 0 | 1;
  labelClean?: 0 | 1;
  morphology?: Morphology;
  solver?: SolverKind;
  cfd3dImage?: string;
  cfd3dSummary?: Cfd3DSummary;
};

export type Split = {
  train: number[];
  cal: number[];
  test: number[];
};

export type ModelName = string;

export type ModelEvaluation = {
  name: ModelName;
  auroc: number;
  f1Pos: number;
  sensitivity: number;
  specificity: number;
  ece: number;
  brier: number;
  accuracy: number;
  predictions: number[];
  trueLabels: (0 | 1)[];
};

export type ConformalSummary = {
  alpha: number;
  q: number;
  empiricalCoverage: number;
  intervalWidthMean: number;
  abstainRate: number;
};

export type ReliabilityBin = {
  pMean: number;
  yMean: number;
  count: number;
};

export type FeatureImportance = {
  feature: FeatureKey;
  delta: number;
};

export type CaseAttribution = {
  caseId: string;
  baseline: number;
  prediction: number;
  contributions: { feature: FeatureKey; delta: number }[];
};

export type SubgroupResult = {
  subgroup: string;
  n: number;
  auroc: number;
  positives: number;
};

export type ResolutionResult = {
  noiseLevel: number;
  auroc: number;
};

export type PipelineConfig = {
  nRealCases: number;
  nSyntheticCases: number;
  labelNoiseRate: number;
  conformalAlpha: number;
  seed: number;
};

export type PipelineResult = {
  version: string;
  generatedAt: string;
  config: PipelineConfig;
  featureKeys: FeatureKey[];
  featureLabels: Record<FeatureKey, string>;
  cohort: CohortCase[];
  split: Split;
  models: ModelEvaluation[];
  bestModel: ModelName;
  conformal: ConformalSummary;
  reliability: ReliabilityBin[];
  importance: FeatureImportance[];
  caseAttributions: CaseAttribution[];
  byLocation: SubgroupResult[];
  byResolution: ResolutionResult[];
  focusCase: FocusCase;
};
