export { default, initSync } from "./baselines_rs.js";
export type {
  InitInput,
  InitOutput,
  SyncInitInput,
} from "./baselines_rs.js";

export type Method =
  | "asls"
  | "arpls"
  | "airpls"
  | "rolling_ball"
  | "polynomial";
export type Method2D = "asls" | "arpls" | "rolling_ball" | "polynomial";
export type NumericArray = ArrayLike<number>;
export type NumericMatrix = ReadonlyArray<ArrayLike<number>>;

export interface BaselineOptions {
  method?: Method;
  lambda?: number;
  lam?: number;
  p?: number;
  maxIter?: number;
  max_iter?: number;
  tol?: number;
  windowSize?: number;
  window_size?: number;
  order?: number;
}

export interface BaselineOptions2D {
  method?: Method2D;
  rows?: number;
  cols?: number;
  lambda?: number;
  lam?: number;
  lambdaRows?: number;
  lambda_rows?: number;
  lambdaCols?: number;
  lambda_cols?: number;
  p?: number;
  maxIter?: number;
  max_iter?: number;
  tol?: number;
  cgMaxIter?: number;
  cg_max_iter?: number;
  cgTol?: number;
  cg_tol?: number;
  windowRows?: number;
  window_rows?: number;
  windowCols?: number;
  window_cols?: number;
  order?: number;
}

export interface FitReport {
  iterations: number;
  converged: boolean;
  tolerance: number;
}

export interface FitResult {
  baseline: Float64Array;
  corrected: Float64Array;
  report: FitReport;
}

export interface FitResult2D extends FitResult {
  rows: number;
  cols: number;
  shape: [number, number];
}

export function fit(data: NumericArray, options?: BaselineOptions): FitResult;
export function baseline(
  data: NumericArray,
  options?: BaselineOptions,
): Float64Array;
export function correct(
  data: NumericArray,
  options?: BaselineOptions,
): Float64Array;
export function baselineWith(data: NumericArray, method: Method): Float64Array;
export function correctWith(data: NumericArray, method: Method): Float64Array;

export function fit2d(
  data: NumericArray | NumericMatrix,
  options?: BaselineOptions2D,
): FitResult2D;
export function fit2d(
  data: NumericArray,
  rows: number,
  cols: number,
  method?: Method2D,
): FitResult2D;
export function baseline2d(
  data: NumericArray | NumericMatrix,
  options?: BaselineOptions2D,
): Float64Array;
export function baseline2d(
  data: NumericArray,
  rows: number,
  cols: number,
  method?: Method2D,
): Float64Array;
export function correct2d(
  data: NumericArray | NumericMatrix,
  options?: BaselineOptions2D,
): Float64Array;
export function correct2d(
  data: NumericArray,
  rows: number,
  cols: number,
  method?: Method2D,
): Float64Array;

export function methods(): Method[];
export function methods2d(): Method2D[];
export const availableMethods: typeof methods;
export const availableMethods2d: typeof methods2d;
