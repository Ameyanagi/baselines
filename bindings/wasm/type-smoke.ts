import {
  baseline,
  baseline2d,
  fit,
  methods,
  type BaselineOptions,
  type FitResult,
  type Method,
} from "./pkg/browser.js";

const options: BaselineOptions = {
  method: "asls",
  lambda: 1e5,
  p: 0.05,
  maxIter: 20,
};
const estimated: Float64Array = baseline([1, 1.1, 4.2, 1.2, 1], options);
const result: FitResult = fit(estimated, { method: "arpls", tol: 1e-4 });
const surface: Float64Array = baseline2d(
  [
    [1, 1, 1],
    [1, 4, 1],
  ],
  { method: "rolling_ball", windowRows: 3, windowCols: 3 },
);
const selected: Method = methods()[0];

void result;
void surface;
void selected;
