import initWasm, {
  availableMethods as rawAvailableMethods,
  availableMethods2d as rawAvailableMethods2d,
  fit2dConfigured,
  fitConfigured,
  initSync,
} from "./baselines_rs.js";

export { initSync };
export default initWasm;

function toFloat64Array(data) {
  if (data instanceof Float64Array) {
    return data;
  }
  if (data == null || typeof data.length !== "number") {
    throw new TypeError("data must be an array or typed array of numbers");
  }
  return Float64Array.from(data);
}

function splitOptions(options, defaultMethod = "asls") {
  if (options == null) {
    return { method: defaultMethod, params: {} };
  }
  if (typeof options !== "object" || Array.isArray(options)) {
    throw new TypeError("options must be an object");
  }
  const { method = defaultMethod, ...params } = options;
  return { method, params };
}

const OPTION_KEYS_1D = [
  "lambda",
  "lam",
  "p",
  "maxIter",
  "max_iter",
  "tol",
  "windowSize",
  "window_size",
  "order",
];
const OPTION_KEYS_2D = [
  "lambda",
  "lam",
  "lambdaRows",
  "lambda_rows",
  "lambdaCols",
  "lambda_cols",
  "p",
  "maxIter",
  "max_iter",
  "tol",
  "cgMaxIter",
  "cg_max_iter",
  "cgTol",
  "cg_tol",
  "windowRows",
  "window_rows",
  "windowCols",
  "window_cols",
  "order",
];

function validateParams(params, allowed) {
  for (const key of Object.keys(params)) {
    if (!allowed.includes(key)) {
      throw new TypeError(
        `unknown option '${key}'; expected one of ${allowed.join(", ")}`,
      );
    }
  }
}

function optionAlias(params, preferred, alias) {
  if (params[preferred] != null && params[alias] != null) {
    throw new TypeError(
      `options cannot contain both '${preferred}' and '${alias}'`,
    );
  }
  return params[preferred] ?? params[alias];
}

function rawFit(data, options) {
  const values = toFloat64Array(data);
  const { method, params } = splitOptions(options);
  validateParams(params, OPTION_KEYS_1D);
  return fitConfigured(
    values,
    method,
    optionAlias(params, "lambda", "lam"),
    params.p,
    optionAlias(params, "maxIter", "max_iter"),
    params.tol,
    optionAlias(params, "windowSize", "window_size"),
    params.order,
  );
}

function report(result) {
  return {
    iterations: result.iterations,
    converged: result.converged,
    tolerance: result.tolerance,
  };
}

/** Fits a baseline and returns the baseline, corrected data, and metadata. */
export function fit(data, options = {}) {
  const result = rawFit(data, options);
  try {
    return {
      baseline: result.baseline,
      corrected: result.corrected,
      report: report(result),
    };
  } finally {
    result.free();
  }
}

/** Estimates a baseline using a method and optional parameters. */
export function baseline(data, options = {}) {
  const result = rawFit(data, options);
  try {
    return result.baseline;
  } finally {
    result.free();
  }
}

/** Returns `data - baseline` using a method and optional parameters. */
export function correct(data, options = {}) {
  const result = rawFit(data, options);
  try {
    return result.corrected;
  } finally {
    result.free();
  }
}

/** Backward-compatible method-selecting baseline helper. */
export function baselineWith(data, method) {
  return baseline(data, { method });
}

/** Backward-compatible method-selecting correction helper. */
export function correctWith(data, method) {
  return correct(data, { method });
}

function isNestedMatrix(data) {
  return (
    Array.isArray(data) &&
    data.length > 0 &&
    data[0] != null &&
    typeof data[0] !== "number" &&
    typeof data[0].length === "number"
  );
}

function normalizeMatrix(data, optionsOrRows, legacyCols, legacyMethod) {
  if (typeof optionsOrRows === "number") {
    return {
      values: toFloat64Array(data),
      rows: optionsOrRows,
      cols: legacyCols,
      method: legacyMethod ?? "asls",
      params: {},
    };
  }

  const options = optionsOrRows ?? {};
  if (typeof options !== "object" || Array.isArray(options)) {
    throw new TypeError("2D options must be an object");
  }
  let { rows, cols } = options;
  const { method = "asls", rows: _rows, cols: _cols, ...params } = options;
  let values;

  if (isNestedMatrix(data)) {
    const inferredRows = data.length;
    const inferredCols = data[0].length;
    if (!data.every((row) => row.length === inferredCols)) {
      throw new TypeError("2D data rows must all have the same length");
    }
    if (rows != null && rows !== inferredRows) {
      throw new RangeError("options.rows does not match the nested data");
    }
    if (cols != null && cols !== inferredCols) {
      throw new RangeError("options.cols does not match the nested data");
    }
    rows = inferredRows;
    cols = inferredCols;
    values = Float64Array.from(data.flatMap((row) => Array.from(row)));
  } else {
    values = toFloat64Array(data);
  }

  if (!Number.isInteger(rows) || rows <= 0) {
    throw new RangeError("options.rows must be a positive integer");
  }
  if (!Number.isInteger(cols) || cols <= 0) {
    throw new RangeError("options.cols must be a positive integer");
  }
  if (values.length !== rows * cols) {
    throw new RangeError("data length does not match rows * cols");
  }

  return { values, rows, cols, method, params };
}

function rawFit2d(data, optionsOrRows, legacyCols, legacyMethod) {
  const { values, rows, cols, method, params } = normalizeMatrix(
    data,
    optionsOrRows,
    legacyCols,
    legacyMethod,
  );
  validateParams(params, OPTION_KEYS_2D);
  return fit2dConfigured(
    values,
    rows,
    cols,
    method,
    optionAlias(params, "lambda", "lam"),
    optionAlias(params, "lambdaRows", "lambda_rows"),
    optionAlias(params, "lambdaCols", "lambda_cols"),
    params.p,
    optionAlias(params, "maxIter", "max_iter"),
    params.tol,
    optionAlias(params, "cgMaxIter", "cg_max_iter"),
    optionAlias(params, "cgTol", "cg_tol"),
    optionAlias(params, "windowRows", "window_rows"),
    optionAlias(params, "windowCols", "window_cols"),
    params.order,
  );
}

/** Fits a 2D baseline and returns flat row-major outputs and metadata. */
export function fit2d(data, optionsOrRows = {}, legacyCols, legacyMethod) {
  const result = rawFit2d(data, optionsOrRows, legacyCols, legacyMethod);
  try {
    return {
      baseline: result.baseline,
      corrected: result.corrected,
      rows: result.rows,
      cols: result.cols,
      shape: [result.rows, result.cols],
      report: report(result),
    };
  } finally {
    result.free();
  }
}

/** Estimates a flat row-major 2D baseline. */
export function baseline2d(data, optionsOrRows = {}, legacyCols, legacyMethod) {
  const result = rawFit2d(data, optionsOrRows, legacyCols, legacyMethod);
  try {
    return result.baseline;
  } finally {
    result.free();
  }
}

/** Corrects flat or nested 2D data and returns a flat row-major array. */
export function correct2d(data, optionsOrRows = {}, legacyCols, legacyMethod) {
  const result = rawFit2d(data, optionsOrRows, legacyCols, legacyMethod);
  try {
    return result.corrected;
  } finally {
    result.free();
  }
}

/** Lists supported one-dimensional methods. */
export function methods() {
  return rawAvailableMethods().split(",");
}

/** Lists supported two-dimensional methods. */
export function methods2d() {
  return rawAvailableMethods2d().split(",");
}

export const availableMethods = methods;
export const availableMethods2d = methods2d;
