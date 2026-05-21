//! Small dense linear algebra helpers.

use crate::{BaselineError, Result};

/// Solves a dense linear system with partial pivoting.
pub(crate) fn solve_dense(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Result<Vec<f64>> {
    let n = rhs.len();
    if matrix.len() != n || matrix.iter().any(|row| row.len() != n) {
        return Err(BaselineError::LengthMismatch {
            name: "matrix",
            expected: n,
            actual: matrix.len(),
        });
    }

    for pivot in 0..n {
        let mut best = pivot;
        for row in pivot + 1..n {
            if matrix[row][pivot].abs() > matrix[best][pivot].abs() {
                best = row;
            }
        }
        if matrix[best][pivot].abs() <= f64::EPSILON {
            return Err(BaselineError::LinearSolve {
                reason: "singular dense linear system",
            });
        }
        matrix.swap(pivot, best);
        rhs.swap(pivot, best);

        let pivot_row = matrix[pivot].clone();
        for row in pivot + 1..n {
            let factor = matrix[row][pivot] / matrix[pivot][pivot];
            for (entry, pivot_entry) in matrix[row][pivot..].iter_mut().zip(&pivot_row[pivot..]) {
                *entry -= factor * pivot_entry;
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }

    let mut solution = vec![0.0; n];
    for row in (0..n).rev() {
        let known = matrix[row][row + 1..]
            .iter()
            .zip(&solution[row + 1..])
            .map(|(coeff, value)| coeff * value)
            .sum::<f64>();
        solution[row] = (rhs[row] - known) / matrix[row][row];
    }
    Ok(solution)
}
