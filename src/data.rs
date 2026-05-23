//! Slice-based data views used by one- and two-dimensional algorithms.

use crate::{BaselineError, Result};

/// Matrix memory layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MatrixLayout {
    /// Rows are contiguous and columns vary fastest.
    #[default]
    RowMajor,
}

/// Matrix shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixShape {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
}

impl MatrixShape {
    /// Creates a non-empty matrix shape.
    pub fn new(rows: usize, cols: usize) -> Result<Self> {
        checked_matrix_len(rows, cols)?;
        Ok(Self { rows, cols })
    }

    /// Returns `rows * cols`.
    #[must_use]
    pub fn len(self) -> usize {
        self.rows * self.cols
    }

    /// Returns whether the shape has no elements.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
}

/// Immutable row-major matrix view over a slice.
#[derive(Debug, Clone, Copy)]
pub struct MatrixView<'a> {
    data: &'a [f64],
    shape: MatrixShape,
    layout: MatrixLayout,
}

impl<'a> MatrixView<'a> {
    /// Creates a validated row-major matrix view.
    pub fn row_major(data: &'a [f64], rows: usize, cols: usize) -> Result<Self> {
        Self::new(data, rows, cols, MatrixLayout::RowMajor)
    }

    /// Creates a validated matrix view.
    pub fn new(data: &'a [f64], rows: usize, cols: usize, layout: MatrixLayout) -> Result<Self> {
        validate_matrix_data("data", data, rows, cols)?;
        Ok(Self {
            data,
            shape: MatrixShape { rows, cols },
            layout,
        })
    }

    /// Returns the underlying data.
    #[must_use]
    pub fn as_slice(&self) -> &'a [f64] {
        self.data
    }

    /// Returns the matrix shape.
    #[must_use]
    pub fn shape(&self) -> MatrixShape {
        self.shape
    }

    /// Returns the number of rows.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.shape.rows
    }

    /// Returns the number of columns.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.shape.cols
    }

    /// Returns the matrix layout.
    #[must_use]
    pub fn layout(&self) -> MatrixLayout {
        self.layout
    }

    /// Returns the number of matrix elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.shape.len()
    }

    /// Returns whether the matrix has no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.shape.is_empty()
    }

    /// Returns the value at `(row, col)`, or `None` if out of bounds.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> Option<f64> {
        self.index(row, col).map(|index| self.data[index])
    }

    /// Returns a contiguous row, or `None` if out of bounds.
    #[must_use]
    pub fn row(&self, row: usize) -> Option<&'a [f64]> {
        if row >= self.shape.rows {
            return None;
        }
        let start = row * self.shape.cols;
        Some(&self.data[start..start + self.shape.cols])
    }

    fn index(&self, row: usize, col: usize) -> Option<usize> {
        if row < self.shape.rows && col < self.shape.cols {
            Some(row * self.shape.cols + col)
        } else {
            None
        }
    }
}

/// Mutable row-major matrix view over a slice.
#[derive(Debug)]
pub struct MatrixViewMut<'a> {
    data: &'a mut [f64],
    shape: MatrixShape,
    layout: MatrixLayout,
}

impl<'a> MatrixViewMut<'a> {
    /// Creates a validated mutable row-major matrix view.
    pub fn row_major(data: &'a mut [f64], rows: usize, cols: usize) -> Result<Self> {
        Self::new(data, rows, cols, MatrixLayout::RowMajor)
    }

    /// Creates a validated mutable matrix view.
    pub fn new(
        data: &'a mut [f64],
        rows: usize,
        cols: usize,
        layout: MatrixLayout,
    ) -> Result<Self> {
        validate_matrix_len("data", rows, cols, data.len())?;
        Ok(Self {
            data,
            shape: MatrixShape { rows, cols },
            layout,
        })
    }

    /// Returns the underlying data.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        self.data
    }

    /// Returns the underlying mutable data.
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        self.data
    }

    /// Returns the matrix shape.
    #[must_use]
    pub fn shape(&self) -> MatrixShape {
        self.shape
    }

    /// Returns the number of rows.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.shape.rows
    }

    /// Returns the number of columns.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.shape.cols
    }

    /// Returns the matrix layout.
    #[must_use]
    pub fn layout(&self) -> MatrixLayout {
        self.layout
    }

    /// Returns the number of matrix elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.shape.len()
    }

    /// Returns whether the matrix has no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.shape.is_empty()
    }

    /// Returns the value at `(row, col)`, or `None` if out of bounds.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> Option<f64> {
        self.index(row, col).map(|index| self.data[index])
    }

    /// Sets the value at `(row, col)`.
    pub fn set(&mut self, row: usize, col: usize, value: f64) -> Result<()> {
        let index = self
            .index(row, col)
            .ok_or(BaselineError::InvalidParameter {
                name: "index",
                reason: "row or column is outside the matrix shape",
            })?;
        self.data[index] = value;
        Ok(())
    }

    /// Returns a contiguous row, or `None` if out of bounds.
    #[must_use]
    pub fn row(&self, row: usize) -> Option<&[f64]> {
        if row >= self.shape.rows {
            return None;
        }
        let start = row * self.shape.cols;
        Some(&self.data[start..start + self.shape.cols])
    }

    /// Returns a mutable contiguous row, or `None` if out of bounds.
    #[must_use]
    pub fn row_mut(&mut self, row: usize) -> Option<&mut [f64]> {
        if row >= self.shape.rows {
            return None;
        }
        let start = row * self.shape.cols;
        Some(&mut self.data[start..start + self.shape.cols])
    }

    fn index(&self, row: usize, col: usize) -> Option<usize> {
        if row < self.shape.rows && col < self.shape.cols {
            Some(row * self.shape.cols + col)
        } else {
            None
        }
    }
}

/// Validates finite row-major matrix input.
pub fn validate_matrix_data(
    name: &'static str,
    data: &[f64],
    rows: usize,
    cols: usize,
) -> Result<()> {
    validate_matrix_len(name, rows, cols, data.len())?;
    for (index, value) in data.iter().enumerate() {
        if !value.is_finite() {
            return Err(BaselineError::NonFiniteInput { index });
        }
    }
    Ok(())
}

/// Validates a matrix slice length against a non-empty shape.
pub fn validate_matrix_len(
    name: &'static str,
    rows: usize,
    cols: usize,
    actual: usize,
) -> Result<()> {
    let expected = checked_matrix_len(rows, cols)?;
    if expected != actual {
        return Err(BaselineError::LengthMismatch {
            name,
            expected,
            actual,
        });
    }
    Ok(())
}

fn checked_matrix_len(rows: usize, cols: usize) -> Result<usize> {
    if rows == 0 {
        return Err(BaselineError::InvalidParameter {
            name: "rows",
            reason: "must be greater than zero",
        });
    }
    if cols == 0 {
        return Err(BaselineError::InvalidParameter {
            name: "cols",
            reason: "must be greater than zero",
        });
    }
    rows.checked_mul(cols)
        .ok_or(BaselineError::InvalidParameter {
            name: "shape",
            reason: "rows * cols overflows usize",
        })
}
