use baselines::{BaselineError, Fit1D, Fit2D, FitReport, MatrixShape, MatrixView, MatrixViewMut};

#[test]
fn fit1d_correction_validates_lengths() {
    let fit = Fit1D {
        baseline: vec![0.5, 1.0, 1.5],
        report: FitReport::new(1, true, 0.0),
    };

    assert_eq!(fit.corrected(&[1.0, 1.5, 3.0]).unwrap(), [0.5, 0.5, 1.5]);

    let mut output = vec![0.0; 3];
    fit.corrected_into(&[1.0, 1.5, 3.0], &mut output).unwrap();
    assert_eq!(output, [0.5, 0.5, 1.5]);

    let error = fit.corrected(&[1.0, 1.5]).unwrap_err();
    assert!(matches!(error, BaselineError::LengthMismatch { .. }));

    let mut short_output = vec![0.0; 2];
    let error = fit
        .corrected_into(&[1.0, 1.5, 3.0], &mut short_output)
        .unwrap_err();
    assert!(matches!(error, BaselineError::LengthMismatch { .. }));
}

#[test]
fn fit2d_tracks_shape_and_corrects_row_major_data() {
    let fit = Fit2D::new(
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        2,
        3,
        FitReport::new(1, true, 0.0),
    )
    .unwrap();

    assert_eq!(fit.shape(), (2, 3));
    assert_eq!(fit.len(), 6);
    assert!(!fit.is_empty());
    assert_eq!(
        fit.corrected(&[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]).unwrap(),
        [0.9, 0.8, 0.7, 1.6, 1.5, 1.4]
    );

    let mut output = vec![0.0; 6];
    fit.corrected_into(&[1.0, 1.0, 1.0, 2.0, 2.0, 2.0], &mut output)
        .unwrap();
    assert_eq!(output, [0.9, 0.8, 0.7, 1.6, 1.5, 1.4]);

    let error = Fit2D::new(vec![0.0; 5], 2, 3, FitReport::new(1, true, 0.0)).unwrap_err();
    assert!(matches!(error, BaselineError::LengthMismatch { .. }));
}

#[test]
fn matrix_view_uses_row_major_indexing() {
    let data = [0.0, 1.0, 2.0, 10.0, 11.0, 12.0];
    let view = MatrixView::row_major(&data, 2, 3).unwrap();

    assert_eq!(view.shape(), MatrixShape { rows: 2, cols: 3 });
    assert_eq!(view.rows(), 2);
    assert_eq!(view.cols(), 3);
    assert_eq!(view.len(), 6);
    assert!(!view.is_empty());
    assert_eq!(view.get(0, 2), Some(2.0));
    assert_eq!(view.get(1, 0), Some(10.0));
    assert_eq!(view.get(2, 0), None);
    assert_eq!(view.row(1).unwrap(), [10.0, 11.0, 12.0]);
}

#[test]
fn matrix_view_mut_updates_row_major_data() {
    let mut data = [0.0; 6];
    let mut view = MatrixViewMut::row_major(&mut data, 2, 3).unwrap();

    view.set(1, 2, 12.0).unwrap();
    view.row_mut(0).unwrap().copy_from_slice(&[1.0, 2.0, 3.0]);

    assert_eq!(view.get(1, 2), Some(12.0));
    assert_eq!(view.row(0).unwrap(), [1.0, 2.0, 3.0]);
    assert_eq!(view.as_slice(), [1.0, 2.0, 3.0, 0.0, 0.0, 12.0]);

    let error = view.set(2, 0, 1.0).unwrap_err();
    assert!(matches!(error, BaselineError::InvalidParameter { .. }));
}

#[test]
fn matrix_views_reject_invalid_input() {
    let error = MatrixView::row_major(&[], 0, 3).unwrap_err();
    assert!(matches!(error, BaselineError::InvalidParameter { .. }));

    let error = MatrixView::row_major(&[1.0, 2.0], 1, 3).unwrap_err();
    assert!(matches!(error, BaselineError::LengthMismatch { .. }));

    let error = MatrixView::row_major(&[1.0, f64::NAN], 1, 2).unwrap_err();
    assert!(matches!(error, BaselineError::NonFiniteInput { index: 1 }));

    let mut output = [0.0; 2];
    let error = MatrixViewMut::row_major(&mut output, 1, 3).unwrap_err();
    assert!(matches!(error, BaselineError::LengthMismatch { .. }));
}
