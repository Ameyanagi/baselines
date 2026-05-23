use baselines::two_d::morphology::{
    Morphology2DParams, imor, mor, noise_median, rolling_ball, rolling_ball_into, tophat,
};
use baselines::{BaselineError, MatrixView, MatrixViewMut};

#[test]
fn two_d_morphology_methods_preserve_constant_surfaces() {
    let data = vec![3.0; 30];
    let input = MatrixView::row_major(&data, 5, 6).unwrap();
    let params = Morphology2DParams {
        window_rows: 5,
        window_cols: 5,
    };

    for fit in [
        rolling_ball(input, params).unwrap(),
        tophat(input, params).unwrap(),
        mor(input, params).unwrap(),
        imor(input, params).unwrap(),
        noise_median(input, params).unwrap(),
    ] {
        assert_eq!(fit.shape(), (5, 6));
        assert!(
            fit.baseline
                .iter()
                .all(|value| (*value - 3.0).abs() < 1e-12)
        );
    }
}

#[test]
fn two_d_morphology_into_reuses_output_buffer() {
    let data = vec![
        1.0, 1.0, 1.0, 1.0, //
        1.0, 5.0, 5.0, 1.0, //
        1.0, 5.0, 5.0, 1.0, //
        1.0, 1.0, 1.0, 1.0,
    ];
    let input = MatrixView::row_major(&data, 4, 4).unwrap();
    let mut output = vec![0.0; data.len()];
    let output_view = MatrixViewMut::row_major(&mut output, 4, 4).unwrap();

    let report = rolling_ball_into(
        input,
        Morphology2DParams {
            window_rows: 3,
            window_cols: 3,
        },
        output_view,
    )
    .unwrap();

    assert!(report.converged);
    assert!(output.iter().all(|value| value.is_finite()));
    assert!(output.iter().all(|value| *value <= 5.0));
}

#[test]
fn two_d_morphology_rejects_invalid_windows_and_output_shapes() {
    let data = vec![1.0; 6];
    let input = MatrixView::row_major(&data, 2, 3).unwrap();

    let error = rolling_ball(
        input,
        Morphology2DParams {
            window_rows: 0,
            window_cols: 3,
        },
    )
    .unwrap_err();
    assert!(matches!(error, BaselineError::InvalidParameter { .. }));

    let mut output = vec![0.0; 6];
    let output = MatrixViewMut::row_major(&mut output, 3, 2).unwrap();
    let error = rolling_ball_into(input, Morphology2DParams::default(), output).unwrap_err();
    assert!(matches!(error, BaselineError::LengthMismatch { .. }));
}
