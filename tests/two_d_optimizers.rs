use baselines::two_d::optimizers::{
    AdaptiveMinmax2DParams, CollabPls2DParams, IndividualAxes2DParams, adaptive_minmax, collab_pls,
    individual_axes, individual_axes_into,
};
use baselines::whittaker::{AslsParams, WhittakerParams};
use baselines::{BaselineError, MatrixView, MatrixViewMut};

#[test]
fn two_d_optimizer_methods_preserve_constant_surfaces() {
    let data = vec![2.5; 30];
    let input = MatrixView::row_major(&data, 5, 6).unwrap();
    let second = vec![3.0; 30];
    let second_input = MatrixView::row_major(&second, 5, 6).unwrap();

    for fit in [
        adaptive_minmax(input, AdaptiveMinmax2DParams::default()).unwrap(),
        individual_axes(input, IndividualAxes2DParams::default()).unwrap(),
    ] {
        assert_eq!(fit.shape(), (5, 6));
        assert!(fit.baseline.iter().all(|value| (*value - 2.5).abs() < 1e-6));
    }

    let fits = collab_pls(&[input, second_input], CollabPls2DParams::default()).unwrap();
    assert_eq!(fits.len(), 2);
    assert!(
        fits[0]
            .baseline
            .iter()
            .all(|value| (*value - 2.5).abs() < 1e-6)
    );
    assert!(
        fits[1]
            .baseline
            .iter()
            .all(|value| (*value - 3.0).abs() < 1e-6)
    );
}

#[test]
fn two_d_individual_axes_into_reuses_output_buffer() {
    let data = vec![
        1.0, 1.1, 1.2, 1.3, 1.4, //
        1.1, 1.2, 1.3, 1.4, 1.5, //
        1.2, 1.3, 1.4, 1.5, 1.6, //
        1.3, 1.4, 1.5, 1.6, 1.7, //
        1.4, 1.5, 1.6, 1.7, 1.8,
    ];
    let input = MatrixView::row_major(&data, 5, 5).unwrap();
    let mut output = vec![0.0; data.len()];
    let output_view = MatrixViewMut::row_major(&mut output, 5, 5).unwrap();

    let report =
        individual_axes_into(input, IndividualAxes2DParams::default(), output_view).unwrap();

    assert!(report.converged);
    assert!(output.iter().all(|value| value.is_finite()));
}

#[test]
fn two_d_optimizers_reject_invalid_parameters_and_shapes() {
    let data = vec![1.0; 30];
    let input = MatrixView::row_major(&data, 5, 6).unwrap();

    let error = adaptive_minmax(
        input,
        AdaptiveMinmax2DParams {
            max_iter: 0,
            ..AdaptiveMinmax2DParams::default()
        },
    )
    .unwrap_err();
    assert!(matches!(error, BaselineError::InvalidParameter { .. }));

    let error = individual_axes(
        input,
        IndividualAxes2DParams {
            asls: AslsParams {
                whittaker: WhittakerParams::default(),
                p: 1.0,
            },
        },
    )
    .unwrap_err();
    assert!(matches!(error, BaselineError::InvalidParameter { .. }));

    let mut output = vec![0.0; data.len()];
    let output = MatrixViewMut::row_major(&mut output, 6, 5).unwrap();
    let error = individual_axes_into(input, IndividualAxes2DParams::default(), output).unwrap_err();
    assert!(matches!(error, BaselineError::LengthMismatch { .. }));

    let error = collab_pls(&[], CollabPls2DParams::default()).unwrap_err();
    assert!(matches!(error, BaselineError::EmptyInput));

    let other = vec![1.0; 25];
    let other_input = MatrixView::row_major(&other, 5, 5).unwrap();
    let error = collab_pls(&[input, other_input], CollabPls2DParams::default()).unwrap_err();
    assert!(matches!(error, BaselineError::LengthMismatch { .. }));
}
