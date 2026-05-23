use baselines::two_d::spline::{
    Irsqr2DParams, MixtureModel2DParams, PsplineAirPls2DParams, PsplineArPls2DParams,
    PsplineAsls2DParams, PsplineBrPls2DParams, PsplineIarPls2DParams, PsplineIasls2DParams,
    PsplineLsrPls2DParams, PsplinePsalsa2DParams, Spline2DParams, Spline2DWorkspace, irsqr,
    mixture_model, pspline_airpls, pspline_arpls, pspline_asls, pspline_asls_into, pspline_brpls,
    pspline_iarpls, pspline_iasls, pspline_lsrpls, pspline_psalsa,
};
use baselines::{BaselineError, MatrixView, MatrixViewMut};

#[test]
fn two_d_spline_methods_preserve_constant_surfaces() {
    let data = vec![2.5; 30];
    let input = MatrixView::row_major(&data, 5, 6).unwrap();

    for fit in [
        pspline_asls(input, PsplineAsls2DParams::default()).unwrap(),
        pspline_iasls(input, PsplineIasls2DParams::default()).unwrap(),
        pspline_airpls(input, PsplineAirPls2DParams::default()).unwrap(),
        pspline_arpls(input, PsplineArPls2DParams::default()).unwrap(),
        pspline_iarpls(input, PsplineIarPls2DParams::default()).unwrap(),
        pspline_psalsa(
            input,
            PsplinePsalsa2DParams {
                k: Some(1.0),
                ..PsplinePsalsa2DParams::default()
            },
        )
        .unwrap(),
        pspline_brpls(input, PsplineBrPls2DParams::default()).unwrap(),
        pspline_lsrpls(input, PsplineLsrPls2DParams::default()).unwrap(),
        irsqr(input, Irsqr2DParams::default()).unwrap(),
        mixture_model(input, MixtureModel2DParams::default()).unwrap(),
    ] {
        assert_eq!(fit.shape(), (5, 6));
        assert!(fit.baseline.iter().all(|value| (*value - 2.5).abs() < 1e-6));
    }
}

#[test]
fn two_d_spline_into_reuses_workspace_and_output_buffer() {
    let rows = 5;
    let cols = 6;
    let data = (0..rows)
        .flat_map(|row| (0..cols).map(move |col| 1.0 + row as f64 * 0.1 + col as f64 * 0.05))
        .collect::<Vec<_>>();
    let input = MatrixView::row_major(&data, rows, cols).unwrap();
    let mut output = vec![0.0; data.len()];
    let output_view = MatrixViewMut::row_major(&mut output, rows, cols).unwrap();
    let mut workspace = Spline2DWorkspace::new(0, 0);

    let report = pspline_asls_into(
        input,
        PsplineAsls2DParams::default(),
        output_view,
        &mut workspace,
    )
    .unwrap();

    assert!(report.iterations > 0);
    assert!(output.iter().all(|value| value.is_finite()));

    let mut second_output = vec![0.0; data.len()];
    let second_output_view = MatrixViewMut::row_major(&mut second_output, rows, cols).unwrap();
    let second_report = pspline_asls_into(
        input,
        PsplineAsls2DParams::default(),
        second_output_view,
        &mut workspace,
    )
    .unwrap();

    assert!(second_report.iterations > 0);
    assert_eq!(output, second_output);
}

#[test]
fn two_d_spline_rejects_invalid_parameters_and_shapes() {
    let data = vec![1.0; 30];
    let input = MatrixView::row_major(&data, 5, 6).unwrap();

    let error = pspline_asls(
        input,
        PsplineAsls2DParams {
            p: 0.0,
            ..PsplineAsls2DParams::default()
        },
    )
    .unwrap_err();
    assert!(matches!(error, BaselineError::InvalidParameter { .. }));

    let error = pspline_arpls(
        input,
        PsplineArPls2DParams {
            spline: Spline2DParams {
                lambda: 0.0,
                ..Spline2DParams::default()
            },
        },
    )
    .unwrap_err();
    assert!(matches!(error, BaselineError::InvalidParameter { .. }));

    let error = irsqr(
        input,
        Irsqr2DParams {
            quantile: 1.0,
            ..Irsqr2DParams::default()
        },
    )
    .unwrap_err();
    assert!(matches!(error, BaselineError::InvalidParameter { .. }));

    let small = vec![1.0; 12];
    let small_input = MatrixView::row_major(&small, 3, 4).unwrap();
    let error = pspline_asls(small_input, PsplineAsls2DParams::default()).unwrap_err();
    assert!(matches!(error, BaselineError::TooShort { .. }));

    let mut output = vec![0.0; data.len()];
    let output = MatrixViewMut::row_major(&mut output, 6, 5).unwrap();
    let mut workspace = Spline2DWorkspace::new(5, 6);
    let error = pspline_asls_into(
        input,
        PsplineAsls2DParams::default(),
        output,
        &mut workspace,
    )
    .unwrap_err();
    assert!(matches!(error, BaselineError::LengthMismatch { .. }));
}
