use baselines::two_d::whittaker::{
    AirPls2DParams, ArPls2DParams, AsPls2DParams, Asls2DParams, BrPls2DParams, DrPls2DParams,
    IarPls2DParams, Iasls2DParams, LsrPls2DParams, Psalsa2DParams, Whittaker2DParams,
    Whittaker2DWorkspace, airpls, arpls, asls, asls_into, aspls, brpls, drpls, iarpls, iasls,
    lsrpls, psalsa,
};
use baselines::{BaselineError, MatrixView, MatrixViewMut};

#[test]
fn two_d_whittaker_methods_preserve_constant_surfaces() {
    let data = vec![2.5; 30];
    let input = MatrixView::row_major(&data, 5, 6).unwrap();

    for fit in [
        asls(input, Asls2DParams::default()).unwrap(),
        iasls(input, Iasls2DParams::default()).unwrap(),
        airpls(input, AirPls2DParams::default()).unwrap(),
        arpls(input, ArPls2DParams::default()).unwrap(),
        drpls(input, DrPls2DParams::default()).unwrap(),
        iarpls(input, IarPls2DParams::default()).unwrap(),
        aspls(input, AsPls2DParams::default()).unwrap(),
        psalsa(
            input,
            Psalsa2DParams {
                k: Some(1.0),
                ..Psalsa2DParams::default()
            },
        )
        .unwrap(),
        brpls(input, BrPls2DParams::default()).unwrap(),
        lsrpls(input, LsrPls2DParams::default()).unwrap(),
    ] {
        assert_eq!(fit.shape(), (5, 6));
        assert!(fit.baseline.iter().all(|value| (*value - 2.5).abs() < 1e-6));
    }
}

#[test]
fn two_d_whittaker_into_reuses_workspace_and_output_buffer() {
    let rows = 4;
    let cols = 5;
    let data = (0..rows)
        .flat_map(|row| (0..cols).map(move |col| 1.0 + row as f64 * 0.1 + col as f64 * 0.05))
        .collect::<Vec<_>>();
    let input = MatrixView::row_major(&data, rows, cols).unwrap();
    let mut output = vec![0.0; data.len()];
    let output_view = MatrixViewMut::row_major(&mut output, rows, cols).unwrap();
    let mut workspace = Whittaker2DWorkspace::new(0);

    let report = asls_into(input, Asls2DParams::default(), output_view, &mut workspace).unwrap();

    assert!(report.iterations > 0);
    assert!(output.iter().all(|value| value.is_finite()));

    let mut second_output = vec![0.0; data.len()];
    let second_output_view = MatrixViewMut::row_major(&mut second_output, rows, cols).unwrap();
    let second_report = asls_into(
        input,
        Asls2DParams::default(),
        second_output_view,
        &mut workspace,
    )
    .unwrap();

    assert!(second_report.iterations > 0);
    assert_eq!(output, second_output);
}

#[test]
fn two_d_whittaker_rejects_invalid_parameters_and_shapes() {
    let data = vec![1.0; 12];
    let input = MatrixView::row_major(&data, 3, 4).unwrap();

    let error = asls(
        input,
        Asls2DParams {
            p: 0.0,
            ..Asls2DParams::default()
        },
    )
    .unwrap_err();
    assert!(matches!(error, BaselineError::InvalidParameter { .. }));

    let error = arpls(
        input,
        ArPls2DParams {
            whittaker: Whittaker2DParams {
                lambda: 0.0,
                ..Whittaker2DParams::default()
            },
        },
    )
    .unwrap_err();
    assert!(matches!(error, BaselineError::InvalidParameter { .. }));

    let error = airpls(
        input,
        AirPls2DParams {
            whittaker: Whittaker2DParams {
                cg_max_iter: 0,
                ..Whittaker2DParams::default()
            },
        },
    )
    .unwrap_err();
    assert!(matches!(error, BaselineError::InvalidParameter { .. }));

    let error = psalsa(
        input,
        Psalsa2DParams {
            k: Some(0.0),
            ..Psalsa2DParams::default()
        },
    )
    .unwrap_err();
    assert!(matches!(error, BaselineError::InvalidParameter { .. }));

    let small = vec![1.0; 6];
    let small_input = MatrixView::row_major(&small, 2, 3).unwrap();
    let error = asls(small_input, Asls2DParams::default()).unwrap_err();
    assert!(matches!(error, BaselineError::TooShort { .. }));

    let mut output = vec![0.0; data.len()];
    let output = MatrixViewMut::row_major(&mut output, 4, 3).unwrap();
    let mut workspace = Whittaker2DWorkspace::new(data.len());
    let error = asls_into(input, Asls2DParams::default(), output, &mut workspace).unwrap_err();
    assert!(matches!(error, BaselineError::LengthMismatch { .. }));
}
