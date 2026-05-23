use baselines::two_d::polynomial::{
    ImodPoly2DParams, ModPoly2DParams, PenalizedPoly2DParams, Poly2DParams, QuantReg2DParams,
    imodpoly, modpoly, penalized_poly, poly, poly_into, quant_reg,
};
use baselines::{BaselineError, MatrixView, MatrixViewMut};

#[test]
fn two_d_polynomial_recovers_planar_surface() {
    let rows = 5;
    let cols = 6;
    let data = (0..rows)
        .flat_map(|row| {
            (0..cols).map(move |col| {
                let x = col as f64;
                let y = row as f64;
                1.5 + 0.2 * x - 0.35 * y
            })
        })
        .collect::<Vec<_>>();
    let input = MatrixView::row_major(&data, rows, cols).unwrap();

    let fit = poly(input, Poly2DParams { order: 1 }).unwrap();

    assert_eq!(fit.shape(), (rows, cols));
    for (actual, expected) in fit.baseline.iter().zip(data) {
        assert!((actual - expected).abs() < 1e-10);
    }
}

#[test]
fn two_d_polynomial_methods_preserve_constant_surfaces() {
    let data = vec![2.5; 42];
    let input = MatrixView::row_major(&data, 6, 7).unwrap();

    for fit in [
        poly(input, Poly2DParams::default()).unwrap(),
        modpoly(input, ModPoly2DParams::default()).unwrap(),
        imodpoly(input, ImodPoly2DParams::default()).unwrap(),
        penalized_poly(input, PenalizedPoly2DParams::default()).unwrap(),
        quant_reg(input, QuantReg2DParams::default()).unwrap(),
    ] {
        assert!(fit.baseline.iter().all(|value| (*value - 2.5).abs() < 1e-8));
    }
}

#[test]
fn two_d_polynomial_into_reuses_output_buffer() {
    let data = vec![
        1.0, 1.1, 1.2, 1.3, //
        1.2, 1.3, 1.4, 1.5, //
        1.4, 1.5, 1.6, 1.7,
    ];
    let input = MatrixView::row_major(&data, 3, 4).unwrap();
    let mut output = vec![0.0; data.len()];
    let output_view = MatrixViewMut::row_major(&mut output, 3, 4).unwrap();

    let report = poly_into(input, Poly2DParams { order: 1 }, output_view).unwrap();

    assert!(report.converged);
    assert!(output.iter().all(|value| value.is_finite()));
}

#[test]
fn two_d_polynomial_rejects_invalid_parameters_and_shapes() {
    let data = vec![1.0; 6];
    let input = MatrixView::row_major(&data, 2, 3).unwrap();

    let error = poly(input, Poly2DParams { order: 3 }).unwrap_err();
    assert!(matches!(error, BaselineError::TooShort { .. }));

    let error = modpoly(
        input,
        ModPoly2DParams {
            max_iter: 0,
            ..ModPoly2DParams::default()
        },
    )
    .unwrap_err();
    assert!(matches!(error, BaselineError::InvalidParameter { .. }));

    let error = quant_reg(
        input,
        QuantReg2DParams {
            quantile: 1.0,
            ..QuantReg2DParams::default()
        },
    )
    .unwrap_err();
    assert!(matches!(error, BaselineError::InvalidParameter { .. }));

    let mut output = vec![0.0; 6];
    let output = MatrixViewMut::row_major(&mut output, 3, 2).unwrap();
    let error = poly_into(input, Poly2DParams::default(), output).unwrap_err();
    assert!(matches!(error, BaselineError::LengthMismatch { .. }));
}
