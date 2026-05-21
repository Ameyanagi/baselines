use baselines::Fit;
use baselines::classification::{
    ClassificationParams, cwt_br, dietrich, fabc, fastchrom, golotvin, rubberband, std_distribution,
};
use baselines::misc::{BeadsParams, beads, interp_pts};
use baselines::morphology::{
    MorphologyParams, amormol, imor, jbcd, mor, mormol, mpls, mpspline, mwmv, rolling_ball, tophat,
};
use baselines::optimizers::{
    LambdaSearchParams, adaptive_minmax, collab_pls, custom_bc, optimize_extended_range,
};
use baselines::polynomial::{
    GoldindecParams, ImodPolyParams, LoessParams, ModPolyParams, PenalizedPolyParams, PolyParams,
    QuantRegParams, goldindec, imodpoly, loess, modpoly, penalized_poly, poly, quant_reg,
};
use baselines::smoothing::{SmoothingParams, ipsa, noise_median, peak_filling, ria, snip, swima};
use baselines::spline::{
    corner_cutting, irsqr, mixture_model, pspline_airpls, pspline_arpls, pspline_asls,
    pspline_aspls, pspline_brpls, pspline_derpsalsa, pspline_drpls, pspline_iarpls, pspline_iasls,
    pspline_lsrpls, pspline_mpls, pspline_psalsa,
};
use baselines::whittaker::{
    AirPlsParams, ArPlsParams, AsPlsParams, AslsParams, BrPlsParams, DerPsalsaParams, DrPlsParams,
    IarPlsParams, IaslsParams, LsrPlsParams, PsalsaParams, airpls, arpls, asls, aspls, brpls,
    derpsalsa, drpls, iarpls, iasls, lsrpls, psalsa,
};

#[test]
fn exposed_1d_algorithms_return_finite_baselines() {
    let y = signal();
    let morph = MorphologyParams { window_size: 11 };
    let smooth = SmoothingParams {
        window_size: 11,
        max_iter: 4,
    };
    let search = LambdaSearchParams {
        start_exp: 2.0,
        stop_exp: 4.0,
        steps: 3,
    };

    let mut fits: Vec<Fit> = vec![
        poly(&y, PolyParams { order: 2 }).unwrap(),
        modpoly(&y, ModPolyParams::default()).unwrap(),
        imodpoly(&y, ImodPolyParams::default()).unwrap(),
        penalized_poly(&y, PenalizedPolyParams::default()).unwrap(),
        loess(&y, LoessParams { window_size: 11 }).unwrap(),
        quant_reg(&y, QuantRegParams::default()).unwrap(),
        goldindec(&y, GoldindecParams::default()).unwrap(),
        asls(&y, AslsParams::default()).unwrap(),
        iasls(&y, IaslsParams::default()).unwrap(),
        airpls(&y, AirPlsParams::default()).unwrap(),
        arpls(&y, ArPlsParams::default()).unwrap(),
        drpls(&y, DrPlsParams::default()).unwrap(),
        iarpls(&y, IarPlsParams::default()).unwrap(),
        aspls(&y, AsPlsParams::default()).unwrap(),
        psalsa(&y, PsalsaParams::default()).unwrap(),
        derpsalsa(&y, DerPsalsaParams::default()).unwrap(),
        brpls(&y, BrPlsParams::default()).unwrap(),
        lsrpls(&y, LsrPlsParams::default()).unwrap(),
        mpls(&y, morph).unwrap(),
        mor(&y, morph).unwrap(),
        imor(&y, morph).unwrap(),
        mormol(&y, morph).unwrap(),
        amormol(&y, morph).unwrap(),
        rolling_ball(&y, morph).unwrap(),
        mwmv(&y, morph).unwrap(),
        tophat(&y, morph).unwrap(),
        mpspline(&y, morph).unwrap(),
        jbcd(&y, morph).unwrap(),
        mixture_model(&y, ArPlsParams::default()).unwrap(),
        irsqr(&y, AslsParams::default()).unwrap(),
        corner_cutting(&y, smooth).unwrap(),
        pspline_asls(&y, AslsParams::default()).unwrap(),
        pspline_iasls(&y, IaslsParams::default()).unwrap(),
        pspline_airpls(&y, AirPlsParams::default()).unwrap(),
        pspline_arpls(&y, ArPlsParams::default()).unwrap(),
        pspline_drpls(&y, DrPlsParams::default()).unwrap(),
        pspline_iarpls(&y, IarPlsParams::default()).unwrap(),
        pspline_aspls(&y, AsPlsParams::default()).unwrap(),
        pspline_psalsa(&y, PsalsaParams::default()).unwrap(),
        pspline_derpsalsa(&y, DerPsalsaParams::default()).unwrap(),
        pspline_mpls(&y, morph).unwrap(),
        pspline_brpls(&y, BrPlsParams::default()).unwrap(),
        pspline_lsrpls(&y, LsrPlsParams::default()).unwrap(),
        noise_median(&y, smooth).unwrap(),
        snip(&y, baselines::SnipParams { max_half_window: 8 }).unwrap(),
        swima(&y, smooth).unwrap(),
        ipsa(&y, smooth).unwrap(),
        ria(&y, smooth).unwrap(),
        peak_filling(&y, smooth).unwrap(),
        dietrich(&y, ClassificationParams { window_size: 11 }).unwrap(),
        golotvin(&y, ClassificationParams { window_size: 11 }).unwrap(),
        std_distribution(&y, ClassificationParams { window_size: 11 }).unwrap(),
        fastchrom(&y, ClassificationParams { window_size: 11 }).unwrap(),
        cwt_br(&y, ClassificationParams { window_size: 11 }).unwrap(),
        fabc(&y, ClassificationParams { window_size: 11 }).unwrap(),
        rubberband(&y).unwrap(),
        optimize_extended_range(&y, search).unwrap(),
        adaptive_minmax(&y, MorphologyParams { window_size: 7 }, morph).unwrap(),
        custom_bc(&y, |values| asls(values, AslsParams::default())).unwrap(),
        interp_pts(
            &y,
            &[
                (0, y[0]),
                (y.len() / 2, y[y.len() / 2]),
                (y.len() - 1, y[y.len() - 1]),
            ],
        )
        .unwrap(),
        beads(&y, BeadsParams::default()).unwrap(),
    ];

    fits.extend(collab_pls(&[y.clone(), y.clone()], AslsParams::default()).unwrap());

    for fit in fits {
        assert_eq!(fit.baseline.len(), y.len());
        assert!(fit.baseline.iter().all(|value| value.is_finite()));
    }
}

fn signal() -> Vec<f64> {
    (0..64)
        .map(|i| {
            let x = i as f64 / 63.0;
            0.8 + 0.2 * x + (-(x - 0.35).powi(2) / 0.002).exp()
        })
        .collect()
}
