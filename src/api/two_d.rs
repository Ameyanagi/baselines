use crate::data::MatrixView;
use crate::two_d::{morphology, optimizers, polynomial, spline, whittaker};
use crate::{Fit2D, Result};

macro_rules! builder_2d {
    ($builder:ident, $params:ty, $fit:path) => {
        #[derive(Debug, Clone, Copy)]
        #[must_use]
        pub struct $builder<'a> {
            input: MatrixView<'a>,
            params: $params,
        }

        impl<'a> $builder<'a> {
            pub(crate) fn new(input: MatrixView<'a>) -> Self {
                Self {
                    input,
                    params: <$params>::default(),
                }
            }

            #[must_use]
            pub fn with_params(mut self, params: $params) -> Self {
                self.params = params;
                self
            }

            #[must_use]
            pub fn params(&self) -> &$params {
                &self.params
            }

            pub fn fit(self) -> Result<Fit2D> {
                $fit(self.input, self.params)
            }
        }
    };
}

macro_rules! setter {
    ($name:ident, $field:ident, $ty:ty) => {
        #[must_use]
        pub fn $name(mut self, value: $ty) -> Self {
            self.params.$field = value;
            self
        }
    };
}

macro_rules! option_setter {
    ($name:ident, $clear:ident, $field:ident, $ty:ty) => {
        #[must_use]
        pub fn $name(mut self, value: $ty) -> Self {
            self.params.$field = Some(value);
            self
        }

        #[must_use]
        pub fn $clear(mut self) -> Self {
            self.params.$field = None;
            self
        }
    };
}

macro_rules! whittaker_2d_setters {
    () => {
        #[must_use]
        pub fn lambda(mut self, value: f64) -> Self {
            self.params.whittaker.lambda = value;
            self
        }

        #[must_use]
        pub fn lambda_rows(mut self, value: f64) -> Self {
            self.params.whittaker.lambda_rows = Some(value);
            self
        }

        #[must_use]
        pub fn shared_lambda_rows(mut self) -> Self {
            self.params.whittaker.lambda_rows = None;
            self
        }

        #[must_use]
        pub fn lambda_cols(mut self, value: f64) -> Self {
            self.params.whittaker.lambda_cols = Some(value);
            self
        }

        #[must_use]
        pub fn shared_lambda_cols(mut self) -> Self {
            self.params.whittaker.lambda_cols = None;
            self
        }

        #[must_use]
        pub fn max_iter(mut self, value: usize) -> Self {
            self.params.whittaker.max_iter = value;
            self
        }

        #[must_use]
        pub fn tol(mut self, value: f64) -> Self {
            self.params.whittaker.tol = value;
            self
        }

        #[must_use]
        pub fn cg_max_iter(mut self, value: usize) -> Self {
            self.params.whittaker.cg_max_iter = value;
            self
        }

        #[must_use]
        pub fn cg_tol(mut self, value: f64) -> Self {
            self.params.whittaker.cg_tol = value;
            self
        }
    };
}

macro_rules! spline_2d_setters {
    () => {
        #[must_use]
        pub fn lambda(mut self, value: f64) -> Self {
            self.params.spline.lambda = value;
            self
        }

        #[must_use]
        pub fn max_iter(mut self, value: usize) -> Self {
            self.params.spline.max_iter = value;
            self
        }

        #[must_use]
        pub fn tol(mut self, value: f64) -> Self {
            self.params.spline.tol = value;
            self
        }

        #[must_use]
        pub fn num_knots_rows(mut self, value: usize) -> Self {
            self.params.spline.num_knots_rows = value;
            self
        }

        #[must_use]
        pub fn num_knots_cols(mut self, value: usize) -> Self {
            self.params.spline.num_knots_cols = value;
            self
        }
    };
}

/// Ergonomic entrypoint for row-major two-dimensional baseline correction.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct Baseline2D<'a> {
    input: MatrixView<'a>,
}

impl<'a> Baseline2D<'a> {
    /// Creates a row-major two-dimensional entrypoint.
    pub fn row_major(data: &'a [f64], rows: usize, cols: usize) -> Result<Self> {
        Ok(Self {
            input: MatrixView::row_major(data, rows, cols)?,
        })
    }

    /// Creates an entrypoint from an existing matrix view.
    pub fn from_view(input: MatrixView<'a>) -> Self {
        Self { input }
    }

    /// Returns the input view.
    #[must_use]
    pub fn as_view(&self) -> MatrixView<'a> {
        self.input
    }

    pub fn asls(self) -> Asls2DBuilder<'a> {
        Asls2DBuilder::new(self.input)
    }

    pub fn iasls(self) -> Iasls2DBuilder<'a> {
        Iasls2DBuilder::new(self.input)
    }

    pub fn airpls(self) -> AirPls2DBuilder<'a> {
        AirPls2DBuilder::new(self.input)
    }

    pub fn arpls(self) -> ArPls2DBuilder<'a> {
        ArPls2DBuilder::new(self.input)
    }

    pub fn drpls(self) -> DrPls2DBuilder<'a> {
        DrPls2DBuilder::new(self.input)
    }

    pub fn iarpls(self) -> IarPls2DBuilder<'a> {
        IarPls2DBuilder::new(self.input)
    }

    pub fn aspls(self) -> AsPls2DBuilder<'a> {
        AsPls2DBuilder::new(self.input)
    }

    pub fn psalsa(self) -> Psalsa2DBuilder<'a> {
        Psalsa2DBuilder::new(self.input)
    }

    pub fn brpls(self) -> BrPls2DBuilder<'a> {
        BrPls2DBuilder::new(self.input)
    }

    pub fn lsrpls(self) -> LsrPls2DBuilder<'a> {
        LsrPls2DBuilder::new(self.input)
    }

    pub fn arpls_eigen(self) -> ArPls2DEigenBuilder<'a> {
        ArPls2DEigenBuilder::new(self.input)
    }

    pub fn poly(self) -> Poly2DBuilder<'a> {
        Poly2DBuilder::new(self.input)
    }

    pub fn modpoly(self) -> ModPoly2DBuilder<'a> {
        ModPoly2DBuilder::new(self.input)
    }

    pub fn imodpoly(self) -> ImodPoly2DBuilder<'a> {
        ImodPoly2DBuilder::new(self.input)
    }

    pub fn penalized_poly(self) -> PenalizedPoly2DBuilder<'a> {
        PenalizedPoly2DBuilder::new(self.input)
    }

    pub fn quant_reg(self) -> QuantReg2DBuilder<'a> {
        QuantReg2DBuilder::new(self.input)
    }

    pub fn rolling_ball(self) -> RollingBall2DBuilder<'a> {
        RollingBall2DBuilder::new(self.input)
    }

    pub fn tophat(self) -> Tophat2DBuilder<'a> {
        Tophat2DBuilder::new(self.input)
    }

    pub fn mor(self) -> Mor2DBuilder<'a> {
        Mor2DBuilder::new(self.input)
    }

    pub fn imor(self) -> Imor2DBuilder<'a> {
        Imor2DBuilder::new(self.input)
    }

    pub fn noise_median(self) -> NoiseMedian2DBuilder<'a> {
        NoiseMedian2DBuilder::new(self.input)
    }

    pub fn pspline_asls(self) -> PsplineAsls2DBuilder<'a> {
        PsplineAsls2DBuilder::new(self.input)
    }

    pub fn pspline_iasls(self) -> PsplineIasls2DBuilder<'a> {
        PsplineIasls2DBuilder::new(self.input)
    }

    pub fn pspline_airpls(self) -> PsplineAirPls2DBuilder<'a> {
        PsplineAirPls2DBuilder::new(self.input)
    }

    pub fn pspline_arpls(self) -> PsplineArPls2DBuilder<'a> {
        PsplineArPls2DBuilder::new(self.input)
    }

    pub fn pspline_iarpls(self) -> PsplineIarPls2DBuilder<'a> {
        PsplineIarPls2DBuilder::new(self.input)
    }

    pub fn pspline_psalsa(self) -> PsplinePsalsa2DBuilder<'a> {
        PsplinePsalsa2DBuilder::new(self.input)
    }

    pub fn pspline_brpls(self) -> PsplineBrPls2DBuilder<'a> {
        PsplineBrPls2DBuilder::new(self.input)
    }

    pub fn pspline_lsrpls(self) -> PsplineLsrPls2DBuilder<'a> {
        PsplineLsrPls2DBuilder::new(self.input)
    }

    pub fn irsqr(self) -> Irsqr2DBuilder<'a> {
        Irsqr2DBuilder::new(self.input)
    }

    pub fn mixture_model(self) -> MixtureModel2DBuilder<'a> {
        MixtureModel2DBuilder::new(self.input)
    }

    pub fn adaptive_minmax(self) -> AdaptiveMinmax2DBuilder<'a> {
        AdaptiveMinmax2DBuilder::new(self.input)
    }

    pub fn individual_axes(self) -> IndividualAxes2DBuilder<'a> {
        IndividualAxes2DBuilder::new(self.input)
    }

    pub fn collab_pls(surfaces: &'a [MatrixView<'a>]) -> CollabPls2DBuilder<'a> {
        CollabPls2DBuilder {
            surfaces,
            params: optimizers::CollabPls2DParams::default(),
        }
    }
}

builder_2d!(Asls2DBuilder, whittaker::Asls2DParams, whittaker::asls);
builder_2d!(Iasls2DBuilder, whittaker::Iasls2DParams, whittaker::iasls);
builder_2d!(
    AirPls2DBuilder,
    whittaker::AirPls2DParams,
    whittaker::airpls
);
builder_2d!(ArPls2DBuilder, whittaker::ArPls2DParams, whittaker::arpls);
builder_2d!(DrPls2DBuilder, whittaker::DrPls2DParams, whittaker::drpls);
builder_2d!(
    IarPls2DBuilder,
    whittaker::IarPls2DParams,
    whittaker::iarpls
);
builder_2d!(AsPls2DBuilder, whittaker::AsPls2DParams, whittaker::aspls);
builder_2d!(
    Psalsa2DBuilder,
    whittaker::Psalsa2DParams,
    whittaker::psalsa
);
builder_2d!(BrPls2DBuilder, whittaker::BrPls2DParams, whittaker::brpls);
builder_2d!(
    LsrPls2DBuilder,
    whittaker::LsrPls2DParams,
    whittaker::lsrpls
);

impl<'a> Asls2DBuilder<'a> {
    whittaker_2d_setters!();
    setter!(p, p, f64);
}

impl<'a> Iasls2DBuilder<'a> {
    whittaker_2d_setters!();
    setter!(p, p, f64);
    setter!(lambda_1, lambda_1, f64);
}

impl<'a> AirPls2DBuilder<'a> {
    whittaker_2d_setters!();
}

impl<'a> ArPls2DBuilder<'a> {
    whittaker_2d_setters!();
}

impl<'a> DrPls2DBuilder<'a> {
    whittaker_2d_setters!();
    setter!(eta, eta, f64);
}

impl<'a> IarPls2DBuilder<'a> {
    whittaker_2d_setters!();
}

impl<'a> AsPls2DBuilder<'a> {
    whittaker_2d_setters!();
    setter!(asymmetric_coef, asymmetric_coef, f64);
}

impl<'a> Psalsa2DBuilder<'a> {
    whittaker_2d_setters!();
    setter!(p, p, f64);
    option_setter!(k, auto_k, k, f64);
}

impl<'a> BrPls2DBuilder<'a> {
    whittaker_2d_setters!();
    setter!(max_iter_2, max_iter_2, usize);
    setter!(tol_2, tol_2, f64);
}

impl<'a> LsrPls2DBuilder<'a> {
    whittaker_2d_setters!();
}

#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct ArPls2DEigenBuilder<'a> {
    input: MatrixView<'a>,
    params: whittaker::ArPls2DEigenParams,
}

impl<'a> ArPls2DEigenBuilder<'a> {
    pub(crate) fn new(input: MatrixView<'a>) -> Self {
        Self {
            input,
            params: whittaker::ArPls2DEigenParams::default(),
        }
    }

    #[must_use]
    pub fn with_params(mut self, params: whittaker::ArPls2DEigenParams) -> Self {
        self.params = params;
        self
    }

    #[must_use]
    pub fn params(&self) -> &whittaker::ArPls2DEigenParams {
        &self.params
    }

    #[must_use]
    pub fn lambda(mut self, value: f64) -> Self {
        self.params.whittaker.lambda = value;
        self
    }

    #[must_use]
    pub fn lambda_rows(mut self, value: f64) -> Self {
        self.params.whittaker.lambda_rows = Some(value);
        self
    }

    #[must_use]
    pub fn shared_lambda_rows(mut self) -> Self {
        self.params.whittaker.lambda_rows = None;
        self
    }

    #[must_use]
    pub fn lambda_cols(mut self, value: f64) -> Self {
        self.params.whittaker.lambda_cols = Some(value);
        self
    }

    #[must_use]
    pub fn shared_lambda_cols(mut self) -> Self {
        self.params.whittaker.lambda_cols = None;
        self
    }

    #[must_use]
    pub fn diff_order(mut self, rows: usize, cols: usize) -> Self {
        self.params.whittaker.diff_order = (rows, cols);
        self
    }

    #[must_use]
    pub fn num_eigens(mut self, rows: usize, cols: usize) -> Self {
        self.params.whittaker.num_eigens = (rows, cols);
        self
    }

    #[must_use]
    pub fn return_dof(mut self, value: bool) -> Self {
        self.params.whittaker.return_dof = value;
        self
    }

    #[must_use]
    pub fn max_iter(mut self, value: usize) -> Self {
        self.params.whittaker.max_iter = value;
        self
    }

    #[must_use]
    pub fn tol(mut self, value: f64) -> Self {
        self.params.whittaker.tol = value;
        self
    }

    #[must_use]
    pub fn cg_max_iter(mut self, value: usize) -> Self {
        self.params.whittaker.cg_max_iter = value;
        self
    }

    #[must_use]
    pub fn cg_tol(mut self, value: f64) -> Self {
        self.params.whittaker.cg_tol = value;
        self
    }

    pub fn fit(self) -> Result<whittaker::Whittaker2DEigenFit> {
        whittaker::arpls_eigen(self.input, self.params)
    }
}

builder_2d!(Poly2DBuilder, polynomial::Poly2DParams, polynomial::poly);
builder_2d!(
    ModPoly2DBuilder,
    polynomial::ModPoly2DParams,
    polynomial::modpoly
);
builder_2d!(
    ImodPoly2DBuilder,
    polynomial::ImodPoly2DParams,
    polynomial::imodpoly
);
builder_2d!(
    PenalizedPoly2DBuilder,
    polynomial::PenalizedPoly2DParams,
    polynomial::penalized_poly
);
builder_2d!(
    QuantReg2DBuilder,
    polynomial::QuantReg2DParams,
    polynomial::quant_reg
);

impl<'a> Poly2DBuilder<'a> {
    setter!(order, order, usize);
}

impl<'a> ModPoly2DBuilder<'a> {
    setter!(order, order, usize);
    setter!(max_iter, max_iter, usize);
    setter!(tol, tol, f64);
}

impl<'a> ImodPoly2DBuilder<'a> {
    setter!(order, order, usize);
    setter!(max_iter, max_iter, usize);
    setter!(tol, tol, f64);
}

impl<'a> PenalizedPoly2DBuilder<'a> {
    setter!(order, order, usize);
    setter!(max_iter, max_iter, usize);
    setter!(tol, tol, f64);
    option_setter!(threshold, auto_threshold, threshold, f64);
    setter!(alpha_factor, alpha_factor, f64);
}

impl<'a> QuantReg2DBuilder<'a> {
    setter!(order, order, usize);
    setter!(quantile, quantile, f64);
    setter!(max_iter, max_iter, usize);
    setter!(tol, tol, f64);
    option_setter!(epsilon, auto_epsilon, epsilon, f64);
}

builder_2d!(
    RollingBall2DBuilder,
    morphology::Morphology2DParams,
    morphology::rolling_ball
);
builder_2d!(
    Tophat2DBuilder,
    morphology::Morphology2DParams,
    morphology::tophat
);
builder_2d!(
    Mor2DBuilder,
    morphology::Morphology2DParams,
    morphology::mor
);
builder_2d!(Imor2DBuilder, morphology::Imor2DParams, morphology::imor);
builder_2d!(
    NoiseMedian2DBuilder,
    morphology::Morphology2DParams,
    morphology::noise_median
);

macro_rules! morphology_2d_window {
    ($builder:ident) => {
        impl<'a> $builder<'a> {
            setter!(window_rows, window_rows, usize);
            setter!(window_cols, window_cols, usize);
        }
    };
}

morphology_2d_window!(RollingBall2DBuilder);
morphology_2d_window!(Tophat2DBuilder);
morphology_2d_window!(Mor2DBuilder);
morphology_2d_window!(NoiseMedian2DBuilder);

impl<'a> Imor2DBuilder<'a> {
    #[must_use]
    pub fn window_rows(mut self, value: usize) -> Self {
        self.params.morphology.window_rows = value;
        self
    }

    #[must_use]
    pub fn window_cols(mut self, value: usize) -> Self {
        self.params.morphology.window_cols = value;
        self
    }

    setter!(max_iter, max_iter, usize);
    setter!(tol, tol, f64);
}

builder_2d!(
    PsplineAsls2DBuilder,
    spline::PsplineAsls2DParams,
    spline::pspline_asls
);
builder_2d!(
    PsplineIasls2DBuilder,
    spline::PsplineIasls2DParams,
    spline::pspline_iasls
);
builder_2d!(
    PsplineAirPls2DBuilder,
    spline::PsplineAirPls2DParams,
    spline::pspline_airpls
);
builder_2d!(
    PsplineArPls2DBuilder,
    spline::PsplineArPls2DParams,
    spline::pspline_arpls
);
builder_2d!(
    PsplineIarPls2DBuilder,
    spline::PsplineIarPls2DParams,
    spline::pspline_iarpls
);
builder_2d!(
    PsplinePsalsa2DBuilder,
    spline::PsplinePsalsa2DParams,
    spline::pspline_psalsa
);
builder_2d!(
    PsplineBrPls2DBuilder,
    spline::PsplineBrPls2DParams,
    spline::pspline_brpls
);
builder_2d!(
    PsplineLsrPls2DBuilder,
    spline::PsplineLsrPls2DParams,
    spline::pspline_lsrpls
);
builder_2d!(Irsqr2DBuilder, spline::Irsqr2DParams, spline::irsqr);
builder_2d!(
    MixtureModel2DBuilder,
    spline::MixtureModel2DParams,
    spline::mixture_model
);

impl<'a> PsplineAsls2DBuilder<'a> {
    spline_2d_setters!();
    setter!(p, p, f64);
}

impl<'a> PsplineIasls2DBuilder<'a> {
    spline_2d_setters!();
    setter!(p, p, f64);
    setter!(lambda_1, lambda_1, f64);
}

impl<'a> PsplineAirPls2DBuilder<'a> {
    spline_2d_setters!();
}

impl<'a> PsplineArPls2DBuilder<'a> {
    spline_2d_setters!();
}

impl<'a> PsplineIarPls2DBuilder<'a> {
    spline_2d_setters!();
}

impl<'a> PsplinePsalsa2DBuilder<'a> {
    spline_2d_setters!();
    setter!(p, p, f64);
    option_setter!(k, auto_k, k, f64);
}

impl<'a> PsplineBrPls2DBuilder<'a> {
    spline_2d_setters!();
    setter!(max_iter_2, max_iter_2, usize);
    setter!(tol_2, tol_2, f64);
}

impl<'a> PsplineLsrPls2DBuilder<'a> {
    spline_2d_setters!();
}

impl<'a> Irsqr2DBuilder<'a> {
    spline_2d_setters!();
    setter!(quantile, quantile, f64);
    option_setter!(epsilon, auto_epsilon, epsilon, f64);
}

impl<'a> MixtureModel2DBuilder<'a> {
    spline_2d_setters!();
    setter!(p, p, f64);
}

builder_2d!(
    AdaptiveMinmax2DBuilder,
    optimizers::AdaptiveMinmax2DParams,
    optimizers::adaptive_minmax
);
builder_2d!(
    IndividualAxes2DBuilder,
    optimizers::IndividualAxes2DParams,
    optimizers::individual_axes
);

impl<'a> AdaptiveMinmax2DBuilder<'a> {
    setter!(order, order, usize);
    setter!(max_iter, max_iter, usize);
    setter!(tol, tol, f64);
}

impl<'a> IndividualAxes2DBuilder<'a> {
    #[must_use]
    pub fn asls_params(mut self, params: crate::whittaker::AslsParams) -> Self {
        self.params.asls = params;
        self
    }

    #[must_use]
    pub fn lambda(mut self, value: f64) -> Self {
        self.params.asls.whittaker.lambda = value;
        self
    }

    #[must_use]
    pub fn max_iter(mut self, value: usize) -> Self {
        self.params.asls.whittaker.max_iter = value;
        self
    }

    #[must_use]
    pub fn tol(mut self, value: f64) -> Self {
        self.params.asls.whittaker.tol = value;
        self
    }

    #[must_use]
    pub fn p(mut self, value: f64) -> Self {
        self.params.asls.p = value;
        self
    }
}

#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct CollabPls2DBuilder<'a> {
    surfaces: &'a [MatrixView<'a>],
    params: optimizers::CollabPls2DParams,
}

impl CollabPls2DBuilder<'_> {
    #[must_use]
    pub fn with_params(mut self, params: optimizers::CollabPls2DParams) -> Self {
        self.params = params;
        self
    }

    #[must_use]
    pub fn params(&self) -> &optimizers::CollabPls2DParams {
        &self.params
    }

    #[must_use]
    pub fn asls_params(mut self, params: whittaker::Asls2DParams) -> Self {
        self.params.asls = params;
        self
    }

    #[must_use]
    pub fn lambda(mut self, value: f64) -> Self {
        self.params.asls.whittaker.lambda = value;
        self
    }

    #[must_use]
    pub fn max_iter(mut self, value: usize) -> Self {
        self.params.asls.whittaker.max_iter = value;
        self
    }

    #[must_use]
    pub fn tol(mut self, value: f64) -> Self {
        self.params.asls.whittaker.tol = value;
        self
    }

    #[must_use]
    pub fn p(mut self, value: f64) -> Self {
        self.params.asls.p = value;
        self
    }

    pub fn fit(self) -> Result<Vec<Fit2D>> {
        optimizers::collab_pls(self.surfaces, self.params)
    }
}
