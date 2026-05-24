use crate::classification::{self, ClassificationFit};
use crate::misc::{self, BeadsCostFunction};
use crate::morphology;
use crate::optimizers;
use crate::polynomial::{self, PenalizedCost};
use crate::smoothing;
use crate::spline;
use crate::whittaker::xy::XyMaskSpec;
use crate::whittaker::{self, WhittakerWorkspace};
use crate::{Fit, FitHistory, FitReport, Result};

macro_rules! builder_1d {
    ($builder:ident, $params:ty, $fit:path) => {
        #[derive(Debug, Clone)]
        #[must_use]
        pub struct $builder<'a> {
            y: &'a [f64],
            params: $params,
        }

        impl<'a> $builder<'a> {
            pub(crate) fn new(y: &'a [f64]) -> Self {
                Self {
                    y,
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

            pub fn fit(self) -> Result<Fit> {
                $fit(self.y, self.params)
            }
        }
    };
}

macro_rules! builder_xy_whittaker {
    ($builder:ident, $params:ty, $fit:path) => {
        #[derive(Debug, Clone)]
        #[must_use]
        pub struct $builder<'a> {
            x: &'a [f64],
            y: &'a [f64],
            params: $params,
            masks: XyMaskSpec<'a>,
        }

        impl<'a> $builder<'a> {
            pub(crate) fn new(x: &'a [f64], y: &'a [f64]) -> Self {
                Self {
                    x,
                    y,
                    params: <$params>::default(),
                    masks: XyMaskSpec::default(),
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

            #[must_use]
            pub fn exclude_range(mut self, start: f64, end: f64) -> Self {
                self.masks.exclude_range(start, end);
                self
            }

            #[must_use]
            pub fn exclude_ranges<I>(mut self, ranges: I) -> Self
            where
                I: IntoIterator<Item = (f64, f64)>,
            {
                self.masks.exclude_ranges(ranges);
                self
            }

            pub fn exclude_mask(mut self, mask: &'a [bool]) -> Result<Self> {
                self.masks.exclude_mask(mask, self.y.len())?;
                Ok(self)
            }

            pub fn baseline_mask(mut self, mask: &'a [bool]) -> Result<Self> {
                self.masks.baseline_mask(mask, self.y.len())?;
                Ok(self)
            }

            #[must_use]
            pub fn clear_masks(mut self) -> Self {
                self.masks.clear();
                self
            }

            pub fn fit(self) -> Result<Fit> {
                $fit(self.x, self.y, self.params, &self.masks)
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

macro_rules! whittaker_setters {
    () => {
        #[must_use]
        pub fn lambda(mut self, value: f64) -> Self {
            self.params.whittaker.lambda = value;
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
    };
}

/// Ergonomic entrypoint for one-dimensional baseline correction.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct Baseline<'a> {
    y: &'a [f64],
}

impl<'a> Baseline<'a> {
    /// Creates a method-chain entrypoint for a one-dimensional signal.
    pub fn new(y: &'a [f64]) -> Self {
        Self { y }
    }

    /// Creates a method-chain entrypoint for a one-dimensional signal with explicit x values.
    pub fn new_xy(x: &'a [f64], y: &'a [f64]) -> Result<BaselineXY<'a>> {
        BaselineXY::new(x, y)
    }

    /// Returns the input signal.
    #[must_use]
    pub fn as_slice(&self) -> &'a [f64] {
        self.y
    }

    pub fn asls(self) -> AslsBuilder<'a> {
        AslsBuilder::new(self.y)
    }

    pub fn airpls(self) -> AirPlsBuilder<'a> {
        AirPlsBuilder::new(self.y)
    }

    pub fn arpls(self) -> ArPlsBuilder<'a> {
        ArPlsBuilder::new(self.y)
    }

    pub fn iasls(self) -> IaslsBuilder<'a> {
        IaslsBuilder::new(self.y)
    }

    pub fn drpls(self) -> DrPlsBuilder<'a> {
        DrPlsBuilder::new(self.y)
    }

    pub fn iarpls(self) -> IarPlsBuilder<'a> {
        IarPlsBuilder::new(self.y)
    }

    pub fn aspls(self) -> AsPlsBuilder<'a> {
        AsPlsBuilder::new(self.y)
    }

    pub fn psalsa(self) -> PsalsaBuilder<'a> {
        PsalsaBuilder::new(self.y)
    }

    pub fn derpsalsa(self) -> DerPsalsaBuilder<'a> {
        DerPsalsaBuilder::new(self.y)
    }

    pub fn brpls(self) -> BrPlsBuilder<'a> {
        BrPlsBuilder::new(self.y)
    }

    pub fn lsrpls(self) -> LsrPlsBuilder<'a> {
        LsrPlsBuilder::new(self.y)
    }

    pub fn poly(self) -> PolyBuilder<'a> {
        PolyBuilder::new(self.y)
    }

    pub fn modpoly(self) -> ModPolyBuilder<'a> {
        ModPolyBuilder::new(self.y)
    }

    pub fn imodpoly(self) -> ImodPolyBuilder<'a> {
        ImodPolyBuilder::new(self.y)
    }

    pub fn penalized_poly(self) -> PenalizedPolyBuilder<'a> {
        PenalizedPolyBuilder::new(self.y)
    }

    pub fn loess(self) -> LoessBuilder<'a> {
        LoessBuilder::new(self.y)
    }

    pub fn quant_reg(self) -> QuantRegBuilder<'a> {
        QuantRegBuilder::new(self.y)
    }

    pub fn goldindec(self) -> GoldindecBuilder<'a> {
        GoldindecBuilder::new(self.y)
    }

    pub fn rolling_ball(self) -> RollingBallBuilder<'a> {
        RollingBallBuilder::new(self.y)
    }

    pub fn tophat(self) -> TophatBuilder<'a> {
        TophatBuilder::new(self.y)
    }

    pub fn mwmv(self) -> MwmvBuilder<'a> {
        MwmvBuilder::new(self.y)
    }

    pub fn mor(self) -> MorBuilder<'a> {
        MorBuilder::new(self.y)
    }

    pub fn mpls(self) -> MplsBuilder<'a> {
        MplsBuilder::new(self.y)
    }

    pub fn imor(self) -> ImorBuilder<'a> {
        ImorBuilder::new(self.y)
    }

    pub fn mormol(self) -> MormolBuilder<'a> {
        MormolBuilder::new(self.y)
    }

    pub fn amormol(self) -> AmormolBuilder<'a> {
        AmormolBuilder::new(self.y)
    }

    pub fn mpspline(self) -> MpsplineBuilder<'a> {
        MpsplineBuilder::new(self.y)
    }

    pub fn jbcd(self) -> JbcdBuilder<'a> {
        JbcdBuilder::new(self.y)
    }

    pub fn snip(self) -> SnipBuilder<'a> {
        SnipBuilder::new(self.y)
    }

    pub fn noise_median(self) -> NoiseMedianBuilder<'a> {
        NoiseMedianBuilder::new(self.y)
    }

    pub fn smoothing_snip(self) -> SmoothingSnipBuilder<'a> {
        SmoothingSnipBuilder::new(self.y)
    }

    pub fn swima(self) -> SwimaBuilder<'a> {
        SwimaBuilder::new(self.y)
    }

    pub fn ipsa(self) -> IpsaBuilder<'a> {
        IpsaBuilder::new(self.y)
    }

    pub fn ria(self) -> RiaBuilder<'a> {
        RiaBuilder::new(self.y)
    }

    pub fn peak_filling(self) -> PeakFillingBuilder<'a> {
        PeakFillingBuilder::new(self.y)
    }

    pub fn irsqr(self) -> IrsqrBuilder<'a> {
        IrsqrBuilder::new(self.y)
    }

    pub fn pspline_asls(self) -> PsplineAslsBuilder<'a> {
        PsplineAslsBuilder::new(self.y)
    }

    pub fn pspline_iasls(self) -> PsplineIaslsBuilder<'a> {
        PsplineIaslsBuilder::new(self.y)
    }

    pub fn pspline_airpls(self) -> PsplineAirPlsBuilder<'a> {
        PsplineAirPlsBuilder::new(self.y)
    }

    pub fn pspline_arpls(self) -> PsplineArPlsBuilder<'a> {
        PsplineArPlsBuilder::new(self.y)
    }

    pub fn pspline_drpls(self) -> PsplineDrPlsBuilder<'a> {
        PsplineDrPlsBuilder::new(self.y)
    }

    pub fn pspline_iarpls(self) -> PsplineIarPlsBuilder<'a> {
        PsplineIarPlsBuilder::new(self.y)
    }

    pub fn pspline_aspls(self) -> PsplineAsPlsBuilder<'a> {
        PsplineAsPlsBuilder::new(self.y)
    }

    pub fn pspline_psalsa(self) -> PsplinePsalsaBuilder<'a> {
        PsplinePsalsaBuilder::new(self.y)
    }

    pub fn pspline_derpsalsa(self) -> PsplineDerPsalsaBuilder<'a> {
        PsplineDerPsalsaBuilder::new(self.y)
    }

    pub fn pspline_mpls(self) -> PsplineMplsBuilder<'a> {
        PsplineMplsBuilder::new(self.y)
    }

    pub fn pspline_brpls(self) -> PsplineBrPlsBuilder<'a> {
        PsplineBrPlsBuilder::new(self.y)
    }

    pub fn pspline_lsrpls(self) -> PsplineLsrPlsBuilder<'a> {
        PsplineLsrPlsBuilder::new(self.y)
    }

    pub fn mixture_model(self) -> MixtureModelBuilder<'a> {
        MixtureModelBuilder::new(self.y)
    }

    pub fn corner_cutting(self) -> CornerCuttingBuilder<'a> {
        CornerCuttingBuilder::new(self.y)
    }

    pub fn dietrich(self) -> DietrichBuilder<'a> {
        DietrichBuilder::new(self.y)
    }

    pub fn golotvin(self) -> GolotvinBuilder<'a> {
        GolotvinBuilder::new(self.y)
    }

    pub fn std_distribution(self) -> StdDistributionBuilder<'a> {
        StdDistributionBuilder::new(self.y)
    }

    pub fn fastchrom(self) -> FastChromBuilder<'a> {
        FastChromBuilder::new(self.y)
    }

    pub fn cwt_br(self) -> CwtBrBuilder<'a> {
        CwtBrBuilder::new(self.y)
    }

    pub fn fabc(self) -> FabcBuilder<'a> {
        FabcBuilder::new(self.y)
    }

    pub fn rubberband(self) -> RubberbandBuilder<'a> {
        RubberbandBuilder { y: self.y }
    }

    pub fn optimize_extended_range(self) -> OptimizeExtendedRangeBuilder<'a> {
        OptimizeExtendedRangeBuilder::new(self.y)
    }

    pub fn custom_bc(self) -> CustomBcBuilder<'a> {
        CustomBcBuilder::new(self.y)
    }

    pub fn adaptive_minmax(self) -> AdaptiveMinmaxBuilder<'a> {
        AdaptiveMinmaxBuilder::new(self.y)
    }

    pub fn beads(self) -> BeadsBuilder<'a> {
        BeadsBuilder::new(self.y)
    }

    pub fn interp_pts(self, points: &'a [(usize, f64)]) -> InterpPtsBuilder<'a> {
        InterpPtsBuilder { y: self.y, points }
    }

    pub fn collab_pls(spectra: &'a [Vec<f64>]) -> CollabPlsBuilder<'a> {
        CollabPlsBuilder {
            spectra,
            params: whittaker::AslsParams::default(),
        }
    }
}

/// Ergonomic entrypoint for one-dimensional baseline correction with explicit x values.
///
/// The x-aware Whittaker builders use finite-difference penalties on the
/// supplied coordinate grid rather than interpolating to a uniform grid.
///
/// # References
///
/// - B. Fornberg, "Generation of Finite Difference Formulas on Arbitrarily
///   Spaced Grids", *Mathematics of Computation*, 1988.
/// - P. H. C. Eilers, "A Perfect Smoother", *Analytical Chemistry*, 2003.
/// - `pybaselines.Baseline` mask semantics are used as a behavioral reference.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct BaselineXY<'a> {
    x: &'a [f64],
    y: &'a [f64],
}

impl<'a> BaselineXY<'a> {
    /// Creates a method-chain entrypoint for a one-dimensional signal with explicit x values.
    pub fn new(x: &'a [f64], y: &'a [f64]) -> Result<Self> {
        whittaker::xy::validate_xy(x, y)?;
        Ok(Self { x, y })
    }

    /// Returns the x-coordinate input.
    #[must_use]
    pub fn x(&self) -> &'a [f64] {
        self.x
    }

    /// Returns the y-value input.
    #[must_use]
    pub fn y(&self) -> &'a [f64] {
        self.y
    }

    /// Configures x-aware AsLS fitting.
    pub fn asls(self) -> AslsXyBuilder<'a> {
        AslsXyBuilder::new(self.x, self.y)
    }

    /// Configures x-aware airPLS fitting.
    pub fn airpls(self) -> AirPlsXyBuilder<'a> {
        AirPlsXyBuilder::new(self.x, self.y)
    }

    /// Configures x-aware arPLS fitting.
    pub fn arpls(self) -> ArPlsXyBuilder<'a> {
        ArPlsXyBuilder::new(self.x, self.y)
    }

    /// Configures x-aware IAsLS fitting.
    pub fn iasls(self) -> IaslsXyBuilder<'a> {
        IaslsXyBuilder::new(self.x, self.y)
    }

    /// Configures x-aware drPLS fitting.
    pub fn drpls(self) -> DrPlsXyBuilder<'a> {
        DrPlsXyBuilder::new(self.x, self.y)
    }

    /// Configures x-aware iarPLS fitting.
    pub fn iarpls(self) -> IarPlsXyBuilder<'a> {
        IarPlsXyBuilder::new(self.x, self.y)
    }

    /// Configures x-aware asPLS fitting.
    pub fn aspls(self) -> AsPlsXyBuilder<'a> {
        AsPlsXyBuilder::new(self.x, self.y)
    }

    /// Configures x-aware psalsa fitting.
    pub fn psalsa(self) -> PsalsaXyBuilder<'a> {
        PsalsaXyBuilder::new(self.x, self.y)
    }

    /// Configures x-aware derpsalsa fitting.
    pub fn derpsalsa(self) -> DerPsalsaXyBuilder<'a> {
        DerPsalsaXyBuilder::new(self.x, self.y)
    }

    /// Configures x-aware brPLS fitting.
    pub fn brpls(self) -> BrPlsXyBuilder<'a> {
        BrPlsXyBuilder::new(self.x, self.y)
    }

    /// Configures x-aware lsrPLS fitting.
    pub fn lsrpls(self) -> LsrPlsXyBuilder<'a> {
        LsrPlsXyBuilder::new(self.x, self.y)
    }
}

builder_1d!(AslsBuilder, whittaker::AslsParams, whittaker::asls);
builder_1d!(AirPlsBuilder, whittaker::AirPlsParams, whittaker::airpls);
builder_1d!(ArPlsBuilder, whittaker::ArPlsParams, whittaker::arpls);
builder_1d!(IaslsBuilder, whittaker::IaslsParams, whittaker::iasls);
builder_1d!(DrPlsBuilder, whittaker::DrPlsParams, whittaker::drpls);
builder_1d!(IarPlsBuilder, whittaker::IarPlsParams, whittaker::iarpls);
builder_1d!(AsPlsBuilder, whittaker::AsPlsParams, whittaker::aspls);
builder_1d!(PsalsaBuilder, whittaker::PsalsaParams, whittaker::psalsa);
builder_1d!(
    DerPsalsaBuilder,
    whittaker::DerPsalsaParams,
    whittaker::derpsalsa
);
builder_1d!(BrPlsBuilder, whittaker::BrPlsParams, whittaker::brpls);
builder_1d!(LsrPlsBuilder, whittaker::LsrPlsParams, whittaker::lsrpls);

builder_xy_whittaker!(AslsXyBuilder, whittaker::AslsParams, whittaker::xy::asls_xy);
builder_xy_whittaker!(
    AirPlsXyBuilder,
    whittaker::AirPlsParams,
    whittaker::xy::airpls_xy
);
builder_xy_whittaker!(
    ArPlsXyBuilder,
    whittaker::ArPlsParams,
    whittaker::xy::arpls_xy
);
builder_xy_whittaker!(
    IaslsXyBuilder,
    whittaker::IaslsParams,
    whittaker::xy::iasls_xy
);
builder_xy_whittaker!(
    DrPlsXyBuilder,
    whittaker::DrPlsParams,
    whittaker::xy::drpls_xy
);
builder_xy_whittaker!(
    IarPlsXyBuilder,
    whittaker::IarPlsParams,
    whittaker::xy::iarpls_xy
);
builder_xy_whittaker!(
    AsPlsXyBuilder,
    whittaker::AsPlsParams,
    whittaker::xy::aspls_xy
);
builder_xy_whittaker!(
    PsalsaXyBuilder,
    whittaker::PsalsaParams,
    whittaker::xy::psalsa_xy
);
builder_xy_whittaker!(
    DerPsalsaXyBuilder,
    whittaker::DerPsalsaParams,
    whittaker::xy::derpsalsa_xy
);
builder_xy_whittaker!(
    BrPlsXyBuilder,
    whittaker::BrPlsParams,
    whittaker::xy::brpls_xy
);
builder_xy_whittaker!(
    LsrPlsXyBuilder,
    whittaker::LsrPlsParams,
    whittaker::xy::lsrpls_xy
);

impl<'a> AslsBuilder<'a> {
    whittaker_setters!();
    setter!(p, p, f64);

    pub fn fit_with_history(self) -> Result<FitHistory> {
        whittaker::asls_with_history(self.y, self.params)
    }

    pub fn fit_into(
        self,
        baseline: &mut [f64],
        workspace: &mut WhittakerWorkspace,
    ) -> Result<FitReport> {
        whittaker::asls_into(self.y, self.params, baseline, workspace)
    }
}

impl<'a> AirPlsBuilder<'a> {
    whittaker_setters!();

    pub fn fit_into(
        self,
        baseline: &mut [f64],
        workspace: &mut WhittakerWorkspace,
    ) -> Result<FitReport> {
        whittaker::airpls_into(self.y, self.params, baseline, workspace)
    }
}

impl<'a> ArPlsBuilder<'a> {
    whittaker_setters!();

    pub fn fit_into(
        self,
        baseline: &mut [f64],
        workspace: &mut WhittakerWorkspace,
    ) -> Result<FitReport> {
        whittaker::arpls_into(self.y, self.params, baseline, workspace)
    }
}

impl<'a> IaslsBuilder<'a> {
    whittaker_setters!();
    setter!(p, p, f64);
    setter!(lambda_1, lambda_1, f64);

    pub fn fit_into(self, baseline: &mut [f64]) -> Result<FitReport> {
        whittaker::iasls_into(self.y, self.params, baseline)
    }
}

impl<'a> DrPlsBuilder<'a> {
    whittaker_setters!();
    setter!(eta, eta, f64);

    pub fn fit_into(self, baseline: &mut [f64]) -> Result<FitReport> {
        whittaker::drpls_into(self.y, self.params, baseline)
    }
}

impl<'a> IarPlsBuilder<'a> {
    whittaker_setters!();
}

impl<'a> AsPlsBuilder<'a> {
    whittaker_setters!();
    setter!(asymmetric_coef, asymmetric_coef, f64);

    pub fn fit_with_history(self) -> Result<FitHistory> {
        whittaker::aspls_with_history(self.y, self.params)
    }
}

impl<'a> PsalsaBuilder<'a> {
    whittaker_setters!();
    setter!(p, p, f64);
    option_setter!(k, auto_k, k, f64);
}

impl<'a> DerPsalsaBuilder<'a> {
    whittaker_setters!();
    setter!(p, p, f64);
    option_setter!(k, auto_k, k, f64);
    option_setter!(
        smooth_half_window,
        auto_smooth_half_window,
        smooth_half_window,
        usize
    );
    setter!(num_smooths, num_smooths, usize);
}

impl<'a> BrPlsBuilder<'a> {
    whittaker_setters!();
    setter!(max_iter_2, max_iter_2, usize);
    setter!(tol_2, tol_2, f64);

    pub fn fit_into(self, baseline: &mut [f64]) -> Result<FitReport> {
        whittaker::brpls_into(self.y, self.params, baseline)
    }
}

impl<'a> LsrPlsBuilder<'a> {
    whittaker_setters!();
}

impl<'a> AslsXyBuilder<'a> {
    whittaker_setters!();
    setter!(p, p, f64);
}

impl<'a> AirPlsXyBuilder<'a> {
    whittaker_setters!();
}

impl<'a> ArPlsXyBuilder<'a> {
    whittaker_setters!();
}

impl<'a> IaslsXyBuilder<'a> {
    whittaker_setters!();
    setter!(p, p, f64);
    setter!(lambda_1, lambda_1, f64);
}

impl<'a> DrPlsXyBuilder<'a> {
    whittaker_setters!();
    setter!(eta, eta, f64);
}

impl<'a> IarPlsXyBuilder<'a> {
    whittaker_setters!();
}

impl<'a> AsPlsXyBuilder<'a> {
    whittaker_setters!();
    setter!(asymmetric_coef, asymmetric_coef, f64);
}

impl<'a> PsalsaXyBuilder<'a> {
    whittaker_setters!();
    setter!(p, p, f64);
    option_setter!(k, auto_k, k, f64);
}

impl<'a> DerPsalsaXyBuilder<'a> {
    whittaker_setters!();
    setter!(p, p, f64);
    option_setter!(k, auto_k, k, f64);
    option_setter!(
        smooth_half_window,
        auto_smooth_half_window,
        smooth_half_window,
        usize
    );
    setter!(num_smooths, num_smooths, usize);
}

impl<'a> BrPlsXyBuilder<'a> {
    whittaker_setters!();
    setter!(max_iter_2, max_iter_2, usize);
    setter!(tol_2, tol_2, f64);
}

impl<'a> LsrPlsXyBuilder<'a> {
    whittaker_setters!();
}

builder_1d!(PolyBuilder, polynomial::PolyParams, polynomial::poly);
builder_1d!(
    ModPolyBuilder,
    polynomial::ModPolyParams,
    polynomial::modpoly
);
builder_1d!(
    ImodPolyBuilder,
    polynomial::ImodPolyParams,
    polynomial::imodpoly
);
builder_1d!(
    PenalizedPolyBuilder,
    polynomial::PenalizedPolyParams,
    polynomial::penalized_poly
);
builder_1d!(LoessBuilder, polynomial::LoessParams, polynomial::loess);
builder_1d!(
    QuantRegBuilder,
    polynomial::QuantRegParams,
    polynomial::quant_reg
);
builder_1d!(
    GoldindecBuilder,
    polynomial::GoldindecParams,
    polynomial::goldindec
);

impl<'a> PolyBuilder<'a> {
    setter!(order, order, usize);
}

impl<'a> ModPolyBuilder<'a> {
    setter!(order, order, usize);
    setter!(max_iter, max_iter, usize);
    setter!(tol, tol, f64);
}

impl<'a> ImodPolyBuilder<'a> {
    setter!(order, order, usize);
    setter!(num_std, num_std, f64);
    setter!(max_iter, max_iter, usize);
    setter!(tol, tol, f64);
}

impl<'a> PenalizedPolyBuilder<'a> {
    setter!(order, order, usize);
    setter!(max_iter, max_iter, usize);
    setter!(tol, tol, f64);
    setter!(cost, cost, PenalizedCost);
    option_setter!(threshold, auto_threshold, threshold, f64);
    setter!(alpha_factor, alpha_factor, f64);
}

impl<'a> LoessBuilder<'a> {
    setter!(window_size, window_size, usize);
}

impl<'a> QuantRegBuilder<'a> {
    setter!(order, order, usize);
    setter!(quantile, quantile, f64);
    setter!(max_iter, max_iter, usize);
    setter!(tol, tol, f64);
    option_setter!(epsilon, auto_epsilon, epsilon, f64);
}

impl<'a> GoldindecBuilder<'a> {
    setter!(order, order, usize);
    setter!(max_iter, max_iter, usize);
    setter!(tol, tol, f64);
    setter!(cost, cost, PenalizedCost);
    setter!(peak_ratio, peak_ratio, f64);
    setter!(alpha_factor, alpha_factor, f64);
    setter!(max_threshold_iter, max_threshold_iter, usize);
    setter!(ratio_tol, ratio_tol, f64);
    setter!(threshold_tol, threshold_tol, f64);
}

builder_1d!(
    RollingBallBuilder,
    morphology::MorphologyParams,
    morphology::rolling_ball
);
builder_1d!(
    TophatBuilder,
    morphology::MorphologyParams,
    morphology::tophat
);
builder_1d!(MwmvBuilder, morphology::MorphologyParams, morphology::mwmv);
builder_1d!(MorBuilder, morphology::MorphologyParams, morphology::mor);
builder_1d!(MplsBuilder, morphology::MorphologyParams, morphology::mpls);
builder_1d!(ImorBuilder, morphology::MorphologyParams, morphology::imor);
builder_1d!(
    MormolBuilder,
    morphology::MorphologyParams,
    morphology::mormol
);
builder_1d!(
    AmormolBuilder,
    morphology::MorphologyParams,
    morphology::amormol
);
builder_1d!(
    MpsplineBuilder,
    morphology::MorphologyParams,
    morphology::mpspline
);
builder_1d!(JbcdBuilder, morphology::MorphologyParams, morphology::jbcd);
builder_1d!(SnipBuilder, morphology::SnipParams, morphology::snip);

macro_rules! morphology_window {
    ($builder:ident) => {
        impl<'a> $builder<'a> {
            setter!(window_size, window_size, usize);
        }
    };
}

morphology_window!(RollingBallBuilder);
morphology_window!(TophatBuilder);
morphology_window!(MwmvBuilder);
morphology_window!(MorBuilder);
morphology_window!(MplsBuilder);
morphology_window!(ImorBuilder);
morphology_window!(MormolBuilder);
morphology_window!(AmormolBuilder);
morphology_window!(MpsplineBuilder);
morphology_window!(JbcdBuilder);

impl<'a> SnipBuilder<'a> {
    setter!(max_half_window, max_half_window, usize);
}

builder_1d!(
    NoiseMedianBuilder,
    smoothing::SmoothingParams,
    smoothing::noise_median
);
builder_1d!(
    SmoothingSnipBuilder,
    morphology::SnipParams,
    smoothing::snip
);
builder_1d!(SwimaBuilder, smoothing::SmoothingParams, smoothing::swima);
builder_1d!(IpsaBuilder, smoothing::SmoothingParams, smoothing::ipsa);
builder_1d!(RiaBuilder, smoothing::SmoothingParams, smoothing::ria);
builder_1d!(
    PeakFillingBuilder,
    smoothing::SmoothingParams,
    smoothing::peak_filling
);

macro_rules! smoothing_window {
    ($builder:ident) => {
        impl<'a> $builder<'a> {
            setter!(window_size, window_size, usize);
            setter!(max_iter, max_iter, usize);
        }
    };
}

smoothing_window!(NoiseMedianBuilder);
smoothing_window!(SwimaBuilder);
smoothing_window!(IpsaBuilder);
smoothing_window!(RiaBuilder);
smoothing_window!(PeakFillingBuilder);

impl<'a> SmoothingSnipBuilder<'a> {
    setter!(max_half_window, max_half_window, usize);
}

builder_1d!(IrsqrBuilder, spline::IrsqrParams, spline::irsqr);
builder_1d!(
    PsplineAslsBuilder,
    whittaker::AslsParams,
    spline::pspline_asls
);
builder_1d!(
    PsplineIaslsBuilder,
    whittaker::IaslsParams,
    spline::pspline_iasls
);
builder_1d!(
    PsplineAirPlsBuilder,
    whittaker::AirPlsParams,
    spline::pspline_airpls
);
builder_1d!(
    PsplineArPlsBuilder,
    whittaker::ArPlsParams,
    spline::pspline_arpls
);
builder_1d!(
    PsplineDrPlsBuilder,
    whittaker::DrPlsParams,
    spline::pspline_drpls
);
builder_1d!(
    PsplineIarPlsBuilder,
    whittaker::IarPlsParams,
    spline::pspline_iarpls
);
builder_1d!(
    PsplineAsPlsBuilder,
    whittaker::AsPlsParams,
    spline::pspline_aspls
);
builder_1d!(
    PsplinePsalsaBuilder,
    whittaker::PsalsaParams,
    spline::pspline_psalsa
);
builder_1d!(
    PsplineDerPsalsaBuilder,
    whittaker::DerPsalsaParams,
    spline::pspline_derpsalsa
);
builder_1d!(
    PsplineMplsBuilder,
    morphology::MorphologyParams,
    spline::pspline_mpls
);
builder_1d!(
    PsplineBrPlsBuilder,
    whittaker::BrPlsParams,
    spline::pspline_brpls
);
builder_1d!(
    PsplineLsrPlsBuilder,
    whittaker::LsrPlsParams,
    spline::pspline_lsrpls
);
builder_1d!(
    MixtureModelBuilder,
    spline::MixtureModelParams,
    spline::mixture_model
);
builder_1d!(
    CornerCuttingBuilder,
    spline::CornerCuttingParams,
    spline::corner_cutting
);

impl<'a> IrsqrBuilder<'a> {
    setter!(lambda, lambda, f64);
    setter!(quantile, quantile, f64);
    setter!(max_iter, max_iter, usize);
    setter!(tol, tol, f64);
    option_setter!(epsilon, auto_epsilon, epsilon, f64);
}

impl<'a> PsplineAslsBuilder<'a> {
    whittaker_setters!();
    setter!(p, p, f64);
}

impl<'a> PsplineIaslsBuilder<'a> {
    whittaker_setters!();
    setter!(p, p, f64);
    setter!(lambda_1, lambda_1, f64);
}

impl<'a> PsplineAirPlsBuilder<'a> {
    whittaker_setters!();
}

impl<'a> PsplineArPlsBuilder<'a> {
    whittaker_setters!();
}

impl<'a> PsplineDrPlsBuilder<'a> {
    whittaker_setters!();
    setter!(eta, eta, f64);
}

impl<'a> PsplineIarPlsBuilder<'a> {
    whittaker_setters!();
}

impl<'a> PsplineAsPlsBuilder<'a> {
    whittaker_setters!();
    setter!(asymmetric_coef, asymmetric_coef, f64);
}

impl<'a> PsplinePsalsaBuilder<'a> {
    whittaker_setters!();
    setter!(p, p, f64);
    option_setter!(k, auto_k, k, f64);
}

impl<'a> PsplineDerPsalsaBuilder<'a> {
    whittaker_setters!();
    setter!(p, p, f64);
    option_setter!(k, auto_k, k, f64);
    option_setter!(
        smooth_half_window,
        auto_smooth_half_window,
        smooth_half_window,
        usize
    );
    setter!(num_smooths, num_smooths, usize);
}

impl<'a> PsplineMplsBuilder<'a> {
    setter!(window_size, window_size, usize);
}

impl<'a> PsplineBrPlsBuilder<'a> {
    whittaker_setters!();
    setter!(max_iter_2, max_iter_2, usize);
    setter!(tol_2, tol_2, f64);
}

impl<'a> PsplineLsrPlsBuilder<'a> {
    whittaker_setters!();
}

impl<'a> MixtureModelBuilder<'a> {
    setter!(lambda, lambda, f64);
    setter!(p, p, f64);
    setter!(num_knots, num_knots, usize);
    setter!(diff_order, diff_order, usize);
    setter!(max_iter, max_iter, usize);
    setter!(tol, tol, f64);
    setter!(symmetric, symmetric, bool);
}

impl<'a> CornerCuttingBuilder<'a> {
    setter!(max_iter, max_iter, usize);
}

builder_1d!(
    DietrichBuilder,
    classification::DietrichParams,
    classification::dietrich
);
builder_1d!(
    GolotvinBuilder,
    classification::GolotvinParams,
    classification::golotvin
);
builder_1d!(
    StdDistributionBuilder,
    classification::StdDistributionParams,
    classification::std_distribution
);
builder_1d!(
    FastChromBuilder,
    classification::FastChromParams,
    classification::fastchrom
);
builder_1d!(
    CwtBrBuilder,
    classification::CwtBrParams,
    classification::cwt_br
);
builder_1d!(
    FabcBuilder,
    classification::FabcParams,
    classification::fabc
);

impl<'a> DietrichBuilder<'a> {
    setter!(smooth_half_window, smooth_half_window, usize);
    setter!(num_std, num_std, f64);
    setter!(interp_half_window, interp_half_window, usize);
    setter!(poly_order, poly_order, usize);
    setter!(max_iter, max_iter, usize);
    setter!(tol, tol, f64);
    setter!(min_length, min_length, usize);
}

impl<'a> GolotvinBuilder<'a> {
    setter!(half_window, half_window, usize);
    setter!(num_std, num_std, f64);
    setter!(sections, sections, usize);
    setter!(smooth_half_window, smooth_half_window, usize);
    setter!(interp_half_window, interp_half_window, usize);
    setter!(min_length, min_length, usize);
}

impl<'a> StdDistributionBuilder<'a> {
    setter!(half_window, half_window, usize);
    setter!(interp_half_window, interp_half_window, usize);
    setter!(fill_half_window, fill_half_window, usize);
    setter!(num_std, num_std, f64);
    setter!(smooth_half_window, smooth_half_window, usize);

    pub fn fit_with_mask(self) -> Result<ClassificationFit> {
        classification::std_distribution_with_mask(self.y, self.params)
    }
}

impl<'a> FastChromBuilder<'a> {
    setter!(half_window, half_window, usize);
    option_setter!(threshold, auto_threshold, threshold, f64);
    option_setter!(min_fwhm, auto_min_fwhm, min_fwhm, usize);
    setter!(interp_half_window, interp_half_window, usize);
    setter!(smooth_half_window, smooth_half_window, usize);
    setter!(max_iter, max_iter, usize);
    setter!(min_length, min_length, usize);

    pub fn fit_with_mask(self) -> Result<ClassificationFit> {
        classification::fastchrom_with_mask(self.y, self.params)
    }
}

impl<'a> CwtBrBuilder<'a> {
    setter!(poly_order, poly_order, usize);
    setter!(num_std, num_std, f64);
    setter!(min_length, min_length, usize);
    setter!(max_iter, max_iter, usize);
    setter!(tol, tol, f64);
    setter!(symmetric, symmetric, bool);

    #[must_use]
    pub fn scales(mut self, scales: Vec<usize>) -> Self {
        self.params.scales = Some(scales);
        self
    }

    #[must_use]
    pub fn auto_scales(mut self) -> Self {
        self.params.scales = None;
        self
    }
}

impl<'a> FabcBuilder<'a> {
    setter!(lambda, lambda, f64);
    setter!(scale, scale, usize);
    setter!(num_std, num_std, f64);
    setter!(diff_order, diff_order, usize);
    setter!(min_length, min_length, usize);
}

#[derive(Debug, Clone)]
#[must_use]
pub struct RubberbandBuilder<'a> {
    y: &'a [f64],
}

impl RubberbandBuilder<'_> {
    pub fn fit(self) -> Result<Fit> {
        classification::rubberband(self.y)
    }
}

builder_1d!(
    OptimizeExtendedRangeBuilder,
    optimizers::LambdaSearchParams,
    optimizers::optimize_extended_range
);
builder_1d!(
    CustomBcBuilder,
    optimizers::CustomBcParams,
    optimizers::custom_bc
);
builder_1d!(
    AdaptiveMinmaxBuilder,
    optimizers::AdaptiveMinmaxParams,
    optimizers::adaptive_minmax
);

impl<'a> OptimizeExtendedRangeBuilder<'a> {
    setter!(start_exp, start_exp, f64);
    setter!(stop_exp, stop_exp, f64);
    setter!(steps, steps, usize);
}

impl<'a> CustomBcBuilder<'a> {
    #[must_use]
    pub fn regions(mut self, regions: Vec<(Option<usize>, Option<usize>)>) -> Self {
        self.params.regions = regions;
        self
    }

    #[must_use]
    pub fn region(mut self, start: Option<usize>, end: Option<usize>) -> Self {
        self.params.regions = vec![(start, end)];
        self
    }

    setter!(sampling, sampling, usize);

    #[must_use]
    pub fn asls_params(mut self, params: whittaker::AslsParams) -> Self {
        self.params.asls = params;
        self
    }

    option_setter!(smooth_lambda, no_smooth_lambda, smooth_lambda, f64);

    pub fn fit_with<F>(self, baseline_fn: F) -> Result<Fit>
    where
        F: FnOnce(&[f64]) -> Result<Fit>,
    {
        optimizers::custom_bc_with(self.y, self.params, baseline_fn)
    }
}

impl<'a> AdaptiveMinmaxBuilder<'a> {
    setter!(poly_order, poly_order, usize);
    setter!(left_constrained_fraction, left_constrained_fraction, f64);
    setter!(right_constrained_fraction, right_constrained_fraction, f64);
    setter!(constrained_weight, constrained_weight, f64);
}

#[derive(Debug, Clone)]
#[must_use]
pub struct CollabPlsBuilder<'a> {
    spectra: &'a [Vec<f64>],
    params: whittaker::AslsParams,
}

impl CollabPlsBuilder<'_> {
    #[must_use]
    pub fn with_params(mut self, params: whittaker::AslsParams) -> Self {
        self.params = params;
        self
    }

    #[must_use]
    pub fn params(&self) -> &whittaker::AslsParams {
        &self.params
    }

    whittaker_setters!();
    setter!(p, p, f64);

    pub fn fit(self) -> Result<Vec<Fit>> {
        optimizers::collab_pls(self.spectra, self.params)
    }
}

builder_1d!(BeadsBuilder, misc::BeadsParams, misc::beads);

impl<'a> BeadsBuilder<'a> {
    setter!(freq_cutoff, freq_cutoff, f64);
    setter!(lam_0, lam_0, f64);
    setter!(lam_1, lam_1, f64);
    setter!(lam_2, lam_2, f64);
    setter!(asymmetry, asymmetry, f64);
    setter!(filter_type, filter_type, usize);
    setter!(cost_function, cost_function, BeadsCostFunction);
    setter!(max_iter, max_iter, usize);
    setter!(tol, tol, f64);
    setter!(eps_0, eps_0, f64);
    setter!(eps_1, eps_1, f64);
    setter!(fit_parabola, fit_parabola, bool);
    option_setter!(
        smooth_half_window,
        no_smooth_half_window,
        smooth_half_window,
        usize
    );
}

#[derive(Debug, Clone)]
#[must_use]
pub struct InterpPtsBuilder<'a> {
    y: &'a [f64],
    points: &'a [(usize, f64)],
}

impl InterpPtsBuilder<'_> {
    pub fn fit(self) -> Result<Fit> {
        misc::interp_pts(self.y, self.points)
    }
}
