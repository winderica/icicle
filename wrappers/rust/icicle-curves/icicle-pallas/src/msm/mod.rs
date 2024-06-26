use crate::curve::CurveCfg;
use icicle_core::{
    curve::{Affine, Curve, Projective},
    error::IcicleResult,
    impl_msm,
    msm::{MSMConfig, MSM},
    traits::IcicleResultWrap,
};
use icicle_cuda_runtime::{
    device_context::DeviceContext,
    error::CudaError,
    memory::{DeviceSlice, HostOrDeviceSlice},
};

impl_msm!("pallas", pallas, CurveCfg);

#[cfg(test)]
pub(crate) mod tests {
    use icicle_core::impl_msm_tests;
    use icicle_core::msm::tests::*;

    use crate::curve::CurveCfg;

    impl_msm_tests!(CurveCfg);
}
