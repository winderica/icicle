use icicle_cuda_runtime::device::check_device;
use icicle_cuda_runtime::{
    device_context::{DeviceContext, DEFAULT_DEVICE_ID},
    memory::{HostOrDeviceSlice, DeviceSlice},
};

use crate::{error::IcicleResult, traits::FieldImpl};

pub mod tests;

/// Struct that encodes VecOps parameters.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct VecOpsConfig<'a> {
    /// Details related to the device such as its id and stream id. See [DeviceContext](@ref device_context::DeviceContext).
    pub ctx: DeviceContext<'a>,
    is_a_on_device: bool,
    is_b_on_device: bool,
    is_result_on_device: bool,
    /// Whether to run the vector operations asynchronously. If set to `true`, the functions will be non-blocking and you'd need to synchronize
    /// it explicitly by running `stream.synchronize()`. If set to false, the functions will block the current CPU thread.
    pub is_async: bool,
}

impl<'a> Default for VecOpsConfig<'a> {
    fn default() -> Self {
        Self::default_for_device(DEFAULT_DEVICE_ID)
    }
}

impl<'a> VecOpsConfig<'a> {
    pub fn default_for_device(device_id: usize) -> Self {
        VecOpsConfig {
            ctx: DeviceContext::default_for_device(device_id),
            is_a_on_device: false,
            is_b_on_device: false,
            is_result_on_device: false,
            is_async: false,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct BitReverseConfig<'a> {
    /// Details related to the device such as its id and stream id. See [DeviceContext](@ref device_context::DeviceContext).
    pub ctx: DeviceContext<'a>,

    /// True if inputs are on device and false if they're on host. Default value: false.
    pub is_input_on_device: bool,

    /// If true, output is preserved on device, otherwise on host. Default value: false.
    pub is_output_on_device: bool,

    /// Whether to run the vector operations asynchronously. If set to `true`, the functions will be non-blocking and you'd need to synchronize
    /// it explicitly by running `stream.synchronize()`. If set to false, the functions will block the current CPU thread.
    pub is_async: bool,
}

impl<'a> Default for BitReverseConfig<'a> {
    fn default() -> Self {
        Self::default_for_device(DEFAULT_DEVICE_ID)
    }
}

impl<'a> BitReverseConfig<'a> {
    pub fn default_for_device(device_id: usize) -> Self {
        BitReverseConfig {
            ctx: DeviceContext::default_for_device(device_id),
            is_input_on_device: false,
            is_output_on_device: false,
            is_async: false,
        }
    }
}

#[doc(hidden)]
pub trait VecOps<F> {
    fn add(
        a: &(impl HostOrDeviceSlice<F> + ?Sized),
        b: &(impl HostOrDeviceSlice<F> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> IcicleResult<()>;

    fn accumulate(
        a: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        b: &(impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> IcicleResult<()>;

    fn sub(
        a: &(impl HostOrDeviceSlice<F> + ?Sized),
        b: &(impl HostOrDeviceSlice<F> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> IcicleResult<()>;

    fn mul(
        a: &(impl HostOrDeviceSlice<F> + ?Sized),
        b: &(impl HostOrDeviceSlice<F> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> IcicleResult<()>;

    fn mul_mat(
        vec: &(impl HostOrDeviceSlice<F> + ?Sized),
        mat: &(impl HostOrDeviceSlice<F> + ?Sized),
        row_ptr: &(impl HostOrDeviceSlice<i32> + ?Sized),
        col_idx: &(impl HostOrDeviceSlice<i32> + ?Sized),
        result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &VecOpsConfig,
    ) -> IcicleResult<()>;

    fn prepare_matrix(
        mat: &(impl HostOrDeviceSlice<F> + ?Sized),
        row_ptr: &(impl HostOrDeviceSlice<i32> + ?Sized),
        col_idx: &(impl HostOrDeviceSlice<i32> + ?Sized),
        ctx: &DeviceContext,
        output_mat: &mut DeviceSlice<F>,
        output_row_ptr: &mut DeviceSlice<i32>,
        output_col_idx: &mut DeviceSlice<i32>,
    ) -> IcicleResult<()>;

    fn compute_t(
        mat_a: &(impl HostOrDeviceSlice<F> + ?Sized),
        row_ptr_a: &(impl HostOrDeviceSlice<i32> + ?Sized),
        col_idx_a: &(impl HostOrDeviceSlice<i32> + ?Sized),
        mat_b: &(impl HostOrDeviceSlice<F> + ?Sized),
        row_ptr_b: &(impl HostOrDeviceSlice<i32> + ?Sized),
        col_idx_b: &(impl HostOrDeviceSlice<i32> + ?Sized),
        mat_c: &(impl HostOrDeviceSlice<F> + ?Sized),
        row_ptr_c: &(impl HostOrDeviceSlice<i32> + ?Sized),
        col_idx_c: &(impl HostOrDeviceSlice<i32> + ?Sized),
        z1: &(impl HostOrDeviceSlice<F> + ?Sized),
        z2: &(impl HostOrDeviceSlice<F> + ?Sized),
        ctx: &DeviceContext,
        result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    ) -> IcicleResult<()>;

    fn transpose(
        input: &(impl HostOrDeviceSlice<F> + ?Sized),
        row_size: u32,
        column_size: u32,
        output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        ctx: &DeviceContext,
        on_device: bool,
        is_async: bool,
    ) -> IcicleResult<()>;

    fn bit_reverse(
        input: &(impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &BitReverseConfig,
        output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    ) -> IcicleResult<()>;

    fn bit_reverse_inplace(
        input: &mut (impl HostOrDeviceSlice<F> + ?Sized),
        cfg: &BitReverseConfig,
    ) -> IcicleResult<()>;
}

fn check_vec_ops_args<'a, F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &(impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig<'a>,
) -> VecOpsConfig<'a> {
    if a.len() != b.len() || a.len() != result.len() {
        panic!(
            "left, right and output lengths {}; {}; {} do not match",
            a.len(),
            b.len(),
            result.len()
        );
    }
    let ctx_device_id = cfg
        .ctx
        .device_id;
    if let Some(device_id) = a.device_id() {
        assert_eq!(device_id, ctx_device_id, "Device ids in a and context are different");
    }
    if let Some(device_id) = b.device_id() {
        assert_eq!(device_id, ctx_device_id, "Device ids in b and context are different");
    }
    if let Some(device_id) = result.device_id() {
        assert_eq!(
            device_id, ctx_device_id,
            "Device ids in result and context are different"
        );
    }
    check_device(ctx_device_id);

    let mut res_cfg = cfg.clone();
    res_cfg.is_a_on_device = a.is_on_device();
    res_cfg.is_b_on_device = b.is_on_device();
    res_cfg.is_result_on_device = result.is_on_device();
    res_cfg
}
fn check_bit_reverse_args<'a, F>(
    input: &(impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &BitReverseConfig<'a>,
    output: &(impl HostOrDeviceSlice<F> + ?Sized),
) -> BitReverseConfig<'a> {
    if input.len() & (input.len() - 1) != 0 {
        panic!("input length must be a power of 2, input length: {}", input.len());
    }
    if input.len() != output.len() {
        panic!(
            "input and output lengths {}; {} do not match",
            input.len(),
            output.len()
        );
    }
    let ctx_device_id = cfg
        .ctx
        .device_id;
    if let Some(device_id) = input.device_id() {
        assert_eq!(
            device_id, ctx_device_id,
            "Device ids in input and context are different"
        );
    }
    if let Some(device_id) = output.device_id() {
        assert_eq!(
            device_id, ctx_device_id,
            "Device ids in output and context are different"
        );
    }
    check_device(ctx_device_id);
    let mut res_cfg = cfg.clone();
    res_cfg.is_input_on_device = input.is_on_device();
    res_cfg.is_output_on_device = output.is_on_device();
    res_cfg
}

pub fn add_scalars<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> IcicleResult<()>
    where
        F: FieldImpl,
        <F as FieldImpl>::Config: VecOps<F>,
{
    let cfg = check_vec_ops_args(a, b, result, cfg);
    <<F as FieldImpl>::Config as VecOps<F>>::add(a, b, result, &cfg)
}

pub fn accumulate_scalars<F>(
    a: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
{
    let cfg = check_vec_ops_args(a, b, a, cfg);
    <<F as FieldImpl>::Config as VecOps<F>>::accumulate(a, b, &cfg)
}

pub fn sub_scalars<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> IcicleResult<()>
    where
        F: FieldImpl,
        <F as FieldImpl>::Config: VecOps<F>,
{
    let cfg = check_vec_ops_args(a, b, result, cfg);
    <<F as FieldImpl>::Config as VecOps<F>>::sub(a, b, result, &cfg)
}

pub fn mul_scalars<F>(
    a: &(impl HostOrDeviceSlice<F> + ?Sized),
    b: &(impl HostOrDeviceSlice<F> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> IcicleResult<()>
    where
        F: FieldImpl,
        <F as FieldImpl>::Config: VecOps<F>,
{
    let cfg = check_vec_ops_args(a, b, result, cfg);
    <<F as FieldImpl>::Config as VecOps<F>>::mul(a, b, result, &cfg)
}

pub fn mul_mat<F>(
    vec: &(impl HostOrDeviceSlice<F> + ?Sized),
    mat: &(impl HostOrDeviceSlice<F> + ?Sized),
    row_ptr: &(impl HostOrDeviceSlice<i32> + ?Sized),
    col_idx: &(impl HostOrDeviceSlice<i32> + ?Sized),
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &VecOpsConfig,
) -> IcicleResult<()>
    where
        F: FieldImpl,
        <F as FieldImpl>::Config: VecOps<F>,
{
    let cfg = {
        let ctx_device_id = cfg
            .ctx
            .device_id;
        if let Some(device_id) = vec.device_id() {
            assert_eq!(device_id, ctx_device_id, "Device ids in a and context are different");
        }
        if let Some(device_id) = mat.device_id() {
            assert_eq!(device_id, ctx_device_id, "Device ids in b and context are different");
        }
        if let Some(device_id) = result.device_id() {
            assert_eq!(
                device_id, ctx_device_id,
                "Device ids in result and context are different"
            );
        }
        check_device(ctx_device_id);

        let mut res_cfg = cfg.clone();
        res_cfg.is_a_on_device = vec.is_on_device();
        res_cfg.is_b_on_device = mat.is_on_device();
        res_cfg.is_result_on_device = result.is_on_device();
        res_cfg
    };
    <<F as FieldImpl>::Config as VecOps<F>>::mul_mat(vec, mat, row_ptr, col_idx, result, &cfg)
}


pub fn prepare_matrix<F>(
    mat: &(impl HostOrDeviceSlice<F> + ?Sized),
    row_ptr: &(impl HostOrDeviceSlice<i32> + ?Sized),
    col_idx: &(impl HostOrDeviceSlice<i32> + ?Sized),
    ctx: &DeviceContext,
    output_mat: &mut DeviceSlice<F>,
    output_row_ptr: &mut DeviceSlice<i32>,
    output_col_idx: &mut DeviceSlice<i32>,
) -> IcicleResult<()> where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
{
    <<F as FieldImpl>::Config as VecOps<F>>::prepare_matrix(
        mat,
        row_ptr,
        col_idx,
        ctx,
        output_mat,
        output_row_ptr,
        output_col_idx,
    )
}

pub fn compute_t<F>(
    mat_a: &(impl HostOrDeviceSlice<F> + ?Sized),
    row_ptr_a: &(impl HostOrDeviceSlice<i32> + ?Sized),
    col_idx_a: &(impl HostOrDeviceSlice<i32> + ?Sized),
    mat_b: &(impl HostOrDeviceSlice<F> + ?Sized),
    row_ptr_b: &(impl HostOrDeviceSlice<i32> + ?Sized),
    col_idx_b: &(impl HostOrDeviceSlice<i32> + ?Sized),
    mat_c: &(impl HostOrDeviceSlice<F> + ?Sized),
    row_ptr_c: &(impl HostOrDeviceSlice<i32> + ?Sized),
    col_idx_c: &(impl HostOrDeviceSlice<i32> + ?Sized),
    z1: &(impl HostOrDeviceSlice<F> + ?Sized),
    z2: &(impl HostOrDeviceSlice<F> + ?Sized),
    ctx: &DeviceContext,
    result: &mut (impl HostOrDeviceSlice<F> + ?Sized),
) -> IcicleResult<()> where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
{
    <<F as FieldImpl>::Config as VecOps<F>>::compute_t(
        mat_a, row_ptr_a, col_idx_a, mat_b, row_ptr_b, col_idx_b, mat_c, row_ptr_c, col_idx_c, z1, z2, ctx, result,
    )
}

pub fn transpose_matrix<F>(
    input: &(impl HostOrDeviceSlice<F> + ?Sized),
    row_size: u32,
    column_size: u32,
    output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    ctx: &DeviceContext,
    on_device: bool,
    is_async: bool,
) -> IcicleResult<()>
    where
        F: FieldImpl,
        <F as FieldImpl>::Config: VecOps<F>,
{
    <<F as FieldImpl>::Config as VecOps<F>>::transpose(input, row_size, column_size, output, ctx, on_device, is_async)
}

pub fn bit_reverse<F>(
    input: &(impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &BitReverseConfig,
    output: &mut (impl HostOrDeviceSlice<F> + ?Sized),
) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
{
    let cfg = check_bit_reverse_args(input, cfg, output);
    <<F as FieldImpl>::Config as VecOps<F>>::bit_reverse(input, &cfg, output)
}

pub fn bit_reverse_inplace<F>(
    input: &mut (impl HostOrDeviceSlice<F> + ?Sized),
    cfg: &BitReverseConfig,
) -> IcicleResult<()>
where
    F: FieldImpl,
    <F as FieldImpl>::Config: VecOps<F>,
{
    let cfg = check_bit_reverse_args(input, cfg, input);
    <<F as FieldImpl>::Config as VecOps<F>>::bit_reverse_inplace(input, &cfg)
}

#[macro_export]
macro_rules! impl_vec_ops_field {
    (
        $field_prefix:literal,
        $field_prefix_ident:ident,
        $field:ident,
        $field_config:ident
    ) => {
        mod $field_prefix_ident {
            use crate::vec_ops::{$field, CudaError, DeviceContext, HostOrDeviceSlice};
            use icicle_core::vec_ops::BitReverseConfig;
            use icicle_core::vec_ops::VecOpsConfig;

            extern "C" {
                #[link_name = concat!($field_prefix, "_add_cuda")]
                pub(crate) fn add_scalars_cuda(
                    a: *const $field,
                    b: *const $field,
                    size: u32,
                    cfg: *const VecOpsConfig,
                    result: *mut $field,
                ) -> CudaError;

                #[link_name = concat!($field_prefix, "_accumulate_cuda")]
                pub(crate) fn accumulate_scalars_cuda(
                    a: *mut $field,
                    b: *const $field,
                    size: u32,
                    cfg: *const VecOpsConfig,
                ) -> CudaError;

                #[link_name = concat!($field_prefix, "_sub_cuda")]
                pub(crate) fn sub_scalars_cuda(
                    a: *const $field,
                    b: *const $field,
                    size: u32,
                    cfg: *const VecOpsConfig,
                    result: *mut $field,
                ) -> CudaError;

                #[link_name = concat!($field_prefix, "_mul_cuda")]
                pub(crate) fn mul_scalars_cuda(
                    a: *const $field,
                    b: *const $field,
                    size: u32,
                    cfg: *const VecOpsConfig,
                    result: *mut $field,
                ) -> CudaError;

                #[link_name = concat!($field_prefix, "_mul_mat_cuda")]
                pub(crate) fn mul_mat_cuda(
                    vec: *const $field,
                    mat: *const $field,
                    row_ptr: *const i32,
                    col_idx: *const i32,
                    n_rows: i32,
                    n_cols: i32,
                    cfg: *const VecOpsConfig,
                    result: *mut $field,
                ) -> CudaError;

                #[link_name = concat!($field_prefix, "_prepare_matrix_cuda")]
                pub(crate) fn prepare_matrix_cuda(
                    mat: *const $field,
                    row_ptr: *const i32,
                    col_idx: *const i32,
                    n_rows: i32,
                    ctx: *const DeviceContext,
                    output_mat: *const $field,
                    output_row_ptr: *const i32,
                    output_col_idx: *const i32,
                ) -> CudaError;

                #[link_name = concat!($field_prefix, "_compute_t_cuda")]
                pub(crate) fn compute_t_cuda(
                    mat_a: *const $field,
                    row_ptr_a: *const i32,
                    col_idx_a: *const i32,
                    mat_b: *const $field,
                    row_ptr_b: *const i32,
                    col_idx_b: *const i32,
                    mat_c: *const $field,
                    row_ptr_c: *const i32,
                    col_idx_c: *const i32,
                    z1: *const $field,
                    z2: *const $field,
                    n_rows: i32,
                    n_cols: i32,
                    ctx: *const DeviceContext,
                    result: *const $field,
                ) -> CudaError;

                #[link_name = concat!($field_prefix, "_transpose_matrix_cuda")]
                pub(crate) fn transpose_cuda(
                    input: *const $field,
                    row_size: u32,
                    column_size: u32,
                    output: *mut $field,
                    ctx: *const DeviceContext,
                    on_device: bool,
                    is_async: bool,
                ) -> CudaError;

                #[link_name = concat!($field_prefix, "_bit_reverse_cuda")]
                pub(crate) fn bit_reverse_cuda(
                    input: *const $field,
                    size: u64,
                    config: *const BitReverseConfig,
                    output: *mut $field,
                ) -> CudaError;
            }
        }

        impl VecOps<$field> for $field_config {
            fn add(
                a: &(impl HostOrDeviceSlice<$field> + ?Sized),
                b: &(impl HostOrDeviceSlice<$field> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::add_scalars_cuda(
                        a.as_ptr(),
                        b.as_ptr(),
                        a.len() as u32,
                        cfg as *const VecOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

            fn accumulate(
                a: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                b: &(impl HostOrDeviceSlice<$field> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::accumulate_scalars_cuda(
                        a.as_mut_ptr(),
                        b.as_ptr(),
                        a.len() as u32,
                        cfg as *const VecOpsConfig,
                    )
                    .wrap()
                }
            }

            fn sub(
                a: &(impl HostOrDeviceSlice<$field> + ?Sized),
                b: &(impl HostOrDeviceSlice<$field> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::sub_scalars_cuda(
                        a.as_ptr(),
                        b.as_ptr(),
                        a.len() as u32,
                        cfg as *const VecOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

            fn mul(
                a: &(impl HostOrDeviceSlice<$field> + ?Sized),
                b: &(impl HostOrDeviceSlice<$field> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::mul_scalars_cuda(
                        a.as_ptr(),
                        b.as_ptr(),
                        a.len() as u32,
                        cfg as *const VecOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

            fn mul_mat(
                vec: &(impl HostOrDeviceSlice<$field> + ?Sized),
                mat: &(impl HostOrDeviceSlice<$field> + ?Sized),
                row_ptr: &(impl HostOrDeviceSlice<i32> + ?Sized),
                col_idx: &(impl HostOrDeviceSlice<i32> + ?Sized),
                result: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                cfg: &VecOpsConfig,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::mul_mat_cuda(
                        vec.as_ptr(),
                        mat.as_ptr(),
                        row_ptr.as_ptr(),
                        col_idx.as_ptr(),
                        (row_ptr.len() - 1) as i32,
                        vec.len() as i32,
                        cfg as *const VecOpsConfig,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }


    fn prepare_matrix(
        mat:  &(impl HostOrDeviceSlice<$field> + ?Sized),
        row_ptr: &(impl HostOrDeviceSlice<i32> + ?Sized),
        col_idx: &(impl HostOrDeviceSlice<i32> + ?Sized),
        ctx: &DeviceContext,
        output_mat: &mut icicle_cuda_runtime::memory::DeviceSlice<$field>,
        output_row_ptr: &mut icicle_cuda_runtime::memory::DeviceSlice<i32>,
        output_col_idx: &mut icicle_cuda_runtime::memory::DeviceSlice<i32>,
    ) -> IcicleResult<()> {
unsafe {
                    $field_prefix_ident::prepare_matrix_cuda(
                        mat.as_ptr(),
                        row_ptr.as_ptr(),
                        col_idx.as_ptr(),
                        (row_ptr.len() - 1) as i32,
                        ctx as *const DeviceContext,
                        output_mat.as_mut_ptr(),
                        output_row_ptr.as_mut_ptr(),
                        output_col_idx.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

    fn compute_t(
        mat_a:  &(impl HostOrDeviceSlice<$field> + ?Sized),
        row_ptr_a: &(impl HostOrDeviceSlice<i32> + ?Sized),
        col_idx_a: &(impl HostOrDeviceSlice<i32> + ?Sized),
        mat_b:  &(impl HostOrDeviceSlice<$field> + ?Sized),
        row_ptr_b: &(impl HostOrDeviceSlice<i32> + ?Sized),
        col_idx_b: &(impl HostOrDeviceSlice<i32> + ?Sized),
        mat_c:  &(impl HostOrDeviceSlice<$field> + ?Sized),
        row_ptr_c: &(impl HostOrDeviceSlice<i32> + ?Sized),
        col_idx_c: &(impl HostOrDeviceSlice<i32> + ?Sized),
        z1:  &(impl HostOrDeviceSlice<$field> + ?Sized),
        z2:  &(impl HostOrDeviceSlice<$field> + ?Sized),
        ctx: &DeviceContext,
        result:  &mut (impl HostOrDeviceSlice<$field> + ?Sized),
    ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::compute_t_cuda(
                        mat_a.as_ptr(),
                        row_ptr_a.as_ptr(),
                        col_idx_a.as_ptr(),
                        mat_b.as_ptr(),
                        row_ptr_b.as_ptr(),
                        col_idx_b.as_ptr(),
                        mat_c.as_ptr(),
                        row_ptr_c.as_ptr(),
                        col_idx_c.as_ptr(),
                        z1.as_ptr(),
                        z2.as_ptr(),
                        (row_ptr_a.len() - 1) as i32,
                        z1.len() as i32,
                        ctx as *const DeviceContext,
                        result.as_mut_ptr(),
                    )
                    .wrap()
                }
            }


            fn transpose(
                input: &(impl HostOrDeviceSlice<$field> + ?Sized),
                row_size: u32,
                column_size: u32,
                output: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                ctx: &DeviceContext,
                on_device: bool,
                is_async: bool,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::transpose_cuda(
                        input.as_ptr(),
                        row_size,
                        column_size,
                        output.as_mut_ptr(),
                        ctx as *const _ as *const DeviceContext,
                        on_device,
                        is_async,
                    )
                    .wrap()
                }
            }

            fn bit_reverse(
                input: &(impl HostOrDeviceSlice<$field> + ?Sized),
                cfg: &BitReverseConfig,
                output: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::bit_reverse_cuda(
                        input.as_ptr(),
                        input.len() as u64,
                        cfg as *const BitReverseConfig,
                        output.as_mut_ptr(),
                    )
                    .wrap()
                }
            }

            fn bit_reverse_inplace(
                input: &mut (impl HostOrDeviceSlice<$field> + ?Sized),
                cfg: &BitReverseConfig,
            ) -> IcicleResult<()> {
                unsafe {
                    $field_prefix_ident::bit_reverse_cuda(
                        input.as_ptr(),
                        input.len() as u64,
                        cfg as *const BitReverseConfig,
                        input.as_mut_ptr(),
                    )
                    .wrap()
                }
            }
        }
    };
}

#[macro_export]
macro_rules! impl_vec_add_tests {
    (
      $field:ident
    ) => {
        #[test]
        pub fn test_vec_add_scalars() {
            check_vec_ops_scalars::<$field>();
        }

        #[test]
        pub fn test_bit_reverse() {
            check_bit_reverse::<$field>()
        }
        #[test]
        pub fn test_bit_reverse_inplace() {
            check_bit_reverse_inplace::<$field>()
        }
    };
}
