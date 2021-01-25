#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::{any::TypeId, marker::PhantomData};

use ff::{Field, PrimeField, ScalarEngine};
use groupy::{CurveAffine, CurveProjective};
// #[cfg(feature = "pairing")]
// use paired::Engine;
// #[cfg(feature = "blst")]
// use blstrs::Engine;
use crate::bls::Engine;
use rust_gpu_tools::opencl;
use std::cmp::min;

use super::multiexp::{
    calc_best_chunk_size, calc_chunk_size, calc_num_groups, calc_window_size, exp_size,
};
use super::{locks, sources, utils, GPUError, GPUResult};

const LOG2_MAX_ELEMENTS: usize = 32; // At most 2^32 elements is supported.
const MAX_LOG2_RADIX: u32 = 8; // Radix256

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

impl Default for CudaInfo {
    fn default() -> Self {
        CudaInfo { device_id: 0 }
    }
}

// TODO: Logically the generic paramater of multiexp should be a group.
// We currently only implement multiexp for G1 (hard coded in function multiexp).
// Multiexp kernel for a single GPU
pub struct SingleMultiexpKernel<E>
where
    E: Engine,
{
    pub(crate) program: opencl::Program,
    pub(crate) exp_bits: usize,
    pub(crate) core_count: usize,
    pub(crate) n: usize,
    pub(crate) priority: bool,
    pub(crate) max_window_size: usize,
    pub(crate) chunk_size_scale: usize,
    pub(crate) best_chunk_size_scale: usize,
    pub(crate) g2_chunk_divider: f32,
    _phantom: std::marker::PhantomData<E::Fr>,
}

impl<E> SingleMultiexpKernel<E>
where
    E: Engine,
{
    pub fn create(d: opencl::Device, priority: bool) -> GPUResult<SingleMultiexpKernel<E>> {
        let src = sources::kernel::<E>(d.brand() == opencl::Brand::Nvidia);

        let exp_bits = exp_size::<E>() * 8;
        let core_count = utils::get_core_count(&d);
        let mem = d.memory();
        let max_window_size = utils::get_max_window_size(&d);
        let chunk_size_scale = utils::get_chunk_size_scale(&d);
        let best_chunk_size_scale = utils::get_best_chunk_size_scale(&d);
        let g2_chunk_divider = utils::get_g2_chunk_divider(&d);
        let max_n = calc_chunk_size::<E>(mem, core_count, chunk_size_scale, max_window_size);
        let best_n = calc_best_chunk_size(max_window_size, core_count, exp_bits, best_chunk_size_scale);
        let n = min(max_n, best_n);

        Ok(SingleMultiexpKernel {
            program: opencl::Program::from_opencl(d, &src)?,
            exp_bits,
            core_count,
            n,
            priority,
            max_window_size,
            chunk_size_scale,
            best_chunk_size_scale,
            g2_chunk_divider,
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn multiexp<G>(
        &mut self,
        bases: &[G],
        exps: &[<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr],
        n: usize,
    ) -> GPUResult<<G as CurveAffine>::Projective>
    where
        G: CurveAffine,
    {
        if locks::PriorityLock::should_break(self.priority) {
            return Err(GPUError::GPUTaken);
        }

        assert_eq!(exps.len(), n);
        assert_eq!(bases.len(), n);

        let exp_bits = exp_size::<E>() * 8;
        let window_size = calc_window_size(n as usize, exp_bits, self.core_count, self.max_window_size);
        let num_windows = ((exp_bits as f64) / (window_size as f64)).ceil() as usize;
        let num_groups = calc_num_groups(self.core_count, num_windows);

        // Each group will have `num_windows` threads and as there are `num_groups` groups, there will
        // be `num_groups` * `num_windows` threads in total.
        // Each thread will use `num_groups` * `num_windows` * `bucket_len` buckets.

        let mut results = vec![<G as CurveAffine>::Projective::zero(); 2 * self.core_count];
        let cuda_info: CudaInfo = Default::default();
        let exps_ptr = exps.as_ptr() as *mut Fr;

        let state = if TypeId::of::<G>() == TypeId::of::<E::G1Affine>() {
            let results_ptr = results.as_ptr() as *mut projective<G1>;
            let bases_ptr = bases.as_ptr() as *mut affine<G1>;
            let input_parameters = G1InputParameters {
                results: results_ptr,
                bases: bases_ptr,
                exps: exps_ptr,
                n: n as u32,
                num_groups: num_groups as u32,
                num_windows: num_windows as u32,
                window_size: window_size as u32,
                core_count: self.core_count as u32,
                cuda_info,
                _phantom_0: PhantomData,
            };
            unsafe { G1_multiexp_cuda(input_parameters) }
        } else if TypeId::of::<G>() == TypeId::of::<E::G2Affine>() {
            let results_ptr = results.as_ptr() as *mut projective<G2>;
            let bases_ptr = bases.as_ptr() as *mut affine<G2>;
            let input_parameters = G2InputParameters {
                results: results_ptr,
                bases: bases_ptr,
                exps: exps_ptr,
                n: n as u32,
                num_groups: num_groups as u32,
                num_windows: num_windows as u32,
                window_size: window_size as u32,
                core_count: self.core_count as u32,
                cuda_info,
                _phantom_0: PhantomData,
            };
            unsafe { G2_multiexp_cuda(input_parameters) }
        } else {
            return Err(GPUError::Simple("Only E::G1 and E::G2 are supported!"));
        };
        match state {
            State_Init_Error => return Err(GPUError::CUDAInitializationError),
            State_Compute_Error => return Err(GPUError::CUDAComputationError),
            State_Compute_Ok => {}
            _ => return Err(GPUError::CUDAUnknownState(state as usize)),
        }

        unsafe { results.set_len(num_groups * num_windows) };

        // Using the algorithm below, we can calculate the final result by accumulating the results
        // of those `NUM_GROUPS` * `NUM_WINDOWS` threads.
        let mut acc = <G as CurveAffine>::Projective::zero();
        let mut bits = 0;
        for i in 0..num_windows {
            let w = min(window_size, exp_bits - bits);
            for _ in 0..w {
                acc.double();
            }
            for g in 0..num_groups {
                acc.add_assign(&results[g * num_windows + i]);
            }
            bits += w; // Process the next window
        }

        Ok(acc)
    }
}

// TODO: Logically the generic paramater of fft should be a finite group.
// We currently only implement fft for Fr (hard coded in function radix_fft_round).
pub struct FFTKernel<E>
where
    E: Engine,
{
    pq: Vec<E::Fr>,
    omegas: Vec<E::Fr>,
    _lock: locks::GPULock, // RFC 1857: struct fields are dropped in the same order as they are declared.
    priority: bool,
}

impl<E> FFTKernel<E>
where
    E: Engine,
{
    pub fn create(priority: bool) -> GPUResult<FFTKernel<E>> {
        let lock = locks::GPULock::lock(0);

        let devices = opencl::Device::all()?;
        if devices.is_empty() {
            return Err(GPUError::Simple("No working GPUs found!"));
        }

        // Select the first device for FFT
        let device = devices[0].clone();

        Ok(FFTKernel {
            pq: vec![],
            omegas: vec![E::Fr::zero(); 32],
            _lock: lock,
            priority,
        })
    }

    /// Share some precalculated values between threads to boost the performance
    fn setup_pq_omegas(&mut self, omega: &E::Fr, n: usize, max_deg: u32) -> GPUResult<()> {
        // Precalculate:
        // [omega^(0/(2^(deg-1))), omega^(1/(2^(deg-1))), ..., omega^((2^(deg-1)-1)/(2^(deg-1)))]
        self.pq = vec![E::Fr::zero(); 1 << max_deg >> 1];
        let twiddle = omega.pow([(n >> max_deg) as u64]);
        self.pq[0] = E::Fr::one();
        if max_deg > 1 {
            self.pq[1] = twiddle;
            for i in 2..(1 << max_deg >> 1) {
                self.pq[i] = self.pq[i - 1];
                self.pq[i].mul_assign(&twiddle);
            }
        }

        // Precalculate [omega, omega^2, omega^4, omega^8, ..., omega^(2^31)]
        self.omegas[0] = *omega;
        for i in 1..LOG2_MAX_ELEMENTS {
            self.omegas[i] = self.omegas[i - 1].pow([2u64]);
        }

        Ok(())
    }

    /// Performs FFT on `a`
    /// * `omega` - Special value `omega` is used for FFT over finite-fields
    /// * `log_n` - Specifies log2 of number of elements
    pub fn radix_fft(&mut self, a: &mut [E::Fr], omega: &E::Fr, log_n: u32) -> GPUResult<()> {
        let n = 1 << log_n;
        let max_deg = min(MAX_LOG2_RADIX, log_n);
        self.setup_pq_omegas(omega, n, max_deg)?;

        if locks::PriorityLock::should_break(self.priority) {
            return Err(GPUError::GPUTaken);
        }

        let src = a.as_mut_ptr();
        let n = 1u32 << log_n;

        let pq_ptr = self.pq.as_ptr() as *mut Fr;
        let omegas_ptr = self.omegas.as_ptr() as *mut Fr;

        let input_parameters = FFTInputParameters {
            x: src as *mut Fr,
            pq: pq_ptr,
            omegas: omegas_ptr,
            n,
            lgn: log_n,
            max_deg,
            cuda_info: Default::default(),
        };
        let state = unsafe { Fr_radix_fft(input_parameters) };
        match state {
            State_Init_Error => return Err(GPUError::CUDAInitializationError),
            State_Compute_Error => return Err(GPUError::CUDAComputationError),
            State_Compute_Ok => {}
            _ => return Err(GPUError::CUDAUnknownState(state as usize)),
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    use crate::{
        bls::Bls12,
        domain::{serial_fft, Scalar},
        multicore::Worker,
        multiexp::multiexp,
        multiexp::FullDensity,
    };
    use ff::{Field, PrimeField, ScalarEngine};
    use groupy::CurveProjective;
    use paired::bls12_381::{Fr, FrRepr, G1Affine, G2Affine};
    use rand_core::SeedableRng;
    use rand_xorshift::XorShiftRng;
    use rust_gpu_tools::opencl;

    const LEN_OF_ARRAY: usize = 100;
    // TODO: Large LOG_D (e.g. 10) currently does not work
    const LOG_D: u32 = 20;

    fn get_first_gpu() -> opencl::Device {
        let devices = opencl::Device::all().expect("Must obtain gpu list");
        let device = devices[0].clone();
        return device;
    }

    fn get_rng() -> XorShiftRng {
        XorShiftRng::from_seed([
            0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06,
            0xbc, 0xe5,
        ])
    }

    fn generate_bases_exps_G1(len_of_array: usize) -> (Vec<G1Affine>, Vec<FrRepr>) {
        let rng = &mut get_rng();
        let bases = (0..len_of_array)
            .map(|_| <Bls12 as crate::bls::Engine>::G1::random(rng).into_affine())
            .collect::<Vec<_>>();
        let exps = (0..len_of_array)
            .map(|_| <Bls12 as ScalarEngine>::Fr::random(rng).into_repr())
            .collect::<Vec<_>>();
        (bases, exps)
    }

    fn generate_bases_exps_G2(len_of_array: usize) -> (Vec<G2Affine>, Vec<FrRepr>) {
        let rng = &mut get_rng();
        let bases = (0..len_of_array)
            .map(|_| <Bls12 as crate::bls::Engine>::G2::random(rng).into_affine())
            .collect::<Vec<_>>();
        let exps = (0..len_of_array)
            .map(|_| <Bls12 as ScalarEngine>::Fr::random(rng).into_repr())
            .collect::<Vec<_>>();
        (bases, exps)
    }

    pub fn my_cpu_multiexp<G>(
        bases: &[G],
        exps: &[<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr],
    ) -> <G as CurveAffine>::Projective
    where
        G: CurveAffine,
        <G as groupy::CurveAffine>::Engine: paired::Engine,
    {
        let pool = Worker::new();
        let bases = Arc::new(bases.iter().map(Clone::clone).collect::<Vec<_>>());
        let exps = Arc::new(exps.iter().map(Clone::clone).collect::<Vec<_>>());

        let cpu_result = multiexp(&pool, (bases, 0), FullDensity, exps, &mut None).wait();
        dbg!(&cpu_result);
        cpu_result.unwrap()
    }

    pub fn my_gpu_multiexp<G>(
        bases: &[G],
        exps: &[<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr],
    ) -> <G as CurveAffine>::Projective
    where
        G: CurveAffine,
    {
        let device = get_first_gpu();
        let mut gpu_multiexp =
            super::super::multiexp::SingleMultiexpKernel::<Bls12>::create(device.clone(), false)
                .expect("Must create gpu kernel");
        let gpu_result = gpu_multiexp.multiexp(bases, exps, bases.len());
        dbg!(&gpu_result);
        gpu_result.unwrap()
    }

    pub fn my_cuda_multiexp<G>(
        bases: &[G],
        exps: &[<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr],
    ) -> <G as CurveAffine>::Projective
    where
        G: CurveAffine,
    {
        let device = get_first_gpu();
        let mut cuda_multiexp = SingleMultiexpKernel::<Bls12>::create(device.clone(), false)
            .expect("Must create cuda kernel");
        let cuda_result = cuda_multiexp.multiexp(&bases, &exps, bases.len());
        dbg!(&cuda_result);
        cuda_result.unwrap()
    }

    #[test]
    fn same_output_for_cuda_multiexp_and_gpu_multiexp_G1() {
        let (bases, exps) = generate_bases_exps_G1(LEN_OF_ARRAY);
        let cuda_result = my_cuda_multiexp(&bases, &exps);
        let gpu_result = my_gpu_multiexp(&bases, &exps);
        assert_eq!(cuda_result, gpu_result);
    }

    #[test]
    fn same_output_for_cuda_multiexp_and_cpu_multiexp_G1() {
        let (bases, exps) = generate_bases_exps_G1(LEN_OF_ARRAY);
        let cuda_result = my_cuda_multiexp(&bases, &exps);
        let cpu_result = my_cpu_multiexp(&bases, &exps);
        assert_eq!(cuda_result, cpu_result);
    }

    #[test]
    fn same_output_for_gpu_multiexp_and_cpu_multiexp_G1() {
        let (bases, exps) = generate_bases_exps_G1(LEN_OF_ARRAY);
        let gpu_result = my_gpu_multiexp(&bases, &exps);
        let cpu_result = my_cpu_multiexp(&bases, &exps);
        assert_eq!(gpu_result, cpu_result);
    }

    #[test]
    fn same_output_for_cuda_multiexp_and_gpu_multiexp_G2() {
        let (bases, exps) = generate_bases_exps_G2(LEN_OF_ARRAY);
        let cuda_result = my_cuda_multiexp(&bases, &exps);
        let gpu_result = my_gpu_multiexp(&bases, &exps);
        assert_eq!(cuda_result, gpu_result);
    }

    #[test]
    fn same_output_for_cuda_multiexp_and_cpu_multiexp_G2() {
        let (bases, exps) = generate_bases_exps_G2(LEN_OF_ARRAY);
        let cuda_result = my_cuda_multiexp(&bases, &exps);
        let cpu_result = my_cpu_multiexp(&bases, &exps);
        assert_eq!(cuda_result, cpu_result);
    }

    #[test]
    fn same_output_for_gpu_multiexp_and_cpu_multiexp_G2() {
        let (bases, exps) = generate_bases_exps_G2(LEN_OF_ARRAY);
        let gpu_result = my_gpu_multiexp(&bases, &exps);
        let cpu_result = my_cpu_multiexp(&bases, &exps);
        assert_eq!(gpu_result, cpu_result);
    }

    fn generate_frs(log_d: u32) -> Vec<Fr> {
        let rng = &mut XorShiftRng::from_seed([
            0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06,
            0xbc, 0xe5,
        ]);
        let d = 1 << log_d;
        (0..d).map(|_| Fr::random(rng)).collect::<Vec<_>>()
    }

    pub fn my_cpu_fft(a: &[Fr], omega: &Fr, log_n: u32) -> Vec<Fr> {
        let mut frs: Vec<Scalar<Bls12>> = a.iter().map(|f| Scalar::<Bls12>(f.clone())).collect();
        serial_fft(&mut frs, omega, log_n);
        frs.iter().map(|x| x.0).collect()
    }

    pub fn my_gpu_fft(a: &[Fr], omega: &Fr, log_n: u32) -> Vec<Fr> {
        let mut frs: Vec<Fr> = a.iter().map(Clone::clone).collect();
        let mut gpu_kern = super::super::fft::FFTKernel::<Bls12>::create(false)
            .expect("Cannot initialize kernel!");
        let gpu_fft_result = gpu_kern.radix_fft(&mut frs, omega, log_n);
        assert!(gpu_fft_result.is_ok());
        frs
    }

    pub fn my_cuda_fft(a: &[Fr], omega: &Fr, log_n: u32) -> Vec<Fr> {
        let mut frs: Vec<Fr> = a.iter().map(Clone::clone).collect();
        let mut cuda_kern = FFTKernel::<Bls12>::create(false).expect("Cannot initialize kernel!");
        let cuda_fft_result = cuda_kern.radix_fft(&mut frs, omega, log_n);
        assert!(cuda_fft_result.is_ok());
        frs
    }

    #[test]
    fn same_output_for_cuda_fft_and_gpu_fft() {
        let log_d = LOG_D;
        let elems = generate_frs(log_d);
        let omega = &Fr::root_of_unity();
        let cuda_result = my_cuda_fft(&elems, omega, log_d);
        let gpu_result = my_gpu_fft(&elems, omega, log_d);
        assert_eq!(cuda_result, gpu_result);
    }

    #[test]
    fn same_output_for_cuda_fft_and_cpu_fft() {
        let log_d = LOG_D;
        let elems = generate_frs(log_d);
        let omega = &Fr::root_of_unity();
        let cuda_result = my_cuda_fft(&elems, omega, log_d);
        let cpu_result = my_cpu_fft(&elems, omega, log_d);
        assert_eq!(cuda_result, cpu_result);
    }

    #[test]
    fn same_output_for_gpu_fft_and_cpu_fft() {
        let log_d = LOG_D;
        let elems = generate_frs(log_d);
        let omega = &Fr::root_of_unity();
        let gpu_result = my_gpu_fft(&elems, omega, log_d);
        let cpu_result = my_cpu_fft(&elems, omega, log_d);
        assert_eq!(gpu_result, cpu_result);
    }
}
