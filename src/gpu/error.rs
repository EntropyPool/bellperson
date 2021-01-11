#[cfg(feature = "gpu")]
use rust_gpu_tools::opencl;

#[derive(thiserror::Error, Debug)]
pub enum GPUError {
    #[error("GPUError: {0}")]
    Simple(&'static str),
    #[cfg(feature = "gpu")]
    #[error("OpenCL Error: {0}")]
    OpenCL(#[from] opencl::GPUError),
    #[cfg(feature = "gpu")]
    #[error("GPU taken by a high priority process!")]
    GPUTaken,
    #[cfg(feature = "gpu")]
    #[error("No kernel is initialized!")]
    KernelUninitialized,
    #[error("GPU accelerator is disabled!")]
    GPUDisabled,
    #[error("Error while initilizing CUDA")]
    CUDAInitializationError,
    #[error("Error while doing CUDA computation")]
    CUDAComputationError,
    #[error("Error while doing CUDA computation")]
    CUDAUnknownState(usize),
}

pub type GPUResult<T> = std::result::Result<T, GPUError>;

#[cfg(feature = "gpu")]
impl From<std::boxed::Box<dyn std::any::Any + std::marker::Send>> for GPUError {
    fn from(e: std::boxed::Box<dyn std::any::Any + std::marker::Send>) -> Self {
        match e.downcast::<Self>() {
            Ok(err) => *err,
            Err(_) => GPUError::Simple("An unknown GPU error happened!"),
        }
    }
}
