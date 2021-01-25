mod error;

pub use self::error::*;

#[cfg(feature = "gpu")]
mod locks;

#[cfg(feature = "gpu")]
pub use self::locks::*;

#[cfg(feature = "gpu")]
mod sources;

#[cfg(feature = "gpu")]
pub use self::sources::*;

#[cfg(feature = "gpu")]
mod utils;

#[cfg(feature = "gpu")]
pub use self::utils::*;

#[cfg(feature = "gpu")]
mod fft;

#[cfg(feature = "gpu")]
pub use self::fft::{FFTKernel as CLFFTKernel};

#[cfg(feature = "gpu")]
mod multiexp;

#[cfg(feature = "gpu")]
pub use self::multiexp::{MultiexpKernel as MultiexpKernel};

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "cuda")]
pub use cuda::FFTKernel as FFTKernel;
#[cfg(all(not(feature = "cuda"), feature = "gpu"))]
pub use self::fft::FFTKernel;

#[cfg(not(feature = "gpu"))]
mod nogpu;

#[cfg(not(feature = "gpu"))]
pub use self::nogpu::*;
