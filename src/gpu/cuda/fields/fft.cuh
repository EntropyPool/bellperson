#include "fields.cuh"

#define MAX_LOG2_LOCAL_WORK_SIZE 7

// TODO: can use CUDA intrinsic bit reverse for faster calculation.
cu_func uint32_t bitreverse(uint32_t n, uint32_t bits) {
    uint32_t r = 0;
    for(int i = 0; i < bits; i++) {
      r = (r << 1) | (n & 1);
      n >>= 1;
    }
    return r;
  }

// omegas is a 32-element array Fr. omegas should be put to constant memory.
// pq is a (1 << max_deg >> 1) element array of Fr.
// access to pq are random, should be put to shared memory.
/*
 * FFT algorithm is inspired from: http://www.bealto.com/gpu-fft_group-1.html
 */
 // You need to allocate (sizeof(Fr) * (1 << deg)) for shared memory "u".
static __global__ void radix_fft(Fr* x, // Source buffer
                          Fr* y, // Destination buffer
                          Fr* pq, // Precalculated twiddle factors
                          Fr* omegas, // [omega, omega^2, omega^4, ...]
                          uint32_t n, // Number of elements
                          uint32_t lgp, // Log2 of `p` (Read more in the link above)
                          uint32_t deg, // 1=>radix2, 2=>radix4, 3=>radix8, ...
                          uint32_t max_deg) // Maximum degree supported, according to `pq` and `omegas`
{
  // Local buffer to store intermediary values
  extern __shared__ Fr u[]; // size of u: 1<<deg, specify in kernel launcher.

  uint32_t lid = threadIdx.x;
  uint32_t lsize = blockDim.x;
  uint32_t index = blockIdx.x;
  uint32_t t = n >> deg;
  uint32_t p = 1 << lgp;
  uint32_t k = index & (p - 1);

  x += index;
  y += ((index - k) << deg) + k;

  uint32_t count = 1 << deg; // 2^deg
  uint32_t counth = count >> 1; // Half of count

  uint32_t counts = count / lsize * lid;
  uint32_t counte = counts + count / lsize;

  // Compute powers of twiddle
  const Fr twiddle = pow_lookup<Fr>(omegas, (n >> lgp >> deg) * k);
  Fr tmp = pow<Fr>(twiddle, counts);
  for(uint32_t i = counts; i < counte; i++) {
    u[i] = mul(tmp, x[i*t]);
    tmp = mul(tmp, twiddle);
  }
  __syncthreads();

  const uint32_t pqshift = max_deg - deg;
  for(uint32_t rnd = 0; rnd < deg; rnd++) {
    const uint32_t bit = counth >> rnd;
    for(uint32_t i = counts >> 1; i < counte >> 1; i++) {
      const uint32_t di = i & (bit - 1);
      const uint32_t i0 = (i << 1) - di;
      const uint32_t i1 = i0 + bit;
      tmp = u[i0];
      u[i0] = add(u[i0], u[i1]);
      u[i1] = sub(tmp, u[i1]);
      if(di != 0) u[i1] = mul(pq[di << rnd << pqshift], u[i1]);
    }

    __syncthreads();
  }

  for(uint32_t i = counts >> 1; i < counte >> 1; i++) {
    y[i*p] = u[bitreverse(i, deg)];
    y[(i+counth)*p] = u[bitreverse(i + counth, deg)];
  }
}

/// Multiplies all of the elements by `field`
template<typename FIELD>
__global__ void mul_by_field(FIELD* elements,
                             uint32_t n,
                             FIELD field) {
  const uint32_t gid = blockIdx.x * blockIdx.x + threadIdx.x;
  elements[gid] = mul(elements[gid], field);
}
  