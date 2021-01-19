#ifndef MULTIEXP_HEADER
#define MULTIEXP_HEADER
#include "fields.cuh"

// types used in our code.
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
template<typename G>
cu_func projective<G> add_mixed(projective<G> a_, affine<G> b) {
    #ifndef BLSTRS
    if(b.inf) {
        return a_;
    }
  #endif

  // cache global memory to a better place. 
  projective<G> a = a_;

  if(eq<G>(a.z, G::Get_ZERO())) {
    a.x = b.x;
    a.y = b.y;
    a.z = G::Get_ONE();
    return a;
  }

  const G z1z1 = sqr<G>(a.z);
  const G u2 = mul<G>(b.x, z1z1);
  const G s2 = mul<G>(mul<G>(b.y, a.z), z1z1);

  if(eq<G>(a.x, u2) && eq<G>(a.y, s2)) {
      return double_op(a);
  }

  const G h = sub(u2, a.x); // H = U2-X1
  const G hh = sqr(h); // HH = H^2
  G i = double_op(hh); i = double_op(i); // I = 4*HH
  G j = mul(h, i); // J = H*I
  G r = sub(s2, a.y); r = double_op(r); // r = 2*(S2-Y1)
  const G v = mul(a.x, i);

  projective<G> ret;

  // X3 = r^2 - J - 2*V
  ret.x = sub(sub(sqr(r), j), double_op(v));

  // Y3 = r*(V-X3)-2*Y1*J
  j = mul(a.y, j); j = double_op(j);
  ret.y = sub(mul(sub(v, ret.x), r), j);

  // Z3 = (Z1+H)^2-Z1Z1-HH
  ret.z = add(a.z, h); ret.z = sub(sub(sqr(ret.z), z1z1), hh);
  return ret;
}
 
template<typename G>
static __global__ void bellman_multiexp(
  affine<G> *bases,
  projective<G> *buckets,
  projective<G> *results,
  Fr *exps,
  uint32_t n,
  uint32_t num_groups,
  uint32_t num_windows,
  uint32_t window_size) {
  // We have `num_windows` * `num_groups` threads per multiexp.
  const uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if(gid >= num_windows * num_groups) return;

  // We have (2^window_size - 1) buckets.
  const uint32_t bucket_len = ((1 << window_size) - 1);

  // Each thread has its own set of buckets in global memory.
  buckets += bucket_len * gid;

  for(uint32_t i = 0; i < bucket_len; i++) buckets[i] = projective<G>::Get_ZERO();

  const uint32_t len = (n + num_groups - 1) / num_groups; // Num of elements in each group

  // This thread runs the multiexp algorithm on elements from `nstart` to `nened`
  // on the window [`bits`, `bits` + `w`)
  const uint32_t nstart = len * (gid / num_windows);
  const uint32_t nend = min(nstart + len, n);
  const uint32_t bits = (gid % num_windows) * window_size;
  const uint16_t w = min((uint16_t)window_size, (uint16_t)((sizeof(Fr) * 8) - bits));

  projective<G> res;// = projective<G>::Get_ZERO();
  projective<G>::COPY_ZERO(&res);

  /*
  if (gid < 5) {
    printf("gid: %d, nstart = %d, nend = %d\n", gid, nstart, nend);
  }
  */

  // temporarily guard it 
  //assert(num_window == 32);
  for(uint32_t i = nstart; i < nend; i++) {
    uint32_t ind = get_bits<Fr>(exps[i], bits, w);
    if(ind--) buckets[ind] = add_mixed<G>(buckets[ind], bases[i]);
  }

  // Summation by parts
  // e.g. 3a + 2b + 1c = a +
  //                    (a) + b +
  //                    ((a) + b) + c
  projective<G> acc = projective<G>::Get_ZERO();
  for(int j = bucket_len - 1; j >= 0; j--) {
    acc = add<G>(acc, buckets[j]);
    res = add<G>(res, acc);
  }

  results[gid] = res;
}
#endif