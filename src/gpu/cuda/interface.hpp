#ifndef INTERFACE_HEADER
#define INTERFACE_HEADER

#include "fields/field_structs.hpp"

typedef enum
{
    Compute_Ok,
    Init_Error, // any errors unrelated to computing
    Compute_Error
} State;

typedef struct
{
    uint32_t device_id;
    // TODO: more
} CudaInfo;

// These are parameters for use
template<typename T>
struct InputParameters {
    projective<T> *results;
    affine<T> *bases;
    Fr *exps;
    uint32_t n;
    uint32_t num_groups;
    uint32_t num_windows;
    uint32_t window_size;
    uint32_t core_count;

    CudaInfo cuda_info;
};

struct FFTInputParameters {
    Fr* x;
    Fr* pq;
    Fr* omegas;
    uint32_t n;
    uint32_t lgn;
    uint32_t max_deg;

    CudaInfo cuda_info;
};

// Type definitions
typedef Fq  G1;
typedef Fq2 G2;

typedef InputParameters<G1> G1InputParameters;
typedef InputParameters<G2> G2InputParameters;

#ifdef __cplusplus
extern "C" {
#endif
State G1_multiexp_cuda(G1InputParameters p);
#ifdef __cplusplus
}
#endif
#ifdef __cplusplus
extern "C" {
#endif
State G2_multiexp_cuda(G2InputParameters p);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
uint64_t G1_multiexp_chunk_size(G1InputParameters p);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
uint64_t G2_multiexp_chunk_size(G2InputParameters p);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
State Fr_radix_fft(FFTInputParameters p);
#ifdef __cplusplus
}
#endif

// #ifdef __cplusplus
// extern "C" {
// #endif
// State Fq_radix_fft(FqFFTInputParameters p);
// #ifdef __cplusplus
// }
// #endif
// #ifdef __cplusplus
// extern "C" {
// #endif
// State Fq2_radix_fft(Fq2FFTInputParameters p);
// #ifdef __cplusplus
// }
// #endif

#endif
