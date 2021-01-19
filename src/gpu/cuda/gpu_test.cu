#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

#include "gpu_support.h"
#include "fields/multiexp.cuh"
#include "fields/fft.cuh"
#include "interface.hpp"

template<typename T>
static State multiexp_cuda(InputParameters<T> p) {
    // initialize cuda
    cudaSetDevice(p.cuda_info.device_id);
    CUcontext context;
    cuCtxCreate(&context, 0, p.cuda_info.device_id);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    affine<T> *d_bases;
    projective<T> *d_buckets;
    projective<T> *d_results;
    Fr *d_exps;

    size_t bases_size = sizeof(affine<T>) * p.n;
    size_t bucket_len = 1 << p.window_size;
    size_t buckets_size = sizeof(projective<T>) * (2 * p.core_count * bucket_len);
    size_t results_size = sizeof(projective<T>) * (2 * p.core_count);
    size_t exps_size = sizeof(Fr) * p.n;

    CUDA_CHECK(cudaMalloc((void**)&d_bases, bases_size));
    CUDA_CHECK(cudaMalloc((void**)&d_buckets, buckets_size));
    CUDA_CHECK(cudaMalloc((void**)&d_results, results_size));
    CUDA_CHECK(cudaMalloc((void**)&d_exps, exps_size));

    CUDA_CHECK(cudaMemcpyAsync(d_bases, p.bases, bases_size, cudaMemcpyHostToDevice));    
    CUDA_CHECK(cudaMemcpyAsync(d_exps, p.exps, exps_size, cudaMemcpyHostToDevice));    
    instantiate_constants();
    CUDA_CHECK(cudaDeviceSynchronize());

    {
        printf("n: %d, num_groups: %d, num_windows: %d, window_size: %d\n",
                p.n, p.num_groups, p.num_windows, p.window_size);
        printf("core_count = %d\n", p.core_count);
    }

    cudaEventRecord(start, 0);

    // Call cuda
    constexpr uint32_t threadsPerBlock = 512; 
    //uint32_t blocksPerGrid = (2 * p.core_count + threadsPerBlock - 1) / threadsPerBlock;
    uint32_t blocksPerGrid = (uint32_t)ceil((2 * p.core_count) / threadsPerBlock);
    bellman_multiexp<T><<<blocksPerGrid, threadsPerBlock>>>
                (d_bases,
                 d_buckets,
                 d_results,
                 d_exps,
                 p.n,
                 p.num_groups,
                 p.num_windows,
                 p.window_size);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed=0;
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("Multiexp Kernel time: %fms\n", elapsed);


    CUDA_CHECK(cudaMemcpyAsync(p.results, d_results, results_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_bases));
    CUDA_CHECK(cudaFree(d_buckets));
    CUDA_CHECK(cudaFree(d_exps));


    CUDA_CHECK(cudaFree(d_results));

    cuCtxDestroy(context);
    return Compute_Ok;
} 

State radix_fft_cuda(FFTInputParameters p) {
    // initialize cuda
    cudaSetDevice(p.cuda_info.device_id);
    CUcontext context;
    cuCtxCreate(&context, 0, p.cuda_info.device_id);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // allocate 
    Fr *d_x;
    Fr *d_y;
    Fr *d_pq;
    Fr *d_omegas;

    size_t x_size = sizeof(Fr) * p.n;
    size_t y_size = sizeof(Fr) * p.n;
    size_t pq_size = sizeof(Fr) * (1 << p.max_deg >> 1);
    size_t omegas_size = sizeof(Fr) * 32;

    {
        //printf("CUDA parameters:\n");
        //printf("pq_size: %d, Fr size: %d, omegas_size: %d\n", pq_size, sizeof(Fr), omegas_size);
        //printf("threads perblock: %d, blocks per grid: %d\n", threadsPerBlock, blocksPerGrid);
        //printf("x/y_size: %d, n: %d\n", x_size, p.n);
        //printf("deg: %d, max_deg %d, lgp: %d, u_size %d\n", p.deg, p.max_deg, p.lgp, u_size);
    }

    CUDA_CHECK(cudaMalloc((void**)&d_x, x_size));
    CUDA_CHECK(cudaMalloc((void**)&d_y, y_size));
    CUDA_CHECK(cudaMalloc((void**)&d_pq, pq_size));
    CUDA_CHECK(cudaMalloc((void**)&d_omegas, omegas_size));

    CUDA_CHECK(cudaMemcpyAsync(d_x, p.x, x_size, cudaMemcpyHostToDevice));    
    CUDA_CHECK(cudaMemcpyAsync(d_pq, p.pq, pq_size, cudaMemcpyHostToDevice));    
    CUDA_CHECK(cudaMemcpyAsync(d_omegas, p.omegas, omegas_size, cudaMemcpyHostToDevice));    


    {
        //printf("CUDA FFT input parameters:\n");
        //printf("N: %ld\n", p.n);
        //printf("x/y size: %ld\n", x_size);
        //printf("pq size: %ld\n", pq_size);
        //printf("omega size: %ld\n", pq_size);
        //printf("u size: %ld\n", u_size);
        //printf("lgp: %d, deg: %d, max_deg: %d\n", p.lgp, p.deg, p.max_deg);
    }

    instantiate_constants();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(start, 0);

    uint log_p = 0;
    while (log_p < p.lgn) {
        uint deg = std::min(p.max_deg, p.lgn - log_p);
        uint32_t threadsPerBlock =
            1 << std::min(deg - 1, (uint32_t)MAX_LOG2_LOCAL_WORK_SIZE);
        uint32_t blocksPerGrid = (p.n >> deg);
        size_t u_size = sizeof(Fr) * (1 << deg);
        radix_fft<<<blocksPerGrid, threadsPerBlock, u_size>>>(
            d_x, d_y, d_pq, d_omegas, p.n, log_p, deg, p.max_deg);

        log_p += deg;
        Fr* tmp = d_x;
        d_x = d_y;
        d_y = tmp;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed=0;
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("FFT Kernel time: %fms\n", elapsed);

    CUDA_CHECK(cudaMemcpyAsync(p.x, d_x, x_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_pq));
    CUDA_CHECK(cudaFree(d_omegas));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_x));
    cuCtxDestroy(context);

    return Compute_Ok;
}

// extern "C" cannot use templates.
extern "C" {
State G1_multiexp_cuda(G1InputParameters p) {
    return multiexp_cuda<G1>(p);
}

State G2_multiexp_cuda(G2InputParameters p) {
    return multiexp_cuda<G2>(p);
}

State Fr_radix_fft(FFTInputParameters p) {
     return radix_fft_cuda(p);
}

}