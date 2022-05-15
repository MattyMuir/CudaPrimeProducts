#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include <primesieve.hpp>
#include "Timer.h"
#include "minilibdiv.h"
#include "util.h"

#define VERIFIABLE 0

#define CCATCH(expr) cudaStatus = expr; if (cudaStatus != cudaSuccess) { fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(cudaStatus)); throw; }
#define TILEHEIGHT 512

__device__ int Min(int a, int b)
{
    return (a < b) ? a : b;
}

__device__ uint64_t FastMod(uint64_t x, const Divider& base)
{
    return x - base.d * Divide(x, base);
}

__global__ void Kernel(int start, const uint64_t* primes, uint64_t* remainders)
{
    int id = threadIdx.x;
    int bNumThreads = blockDim.x;

    int batchStart = start + blockIdx.x * bNumThreads;
    int end = batchStart + bNumThreads - 1;

    // Value of n for current thread
    int n = batchStart + id;

    // Number of tiles vertically = ceil(end / TILEHEIGHT)
    int yTileNum = (end + TILEHEIGHT - 1) / TILEHEIGHT;

    __shared__ uint64_t shPrimes[TILEHEIGHT / 2];

    uint64_t prod = 1;
    Divider modDiv = GenDiv(primes[n]);
    for (int ty = 0; ty < yTileNum; ty++)
    {
        // Perform first layer of multiplication and store results in shPrimes
        for (int i = id; i < TILEHEIGHT / 2; i += bNumThreads)
        {
            int primesIndex = ty * TILEHEIGHT + i * 2;
            shPrimes[i] = primes[primesIndex] * primes[primesIndex + 1];
        }

        // Sync threads to ensure shared memory is complete before proceeding
        __syncthreads();

        // Check if thread is active for current tile
        if (n > ty * TILEHEIGHT)
        {
            // Tile will only be partially completed
            if (n - ty * TILEHEIGHT <= TILEHEIGHT)
            {
                int iThresh = Min((n - ty * TILEHEIGHT) / 2, TILEHEIGHT / 2);

                for (int i = 0; i < iThresh; i++)
                {
                    prod *= FastMod(shPrimes[i], modDiv);
                    prod = FastMod(prod, modDiv);
                }

                if (n & 1)
                    prod *= primes[n - 1];
                prod = FastMod(prod, modDiv);

                // Now that the partial tile is complete, the whole prod is
                // Check final remainder
                if (prod == modDiv.d - 1)
                    printf("%d\n", n);

#if VERIFIABLE
                remainders[n - start] = prod;
#endif
            }
            else // Tile will be fully completed
            {
                for (int i = 0; i < TILEHEIGHT / 2; i++)
                {
                    prod *= FastMod(shPrimes[i], modDiv);
                    prod = FastMod(prod, modDiv);
                }
            }
        }
        // Sync threads to ensure all are finished before changing shared memory
        __syncthreads();
    }
}

int main()
{
    cudaError_t cudaStatus;

    int start = IntInput("Start: ");
    int end = IntInput("End: ");

    // ===== Prepare Kernel Parameters =====
    int th = 1024;
    int b = (end - start + th - 1) / th;

    // Round 'end' to nearest multiple of 'th'
    end = start + th * b;
    int size = end + 1;

    // ===== Generate Primes =====
    std::vector<uint64_t> primes;

    TIMER(genprimes);
    primesieve::generate_n_primes(size, &primes);
    STOP_LOG(genprimes);

    int devNum = 0;
    cudaGetDeviceCount(&devNum);
    std::cout << "Devices: " << devNum << '\n';

    uint64_t* devPrimes = nullptr;

    size_t bufSize = size * sizeof(uint64_t);
    CCATCH(cudaMalloc(&devPrimes, bufSize));
    CCATCH(cudaMemcpy(devPrimes, primes.data(), bufSize, cudaMemcpyHostToDevice));

    // ===== Remainders =====
    uint64_t* remainders = nullptr;
#if VERIFIABLE
    CCATCH(cudaMallocManaged(&remainders, size * sizeof(*remainders)));
    CCATCH(cudaMemset(remainders, 0, size * sizeof(*remainders)));
#endif

    // ===== Running Kernel =====
    std::cout << "Running kernel...\n";
    freopen("log.log", "w", stdout);
    std::cout << "=============\n" << std::flush;
    TIMER(kernel);

    Kernel<<< b, th >>>(start, devPrimes, remainders);
    CCATCH(cudaGetLastError());
    CCATCH(cudaDeviceSynchronize());

    std::cout << "=============\n";
    STOP_LOG(kernel);
    std::cout << "Checked: (" << start << ", " << end << ")\n";

#if VERIFIABLE
    SaveRemainders(start, end, remainders, th * b);
#endif

    // ===== Free Memory =====
    cudaFree(devPrimes);
#if VERIFIABLE
    cudaFree(remainders);
#endif
}