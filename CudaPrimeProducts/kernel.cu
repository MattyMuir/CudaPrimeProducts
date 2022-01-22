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

#define VERIFIABLE 0

#define CCATCH(expr) cudaStatus = expr; if (cudaStatus != cudaSuccess) { fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(cudaStatus)); throw; }
#define CLOG(expr) if (blockIdx.x == 0 && threadIdx.x == 429) { expr; }
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

    start += blockIdx.x * bNumThreads;
    int end = start + bNumThreads - 1;

    __shared__ uint64_t shPrimes[TILEHEIGHT / 2];

    // Number of tiles vertically = ceil(end / TILEHEIGHT)
    int yTileNum = (end + TILEHEIGHT - 1) / TILEHEIGHT;

    // Value of n for current thread
    int n = start + id;
    uint64_t prod = 1;

    uint64_t modPrime = primes[n];
    Divider modDiv = GenDiv(modPrime);

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
            int iThresh = Min((n - ty * TILEHEIGHT) / 2, TILEHEIGHT / 2);
            int i;
            for (i = 0; i < iThresh; i++)
            {
                prod *= FastMod(shPrimes[i], modDiv);
                prod = FastMod(prod, modDiv);
            }

            if (i < TILEHEIGHT / 2 || n == ty * TILEHEIGHT + TILEHEIGHT) // Tile was only partially completed, do final multiplication and check value
            {
                if (n & 1 == 1)
                    prod *= primes[n - 1];

                prod %= modPrime;
                if (prod == modPrime - 1)
                    printf("%d\n", n);

#if VERIFIABLE == 1
                remainders[n] = (prod + 1) % modPrime;
#endif
            }
        }
        // Sync threads to ensure all are finished before changing shared memory
        __syncthreads();
    }
}

void SaveRemainders(std::string path, uint64_t* remainders, int num)
{
    std::cout << "Saving...\n";
    std::ofstream file;
    file.open(path);

    for (int i = 0; i < num; i++)
        file << remainders[i] << "\n";

    file.close();
    std::cout << "Saved.\n";
}

int main()
{
    cudaError_t cudaStatus;

    int start = 1;
    int end = 1e6;
    int size = end + 1;

    // ===== Generate Primes =====
    std::vector<uint64_t> primes;

    primesieve::generate_n_primes(size, &primes);

    uint64_t* devPrimes = nullptr;

    cudaMalloc(&devPrimes, size * sizeof(*devPrimes));
    cudaMemcpy(devPrimes, &(primes[0]), size * sizeof(primes[0]), cudaMemcpyHostToDevice);

    // ===== Remainders =====
    uint64_t* remainders = nullptr;
#if VERIFIABLE == 1
    cudaMallocManaged(&remainders, size * sizeof(*remainders));
    cudaMemset(remainders, 0, size * sizeof(*remainders));
#endif

    // ===== Run Kernel =====
    int th = 1024;
    int b = (end - start) / th;

    std::cout << "Running kernel...\n"; Timer t;

    Kernel << < b, th >> > (start, devPrimes, remainders);
    CCATCH(cudaGetLastError());
    CCATCH(cudaDeviceSynchronize());

    t.Stop(); std::cout << "Finished, kernel took: " << t.duration * 0.001 << "ms\n";

#if VERIFIABLE == 1
    SaveRemainders("C:\\Users\\matty\\Desktop\\rems.txt", remainders, th * b);
#endif

    // ===== Free Memory =====
    cudaFree(devPrimes);
#if VERIFIABLE == 1
    cudaFree(remainders);
#endif
}