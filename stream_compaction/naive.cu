#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void inclusiveScanAdd(const int n, const int stride, const int* idata, int* odata) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) {
                return;
            }
            if (stride > idx) {
                odata[idx] = idata[idx];
            }
            else 
            {
                odata[idx] = idata[idx] + idata[idx - stride];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            size_t sizeInBytes = n * sizeof(int);

            int* d_odata;
            int* d_idata;
            cudaMalloc(&d_odata, sizeInBytes);
            checkCUDAError("cudaMalloc d_odata failed");
            cudaMalloc(&d_idata, sizeInBytes);
            checkCUDAError("cudaMalloc d_idata failed");
            
            cudaMemcpy(d_idata, idata, sizeInBytes, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy d_idata failed");

            int blockSize = 1024;
            int gridSize = divup(n, blockSize);

            timer().startGpuTimer();

            for (int i = 0; i < ilog2ceil(n); i++) {
                int stride = 1 << i;
                inclusiveScanAdd << <gridSize, blockSize >> > (n, stride, d_idata, d_odata);
                checkCUDAError("inclusiveScanAdd failed");
                std::swap(d_odata, d_idata);
            }

            timer().endGpuTimer();

            // due to the fact that we want to make the inclusive scan exclusive instead
            odata[0] = 0;
            cudaMemcpy(odata + 1, d_idata, sizeInBytes - sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy odata failed");

            cudaFree(d_odata);
            checkCUDAError("cudaFree d_odata failed");
            cudaFree(d_idata);
            checkCUDAError("cudaFree d_idata failed");

        }
    }
}
