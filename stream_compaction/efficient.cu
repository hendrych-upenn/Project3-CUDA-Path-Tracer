#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        using StreamCompaction::Common::kernMapToBoolean;
        using StreamCompaction::Common::kernScatter;

        const int maxBlockSize = 64;

        // assumes padding to block size;
        __global__ void kernBlockScan(int blockLog2Ceil, int* data) {
            int blockOffset = blockDim.x * blockIdx.x;
            int idx = threadIdx.x + 1;
            // upsweep
            for (int i = 1; i <= blockLog2Ceil; i++) {
                int pos = idx * (1 << i) - 1;
                if (pos < blockDim.x) {
                    pos += blockOffset;
                    int offset = 1 << (i - 1);
                    data[pos] = data[pos] + data[pos - offset];
                }
                __syncthreads();
            }
            // downsweep
            // set root to 0

            if (threadIdx.x == blockDim.x - 1) {
                data[blockOffset + blockDim.x - 1] = 0;
            }
            __syncthreads();

            for (int i = blockLog2Ceil; i > 0; i--) {
                int pos = idx * (1 << i) - 1;
                if (pos < blockDim.x) {
                    pos += blockOffset;
                    int offset = 1 << (i - 1);
                    int t = data[pos - offset];
                    data[pos - offset] = data[pos];
                    data[pos] = t + data[pos];
                }

                __syncthreads();
            }
        }

        // assumes padding to block size;
        __global__ void kernBlockScanStoreSum(int blockLog2Ceil, int* data, int* sums) {
            int blockOffset = blockDim.x * blockIdx.x;
            int idx = threadIdx.x + 1;
            // upsweep
            for (int i = 1; i <= blockLog2Ceil; i++) {
                int pos = idx * (1 << i) - 1;
                if (pos < blockDim.x) {
                    pos += blockOffset;
                    int offset = 1 << (i - 1);
                    data[pos] = data[pos] + data[pos - offset];
                }
                __syncthreads();
            }
            // store sum for cross block sum later
            if (threadIdx.x == 0) {
                sums[blockIdx.x] = data[blockOffset + blockDim.x - 1];
            }
            __syncthreads();
            // downsweep
            // set root to 0

            if (threadIdx.x == blockDim.x - 1) {
                data[blockOffset + blockDim.x - 1] = 0;
            }
            __syncthreads();

            for (int i = blockLog2Ceil; i > 0; i--) {
                int pos = idx * (1 << i) - 1;
                if (pos < blockDim.x) {
                    pos += blockOffset;
                    int offset = 1 << (i - 1);
                    int t = data[pos - offset];
                    data[pos - offset] = data[pos];
                    data[pos] = t + data[pos];
                }

                __syncthreads();
            }
        }

        // assumes padding to block size;
        __global__ void kernAddSums(int* data, const int* sums) {
            int idx = blockDim.x * (blockIdx.x) + threadIdx.x;
            data[idx] += sums[blockIdx.x];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {


            if (n <= 0) {
                return;
            }

            std::vector<int*> scanArrays{};
            std::vector<int> scanArrayLens{};

            // ceil to next maxBlockSize
            int scanArrayLen = divup(n, maxBlockSize) * maxBlockSize;
            while (scanArrayLen > maxBlockSize) {
                int* d_array;
                cudaMalloc(&d_array, scanArrayLen * sizeof(int));
                checkCUDAError("cudaMalloc scanArrayLen failed");
                cudaMemset(d_array, 0, scanArrayLen * sizeof(int));
                scanArrays.push_back(d_array);
                scanArrayLens.push_back(scanArrayLen);
                //fprintf(stderr, "Size %i\n", scanArrayLen);
                // divide by maxBlockSize then ceil to it
                scanArrayLen = divup(scanArrayLen / maxBlockSize, maxBlockSize) * maxBlockSize;
            }
            {
                // scanArrayLen = maxBlockSize now
                int* d_array;
                cudaMalloc(&d_array, scanArrayLen * sizeof(int));
                checkCUDAError("cudaMalloc scanArrayLen failed");
                cudaMemset(d_array, 0, scanArrayLen * sizeof(int));
                scanArrays.push_back(d_array);
                scanArrayLens.push_back(scanArrayLen);
            }

            cudaMemcpy(scanArrays[0], idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy idata failed");

            timer().startGpuTimer();

            for (int i = 0; i < scanArrays.size() - 1; i++) {
                int arrayLen = scanArrayLens[i];
                kernBlockScanStoreSum << < arrayLen / maxBlockSize, maxBlockSize >> > (ilog2ceil(maxBlockSize), scanArrays[i], scanArrays[i + 1]);
                checkCUDAError("kernBlockScanStoreSum failed");
            }
            kernBlockScan << <1, maxBlockSize >> > (ilog2ceil(maxBlockSize), scanArrays.back());
            checkCUDAError("kernBlockScan failed");

            for (int i = scanArrays.size() - 2; i >= 0; i--) {
                int arrayLen = scanArrayLens[i];
                kernAddSums << < arrayLen / maxBlockSize, maxBlockSize >> > (scanArrays[i], scanArrays[i + 1]);
                checkCUDAError("kernAddSums failed");
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, scanArrays[0], n * sizeof(int), cudaMemcpyDeviceToHost);
            //cudaMemcpy(odata, scanArrays.back(), scanArrayLens.back() * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy odata failed");

            for (int i = 0; i < scanArrays.size(); i++) {
                cudaFree(scanArrays[i]);
                checkCUDAError("cudaFree scanArrays[i] failed");
            }
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            if (n <= 0) {
                return -1;
            }

            int topArrayLen = divup(n, maxBlockSize) * maxBlockSize;

            int* d_idata;
            cudaMalloc(&d_idata, topArrayLen * sizeof(int));
            checkCUDAError("cudaMalloc d_idata failed");
            cudaMemset(d_idata, 0, topArrayLen * sizeof(int));
            checkCUDAError("cudaMemset d_idata failed");
            cudaMemcpy(d_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy d_idata failed");

            int* d_bools;
            cudaMalloc(&d_bools, topArrayLen * sizeof(int));
            checkCUDAError("cudaMalloc d_bools failed");


            int* d_odata;
            cudaMalloc(&d_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc d_odata failed");

            std::vector<int*> scanArrays{};
            std::vector<int> scanArrayLens{};

            // ceil to next maxBlockSize
            int scanArrayLen = topArrayLen;
            while (scanArrayLen > maxBlockSize) {
                int* d_array;
                cudaMalloc(&d_array, scanArrayLen * sizeof(int));
                checkCUDAError("kernMapToBoolean failed");
                cudaMemset(d_array, 0, scanArrayLen * sizeof(int));
                checkCUDAError("cudaMemset d_array failed");
                scanArrays.push_back(d_array);
                scanArrayLens.push_back(scanArrayLen);
                //fprintf(stderr, "Size %i\n", scanArrayLen);
                // divide by maxBlockSize then ceil to it
                scanArrayLen = divup(scanArrayLen / maxBlockSize, maxBlockSize) * maxBlockSize;
            }
            {
                // scanArrayLen = maxBlockSize now
                int* d_array;
                cudaMalloc(&d_array, scanArrayLen * sizeof(int));
                checkCUDAError("cudaMalloc scanArrayLen failed");
                cudaMemset(d_array, 0, scanArrayLen * sizeof(int));
                checkCUDAError("cudaMemset d_array failed");
                scanArrays.push_back(d_array);
                scanArrayLens.push_back(scanArrayLen);
            }

            timer().startGpuTimer();

            kernMapToBoolean << <topArrayLen / maxBlockSize, maxBlockSize >> > (topArrayLen, d_bools, d_idata);
            checkCUDAError("kernMapToBoolean failed");

            cudaMemcpy(scanArrays[0], d_bools, topArrayLen * sizeof(int), cudaMemcpyDeviceToDevice);
            checkCUDAError("cudaMemcpy d_bools failed");

            for (int i = 0; i < scanArrays.size() - 1; i++) {
                int arrayLen = scanArrayLens[i];
                kernBlockScanStoreSum << < arrayLen / maxBlockSize, maxBlockSize >> > (ilog2ceil(maxBlockSize), scanArrays[i], scanArrays[i + 1]);
                checkCUDAError("kernBlockScanStoreSum failed");
            }
            kernBlockScan << <1, maxBlockSize >> > (ilog2ceil(maxBlockSize), scanArrays.back());
            checkCUDAError("kernBlockScan failed");

            for (int i = scanArrays.size() - 2; i >= 0; i--) {
                int arrayLen = scanArrayLens[i];
                kernAddSums << < arrayLen / maxBlockSize, maxBlockSize >> > (scanArrays[i], scanArrays[i + 1]);
                checkCUDAError("kernAddSums failed");
            }

            int* d_indices = scanArrays[0];


            kernScatter << <topArrayLen / maxBlockSize, maxBlockSize >> > (n, d_odata, d_idata, d_bools, d_indices);
            checkCUDAError("kernScatter failed");

            timer().endGpuTimer();

            cudaDeviceSynchronize();

            int compactLen;
            cudaMemcpy(&compactLen, &d_indices[topArrayLen - 1], sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy compactLen failed");
            int lastBool;
            cudaMemcpy(&lastBool, &d_bools[topArrayLen - 1], sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy lastBool failed");
            compactLen += lastBool;

            cudaMemcpy(odata, d_odata, compactLen * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy d_odata failed");

            for (int i = 0; i < scanArrays.size(); i++) {
                cudaFree(scanArrays[i]);
                checkCUDAError("cudaFree scanArrays[i] failed");
            }
            cudaFree(d_bools);
            checkCUDAError("cudaFree d_bools failed");
            cudaFree(d_idata);
            checkCUDAError("cudaFree d_idata failed");
            cudaFree(d_odata);
            checkCUDAError("cudaFree d_odata failed");

            return compactLen;
        }
    }
}
