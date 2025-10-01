#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int sum = 0;
            for (int i = 0; i < n; i++) {
                odata[i] = sum;
                sum += idata[i];
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int index = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i])
                {
                    odata[index++] = idata[i];
                }
            }
            timer().endCpuTimer();
            return index;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            int* bool_array = new int[n];
            int* scanned_array = new int[n];
            timer().startCpuTimer();
            // TODO
            for (int i = 0; i < n; i++) {
                bool_array[i] = idata[i] == 0 ? 0 : 1;
            }
            int sum = 0;
            for (int i = 0; i < n; i++) {
                scanned_array[i] = sum;
                sum += bool_array[i];
            }
            for (int i = 0; i < n; i++) {
                if (bool_array[i] == 1) {
                    odata[scanned_array[i]] = idata[i];
                }
            }
            int length = scanned_array[n - 1] + bool_array[n - 1];
            timer().endCpuTimer();
            delete [] bool_array;
            delete [] scanned_array;
            return length;
        }
    }
}
