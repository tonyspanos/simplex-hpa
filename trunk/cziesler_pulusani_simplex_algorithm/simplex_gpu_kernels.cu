//===========================================================================//
// File Name:   simplex_gpu_kernels.cpp
// Authors:     Cody Cziesler, Praneeth Pulusani
//
// Description: The simplex algorithm implemented on a GPU
//
//===========================================================================//

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <ctime>
#include "stdio.h" 

#include "common_gpu.h"

// Hack to allow printf's to work in the kernels
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
   #define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif

// Defines so that we can call this function
float* get_array_from_file (string fileprefix, int *width, int *height, bool is_max);

//===========================================================================//
// normalize_kernel
//
// Scale each column of the pivot row by input "scale"
//
// @param  arr        - the 1D array representation of the matrix
// @param  scale      - arr[pivotRow * width + pivotColumn]
// @param  height     - the height of arr
// @param  width      - the width of arr
// @param  pivotRow   - the pivot row
//===========================================================================//
__global__ void normalize_kernel (float* arr, float scale, int height, int width, int pivotRow) {

  // Indexing
  long yId = threadIdx.y + (blockIdx.y * blockDim.y); // Row

  // Skip if not within size
  if (yId >= width) {
    return;
  }

  // Scale the row by "scale"
  arr[INDEX(yId, pivotRow)] /= scale;

}

//===========================================================================//
// row_reduce_kernel
//
// Perform row reduction on each row
//
// @param  arr         - the 1D array representation of the matrix
// @param  height      - the height of arr
// @param  width       - the width of arr
// @param  pivotRow    - the pivot row
// @param  pivotColumn - the pivot column
// @param  divider     - arr[INDEX(pivotColumn, PivotRow)]
//===========================================================================//
__global__ void row_reduce_kernel (float* arr, int height, int width, int pivotRow, int pivotColumn) { //, float divider) {

  // Indexing
  long xId = threadIdx.x + (blockIdx.x * blockDim.x); // Column
  long yId = threadIdx.y + (blockIdx.y * blockDim.y); // Row

  // Skip if not within size, or on pivotRow
  if ( (yId >= height) || (xId >= width) || (yId == pivotRow) ) {
    return;
  }

  // Row Reduction for each pixel
  arr[INDEX(xId, yId)] = arr[INDEX(xId, yId)] - (arr[INDEX(xId, pivotRow)] * 
    (arr[INDEX(pivotColumn, yId)] / arr[INDEX(pivotColumn, pivotRow)]));
}

//===========================================================================//
// simplex_gpu
//
// Puts the whole Simplex Algorithm together
//
// @param  arr        - the 1D array representation of the matrix
// @param  width      - the width of arr
// @param  height     - the height of arr
//
// @return num_iterations - the number of iterations it took to complete
//===========================================================================//
int simplex_gpu (float *arr, int width, int height) {

  DBGPRINT("Simplex GPU function entered ");

  // A status flag
  cudaError_t status;

  // Variables
  int num_iterations = 0;
  float scale;
  int pivotRow, pivotColumn;

  long size = width * height;

  // Number of bytes in the matrix. 
  int bytes = size * sizeof(float); 

  // Pointers to the device arrays 
  float *arr_d; 

  // Allocate memory on the device to store each matrix 
  status = cudaMalloc((void**) &arr_d, bytes); 
  if (status != cudaSuccess) {
    cout << "Malloc failed: " << cudaGetErrorString(status) << endl;
    return -1;
  }

  // Copy the host input data to the device 
  status = cudaMemcpy(arr_d, arr, bytes, cudaMemcpyHostToDevice); 
  if (status != cudaSuccess) {
    cout << "Memcpy 1 failed: " << cudaGetErrorString(status) << endl;
    return -1;
  }

  // Block and Grid dimensions
  int size_side_RR = 32;
  int tile_size_N  = 512;
  long dim_size_RR = (long)ceil(size / (float)(size_side_RR*size_side_RR));
  long dim_size_N  = (long)ceil(size / (float)tile_size_N);

  // Row Reduction Dimensions (2D)
  dim3 dimGridRR(dim_size_RR, dim_size_RR);
  dim3 dimBlockRR(size_side_RR, size_side_RR);

  // Normalize Dimensions (1D)
  dim3 dimGridN(1, dim_size_N);
  dim3 dimBlockN(1, tile_size_N);

  // Variables to capture the runtime information
  clock_t start, end; //, norm_k_start, norm_k_end, row_k_start, row_k_end, p_start, p_end;
  float total_time = 0;
  
  /////////////////////////////////////////////////
  // Repeat until the bottom row is all positive //
  /////////////////////////////////////////////////
  while (!is_indicator_positive_gpu (arr, width, height)) {

    start = clock();
    // If number of iterations exceed the threshold, no solutions were found
    if (num_iterations > MAX_ITER) {
      return num_iterations;
    }

    DBGPRINT("Iteration " << num_iterations);

    // Do the gaussian elimination part

    pivotColumn = get_pivot_column_index_gpu (arr,width,height);
    pivotRow    = get_pivot_row_index_gpu (arr,width,height,pivotColumn);

    // Normalization
    scale = arr[INDEX(pivotColumn, pivotRow)];
    DBGPRINT("Before normalize kernel");
    normalize_kernel<<<dimGridN, dimBlockN>>>(arr_d, scale, height, width, pivotRow);
    cudaThreadSynchronize();


    // Check for CUDA errors
    status = cudaGetLastError();
    if (status != cudaSuccess) {
      cout << "Normalize Kernel failed: " << cudaGetErrorString(status) << endl;
      return -1;
    }

    // Do the division outside of the kernel
    float divider = arr[INDEX(pivotColumn, pivotRow)];

    // Row reduction
    DBGPRINT("Before row reduction kernel");

    row_reduce_kernel<<<dimGridRR, dimBlockRR>>>(arr_d, height, width, pivotRow, pivotColumn); //, divider);
    // Synch here before the mem copy
    cudaThreadSynchronize();

    // Check for CUDA errors
    status = cudaGetLastError();
    if (status != cudaSuccess) {
      cout << "Row Reduce Kernel failed: " << cudaGetErrorString(status) << endl;
      return -1;
    }

    // Increment the number of iterations
    num_iterations++;

    #ifdef DEBUG
     // print_matrix_gpu (arr, width, height);
    #endif

    end = clock();
    total_time += (end-start);

    // Copy the host input data to the device 
    status = cudaMemcpy(arr, arr_d, bytes, cudaMemcpyDeviceToHost); 
    if (status != cudaSuccess) {
      cout << "Memcpy 2 failed: " << cudaGetErrorString(status) << endl;
      return -1;
    }
  }

  // Free to prevent out-of-memory error
  cudaFree (arr_d);

  cout << "GPU Computation time: " << (float)total_time * 1000 / (float)CLOCKS_PER_SEC << " ms\n";

  return num_iterations;

}

//===========================================================================//
// simplex_gpu_zero_copy
//
// Puts the whole Simplex Algorithm together, uses zero_copy memory
//
// Algorithm for zero-copy:
//   cudaHostAlloc host array
//   initialize host array
//   cudaHostGetDevicePointer (&device, host, 0)
//   run kernel with device array
//   synchronize threads
//   save answer array from host array
//   cudaFreeHost host array
//
//
// @param  *tgpu          - a pointer to a float variable to return the 
//                          execution time
//
// @return num_iterations - the number of iterations it took to complete
//===========================================================================//
int simplex_gpu_zero_copy (float *tgpu) {

  DBGPRINT("Simplex GPU function entered ");

  // A status flag
  cudaError_t status;

  // Check to see if the device supports zero-copy
  cudaDeviceProp prop;
  int whichDevice;
  status = cudaGetDevice (&whichDevice);
  if (status != cudaSuccess) {
    cout << "Get Device failed: " << cudaGetErrorString(status) << endl;
    return -1;
  }
  status = cudaGetDeviceProperties (&prop, whichDevice);
  if (status != cudaSuccess) {
    cout << "Get Device Properties failed: " << cudaGetErrorString(status) << endl;
    return -1;
  }
  if (prop.canMapHostMemory != 1) {
    printf ("Device cannot map memory\n");
    return -1;
  }

  // Get the height, width, and matrix
  int height, width;
  float * arr = get_array_from_file ("cody_1000", &width, &height, 0);

  // Variables
  int num_iterations = 0;
  float scale;
  int pivotRow, pivotColumn;
  long size = width * height;

  // Number of bytes in the matrix. 
  int bytes = size * sizeof(float); 

  // Pointers to the device arrays 
  float *arr_d, *arr_h; 

  // Allocate memory on the host to store each matrix 
  status = cudaHostAlloc( (void**) &arr_h, bytes, cudaHostAllocMapped);
  if (status != cudaSuccess) {
    cout << "Malloc failed: " << cudaGetErrorString(status) << endl;
    return -1;
  }

  // Get the device pointer
  status = cudaHostGetDevicePointer( &arr_d, arr_h, 0 );
  if (status != cudaSuccess) {
    cout << "Pointer failed: " << cudaGetErrorString(status) << endl;
    cudaFree (arr_d);
    return -1;
  }

  // Copy the array into the page-locked host array
  memcpy (arr_h, arr, bytes);

  // Block and Grid dimensions
  int size_side_RR = 32;
  int tile_size_N  = 256;
  long dim_size_RR = (long)ceil(size / (float)(size_side_RR*size_side_RR));
  long dim_size_N  = (long)ceil(size / (float)tile_size_N);

  // Row Reduction Dimensions (2D)
  dim3 dimGridRR(dim_size_RR, dim_size_RR);
  dim3 dimBlockRR(size_side_RR, size_side_RR);

  // Normalize Dimensions (1D)
  dim3 dimGridN(1, dim_size_N);
  dim3 dimBlockN(1, tile_size_N);

  // Variables to capture the runtime information
  clock_t start, end;

  start = clock();
  /////////////////////////////////////////////////
  // Repeat until the bottom row is all positive //
  /////////////////////////////////////////////////
  while (!is_indicator_positive_gpu (arr_h, width, height)) {

    // If number of iterations exceed the threshold, no solutions were found
    if (num_iterations > MAX_ITER) {
      cudaFree (arr_d);
      return num_iterations;
    }

    DBGPRINT("Iteration " << num_iterations);

    // Do the gaussian elimination part
    pivotColumn = get_pivot_column_index_gpu (arr_h,width,height);
    pivotRow    = get_pivot_row_index_gpu (arr_h,width,height,pivotColumn);

    scale = arr_h[INDEX(pivotColumn, pivotRow)];

    // Normalization
    DBGPRINT("Before normalize kernel");
    normalize_kernel<<<dimGridN, dimBlockN>>>(arr_d, scale, height, width, pivotRow);
    cudaThreadSynchronize();

    // Check for CUDA errors
    status = cudaGetLastError();
    if (status != cudaSuccess) {
      cout << "Normalize failed: " << cudaGetErrorString(status) << endl;
      cudaFree (arr_d);
      return -1;
    }

    // Do the division outside of the kernel
    float divider = arr_h[INDEX(pivotColumn, pivotRow)];

    // Row reduction
    DBGPRINT("Before row reduction kernel");
    row_reduce_kernel<<<dimGridRR, dimBlockRR>>>(arr_d, height, width, pivotRow, pivotColumn); //, divider);
    cudaThreadSynchronize();

    // Check for CUDA errors
    status = cudaGetLastError();
    if (status != cudaSuccess) {
      cout << "Row Reduce Kernel failed: " << cudaGetErrorString(status) << endl;
      cudaFree (arr_d);
      return -1;
    }

    // Increment the number of iterations
    num_iterations++;
    #ifdef DEBUG
      print_matrix_gpu (arr_h, width, height);
    #endif

  }
  end = clock();

  // Get the information from arr_h
  cout << "Z: " << arr_h[width*height-1] << "\n";
  cout << "The solution took " << num_iterations << " iterations\n";

  // Free to prevent out-of-memory error
  cudaFreeHost (arr_h);

  // Save the execution time
  *tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC;

  // Return the number of iterations
  return num_iterations;

}

///////////////////////////////////////////////////////////////////////////////
// Cody's temp workspace for testing
//
// This is testing the zero-copy memory. Adaoted from CUDA by Example (2010)
//
// Unused in the Simplex Algorithm, but used to test out zero-copy memory
//
///////////////////////////////////////////////////////////////////////////////
const int N = 33*1024*1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = min(32, (N+threadsPerBlock-1)/threadsPerBlock);

__global__ void dot (int size, float *a, float *b, float *c) {
  __shared__ float cache[threadsPerBlock];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int cacheIndex = threadIdx.x;
  float temp = 0;
  while (tid < size) {
    temp += a[tid] * b[tid];
    tid += blockDim.x * gridDim.x;
  }

  cache[cacheIndex] = temp;

  __syncthreads();

  int i = blockDim.x/2;
  while (i != 0) {
    if (cacheIndex < i) {
      cache[cacheIndex] += cache[cacheIndex + i];
    }
    __syncthreads();
    i /= 2;
  }

  if (cacheIndex == 0) {
    c[blockIdx.x] = cache[0];
  }

}

float cuda_test (int size) {
  float *a, *b, c, *partial_c;
  float *dev_a, *dev_b, *dev_partial_c;

  cudaHostAlloc ((void**) &a, size*sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc ((void**) &b, size*sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped);
  cudaHostAlloc ((void**) &partial_c, blocksPerGrid*sizeof(float), cudaHostAllocMapped);

  for (int i = 0; i < size; i++) {
    a[i] = (float)i;
    b[i] = (float)i*2;
  }

  cudaHostGetDevicePointer (&dev_a, a, 0);
  cudaHostGetDevicePointer (&dev_b, b, 0);
  cudaHostGetDevicePointer (&dev_partial_c, partial_c, 0);

  dot<<<blocksPerGrid, threadsPerBlock>>>(size, dev_a, dev_b, dev_partial_c);

  cudaThreadSynchronize();

  c = 0;
  for (int i = 0; i < blocksPerGrid; i++) {
    c += partial_c[i];
  }

  cudaFreeHost(a);
  cudaFreeHost(b);
  cudaFreeHost(partial_c);

  printf ("Value: %f\n", c);

  return 0;
}
