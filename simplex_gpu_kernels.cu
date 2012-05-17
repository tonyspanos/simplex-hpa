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

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
   #define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif

float* get_array_from_file (string fileprefix, int *width, int *height, bool is_max);

//===========================================================================//
// normalize_kernel
//
// Scale each column of the pivot row by input "scale"
//
// @param  arr        - the 1D array representation of the matrix
// @param  scale      - arr[pivotRow * width + pivotColumn]
// @param  height     - the height of arr
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

  /*
  if (yId == 0) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        printf("%f, ", arr[INDEX(w,h)]);
      }
      printf ("\n");
    }
    printf ("\n");
  }
  */

}

//===========================================================================//
// row_reduce_kernel
//
// Perform row reduction on each row
//
// @param  arr        - the 1D array representation of the matrix
// @param  divider    - arr[INDEX(pivotColumn, PivotRow)]
// @param  height     - the height of arr
//===========================================================================//
__global__ void row_reduce_kernel (float* arr, int height, int width, int pivotRow, int pivotColumn, float divider) {

  // Indexing
  long xId = threadIdx.x + (blockIdx.x * blockDim.x); // Column
  long yId = threadIdx.y + (blockIdx.y * blockDim.y); // Row

  // Skip if not within size, or on pivotRow
  if ( (yId >= height) || (xId >= width) || (yId == pivotRow) ) {
    return;
  }
  
  // Find the scale (will be different for each row
  float scale = arr[INDEX(pivotColumn, yId)];

  // Row Reduction for each pixel
  arr[INDEX(xId, yId)] -= (arr[INDEX(xId, pivotRow)] * (scale * divider));

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
    cout << "Malloc Kernel failed: " << cudaGetErrorString(status) << endl;
    return -1;
  }

  // Copy the host input data to the device 
  status = cudaMemcpy(arr_d, arr, bytes, cudaMemcpyHostToDevice); 
  if (status != cudaSuccess) {
    cout << "Memcpy 1 Kernel failed: " << cudaGetErrorString(status) << endl;
    return -1;
  }

  int tile_size = 512;
  int tile_size_N = 256;
  long dim_size   = (long)ceil(size / (float)tile_size);
  long dim_size_N = (long)ceil(size / (float)tile_size_N);

  /*
  cout << "SIZE:        " << size << endl;
  cout << "TILE_SIZE_N: " << tile_size_N << endl;
  cout << "DIM SIZE:    " << dim_size << endl;
  cout << "DIM SIZE_N:  " << dim_size_N << endl;
  */

  // Row Reduction Dimensions (2D)
  dim3 dimGridRR(dim_size, dim_size);
  dim3 dimBlockRR(16, 16);

  // Normalize Dimensions (1D)
  dim3 dimGridN(1, dim_size_N);
  dim3 dimBlockN(1, tile_size_N);

  /////////////////////////////////////////////////
  // Repeat until the bottom row is all positive //
  /////////////////////////////////////////////////
  while (!is_indicator_positive_gpu (arr, width, height)) {
    // If number of iterations exceed the threshold, no solutions were found
    if (num_iterations > MAX_ITER) {
      return num_iterations;
    }

    DBGPRINT("Iteration " << num_iterations);

    // Do the gaussian elimination part
    pivotColumn = get_pivot_column_index_gpu (arr,width,height);
    pivotRow    = get_pivot_row_index_gpu (arr,width,height,pivotColumn);

	  DBGPRINT("Iteration " );
    // Normalization
    scale = arr[INDEX(pivotColumn, pivotRow)];
    normalize_kernel<<<dimGridN, dimBlockN>>>(arr_d, scale, height, width, pivotRow);
	  cudaThreadSynchronize();

    // Check for CUDA errors
    status = cudaGetLastError();
    if (status != cudaSuccess) {
      cout << "Normalize Kernel failed: " << cudaGetErrorString(status) << endl;
      return -1;
    }

	  DBGPRINT("Before row reduction ");
    // Do the division outside of the kernel
    float divider = 1 / arr[INDEX(pivotColumn, pivotRow)];

    // Row reduction
    row_reduce_kernel<<<dimGridRR, dimBlockRR>>>(arr_d, height, width, pivotRow, pivotColumn, divider);
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
	  // Copy the host input data to the device 
	  status = cudaMemcpy(arr, arr_d, bytes, cudaMemcpyDeviceToHost); 
    if (status != cudaSuccess) {
      cout << "Memcpy 2 Kernel failed: " << cudaGetErrorString(status) << endl;
      return -1;
    }
  }

  // Free to prevent out-of-memory error
  cudaFree (arr_d);

  //print_matrix_gpu(arr,width,height);

  return num_iterations;

}

// Page locked memory
int simplex_gpu_page_locked_mem (float *tgpu) { // (float *arr, int width, int height) {

  int height, width;
  float * arr  = get_array_from_file ("cody_1000", &width, &height, 0);

  DBGPRINT("Simplex GPU function entered ");
  // A status flag
  cudaError_t status;

  int num_iterations = 0;
  float scale;
  int pivotRow, pivotColumn;

  long size = width * height;

  // Number of bytes in the matrix. 
  int bytes = size * sizeof(float); 
  // Pointers to the device arrays 
  float *arr_d, *arr_h; 

  // Allocate memory on the device to store each matrix 
  status = cudaHostAlloc( (void**) &arr_h, bytes, cudaHostAllocMapped | cudaHostAllocWriteCombined);
  if (status != cudaSuccess) {
    cout << "Malloc Kernel failed: " << cudaGetErrorString(status) << endl;
    //cudaFree (arr_d);
    return -1;
  }

  // Fill the memory host with data
  /*
  for (int i = 0; i < size; i++) {
    arr_h[i] = arr[i];
  }
  */
  memcpy (arr_h, arr, bytes);

  // Get the device pointer
  status = cudaHostGetDevicePointer( &arr_d, arr_h, 0 );
  if (status != cudaSuccess) {
    cout << "Pointer Kernel failed: " << cudaGetErrorString(status) << endl;
    cudaFree (arr_d);
    return -1;
  }

  int tile_size = 512;
  int tile_size_N = 256;
  long dim_size   = (long)ceil(size / (float)tile_size);
  long dim_size_N = (long)ceil(size / (float)tile_size_N);

  /*
  cout << "SIZE:        " << size << endl;
  cout << "TILE_SIZE_N: " << tile_size_N << endl;
  cout << "DIM SIZE:    " << dim_size << endl;
  cout << "DIM SIZE_N:  " << dim_size_N << endl;
  */

  // Row Reduction Dimensions (2D)
  dim3 dimGridRR(dim_size, dim_size);
  dim3 dimBlockRR(16, 16);

  // Normalize Dimensions (1D)
  dim3 dimGridN(1, dim_size_N);
  dim3 dimBlockN(1, tile_size_N);

  // Variables to capture the runtime information
  //float tgpu;
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

	  DBGPRINT("Iteration " );
    // Normalization
    scale = arr_h[INDEX(pivotColumn, pivotRow)];
    normalize_kernel<<<dimGridN, dimBlockN>>>(arr_d, scale, height, width, pivotRow);
	  cudaThreadSynchronize();

    // Check for CUDA errors
    status = cudaGetLastError();
    if (status != cudaSuccess) {
      cout << "Normalize Kernel failed: " << cudaGetErrorString(status) << endl;
      cudaFree (arr_d);
      return -1;
    }

	  DBGPRINT("Before row reduction ");
    // Do the division outside of the kernel
    float divider = 1 / arr_h[INDEX(pivotColumn, pivotRow)];

    // Row reduction
    row_reduce_kernel<<<dimGridRR, dimBlockRR>>>(arr_d, height, width, pivotRow, pivotColumn, divider);
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
     // print_matrix_gpu (arr, width, height);
    #endif
 
  }
  end = clock();

  // Fill the memory host with data
  /*for (int i = 0; i < size; i++) {
    arr[i] = arr_h[i];
  }*/
  memcpy (arr, arr_h, bytes);

  //print_matrix_gpu(arr,width,height);

  // Free to prevent out-of-memory error
  cudaFreeHost (arr_h);

  //print_matrix_gpu(arr,width,height);

  *tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC;

  // Determine whether a solution was found
  //cout << "The GPU took " << *tgpu << " ms, Z = " << arr[width*height-1] << endl;

  return num_iterations;

}
