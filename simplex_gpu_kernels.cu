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
#include "stdio.h" 

#include "common.h"

//===========================================================================//
// normalize_kernel
//
// Scale each column of the pivot row by input "scale"
//
// @param  arr        - the 1D array representation of the matrix
// @param  scale      - arr[pivotRow * width + pivotColumn]
// @param  height     - the height of arr
//===========================================================================//
__global__ void normalize_kernel (float* arr, float scale, int height) {

  // Indexing
  int yId = threadIdx.y + (blockIdx.y * blockDim.y); // Row

  // Skip if not within size
  if (yId >= height) {
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
// @param  arr        - the 1D array representation of the matrix
// @param  divider    - arr[INDEX(pivotColumn, PivotRow)]
// @param  height     - the height of arr
//===========================================================================//
__global__ void row_reduce_kernel (float* arr, float divider, int height, int width, int pivotRow, int pivotColumn) {

  // Indexing
  int xId = threadIdx.x + (blockIdx.x * blockDim.x); // Column
  int yId = threadIdx.y + (blockIdx.y * blockDim.y); // Row
  float scale;

  // Skip if not within size, or on pivotRow
  if ( (yId >= height) || (xId >= width) || (yId == pivotRow) ) {
    return;
  }

  // Find the scale (will be different for each row
  scale = arr[INDEX(pivotColumn, yId)];

  // Row Reduction for each pixel
  arr[INDEX(xId, yId)] -= (arr[INDEX(xId, pivotRow)] * (scale / divider));

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

  // A status flag
  cudaError_t status;

  int num_iterations = 0;
  float scale, divider;
  int pivotRow, pivotColumn;
  int tile_size = 512;
  int size = width * height;

  int dim_size = (int)ceil((float)size / (float)tile_size);

  // Row Reduction Dimensions (2D)
  dim3 dimGridRR(tile_size, tile_size);
  dim3 dimBlockRR(dim_size, dim_size);

  // Normalize Dimensions (1D)
  dim3 dimGridN(1, tile_size);
  dim3 dimBlockN(1, dim_size);

  /////////////////////////////////////////////////
  // Repeat until the bottom row is all positive //
  /////////////////////////////////////////////////
  while (!is_indicator_positive (arr, width, height)) {

    // If number of iterations exceed the threshold, no solutions were found
    if (num_iterations > MAX_ITER) {
      return num_iterations;
    }

    DBGPRINT("Iteration " << num_iterations);

    // Do the gaussian elimination part
    pivotColumn = get_pivot_column_index (arr,width,height);
    pivotRow    = get_pivot_row_index (arr,width,height,pivotColumn);

    // Normalization
    scale = arr[INDEX(pivotColumn, pivotRow)];
    normalize_kernel<<<dimGridN, dimBlockB>>>(arr, scale, height);

    // Check for CUDA errors
    status = cudaGetLastError();
    if (status != cudaSuccess) {
      cout << "Kernel failed: " << cudaGetErrorString(status) << endl;
      return -1;
    }

    // Row reduction
    divider = arr[INDEX(pivotColumn, pivotRow)];
    row_reduce_kernel<<<dimGridRR, dimBlockRR>>>(arr, divider, height, width, pivotRow, pivotColumn);

    // Check for CUDA errors
    status = cudaGetLastError();
    if (status != cudaSuccess) {
      cout << "Kernel failed: " << cudaGetErrorString(status) << endl;
      return -1;
    }

    // Increment the number of iterations
    num_iterations++;
    #ifdef DEBUG
      print_matrix (arr, width, height);
    #endif
  }

  return num_iterations;

}
