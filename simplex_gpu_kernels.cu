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

#include "common_gpu.h"
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
   #define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif
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
  //printf("kernel\n");
	
  // Indexing
  int yId = threadIdx.y + (blockIdx.y * blockDim.y); // Row

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
// @param  arr        - the 1D array representation of the matrix
// @param  divider    - arr[INDEX(pivotColumn, PivotRow)]
// @param  height     - the height of arr
//===========================================================================//
__global__ void row_reduce_kernel (float* arr, int height, int width, int pivotRow, int pivotColumn) {

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

  float divider = arr[INDEX(pivotColumn, pivotRow)];
  // Row Reduction for each pixel
  //printf("x: %d, y: %d, value: %f = %f- %f* (%f/%f)\n",xId,yId,arr[INDEX(xId, yId)],arr[INDEX(xId, yId)],arr[INDEX(xId, pivotRow)],scale,divider);
  arr[INDEX(xId, yId)] -= (arr[INDEX(xId, pivotRow)] * (scale / divider));

  //printf("pivotColumn: %d, width: %d, height: %d, gausselim col:%d,row:%d,pivot row:%d,scale/div:%f\n",pivotColumn, width,height,xId,yId,pivotRow,scale/divider);


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
	cout<<"hello there"<<endl;
  DBGPRINT("Simplex GPU function entered ");
  // A status flag
  cudaError_t status;

  int num_iterations = 0;
  float scale;
  int pivotRow, pivotColumn;
  int tile_size = 512;
  int size = width * height;


  // Number of bytes in the matrix. 
  int bytes = size * sizeof(float); 
  // Pointers to the device arrays 
  float *arr_d; 
  // Allocate memory on the device to store each matrix 
  cudaMalloc((void**) &arr_d, bytes); 
  // Copy the host input data to the device 
  cudaMemcpy(arr_d, arr, bytes, cudaMemcpyHostToDevice); 

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
  while (!is_indicator_positive_gpu (arr, width, height)) {
    cout<<"hello there1"<<endl;
    // If number of iterations exceed the threshold, no solutions were found
    if (num_iterations > MAX_ITER) {
      return num_iterations;
    }

    DBGPRINT("Iteration " << num_iterations);

    // Do the gaussian elimination part
    pivotColumn = get_pivot_column_index_gpu (arr,width,height);
    pivotRow    = get_pivot_row_index_gpu (arr,width,height,pivotColumn);

	DBGPRINT("Iteration " );
	cout<<"hello there normalization"<<endl;
    // Normalization
    scale = arr[INDEX(pivotColumn, pivotRow)];
    normalize_kernel<<<dimGridN, dimBlockN>>>(arr_d, scale, height, width, pivotRow);
	cudaThreadSynchronize();

    // Check for CUDA errors
    status = cudaGetLastError();
    if (status != cudaSuccess) {
      cout << "Kernel failed: " << cudaGetErrorString(status) << endl;
      return -1;
    }
	cout<<"hello there row reduction"<<endl;
	DBGPRINT("Before row reduction ");
    // Row reduction
    row_reduce_kernel<<<dimGridRR, dimBlockRR>>>(arr_d, height, width, pivotRow, pivotColumn);
	cudaThreadSynchronize();

	//return 0;
    // Check for CUDA errors
    status = cudaGetLastError();
    if (status != cudaSuccess) {
      cout << "Kernel failed: " << cudaGetErrorString(status) << endl;
      return -1;
    }

    // Increment the number of iterations
    num_iterations++;
    #ifdef DEBUG
     // print_matrix_gpu (arr, width, height);
    #endif
	 // Copy the host input data to the device 
	 cudaMemcpy(arr, arr_d, bytes, cudaMemcpyDeviceToHost); 
	  // print_matrix_gpu(arr,width,height);
  }
print_matrix_gpu(arr,width,height);

  cout << "Printing GPU solution\n";

  return num_iterations;

}
