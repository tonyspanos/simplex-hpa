//===========================================================================//
// File Name:   simplex_cpu.cpp
// Authors:     Cody Cziesler, Praneeth Pulusani
//
// Description: The simplex algorithm implemented on a CPU
//
//===========================================================================//

#include <iostream>
#include <string>
#include <ctime>

#include "common.h"

using namespace std;

#define ITERS 10

// Function declarations
float* get_array_from_file (string fileprefix, int *width, int *height, bool is_max);
int simplex_gpu (float *arr, int width, int height);
int simplex_gpu_zero_copy (float *tgpu);

//===========================================================================//
// gaussian_eliminate
//
// Perform the gaussian elimination part of the simplex algorithm
//
// @param  arr        - the 1D array representation of the matrix
// @param  width      - the width of arr
// @param  height     - the height of arr
//===========================================================================//
void gaussian_eliminate (float *arr, int width, int height) {

  int pivotColumn = get_pivot_column_index (arr,width,height);
  int pivotRow    = get_pivot_row_index (arr,width,height, pivotColumn);

  // Normalization
  float scale = arr[pivotRow*width+pivotColumn];
  for (int col = 0; col < width; col++) {
    arr[INDEX(col, pivotRow)] /= scale;
  }

  float divider = arr[INDEX(pivotColumn, pivotRow)];

  // Row reduction
  for (int row = 0; row < height; row++) {
    if (row != pivotRow && divider != 0) {
      scale = arr[row*width+pivotColumn];
      for (int col = 0; col < width; col++) {
        arr[INDEX(col, row)] -= (arr[INDEX(col, pivotRow)] * (scale / divider));
      }
    }
  }
}

//===========================================================================//
// simplex_cpu
//
// Puts the whole Simplex Algorithm together
//
// @param  arr        - the 1D array representation of the matrix
// @param  width      - the width of arr
// @param  height     - the height of arr
//
// @return num_iterations - the number of iterations it took to complete
//===========================================================================//
int simplex_cpu (float *arr, int width, int height) {

  int num_iterations = 0;

  // Repeat until the bottom row is all positive
  while (!is_indicator_positive (arr, width, height)) {

    // If number of iterations exceed the threshold, no solutions were found
    if (num_iterations > MAX_ITER) {
      return num_iterations;
    }

    DBGPRINT("Iteration " << num_iterations);
    
    // Do the gaussian elimination part
    gaussian_eliminate (arr, width, height);

    // Increment the number of iterations
    num_iterations++;
    #ifdef DEBUG
      print_matrix (arr, width, height);
    #endif

  }

  return num_iterations;
}

float cuda_test (int size);

//===========================================================================//
// main
//
// The main entrance point to the program
//===========================================================================//
int main() {

  cout << "Starting Simplex CPU Calculations\n";

  int width, height;

  // The array is stored in a 1D vector, where the elements can be accessed by
  // the INDEX(x,y) macro
  #ifdef USE_KNOWN_MATRIX
    /*
    // Maximize
    float arr_ref[] = {  2,  1, 1, 1, 0, 0, 14,
                         4,  2, 3, 0, 1, 0, 28, 
                         2,  5, 5, 0, 0, 1, 30,
                        -1, -2, 1, 0, 0, 0, 0   };  
    width = 7;
    height = 4;
    */
    /*
    float arr_ref[] = {  1,  0,  0,  0,  0, 1, 0, 0, 0, 0, 40,
                         0,  1,  1,  0,  0, 0, 1, 0, 0, 0, 35,
                         0,  0,  1,  1,  0, 0, 0, 1, 0, 0, 45,
                         0,  1,  1,  0,  1, 0, 0, 0, 1, 0, 28,
                        -5, -2, -3, -4,  2, 0, 0, 0, 0, 1, 0 };
    width = 11;
    height = 5;
    */

    float arr_ref[] = {  1,  0,  0,  0,  0, 1, 0, 0, 0, 0, 0, 40,
                         1,  0,  0,  0,  0, 0, 1, 0, 0, 0, 0, 41,
                         0,  1,  1,  0,  0, 0, 0, 1, 0, 0, 0, 35,
                         0,  0,  1,  1,  0, 0, 0, 0, 1, 0, 0, 45,
                         0,  1,  1,  0,  1, 0, 0, 0, 0, 1, 0, 28,
                        -5, -2, -3, -4,  2, 0, 0, 0, 0, 0, 1, 0 };
    width = 12;
    height = 6;

    /* 
    // Maximize
    float arr_ref[] = { 1,  1, 1, 0, 4,
                    2,  1, 0, 1, 5,
                   -3, -4, 0, 0, 0 };
    width = 5;
    height = 3;
    */

    /*
    // Minimize
    float arr_ref[] = { 1,  2, 1, 0, 6,
                    3,  2, 0, 1, 12,
                   -2,  1, 0, 0, 0 };
    width = 5;
    height = 3;
    */

    /*
    // Minimize
    float arr_ref[] = { 1,  3, 1, 0, 2, 
                    2,  2, 0, 1, 5, 
                   -4, -3, 0, 0, 0 };
    width = 5;
    height = 3;
    */
  #else
    float * arr_ref  = get_array_from_file ("cody_200", &width, &height, 0);
  #endif

  // Print out relevant information
  cout << "Width:  " << width << endl;
  cout << "Height: " << height << endl << endl;

  // Variables to capture the runtime information
  float tcpu, tgpu;
  clock_t start, end;

  // Integers to hold the number of iterations returned from simplex_*
  int num_iter_cpu, num_iter_gpu;

  // Allocate arrays
  float * arr     = (float*) malloc (sizeof(float) * width * height);
  float * arr_gpu = (float*) malloc (sizeof(float) * width * height);

  // Do the calculation on a CPU
  start = clock();
  for (int i = 0; i < ITERS; i++) {
    memcpy (arr, arr_ref, width*height*sizeof(float));
    num_iter_cpu = simplex_cpu (arr, width, height);
  }
  end = clock();

  // CPU Printouts
  tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
  cout << "*** CPU ***\n";
  if (num_iter_cpu > MAX_ITER) {
    cout << "No solution was found in " << num_iter_cpu << " iterations\n";
  } else {
    cout << "Z = " << arr[width*height - 1] << endl;
    cout << "The solution took " << num_iter_cpu << " iterations\n";
  }
  cout << endl << endl;

  // Page Locked Memory
  cout << "*** GPU zero-copy ***" << endl;
  float tgpu_pl = 0;
  start = clock();
  for (int i = 0; i < ITERS; i++) {
    memcpy (arr_gpu, arr_ref, width*height*sizeof(float));
    float tmp_tgpu = 0;
    simplex_gpu_zero_copy (&tmp_tgpu);
    tgpu_pl += tmp_tgpu;
  }
  end = clock();
  tgpu_pl /= ITERS;

  // Do the calculation on a CPU
  start = clock();
  for (int i = 0; i < ITERS; i++) {
    memcpy (arr_gpu, arr_ref, width*height*sizeof(float));
    num_iter_gpu = simplex_gpu (arr_gpu, width, height);
  }
  end = clock();

  // GPU Printouts
  tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
  cout << endl << endl << "*** GPU ***\n";
  if (num_iter_gpu > MAX_ITER) {
    cout << "No solution was found in " << num_iter_gpu << " iterations\n";
  } else {
    cout << "Z = " << arr_gpu[width*height - 1] << endl;
    cout << "The solution took " << num_iter_gpu << " iterations\n" << endl << endl;
  }

  // Final Printouts
  cout << "-----------------------------------------------------\n";
  cout << "The CPU took " << tcpu << " ms" << endl << endl;

  cout << "The GPU took " << tgpu << " ms" << endl;
  cout << "Speedup (CPU/GPU) = " << tcpu/tgpu << endl << endl;

  cout << "The GPU (zero-copy) took " << tgpu_pl << " ms" << endl;
  cout << "Speedup (CPU/GPU_ZC) = " << tcpu/tgpu_pl << endl;

  #ifndef USE_KNOWN_MATRIX
    free (arr);
  #endif
  
}
