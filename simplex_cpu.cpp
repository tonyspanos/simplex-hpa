//===========================================================================//
// File Name:   simplex_cpu.cpp
// Authors:     Cody Cziesler, Praneeth Pulusani
//
// Description: The simplex algorithm implemented on a CPU
//
//===========================================================================//

#include <iostream>
#include <string>
//#include <time.h>

#include "common.h"

using namespace std;

// Declaration from fileio.cpp
float* get_array_from_file (string fileprefix, int *width, int *height, bool is_max);

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
  float scale;

  int pivotColumn = get_pivot_column_index (arr,width,height);
  int pivotRow    = get_pivot_row_index (arr,width,height, pivotColumn);

  // Normalization
  scale = arr[pivotRow*width+pivotColumn];
  //cout << "PC: " << pivotColumn << " PR: " << pivotRow << " SCALE: " << scale << endl;
  for (int col = 0; col < width; col++) {
    arr[INDEX(col, pivotRow)] /= scale;
  }

  float divider = arr[INDEX(pivotColumn, pivotRow)];
  //cout << "index: " << INDEX(pivotColumn, pivotRow) << endl;
  //cout << "PC: " << pivotColumn << " PR: " << pivotRow << " DIV: " << divider << endl;

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
    // Maximize
    float arr[] = {  2,  1, 1, 1, 0, 0, 14,
                     4,  2, 3, 0, 1, 0, 28, 
                     2,  5, 5, 0, 0, 1, 30,
                    -1, -2, 1, 0, 0, 0, 0   };   
    width = 7;
    height = 4;

    /* 
      // Maximize
      float arr[] = { 1,  1, 1, 0, 4,
                      2,  1, 0, 1, 5,
                     -3, -4, 0, 0, 0 };
      */

      /*
      // Minimize
      float arr[] = { 1,  2, 1, 0, 6,
                      3,  2, 0, 1, 12,
                     -2,  1, 0, 0, 0 };
      */

      /*
      // Minimize
      float arr[] = { 1,  3, 1, 0, 2, 
                      2,  2, 0, 1, 5, 
                     -4, -3, 0, 0, 0 };
    */
  #endif  

  #ifndef USE_KNOWN_MATRIX
    float * arr = get_array_from_file ("adlittle", &width, &height, 0);
  #endif

  // Print out relevant information
  cout << "Width:  " << width << endl;
  cout << "Height: " << height << endl << endl;

  cout << "Starting matrix:\n";
  print_matrix (arr, width, height);

  // Do the calculation
  int num_iter = simplex_cpu (arr, width, height);

  // Determine whether a solution was found
  if (num_iter > MAX_ITER) {
    cout << "No solution was found in " << num_iter << " iterations\n";
  } else {
    cout << "\nSolution matrix:\n";
    print_matrix (arr, width, height);
    cout << "The solution took " << num_iter << " iterations\n";
  }

  #ifndef USE_KNOWN_MATRIX
    free (arr);
  #endif
  
}
