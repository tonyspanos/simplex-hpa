//===========================================================================//
// File Name:   common_gpu.h
// Authors:     Cody Cziesler, Praneeth Pulusani
//
// Description: Helper functions to be used for the Simplex Algorithm on both
//              the CPU and GPU
//
//===========================================================================//

#ifndef COMMON_GPU_H
#define COMMON_GPU_H

#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <time.h>

using namespace std;

// Defines used for the program
#define INDEX(x,y) ((x) + width * (y))
#define MAX_ITER 1000

// #define DEBUG
#ifdef DEBUG
  #define DBGPRINT(s) (cout << s << endl)
#else
  #define DBGPRINT(S)
#endif

//===========================================================================//
// print_matrix_gpu
//
// Prints the array in a nice-to-read format
//
// @param  arr        - the 1D array representation of the matrix
// @param  width      - the width of arr
// @param  height     - the height of arr
//===========================================================================//
void print_matrix_gpu (float * arr, int width, int height) {
  // Don't print if height or width is too large
  if (height > 20 || width > 20) {
    //return;
  }
  cout << "------------------------------------------------------------------------------\n";
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      cout.width(13);
      // Print out 0 if element is close to 0
      if ( arr[INDEX(x,y)] < 1e-6 && arr[INDEX(x,y)] > -1e-6) {
        cout << 0;
      } else {
        cout << setprecision (5) << arr[INDEX(x,y)];
      }
      if (x != width-1) {
        cout << ", ";
      }
    }
    cout << ";" << endl;
  }
  cout << "------------------------------------------------------------------------------\n\n";

}

//===========================================================================//
// get_pivot_column_index_gpu
//
// Find the most negative number on the bottom row. In case of a tie, select
// the column farthest to the right
//
// @param  arr        - the 1D array representation of the matrix
// @param  width      - the width of arr
// @param  height     - the height of arr
//
// @return pivot_col  - the pivot column
//===========================================================================//
int get_pivot_column_index_gpu (float *arr, int width, int height) {
  int pivot_col = 0;
  float min_val = 0;

  // Loop through each column
  for (int x = 0; x < width; x++) {
    // Save this for later use
    float cur_val = arr[INDEX(x, height - 1)];

    DBGPRINT("\nx: " << x);
    DBGPRINT("cur_val: " << cur_val);
    DBGPRINT("min_val: " << min_val);
    DBGPRINT("pivot_col: " << pivot_col);
    
    // If current value is less than the min value, or x is 0, swap
    // min_val with cur_val and set the pivot_col to the current x
    // value
    if ((cur_val <= min_val) || (x == 0)) {
      DBGPRINT("pivotCol swap!");
      min_val = cur_val;
      pivot_col = x;
    }
  }

  // Return the pivot column
  return pivot_col;
}

//===========================================================================//
// get_pivot_row_index_gpu
//
// Gets the pivot row based on the ratio. In case of a tie, select the row 
// furthest toward the bottom.
//
// @param  arr              - the 1D array representation of the matrix
// @param  width            - the width of arr
// @param  height           - the height of arr
// @param  pivotColunmIndex - the column where the pivot row resides
//
// @return pivotRowIndex    - the pivot row
//===========================================================================//
int get_pivot_row_index_gpu (float *arr, int width, int height, int pivotColumnIndex) {

  int solutionColumnIndex = width - 1;

  // Pivot row index will be of the row whose ratio of solution col element 
  // to current element is the lowest
  int pivotRowIndex = 0;

  float lowestRatio;
  
  if (arr[INDEX(pivotColumnIndex, 0)] != 0) {
    lowestRatio = arr[INDEX(solutionColumnIndex, 0)] / arr[INDEX(pivotColumnIndex, 0)];
  } else {
    lowestRatio = 100000;
  }

  // h=1: because computed the first ratio already in the two lines above
  // h<height-1: do not consider the last row because it is the indicator row
  for (int h = 1; h < height - 1; h++){
    DBGPRINT("h: " << h);
    DBGPRINT("lowestRatio: " << lowestRatio);
    DBGPRINT("pivotRowIndex: " << pivotRowIndex << endl);
    if (arr[INDEX(pivotColumnIndex, h)] != 0) {
      float ratio = arr[INDEX(solutionColumnIndex, h)] / arr[INDEX(pivotColumnIndex, h)];
      // Update the pivot row
      if (ratio <= lowestRatio){
        lowestRatio   = ratio;
        pivotRowIndex = h;
        DBGPRINT("pivotRow swap!");
      }
    }
  }

  return pivotRowIndex;
}

//===========================================================================//
// is_indicator_positive_gpu
//
// Loop through the bottom row of arr to see if all of the values are 
// positive
//
// @param  arr        - the 1D array representation of the matrix
// @param  width      - the width of arr
// @param  height     - the height of arr
//
// @return            - true if positive, false otherwise
//===========================================================================//
bool is_indicator_positive_gpu (float *arr, int width, int height){
  // Loop through all columns in bottom row
  for (int col = 0; col < width; col++) {
    // If negative, break and return false
    if (arr[INDEX(col, height - 1)] < 0) {
      return false;
    }
  }
  return true;
}

//===========================================================================//
// initialize_matrix_gpu
//
// Initialize the values of the matrix to random integers.
//
// NOTE: This doesn't always give matrices that work (CODY)
//
// @param  arr        - the 1D array representation of the matrix
// @param  width      - the width of arr
// @param  height     - the height of arr
//===========================================================================//
void initialize_matrix_gpu (float *arr, int width, int height) {

  // Seed random number generator
  srand ( (unsigned int) time(NULL) );

  for (int x = 0; x < (width - 1) / 2; x++) {
    for (int y = 0; y < height; y++) {
      if (y == height-1) {
        arr[INDEX(x,y)] = (float) (rand() % 20) - 10;
      } else {
        arr[INDEX(x,y)] = (float) (rand() % 20);
      }
    }
  }

  for (int x = (width - 1) / 2; x < width - 1; x++) {
    for (int y = 0; y < height; y++) {
      DBGPRINT("x: " << x << ", y: " << y);
      if (x - ((width - 1) / 2) == y) {
        arr[INDEX(x,y)] = 1;
      } else {
        arr[INDEX(x,y)] = 0;
      }
    }
  }

  for (int y = 0; y < height; y++) {
    if (y == height-1) {
      arr[INDEX(width-1, y)] = 0;
    } else {
      arr[INDEX(width-1, y)] = (float) (rand() % 20);
    }
  }

}

#endif
