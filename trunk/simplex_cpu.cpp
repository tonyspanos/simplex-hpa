//===========================================================================//
// File Name:   simplex_cpu.cpp
// Authors:     Cody Cziesler, Praneeth Pulusani
//
// Description: The simplex algorithm implemented on a CPU
//
//===========================================================================//

#include <iostream>
using namespace std;

#define INDEX(x,y) ((x) + width * (y))

#ifdef DEBUG
  #define DBGPRINT(s) (cout << s << endl)
#else
  #define DBGPRINT(S)
#endif

// Made global so that functions can get this value without it being passed in
int num_rest = 3;
int num_vars = 3;

//===========================================================================//
// print_matrix
//
// Prints the array in a nice-to-read format
//
// @param  arr        - the 1D array representation of the matrix
// @param  width      - the width of arr
// @param  height     - the height of arr
//===========================================================================//
void print_matrix (float * arr, int width, int height) {
  cout << "----------------------------------------\n";
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      cout.width(5);
      // Print out 0 if element is close to 0
      if ( arr[INDEX(x,y)] < 1e-6 && arr[INDEX(x,y)] > -1e-6) {
        cout << 0;
      } else {
        cout << arr[INDEX(x,y)];
      }
      if (x != width-1) {
        cout << ", ";
      }
    }
    cout << endl;
  }
  cout << "----------------------------------------\n\n";
}

//===========================================================================//
// get_pivot_column_index
//
// Find the most negative number on the bottom row
//
// @param  arr        - the 1D array representation of the matrix
// @param  width      - the width of arr
// @param  height     - the height of arr
//
// @return pivot_col  - the pivot column
//===========================================================================//
int get_pivot_column_index (float *arr, int width, int height) {
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
      DBGPRINT("swap!");
      min_val = cur_val;
      pivot_col = x;
    }
  }

  // Return the pivot column
  return pivot_col;
}

//===========================================================================//
// get_pivot_row_index
//
// Gets the pivot row based on the ratio
//
// @param  arr        - the 1D array representation of the matrix
// @param  width      - the width of arr
// @param  height     - the height of arr
//
// @return pivotRowIndex - the pivot row
//===========================================================================//
int get_pivot_row_index(float *arr, int width, int height){
  int solutionColumnIndex = width - 1;

  // Pivot column index will be of the column where the least number in the 
  // indicator row resides
  int pivotColumnIndex = get_pivot_column_index(arr,width,height);

  // pivot row index will be of the row whose ratio of solution col element 
  // to current element is the lowest
  int pivotRowIndex=0;  //default

  float lowestRatio=arr[0*width+solutionColumnIndex]/arr[0*width+pivotColumnIndex];//default

  // h=1: because computed the first ratio already in the two lines above
  // h<height-1: do not consider the last row because it is the indicator row
  for(int h=1;h<height-1;h++){
    float ratio = arr[h*width+solutionColumnIndex]/arr[h*width+pivotColumnIndex];
    // update the pivot row
    if(ratio<=lowestRatio){
      lowestRatio=ratio;
      pivotRowIndex=h;
    }
  }
  return pivotRowIndex;
}

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
  int counter = 0;
  float scale;

  int pivotColumn = get_pivot_column_index(arr, width,height);
  int pivotRow    = get_pivot_row_index(arr, width,height);

  // Normalization
  scale = arr[pivotRow*width+pivotColumn];
  for (int col = 0; col < width; col++) {
    arr[pivotRow*width + col] = arr[pivotRow*width + col] / scale;
  }

  float divider   = arr[pivotRow*width+pivotColumn];

  // Row reduction
  for (int row = 0; row < height; row++) {
    if (row != pivotRow && divider != 0) {
      scale = arr[row*width+pivotColumn];
      for (int col = 0; col < width; col++) {
       arr[row*width+col] = arr[row*width+col] - (arr[pivotRow*width+col] * (scale / divider));
      }
    }
  }
}

//===========================================================================//
// is_indicator_positive
//
// @param  arr        - the 1D array representation of the matrix
// @param  width      - the width of arr
// @param  height     - the height of arr
//
// @return            - true if positive, false otherwise
//===========================================================================//
bool is_indicator_positive(float *arr, int width, int height){
  int indicatorRowIndex=height-1;
  bool is_ind_pos=true;
  for(int col = 0; col<width;col++){
    if(arr[indicatorRowIndex*width+col]<0){
      is_ind_pos=false;
    }
  }
  return is_ind_pos;
}

//===========================================================================//
// simplex_cpu
//
// Puts the whole Simplex Algorithm together
//
// @param  arr        - the 1D array representation of the matrix
// @param  width      - the width of arr
// @param  height     - the height of arr
//===========================================================================//
void simplex_cpu (float *arr, int width, int height) {

  // Repeat until the bottom row is all positive
  while (!is_indicator_positive (arr, width, height)) {
    gaussian_eliminate (arr, width, height);
    //print_matrix (arr, width, height);
  }

}

//===========================================================================//
// main
//
// The main entrance point to the program
//===========================================================================//
int main() {

  cout << "Starting Simplex CPU Calculations\n";

  // Number of restrictions
  num_rest = 2;

  // Number of variables
  num_vars = 2;

  int width  = (2*num_vars)+1;
  int height = num_rest + 1;
  int size   = width * height;

  /*
  float arr[] = {  2,  1, 1, 1, 0, 0, 14,
                   4,  2, 3, 0, 1, 0, 28, 
                   2,  5, 5, 0, 0, 1, 30,
                  -1, -2, 1, 0, 0, 0, 0   };
  */

  /* // Maximize
  float arr[] = { 1,  1, 1, 0, 4,
                  2,  1, 0, 1, 5,
                 -3, -4, 0, 0, 0 };
  */

  /*// Minimize
  float arr[] = { 1,  2, 1, 0, 6,
                  3,  2, 0, 1, 12,
                 -2,  1, 0, 0, 0 };
  */

  // Minimize
  float arr[] = { 1,  3, 1, 0, 2, 
                  2,  2, 0, 1, 5, 
                 -4, -3, 0, 0, 0 };


  cout << "Width:  " << width << endl;
  cout << "Height: " << height << endl << endl;
 
  cout << "Starting matrix:\n";
  print_matrix (arr, width, height);

  simplex_cpu (arr, width, height);

  cout << "\nSolution matrix:\n";
  print_matrix (arr, width, height);

}