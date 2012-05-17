#include <iostream>
#include <fstream>
#include <string>
using namespace std;

//===========================================================================//
// get_array_from_file
//
// Calls the "simpleparser.jar" program to parse the simplex file
//
// @param  fileprefix - a string containing the prefix to the file to open
// @param  *width     - a pointer to the width variable
// @param  *height    - a pointer to the height variable
// @param  is_max     - 1 if maximize, 0 otherwise
//
// @return matrix     - an array containing the matrix read in
//===========================================================================//
float* get_array_from_file (string fileprefix, int *width, int *height, bool is_max) {

	// Java call usage java -jar simplexparser.jar <filename without extension>
	// reads in filename.csv, writes out filename.nlv

  /*
  // Execute the Java program
  string execCommand;
  if (is_max) {
    execCommand = "java -jar simplexparser.jar " + fileprefix + " maximize";
  } else {
    execCommand = "java -jar simplexparser.jar " + fileprefix;
  }
	system (execCommand.c_str());
  */

	string line;
	string nlvfilename = fileprefix+".nlv";
	ifstream myfile (nlvfilename.c_str());
	int rows;
	int cols;

	// Keeps track of elements, but probably not needed because rows*cols gives it anyway
	int elem_counter;
	float *matrix;

  // As long as file is open...
	if (myfile.is_open()) {

		// First line has number of rows
		getline (myfile,line);
		rows = atoi(line.c_str());

		// Seccond line has number of cols
		getline (myfile,line);
		cols = atoi(line.c_str());

    // Malloc the matrix
		matrix = new float[rows*cols];

		// As long as end of file is not reached and errors dont pop up, read a new line
		elem_counter = 0;
    while ( myfile.good() ){
			getline (myfile,line);
			// Add the element of a 2d matrix to a single dimension array
			matrix[elem_counter++] = (float) atof(line.c_str());
		}
		myfile.close();

    /*
		// Print the matrix to test if it read the file correctly
		elem_counter = 0;
    for(int r = 0; r < rows; r++){
			for(int c = 0; c < cols; c++, elem_counter++){
				cout << matrix[elem_counter] << " ";
			}
			cout << endl;
		}
    */

    // Save width and height variables
    *width  = cols;
    *height = rows;

	} else {
			cout << "Unable to open file\n"; 
      return 0;
	}

  // Return the Matrix
	return matrix;

}