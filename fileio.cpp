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
//
// @return matrix     - an array containing the matrix read in
//===========================================================================//
float* get_array_from_file (string fileprefix, int *width, int *height, bool is_max) {



	//java call usage java -jar simplexparser.jar <filename without extension>
	//reads in filename.csv, writes out filename.nlv

  string execCommand;
  if (is_max) {
  //  execCommand = "java -jar simplexparser.jar " + fileprefix + " maximize";
  } else {
  //  execCommand = "java -jar simplexparser.jar " + fileprefix;
  }

	//system (execCommand.c_str());
	string line;
	string nlvfilename = fileprefix+".nlv";
	ifstream myfile (nlvfilename.c_str());
	int rows;
	int cols;
	//keeps track of elements, but probably not needed because rows*cols gives it anyway
	int elem_counter;
	float *matrix;



	if (myfile.is_open()) {
		//first line has number of rows
		getline (myfile,line);
		rows = atoi(line.c_str());
    //cout << "ROWS: " << rows << endl;

		//seccond line has number of cols
		getline (myfile,line);
		cols = atoi(line.c_str());
    //cout << "COLS: " << cols << endl;

		elem_counter = 0;

		matrix = new float[rows*cols];

		//as long as end of file is not reached and errors dont pop up, read a new line
		while ( myfile.good() ){
			getline (myfile,line);
			//add the element of a 2d matrix to a single dimension array
			matrix[elem_counter] = atof(line.c_str());
			elem_counter++;
		}
		myfile.close();

    //cout << elem_counter << endl;
    //cout << (rows * cols) << endl;

    /*
		//print the matrix to test if it read the file correctly
		elem_counter=0;
    for(int r =0; r<rows;r++){
			for(int c =0;c<cols;c++,elem_counter++){
				cout<<matrix[elem_counter]<<" ";
			}
			cout<<endl;
		}
    */

    // Save width and height variables
    *width  = cols;
    *height = rows;

	} else {
			cout << "Unable to open file\n"; 
      return 0;
	}

	return matrix;

}