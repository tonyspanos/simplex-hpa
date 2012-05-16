#include <iostream>
#include <fstream>
#include <string>
using namespace std;

// string fileprefix="simplex";

float* get_array_from_file (string fileprefix, int *width, int *height) {

	//java call usage java -jar simplexparser.jar <filename without extension>
	//reads in filename.csv, writes out filename.nlv
	string execCommand = "java -jar simplexparser.jar "+fileprefix;
	system(execCommand.c_str());
	string line;
	string nlvfilename=fileprefix+".nlv";
	ifstream myfile (nlvfilename.c_str());
	int rows;
	int cols;
	//keeps track of elements, but probably not needed because rows*cols gives it anyway
	int elem_counter;
	float *matrix;

	if (myfile.is_open()) {
		//first line has number of rows
		getline (myfile,line);
		//cout<<line;
		rows = atoi(line.c_str());
		//seccond line has number of cols
		getline (myfile,line);
		cols=atoi(line.c_str());

		elem_counter=0;

		matrix = new float[rows*cols];
    //matrix = (float*) malloc (sizeof(float) * rows * cols);

		//as long as end of file is not reached and errors dont pop up, read a new line
		while ( myfile.good() ){
			getline (myfile,line);
			//add the element of a 2d matrix to a single dimension array
			matrix[elem_counter]=atof(line.c_str());
			//cout << line << endl;
			elem_counter++;
		}
		myfile.close();

		//print the matrix to test if it read the file correctly
		elem_counter=0;
		/*
    for(int r =0; r<rows;r++){
			for(int c =0;c<cols;c++,elem_counter++){
				cout<<matrix[elem_counter]<<" ";
			}
			cout<<endl;
		}
    */

    // Save width and height
    *width  = cols;
    *height = rows;

	} else {
			cout << "Unable to open file"; 
      return 0;
	}
  //return true;
	return matrix;
}