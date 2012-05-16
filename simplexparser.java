import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.StringTokenizer;

public class simplexparser {
	public static void main(String[] args) throws IOException {
		ArrayList<Float> matrix1d = new ArrayList<Float>();

		File file = new File(args[0]+".csv");

		BufferedReader bufRdr = new BufferedReader(new FileReader(file));
		String line = null;
		int rows = 0;
		int elems = 0;

		// read each line of text file
		bufRdr.readLine();
		while ((line = bufRdr.readLine()) != null) {

			StringTokenizer st = new StringTokenizer(line, ";");
			while (st.hasMoreTokens()) {
				// get next token and store it in the array
				try {
					matrix1d.add(Float.parseFloat(st.nextToken()));
					elems++;
				} catch (NumberFormatException e) {

				}

			}
			rows++;
		}
		rows = rows - 1;//the first row has variable names
		System.out.println("rows:" + rows + ",cols:" + elems / rows);
		// close the file
		bufRdr.close();
		int cols=matrix1d.size()/rows;


		
		
		
		// Create file
		FileWriter fstream = new FileWriter(args[0]+".nlv");
		BufferedWriter out = new BufferedWriter(fstream);
		//write number of rows and number of columns at the top
		out.write(rows + "");
		out.newLine();
		out.write(cols+cols-1 + "");
		// out.newLine();
		//add rest of the matrix element into new line value file
		/*for (Float f : matrix1d) {
			out.newLine();
			out.write(f.toString());
			// out.newLine();
		}*/
		
		
		//add slack variables and write the file
		for(int r=0;r<rows;r++){
			ArrayList<Float> rowArray = new ArrayList<Float>();
			//copy the row except the solution element
			for (int c =0;c<cols-1;c++){
				rowArray.add(matrix1d.get(r*cols+c));
				
			}
			//add slack variables/ rowlength-1 is doubled
			for(int numslack=0;numslack< cols-1;numslack++){
				if(r==numslack){
					rowArray.add(1.0f);
				}else{
					rowArray.add(0.0f);
				}
			}
			//add the solution column entry
			rowArray.add(matrix1d.get(r*cols+(cols-1)));
			for (Float f : rowArray) {
				out.newLine();
				out.write(f.toString());
				// out.newLine();
			}
		}
		
		

		// Close the output stream
		out.close();

	}
}
