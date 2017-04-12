
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <vector>
#include <random>
//#include <math.h>
#include "Eigen\Eigen\Dense"
#include "Eigen\Eigen\src\SVD\JacobiSVD.h"
using Eigen::MatrixXd;
using Eigen::MatrixXf;
using namespace Eigen;

using namespace std;

double string_to_double( const std::string& s ){
	std::istringstream i(s);
	double x;
   if (!(i >> x))
return 0;
   return x;
 } 
//for polynomial fitting
const int m=4;//m-1 is the degree of the polynomial we are fitting for the 2nd and third components
const int m1=2;//m1-1 is the degree of the polynomial we are fitting to the 1st component. We are going to fit Ln(x) rather than x 
MatrixXd polyCoeffic1(m1,1);
MatrixXd polyCoeffic2(m,1);
MatrixXd polyCoeffic3(m,1);

double Vol_1(double Tau){
	//return 0.0064306548; //This is whats in his code
	if (Tau==0.0){
		Tau=0.0001;
	}
	//return (0.004398239+0.00082817 * log(Tau));
	return (polyCoeffic1(0,0)+polyCoeffic1(1,0) * log(Tau));

}

double Vol_2(double Tau){

	//return (0.0035565431) + Tau * (0.0005683999) + Tau * Tau * (-0.0001181915) + Tau * Tau * Tau * (0.0000035939);
	return polyCoeffic2(0,0) + Tau * polyCoeffic2(1,0) + Tau * Tau * polyCoeffic2(2,0) + Tau * Tau * Tau * polyCoeffic2(3,0);
}

double Vol_3(double Tau){
	return polyCoeffic3(0,0) + Tau * polyCoeffic3(1,0) + Tau * Tau * polyCoeffic3(2,0) + Tau * Tau * Tau * polyCoeffic3(3,0);
}

double M(double Tau){
	
	int N;
	double M1, M2, M3;
	double dTau;
	
	if(Tau==0){
		return 0;
	}else{
		
		dTau= 0.01;
		N=Tau/dTau; 
		dTau=Tau/N; //recalc dTau
		
		//using trapezium rule to compute M1
		M1 = 0.5 * Vol_1(0);
        for(int i=1; i<N ; i++){
            M1 = M1 + Vol_1(i * dTau);
		}
        
        M1 = M1 + 0.5 * Vol_1(Tau);
        M1 = M1 * dTau;
        M1 = Vol_1(Tau) * M1; //Vol_1 represents v_i(t,T) and M1 represents integral part for one factor (Slide 15)	

		//using trapezium rule to compute M2
        M2 = 0.5 * Vol_2(0);
        for(int i=1; i<N ; i++){
            M2 = M2 + Vol_2(i * dTau);
		}
        M2 = M2 + 0.5 * Vol_2(Tau);
        M2 = M2 * dTau;
        M2 = Vol_2(Tau) * M2;

		//using trapezium rule to compute M3
        M3 = 0.5 * Vol_3(0);
        for(int i=1; i<N ; i++){
            M3 = M3 + Vol_3(i * dTau);
		}
        M3 = M3 + 0.5 * Vol_3(Tau);
        M3 = M3 * dTau;
        M3 = Vol_3(Tau) * M3;

		return M1 + M2 + M3; //sum for multi-factor
	}
}



int	main()
{

	/*This is just for testing
	vector < vector < vector<double> > > third;
	vector < vector<double> >  second;
	vector<double> first;

	for(int j=0;j<3;j++){
		for(int i=0;i<3;i++){
			first.clear();
			first.push_back(i);
			first.push_back(i+1);
			first.push_back(i+2);
			second.push_back(first);
		}
		third.push_back(second);
		second.clear();
	}
	
	cout << "000 is " << third[0][0][0] << '\n';
	cout << "111 is " << third[1][1][1] << '\n';
	cout << "222 is " << third[2][2][2] << '\n';

	//*/
	/*/*//*/*//*//*//*/*//*/*//*/*//*/*//*//*//*/*//*/*//*/*//*/*//*/
							read in the data
	/*///*/*//*//*//*/*//*/*//*/*//*/*//*//*//*/*//*/*//*/*//*/*//*/
	ifstream infile("csv data.csv"); // for example

	/*/*//*/*//*//*//*/*//*/*//*/*//*/*//*//*//*/*//*/*//*/*//*/*//*/
							put the data into vectors
	/*///*/*//*//*//*/*//*/*//*/*//*/*//*//*//*/*//*/*//*/*//*/*//*/

	string line = "";
	vector< vector<double> > data;
	

	while (getline(infile, line)){ //while there are lines in the file, get the line
        stringstream strstr(line); //put the line in a stringstream
        string word = ""; //make a blank word
        double number;//make a new number
		vector<double> row;//make a vector row to put the line into

		while (getline(strstr,word, ',')){//put the word separated by ',' into string "word"
				number = string_to_double(word);//convert the "word" to a double
				row.push_back(number);//put the new number from the line into the row
		}
		data.push_back(row);//put the row into the whole matrix
	}


	
	//vector< vector<double> > all_words;

	//for(unsigned i=0; i<1265; i++){
	//	for(unsigned j=0; j<52; j++){
	//		double a = all_words[i][j];
	//		cout << a << '\n';
	//	}
	//}

	int num_columns;
	int num_rows;
	cout << "Checking input data... " << '\n';
	num_columns = data[1].size();
	cout << "The number of columns in the data is " << num_columns << '\n';
	num_rows = data.size();
	cout << "The number of rows in the data is " << num_rows << '\n';


	//just a test
	/*double a = data[0][0];
	cout << a << '\n';
	a = data[26][32];
	cout << a << '\n';*/
	
	/*/*//*/*//*//*//*/*//*/*//*/*//*/*//*//*//*/*//*/*//*/*//*/*//*/
							now get the differences
	/*///*/*//*//*//*/*//*/*//*/*//*/*//*//*//*/*//*/*//*/*//*/*//*/

	vector< vector<double> > differences;

	for(int i=0; i<num_rows-1; i++){ //Loop over only go to rows-1 since because its differences we have 1 less

		vector<double> row;//make a row to put the row into

		for(int j=0; j<num_columns; j++){
			if (i==0||j==0){//if we are on the 0'th row or column, then just put in the data directly as it is header info
				row.push_back(data[i][j]);//put the data in without change
		}
			else{
					row.push_back(data[i+1][j]-data[i][j]);//put the differences of the data in
				}
		}
		differences.push_back(row);//put the row into the whole matrix
	}

	//just a test
	/*a = differences[0][0];
	cout << a << '\n';
	a = differences[26][32];
	cout << a << '\n';
	cout << "differences size is " <<differences.size() << '\n';*/

	ofstream myfile;
	myfile.open ("differences.csv");
	for(int i=0; i<num_rows-1; i++){ //only go to rows-1 since its differences we have 1 less
			for(int j=0; j<num_columns; j++){
				myfile <<differences[i][j] <<",";
			}
			myfile << "\n";
	}

	myfile.close();

	/*/*//*/*//*//*//*/*//*/*//*/*//*/*//*//*//*/*//*/*//*/*//*/*//*/
							now get the covariance matrix
	/*///*/*//*//*//*/*//*/*//*/*//*/*//*//*//*/*//*/*//*/*//*/*//*/

	vector< vector<double> > covariancematrix;
	vector<double> x;
	vector<double> y;
	covariancematrix.resize(num_columns);
	for(int h=0; h<num_columns; h++){
			covariancematrix[h].resize(num_columns);
	}
	x.resize(num_rows-2);
	y.resize(num_rows-2);

	for(int i=0; i<num_columns; i++){ //for each column of the differences matrix
		
		for(int j=1; j<num_rows-1; j++){//only from 1 onwards as we dont want to header info
			x[j-1]=differences[j][i];//first construct the x column to compare to y
		}

		for(int h=0; h<num_columns; h++){
			for(int j=1; j<num_rows-1; j++){//only from 1 onwards as we dont want to header info
				y[j-1]=differences[j][h];//then construct the y column we want to compare x to.
			}
			if(h==0||i==0){//so if we are in the edge LHS of the matrix for either comparison vector then its just header information.
				if(i==0){
					covariancematrix[i][h]=differences[i][h];//then just make it equal the data
				}else{
					covariancematrix[i][h]=differences[h][i];//except in the case of the covariance matrix we want the side info the same as the top header info hence[h][i] rather than [i][h]
				}
			}else if(h<i){
				covariancematrix[i][h]=covariancematrix[h][i];
			}else{
				//calculate covariance
				double xmean; 
				double sumx = 0;
				double ymean; 
				double sumy = 0;
				for(unsigned p = 0; p < x.size(); p++){
					sumx += x[p];
					sumy += y[p];
				}
				xmean=sumx/x.size();
				ymean=sumy/y.size();
				
				double total = 0;
				
				for(int k = 0; k < x.size(); k++)
				{
					total += (x[k] - xmean) * (y[k] - ymean);
				}
				
				covariancematrix[i][h]=((total/x.size())* 252 / 10000);
			}


		}

	}

	//just a test
	/*a = covariancematrix[0][0];
	cout <<"0,0 = "<< a << '\n';
	a = covariancematrix[26][32];
	cout <<"26,32 ="<< a << '\n';
	a = covariancematrix[1][1];
	cout <<"1,1 ="<< a << '\n';
	a = covariancematrix[2][2];
	cout <<"2,2 ="<< a << '\n';*/
	cout <<"please wait, computing..."<<endl;

	myfile.open ("covariance matrix.csv");
	for(int i=0; i<num_columns; i++){ //only go to rows-1 since its differences we have 1 less
			for(int j=0; j<num_columns; j++){
				myfile <<covariancematrix[i][j] <<",";
			}
			myfile << "\n";
	}

	myfile.close();
	
	/*/*//*/*//*//*//*/*//*/*//*/*//*/*//*//*//*/*//*/*//*/*//*/*//*/
							now for eiganvalues and eigenvectors
	/*///*/*//*//*//*/*//*/*//*/*//*/*//*//*//*/*//*/*//*/*//*/*//*/



	MatrixXd covmat(num_columns-1,num_columns-1);
	for(int i=1; i<num_columns; i++){ //exclude side headers at 0
			for(int j=1; j<num_columns; j++){//exclude side headers at 0
				covmat(i-1,j-1)=covariancematrix[i][j];
			}
	}


	//MatrixXd eigencalc(MatrixXd covmat, const int rowscols){
	
	/*/*//*/*//*//*//*/*//*/*//*/*//*/*//*//*//*/*//*/*//*/*//*/*//*/
							The Power Method
	/*///*/*//*//*//*/*//*/*//*/*//*/*//*//*//*/*//*/*//*/*//*/*//*/

	double beta,betaprevious,error,normalise;
	MatrixXd Vector(num_columns-1,1);
	MatrixXd VolMatrix(num_columns-1,3);
	MatrixXd temp=covmat;//we will work on this temp matrix


	for(int k=0; k<3; k++){
		
		for(int i=0; i<num_columns-1; i++){
			Vector(i,0)=1;//first reset the Vector. This is our initial guess for the eigenvector
		}
		
		betaprevious=0;
		error=1;//just to initialise it
		
		while(error>0.000000001){

			Vector=temp*Vector;//we iterate over Vector
			beta=Vector.maxCoeff();//this is the element of the vector with the largest modulus
			for(int j=0; j<num_columns-1; j++){
				Vector(j,0)=Vector(j,0)/beta;
			}

			error=abs(beta-betaprevious);//recalc the error
			betaprevious=beta;//reset beta
		}

		normalise=0;//now we create the constant for normalising the vector
		for(int i=0; i<num_columns-1; i++){
			normalise=normalise+Vector(i,0)*Vector(i,0);
		}
		normalise=sqrt(normalise);

		for(int i=0; i<num_columns-1; i++){
			Vector(i,0)=sqrt(beta)*Vector(i,0)/normalise;//create the vol
			VolMatrix(i,k)=Vector(i,0);
		}

		//now find the new matrix M-lamda*v1*v1 transpose
		MatrixXd newmatrix(num_columns-1,num_columns-1);
		
		for(int i=0; i<num_columns-1; i++){
			for(int j=0; j<num_columns-1; j++){
				newmatrix(i,j)= temp(i,j)-Vector(i,0)*Vector(j,0);
			}
		}
		temp=newmatrix;

	}
	//}


	//Out put the vol matrix
	myfile.open ("vols.csv");
	for(int i=0; i<num_columns-1; i++){
		for(int j=0; j<3; j++){ 
			myfile << VolMatrix(i,j) << ",";
		}
		myfile << "\n";
	}

	myfile.close();


	/*/*//*/*//*//*//*/*//*/*//*/*//*/*//*//*//*/*//*/*//*/*//*/*//*/
	Now again using the MatrixXd methods.
	/*///*/*//*//*//*/*//*/*//*/*//*/*//*//*//*/*//*/*//*/*//*/*//*/
	
	/*EigenSolver<MatrixXd> eigensolver(covmat);
	if (eigensolver.info() != Success) abort();

	//Out put the eigenvalues and eigenvectors
	myfile.open ("eigen.csv");

	//eigenvalues
	for(int i=0; i<eigensolver.eigenvalues().size(); i++){
		myfile << eigensolver.eigenvalues()[i].real() << ",";
	}
	myfile << "\n";
	myfile << "\n";


	//eigenvectors
	for(int i=0; i<eigensolver.eigenvectors().rows(); i++){ 
			for(int j=0; j<eigensolver.eigenvectors().cols(); j++){
				myfile << eigensolver.eigenvectors()(i,j).real() << ",";
			}
			myfile << "\n";
	}

	myfile.close();*/

	

	/*/*//*/*//*//*//*/*//*/*//*/*//*/*//*//*//*/*//*/*//*/*//*/*//*/
	Step 8: Curve Fitting
	/*///*/*//*//*//*/*//*/*//*/*//*/*//*//*//*/*//*/*//*/*//*/*//*/
	
	int n=num_columns-1;  //n is the number of observations
	int eigenpos=0;//the position we are currently fitting. 0 is position PC1

	//now we have the matrices of Y, alpha and Vandermonde for each component.
	MatrixXd Y1(n,1);
	MatrixXd Y2(n,1);
	MatrixXd Y3(n,1);
	MatrixXd Vandermonde1(n,m1);
	MatrixXd Vandermonde2(n,m);
	MatrixXd Vandermonde3(n,m);

	
	
	//First fit number 1, ie position 0
	//Each of the Y components are the eigenvector times the square root of the eigen value.
	for(int i=0; i<n; i++){ 
		Y1(i,0)=VolMatrix(i,eigenpos);
		//number 1 so position 0			
	}
	//then we populate the vandermonde matrix
	for(int i=0; i<n; i++){ 
			for(int j=0; j<m1; j++){
				Vandermonde1(i,j)=pow(log(covariancematrix[i+1][0]),j);	
				//special case, we are using natural log of the maturity
			}
	}
	//now populate the matrix of polynomial coefficients using the formula described in the report
	polyCoeffic1 = ((Vandermonde1.transpose() * Vandermonde1).inverse() * Vandermonde1.transpose()) * Y1;

	//Now fit number 2, ie position 1
	eigenpos=1;

	for(int i=0; i<n; i++){ 
		Y2(i,0)=VolMatrix(i,eigenpos);
		//number 2 so position 1				
	}
	for(int i=0; i<n; i++){ 
		for(int j=0; j<m; j++){
			Vandermonde2(i,j)=pow(covariancematrix[i+1][0],j);	
		}
	}

	polyCoeffic2 = ((Vandermonde2.transpose() * Vandermonde2).inverse() * Vandermonde2.transpose()) * Y2;
	
	//Now fit number 3, ie position 2
	eigenpos=2;

	for(int i=0; i<n; i++){ 
		Y3(i,0)=VolMatrix(i,eigenpos);
		//number 3 so position 2			
	}
	
	for(int i=0; i<n; i++){ 
		for(int j=0; j<m; j++){
			Vandermonde3(i,j)=pow(covariancematrix[i+1][0],j);					
		}
	}
	polyCoeffic3 = ((Vandermonde3.transpose() * Vandermonde3).inverse() * Vandermonde3.transpose()) * Y3;
	
	//now output it all to a file.
	myfile.open ("fitteddata.csv");
	myfile <<"maturity,vol1,vol1fitted,vol2,vol2fitted,vol3,vol3fitted\n";
	for(int i=0; i<n; i++){ 
		myfile << covariancematrix[i+1][0]<< "," 
			   << Y1(i,0)<< "," << polyCoeffic1(0,0)+polyCoeffic1(1,0)*log(covariancematrix[i+1][0]) << "," 
			   << Y2(i,0)<< "," << polyCoeffic2(0,0)+polyCoeffic2(1,0)*covariancematrix[i+1][0]+polyCoeffic2(2,0)*pow(covariancematrix[i+1][0],2)+polyCoeffic2(3,0)*pow(covariancematrix[i+1][0],3) << "," 
		      << Y3(i,0)<< "," << polyCoeffic3(0,0)+polyCoeffic3(1,0)*covariancematrix[i+1][0]+polyCoeffic3(2,0)*pow(covariancematrix[i+1][0],2)+polyCoeffic3(3,0)*pow(covariancematrix[i+1][0],3) << "\n";
	}
	myfile << "\n";
	myfile.close();

	//cin.get();// this is just to keep the console window open
	

	/*/*//*/*//*//*//*/*//*/*//*/*//*/*//*//*//*/*//*/*//*/*//*/*//*/
	Polynomials n stuff
	/*///*/*//*//*//*/*//*/*//*/*//*/*//*//*//*/*//*/*//*/*//*/*//*/
	/*int m=4;
	int n=5;
	MatrixXd polyCoeffic(m,1);
	polyCoeffic(0,0)=0;
	polyCoeffic(1,0)=0;
	polyCoeffic(2,0)=0;
	polyCoeffic(3,0)=0;
	MatrixXd X(n,1);
	X(0,0)=1;
	X(1,0)=2;
	X(2,0)=3;
	X(3,0)=4;
	X(4,0)=5;
	MatrixXd Y(n,1);
	Y(0,0)=1;
	Y(1,0)=2;
	Y(2,0)=3;
	Y(3,0)=4;
	Y(4,0)=5;
	MatrixXd Vandermonde(n,m);
	for(int i=0; i<n; i++){ 
			for(int j=0; j<m; j++){
				Vandermonde(i,j)=pow(X(i,0),j);
				cout << "Vandermonde (" << i << "," << j << ") is "<< Vandermonde(i,j) << '\n';

			}
	}
	polyCoeffic = ((Vandermonde.transpose() * Vandermonde).inverse() * Vandermonde.transpose()) * Y;
	
	for(int j=0; j<m; j++){

				cout << "Poly Coeffs " << j << " is "<< polyCoeffic(j,0) << '\n';

	}

	cin.get();// this is just to keep the console window open*/

	/*/*//*/*//*//*//*/*//*/*//*/*//*/*//*//*//*/*//*/*//*/*//*/*//*/
	//this is just for testing:
	
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0.0,1.0);
	double randomtest1 = distribution(generator);
	double randomtest2 = distribution(generator);
	double randomtest3 = distribution(generator);
	double randomtest4 = distribution(generator);
	double randomtest5 = distribution(generator);
	while (true){
	randomtest1 = distribution(generator);
	randomtest2 = distribution(generator);
	randomtest3 = distribution(generator);
	randomtest4 = distribution(generator);
	randomtest5 = distribution(generator);
	cout << "random test 1 is:" << randomtest1 << endl;
	cout << "random test 2 is:" << randomtest2 << endl;
	cout << "random test 3 is:" << randomtest3 << endl;
	cout << "random test 4 is:" << randomtest4 << endl;
	cout << "random test 5 is:" << randomtest5 << endl;
	system("pause");
	}
	/*///*/*//*//*//*/*//*/*//*/*//*/*//*//*//*/*//*/*//*/*//*/*//*/

	/*/*//*/*//*//*//*/*//*/*//*/*//*/*//*//*//*/*//*/*//*/*//*/*//*/
	Steip nine: Now we make the final Monte Carlo matrix.
	/*///*/*//*//*//*/*//*/*//*/*//*/*//*//*//*/*//*/*//*/*//*/*//*/


	const int row_steps=1002;// 10 years so 10*100 (+2 for header info and last row of observations)
	const int iterations=1000;
	const double dT= 0.01;
	double T;
	int on_row_number;//which row number we are currently on
	vector < vector < vector<double> > > MC;

	//MatrixXd MC(iterations,row_steps,num_columns);// (iterations,row_steps, columns)
		//MC is a vector of interations
	//each iteration is a vector of rows
	//each row is a vector of doubles

	//This is for the random number generation. ie the pricipal components.
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0.0,1.0);
	
		//Now we populate each the Monte Carlo (MC) matrix for each iteration
		vector < vector<double> > iteration;
		double zcBondPrice[iterations]; //and array of each value of the bond price.... for bond pricing
		double zcBondPriceFinal=0;//initialise the final price...for bond pricing
		const int capLengthPeriods=4;//  4 means a 2 year capfor 6 month
		const int tenor=6;// tenor in months
		int stepsPerPeriod;
		if(tenor<13)//fixes a bug where the calculation below will only work up to one year (ie 12/12). 13 onwards requires the second calculation. Just a c++ quirk.
		{
			stepsPerPeriod=(1/dT)/(12/tenor);//number of steps that make up a period. In this case its 6 months with dt=0.01 so 50 altogether. 1/dT is 100 remember
		}else
		{
			stepsPerPeriod=(1/dT)*(tenor/12);//number of steps that make up a period. In this case its 6 months with dt=0.01 so 50 altogether. 1/dT is 100 remember
		}
		cout << "Steps per period is " << stepsPerPeriod <<'\n';
		
		double capStrike=0.02;//2% strike on the cap
		double capPriceFinal=0;
		for(int n=0; n<iterations; n++){ 

			//first initialise the variables
			on_row_number=0;
			T=0;
			cout <<"We are on iteration: "<< n << '\n';//this is just for testing

			vector<double> row;//create a row to put the first line into
			
			//first populate the top row with the header info
			//first come row number 0
			for(int j=0; j<num_columns; j++){//then for each column add the top line
				row.push_back(differences[0][j]);	//create the 0th row
			}
			
			iteration.push_back(row); //push the first row into the given iteration
			//MC[n].push_back(row);//push the first row into the given iteration
			
			row.clear(); //empty the vector
			on_row_number+=1;//now at 1

			//now row 1 which is pre determined
			row.push_back(T);//add the tenor to the row, currently zero
			for(int j=1; j<num_columns; j++){//then for each column from 1 onwards since we already added the 0th
				
				row.push_back((data[num_rows-1][j])/100);	//create the 1st row. Dividing by a hundred to make it a percentage
			}
			iteration.push_back(row);//push the first row into the given iteration
			T+=dT;//increment T
			row.clear();//clear the row
			row.push_back(T);//add the tenor to the row
			on_row_number+=1;//incremement the number of rows now at 2

			//This is for the random number generation. ie the pricipal components.
			//std::default_random_engine generator;
			//std::normal_distribution<double> distribution(0.0,1.0);
			vector<double> dX_1;
			vector<double> dX_2;
			vector<double> dX_3;
			for(int i=0; i<row_steps-2; i++){ //the -2 is there cause the header AND the first line dont need an RNG
				//for each of the dX_i we fill them up with 1000 increments of uncorrelated Brownian Motions
				dX_1.push_back(distribution(generator));
				dX_2.push_back(distribution(generator));
				dX_3.push_back(distribution(generator));
			}
			for(int k=on_row_number; k<row_steps; k++){
				for(int j=1; j<num_columns; j++){//then for each column 
					// F + drift*dt+SUM(vol*dX_i)*SQRT(dt)+dF/dtau*dt
					double F=iteration[on_row_number-1][j];
					double drift=M(iteration[0][j]);//MC[n][0][j] = the tenor
					double dF;
					if((j+1)!=num_columns){//just checking we are not at the edge
						dF=(iteration[on_row_number-1][j+1]-iteration[on_row_number-1][j]);
					}else{//if we are at the edge we have to modify the formula a bit.
						dF=(iteration[on_row_number-1][j]-iteration[on_row_number-1][j-1]);
					}

					double dTau;
					if((j+1)!=num_columns){//just checking we are not at the edge
						dTau=(iteration[0][j+1]-iteration[0][j]);//for this one its always row zero we are dealing with
					}else{//if we are at the edge we have to modify the formula a bit.
						dTau=(iteration[0][j]-iteration[0][j-1]);
					}
				
					//For this formula we use on_row_number-2 as the dX_i vectors are of length row_steps-2. 
					double formula=F+drift*dT+(Vol_1(iteration[0][j])*dX_1[on_row_number-2]+Vol_2(iteration[0][j])*dX_2[on_row_number-2]+Vol_3(iteration[0][j])*dX_3[on_row_number-2])*sqrt(dT)+dF/dTau*dT;
					row.push_back(formula);	
				}
				iteration.push_back(row);//push the row into the interation.
				T+=dT;//increment T
				row.clear();//clear the row
				row.push_back(T);//add the tenor to the row
				on_row_number+=1;//incremement the number of rows
			}
			MC.push_back(iteration);
			iteration.clear();//empty the vector

			//Ok now we actually price the derivatives. First up is the ZC bond
			zcBondPrice[n]=0;//initialise
			for(int j=1; j<stepsPerPeriod+1; j++){//stepsPerPeriod is 50 if its 6 months just cause its a 6 month bond say
				zcBondPrice[n]-=MC[n][j][1]*dT;// 1 cause thats the column number. 
			}
			zcBondPrice[n]=exp(zcBondPrice[n]);
			cout <<"Bond price for iteration "<< n << " is "<< zcBondPrice[n] <<'\n';
				
			zcBondPriceFinal+=zcBondPrice[n];//sum of all the bond prices
			cout << "Bond rolling average is: " << zcBondPriceFinal/(n+1) <<'\n';

			cout << "Tenor is: " << tenor <<'\n';
			
			//now for the cap
			double tempCapPrice=0;//just for output	
			double capDfs[capLengthPeriods];//initialise the discount factors for the cap. capLengthPeriods represents 4 perdiods, or 2 years
			double capCashflows[capLengthPeriods];//initialise the actual cashflows
			for(int p=0; p<capLengthPeriods; p++){
				capDfs[p]=0;//initilialise the discount factors
				double tempLibor=0;//just for use in the following function
				//now we get the actual discount factors
				for(int j=1; j<(stepsPerPeriod*(p+1))+1; j++){//50 just cause its a 6 month bond say so like 1 to 51 then 1 to 101 etc
					capDfs[p]-=MC[n][j][1]*dT;// 1 cause we use the shortest tenor for the DFs
				}
				capDfs[p]=exp(capDfs[p]);//this will give us the DF for period p on interation n
				//for(int q=1; q<stepsPerPeriod+1; q++){
				//	tempLibor+=MC[n][stepsPerPeriod*p+q][2]*0.01;//2 because column 2 for 6 month libor
				//}
				//tempLibor=2*sqrt(exp((12/tenor)*tempLibor)-1);
				//lets just try the averages method
				for(int q=1; q<stepsPerPeriod+1; q++){
					tempLibor+=MC[n][stepsPerPeriod*p+q+1][2];//2 because column 2 for 6 month libor
				}
				tempLibor=tempLibor/stepsPerPeriod;

				capCashflows[p]=capDfs[p]*max((tempLibor-capStrike),0.0)*tenor/12;//2 cause that the 6month column number
				capPriceFinal+=capCashflows[p];
				tempCapPrice+=capCashflows[p];
			}
			cout <<"Cap length in periods is "<< capLengthPeriods <<" cap strike is "<< capStrike << '\n';
			cout <<"Cap price for iteration "<< n << " is "<< tempCapPrice <<'\n';

			cout << "Cap rolling average is: " << capPriceFinal/(n+1) <<'\n';
			
		}
		zcBondPriceFinal=zcBondPriceFinal/iterations;//and divide by the number of simulations to get the average price
		capPriceFinal=capPriceFinal/iterations;

		cout << "Final Value of Bond is: " << zcBondPriceFinal <<'\n';
		cout << "Final Value of Cap is: " << capPriceFinal <<'\n';
	/*/*//*/*//*//*//*/*//*/*//*/*//*/*//*//*//*/*//*/*//*/*//*/*//*/
	Ok now we actually price the derivatives. First up is the ZC bond
	
	cout <<"The value of MC 000 is "<<  MC[0][0][0] <<'\n';
	//cout <<"The value of MC 101010 is "<<  MC.at(10).at(10).at(10) <<'\n';
	//cout <<"The value of MC 202020 is "<<  MC.at(20).at(20).at(20) <<'\n';
	//cout <<"The value of MC 303030 is "<<  MC.at(30).at(30).at(30)<<'\n';
	double zcBondPrice[iterations]; //and array of each value of the bond price
	for(int i=0; i<iterations; i++){
		zcBondPrice[i]=0;//initialise
	}
	
	double zcBondPriceFinal=0;//initialise the final price
	myfile.open ("mcdatatest.csv");//test output
	for(int i=0; i<iterations; i++){
		for(int j=1; j<51; j++){//50 just cause its a 5 year bond say

		zcBondPrice[i]-=MC[i][j][1]*dT;// 1 cause thats the column number

	
		//this is just test output
		
		
		myfile << MC[i][j][1] << ",";//test output
		
		

		}
		zcBondPrice[i]=exp(zcBondPrice[i]);
		cout <<"Bond price for iteration "<< i << " is "<< zcBondPrice[i] <<'\n';
		myfile << "\n";//test output
		zcBondPriceFinal+=zcBondPrice[i];//sum of all the bond prices
		cout << "rolling average is: " << zcBondPriceFinal/(i+1) <<'\n';
	}
	myfile.close();//test output



	zcBondPriceFinal=zcBondPriceFinal/iterations;//and divide by the number of simulations to get the average price

	cout << "Final Value of Bond is: " << zcBondPriceFinal <<'\n';
	/*///*/*//*//*//*/*//*/*//*/*//*/*//*//*//*/*//*/*//*/*//*/*//*/
	cin.get();// this is just to keep the console window open
    return 0;
}


