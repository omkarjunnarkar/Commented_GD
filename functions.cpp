/*
***********************************************************************************************************************************************************

												 Gradient Descent Method
												Fitting Routine Functions

Author: Omkar Junnarkar, Semester-3 MSc. Computational Material Science
Matriculation Nr.: 66157	Email: omkar.junnarkar@student.tu-freiberg.de
IDE : Microsoft Visual Studio 2019

iostream: For Input-Output of the C++ Program
Eigen/Dense: For Using Dense Matrix Interface of Linear Algebra Library Eigen
iomanip: To manipulate the number of decimal places in output
math: For arithmetic operations
fstream: To create the stream of file and write
functions.h: Contains the Fitting Routine Header
*/

#include<iostream>
#include<Eigen/Dense>
#include<iomanip>
#include<math.h>
#include<fstream>
#include"functions.h"

/*
To reduce the effort of specifying libraries/class for each command/data type
*/
using namespace std;
using namespace Eigen;

/*-----------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*
	Arguments : Parameters of Equation, Input(x) for Equation
	para( , ) : List of Parameters
	x( , ) : List of Input values for the equation

	Output : Matrix 'y', data points of the funcction y=f(parameters,x)
*/
MatrixXd function_y(MatrixXd para, MatrixXd x) {
	int number_of_data_points = x.rows();
	MatrixXd y(number_of_data_points, 1);
	for (int i = 0; i < number_of_data_points; i++) {

		//y(i, 0) = para(0, 0) * pow(x(i, 0), 2) + para(1, 0) * exp(pow(z(i, 0), 2) + 1) + para(2, 0);

		y(i, 0) = para(0, 0) * pow(x(i, 0), 3) + para(1, 0) * pow(x(i, 0), 2) + para(2, 0) * x(i, 0) + para(3, 0) * exp(para(4, 0) * x(i, 0) + para(5, 0));
		
		//y(i, 0) = para(0, 0) * pow(x(i, 0), 4) + para(1, 0) * exp(para(2, 0) * x(i, 0));
	}
	return y;
};

/*-----------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*
	Jacobian Matrix : the matrix containing values of First Order Partial Derivatives w.r.t all parameters at All Data Points
	Size : (Number of Data Points, Parameters)
	The Partial Derivatives of the functions with respect to each Parameter can be estimated by using Finite Difference Method as follows :

	Df/Da = [f(a + DELa) - f(a)]/DELa , Df/Db = [f(b + DELb) - f(b)]/DELb , Df/Dc = [f(c + DELc) - f(c)]/DELc
	where a,b,c are paramerters and DELa,DELb,DELc are the deflections (given by initial deflection)

	Thus by computing function values of these derivates at all data points 'x', All Elements of Jacobian can be obtained.
	[ Refer Report/Manual for Details ]
	y_deflected : Function value obtained by deflecting parameter

	Arguments: Estimated Parameters, Ínitial Deflection, 'y' measured, Input data 'x'
	Output : Jacobian Matrix
*/
MatrixXd getJacobianMatrix(MatrixXd para_est, MatrixXd deflection, MatrixXd ym, MatrixXd input) {

	MatrixXd Jacobian_Matrix(ym.rows(), para_est.rows());
	MatrixXd y = function_y(para_est, input);
	MatrixXd y_deflected(ym.rows(), 1);

	for (int i = 0; i < para_est.rows(); i++) {

		para_est(i, 0) = para_est(i, 0) + deflection(i, 0);		/*Changing the parameters one by one */

		y_deflected = function_y(para_est, input);				/*Computing the deflected function arrray */
		for (int j = 0; j < input.rows(); j++) {

			// [f(v, p + dp) - f(v, p) ] / [dp] 

			Jacobian_Matrix(j, i) = (y_deflected(j, 0) - y(j, 0)) / deflection(i, 0);
		}
		para_est(i, 0) = para_est(i, 0) - deflection(i, 0);		/*Bringing back the parametes to original value*/
	}
	return Jacobian_Matrix;
};

/*-----------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*
	Gradient Descent Method  :

	Arguments : Estimated Parameters, Initial Deflection of Parameters, Measured 'y' values, Data Input 'x'
	Output : True Parameters

	Computation Strategy & Variables:

	-> Difference between measured y values and estimated y values is computed (d) along with error
	-> Alpha is the 'Learning Rate' which is data sensitive. It defines the size of steps taken in the direction of descent.
	-> Change in Parameters (dp) is computed by ALPHA*J.transpose*d -> Parameters are updated and new error is computed
	-> Iterations continue untill problem converges

	Learning Rate has influence on the behaviour of routine. Initial value is set to 8e-8. In case of well behaved functions, alpha can be Increased to reach the solution faster.
	[ Refer Report/Manual for Details ]

*/

MatrixXd GradientDescent(MatrixXd para_guess, MatrixXd deflection, MatrixXd ym, MatrixXd input) {

	cout << "-> Entered Gradient Descent\n";

	ofstream errorfile("ErrorNorm.csv");
	/* Number of Parameters */
	int npara = para_guess.rows(), ndata = ym.rows();

	MatrixXd d(ndata, 1);
	MatrixXd J(ndata, npara);
	double error,error_gd;

	double alpha =8e-8;
	MatrixXd para_est = para_guess;			/* Estimated Parameters*/
	MatrixXd para_gd, y_est_gd, y_est;
	int maxiter = 100000, counter=0;		/* Maximum Iterations Allowed, Counter for Iterations */
	
	while (counter < maxiter) {
		
		cout << "--> Iteration : " << counter << endl;

		if (counter == 0) {					/* Only to be done for First Iteration */
			y_est = function_y(para_est, input);
			d = ym - y_est;
		}
		else {
			y_est = y_est_gd;
			para_est = para_gd;
			error = error_gd;
			d = ym - y_est;
		}

		J = getJacobianMatrix(para_est, deflection, ym,input);
		MatrixXd dp = alpha * J.transpose() * d;	/*Computing change in parameters*/
		//cout << "dp: \n" << dp;
		para_gd = para_est + dp;					/* Update of parameters*/
		y_est_gd= function_y(para_gd, input);		/* 'y' values from updated parameters*/
		MatrixXd d_gd = ym - y_est_gd;				/* difference betwen measured and computed values*/
		MatrixXd temp2 = d_gd.transpose() * d_gd;	/* Present Error */
		error_gd = temp2(0,0);	
		cout << "err= " << error_gd << endl;

		/* Stopping Criterion */
		if (error_gd < 1e-2) {
			counter = maxiter;
		}
		else counter++;
		
		/* Writing File */
		errorfile << error_gd << endl;
		
	}

	errorfile.close();

	/* Returning Final Set of Obtained Parameters */

	return para_gd;
};

/*-----------------------------------------------------------------------------------------------------------------------------------------------------------*/
