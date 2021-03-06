/*
***********************************************************************************************************************************************************

												Gradient Descent Method
											  Header File Fitting Routine

Author: Omkar Junnarkar, Semester-3 MSc. Computational Material Science
Matriculation Nr.: 66157	Email: omkar.junnarkar@student.tu-freiberg.de
IDE : Microsoft Visual Studio 2019

*/
#include<iostream>
#include<Eigen/Dense>

using namespace std;
using namespace Eigen;

MatrixXd function_y(MatrixXd para, MatrixXd x);
MatrixXd getJacobianMatrix(MatrixXd para_est, MatrixXd deflection, MatrixXd ym, MatrixXd input);
MatrixXd GradientDescent(MatrixXd para_guess, MatrixXd deflection, MatrixXd ym, MatrixXd input);
