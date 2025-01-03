/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */

#include "matvec_oper.h"

using namespace std;

void matmat(const int n, const double * matr1, const double * matr2, double * matr_output)
{
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			matr_output[i*n+j] = 0.0;
			for (int m = 0; m < n; m++) {
				matr_output[i*n+j] += matr1[i*n+m]*matr2[m*n+j];
			}
		}
	}	
}

void matvec(const int n, const double * matr, const double * vec, double * vec_output)
{
	for (int ii = 0; ii < n; ii++) {
		vec_output[ii] = 0.0;
		for (int jj = 0; jj < n; jj++) {
			vec_output[ii] += matr[ii*n+jj]*vec[jj];
		}
	}
}

double vecvec(const int n, const double * vec1, const double * vec2)
{
	double sum = 0.0;

	for (int i = 0; i < n; i++)
		sum += vec1[i]*vec2[i];

	return sum;
}

void orthoproject(int n, double * matr)
{
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (i == j) matr[i*n+j] = 1.0-1.0/n;
			else matr[i*n+j] = 0-1.0/n;
		}
	}
}

void gen_matr(const int n, const double * orth_matr, const double * matrix_input, double * matrix)
{
	double * matrix_dummy;

	matrix_dummy = new double [n*n];
	
	matmat(n, orth_matr, matrix_input, matrix);

	for (int i = 0; i < n; i++) 
		for (int j = 0; j < n; j++) 
			matrix_dummy[i*n+j] = matrix[i*n+j];
			
	matmat(n, matrix_dummy, orth_matr, matrix);

	/*for (int i = 0; i < n; i++) {
		double sum = 0.0;
		for (int j = 0; j < n; j++) {
			//cout << matrix[i*n+j] << " ";
			sum += matrix[i*n+j];
		}	
		//cout << endl;
		cout << sum << endl;
	}*/

	delete [] matrix_dummy;
}
