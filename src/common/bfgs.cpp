/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Alexander Shapeev, Evgeny Podryabinkin, Ivan Novikov
 */

#include <string>
#include <iostream>
#include "bfgs.h"


//! Input: x was set either before iterations or changed during iterations;
//! f, g are set for the current x before the call.
//! The function sets new x.
//! Internally, it changes the protected members
//! alpha and if outside linesearch then it changes p.
const Array1D& BFGS::Iterate(double f, const Array1D& g) {
	p_dot_g = ScalarProd(&g[0], &p[0]);

	// checking Wolfe conditions for the function
	if (f > f_start + wolfe_c1*linesearch.x()*p_dot_g_start
		|| std::abs(p_dot_g) > wolfe_c2*std::abs(p_dot_g_start)) {
		// Linesearch: increased or reduced too little, going to decrease the step
		is_in_linesearch_ = true;
		iter_step+=1;

		linesearch.Iterate(f, p_dot_g);
		for (int i = 0; i < size; i++) x_[i] = x_start[i] + linesearch.x() * p[i];
	} else {
		// if we were in linesearch, we say we are no longer in linesearch
		// else, we should say we ARE in linesearch
		// (not to be caught in an infinite cycle)
		is_in_linesearch_ = !is_in_linesearch_;
		iter_step = 0;

		UpdateInvHess(g);
		// Doing a proper (non-linesearch iteration)
		// forming a descent (hopefully) direction p
		FillWithZero(p);
		for (int i = 0; i < size; i++) // TODO: change to cblas_dgemv
			for (int j = 0; j < size; j++)
				p[i] -= inv_hess(i, j) * g[j];
		p_dot_g = ScalarProd(&g[0], &p[0]);
		linesearch.Reset();
		// setting initial point for linearsearch
		SetStart(f, g);

		if (p_dot_g > 0) {
			std::ofstream ofs("bfgs.log");
			ofs.precision(16);
			ofs << size << '\n';
			for(int i=0; i<size; i++)
				for(int j=0; j<size; j++)
					ofs << inv_hess(i,j) << " ";
			ofs.close();
			ERROR("BFGS: stepping in accend direction detected.");
		}

		linesearch.Iterate(f, p_dot_g);
		for (int i = 0; i < size; i++)
			x_[i] = x_start[i] + linesearch.x() * p[i];
	}

	return x_;
}


const Array1D& BFGS::Iterate2(double _lr, const Array1D& g) {
	for (int i = 0; i < size; i++) {
		p[i] -= inv_hess(i, i) * g[i];
		x_[i] += _lr * p[i];
	}
	
}


//! The step x is too far, make a smaller step (as a part of linesearch).
//! Sets new x and alpha
const Array1D& BFGS::ReduceStep(double _coeff) {
	linesearch.ReduceStep(_coeff);
	for (int i = 0; i < size; i++)
		x_[i] = x_start[i] + linesearch.x() * p[i];

	return x_;
}

void BFGS::UpdateInvHess(const Array1D& g)
{
	for (int i = 0; i<size; i++)
		delta_grad[i] = g[i] - g_start[i];

	const double alpha = linesearch.x();
	double py = 0.0;
	double yCy = 0.0;
	for (int i = 0; i<size; i++)
		py += alpha * p[i] * delta_grad[i];

	// if py==0 then this is the very first iteration and no updating inv_hess needed
	if (py == 0) return;

	FillWithZero(yC);
	for (int i = 0; i<size; i++) {
		for (int j = 0; j<size; j++)
			yC[i] += inv_hess(i, j) * delta_grad[j];
		yCy += yC[i] * delta_grad[i];
	}

	double foo = (py + yCy) / (py*py);
	for (int i = 0; i<size; i++)
		for (int j = 0; j<size; j++)
			inv_hess(i, j) += alpha*alpha*p[i] * p[j] * foo - alpha*(p[i] * yC[j] + yC[i] * p[j]) / py;
}

void BFGS::Set_x(const double * x, int _size) {
	Resize(_size);
	memcpy(&x_[0], &x[0], _size * sizeof(double));
	f_start = HUGE_DOUBLE;
	p_dot_g_start = HUGE_DOUBLE;

	// reset linesearch
	linesearch.Reset();
	is_in_linesearch_ = false;
}

void BFGS::Restart() {
	inv_hess.set(0.0);
	for (int i = 0; i < size; i++)
		inv_hess(i, i) = 1.0;
	f_start = HUGE_DOUBLE;
	p_dot_g_start = HUGE_DOUBLE;
	linesearch.Reset();
	is_in_linesearch_ = false;
}
