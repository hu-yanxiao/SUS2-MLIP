/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Alexander Shapeev, Evgeny Podryabinkin, Ivan Novikov
 */

#ifndef MLIP_BFGS_H
#define MLIP_BFGS_H

// #define MLIP_LINESEARCH_DEBUG

#include <cmath>
#include <iostream>
#include "multidimensional_arrays.h"
#include "utils.h"


//! Linesearch

//! Assumption: f'(0)<0
//! A tyipcal usage:
//! \code
//! Linesearch ls; // minimizes f(x) = 0
//! while(!converge) {
//!		ls.Iterate(f(ls.x()),f'(ls.x())); // sets new ls.x
//! }
//! \endcode
class Linesearch
{
protected:
	double curr_x;

	double left_x; //! left bound for the root
	double left_f;
	double left_g;
	double right_x; //! right bound for the root, sometimes right_x has to expand - check right_g
	double right_f;
	double right_g;

	double prev_x;
	double prev_f;
	double prev_g;

public:

	//! resets linesearch for minimizing another function.
	void Reset(double _initial_step = 1.0) {
		left_x = curr_x = 0.0;
		right_x = _initial_step;
		right_f = HUGE_DOUBLE;
		right_g = HUGE_DOUBLE;
	}

	Linesearch() { Reset(); }

	double x() { return curr_x; }

	void ReduceStep(double ratio = 0.25) {
		curr_x = prev_x + ratio * (curr_x - prev_x);
	}

	//! make an iteration, sets new x()
	void Iterate(double curr_f, double curr_g) {
		if (curr_x == 0) {
			// this is the very first iteration
			left_x = curr_x;
			left_f = curr_f;
			left_g = curr_g;

			if (curr_g > 0) 
				ERROR("Linesearch with increasing funcion!");

			prev_x = curr_x;
			prev_f = curr_f;
			prev_g = curr_g;

			curr_x = right_x;
			return;
		}
#ifdef MLIP_LINESEARCH_DEBUG
		std::cerr.precision(16);
		std::cerr << "ls dump: {"
			<< left_x << ", "
			<< curr_x << ", "
			<< right_x << ", "
			<< left_f << ", "
			<< curr_f << ", "
			<< right_f << ", "
			<< left_g << ", "
			<< curr_g << ", "
			<< right_g << ", "
			<< prev_x << ", "
			<< prev_f << ", "
			<< prev_g << "}" 
			<< std::endl;
#endif // MLIP_LINESEARCH_DEBUG

		// Now prev_x, prev_f, prev_g are set
		if (prev_x == curr_x) ERROR("prev_x == curr_x");

		if (curr_x == right_x || (curr_x > left_x && right_f == 9e99)) {
			right_x = curr_x;
			right_f = curr_f;
			right_g = curr_g;
		}
		// right_... are already set here

		// update left_ and right_
		if (right_g < 0 && right_f < left_f) {
			if (curr_x > right_x) {
				left_x = right_x;
				left_f = right_f;
				left_g = right_g;
				right_x = curr_x;
				right_f = curr_f;
				right_g = curr_g;
			}
		} else {

			// update left_x
			if (curr_g < 0 && curr_f < left_f && curr_x > left_x) {
				if (curr_x > right_x) {
					if (curr_f > right_f) {
						prev_x = curr_x; prev_f = curr_f; prev_g = curr_g;
						left_x = right_x; left_f = right_f; left_g = right_g;
						right_x = curr_x; right_f = curr_f; right_g = curr_g;
						curr_x = 0.5 * (left_x + right_x);
						return;
					}
					right_x = curr_x; right_f = curr_f; right_g = curr_g;
				}
				else {
					left_x = curr_x; left_f = curr_f; left_g = curr_g;
				}
			}

			// update right_x
			if (curr_g > 0 && curr_x < right_x) {
				right_x = curr_x;
				right_f = curr_f;
				right_g = curr_g;
			}
		}
		// check weak Wolfe conditions
		if (curr_f > left_f + 0.1 * left_g * (curr_x - left_x)) {
			// a robust iteration: we reconstruct log(f - delta_f) with a parabola
			// with left_f, left_g, curr_f

			double slope = -left_g * (curr_x - left_x);
			double f0 = std::log(slope);
			double g0 = left_g / slope;
			double f1 = std::log(curr_f - left_f + slope);

			double next_x = left_x - 0.5 * g0 * (curr_x - left_x) * (curr_x - left_x)
				/ (f1 - f0 - g0 * (curr_x - left_x));

			if (next_x > right_x && right_g > 0) {
				// super robust here: dihotomy
				next_x = left_x + 0.5 * (curr_x - left_x);
			} else if ((next_x - left_x > 3.0 * right_x - 2.0 * left_x) && right_g < 0) {
				// expand the right boundary, but not too much
				next_x = left_x + 3.0 * (right_x - left_x);
			}

#ifdef MLIP_LINESEARCH_DEBUG
			std::cerr << "ls: robust, left_f - delta_f: " << slope;
			std::cerr << " left_x: " << left_x;
			std::cerr << " curr_x: " << curr_x;
			std::cerr << " next_x: " << next_x;
			std::cerr << std::endl;
#endif // MLIP_LINESEARCH_DEBUG

			prev_x = curr_x; prev_f = curr_f; prev_g = curr_g;
			curr_x = next_x;
			return;
		}

		if ((curr_g - prev_g) / (curr_x - prev_x) < 0.0) {
			// second derivative < 0, careful here
			if (right_g < 0) {
				prev_x = curr_x; prev_f = curr_f; prev_g = curr_g;
				curr_x = 3 * right_x - 2 * left_x;
				return;
			}
			// Now right_g < 0
			prev_x = curr_x; prev_f = curr_f; prev_g = curr_g;
			curr_x += 0.5*(right_x - curr_x);
			return;
		}

		// we can now do a secant method
		double new_x = curr_x - curr_g * (curr_x - prev_x) / (curr_g - prev_g);
		prev_x = curr_x; prev_f = curr_f; prev_g = curr_g;
		curr_x = new_x;
		if (curr_x > right_x) {
			if (right_g > 0) {
				curr_x = 0.5*left_x + 0.5*right_x;
			} else {
				if (curr_x > 3 * right_x - 2 * left_x)
					curr_x = 3 * right_x - 2 * left_x;
			}
			return;
		}
		if (curr_x < left_x) {
#ifdef MLIP_LINESEARCH_DEBUG
			std::cerr << "ls: curr_x < left_x, making dihotomy" << std::endl;
#endif // MLIP_LINESEARCH_DEBUG
			prev_x = curr_x; prev_f = curr_f; prev_g = curr_g;
			curr_x = 0.5*left_x + 0.5*right_x;
		return;
		}
	}
};

//! The BFGS interface.

//! A tyipcal usage:
//! \code
//! Array1D x(n);
//! <fill x with initial values>
//! bfgs.Set_x(x);									// Initiates internal state of BFGS engine
//! // NOTE: bfgs.x should be changed "by hand" from this point
//! while (<convergence is not achieved>) {
//!   if (<time-to-time> && is_in_linesearch)			// note: we may do some special expensive trick to find better_x
//!		set_x(better_x);								//       best to set while outside linesearch
//!   <calculate value of funcion f(x) and gradient g(x)>
//!   x = bfgs.Iterate(f, g);
//!   while(<the step is too large>)
//!     bfgs.ReduceStep();
//! }
//! \endcode
//!
//! For restarting iterations (if inv.hess becomes bad) do
//! \code
//! bfgs.Restart();
//! bfgs.inv_hess[..][..] = ..; // set inv_hess manually
//! bfgs.Set_x(x);
//!
//! If bfgs.is_in_linesearch was set false after bfgs.Iterate(), then the old bfgs.x was the last
//! iteration of linesearch, and new bfgs.x is the first iteration of linesearch
//!
//! \endcode
class BFGS
{
private:
	inline double ScalarProd(const double* v1, const double* v2);
	Array1D delta_grad;									//!< needed for inv_hess update
	Array1D yC;											//!< needed for inv_hess update

protected:
	int size = 0;										//!< size of x, g, etc.
	Array1D p;											//!< search (descent) direction
	double p_dot_g;

	Linesearch linesearch;								//!< linesearch engine

	Array1D x_start;									//!< linesearch initial x
	double f_start = HUGE_DOUBLE;						//!< linesearch initial f
	Array1D g_start;									//!< linesearch initial g -- needed for inv_hess update
	double p_dot_g_start;								//!< linesearch initial p_dot_g (g can change during the linesearch)

	Array1D x_;											//< current x
	bool is_in_linesearch_ = false;

	inline void SetStart(double f, const Array1D& g);	//! set initial point (x,g,f,p_dot_g) for linearsearch
	void UpdateInvHess(const Array1D& g);				//! update inv_hess based on g, g_start, alpha, p
	inline void Resize(int _size);						//! resize all vectors and matrices, sets inv_hess to the identity matrix

public:
	Array2D inv_hess;
	double lr = 100;
	int iter_step = 0;
	bool is_in_linesearch() { return is_in_linesearch_; }

	double wolfe_c1 = 0.1;
	double wolfe_c2 = 0.5;
	
	void Set_x(const double *x, int _size);				//! sets x and resets (i.e., quits) linesearch. Does not reset Hessian if size is not changed
	void Set_x(Array1D x) { Set_x(x.data(), (int)x.size()); };//! sets x and resets (i.e., quits) linesearch

	//! resets hess and restarts the iteration (i.e., quits linesearch)
	//! inv_hess can be set manually immediately after calling Restart()
	void Restart();

	double x(int i) { return x_[i]; }					//! read-only access to x
	const Array1D& Iterate(double f, const Array1D& g);	//! Make a BFGS iteration
	const Array1D& Iterate2(double _lr, const Array1D& g);	//! Make a BFGS iteration
	const Array1D& ReduceStep(double _coeff = 0.25);	//! Reduce the step (e.g., if x was unphysical)
};


inline double BFGS::ScalarProd(const double* v1, const double* v2) {
	double res = 0.0;
	for (int i = 0; i<size; i++)
		res += v1[i] * v2[i];
	return res;
}

inline void BFGS::SetStart(double f, const Array1D& g) {
	memcpy(&x_start[0], &x_[0], size * sizeof(double));
	memcpy(&g_start[0], &g[0], size * sizeof(double));
	f_start = f;
	p_dot_g_start = p_dot_g;
}

inline void BFGS::Resize(int _size) {
	if (size == _size) return;
	size = _size;
	x_.resize(size);
	x_start.resize(size);
	g_start.resize(size);
	delta_grad.resize(size);
	yC.resize(size);
	p.resize(size);
	inv_hess.resize(size, size);
	inv_hess.set(0.0);
	for (int i=0; i<size; i++)
		inv_hess(i, i) = 1.0;
}



#endif //MLIP_BFGS_H
