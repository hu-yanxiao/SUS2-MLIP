/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Alexander Shapeev, Evgeny Podryabinkin, Ivan Novikov
 */

#include <algorithm>
#include <string>
#include <iostream>
#include "bfgs.h"

#ifdef MLIP_MPI
#include <mpi.h>
#endif

#ifdef MLIP_INTEL_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif

namespace {

int FirstNonFinite(const Array1D& values, int size)
{
	for (int i = 0; i < size; ++i)
		if (!std::isfinite(values[i]))
			return i;
	return -1;
}

void CheckFiniteArray(const Array1D& values, int size, const std::string& name)
{
	const int bad_index = FirstNonFinite(values, size);
	if (bad_index >= 0)
		ERROR(name + " contains a non-finite value at index " + std::to_string(bad_index));
}

bool BFGSShouldBacktrackNonFinite(bool is_in_linesearch, const Linesearch& linesearch)
{
	return is_in_linesearch && linesearch.x() != 0.0;
}

} // namespace

void BFGS::UseDistributedDense(int rank, int size
#ifdef MLIP_MPI
	, MPI_Comm comm
#endif
)
{
#ifndef MLIP_MPI
	if (size > 1)
		ERROR("Distributed dense BFGS requires MPI support.");
#endif

	const bool new_use_distributed = (size > 1);
	const bool layout_changed = use_distributed_dense_ != new_use_distributed
		|| distributed_rank_ != rank
		|| distributed_size_ != std::max(size, 1);

	use_distributed_dense_ = new_use_distributed;
	distributed_rank_ = new_use_distributed ? rank : 0;
	distributed_size_ = new_use_distributed ? size : 1;
#ifdef MLIP_MPI
	distributed_comm_ = new_use_distributed ? comm : MPI_COMM_WORLD;
#endif

	if (layout_changed && this->size > 0) {
		const int old_size = this->size;
		this->size = 0;
		Resize(old_size);
	}
}

void BFGS::UpdateDistributedLayout()
{
	if (distributed_size_ <= 0)
		distributed_size_ = 1;

	distributed_row_counts_.assign(distributed_size_, 0);
	distributed_row_displs_.assign(distributed_size_, 0);
	distributed_elem_counts_.assign(distributed_size_, 0);
	distributed_elem_displs_.assign(distributed_size_, 0);

	const int base_row_count = size / distributed_size_;
	const int remainder = size % distributed_size_;
	int row_displ = 0;
	int elem_displ = 0;
	for (int rank = 0; rank < distributed_size_; ++rank) {
		const int row_count = base_row_count + (rank < remainder ? 1 : 0);
		distributed_row_counts_[rank] = row_count;
		distributed_row_displs_[rank] = row_displ;
		distributed_elem_counts_[rank] = row_count * size;
		distributed_elem_displs_[rank] = elem_displ;
		row_displ += row_count;
		elem_displ += row_count * size;
	}

	distributed_row_start_ = distributed_row_displs_[distributed_rank_];
	distributed_row_count_ = distributed_row_counts_[distributed_rank_];
	distributed_inv_hess_rows_.assign(distributed_row_count_ * size, 0.0);
	distributed_local_buffer_.assign(distributed_row_count_, 0.0);

	for (int local_row = 0; local_row < distributed_row_count_; ++local_row) {
		const int global_row = distributed_row_start_ + local_row;
		distributed_inv_hess_rows_[local_row * size + global_row] = 1.0;
	}
}

void BFGS::DenseMatVec(const Array1D& v, Array1D& out)
{
	if ((int)out.size() != size)
		out.resize(size);

	if (!use_distributed_dense_) {
#ifdef MLIP_INTEL_MKL
		cblas_dsymv(CBLAS_ORDER::CblasRowMajor,
					CBLAS_UPLO::CblasUpper,
					size,
					1.0,
					&inv_hess(0, 0), size,
					&v[0], 1,
					0.0,
					&out[0], 1);
#else
		cblas_dgemv(CBLAS_ORDER::CblasRowMajor,
					CBLAS_TRANSPOSE::CblasNoTrans,
					size, size,
					1.0,
					&inv_hess(0, 0), size,
					&v[0], 1,
					0.0,
					&out[0], 1);
#endif
		return;
	}

#ifndef MLIP_MPI
	ERROR("Distributed dense BFGS requires MPI support.");
#else
	if (distributed_row_count_ > 0) {
		cblas_dgemv(CBLAS_ORDER::CblasRowMajor,
					CBLAS_TRANSPOSE::CblasNoTrans,
					distributed_row_count_, size,
					1.0,
					distributed_inv_hess_rows_.data(), size,
					&v[0], 1,
					0.0,
					distributed_local_buffer_.data(), 1);
	}

	MPI_Allgatherv(distributed_row_count_ > 0 ? distributed_local_buffer_.data() : nullptr,
				   distributed_row_count_,
				   MPI_DOUBLE,
				   &out[0],
				   distributed_row_counts_.data(),
				   distributed_row_displs_.data(),
				   MPI_DOUBLE,
				   distributed_comm_);
#endif
}

void BFGS::MirrorUpperToLower()
{
	for (int i = 0; i < size; ++i)
		for (int j = i + 1; j < size; ++j)
			inv_hess(j, i) = inv_hess(i, j);
}

void BFGS::FormDenseDirection(const Array1D& g)
{
	DenseMatVec(g, p);
	for (int i = 0; i < size; ++i)
		p[i] = -p[i];
}

void BFGS::SetInvHessDiagonal(const Array1D& diag)
{
	if ((int)diag.size() != size)
		ERROR("BFGS::SetInvHessDiagonal(): diagonal size mismatch.");

	if (!use_distributed_dense_) {
		inv_hess.set(0.0);
		for (int i = 0; i < size; ++i)
			inv_hess(i, i) = diag[i];
		MirrorUpperToLower();
		return;
	}

	std::fill(distributed_inv_hess_rows_.begin(), distributed_inv_hess_rows_.end(), 0.0);
	for (int local_row = 0; local_row < distributed_row_count_; ++local_row) {
		const int global_row = distributed_row_start_ + local_row;
		distributed_inv_hess_rows_[local_row * size + global_row] = diag[global_row];
	}
}

void BFGS::MaskCoordinates(const std::vector<int>& indices)
{
	if (indices.empty())
		return;

	if ((int)mask_workspace_.size() != size)
		mask_workspace_.assign(size, 0);
	else
		std::fill(mask_workspace_.begin(), mask_workspace_.end(), 0);
	for (int idx : indices) {
		if (idx < 0 || idx >= size)
			ERROR("BFGS::MaskCoordinates(): index out of range.");
		mask_workspace_[idx] = 1;
	}

	if (!use_distributed_dense_) {
		for (int idx : indices) {
			for (int j = 0; j < size; ++j) {
				inv_hess(idx, j) = 0.0;
				inv_hess(j, idx) = 0.0;
			}
			inv_hess(idx, idx) = 1.0;
		}
	} else {
		for (int local_row = 0; local_row < distributed_row_count_; ++local_row) {
			const int global_row = distributed_row_start_ + local_row;
			double* row = distributed_inv_hess_rows_.data() + local_row * size;
			if (mask_workspace_[global_row]) {
				std::fill(row, row + size, 0.0);
				row[global_row] = 1.0;
				continue;
			}
			for (int idx : indices)
				row[idx] = 0.0;
		}
	}

	for (int idx : indices)
		p[idx] = 0.0;

	if (!use_distributed_dense_)
		MirrorUpperToLower();
}


//! Input: x was set either before iterations or changed during iterations;
//! f, g are set for the current x before the call.
//! The function sets new x.
//! Internally, it changes the protected members
//! alpha and if outside linesearch then it changes p.
const Array1D& BFGS::Iterate(double f, const Array1D& g) {
	const int bad_grad_index = FirstNonFinite(g, size);
	if (!std::isfinite(f) || bad_grad_index >= 0) {
		if (BFGSShouldBacktrackNonFinite(is_in_linesearch_, linesearch)) {
			is_in_linesearch_ = true;
			iter_step += 1;
			linesearch.ReduceStep(0.25);
			for (int i = 0; i < size; i++)
				x_[i] = x_start[i] + linesearch.x() * p[i];
			CheckFiniteArray(x_, size, "BFGS backtracked x");
			return x_;
		}
		if (!std::isfinite(f))
			ERROR("BFGS received a non-finite function value outside line search.");
		ERROR("BFGS received a non-finite gradient outside line search at index "
		      + std::to_string(bad_grad_index));
	}

	p_dot_g = ScalarProd(&g[0], &p[0]);
	if (!std::isfinite(p_dot_g))
		ERROR("BFGS produced a non-finite directional derivative.");

	// checking Wolfe conditions for the function
	if (f > f_start + wolfe_c1*linesearch.x()*p_dot_g_start
		|| std::abs(p_dot_g) > wolfe_c2*std::abs(p_dot_g_start)) {
		// Linesearch: increased or reduced too little, going to decrease the step
		is_in_linesearch_ = true;
		iter_step+=1;

		linesearch.Iterate(f, p_dot_g);
		if (linesearch.stagnated())
			iter_step = 31;
		for (int i = 0; i < size; i++) x_[i] = x_start[i] + linesearch.x() * p[i];
		CheckFiniteArray(x_, size, "BFGS line-search x");
	} else {
		// if we were in linesearch, we say we are no longer in linesearch
		// else, we should say we ARE in linesearch
		// (not to be caught in an infinite cycle)
		is_in_linesearch_ = !is_in_linesearch_;
		iter_step = 0;

		UpdateInvHess(g);
		// Doing a proper (non-linesearch iteration)
		// forming a descent (hopefully) direction p
		FormDenseDirection(g);
		CheckFiniteArray(p, size, "BFGS search direction");
		p_dot_g = ScalarProd(&g[0], &p[0]);
		if (!std::isfinite(p_dot_g))
			ERROR("BFGS produced a non-finite directional derivative after Hessian update.");
		linesearch.Reset();
		// setting initial point for linearsearch
		SetStart(f, g);

		if (p_dot_g > 0) {
			if (!use_distributed_dense_) {
				std::ofstream ofs("bfgs.log");
				ofs.precision(16);
				ofs << size << '\n';
				for(int i=0; i<size; i++)
					for(int j=0; j<size; j++)
						ofs << inv_hess(i,j) << " ";
				ofs.close();
			}
			ERROR("BFGS: stepping in accend direction detected.");
		}

		linesearch.Iterate(f, p_dot_g);
		for (int i = 0; i < size; i++)
			x_[i] = x_start[i] + linesearch.x() * p[i];
		CheckFiniteArray(x_, size, "BFGS next x");
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
	if (!std::isfinite(py))
		ERROR("BFGS Hessian update produced non-finite curvature.");
	const double curvature_floor = 64.0 * std::numeric_limits<double>::epsilon();
	if (py <= curvature_floor) return;

	DenseMatVec(delta_grad, yC);
	for (int i = 0; i < size; i++)
		yCy += yC[i] * delta_grad[i];
	if (!std::isfinite(yCy))
		ERROR("BFGS Hessian update produced non-finite yHy.");

	double foo = (py + yCy) / (py*py);
	if (!std::isfinite(foo))
		return;

	if (!use_distributed_dense_) {
#ifdef MLIP_INTEL_MKL
		cblas_dsyr(CBLAS_ORDER::CblasRowMajor,
				   CBLAS_UPLO::CblasUpper,
				   size,
				   alpha * alpha * foo,
				   &p[0], 1,
				   &inv_hess(0, 0), size);
		cblas_dsyr2(CBLAS_ORDER::CblasRowMajor,
				    CBLAS_UPLO::CblasUpper,
				    size,
				    -alpha / py,
				    &p[0], 1,
				    &yC[0], 1,
				    &inv_hess(0, 0), size);
		MirrorUpperToLower();
#else
		cblas_dger(CBLAS_ORDER::CblasRowMajor,
					size, size,
					alpha * alpha * foo,
					&p[0], 1,
					&p[0], 1,
					&inv_hess(0, 0), size);
		cblas_dger(CBLAS_ORDER::CblasRowMajor,
					size, size,
					-alpha / py,
					&p[0], 1,
					&yC[0], 1,
					&inv_hess(0, 0), size);
		cblas_dger(CBLAS_ORDER::CblasRowMajor,
					size, size,
					-alpha / py,
					&yC[0], 1,
					&p[0], 1,
					&inv_hess(0, 0), size);
#endif
		return;
	}

	for (int local_row = 0; local_row < distributed_row_count_; ++local_row) {
		const int global_row = distributed_row_start_ + local_row;
		double* row = distributed_inv_hess_rows_.data() + local_row * size;
		const double p_coeff = alpha * alpha * foo * p[global_row] - alpha / py * yC[global_row];
		const double y_coeff = -alpha / py * p[global_row];
		cblas_daxpy(size, p_coeff, &p[0], 1, row, 1);
		cblas_daxpy(size, y_coeff, &yC[0], 1, row, 1);
	}
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
	if (!use_distributed_dense_) {
		inv_hess.set(0.0);
		for (int i = 0; i < size; i++)
			inv_hess(i, i) = 1.0;
	}
	else {
		std::fill(distributed_inv_hess_rows_.begin(), distributed_inv_hess_rows_.end(), 0.0);
		for (int local_row = 0; local_row < distributed_row_count_; ++local_row) {
			const int global_row = distributed_row_start_ + local_row;
			distributed_inv_hess_rows_[local_row * size + global_row] = 1.0;
		}
	}
	f_start = HUGE_DOUBLE;
	p_dot_g_start = HUGE_DOUBLE;
	linesearch.Reset();
	is_in_linesearch_ = false;
}
