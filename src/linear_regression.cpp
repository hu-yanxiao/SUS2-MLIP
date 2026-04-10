/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Alexander Shapeev, Evgeny Podryabinkin
 */


#include "linear_regression.h"
#ifdef MLIP_MPI
#	include <mpi.h>
#endif

#include <sstream>

using namespace std;

const char* LinearRegression::tagname = {"fit"};

LinearRegression::LinearRegression(	AnyLocalMLIP* _p_mtp,
									double _wgt_eqtn_energy,
									double _wgt_eqtn_force,
									double _wgt_eqtn_stress,
									double _wgt_rel_forces) :
	AnyTrainer(_p_mtp, _wgt_eqtn_energy, _wgt_eqtn_force, _wgt_eqtn_stress, _wgt_rel_forces),
	p_mtp((MTP*)_p_mtp)
{
	int n = p_mtp->CoeffCount();		// Matrix size and basis functions count

	p_matrix1 = new double[n*n];
	p_matrix2 = new double[n*n];
	p_vector1 = new double[n];
	p_vector2 = new double[n];

	quad_opt_vec = p_vector1;
	quad_opt_matr = p_matrix1;

	quad_opt_eqn_count = 0;

	ClearSLAE();
}

LinearRegression::~LinearRegression()
{
	if (p_matrix1 != nullptr) delete[] p_matrix1;
	if (p_matrix2 != nullptr) delete[] p_matrix2;
	if (p_vector1 != nullptr) delete[] p_vector1;
	if (p_vector2 != nullptr) delete[] p_vector2;
}

void LinearRegression::ClearSLAE()
{
	const int n = p_mtp->CoeffCount();		// Matrix size 

	quad_opt_eqn_count = 0;
	quad_opt_scalar = 0.0;
	memset(quad_opt_vec, 0, n*sizeof(double));
	memset(quad_opt_matr, 0, n*n*sizeof(double));
}

void LinearRegression::CopySLAE()
{
	int n = p_mtp->CoeffCount();	// Matrix size 

	if (quad_opt_matr == p_matrix1)
	{
		memcpy(p_matrix2, p_matrix1, n*n*sizeof(double));
		memcpy(p_vector2, p_vector1, n*sizeof(double));
	}
	else if (quad_opt_matr == p_matrix2)
	{
		memcpy(p_matrix1, p_matrix2, n*n*sizeof(double));
		memcpy(p_vector1, p_vector2, n*sizeof(double));
	}
	else
		ERROR("Invalid internal state");
}

void LinearRegression::RestoreSLAE()
{
	if (quad_opt_matr == p_matrix1)
	{
		quad_opt_matr = p_matrix2;
		quad_opt_vec = p_vector2;
	}
	else if (quad_opt_matr == p_matrix2)
	{
		quad_opt_matr = p_matrix1;
		quad_opt_vec = p_vector1;
	}
	else
		ERROR("Invalid internal state");
}

void LinearRegression::AddToSLAE(Configuration& cfg, double weight)
{
	if (cfg.size() == 0)				// 
		return;

	int n = p_mtp->alpha_count;		// Matrix size and basis functions count

	double wgt_energy = wgt_eqtn_energy / cfg.size();
	double wgt_forces = wgt_eqtn_forces;
	double wgt_stress = wgt_eqtn_stress / cfg.size();

	if (weighting == "structures")
	{
		wgt_energy /= cfg.size();
		wgt_stress /= cfg.size();

		wgt_forces /= cfg.size();
	}
	else if (weighting == "molecules")
	{
		wgt_energy *= cfg.size();
		wgt_stress *= cfg.size();
	}

	if (((wgt_forces > 0) && cfg.has_forces()) ||
		((wgt_stress > 0) && cfg.has_stresses()))
		p_mtp->CalcEFSGrads(cfg, energy_cmpnts, forces_cmpnts, stress_cmpnts);
	else if ((wgt_energy > 0) && cfg.has_energy())
		p_mtp->CalcEnergyGrad(cfg, energy_cmpnts);

	if ((wgt_energy > 0) && cfg.has_energy())
	{
		for (int i = 0; i < n; i++)
			for (int j = i; j < n; j++)
				quad_opt_matr[i*n + j] += weight * wgt_energy * energy_cmpnts[i] * energy_cmpnts[j];

		for (int i = 0; i < n; i++)
			quad_opt_vec[i] += weight * wgt_energy * energy_cmpnts[i] * cfg.energy;
		
		quad_opt_scalar += weight * wgt_energy * cfg.energy * cfg.energy;

		quad_opt_eqn_count += (weight > 0) ? 1 : ((weight < 0) ? -1 : 0);
	}

	if ((wgt_forces > 0) && cfg.has_forces())
		for (int ind = 0; ind < cfg.size(); ind++)
		{
			double wgt = (wgt_rel_forces<=0.0) ?
				weight * wgt_forces :
				weight * wgt_forces * wgt_rel_forces / (cfg.force(ind).NormSq() + wgt_rel_forces);

			for (int a=0; a<3; a++)
				for (int i=0; i<n; i++)
				{
					for (int j=i; j<n; j++)
						quad_opt_matr[i*n + j] += 
							wgt * forces_cmpnts(ind, a, i) * forces_cmpnts(ind, a, j);

					quad_opt_vec[i] += wgt * forces_cmpnts(ind, a, i) * cfg.force(ind, a);
				}
		
			for (int a = 0; a < 3; a++)
				quad_opt_scalar += weight * wgt_forces * cfg.force(ind, a) * cfg.force(ind, a);

			quad_opt_eqn_count += 3 * ((weight > 0) ? 1 : ((weight < 0) ? -1 : 0));
		}

	if ((wgt_stress > 0) && cfg.has_stresses())
	{
		for (int a = 0; a < 3; a++)
			for (int b = 0; b < 3; b++)
			{
				for (int i = 0; i < n; i++)
				{
					for (int j = i; j < n; j++)
						quad_opt_matr[i*n + j] += 
							weight * wgt_stress * stress_cmpnts(a,b,i) * stress_cmpnts(a,b,j);

					quad_opt_vec[i] += 
						weight * wgt_stress * stress_cmpnts(a, b, i) * cfg.stresses[a][b];
				}

				quad_opt_scalar += weight * wgt_stress * cfg.stresses[a][b] * cfg.stresses[a][b];
			}

		quad_opt_eqn_count += 6 * ((weight > 0) ? 1 : ((weight < 0) ? -1 : 0));
	}
}

void LinearRegression::RemoveFromSLAE(Configuration & cfg)
{
	AddToSLAE(cfg, -1);
}

void LinearRegression::JoinSLAE(LinearRegression& to_add)
{
	if (p_mtp->alpha_count != to_add.p_mtp->alpha_count)
		ERROR("Incompatible SLAE sizes detected");
	
	for (int i=0; i<p_mtp->alpha_count*p_mtp->alpha_count; i++)
		quad_opt_matr[i] += to_add.quad_opt_matr[i];
	for (int i = 0; i < p_mtp->alpha_count; i++)
		quad_opt_vec[i] += to_add.quad_opt_vec[i];
	quad_opt_scalar += to_add.quad_opt_scalar;
	quad_opt_eqn_count += to_add.quad_opt_eqn_count;
}

void LinearRegression::SymmetrizeSLAE()
{
	int n = p_mtp->alpha_count;		// Matrix size

	for (int i = 0; i < n; i++) 
		for (int j = i + 1; j < n; j++) 
			quad_opt_matr[j*n + i] = quad_opt_matr[i*n + j];
}

void LinearRegression::SolveSLAE()
{
	const double reg_param_soft = 1e-13;
	const double reg_param_hard = 1e-13;

	const int n = p_mtp->CoeffCount();		// Matrix size
	double* regress_coeffs = p_mtp->Coeff();

	// Regularization (soft)
	for (int i=0; i<n; i++)
		quad_opt_matr[i*n + i] += reg_param_soft*quad_opt_matr[i*n + i];

	// Gaussian Elimination with leading row swaps
	for (int i=0; i<n-1; i++) 
	{
		double& m_ii = quad_opt_matr[i*n + i];

		// find row j0 with maximal in modulus element max_el in i-th column
		int j0 = i;
		double max_el = m_ii;
		for (int j=i+1; j<n; j++)
		{
			double foo = fabs(quad_opt_matr[j*n + i]);
			if (foo > max_el)
			{
				max_el = foo;
				j0 = j;
			}
		}
		if (j0 != i)// swap rows
		{
			for (int k=0; k<n; k++)
				swap(quad_opt_matr[i*n + k], quad_opt_matr[j0*n + k]);
			swap(quad_opt_vec[i], quad_opt_vec[j0]);
		}
		// strong regularization
		if (fabs(m_ii) < reg_param_hard)
			m_ii = reg_param_hard;	
		
		// Gsussian elimination
		for (int j=i+1; j<n; j++) {
			double ratio = quad_opt_matr[j*n + i] / m_ii;
			for (int k=i; k<n; k++) 
				quad_opt_matr[j*n + k] -= ratio * quad_opt_matr[i*n + k];
			quad_opt_vec[j] -= ratio * quad_opt_vec[i];
		}
	}
	// Calculating regression coefficients
	regress_coeffs[n - 1] = quad_opt_vec[n - 1] / quad_opt_matr[(n - 1)*n + (n - 1)];
	for (int i=n-2; i>=0; i--) {
		double temp = quad_opt_vec[i];
		for (int j=i+1; j<n; j++) {
			temp -= (quad_opt_matr[i*n + j] * regress_coeffs[j]);
		}
		regress_coeffs[i] = temp / quad_opt_matr[i*n + i];
	}
}

void LinearRegression::Train(std::vector<Configuration>& train_set)
{
	std::stringstream logstrm1;
	logstrm1 << "Fitting: training over " << train_set.size() << " configurations" << endl;
	MLP_LOG(tagname,logstrm1.str());

	ClearSLAE();

	int cntr = 1;
	for (Configuration& cfg : train_set)
	{
		AddToSLAE(cfg);
		logstrm1 << "Fitting: Adding to SLAE cfg#" << cntr++ << endl;
		MLP_LOG(tagname,logstrm1.str()); logstrm1.str("");
	}

#ifdef MLIP_MPI
	int mpi_rank;
	int mpi_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	int quad_opt_eqn_count_ = 0;
	MPI_Allreduce(&quad_opt_eqn_count, &quad_opt_eqn_count_, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	quad_opt_eqn_count = quad_opt_eqn_count_;
#endif

	logstrm1 << "Fitting: training over " << quad_opt_eqn_count << " equations" << endl;
	MLP_LOG(tagname,logstrm1.str()); logstrm1.clear();
	Train();
	logstrm1 << "Fitting: training complete" << endl; 
	MLP_LOG(tagname,logstrm1.str());
}

void LinearRegression::Train()
{
	SymmetrizeSLAE();

	if (quad_opt_eqn_count < p_mtp->alpha_count)
		Warning("Fitting: Not enough equations for training. Eq. count = " + 
				std::to_string(quad_opt_eqn_count) + " , whereas number of MTP parameters = " +
				std::to_string(p_mtp->alpha_count));


#ifdef MLIP_MPI		// here we sum all SLAE	
	// quad_opt_scalar summation
	double buff_opt_scalar;
	MPI_Allreduce(&quad_opt_scalar, &buff_opt_scalar, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	quad_opt_scalar = buff_opt_scalar;

	// taking pointers to buffer arrays
	int n = p_mtp->alpha_count;		// Matrix size 
	double* buff_opt_matr = nullptr;
	double* buff_opt_vec = nullptr;
	if (quad_opt_matr == p_matrix1)
	{
		buff_opt_matr = p_matrix2;
		buff_opt_vec = p_vector2;
	}
	else
	{
		buff_opt_matr = p_matrix1;
		buff_opt_vec = p_vector1;
	}

	// vec summation
	MPI_Allreduce(quad_opt_vec, buff_opt_vec, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	// matrix summation
	MPI_Allreduce(quad_opt_matr, buff_opt_matr, n*n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	// switch to MPI_Reduced matrix
	RestoreSLAE();
#endif

	CopySLAE();
	SolveSLAE();
	RestoreSLAE();
}

