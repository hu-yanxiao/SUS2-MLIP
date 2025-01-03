/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Alexander Shapeev, Evgeny Podryabinkin
 */

#ifdef MLIP_MPI
#	include <mpi.h>
#endif

#include "../linear_regression.h"
//#include "train.h"

using namespace std;


#ifdef MLIP_MPI
void Train(	const string& mtpfnm, 
			const std::string& cfgfnm, 
			double ene_eq_wgt ,
			double frc_eq_wgt ,
			double str_eq_wgt ,
			double relfrc_wgt ,
			const std::string mtrxfnm,
			std::ostream* p_log)
{
	int mpi_rank=-1;
	int mpi_size=-1;

	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	if (p_log != nullptr)
	*p_log << "mpi_rank = " << mpi_rank << ", mpi_size = "
		<< mpi_size << ", MTP: " << mtpfnm << ", database: " << cfgfnm << endl;

	Configuration cfg;

	MPI_Barrier(MPI_COMM_WORLD);

	MTP MTP(mtpfnm);
	LinearRegression learner(&MTP, ene_eq_wgt, frc_eq_wgt, str_eq_wgt);
	learner.wgt_rel_forces = relfrc_wgt;

	ifstream ifs(cfgfnm);
	int proc_cntr=0;
	for (int cntr=0; cfg.Load(ifs); cntr++)
	{
		if (cntr % mpi_size == mpi_rank)
		{
			double wgt = 1.0;///((cfg.energy/cfg.size()) + 7.7);
			learner.AddToSLAE(cfg, wgt);
			if (mpi_rank == 0)
			{
				if (p_log != nullptr)
				{
					*p_log << cntr << '\r';
					p_log->flush();
				}
			}
			proc_cntr++;
		}
	}
	ifs.close();
	if (p_log != nullptr)
		*p_log << "process#" << mpi_rank << ": " << proc_cntr << " configuration processed\n";

	//learner.Train();
	//string fnm = "partialearned.mtp" + to_string(rank);
	//	MTP.Save(fnm.c_str());

	double* mtrx = NULL;
	double* rp = NULL;
	int n = MTP.alpha_count;

	if (mpi_rank == 0)
	{
		mtrx = new double[n*n];
		rp = new double[n];
		memset(mtrx, 0, n*n*sizeof(double));
		memset(rp, 0, n*sizeof(double));
	}

	MPI_Reduce(learner.quad_opt_matr, mtrx, n*n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(learner.quad_opt_vec, rp, n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	int eqncnt = 0;
	double scalar = 0.0;
	MPI_Reduce(&learner.quad_opt_eqn_count, &eqncnt, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&learner.quad_opt_scalar, &scalar, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if (mpi_rank == 0)
	{
		memcpy(learner.quad_opt_matr, mtrx, n*n*sizeof(double));
		memcpy(learner.quad_opt_vec, rp, n*sizeof(double));
		learner.quad_opt_scalar = scalar;
		learner.quad_opt_eqn_count = eqncnt;

		if (mtrxfnm != "")
		{
			// writing SLAE matrix
			ofstream ofs(mtrxfnm, ios::binary);
			ofs.write((char*)&n, sizeof(n));
			ofs.write((char*)&learner.quad_opt_eqn_count, sizeof(learner.quad_opt_eqn_count));
			ofs.write((char*)&learner.quad_opt_scalar, sizeof(double));
			ofs.write((char*)learner.quad_opt_vec, n*sizeof(double));
			ofs.write((char*)learner.quad_opt_matr, n*n*sizeof(double));
			ofs.close();
		}

		delete[] mtrx;
		delete[] rp;

		learner.Train();

		MTP.Save(mtpfnm);
	}
}
#else
void Train(	const std::string& mtpfnm, 
			const std::string& cfgfnm, 
			double ene_eq_wgt ,
			double frc_eq_wgt ,
			double str_eq_wgt ,
			double relfrc_wgt ,
			const std::string mtrxfnm ,
			std::ostream* p_log )
{

	Configuration cfg;

	MTP MTP(mtpfnm);
	LinearRegression learner(&MTP, ene_eq_wgt, frc_eq_wgt, str_eq_wgt);
	learner.wgt_rel_forces = relfrc_wgt;

	ifstream ifs(cfgfnm);
	int proc_cntr=0;
	for (int cntr = 0; cfg.Load(ifs); cntr++)
	{
		double wgt = 1.0;///((cfg.energy/cfg.size()) + 7.7);
		learner.AddToSLAE(cfg, wgt);
		*p_log << cntr << '\r';
		p_log->flush();
		proc_cntr++;
	}
	ifs.close();

	int n = MTP.alpha_count;

	if (mtrxfnm != "")
	{
		// writing SLAE matrix
		ofstream ofs(mtrxfnm, ios::binary);
		ofs.write((char*)&n, sizeof(n));
		ofs.write((char*)&learner.quad_opt_eqn_count, sizeof(learner.quad_opt_eqn_count));
		ofs.write((char*)&learner.quad_opt_scalar, sizeof(double));
		ofs.write((char*)learner.quad_opt_vec, n*sizeof(double));
		ofs.write((char*)learner.quad_opt_matr, n*n*sizeof(double));
		ofs.close();
	}

	learner.Train();

	MTP.Save(mtpfnm);
}
#endif
