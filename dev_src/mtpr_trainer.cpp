/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */

#ifdef MLIP_MPI
#	include <mpi.h>
#endif

#include "mtpr_trainer.h"

#ifdef ALGLIB
#	include "alglib/optimization.h"
#endif

#ifdef MLIP_INTEL_MKL
#	include <mkl_lapacke.h>
#	include <mkl_cblas.h>
#	include <mkl_service.h>
#else
#	include <cblas.h>
#endif

#include <ctime>
#include <cstdlib>
#include <cstring>
#include <array>
#include <limits>
#include <random>
#include <sstream>

using namespace std;

namespace {

std::string CurrentTimestamp()
{
	std::time_t now = std::time(nullptr);
	char buf[32];
	std::strftime(buf, sizeof(buf), "%F %T", std::localtime(&now));
	return std::string(buf);
}

constexpr std::size_t kNeighborhoodCacheBudgetBytesPerRank = 4ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr int kOptimizerReducePrefix = 1;
constexpr int kAcceptedDiagDoubleCount = 11;
constexpr int kAcceptedDiagCountCount = 3;
constexpr int kTraceFlushInterval = 16;

double EstimateConfigurationVolume(const Configuration& cfg)
{
	double volume = std::abs(cfg.lattice.det());
	if (volume > 1.0e-12)
		return volume;

	if (cfg.size() == 0)
		return 0.0;

	Vector3 min_pos = cfg.pos(0);
	Vector3 max_pos = cfg.pos(0);
	for (int i = 1; i < cfg.size(); ++i) {
		for (int a = 0; a < 3; ++a) {
			min_pos[a] = std::min(min_pos[a], cfg.pos(i, a));
			max_pos[a] = std::max(max_pos[a], cfg.pos(i, a));
		}
	}

	const double dx = std::max(1.0e-6, max_pos[0] - min_pos[0]);
	const double dy = std::max(1.0e-6, max_pos[1] - min_pos[1]);
	const double dz = std::max(1.0e-6, max_pos[2] - min_pos[2]);
	return dx * dy * dz;
}

std::size_t EstimateNeighborhoodCacheBytes(const std::vector<Configuration>& configs, double cutoff)
{
	const double sphere_volume = (4.0 / 3.0) * M_PI * cutoff * cutoff * cutoff;
	long double total_bytes = 0.0L;
	for (const Configuration& cfg : configs) {
		if (cfg.size() == 0)
			continue;
		const double volume = EstimateConfigurationVolume(cfg);
		const double number_density = volume > 1.0e-12 ? static_cast<double>(cfg.size()) / volume : 0.0;
		const double expected_neighbors = std::max(1.0, sphere_volume * number_density);
		const long double entries = static_cast<long double>(cfg.size()) * expected_neighbors;
		total_bytes += static_cast<long double>(cfg.size()) * 160.0L;
		total_bytes += entries * 96.0L;
	}
	if (total_bytes < 0.0L)
		return 0;
	if (total_bytes > static_cast<long double>(std::numeric_limits<std::size_t>::max()))
		return std::numeric_limits<std::size_t>::max();
	return static_cast<std::size_t>(total_bytes);
}

bool ShouldCacheNeighborhoods(const std::vector<Configuration>& configs, double cutoff, std::size_t* estimated_bytes = nullptr)
{
	const std::size_t bytes = EstimateNeighborhoodCacheBytes(configs, cutoff);
	if (estimated_bytes != nullptr)
		*estimated_bytes = bytes;
	const char* env = std::getenv("MLIP_CACHE_NEIGHBORHOODS");
	if (env != nullptr) {
		if (std::strcmp(env, "0") == 0 || std::strcmp(env, "false") == 0 || std::strcmp(env, "FALSE") == 0)
			return false;
		if (std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0 || std::strcmp(env, "TRUE") == 0)
			return true;
	}
	return bytes <= kNeighborhoodCacheBudgetBytesPerRank;
}

bool TryBuildNeighborhoods(const std::vector<Configuration>& configs,
						   double cutoff,
						   std::vector<Neighborhoods>& neighborhoods_out)
{
	try {
		neighborhoods_out.clear();
		neighborhoods_out.reserve(configs.size());
		for (const Configuration& cfg : configs)
			neighborhoods_out.emplace_back(cfg, cutoff);
		return true;
	}
	catch (const std::bad_alloc&) {
		neighborhoods_out.clear();
		neighborhoods_out.shrink_to_fit();
		return false;
	}
}

int SuggestedLinearSolveThreads(int n)
{
#ifdef MLIP_INTEL_MKL
	const char* env = std::getenv("MLIP_LINEAR_SOLVE_THREADS");
	if (env != nullptr) {
		int parsed = std::atoi(env);
		if (parsed > 0)
			return parsed;
	}
	if (n >= 1000 && n <= 5000)
		return 8;
#endif
	return 1;
}

void SolveSLAEGaussian(int n, double* matrix, double* rhs)
{
	for (int i = 0; i < (n - 1); i++) {
		for (int j = (i + 1); j < n; j++) {
			double ratio = matrix[j*n + i] / matrix[i*n + i];
			for (int count = i; count < n; count++)
				matrix[j*n + count] -= ratio * matrix[i*n + count];
			rhs[j] -= ratio * rhs[i];
		}
	}

	rhs[n - 1] /= matrix[(n - 1)*n + (n - 1)];
	for (int i = (n - 2); i >= 0; i--) {
		double temp = rhs[i];
		for (int j = (i + 1); j < n; j++)
			temp -= matrix[i*n + j] * rhs[j];
		rhs[i] = temp / matrix[i*n + i];
	}
}

}


void MTPR_trainer::shift(bool shift_)
{
p_mlmtpr->shift_ = shift_;

}

#ifdef MLIP_MPI
void MTPR_trainer::ConfigureTrainComm(bool has_local_work, int world_rank, int world_size)
{
	if (train_comm_owned_ && train_comm_ != MPI_COMM_NULL) {
		MPI_Comm_free(&train_comm_);
	}
	train_comm_owned_ = false;
	train_rank_active_ = has_local_work;
	train_comm_is_world_ = true;
	train_rank_ = 0;
	train_size_ = 1;
	train_comm_ = MPI_COMM_WORLD;

	if (world_size <= 1) {
		train_rank_active_ = true;
		return;
	}

	MPI_Comm_split(MPI_COMM_WORLD, has_local_work ? 0 : MPI_UNDEFINED, world_rank, &train_comm_);
	int active_count = 0;
	int active_flag = has_local_work ? 1 : 0;
	MPI_Allreduce(&active_flag, &active_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	train_comm_is_world_ = (active_count == world_size);
	if (!has_local_work) {
		train_comm_ = MPI_COMM_NULL;
		train_rank_ = -1;
		train_size_ = 0;
		return;
	}

	train_comm_owned_ = true;
	MPI_Comm_rank(train_comm_, &train_rank_);
	MPI_Comm_size(train_comm_, &train_size_);
}

void MTPR_trainer::BroadcastCoeffsWorld(int root_world_rank)
{
	MPI_Bcast(&p_mlmtpr->Coeff()[0], p_mlmtpr->CoeffCount(), MPI_DOUBLE, root_world_rank, MPI_COMM_WORLD);
}
#endif


void MTPR_trainer::LoadWeights(ifstream& ifs)
{

	string next;


	ifs >> wgt_eqtn_forces;
	//cout << "forces coeffitient = " << wgt_eqtn_forces << "\n";
	ifs >> next;
	ifs >> next;

	ifs >> wgt_eqtn_stress;
	//cout << "stress coeffitient = " << wgt_eqtn_stress << "\n";
	ifs >> next;
	ifs >> next;

	ifs >> wgt_eqtn_constr;
	//cout << "stress coeffitient = " << wgt_eqtn_stress << "\n";
	ifs >> next;
	ifs >> next;

}



void MTPR_trainer::ClearSLAE()
{

	int n = p_mlmtpr->alpha_count - 1 + p_mlmtpr->species_count;	// Matrix size

	if (quad_opt_allocated_n != n || quad_opt_vec == nullptr || quad_opt_matr == nullptr) {
		delete[] quad_opt_vec;
		delete[] quad_opt_matr;
		quad_opt_vec = new double[n];
		quad_opt_matr = new double[n*n];
		quad_opt_allocated_n = n;
	}

	quad_opt_eqn_count = 0;
	quad_opt_scalar = 0.0;

	memset(quad_opt_vec, 0, n * sizeof(double));
	memset(quad_opt_matr, 0, n * n * sizeof(double));
}

MTPR_trainer::~MTPR_trainer()
{
#ifdef MLIP_MPI
	if (train_comm_owned_ && train_comm_ != MPI_COMM_NULL)
		MPI_Comm_free(&train_comm_);
#endif
	delete[] quad_opt_vec;
	delete[] quad_opt_matr;
	quad_opt_vec = nullptr;
	quad_opt_matr = nullptr;
	quad_opt_allocated_n = 0;
}

void MTPR_trainer::SymmetrizeSLAE()
{
	int n = p_mlmtpr->alpha_count + p_mlmtpr->species_count - 1;		// Matrix size

	for (int i = 0; i < n; i++)
		for (int j = i + 1; j < n; j++)
			quad_opt_matr[j*n + i] = quad_opt_matr[i*n + j];
}

void MTPR_trainer::SolveSLAE()
{

	SymmetrizeSLAE();

	double gammareg = 1e-13;

	int n = p_mlmtpr->alpha_count - 1 + p_mlmtpr->species_count;		// Matrix size

	for (int i = 0; i < n; i++)
		quad_opt_matr[i*n + i] += gammareg*(1 + quad_opt_matr[i*n + i]);

	p_mlmtpr->LinCoeff();	/// TO MOVE TO MLMTPR

#ifdef MLIP_INTEL_MKL
	std::vector<double> matrix_work(quad_opt_matr, quad_opt_matr + n * n);
	std::vector<double> rhs_work(quad_opt_vec, quad_opt_vec + n);
	const int solve_threads = SuggestedLinearSolveThreads(n);
	int previous_threads = 0;
	if (solve_threads > 1)
		previous_threads = mkl_set_num_threads_local(solve_threads);

	int info = LAPACKE_dposv(LAPACK_ROW_MAJOR, 'U', n, 1,
							 matrix_work.data(), n,
							 rhs_work.data(), 1);
	if (info != 0) {
		matrix_work.assign(quad_opt_matr, quad_opt_matr + n * n);
		rhs_work.assign(quad_opt_vec, quad_opt_vec + n);
		std::vector<int> ipiv(n);
		info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, 1,
							 matrix_work.data(), n,
							 ipiv.data(),
							 rhs_work.data(), 1);
	}

	if (info != 0) {
		matrix_work.assign(quad_opt_matr, quad_opt_matr + n * n);
		rhs_work.assign(quad_opt_vec, quad_opt_vec + n);
		SolveSLAEGaussian(n, matrix_work.data(), rhs_work.data());
	}

	memcpy(quad_opt_matr, matrix_work.data(), n * n * sizeof(double));
	memcpy(quad_opt_vec, rhs_work.data(), n * sizeof(double));
	if (solve_threads > 1)
		mkl_set_num_threads_local(previous_threads);
#else
	SolveSLAEGaussian(n, quad_opt_matr, quad_opt_vec);
#endif

	for (int i = 0; i < n; i++)
		p_mlmtpr->linear_coeffs[i] = quad_opt_vec[i];
	//double e0_ = 0;
	//for (int i = 0; i < p_mlmtpr->species_count; i++)
	//{
	//	e0_ += p_mlmtpr->regression_coeffs[i] / p_mlmtpr->species_count;
	//}

//	double e_0 = 0;
//	for (int i = 0; i < p_mlmtpr->species_count; i++) {
//		e_0 += p_mlmtpr->linear_coeffs[i] / p_mlmtpr->species_count;
//	}
//	std::random_device rand_device;
//	std::default_random_engine generator(rand_device());
//	std::uniform_real_distribution<> uniform(-0.05, 0.05);
	for (int i = 0; i < p_mlmtpr->species_count; i++) {
		p_mlmtpr->regression_coeffs[i] = p_mlmtpr->linear_coeffs[i] - 1.0;
		p_mlmtpr->linear_coeffs[i] = 1.0;
	}
	for (int i = 0; i < n; i++) {
		p_mlmtpr->regression_coeffs[p_mlmtpr->regression_coeffs.size() - n + i] = p_mlmtpr->linear_coeffs[i];
	}
	for (int i = (int)p_mlmtpr->regression_coeffs.size() - n + p_mlmtpr->species_count; i < (int)p_mlmtpr->regression_coeffs.size(); i++)
		p_mlmtpr->regression_coeffs[i] /= p_mlmtpr->linear_mults[i - (p_mlmtpr->regression_coeffs.size() - n + p_mlmtpr->species_count)] * 1.0;


}


void MTPR_trainer::AddToSLAE(Configuration& cfg, double weight, const Neighborhoods* neighborhoods)
{
	if (cfg.size() == 0)				// 
		return;

	int n = p_mlmtpr->alpha_count - 1 + p_mlmtpr->species_count;		// Matrix size
      //  {std::cout<<n<<" "<<std::endl;}
     //   {std::cout<<n<<" "<<  (int)p_mlmtpr->energy_cmpnts.size() <<std::endl;}
	if (neighborhoods != nullptr)
		p_mlmtpr->CalcEFSComponents(cfg, *neighborhoods);
	else
		p_mlmtpr->CalcEFSComponents(cfg);
//        int w=p_mlmtpr->energy_cmpnts.size();
      //  {std::cout<<n<<" "<<std::endl;}
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

	cout.precision(15);

	int fn = norm_by_forces;
	double d = 0.1;
	double avef = 0;

	if (cfg.has_forces())
		for (int ind = 0; ind < cfg.size(); ind++)
			avef += cfg.force(ind).NormSq() / cfg.size();


	if (cfg.has_energy())
	{
		const double alpha = weight * wgt_energy * d / (d + fn*avef);
		cblas_dger(CBLAS_ORDER::CblasRowMajor, n, n,
			alpha,
			p_mlmtpr->energy_cmpnts, 1,
			p_mlmtpr->energy_cmpnts, 1,
			quad_opt_matr, n);
		cblas_daxpy(n, alpha * cfg.energy, p_mlmtpr->energy_cmpnts, 1, quad_opt_vec, 1);
		quad_opt_scalar += alpha * cfg.energy * cfg.energy;

		quad_opt_eqn_count += (weight > 0) ? 1 : ((weight < 0) ? -1 : 0);
	}

	if ((wgt_eqtn_forces > 0) && (cfg.has_forces()))
	{
		const double alpha = weight * wgt_forces * d / (d + fn*avef);
		const int force_rows = 3 * cfg.size();
		const bool use_force_block = (n >= 1000 && n <= 5000 && force_rows > 0);
		if (use_force_block) {
			lin_force_block_.assign(static_cast<size_t>(force_rows) * n, 0.0);
			lin_force_rhs_.assign(force_rows, 0.0);
			for (int ind = 0; ind < cfg.size(); ind++) {
				for (int a = 0; a < 3; a++) {
					const int row = 3 * ind + a;
					lin_force_rhs_[row] = cfg.force(ind, a);
					for (int i = 0; i < n; i++)
						lin_force_block_[static_cast<size_t>(row) * n + i] = p_mlmtpr->forces_cmpnts(ind, i, a);
				}
			}
			cblas_dgemm(CBLAS_ORDER::CblasRowMajor,
				CBLAS_TRANSPOSE::CblasTrans,
				CBLAS_TRANSPOSE::CblasNoTrans,
				n, n, force_rows,
				alpha,
				lin_force_block_.data(), n,
				lin_force_block_.data(), n,
				1.0,
				quad_opt_matr, n);
			cblas_dgemv(CBLAS_ORDER::CblasRowMajor,
				CBLAS_TRANSPOSE::CblasTrans,
				force_rows, n,
				alpha,
				lin_force_block_.data(), n,
				lin_force_rhs_.data(), 1,
				1.0,
				quad_opt_vec, 1);
			for (double force_value : lin_force_rhs_)
				quad_opt_scalar += alpha * force_value * force_value;
			quad_opt_eqn_count += force_rows * ((weight > 0) ? 1 : ((weight < 0) ? -1 : 0));
		} else {
			for (int ind = 0; ind < cfg.size(); ind++)
			{
				for (int a = 0; a < 3; a++) {
					double* force_cmp = &p_mlmtpr->forces_cmpnts(ind, 0, a);
					cblas_dger(CBLAS_ORDER::CblasRowMajor, n, n,
						alpha,
						force_cmp, 3,
						force_cmp, 3,
						quad_opt_matr, n);
					cblas_daxpy(n, alpha * cfg.force(ind, a), force_cmp, 3, quad_opt_vec, 1);
				}

				for (int a = 0; a < 3; a++)
					quad_opt_scalar += alpha * cfg.force(ind, a) * cfg.force(ind, a);

				quad_opt_eqn_count += 3 * ((weight > 0) ? 1 : ((weight < 0) ? -1 : 0));
			}
		}
	}

	if ((wgt_eqtn_stress > 0) && (cfg.has_stresses()))
	{
		const double alpha = weight * wgt_stress;
		const int stress_rows = 9;
		const bool use_stress_block = (n >= 1000 && n <= 5000);
		if (use_stress_block) {
			lin_stress_block_.assign(static_cast<size_t>(stress_rows) * n, 0.0);
			lin_stress_rhs_.assign(stress_rows, 0.0);
			int row = 0;
			for (int a = 0; a < 3; a++)
				for (int b = 0; b < 3; b++) {
					lin_stress_rhs_[row] = cfg.stresses[a][b];
					double* stress_cmp = &p_mlmtpr->stress_cmpnts[0][a][b];
					for (int i = 0; i < n; i++)
						lin_stress_block_[static_cast<size_t>(row) * n + i] = stress_cmp[i * 9];
					row++;
				}
			cblas_dgemm(CBLAS_ORDER::CblasRowMajor,
				CBLAS_TRANSPOSE::CblasTrans,
				CBLAS_TRANSPOSE::CblasNoTrans,
				n, n, stress_rows,
				alpha,
				lin_stress_block_.data(), n,
				lin_stress_block_.data(), n,
				1.0,
				quad_opt_matr, n);
			cblas_dgemv(CBLAS_ORDER::CblasRowMajor,
				CBLAS_TRANSPOSE::CblasTrans,
				stress_rows, n,
				alpha,
				lin_stress_block_.data(), n,
				lin_stress_rhs_.data(), 1,
				1.0,
				quad_opt_vec, 1);
			for (double stress_value : lin_stress_rhs_)
				quad_opt_scalar += alpha * stress_value * stress_value;
		} else {
			for (int a = 0; a < 3; a++)
				for (int b = 0; b < 3; b++) {
					double* stress_cmp = &p_mlmtpr->stress_cmpnts[0][a][b];
					cblas_dger(CBLAS_ORDER::CblasRowMajor, n, n,
						alpha,
						stress_cmp, 9,
						stress_cmp, 9,
						quad_opt_matr, n);
					cblas_daxpy(n, alpha * cfg.stresses[a][b], stress_cmp, 9, quad_opt_vec, 1);
					quad_opt_scalar += alpha * cfg.stresses[a][b] * cfg.stresses[a][b];
				}
		}

		quad_opt_eqn_count += 6 * ((weight > 0) ? 1 : ((weight < 0) ? -1 : 0));
	}
}


double* MTPR_trainer::ConstructLinHessian()
{
	ERROR("MTPR_trainer::ConstructLinHessian() requires revision and refactoring!");

	//for (auto& cfg : training_set)
	//	AddForTrain(cfg);

	SymmetrizeSLAE();

	int linsize = p_mlmtpr->alpha_scalar_moments + p_mlmtpr->species_count;

	//int m = (int)training_set.size();
	int M = 1;

	//MPI_Allreduce(&m, &M, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

	//cout << "M" << M << endl;

	double* Hess = new double[linsize*linsize];

	for (int i = 0; i < linsize; i++)
		for (int j = 0; j < linsize; j++)
			Hess[i*linsize + j] = 2 * quad_opt_matr[i*linsize + j] / M;


	return Hess;
}

void MTPR_trainer::TrainLinear(int prank,
								   vector<Configuration>& training_set,
								   const std::vector<Neighborhoods>* neighborhoods,
								   const std::string& context)
{
#ifdef MLIP_MPI
	if (!train_rank_active_)
		return;
	const int mpi_rank = train_rank_;
	const MPI_Comm train_comm = train_comm_;
#else
	const int mpi_rank = prank;
#endif
	const long long local_cfg_count = static_cast<long long>(training_set.size());
	long long local_atom_count = 0;
	for (const Configuration& cfg : training_set)
		local_atom_count += cfg.size();

	p_mlmtpr->Orthogonalize();

	ClearSLAE();

	double build_start = 0.0;
#ifdef MLIP_MPI
	build_start = MPI_Wtime();
#else
	build_start = static_cast<double>(clock()) / CLOCKS_PER_SEC;
#endif
	for (size_t i = 0; i < training_set.size(); ++i)
		AddToSLAE(training_set[i], 1.0, neighborhoods == nullptr ? nullptr : &(*neighborhoods)[i]);
	double build_seconds = 0.0;
#ifdef MLIP_MPI
	build_seconds = MPI_Wtime() - build_start;
#else
	build_seconds = static_cast<double>(clock()) / CLOCKS_PER_SEC - build_start;
#endif

#ifdef MLIP_MPI

	int n = p_mlmtpr->alpha_count - 1 + p_mlmtpr->species_count;		// Matrix size
	const std::string prefix = context.empty() ? "TrainLinear" : "TrainLinear[" + context + "]";
	double reduce_start = MPI_Wtime();
	if (mpi_rank == 0) {
		std::cout << "[" << CurrentTimestamp() << "] TrainLinear build done"
		          << " ctx=" << prefix
		          << " n=" << n
		          << " cfg_local=" << local_cfg_count
		          << " atoms_local=" << local_atom_count
		          << " eq_local=" << quad_opt_eqn_count
		          << " build_s=" << build_seconds
		          << std::endl;
	}
	if (mpi_rank == 0) {
		MPI_Reduce(MPI_IN_PLACE, quad_opt_matr, n*n, MPI_DOUBLE, MPI_SUM, 0, train_comm);
		MPI_Reduce(MPI_IN_PLACE, quad_opt_vec, n, MPI_DOUBLE, MPI_SUM, 0, train_comm);
		MPI_Reduce(MPI_IN_PLACE, &quad_opt_scalar, 1, MPI_DOUBLE, MPI_SUM, 0, train_comm);
		const double reduce_seconds = MPI_Wtime() - reduce_start;
		std::cout << "[" << CurrentTimestamp() << "] TrainLinear reduce done"
		          << " ctx=" << prefix
		          << " reduce_s=" << reduce_seconds << std::endl;
		const double solve_start = MPI_Wtime();
		SolveSLAE();
		const double solve_seconds = MPI_Wtime() - solve_start;
		std::cout << "[" << CurrentTimestamp() << "] TrainLinear solve done"
		          << " ctx=" << prefix
		          << " solve_s=" << solve_seconds << std::endl;
	} else {
		MPI_Reduce(quad_opt_matr, quad_opt_matr, n*n, MPI_DOUBLE, MPI_SUM, 0, train_comm);
		MPI_Reduce(quad_opt_vec, quad_opt_vec, n, MPI_DOUBLE, MPI_SUM, 0, train_comm);
		MPI_Reduce(&quad_opt_scalar, &quad_opt_scalar, 1, MPI_DOUBLE, MPI_SUM, 0, train_comm);
	}

#else
	SolveSLAE();
#endif
#ifdef MLIP_MPI
	double bcast_start = MPI_Wtime();
	MPI_Bcast(&p_mlmtpr->Coeff()[0], p_mlmtpr->CoeffCount(), MPI_DOUBLE, 0, train_comm);
	double bcast_seconds = MPI_Wtime() - bcast_start;
	if (mpi_rank == 0) {
		std::cout << "[" << CurrentTimestamp() << "] TrainLinear bcast done"
		          << " ctx=" << prefix
		          << " bcast_s=" << bcast_seconds
		          << std::endl;
	}
#endif


}

void MTPR_trainer::random_sample(int prank, std::vector<Configuration>& training_set, int max_step, const std::vector<Neighborhoods>* neighborhoods) {
//	int n_coeffe= p_mlip->CoeffCount();
//	double* x = p_mlip->Coeff();
#ifdef MLIP_MPI
	if (!train_rank_active_)
		return;
	const int mpi_rank = train_rank_;
	const MPI_Comm train_comm = train_comm_;
#else
	const int mpi_rank = prank;
#endif
	int n_coeffe= p_mlmtpr->CoeffCount();
	double* x = p_mlmtpr->Coeff();
	double c_l = 0.0;
	double std_l = 0.0;
	double p_l = 1e10;
	std::vector<double> _x(n_coeffe);
	for (int i = 0; i < n_coeffe; i++)
		_x[i] = x[i];

	int num_step = 0;
	int m = (int)training_set.size();
	int K = m;

#ifdef MLIP_MPI
	MPI_Allreduce(&m, &K, 1, MPI_INT, MPI_SUM, train_comm);
#endif
	TrainLinear(prank, training_set, neighborhoods, "random_sample init");
	if (mpi_rank == 0) { std::cout << _x.size() << " " << n_coeffe << std::endl; }
	ObjectiveFunction(training_set, neighborhoods);
	loss_ /= K;
	std_ /= K;
	if (mpi_rank == 0) { std::cout << "__________....__________ " << std::endl; }
#ifdef MLIP_MPI
	MPI_Reduce(&loss_, &c_l, 1, MPI_DOUBLE, MPI_SUM, 0, train_comm);
	MPI_Reduce(&std_, &std_l, 1, MPI_DOUBLE, MPI_SUM, 0, train_comm);
#else
	c_l = loss_;
	std_l = std_;
	p_l = loss_ + std_;
#endif
	if (mpi_rank == 0) { std::cout << "__________....__________ " << std::endl; }
	while (num_step < max_step) {
		if (mpi_rank == 0) {
			std::random_device rand_device;
			std::mt19937_64 generator(rand_device());

			std::cout << "Random sample of nonlinear coefficients" << std::endl;
			p_mlmtpr->RandomizeNonlinearCoeffs(generator, 0.5, true, 0.20);
		}
#ifdef MLIP_MPI
		MPI_Bcast(&x[0], p_mlmtpr->CoeffCount(), MPI_DOUBLE, 0, train_comm);
#endif
		TrainLinear(prank, training_set, neighborhoods, "random_sample step " + std::to_string(num_step));
		ObjectiveFunction(training_set, neighborhoods);
		loss_ /= K;
		std_ /= K;
#ifdef MLIP_MPI
		MPI_Reduce(&loss_, &c_l, 1, MPI_DOUBLE, MPI_SUM, 0, train_comm);
		MPI_Reduce(&std_, &std_l, 1, MPI_DOUBLE, MPI_SUM, 0, train_comm);
#else
		c_l = loss_;
		std_l = std_;
#endif
		if (mpi_rank == 0) {
			if (std_l + c_l < p_l) {
				p_l = std_l + c_l;
				for (int i = 0; i < n_coeffe; i++)
					_x[i] = x[i];
				std::cout << "num_step: " << num_step << " f= " << c_l << "   std^2= " << std_l/std_scaling << "\t (*opt)"
					<< std::endl;
				num_step += 1;
			}
			else {
				std::cout << "num_step: " << num_step << " f= " << c_l << "   std^2= " << std_l/std_scaling << std::endl;
				num_step += 1;
			}
		}
#ifdef MLIP_MPI
		MPI_Bcast(&num_step, 1, MPI_INT, 0, train_comm);
		MPI_Bcast(&_x[0], n_coeffe, MPI_DOUBLE, 0, train_comm);
#endif
	}
	for (int i = 0; i < n_coeffe; i++)
		x[i] = _x[i];

#ifdef MLIP_MPI
	MPI_Bcast(&x[0], p_mlmtpr->CoeffCount(), MPI_DOUBLE, 0, train_comm);
#endif
	std::vector<double>().swap(_x);
}


#ifndef ALGLIB
void MTPR_trainer::Train(std::vector<Configuration>& training_set) //with Shapeev bfgs
{

	//cout << max_step_count << endl;

	int n = p_mlip->CoeffCount();
	double *x = p_mlip->Coeff();

	int nlin = p_mlmtpr->alpha_count + p_mlmtpr->species_count - 1;

	p_mlmtpr->max_radial.resize(p_mlmtpr->species_count*p_mlmtpr->species_count*p_mlmtpr->radial_func_count);

	for (int i = 0; i < p_mlmtpr->max_radial.size(); i++)
		p_mlmtpr->max_radial[i] = 1e-10;

	int prank = 0;
	int psize = 1;
	std::stringstream logstrm1;

#ifdef MLIP_MPI
	if (!train_rank_active_)
		return;
	prank = train_rank_;
	psize = train_size_;
	MPI_Comm train_comm = train_comm_;
	const bool allow_distributed_bfgs = train_comm_is_world_;
	bfgs.UseDistributedDense(prank, allow_distributed_bfgs ? psize : 1, train_comm);
	if (prank == 0) {
		logstrm1 << "MTPR parallel training started" << endl;
		// 		if (GetLogStream()!=nullptr) GetLogStream()->precision(15);
		MLP_LOG("dev", logstrm1.str()); logstrm1.str("");
	}
#else
	if (prank == 0) {

		logstrm1 << "MTPR serial(?!?) training started" << endl;
		// 		if (GetLogStream()!=nullptr) GetLogStream()->precision(15);
		MLP_LOG("dev", logstrm1.str()); logstrm1.str("");

	}
#endif

	int m = (int)training_set.size(); // train set size on the current core
	int K = 0;                     // train set size over all cores

	K = m;

#ifdef MLIP_MPI												   
	MPI_Allreduce(&m, &K, 1, MPI_INT, MPI_SUM, train_comm);
#endif

	std::vector<Neighborhoods> training_neighborhoods;
	const long long local_atom_count = [&training_set]() {
		long long total_atoms = 0;
		for (const Configuration& cfg : training_set)
			total_atoms += cfg.size();
		return total_atoms;
	}();
	std::size_t estimated_cache_bytes = 0;
	bool cache_training_neighborhoods =
		ShouldCacheNeighborhoods(training_set, p_mlmtpr->CutOff(), &estimated_cache_bytes);
	if (cache_training_neighborhoods) {
		const bool built_ok = TryBuildNeighborhoods(training_set, p_mlmtpr->CutOff(), training_neighborhoods);
		if (built_ok) {
			if (prank == 0)
				std::cout << "[" << CurrentTimestamp() << "] BFGS neighborhood cache enabled"
				          << " estimated_bytes=" << estimated_cache_bytes << std::endl;
		} else {
			cache_training_neighborhoods = false;
			if (prank == 0)
				std::cout << "[" << CurrentTimestamp() << "] BFGS neighborhood cache fallback"
				          << " estimated_bytes=" << estimated_cache_bytes
				          << " reason=bad_alloc" << std::endl;
		}
	} else if (prank == 0) {
		std::cout << "[" << CurrentTimestamp() << "] BFGS neighborhood cache disabled"
		          << " estimated_bytes=" << estimated_cache_bytes
		          << " budget_bytes=" << kNeighborhoodCacheBudgetBytesPerRank
		          << std::endl;
	}

	bfgs.Set_x(x, n);
	const bool distributed_bfgs = bfgs.UsingDistributedDense();
	const bool need_std_terms = NeedStdTerms();

	Array1D inv_hess_diag(n, 1.0);
	for (int i = n - nlin + p_mlmtpr->species_count; i < n; i++)
		inv_hess_diag[i] /= p_mlmtpr->linear_mults[i - (n - nlin + p_mlmtpr->species_count)] * p_mlmtpr->linear_mults[i - (n - nlin + p_mlmtpr->species_count)];
	bfgs.SetInvHessDiagonal(inv_hess_diag);
	const int species_coeff_begin = n - nlin;
	std::vector<int> species_coeff_indices(p_mlmtpr->species_count);
	for (int i = 0; i < p_mlmtpr->species_count; ++i)
		species_coeff_indices[i] = species_coeff_begin + i;
	auto reset_bfgs_state = [&]() {
		bfgs.Restart();
		bfgs.Set_x(x, n);
		bfgs.SetInvHessDiagonal(inv_hess_diag);
	};
	bool freeze_species_coeffs = do_lin && do_lin_step_limit > 0;
	if (freeze_species_coeffs)
		bfgs.MaskCoordinates(species_coeff_indices);
	std::vector<double> optimizer_reduce_local(n + kOptimizerReducePrefix, 0.0);
	std::vector<double> optimizer_reduce_global(n + kOptimizerReducePrefix, 0.0);
	std::array<double, kAcceptedDiagDoubleCount> accepted_diag_local{};
	std::array<double, kAcceptedDiagDoubleCount> accepted_diag_global{};
	std::array<long long, kAcceptedDiagCountCount> accepted_count_local{};
	std::array<long long, kAcceptedDiagCountCount> accepted_count_global{};

	int num_step = 0;

	double linf = 9e99;
	double loss_reduced_by = 0.0;
	double loss_prev = 9e99;
	double std_l = 0.0;
	double stdd_l = 0.0;
	double mean_1_l = 0.0;
	double mean_2_l = 0.0;
	double mean_3_l = 0.0;
	double energy_mae_mev_atom_l = 0.0;
	double energy_rmse_mev_atom_l = 0.0;
	double force_mae_mev_a_l = 0.0;
	double force_rmse_mev_a_l = 0.0;
	double stress_mae_ev_l = 0.0;
	double stress_rmse_ev_l = 0.0;



	bool converge = false;

	double max_shift = 0.1*random_perturb;
	double cooling_rate = 0.2;
	bool linesearch = false;

	std::random_device random_device;
	std::default_random_engine eng(random_device());
	std::uniform_real_distribution<double> distr(-1, 1);
	std::ofstream bfgs_trace_stream;
	if (prank == 0 && !bfgs_trace_file.empty()) {
		bfgs_trace_stream.open(bfgs_trace_file, std::ios::out | std::ios::trunc);
		if (!bfgs_trace_stream.is_open())
			ERROR("Can't open BFGS trace file " + bfgs_trace_file + " for writing");
		bfgs_trace_stream << "step,total_loss,efs_loss,energy_mae_mev_atom,force_mae_mev_a,stress_mae_ev\n";
		bfgs_trace_stream << std::scientific << std::setprecision(17);
	}
	int trace_lines_since_flush = 0;

	while (!converge)
	{
		//if (prank==0)
		//cout << "itr" << endl;

		if (!linesearch)
		{
			bool external_x_modified = false;
			bool external_x_needs_broadcast = false;
			const bool random_perturb_applied = (max_shift != 0.0);
			if (prank == 0) {
				for (int i = 0; i < n - nlin; i++)
					x[i] += distr(eng)*max_shift;
			}
			external_x_modified = random_perturb_applied;
			external_x_needs_broadcast = distributed_bfgs && random_perturb_applied;

			const bool should_freeze_species_coeffs = do_lin && num_step < do_lin_step_limit;
			if (should_freeze_species_coeffs != freeze_species_coeffs) {
				freeze_species_coeffs = should_freeze_species_coeffs;
				reset_bfgs_state();
				if (freeze_species_coeffs)
					bfgs.MaskCoordinates(species_coeff_indices);
				if (prank == 0) {
					std::cout << "[" << CurrentTimestamp() << "] "
					          << "species_coeffs BFGS freeze "
					          << (freeze_species_coeffs ? "enabled" : "disabled")
					          << " at step=" << num_step << std::endl;
				}
			} else if (freeze_species_coeffs) {
				bfgs.MaskCoordinates(species_coeff_indices);
			}

			const bool run_train_linear = do_lin
				&& num_step < do_lin_step_limit
				&& (num_step % do_lin_frequency == 0);
			if (run_train_linear)
			{
				/*
				for (int i=n-nlin+p_mlmtpr->species_count;i<n;i++)
					for (int j=n-nlin+p_mlmtpr->species_count;j<n;j++)
									bfgs.inv_hess(i,j)*=p_mlmtpr->linear_mults[i-(n-nlin+p_mlmtpr->species_count)]*p_mlmtpr->linear_mults[j-(n-nlin+p_mlmtpr->species_count)];

							p_mlmtpr->Perform_scaling();

				for (int i=n-nlin+p_mlmtpr->species_count;i<n;i++)
					for (int j=n-nlin+p_mlmtpr->species_count;j<n;j++)
									bfgs.inv_hess(i,j)/=p_mlmtpr->linear_mults[i-(n-nlin+p_mlmtpr->species_count)]*p_mlmtpr->linear_mults[j-(n-nlin+p_mlmtpr->species_count)];

				*/
				TrainLinear(prank,
						  training_set,
						  cache_training_neighborhoods ? &training_neighborhoods : nullptr,
						  "bfgs refresh step " + std::to_string(num_step));
				external_x_modified = true;
				external_x_needs_broadcast = false; // TrainLinear already broadcasts coeffs.
			}

#ifdef MLIP_MPI
			if (distributed_bfgs && external_x_needs_broadcast)
				MPI_Bcast(&x[0], n, MPI_DOUBLE, 0, train_comm);
#endif
			if (external_x_modified) {
				if (distributed_bfgs || prank == 0)
					bfgs.Set_x(x, n);
				if (freeze_species_coeffs)
					bfgs.MaskCoordinates(species_coeff_indices);
			}
		}

		std::memcpy(x, bfgs.Data(), n * sizeof(double));

		CalcObjectiveFunctionGrad(training_set, cache_training_neighborhoods ? &training_neighborhoods : nullptr);

		loss_ /= K;
		std_ /= K;
        stdd_/=K;
		mean_1 /= K;
		mean_2 /= K;
		mean_3 /= K;
		for (int i = 0; i < n; i++)
			loss_grad_[i] /= K;

#ifdef MLIP_MPI
		optimizer_reduce_local[0] = loss_;
		std::memcpy(optimizer_reduce_local.data() + kOptimizerReducePrefix,
		            &loss_grad_[0],
		            n * sizeof(double));
		MPI_Reduce(optimizer_reduce_local.data(),
		           prank == 0 ? optimizer_reduce_global.data() : optimizer_reduce_local.data(),
		           n + kOptimizerReducePrefix,
		           MPI_DOUBLE,
		           MPI_SUM,
		           0,
		           train_comm);
		if (prank == 0) {
			bfgs_f = optimizer_reduce_global[0];
			std::memcpy(&bfgs_g[0],
			            optimizer_reduce_global.data() + kOptimizerReducePrefix,
			            n * sizeof(double));
		}

#else
		bfgs_f = loss_;
		memcpy(&bfgs_g[0], &loss_grad_[0], p_mlmtpr->CoeffCount() * sizeof(double));
#endif	
		if (freeze_species_coeffs) {
			for (int idx : species_coeff_indices)
				bfgs_g[idx] = 0.0;
		}

#ifdef MLIP_MPI
		if (distributed_bfgs) {
			if (prank == 0) {
				optimizer_reduce_global[0] = bfgs_f;
				std::memcpy(optimizer_reduce_global.data() + kOptimizerReducePrefix,
				            &bfgs_g[0],
				            n * sizeof(double));
			}
			MPI_Bcast(optimizer_reduce_global.data(),
			          n + kOptimizerReducePrefix,
			          MPI_DOUBLE,
			          0,
			          train_comm);
			if (prank != 0) {
				bfgs_f = optimizer_reduce_global[0];
				std::memcpy(&bfgs_g[0],
				            optimizer_reduce_global.data() + kOptimizerReducePrefix,
				            n * sizeof(double));
			}
		}
#endif

		if ((distributed_bfgs || prank == 0) && !converge) {
				bfgs.Iterate(bfgs_f, bfgs_g);

				while (abs(bfgs.x(p_mlmtpr->species_count*p_mlmtpr->species_count) - x[p_mlmtpr->species_count*p_mlmtpr->species_count]) > 0.5) {
					bfgs.ReduceStep(0.25);
				}
		}

#ifdef MLIP_MPI
		if (!distributed_bfgs) {
			if (prank == 0)
				std::memcpy(x, bfgs.Data(), n * sizeof(double));
			MPI_Bcast(&x[0], n, MPI_DOUBLE, 0, train_comm);
			if (prank != 0) {
				bfgs.Set_x(x, n);
				if (freeze_species_coeffs)
					bfgs.MaskCoordinates(species_coeff_indices);
			}
		}
#endif

		if (prank == 0)
			if (!converge) {
				if (bfgs.iter_step > 30) {
					converge = true;
					logstrm1 << "BFGS ended due to linesearch  more than  30 iterations" << endl;
                                        logstrm1 << "d_x= "<< bfgs.x(0) - x[0] << endl;
					MLP_LOG("dev", logstrm1.str()); logstrm1.str("");
				}
					

			}

		if (distributed_bfgs || prank == 0)
			linesearch = bfgs.is_in_linesearch();

	#ifdef MLIP_MPI
		if (!distributed_bfgs) {
				int linesearch_state = linesearch ? 1 : 0;
				MPI_Bcast(&linesearch_state, 1, MPI_INT, 0, train_comm);
				linesearch = (linesearch_state != 0);
			}
		#endif

		if (!linesearch)
		{
			accepted_diag_local[0] = metric_energy_abs_sum_;
			accepted_diag_local[1] = metric_energy_sq_weighted_sum_;
			accepted_diag_local[2] = metric_force_abs_component_sum_;
			accepted_diag_local[3] = metric_force_sq_component_sum_;
			accepted_diag_local[4] = metric_stress_abs_component_sum_;
			accepted_diag_local[5] = metric_stress_sq_component_sum_;
			accepted_diag_local[6] = std_;
			accepted_diag_local[7] = stdd_;
			accepted_diag_local[8] = mean_1;
			accepted_diag_local[9] = mean_2;
			accepted_diag_local[10] = mean_3;
			accepted_count_local[0] = metric_energy_atom_count_;
			accepted_count_local[1] = metric_force_component_count_;
			accepted_count_local[2] = metric_stress_component_count_;

#ifdef MLIP_MPI
			MPI_Request accepted_reduce_requests[2];
			MPI_Ireduce(accepted_diag_local.data(),
			            prank == 0 ? accepted_diag_global.data() : accepted_diag_local.data(),
			            kAcceptedDiagDoubleCount,
			            MPI_DOUBLE,
			            MPI_SUM,
			            0,
			            train_comm,
			            &accepted_reduce_requests[0]);
			MPI_Ireduce(accepted_count_local.data(),
			            prank == 0 ? accepted_count_global.data() : accepted_count_local.data(),
			            kAcceptedDiagCountCount,
			            MPI_LONG_LONG_INT,
			            MPI_SUM,
			            0,
			            train_comm,
			            &accepted_reduce_requests[1]);
#else
			accepted_diag_global = accepted_diag_local;
			accepted_count_global = accepted_count_local;
#endif

			if (prank == 0)
			{
				if (loss_prev < bfgs_f)
				{
					max_shift *= (1 - cooling_rate);

					logstrm1 << "*" << endl;
					MLP_LOG("dev", logstrm1.str()); logstrm1.str("");
				}

				if (abs(loss_prev - bfgs_f) < 1e-13)
				{
					converge = true;
					logstrm1 << "BFGS ended due to small decr. for 1 iteration" << endl;
					MLP_LOG("dev", logstrm1.str()); logstrm1.str("");
				}

				loss_prev = bfgs_f;
			}

#ifdef MLIP_MPI
			MPI_Waitall(2, accepted_reduce_requests, MPI_STATUSES_IGNORE);
#endif

			if (prank == 0)
			{
				std_l = accepted_diag_global[6];
				stdd_l = accepted_diag_global[7];
				mean_1_l = accepted_diag_global[8];
				mean_2_l = accepted_diag_global[9];
				mean_3_l = accepted_diag_global[10];
				const long long energy_mae_count_global = accepted_count_global[0];
				const long long force_mae_count_global = accepted_count_global[1];
				const long long stress_mae_count_global = accepted_count_global[2];
				energy_mae_mev_atom_l = energy_mae_count_global > 0 ?
					1000.0 * accepted_diag_global[0] / static_cast<double>(energy_mae_count_global) : 0.0;
				energy_rmse_mev_atom_l = energy_mae_count_global > 0 ?
					1000.0 * std::sqrt(accepted_diag_global[1] / static_cast<double>(energy_mae_count_global)) : 0.0;
				force_mae_mev_a_l = force_mae_count_global > 0 ?
					1000.0 * accepted_diag_global[2] / static_cast<double>(force_mae_count_global) : 0.0;
				force_rmse_mev_a_l = force_mae_count_global > 0 ?
					1000.0 * std::sqrt(accepted_diag_global[3] / static_cast<double>(force_mae_count_global)) : 0.0;
				stress_mae_ev_l = stress_mae_count_global > 0 ?
					accepted_diag_global[4] / static_cast<double>(stress_mae_count_global) : 0.0;
				stress_rmse_ev_l = stress_mae_count_global > 0 ?
					std::sqrt(accepted_diag_global[5] / static_cast<double>(stress_mae_count_global)) : 0.0;

				last_train_error_summary_.energy_mae_mev_atom = energy_mae_mev_atom_l;
				last_train_error_summary_.energy_rmse_mev_atom = energy_rmse_mev_atom_l;
				last_train_error_summary_.force_mae_mev_a = force_mae_mev_a_l;
				last_train_error_summary_.force_rmse_mev_a = force_rmse_mev_a_l;
				last_train_error_summary_.stress_mae_ev = stress_mae_ev_l;
				last_train_error_summary_.stress_rmse_ev = stress_rmse_ev_l;
				have_last_train_error_summary_ = true;

				const double efs_loss = bfgs_f - std_l - stdd_l;
				if (curr_pot_name != "")
					p_mlmtpr->Save(curr_pot_name);
				if (bfgs_trace_stream.is_open()) {
					bfgs_trace_stream << num_step << ','
					                  << bfgs_f << ','
					                  << efs_loss << ','
					                  << energy_mae_mev_atom_l << ','
					                  << force_mae_mev_a_l << ','
					                  << stress_mae_ev_l << '\n';
					if (++trace_lines_since_flush >= kTraceFlushInterval) {
						bfgs_trace_stream.flush();
						trace_lines_since_flush = 0;
					}
				}
				logstrm1 << "[" << CurrentTimestamp() << "] "
						 << "step=" << num_step
						 << " total_loss=" << std::fixed << std::setprecision(3) << bfgs_f
						 << " efs_loss=" << std::fixed << std::setprecision(3) << efs_loss
						 << " energy_MAE(meV/atom)=" << std::fixed << std::setprecision(3) << energy_mae_mev_atom_l
						 << " force_MAE(meV/A)=" << std::fixed << std::setprecision(3) << force_mae_mev_a_l
						 << " stress_MAE(eV)=" << std::fixed << std::setprecision(3) << stress_mae_ev_l
						 << endl;
				MLP_LOG("dev", logstrm1.str()); logstrm1.str("");

				num_step++;

				if (num_step % 60 == 1) linf = bfgs_f;
				if (num_step % 60 == 0)
				{
					if ((linf - bfgs_f) / bfgs_f < linstop && (linf - bfgs_f) / bfgs_f < loss_reduced_by && num_step > 120)
					{
						converge = true;
						logstrm1 << "BFGS ended due to small decr. in 60 iterations" << endl;
						MLP_LOG("dev", logstrm1.str()); logstrm1.str("");
					}
					loss_reduced_by = (linf - bfgs_f) / bfgs_f;
				}

				if (num_step >= max_step_count)
				{
					converge = true;

					logstrm1 << "step limit reached" << endl;
					MLP_LOG("dev", logstrm1.str()); logstrm1.str("");
				}
			}
		}

	#ifdef MLIP_MPI
			int iteration_state[3] = {
				converge ? 1 : 0,
				linesearch ? 1 : 0,
				num_step
			};
			MPI_Bcast(iteration_state, 3, MPI_INT, 0, MPI_COMM_WORLD);
			converge = (iteration_state[0] != 0);
			linesearch = (iteration_state[1] != 0);
			num_step = iteration_state[2];
	#endif
	}

	p_mlmtpr->inited = true;
	have_hess = true;
	if (bfgs_trace_stream.is_open())
		bfgs_trace_stream.flush();

	if (prank == 0)
	{
		const double joint_std_report = need_std_terms ? (std_l / std_scaling) : 0.0;
		const double center_std_report = need_std_terms ? (stdd_l / stdd_scaling) : 0.0;
		logstrm1 << "MTPR training ended:" << "\t joint_std^2:" << joint_std_report <<  "\t center_std^2:"  << center_std_report <<"   " << mean_1_l <<"   " << mean_2_l << "   " << mean_3_l << "\t efs:" << bfgs_f - std_l - 0 * stdd_l << endl;
		MLP_LOG("dev", logstrm1.str()); logstrm1.str("");
	}
}
#else
#ifdef MLIP_MPI
void MTPR_trainer::Train2(vector<Configuration>& train_set)
{
	int mpi_rank;
	int mpi_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	if (mpi_rank == 0) cout << "Parallel training started (using AlgLib BFGS)" << endl;

	int size = p_mlmtpr->CoeffCount();
	double *coeffs = p_mlmtpr->Coeff();

	alglib::real_1d_array x;
	x.setcontent(size, coeffs);

	//double prev_func, curr_func;
	int needNextIterInt;
	int curr_iter = 0;
	alglib::ae_int_t m = x.length();
	alglib::minlbfgsstate state;
	alglib::minlbfgsreport rep;

	int isPrintFunc = 1;
	double epsx = 0.0;
	double epsg = 0.0;
	double epsf = 1e-13;
	int maxits = 1500;

	int mm = (int)train_set.size(); // train set size on the current core
	int K = 0;                     // train set size over all cores

	K = mm;

	MPI_Allreduce(&mm, &K, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

	alglib::minlbfgscreate(x.length(), m, x, state);
	alglib::minlbfgssetcond(state, epsg, epsf, epsx, maxits);

	alglib_impl::ae_state _alglib_env_state;
	alglib_impl::ae_state_init(&_alglib_env_state);
	try {
		if (mpi_rank == 0) {
			if (alglib_impl::minlbfgsiteration(state.c_ptr(), &_alglib_env_state) && (curr_iter < maxits)) needNextIterInt = 1;
			else needNextIterInt = 0;
		}
		MPI_Bcast(&needNextIterInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
		state.needfg = true;
		while (needNextIterInt == 1) {
			if (state.needfg) {
				if (mpi_rank == 0)
					memcpy(coeffs, state.x.getcontent(), sizeof(double) * x.length());
				MPI_Bcast(coeffs, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
				CalcObjectiveFunctionGrad(train_set);
				loss_ /= K;
				for (int i = 0; i < size; i++)
					loss_grad_[i] /= K;
				MPI_Reduce(&loss_, &state.f, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
				std::cout.precision(12);
				if (mpi_rank == 0 && isPrintFunc == 1) {
					std::cout << "curr_iter = " << curr_iter << ", func = " << state.f << std::endl;
				}
				MPI_Reduce(&loss_grad_[0], &state.g[0], size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
				//if (mpi_rank == 0) {
				//	for (int i = 0; i < size; i++)
				//		std::cout << state.g[i] << " ";
				//	std::cout << std::endl;
				//}
				if (mpi_rank == 0) {
					if (alglib_impl::minlbfgsiteration(state.c_ptr(), &_alglib_env_state) && (curr_iter < maxits))
						needNextIterInt = 1;
					else {
						needNextIterInt = 0;
						//std::cout << "first stop criterion" << std::endl;	
					}
				}
				//check progress of functional minimization each 100 iterations
				//if (mpi_rank == 0 && needNextIterInt == 1) {
				//	if (curr_iter == 0) prev_func = loss_;
				//	if (curr_iter % 100 == 0 && curr_iter != 0) {
				//		curr_func = loss_;
						//std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ";
						//std::cout << fabs((curr_func - prev_func) / prev_func) << std::endl;
				//		if (fabs((curr_func - prev_func) / prev_func) < 1E-5) {
				//			needNextIterInt = 0;
				//			std::cout << "second stop criterion" << std::endl;
				//		}
				//		else {
				//			prev_func = curr_func;
				//		}
				//	}					
				//}
				MPI_Bcast(&needNextIterInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
				if (mpi_rank == 0) {
					curr_iter++;
					//std::cout << curr_iter << std::endl;
				}
				/*if (mpi_rank == 0) {
					for (int i = 0; i < size; i++)
						std::cout << coeffs[i] << ", ";
					std::cout << std::endl;
				}*/
				if (mpi_rank == 0) {
					p_mlmtpr->Save(curr_pot_name);
				}
				continue;
			}
			if (state.xupdated) {
				if (mpi_rank == 0)
					memcpy(coeffs, state.x.getcontent(), sizeof(double) * x.length());
				MPI_Bcast(coeffs, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
				double loss = ObjectiveFunction(train_set);
				//state.f=f;
				MPI_Allreduce(&loss, &state.f, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
				if (mpi_rank == 0) {
					if (alglib_impl::minlbfgsiteration(state.c_ptr(), &_alglib_env_state) && (curr_iter < maxits)) needNextIterInt = 1;
					else needNextIterInt = 0;
				}
				continue;
			}
			throw alglib::ap_error("ALGLIB: error in 'minlbfgsoptimize' (some derivatives were not provided?)");
		}
		alglib_impl::ae_state_clear(&_alglib_env_state);
	}
	catch (alglib_impl::ae_error_type) {
		throw alglib::ap_error(_alglib_env_state.error_msg);
	}

	if (mpi_rank == 0) {
		alglib::minlbfgsresults(state, x, rep);
		memcpy(coeffs, state.x.getcontent(), sizeof(double) * x.length());
	}
	MPI_Bcast(coeffs, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//printf("%d\n", int(rep.terminationtype)); 
	//printf("%d\n", int(rep.iterationscount));
	//printf("%d\n", int(rep.inneriterationscount));
	//printf("%s\n", x.tostring(x.length()).c_str());
}
#endif
#endif
