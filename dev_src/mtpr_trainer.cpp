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

#include <sstream>

using namespace std;


void MTPR_trainer::shift(bool shift_)
{
p_mlmtpr->shift_ = shift_;

}


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

	quad_opt_vec = new double[n];
	quad_opt_matr = new double[n*n];

	quad_opt_eqn_count = 0;
	quad_opt_scalar = 0.0;

	memset(quad_opt_vec, 0, n * sizeof(double));
	memset(quad_opt_matr, 0, n * n * sizeof(double));
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

	// Gaussian Elimination
	for (int i = 0; i < (n - 1); i++) {
		for (int j = (i + 1); j < n; j++) {
			double ratio = quad_opt_matr[j*n + i] / quad_opt_matr[i*n + i];
			for (int count = i; count < n; count++) {
				quad_opt_matr[j*n + count] -= (ratio * quad_opt_matr[i*n + count]);
			}
			quad_opt_vec[j] -= (ratio * quad_opt_vec[i]);
		}
	}


	p_mlmtpr->linear_coeffs = p_mlmtpr->LinCoeff();	/// TO MOVE TO MLMTPR

	p_mlmtpr->linear_coeffs[n - 1] = quad_opt_vec[n - 1] / quad_opt_matr[(n - 1)*n + (n - 1)];

	for (int i = (n - 2); i >= 0; i--) {
		double temp = quad_opt_vec[i];
		for (int j = (i + 1); j < n; j++) {
			temp -= (quad_opt_matr[i*n + j] * p_mlmtpr->linear_coeffs[j]);
		}
		p_mlmtpr->linear_coeffs[i] = temp / quad_opt_matr[i*n + i];
	}
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


void MTPR_trainer::AddToSLAE(Configuration& cfg, double weight)
{
	if (cfg.size() == 0)				// 
		return;

	int n = p_mlmtpr->alpha_count - 1 + p_mlmtpr->species_count;		// Matrix size
      //  {std::cout<<n<<" "<<std::endl;}
     //   {std::cout<<n<<" "<<  (int)p_mlmtpr->energy_cmpnts.size() <<std::endl;}
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
		for (int i = 0; i < n; i++)
			for (int j = i; j < n; j++)
				quad_opt_matr[i*n + j] += weight * wgt_energy * p_mlmtpr->energy_cmpnts[i] * p_mlmtpr->energy_cmpnts[j] * d / (d + fn*avef);

		for (int i = 0; i < n; i++)
			quad_opt_vec[i] += weight * wgt_energy * p_mlmtpr->energy_cmpnts[i] * cfg.energy*d / (d + fn*avef);

		quad_opt_scalar += weight * wgt_energy * cfg.energy * cfg.energy *d / (d + fn*avef);

		quad_opt_eqn_count += (weight > 0) ? 1 : ((weight < 0) ? -1 : 0);
	}

	if ((wgt_eqtn_forces > 0) && (cfg.has_forces()))
	{
		for (int i = 0; i < n; i++)
			for (int j = i; j < n; j++)
				for (int ind = 0; ind < cfg.size(); ind++)
					for (int a = 0; a < 3; a++)
						quad_opt_matr[i*n + j] += weight * wgt_forces * p_mlmtpr->forces_cmpnts(ind, i, a) * p_mlmtpr->forces_cmpnts(ind, j, a)*d / (d + fn*avef);

		for (int ind = 0; ind < cfg.size(); ind++)
		{
			for (int i = 0; i < n; i++)
				for (int a = 0; a < 3; a++)
					quad_opt_vec[i] += weight * wgt_forces * p_mlmtpr->forces_cmpnts(ind, i, a) * cfg.force(ind, a)*d / (d + fn*avef);

			for (int a = 0; a < 3; a++)
				quad_opt_scalar += weight * wgt_forces * cfg.force(ind, a) * cfg.force(ind, a) *d / (d + fn*avef);

			quad_opt_eqn_count += 3 * ((weight > 0) ? 1 : ((weight < 0) ? -1 : 0));
		}
	}

	if ((wgt_eqtn_stress > 0) && (cfg.has_stresses()))
	{
		for (int i = 0; i < n; i++)
			for (int j = i; j < n; j++)
				for (int a = 0; a < 3; a++)
					for (int b = 0; b < 3; b++)
						quad_opt_matr[i*n + j] += weight * wgt_stress * p_mlmtpr->stress_cmpnts[i][a][b] * p_mlmtpr->stress_cmpnts[j][a][b];

		for (int i = 0; i < n; i++)
			for (int a = 0; a < 3; a++)
				for (int b = 0; b < 3; b++)
					quad_opt_vec[i] += weight * wgt_stress * p_mlmtpr->stress_cmpnts[i][a][b] * cfg.stresses[a][b];

		for (int a = 0; a < 3; a++)
			for (int b = 0; b < 3; b++)
				quad_opt_scalar += weight * wgt_stress * cfg.stresses[a][b] * cfg.stresses[a][b];

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

void MTPR_trainer::TrainLinear(int prank, vector<Configuration>& training_set)
{
  
	p_mlmtpr->Orthogonalize();

	ClearSLAE();
 
	for (auto& cfg : training_set)
		AddToSLAE(cfg);
   
#ifdef MLIP_MPI

	double* mtrx = nullptr;
	double* rp = nullptr;
	double scalar = 0;

	int n = p_mlmtpr->alpha_count - 1 + p_mlmtpr->species_count;		// Matrix size

	if (prank == 0)
	{       
		mtrx = new double[n*n];
		rp = new double[n];
		memset(mtrx, 0, n*n * sizeof(double));
		memset(rp, 0, n * sizeof(double));
                

	}

	MPI_Reduce(quad_opt_matr, mtrx, n*n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(quad_opt_vec, rp, n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&quad_opt_scalar, &scalar, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);


	if (prank == 0)
	{
               
		memcpy(quad_opt_matr, mtrx, n*n * sizeof(double));
		memcpy(quad_opt_vec, rp, n * sizeof(double));
		quad_opt_scalar = scalar;

		SolveSLAE();
              
		delete[] mtrx;
		delete[] rp;
	}

#else
	SolveSLAE();
#endif
#ifdef MLIP_MPI
        MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(&p_mlmtpr->Coeff()[0], p_mlmtpr->CoeffCount(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif


}

void MTPR_trainer::random_sample(int prank, std::vector<Configuration>& training_set, int max_step) {
	int nlin= p_mlmtpr->alpha_count + p_mlmtpr->species_count - 1;
//	int n_coeffe= p_mlip->CoeffCount();
//	double* x = p_mlip->Coeff();
        int n_coeffe= p_mlmtpr->CoeffCount();
        double* x = p_mlmtpr->Coeff();
	double c_l;
	double std_l;
	double p_l=1e10;
	std::vector<double> _x;
        _x.resize(n_coeffe);
	_x.reserve(n_coeffe);
        for (int i = 0;i < n_coeffe;i++) {
                                        _x[i] = x[i];
                                }

	//
	int n_s = p_mlmtpr->species_count;
	int n_r = p_mlmtpr->radial_func_count;
	int n_k = p_mlmtpr->K_;
	int n_rb =p_mlmtpr->Get_RB_size();

	//
	int num_step = 0;

	int m = (int)training_set.size(); // train set size on the current core
	int K = 0;                     // train set size over all cores
        double _loss = 0;
	K = m;

#ifdef MLIP_MPI												   
	MPI_Allreduce(&m, &K, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
             TrainLinear(prank, training_set);
             if (prank == 0 ) {std::cout << _x.size() <<" " <<n_coeffe<<std::endl; }
		//	 CalcObjectiveFunctionGrad(training_set);
	      _loss= ObjectiveFunction(training_set);
             loss_ /= K;
             std_ /= K;
if (prank == 0 ) {std::cout <<"__________....__________ " <<std::endl; }
#ifdef MLIP_MPI
             MPI_Barrier(MPI_COMM_WORLD);
             MPI_Reduce(&loss_, &c_l, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);  
             MPI_Reduce(&std_, &std_l, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#else
             c_l = loss_;
             std_l = std_;
             p_l = loss_ + std_ ;
#endif
        if (prank == 0 ) {std::cout <<"__________....__________ " <<std::endl; }
	while (num_step < max_step) {
		if (prank == 0) {
			std::random_device rand_device;
			std::default_random_engine generator(rand_device());
			std::uniform_real_distribution<> uniform(-1.0, 1.0);

			std::cout << "Random sample of radial coefficients" << std::endl;

			for (int k = 0; k < 1; k++)
				for (int l = k; l < 1; l++)
					for (int i = 0; i < n_r; i++) {
						for (int j = 0; j < n_rb; j++)
						{	x[n_s+2*n_s*n_s*n_k+i * (n_rb+n_s) + j]= 0.5 * uniform(generator);
                                                        
                                               //      if (j >= n_rb){ p_mlmtpr->regression_coeffs[i * (n_rb+n_s)+j]= 1.00 ;}    
                                                 //    if (num_step < 3) {std::cout << p_mlmtpr->regression_coeffs[i * n_r + j] << std::endl;} 
                                                }
						//      mtpr.regression_coeffs[k*mtpr.radial_func_count*rb_size +
						//              i*rb_size + min(i, rb_size - 1)] = 5e-7 * uniform(generator);
					}

		    //  if (num_step< 1) {std::cout << x[0] <<  x[-1]  << std::endl;}
                       
                }
#ifdef MLIP_MPI
        //        MPI_Barrier(MPI_COMM_WORLD);
	//	MPI_Bcast(&(p_mlmtpr->regression_coeffs[0]), n_coeffe, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Barrier(MPI_COMM_WORLD);
                MPI_Bcast(&x[0], p_mlmtpr->CoeffCount(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
     //         if (prank == 0 && num_step< 3) {
      //                for (int i = 0; i < n_r; i++) {
       //                    for (int j = 0; j < n_rb+ n_s; j++){
        //                                                  std::cout << p_mlmtpr->regression_coeffs[i*(n_rb+n_s)+j] << std::endl;
         //                                                  }
           //                                                       }      }
		TrainLinear(prank, training_set);
//              if (prank == 0 && num_step< 3) {std::cout << p_mlmtpr->regression_coeffs[0] << " " <<p_mlmtpr->regression_coeffs[(n_rb+n_s)*n_r-3]  <<" " <<n_coeffe<<std::endl; }
	//	CalcObjectiveFunctionGrad(training_set);
		_loss= ObjectiveFunction(training_set);
		loss_ /= K;
		std_ /= K;
#ifdef MLIP_MPI
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Reduce(&loss_, &c_l, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); // get c_l in rank 0 
		MPI_Reduce(&std_, &std_l, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); 
#else
		c_l = loss_;
		std_l = std_;
#endif
		if (prank == 0) {
			if (std_l+c_l  < p_l) {
				p_l = std_l+c_l;
				for (int i = 0;i < n_coeffe;i++) {
					_x[i] = x[i];
				}
				std::cout << "num_step: " << num_step << " f= " << c_l << "   std^2= " << std_l/std_scaling <<"\t (*opt)"
					<< std::endl;

				num_step += 1;
				
			}
			else { 
				std::cout << "num_step: " << num_step << " f= " << c_l <<"   std^2= " << std_l/std_scaling<< std::endl;
				num_step += 1;
			}	
		}
#ifdef MLIP_MPI
                MPI_Barrier(MPI_COMM_WORLD);
		MPI_Bcast(&num_step, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&_x[0], n_coeffe, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
	
	}
	for (int i = 0;i < n_coeffe;i++) {
		x[i] = _x[i];
	}

#ifdef MLIP_MPI
                MPI_Barrier(MPI_COMM_WORLD);
               // MPI_Bcast(&(p_mlmtpr->regression_coeffs[0]), n_coeffe, MPI_DOUBLE, 0, MPI_COMM_WORLD);
               MPI_Bcast(&x[0], p_mlmtpr->CoeffCount(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
	std::vector<double>().swap(_x);
	
	
	//
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
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Comm_rank(MPI_COMM_WORLD, &prank);
	MPI_Comm_size(MPI_COMM_WORLD, &psize);
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
	MPI_Allreduce(&m, &K, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif

	bfgs.Set_x(x, n);

	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			if (i == j)
				bfgs.inv_hess(i, j) = 1;
			else
				bfgs.inv_hess(i, j) = 0;

	for (int i = n - nlin + p_mlmtpr->species_count; i < n; i++)
		bfgs.inv_hess(i, i) /= p_mlmtpr->linear_mults[i - (n - nlin + p_mlmtpr->species_count)] * p_mlmtpr->linear_mults[i - (n - nlin + p_mlmtpr->species_count)];

	int num_step = 0;

	double linf = 9e99;
	double loss_reduced_by = 0.0;
	double loss_prev = 9e99;
	double std_l;
    double stdd_l ;
	double mean_1_l;
	double mean_2_l;
	double mean_3_l;
        int lin_freq= 100 ;



	bool converge = false;

	double max_shift = 0.1*random_perturb;
	double cooling_rate = 0.2;
	bool linesearch = false;

	std::random_device random_device;
	std::default_random_engine eng(random_device());
	std::uniform_real_distribution<double> distr(-1, 1);

	while (!converge)
	{
		//if (prank==0)
		//cout << "itr" << endl;

		if (!linesearch)
		{
			for (int i = 0; i < n - nlin; i++)
				x[i] += distr(eng)*max_shift;

			if (prank == 0)
				bfgs.Set_x(x, n);

#ifdef MLIP_MPI
			MPI_Bcast(&x[0], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
			


                        if (num_step >= 0 & num_step < 400) {lin_freq =50;}
                        if (num_step >= 400 & num_step < 800) {lin_freq =60;}
                        if (num_step >= 800 ) {lin_freq =70;}

                        
                        
			if (num_step % lin_freq  == 0 )
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
                                if (num_step < 3000 && do_lin) 
                                {
                                 TrainLinear(prank, training_set);
                                }
			//	TrainLinear(prank, training_set);

				if (prank == 0)
					bfgs.Set_x(x, n);
			}
			
			if (prank == 0)
				if (curr_pot_name != "")
					p_mlmtpr->Save(curr_pot_name);
		}

		for (int i = 0; i < n; i++)
			x[i] = bfgs.x(i);

#ifdef MLIP_MPI
		MPI_Bcast(&x[0], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
		CalcObjectiveFunctionGrad(training_set);

		loss_ /= K;
		std_ /= K;
        stdd_/=K;
		mean_1 /= K;
		mean_2 /= K;
		mean_3 /= K;
		for (int i = 0; i < n; i++)
			loss_grad_[i] /= K;

#ifdef MLIP_MPI
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Reduce(&loss_, &bfgs_f, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&loss_grad_[0], &bfgs_g[0], n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&std_, &std_l, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&stdd_, &stdd_l, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&mean_1, &mean_1_l, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&mean_2, &mean_2_l, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&mean_3, &mean_3_l, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

#else
		bfgs_f = loss_;
		std_l = std_;
		stdd_l = stdd_;
		mean_1_l = mean_1;
		mean_2_l = mean_2;
		mean_3_l = mean_3;
		memcpy(&bfgs_g[0], &loss_grad_[0], p_mlmtpr->CoeffCount() * sizeof(double));
#endif	

		if (prank == 0)
			if (!converge) {
				bfgs.Iterate(bfgs_f, bfgs_g);

				while (abs(bfgs.x(p_mlmtpr->species_count*p_mlmtpr->species_count) - x[p_mlmtpr->species_count*p_mlmtpr->species_count]) > 0.5) {
					bfgs.ReduceStep(0.25);
				}
				if (bfgs.iter_step > 30) {
					converge = true;
					logstrm1 << "BFGS ended due to linesearch  more than  30 iterations" << endl;
                                        logstrm1 << "d_x= "<< bfgs.x(0) - x[0] << endl;
					MLP_LOG("dev", logstrm1.str()); logstrm1.str("");
				}
					

			}

		linesearch = bfgs.is_in_linesearch();

		//if (prank == 0 && !linesearch) cout << num_step << " " << bfgs_f << endl;

		if (!linesearch)
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
				//logstrm1 << "BFGS iter " << num_step << ": f=" << bfgs_f << "\t joint_std^2:" << std_l/std_scaling <<  "\t center_std^2:"  << stdd_l/stdd_scaling << "\t" << mean_1_l << "\t" << mean_2_l << "\t" << mean_3_l << "\t efs:" << bfgs_f - std_l - stdd_l <<endl;
				logstrm1 << "BFGS iter " << std::setw(6) << num_step << ": f=" << std::fixed << std::setprecision(6) << std::right << bfgs_f << "  joint_std^2:" << std_l/std_scaling <<      "  center_std^2:"  << stdd_l/stdd_scaling << "\t" << std::fixed << std::setprecision(3) << std::right << mean_1_l << "  " << std::fixed << std::setprecision(3) << std::right << mean_2_l << "  " << std::    fixed << std::setprecision(3) << std::right << mean_3_l << "\t efs:" << std::fixed << std::setprecision(3) << std::right << bfgs_f - std_l - stdd_l <<endl;
				MLP_LOG("dev", logstrm1.str()); logstrm1.str("");

				//cout << num_step << " " << bfgs_f << endl;
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

#ifdef MLIP_MPI
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Bcast(&converge, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
		MPI_Bcast(&linesearch, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
		MPI_Bcast(&num_step, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
	}

	p_mlmtpr->inited = true;
	have_hess = true;

	if (prank == 0)
	{
		logstrm1 << "MTPR training ended:" << "\t joint_std^2:" << std_l/std_scaling <<  "\t center_std^2:"  << stdd_l/stdd_scaling <<"   " << mean_1_l <<"   " << mean_2_l << "   " << mean_3_l << "\t efs:" << bfgs_f - std_l - 0 * stdd_l << endl;
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
