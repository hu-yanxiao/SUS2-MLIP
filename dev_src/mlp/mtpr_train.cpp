/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */

#include <algorithm>
#include <cstdio>
#include <ctime>
#include <sys/stat.h>
#include <unistd.h>

#include "mtpr_train.h"

using namespace std;

char* mtpfnm;
char* cfgfnm;

int prank = 0;
int psize = 1;

vector<Configuration> training_set;
vector<Configuration> validSet;

struct DatasetStats {
	long long cfg_count = 0;
	long long atom_count = 0;
};

namespace {

std::string SanitizeShardTag(const std::string& value)
{
	std::string sanitized;
	sanitized.reserve(value.size());
	for (char ch : value) {
		if ((ch >= 'a' && ch <= 'z') ||
		    (ch >= 'A' && ch <= 'Z') ||
		    (ch >= '0' && ch <= '9'))
			sanitized.push_back(ch);
		else
			sanitized.push_back('_');
	}
	return sanitized;
}

std::string MakeShardDir(const std::string& cfgfnm, const std::string& label)
{
	std::string basename = cfgfnm;
	size_t slash = basename.find_last_of("/\\");
	if (slash != std::string::npos)
		basename = basename.substr(slash + 1);
	return ".cfg_shards_" + label + "_" + SanitizeShardTag(basename) + "_" +
		std::to_string(static_cast<long long>(std::time(nullptr))) + "_" +
		std::to_string(static_cast<long long>(getpid()));
}

DatasetStats LoadConfigsFromFile(const std::string& cfgfnm, std::vector<Configuration>& target)
{
	DatasetStats stats;
	target.clear();
	Configuration cfg;
	std::ifstream ifs(cfgfnm, std::ios::binary);
	for (; cfg.Load(ifs); ) {
		target.push_back(cfg);
		stats.cfg_count++;
		stats.atom_count += cfg.size();
	}
	return stats;
}

void BroadcastString(std::string& value)
{
#ifdef MLIP_MPI
	int rank = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int length = static_cast<int>(value.size());
	MPI_Bcast(&length, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (rank != 0)
		value.resize(length);
	if (length > 0)
		MPI_Bcast(&value[0], length, MPI_CHAR, 0, MPI_COMM_WORLD);
#else
	(void)value;
#endif
}

DatasetStats LoadConfigsViaRank0Shards(const std::string& cfgfnm,
									   int proc_rank,
									   int proc_size,
									   std::vector<Configuration>& target,
									   const std::string& label)
{
	if (proc_size <= 1)
		return LoadConfigsFromFile(cfgfnm, target);

	DatasetStats stats;
	std::string shard_dir;

	if (proc_rank == 0) {
		shard_dir = MakeShardDir(cfgfnm, label);
		if (mkdir(shard_dir.c_str(), 0777) != 0)
			ERROR("Unable to create shard directory " + shard_dir);

		std::vector<std::ofstream> shard_streams(proc_size);
		for (int rank = 0; rank < proc_size; rank++) {
			const std::string shard_path = shard_dir + "/rank_" + std::to_string(rank) + ".bin";
			shard_streams[rank].open(shard_path, std::ios::binary | std::ios::trunc);
			if (!shard_streams[rank].is_open())
				ERROR("Unable to open shard file " + shard_path);
		}

		Configuration cfg;
		std::ifstream ifs(cfgfnm, std::ios::binary);
		for (int cntr = 0; cfg.Load(ifs); cntr++) {
			const int owner = cntr % proc_size;
			cfg.SaveBin(shard_streams[owner]);
		}
		for (std::ofstream& ofs : shard_streams)
			ofs.close();
	}

	BroadcastString(shard_dir);

	const std::string local_shard = shard_dir + "/rank_" + std::to_string(proc_rank) + ".bin";
	stats = LoadConfigsFromFile(local_shard, target);

#ifdef MLIP_MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif

	if (proc_rank == 0) {
		for (int rank = 0; rank < proc_size; rank++) {
			const std::string shard_path = shard_dir + "/rank_" + std::to_string(rank) + ".bin";
			std::remove(shard_path.c_str());
		}
		rmdir(shard_dir.c_str());
	}

#ifdef MLIP_MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif

	return stats;
}

} // namespace

DatasetStats AddConfigs(const string cfgfnm, NonLinearRegression& dtr, int proc_rank, int proc_size)
{
	(void)dtr;

#ifdef MLIP_MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif

	return LoadConfigsViaRank0Shards(cfgfnm, proc_rank, proc_size, training_set, "train");
}

void Rescale(MTPR_trainer& trainer, MLMTPR& mtpr)
{
	double min_scaling = mtpr.scaling;
	double max_scaling = mtpr.scaling;
	int ind;
	do {
		double condition_number[5];
		double scaling = mtpr.scaling;
		double scalings[5] = { scaling / 1.21,scaling / 1.11, scaling, scaling * 1.1, scaling * 1.2 };
		vector<double> coeffs;
		coeffs.resize(mtpr.linear_coeffs.size() - mtpr.species_count + 2);
		if (prank == 0) std::cout << "Rescaling...\n";
		for (int j = 0; j < 5; j++) {
			if (prank == 0) std::cout << "   scaling = " << scalings[j] << ", condition number = ";
			mtpr.scaling = scalings[j];
			//                    if (prank == 0) std::cout << " _check  ";
			trainer.TrainLinear(prank, training_set);
			//                      if (prank == 0) std::cout << " _check  ";
			mtpr.LinCoeff();
			//                        if (prank == 0) std::cout << " _check  ";
			double rms = 13.0;
			coeffs[0] = -3.0;
			coeffs[1] = -2.0;
			for (int i = 0; i < mtpr.linear_coeffs.size() - mtpr.species_count; i++) {

				coeffs[i + 2] = std::abs(mtpr.linear_coeffs[i + mtpr.species_count]);
				rms += coeffs[i] * coeffs[i];
			}
			rms = sqrt(rms);
			std::sort(coeffs.begin(), coeffs.end());

			double median = coeffs[coeffs.size() / 2];

			condition_number[j] = rms / median;
			if (prank == 0) std::cout << rms / median << "\n";
		}

		// finds minimal condition number
		ind = 2;
		for (int j = 0; j < 5; j++)
			if (condition_number[j] < condition_number[ind]) ind = j;

		mtpr.scaling = scalings[ind];
		if (prank == 0) std::cout << "Rescaling to " << mtpr.scaling << "... ";
		trainer.TrainLinear(prank, training_set);
		if (prank == 0) std::cout << "done" << std::endl;
		if ((min_scaling < mtpr.scaling) && (mtpr.scaling < max_scaling))
			ind = 2;
		else {
			min_scaling = std::min(min_scaling, mtpr.scaling);
			max_scaling = std::max(max_scaling, mtpr.scaling);
		}
	} while (ind != 2);
}

void Train_MTPR(std::vector<std::string>& args, std::map<std::string, std::string>& opts)
{
	//args[0] - potname
	//args[1] - ts_name

	double weight_energy = 1.0;
	if (opts["energy-weight"] != "")
		weight_energy = stod(opts["energy-weight"]);
        double weight_std = 0.0;
                if(opts["std-weight"] != "")
                        weight_std = stod(opts["std-weight"]);
        

        double weight_stdd = 0.0000;
                if(opts["stdd-weight"] != "")
                        weight_stdd = stod(opts["stdd-weight"]);

	double weight_force = 0.01;
	if (opts["force-weight"] != "")
		weight_force = stod(opts["force-weight"]);

	double weight_stress = 0.001;
	if (opts["stress-weight"] != "")
		weight_stress = stod(opts["stress-weight"]);

	double scale_by_force = 0.0;
	if (opts["scale-by-force"] != "")
		scale_by_force = stod(opts["scale-by-force"]);

	string validfnm = "";
	if (opts["valid-cfgs"] != "")
		validfnm = opts["valid-cfgs"];

	int maxits = 1000;
	if (opts["max-iter"] != "")
		maxits = stoi(opts["max-iter"]);

	bool skip_preinit = false;
	if (opts["skip-preinit"] != "")
		skip_preinit = true;
        bool do_shift =true;
        if (opts["shift"] != "")
                do_shift = false;

        bool do_lin = false;
        if (opts["do-lin"] != "")
                do_lin = true;


	string curr_fnm = "";
	if (opts["curr-pot-name"] != "")
		curr_fnm = opts["curr-pot-name"];

	string bfgs_trace_fnm = "";
	if (opts["bfgs-trace-file"] != "")
		bfgs_trace_fnm = opts["bfgs-trace-file"];

	string trained_fnm = "Trained.mtp_";
	if (opts["trained-pot-name"] != "")
		trained_fnm = opts["trained-pot-name"];

	double bfgs_conv_tol = 1e-3;
	if (opts["bfgs-conv-tol"] != "")
		bfgs_conv_tol = stod(opts["bfgs-conv-tol"]);
        
        bool do_sample = true;
	if (opts["do-samp"] != "")
		do_sample = false;


	string weighting = "vibrations";
	if (opts["weighting"] != "")
		weighting = opts["weighting"];

	if (opts["init-params"] == "")
		opts["init-params"] = "random";
	if (opts["init-params"] != "random" && opts["init-params"] != "same")
		ERROR("--init-params should be 'random' or 'same'");

	bool mindist_update = false;
	if (opts["update-mindist"] != "")
		mindist_update = true;

	SetTagLogStream("dev", &std::cout);
	int end = 1;
	MLMTPR mtpr = MLMTPR();
	for (int i = 0; i < end; i++) {
		try {
			mtpr.Load(args[0]);
			end = 1;
		}
		catch (MlipException& exp) {
			std::cout << exp.What() << std::endl;
			end = 10;
		}
	}
#ifdef MLIP_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &prank);
	MPI_Comm_size(MPI_COMM_WORLD, &psize);
#endif

#ifdef MLIP_MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif

	MTPR_trainer trainer(&mtpr, weight_energy, weight_force, weight_stress, scale_by_force, 1e-9, curr_fnm, 0);
	//LOSS FUNCTIONAL MODIFICATION!!!
	trainer.weighting = weighting;
        trainer.do_lin=do_lin;
	trainer.std_scaling = weight_std;
        trainer.stdd_scaling = weight_stdd;
	trainer.linstop = bfgs_conv_tol;	//if in 100 iterations loss decreases less than this, BFGS is finished
	trainer.curr_pot_name = curr_fnm;
	trainer.bfgs_trace_file = bfgs_trace_fnm;

	if (prank == 0)
		std::cout << "SUS2-MLIP V1.0.0"
		          << " | potential from " << args[0]
		          << ", database: " << args[1] << std::endl;

	Configuration cfg;
	DatasetStats train_stats_local;
	DatasetStats valid_stats_local;
	if (validfnm != "")
	{
		try
		{
			valid_stats_local = LoadConfigsViaRank0Shards(validfnm, prank, psize, validSet, "valid");
			if (prank == 0)
				std::cout << "validation set: " << validfnm << std::endl;
		}
		catch (MlipException& exp)
		{
			std::cout << exp.What() << std::endl;
		}
	}

	for (int i = 0; i < end; i++)
	{
		try
		{
#ifdef MLIP_MPI
			MPI_Barrier(MPI_COMM_WORLD);
#endif
			training_set.clear();
			train_stats_local = AddConfigs(args[1], trainer, prank, psize);
			end = 1;
		}
		catch (MlipException& exp)
		{
			std::cout << exp.What() << std::endl;
			end = 10;
		}
	}

	long long train_cfg_total = train_stats_local.cfg_count;
	long long train_atom_total = train_stats_local.atom_count;
	long long valid_cfg_total = valid_stats_local.cfg_count;
	long long valid_atom_total = valid_stats_local.atom_count;
#ifdef MLIP_MPI
	long long train_cfg_local = train_stats_local.cfg_count;
	long long train_atom_local = train_stats_local.atom_count;
	long long valid_cfg_local = valid_stats_local.cfg_count;
	long long valid_atom_local = valid_stats_local.atom_count;
	MPI_Allreduce(&train_cfg_local, &train_cfg_total, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&train_atom_local, &train_atom_total, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&valid_cfg_local, &valid_cfg_total, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(&valid_atom_local, &valid_atom_total, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
#endif

        if (prank == 0) {std::cout <<"num_of_species: " <<mtpr.species_count  <<std::endl;
                         std::cout <<"training structures: " << train_cfg_total << std::endl;
                         std::cout <<"training atoms: " << train_atom_total << std::endl;
                         if (validfnm != "") {
                                std::cout <<"validation structures: " << valid_cfg_total << std::endl;
                                std::cout <<"validation atoms: " << valid_atom_total << std::endl;
                         }
                         std::cout <<"num_of_parameters: " <<mtpr.regression_coeffs.size()  <<std::endl;
                         std::cout <<"num_of_scalar_basis_functions: " <<mtpr.alpha_scalar_moments  <<std::endl;
						 std::cout << "num_of_L_channels: " << mtpr.L << std::endl;
						 std::cout << "num_of_scaling_channels:" << mtpr.K_ << std::endl;
						 std::cout << "scaling map:" <<  std::endl;
						 for (int i =0; i<mtpr.mu_to_K.size();i++)
						 {
							std::cout << mtpr.mu_to_K[i]<< "  " ;
						 }
						 std::cout << std::endl;

		                    }
	//random initialization of radial coefficients
	if (opts["init-params"] == "random" && !mtpr.inited) {
		if (prank == 0) {
			std::random_device rand_device;
			std::default_random_engine generator(rand_device());
			std::uniform_real_distribution<> uniform(-1.0, 1.0);

			std::cout << "Random initialization of radial coefficients" << std::endl;
			int rb_size = mtpr.Get_RB_size();
			for (int k = 0; k < 1; k++)
				for (int l = k; l < 1; l++)
					for (int i = 0; i < mtpr.radial_func_count; i++) {
						for (int j = 0; j < rb_size + mtpr.species_count; j++) {
							mtpr.regression_coeffs[mtpr.species_count+2*mtpr.species_count*mtpr.species_count* mtpr.K_ +(k * mtpr.species_count + l) * mtpr.radial_func_count * rb_size +
							i * (rb_size+ mtpr.species_count) + j ]
							= 5e-3 * uniform(generator);
                                                if (j >= rb_size) {mtpr.regression_coeffs[mtpr.species_count+2*mtpr.species_count*mtpr.species_count* mtpr.K_ +(k * mtpr.species_count + l) * mtpr.radial_func_count * rb_size +
                                                        i * (rb_size+ mtpr.species_count)+j ] = 1.0 ; }

                                                 }
						//	mtpr.regression_coeffs[k*mtpr.radial_func_count*rb_size +
						//		i*rb_size + min(i, rb_size - 1)] = 5e-7 * uniform(generator);
					}
		}
    //     if (prank == 0) {std::cout << mtpr.regression_coeffs[mtpr.radial_func_count*( mtpr.Get_RB_size() + mtpr.species_count)-4] << std::endl;}
#ifdef MLIP_MPI
		MPI_Bcast(&mtpr.Coeff()[0], mtpr.CoeffCount(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Barrier(MPI_COMM_WORLD);
#endif
	}
         trainer.shift(do_shift);
	if (!mtpr.inited && maxits > 0 && !skip_preinit) {
		trainer.max_step_count = 75;
//		trainer.random_sample(prank, training_set);
//		if (prank == 0){
//		                        mtpr.Save(trained_fnm);
//		                                                mtpr.Save_2("unfixed1.mtp");
//		                                                                  
//
               if (prank == 0){
                        mtpr.Save("ini.mtp");
                      
                      }

               Rescale(trainer, mtpr);
               if (do_sample){
               trainer.random_sample(prank, training_set, 10);
               }
		if (prank == 0)
			std::cout << "Pre-training started" << std::endl;

		trainer.Train(training_set);

		
                //trainer.random_sample(prank, training_set, 20);
		if (prank == 0)
			std::cout << "Pre-training ended" << std::endl;
	}

	//getting the lowest min_dist for the training setZ
	double min_dist = 999;
	for (int i = 0; i < training_set.size(); i++) {
		if (training_set[i].MinDist() < min_dist)
			min_dist = training_set[i].MinDist();
	}

	double total_min_dist = min_dist;

	//finding minimum distance in configuration among the processes
#ifdef MLIP_MPI
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Allreduce(&min_dist, &total_min_dist, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#endif

	if (mindist_update) {
		if (prank == 0)
		{
			std::cout << "Found configuration with mindist=" << total_min_dist << ", MTP's mindist will be decreased\n";
		}
		mtpr.p_RadialBasis->min_dist = 0.99 * total_min_dist;
	}

	trainer.max_step_count = maxits;				//maximum step count (linesearch doesn't count)

	if (maxits > 0) {
		if (prank == 0) {
			std::cout << "Modified by Hu Yanxiao. " << std::endl;
                        std::cout << "BFGS iterations count set to " << trainer.max_step_count << std::endl;
			std::cout << "BFGS convergence tolerance set to " << bfgs_conv_tol << std::endl;
                       
			if ((weight_energy != 0) || (weight_force != 0) || (weight_stress != 0)) {
				std::cout << "Energy weight: " << weight_energy << std::endl;
				std::cout << "Force weight: " << weight_force << std::endl;
				std::cout << "Stress weight: " << weight_stress << std::endl;
                                std::cout << "std weight: " << trainer.std_scaling << std::endl;
                                std::cout << "center_std weight: " << trainer.stdd_scaling << std::endl;
			}
		}
                //trainer.std_scaling /= 10000 ;
	//        trainer.random_sample(prank, training_set);
//          	Rescale(trainer, mtpr);
		bool f = true;
		trainer.shift(f);
		trainer.Train(training_set);
		//string train_name = "pp.mtp";
//		if (prank == 0)
//			mtpr.Save("loop_1.mtp");
//		if (prank == 0) {std::cout << "loop_2 start:" << std::endl;}
//		//trainer.std_scaling *= 10 ;
//		trainer.TrainLinear(prank, training_set);
//		Rescale(trainer, mtpr);
 //               trainer.Train(training_set);
  //              if (prank == 0) {mtpr.Save("loop_2.mtp");}
//                        
//                if (prank == 0) {std::cout << "loop_3 start:" << std::endl;}
//                //trainer.std_scaling *= 10 ;
//                trainer.TrainLinear(prank, training_set);
//                Rescale(trainer, mtpr);
//                trainer.Train(training_set);
//                if (prank == 0) {mtpr.Save("loop_3.mtp");}
//                if (prank == 0) {std::cout << "loop_4 start:" << std::endl;}
                //trainer.std_scaling *= 100 ;
//                trainer.TrainLinear(prank, training_set);
 //               Rescale(trainer, mtpr);
  //              trainer.Train(training_set);
   //             if (prank == 0) {mtpr.Save("loop_4.mtp");}

		if (prank == 0){
			mtpr.Save(trained_fnm);
     //                   mtpr.Save_2("unfixed1.mtp");
                      }
     //           Rescale(trainer, mtpr);
     //           if (prank==0){ mtpr.Save_2("unfixed2.mtp");
      //                 }
	}
	ErrorMonitor errmon, bufferrmon;
	std::cout.precision(15);
	if (trainer.HasLastTrainErrorSummary()) {
		if (prank == 0) {
			const MTPR_trainer::TrainErrorSummary& train_summary = trainer.LastTrainErrorSummary();
			std::cout << "\n=== Train Summary ===\n"
			          << std::fixed << std::setprecision(3)
			          << "Structures           : " << train_cfg_total << "\n"
			          << "Atoms                : " << train_atom_total << "\n"
			          << "Energy MAE (meV/atom): " << train_summary.energy_mae_mev_atom << "\n"
			          << "Energy RMSE(meV/atom): " << train_summary.energy_rmse_mev_atom << "\n"
			          << "Force MAE  (meV/A)   : " << train_summary.force_mae_mev_a << "\n"
			          << "Force RMSE (meV/A)   : " << train_summary.force_rmse_mev_a << "\n"
			          << "Stress MAE (eV)      : " << train_summary.stress_mae_ev << "\n"
			          << "Stress RMSE(eV)      : " << train_summary.stress_rmse_ev << "\n"
			          << "=====================\n";
		}
	} else {
		errmon.reset();
		double train_energy_abs_sum = 0.0;
		double train_energy_sq_weighted_sum = 0.0;
		double train_force_abs_component_sum = 0.0;
		double train_force_sq_component_sum = 0.0;
		double train_stress_abs_component_sum = 0.0;
		double train_stress_sq_component_sum = 0.0;
		long long train_force_component_count = 0;
		long long train_stress_component_count = 0;

		for (Configuration& cfg_orig : training_set)
		{
			cfg = cfg_orig;
			mtpr.CalcEFS(cfg);
			errmon.collect(cfg_orig, cfg);
			if (cfg_orig.has_energy() && cfg.has_energy())
			{
				const double dE = cfg_orig.energy - cfg.energy;
				train_energy_abs_sum += std::abs(dE);
				train_energy_sq_weighted_sum += dE * dE / cfg.size();
			}
			if (cfg_orig.has_forces() && cfg.has_forces()) {
				for (int i = 0; i < cfg.size(); i++)
					for (int a = 0; a < 3; a++)
					{
						const double dF = cfg.force(i)[a] - cfg_orig.force(i)[a];
						train_force_abs_component_sum += std::abs(dF);
						train_force_sq_component_sum += dF * dF;
					}
				train_force_component_count += static_cast<long long>(3) * cfg.size();
			}
			if (cfg_orig.has_stresses() && cfg.has_stresses()) {
				const double d0 = cfg.stresses[0][0] - cfg_orig.stresses[0][0];
				const double d1 = cfg.stresses[1][1] - cfg_orig.stresses[1][1];
				const double d2 = cfg.stresses[2][2] - cfg_orig.stresses[2][2];
				const double d3 = cfg.stresses[1][2] - cfg_orig.stresses[1][2];
				const double d4 = cfg.stresses[0][2] - cfg_orig.stresses[0][2];
				const double d5 = cfg.stresses[0][1] - cfg_orig.stresses[0][1];
				train_stress_abs_component_sum += std::abs(d0) + std::abs(d1) + std::abs(d2) +
				                                  std::abs(d3) + std::abs(d4) + std::abs(d5);
				train_stress_sq_component_sum += d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5;
				train_stress_component_count += 6;
			}

		}
		bufferrmon.reset();
#ifdef MLIP_MPI
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Reduce(&errmon.ene_all.max, &bufferrmon.ene_all.max, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.ene_all.sum, &bufferrmon.ene_all.sum, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.epa_all.max, &bufferrmon.epa_all.max, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.epa_all.sum, &bufferrmon.epa_all.sum, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.frc_all.max, &bufferrmon.frc_all.max, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.frc_all.sum, &bufferrmon.frc_all.sum, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.ene_all.count, &bufferrmon.ene_all.count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.epa_all.count, &bufferrmon.epa_all.count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.frc_all.count, &bufferrmon.frc_all.count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.str_all.count, &bufferrmon.str_all.count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.str_all.max, &bufferrmon.str_all.max, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.str_all.sum, &bufferrmon.str_all.sum, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.vir_all.count, &bufferrmon.vir_all.count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.vir_all.max, &bufferrmon.vir_all.max, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.vir_all.sum, &bufferrmon.vir_all.sum, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		double train_energy_abs_sum_global = 0.0;
		double train_energy_sq_weighted_sum_global = 0.0;
		double train_force_abs_component_sum_global = 0.0;
		double train_force_sq_component_sum_global = 0.0;
		double train_stress_abs_component_sum_global = 0.0;
		double train_stress_sq_component_sum_global = 0.0;
		long long train_force_component_count_global = 0;
		long long train_stress_component_count_global = 0;
		MPI_Reduce(&train_energy_abs_sum, &train_energy_abs_sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&train_energy_sq_weighted_sum, &train_energy_sq_weighted_sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&train_force_abs_component_sum, &train_force_abs_component_sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&train_force_sq_component_sum, &train_force_sq_component_sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&train_stress_abs_component_sum, &train_stress_abs_component_sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&train_stress_sq_component_sum, &train_stress_sq_component_sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&train_force_component_count, &train_force_component_count_global, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&train_stress_component_count, &train_stress_component_count_global, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
			if (prank == 0)
			{
				const double train_energy_mae_mev_atom =
					train_atom_total > 0 ? 1000.0 * train_energy_abs_sum_global / static_cast<double>(train_atom_total) : 0.0;
				const double train_energy_rmse_mev_atom =
					train_atom_total > 0 ? 1000.0 * std::sqrt(train_energy_sq_weighted_sum_global / static_cast<double>(train_atom_total)) : 0.0;
				const double train_force_mae_mev_a =
					train_force_component_count_global > 0 ? 1000.0 * train_force_abs_component_sum_global / static_cast<double>(train_force_component_count_global) : 0.0;
				const double train_force_rmse_mev_a =
					train_force_component_count_global > 0 ? 1000.0 * std::sqrt(train_force_sq_component_sum_global / static_cast<double>(train_force_component_count_global)) : 0.0;
				const double train_stress_mae_ev =
					train_stress_component_count_global > 0 ? train_stress_abs_component_sum_global / static_cast<double>(train_stress_component_count_global) : 0.0;
				const double train_stress_rmse_ev =
					train_stress_component_count_global > 0 ? std::sqrt(train_stress_sq_component_sum_global / static_cast<double>(train_stress_component_count_global)) : 0.0;
				std::cout << "\n=== Train Summary ===\n"
				          << std::fixed << std::setprecision(3)
				          << "Structures           : " << train_cfg_total << "\n"
				          << "Atoms                : " << train_atom_total << "\n"
				          << "Energy MAE (meV/atom): " << train_energy_mae_mev_atom << "\n"
				          << "Energy RMSE(meV/atom): " << train_energy_rmse_mev_atom << "\n"
				          << "Force MAE  (meV/A)   : " << train_force_mae_mev_a << "\n"
				          << "Force RMSE (meV/A)   : " << train_force_rmse_mev_a << "\n"
				          << "Stress MAE (eV)      : " << train_stress_mae_ev << "\n"
				          << "Stress RMSE(eV)      : " << train_stress_rmse_ev << "\n"
				          << "=====================\n";
			}

#else
		const double train_energy_mae_mev_atom =
			train_atom_total > 0 ? 1000.0 * train_energy_abs_sum / static_cast<double>(train_atom_total) : 0.0;
		const double train_energy_rmse_mev_atom =
			train_atom_total > 0 ? 1000.0 * std::sqrt(train_energy_sq_weighted_sum / static_cast<double>(train_atom_total)) : 0.0;
		const double train_force_mae_mev_a =
			train_force_component_count > 0 ? 1000.0 * train_force_abs_component_sum / static_cast<double>(train_force_component_count) : 0.0;
		const double train_force_rmse_mev_a =
			train_force_component_count > 0 ? 1000.0 * std::sqrt(train_force_sq_component_sum / static_cast<double>(train_force_component_count)) : 0.0;
		const double train_stress_mae_ev =
			train_stress_component_count > 0 ? train_stress_abs_component_sum / static_cast<double>(train_stress_component_count) : 0.0;
		const double train_stress_rmse_ev =
			train_stress_component_count > 0 ? std::sqrt(train_stress_sq_component_sum / static_cast<double>(train_stress_component_count)) : 0.0;
		std::cout << "\n=== Train Summary ===\n"
		          << std::fixed << std::setprecision(3)
		          << "Structures           : " << train_cfg_total << "\n"
		          << "Atoms                : " << train_atom_total << "\n"
		          << "Energy MAE (meV/atom): " << train_energy_mae_mev_atom << "\n"
		          << "Energy RMSE(meV/atom): " << train_energy_rmse_mev_atom << "\n"
		          << "Force MAE  (meV/A)   : " << train_force_mae_mev_a << "\n"
		          << "Force RMSE (meV/A)   : " << train_force_rmse_mev_a << "\n"
		          << "Stress MAE (eV)      : " << train_stress_mae_ev << "\n"
		          << "Stress RMSE(eV)      : " << train_stress_rmse_ev << "\n"
		          << "=====================\n";
#endif
	}

	errmon.reset();
	bufferrmon.reset();
#ifdef MLIP_MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif

	if (validfnm != "")
	{
		double valid_energy_abs_sum = 0.0;
		double valid_energy_sq_weighted_sum = 0.0;
		double valid_force_abs_component_sum = 0.0;
		double valid_force_sq_component_sum = 0.0;
		double valid_stress_abs_component_sum = 0.0;
		double valid_stress_sq_component_sum = 0.0;
		long long valid_force_component_count = 0;
		long long valid_stress_component_count = 0;
		for (Configuration& cfg_orig : validSet)
		{
			cfg = cfg_orig;
			mtpr.CalcEFS(cfg);
			errmon.collect(cfg_orig, cfg);
			if (cfg_orig.has_energy() && cfg.has_energy())
			{
				const double dE = cfg_orig.energy - cfg.energy;
				valid_energy_abs_sum += std::abs(dE);
				valid_energy_sq_weighted_sum += dE * dE / cfg.size();
			}
			if (cfg_orig.has_forces() && cfg.has_forces()) {
				for (int i = 0; i < cfg.size(); i++)
					for (int a = 0; a < 3; a++)
					{
						const double dF = cfg.force(i)[a] - cfg_orig.force(i)[a];
						valid_force_abs_component_sum += std::abs(dF);
						valid_force_sq_component_sum += dF * dF;
					}
				valid_force_component_count += static_cast<long long>(3) * cfg.size();
			}
			if (cfg_orig.has_stresses() && cfg.has_stresses()) {
				const double d0 = cfg.stresses[0][0] - cfg_orig.stresses[0][0];
				const double d1 = cfg.stresses[1][1] - cfg_orig.stresses[1][1];
				const double d2 = cfg.stresses[2][2] - cfg_orig.stresses[2][2];
				const double d3 = cfg.stresses[1][2] - cfg_orig.stresses[1][2];
				const double d4 = cfg.stresses[0][2] - cfg_orig.stresses[0][2];
				const double d5 = cfg.stresses[0][1] - cfg_orig.stresses[0][1];
				valid_stress_abs_component_sum += std::abs(d0) + std::abs(d1) + std::abs(d2) +
				                                  std::abs(d3) + std::abs(d4) + std::abs(d5);
				valid_stress_sq_component_sum += d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5;
				valid_stress_component_count += 6;
			}
		}
		bufferrmon.reset();
#ifdef MLIP_MPI
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Reduce(&errmon.ene_all.max, &bufferrmon.ene_all.max, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.ene_all.sum, &bufferrmon.ene_all.sum, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.epa_all.max, &bufferrmon.epa_all.max, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.epa_all.sum, &bufferrmon.epa_all.sum, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.frc_all.max, &bufferrmon.frc_all.max, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.frc_all.sum, &bufferrmon.frc_all.sum, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.ene_all.count, &bufferrmon.ene_all.count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.epa_all.count, &bufferrmon.epa_all.count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.frc_all.count, &bufferrmon.frc_all.count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.str_all.count, &bufferrmon.str_all.count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.str_all.max, &bufferrmon.str_all.max, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.str_all.sum, &bufferrmon.str_all.sum, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.vir_all.count, &bufferrmon.vir_all.count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.vir_all.max, &bufferrmon.vir_all.max, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&errmon.vir_all.sum, &bufferrmon.vir_all.sum, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		double valid_energy_abs_sum_global = 0.0;
		double valid_force_abs_component_sum_global = 0.0;
		double valid_stress_abs_component_sum_global = 0.0;
		long long valid_force_component_count_global = 0;
		long long valid_stress_component_count_global = 0;
		MPI_Reduce(&valid_energy_abs_sum, &valid_energy_abs_sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		double valid_energy_sq_weighted_sum_global = 0.0;
		double valid_force_sq_component_sum_global = 0.0;
		double valid_stress_sq_component_sum_global = 0.0;
		MPI_Reduce(&valid_force_abs_component_sum, &valid_force_abs_component_sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&valid_stress_abs_component_sum, &valid_stress_abs_component_sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&valid_energy_sq_weighted_sum, &valid_energy_sq_weighted_sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&valid_force_sq_component_sum, &valid_force_sq_component_sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&valid_stress_sq_component_sum, &valid_stress_sq_component_sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&valid_force_component_count, &valid_force_component_count_global, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&valid_stress_component_count, &valid_stress_component_count_global, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		if (prank == 0)
		{
			const double valid_energy_mae_mev_atom =
				valid_atom_total > 0 ? 1000.0 * valid_energy_abs_sum_global / static_cast<double>(valid_atom_total) : 0.0;
			const double valid_energy_rmse_mev_atom =
				valid_atom_total > 0 ? 1000.0 * std::sqrt(valid_energy_sq_weighted_sum_global / static_cast<double>(valid_atom_total)) : 0.0;
			const double valid_force_mae_mev_a =
				valid_force_component_count_global > 0 ? 1000.0 * valid_force_abs_component_sum_global / static_cast<double>(valid_force_component_count_global) : 0.0;
			const double valid_force_rmse_mev_a =
				valid_force_component_count_global > 0 ? 1000.0 * std::sqrt(valid_force_sq_component_sum_global / static_cast<double>(valid_force_component_count_global)) : 0.0;
			const double valid_stress_mae_ev =
				valid_stress_component_count_global > 0 ? valid_stress_abs_component_sum_global / static_cast<double>(valid_stress_component_count_global) : 0.0;
			const double valid_stress_rmse_ev =
				valid_stress_component_count_global > 0 ? std::sqrt(valid_stress_sq_component_sum_global / static_cast<double>(valid_stress_component_count_global)) : 0.0;
			std::cout << "\n=== Validation Summary ===\n"
			          << std::fixed << std::setprecision(3)
			          << "Structures           : " << valid_cfg_total << "\n"
			          << "Atoms                : " << valid_atom_total << "\n"
			          << "Energy MAE (meV/atom): " << valid_energy_mae_mev_atom << "\n"
			          << "Energy RMSE(meV/atom): " << valid_energy_rmse_mev_atom << "\n"
			          << "Force MAE  (meV/A)   : " << valid_force_mae_mev_a << "\n"
			          << "Force RMSE (meV/A)   : " << valid_force_rmse_mev_a << "\n"
			          << "Stress MAE (eV)      : " << valid_stress_mae_ev << "\n"
			          << "Stress RMSE(eV)      : " << valid_stress_rmse_ev << "\n"
			          << "==========================\n";
		}
#else
		const double valid_energy_mae_mev_atom =
			valid_atom_total > 0 ? 1000.0 * valid_energy_abs_sum / static_cast<double>(valid_atom_total) : 0.0;
		const double valid_energy_rmse_mev_atom =
			valid_atom_total > 0 ? 1000.0 * std::sqrt(valid_energy_sq_weighted_sum / static_cast<double>(valid_atom_total)) : 0.0;
		const double valid_force_mae_mev_a =
			valid_force_component_count > 0 ? 1000.0 * valid_force_abs_component_sum / static_cast<double>(valid_force_component_count) : 0.0;
		const double valid_force_rmse_mev_a =
			valid_force_component_count > 0 ? 1000.0 * std::sqrt(valid_force_sq_component_sum / static_cast<double>(valid_force_component_count)) : 0.0;
		const double valid_stress_mae_ev =
			valid_stress_component_count > 0 ? valid_stress_abs_component_sum / static_cast<double>(valid_stress_component_count) : 0.0;
		const double valid_stress_rmse_ev =
			valid_stress_component_count > 0 ? std::sqrt(valid_stress_sq_component_sum / static_cast<double>(valid_stress_component_count)) : 0.0;
		std::cout << "\n=== Validation Summary ===\n"
		          << std::fixed << std::setprecision(3)
		          << "Structures           : " << valid_cfg_total << "\n"
		          << "Atoms                : " << valid_atom_total << "\n"
		          << "Energy MAE (meV/atom): " << valid_energy_mae_mev_atom << "\n"
		          << "Energy RMSE(meV/atom): " << valid_energy_rmse_mev_atom << "\n"
		          << "Force MAE  (meV/A)   : " << valid_force_mae_mev_a << "\n"
		          << "Force RMSE (meV/A)   : " << valid_force_rmse_mev_a << "\n"
		          << "Stress MAE (eV)      : " << valid_stress_mae_ev << "\n"
		          << "Stress RMSE(eV)      : " << valid_stress_rmse_ev << "\n"
		          << "==========================\n";
#endif
	}
#ifdef MLIP_MPI
	MPI_Barrier(MPI_COMM_WORLD);
	//MPI_Finalize();
#endif
}
