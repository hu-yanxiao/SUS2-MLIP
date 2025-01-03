/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin
 */

#include "relaxation.h"
#include <sstream>
#ifdef MLIP_MPI
#	include <mpi.h>
#endif //MLIP_MPI


using namespace std;


const char* Relaxation::tagname = {"relaxation"};

void Relaxation::WriteLog()
{
    std::stringstream logstrm1;
	logstrm1	<< "cfg " << struct_cntr
				<< "\titr " << step
				<< "\t dx " << max_step
				<< "\t md " << curr_mindist
				<< "\te/a " << cfg.energy/cfg.size()
				<< "\tent " << f
				<< "\tfrc " << max_force
				<< "\tstr " << max_stress;
	if(cfg.features["MV_grade"]!="")
    	    logstrm1		<< "\tgrade " << cfg.features["MV_grade"];	
	logstrm1		<< std::endl;
    MLP_LOG(tagname,logstrm1.str());
};

void Relaxation::EFStoFuncGrad()
{
	double vol = fabs(cfg.lattice.det());

	f = cfg.energy + pressure*vol / 160.21766208;

	if (relax_cell_flag)
	{
		Matrix3 Stress = cfg.stresses - Matrix3::Id()*(1.0/160.21766208)*pressure*vol;
		Matrix3 dEdL = cfg.lattice.inverse().transpose() * Stress;
		for (int a=0; a<3; a++)
			for (int b=0; b<3; b++)
				g[3*a+b] = -dEdL[a][b];

		max_stress = Stress.MaxAbs() * 160.21766208 / fabs(cfg.lattice.det());
	}
	else
		max_stress = cfg.virial().MaxAbs() * 160.21766208;	// for correct output in log

	max_force = 0.0;
	if (relax_atoms_flag)
	{
		for (int i=0; i<cfg.size(); i++)
		{
			Vector3 frac_force = cfg.lattice * cfg.force(i);
			g[offset+3*i+0] = -frac_force[0];
			g[offset+3*i+1] = -frac_force[1];
			g[offset+3*i+2] = -frac_force[2];
			max_force = __max(max_force, cfg.force(i).Norm());
		}
	}
	else // for correct output in log
		for (int i=0; i<cfg.size(); i++)
			max_force = __max(max_force, cfg.force(i).Norm());
}

void Relaxation::XtoLatPos()
{
	max_step = 0.0;

	if (relax_cell_flag)
	{
		// Calculation of the new lattice in Cartesian coordinates
		Matrix3 dfrm = cfg.lattice.inverse();
		dfrm *= Matrix3(x[0], x[1], x[2],
						x[3], x[4], x[5],
						x[6], x[7], x[8]);
		Matrix3 new_lattice(cfg.lattice*dfrm);

		if (relax_atoms_flag) 
			cfg.lattice = new_lattice;
		else
			cfg.Deform(dfrm);

		// max_step calculation
		Vector3 delta_l1(x[0]-x_prev[0], x[1]-x_prev[1], x[2]-x_prev[2]);
		Vector3 delta_l2(x[3]-x_prev[3], x[4]-x_prev[4], x[5]-x_prev[5]);
		Vector3 delta_l3(x[6]-x_prev[6], x[7]-x_prev[7], x[8]-x_prev[8]);
		max_step = __max(__max(delta_l1.Norm(), delta_l2.Norm()), delta_l3.Norm());
	}

	if (relax_atoms_flag)
	{
		// Calculation of the new atomic positions in Cartesian coordinates
		Matrix3 LT(cfg.lattice.transpose());
		for (int i=0; i<cfg.size(); i++)
		{
			Vector3 frac_coord(x[offset+3*i+0], x[offset+3*i+1], x[offset+3*i+2]);
			Vector3 period(floor(frac_coord[0]), floor(frac_coord[1]), floor(frac_coord[2]));
			frac_coord -= period;
			cfg.pos(i) = LT*frac_coord;
		}

		// max_step calculation
		for (int i=0; i<cfg.size(); i++)
		{
			Vector3 delta_x(x[offset+3*i+0] - x_prev[offset+3*i+0],
							x[offset+3*i+1] - x_prev[offset+3*i+1],
							x[offset+3*i+2] - x_prev[offset+3*i+2]);
			max_step = __max(max_step, (delta_x * cfg.lattice).Norm());
		}
	}

	cfg.set_new_id();
}

void Relaxation::LatPosToX()
{
	if (relax_cell_flag)
	{
		//Lattice relaxation in Cartesian coordinates
		for (int a=0; a<3; a++)
			for (int b=0; b<3; b++)
				x[3*a+b] = cfg.lattice[a][b];
	}
	if (relax_atoms_flag)
	{
		Matrix3 invLT(cfg.lattice.inverse().transpose());
		for (int i=0; i<cfg.size(); i++)
		{
			Vector3 frac_coord = cfg.direct_pos(i);
			x[offset+3*i+0] = frac_coord[0];
			x[offset+3*i+1] = frac_coord[1];
			x[offset+3*i+2] = frac_coord[2];
		}
	}
}

void Relaxation::Init()
{
	// setting flags
	relax_atoms_flag = (tol_force > 0);
	relax_cell_flag = (tol_stress > 0);
	if (!relax_atoms_flag && !relax_cell_flag)
		ERROR("No relaxation activated");
	offset = relax_cell_flag ? 9 : 0;

	// Checking and repairing initial configuration
	cfg.MoveAtomsIntoCell();
	curr_mindist = cfg.MinDist();
	if (curr_mindist < mindist_constraint)
		ERROR("Relaxation: Minimal distance in intial configuration is too short. " +
							 to_string(curr_mindist));

	// initalization of BFGS argument array 'x'
	int new_size = offset + (relax_atoms_flag ? 3*cfg.size() : 0);
	x.resize(new_size);
	x_prev.resize(new_size);
	g.resize(new_size);
	LatPosToX();
	bfgs.Set_x(x.data(), new_size);
	bfgs.Restart();						// Cleaning Hessian
	max_step = 0.0;

	// setting the initial Hessian
	const Matrix3 LT = cfg.lattice * cfg.lattice.transpose();
	// setting force constants;
	if (relax_atoms_flag) 
	{
		const double force_const = 10; // typical force constant in eV/A^2
		const Matrix3 force_const_inv = (1.0 / force_const) * LT.inverse();
		for (int i=0; i<cfg.size(); i++) 
			for (int a=0; a<3; a++)
				for (int b=0; b<3; b++)
					bfgs.inv_hess(offset + 3*i + a, offset + 3*i + b) = force_const_inv[a][b];
	}
	// setting the elastic constants
	if (relax_cell_flag) 
	{
		const double el_const = 500; // typical (slightly overestimated) elast. constant in GPa
		const Matrix3 el_const_inv = (cfg.lattice.transpose() * cfg.lattice) *
									 (160.2176487 / el_const / fabs(cfg.lattice.det()));
		for (int i=0; i<3; i++) 
			for (int a=0; a<3; a++)
				for (int b=0; b<3; b++)
					bfgs.inv_hess(3*i + a, 3*i + b) = el_const_inv[a][b];
	}
}

Relaxation::Relaxation(	AnyPotential* _pot,
						const Configuration& _cfg,
						double _tol_force,
						double _tol_stress )
{
	p_potential = _pot;
	cfg = _cfg;
	tol_force = _tol_force;
	tol_stress = _tol_stress;

// 	SetLogStream(log_output);
}

Relaxation::Relaxation(AnyPotential* _p_potential, const Settings& settings)
{
	InitSettings();
	ApplySettings(settings);
	
	Message("Using internal relaxation driver");

	p_potential = _p_potential;

//	InitSettings(); // no need - to be deleted

#ifdef MLIP_MPI
	int mpi_rank;
	int mpi_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	if (!log_output.empty() &&
		log_output != "stdout" &&
		log_output != "stderr")
		log_output += '_' + to_string(mpi_rank);
#endif //MLIP_MPI

	PrintSettings();

// 	SetLogStream(log_output);

  SetTagLogStream(tagname, log_output);

	Message("Relaxation initialization complete");
}

Relaxation::~Relaxation()
{
}

bool Relaxation::CheckConvergence()
{
	return	((max_force < tol_force) || (!relax_atoms_flag)) && 
			((max_stress < tol_stress) || (!relax_cell_flag));
}

bool Relaxation::FixStep()
{
	while (curr_mindist < mindist_constraint || max_step > maxstep_constraint)
	{
		x = bfgs.ReduceStep(0.5);
		XtoLatPos();
		curr_mindist = cfg.MinDist();

		//logstrm	<< " -> (" << curr_mindist
		//		<< ", " << max_step
		//		<< ")";

		if (max_step < minstep_constraint)
			return false;
	}

	return true;
}

void Relaxation::Run()
{
	struct_cntr++;
	if (p_potential!=nullptr && cfg.size()>0 && itr_limit>0)
	{
		p_potential->CalcEFS(cfg);
		Init();
		Message("Relaxation: starting relaxation cfg#" + to_string(struct_cntr));
	}
	else
	{
		Warning("Relaxation: Unable to start relaxation");
		return;
	}

  std::stringstream logstrm1;
	// Run main loop
	for (step=0; step<itr_limit; step++)
	{
		p_potential->CalcEFS(cfg);
		EFStoFuncGrad();

		WriteLog();

		//	check convergence
		if (CheckConvergence())
		{
			logstrm1 << '\n';
			MLP_LOG(tagname,logstrm1.str());
			cfg.features["min_dist"] = to_string(cfg.MinDist());
			cfg.features["from"] = "relaxation_OK";
			Message("Relaxation of cfg#" + to_string(struct_cntr) + " complete");
			return;
		}

		// backuping previous x
		memcpy(x_prev.data(), x.data(), x.size()*sizeof(double));
		
		// Do BFGS iteration
		try { x = bfgs.Iterate(f, g); }
		catch (const MlipException& excep) {
			Warning("Relaxation: BFGS crashed. Error message:\n" + excep.message );
			bfgs.Restart();
			LatPosToX();
			bfgs.Set_x(x.data(), (int)x.size());
		}

		// updating lattice and atomic positions
		XtoLatPos();
		curr_mindist = cfg.MinDist();

		// May appear either while LOTF or if minstep_constraint is too high
		if (max_step < minstep_constraint)
		{
			Warning("Relaxation: Too small steps detected. Breaking linesearch");
			bfgs.Set_x(x);	// Reseting linesearch
		}

		if (curr_mindist < mindist_constraint || max_step > maxstep_constraint )
		{
				//logstrm << "Relaxation: cfg " << struct_cntr
				//		<< " Fixing step: (mindist = " << curr_mindist
				//		<< ", max_step = " << max_step
				//		<< ")";
				//if (GetLogStream() != nullptr) GetLogStream()->flush();

			// trying to fix step to meet the constraints
			if (FixStep())	
				;//logstrm	<< " - Ok" << endl;
			else	// unable to meet the constrains
			{
				logstrm1	<< '\n';
                MLP_LOG(tagname,logstrm1.str());
				Warning("Relaxation failed. Unable to fit to the constraints");
				// load configuration with minimal achieved energy
				bfgs.ReduceStep(0.0); //roll back to previous x
				cfg.features["min_dist"] = to_string(curr_mindist);
				if (curr_mindist < mindist_constraint)
				{
					cfg.features["from"] = "relaxation_MINDIST_ERROR";
					throw MlipException("Unable to fit to mindist constraint");
				}
				else
				{
					cfg.features["from"] = "relaxation_ERROR";
					ERROR("Relaxation error");
				}
				return;
			}
		}
	}

	if (!CheckConvergence())
	{// load configuration (positions) with minimal achieved energy
		logstrm1	<< '\n';
        MLP_LOG(tagname,logstrm1.str());
		Warning("Relaxation: convergence has not achieved");
		bfgs.ReduceStep(0.0);
		XtoLatPos();
		p_potential->CalcEFS(cfg);
		cfg.features["from"] = "relaxation_NOT_CONVERGED";
		cfg.features["min_dist"] = to_string(cfg.MinDist());
	}
	else
		ERROR("Really strange situation");
}

void Relaxation::PrintDebug(const char * fnm)	// for testing purposes
{
	std::ofstream ofs(fnm, std::ios::app);
	cfg.MoveAtomsIntoCell();
	for (int i = 0; i < cfg.size(); i++)
		ofs << cfg.pos(i, 0) << ' ' << cfg.pos(i, 1) << ' ' << cfg.pos(i, 2) << '\t';
	ofs << std::endl;
	ofs.close();
}
