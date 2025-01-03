/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin
 */


#include "../mlip_wrapper.h"
#include "../mlip.h"


using namespace std;


#define NEIGHMASK 0x3FFFFFFF
#define DEFAULTCUTOFF 5.0


MLIP_Wrapper *MLIP_wrp = nullptr;
Configuration comm_conf;
AnyLocalMLIP* p_mlip; // AnyLocalMLIP is needed for the neighborhood mode
double cutoff;
std::ofstream logfilestream;
bool reorder_atoms = true;


// Initilizes MLIP
void MLIP_init(const char * settings_filename,	// settings filename
			   const char * log_filename,	// filename for logging communication 
			   int ntypes,					// Number of atom types 
			   double& rcut,				// MLIP's cutoff radius returned to driver
			   int& mode)					// If 0 call of MLIP_calc_nbh() from driver is allowed. 
{
	if (log_filename != nullptr)	// log to file
	{
		logfilestream.open(log_filename);
		if (!logfilestream)
			Warning((std::string)"Cannot open file \"" + log_filename + "\" for writing MLIP log");
		else
			SetStreamForOutput(&logfilestream);
	}
	else							// log to stdout
		SetStreamForOutput(&std::cout);

	if (MLIP_wrp != nullptr)
		delete MLIP_wrp;

	Settings settings;
	settings.Load(settings_filename);

	try
	{
		MLIP_wrp = new MLIP_Wrapper(settings);
	}
	catch (MlipException& exception)
	{
		Message(exception.What());
		exit(9991);
	}

	p_mlip = (AnyLocalMLIP*)MLIP_wrp->p_mlip;
	if (p_mlip != nullptr)
		rcut = p_mlip->CutOff();
	else
		rcut = DEFAULTCUTOFF;

	mode = 0;
	mode += settings["mlip"] == "void";
	mode += settings["calculate-efs"] == "FALSE";
	mode += settings["fit"] == "TRUE";
	mode += settings["select"] == "TRUE";
	mode += settings["write-cfgs"] == "TRUE";
	mode += settings["check-errors"] == "TRUE";
	mode += settings["log"] == "TRUE";

	// TODO: fix
	if (settings["mlip"] == "sw") mode=1;

	//if (mode)
	//	Message("MLIP has been linked in configuration mode");
	//else
	//	Message("MLIP has been linked in neighborhoods mode");
	
	if (mode &&
		settings["abinitio"] == "lammps")	// LAMMPS used for EFS calculation; 4 is the LAMMPS abinitio potential
			reorder_atoms = true;					// atoms in cfg should be ordered according their ids in LAMMPS for consistency
	else 
			reorder_atoms = false;

	if ((AnyLocalMLIP*)MLIP_wrp->p_mlip == nullptr)
		rcut = DEFAULTCUTOFF;
	cutoff = rcut;
}

// calculates EFS for configuration
void MLIP_calc_cfg(	int n,			// input parameter: number of atoms
					double* lat,	// input parameter: lattice (9 double numbers)
					double** x,		// input parameter: atomic positions (cartesian, n x 3 double numbers)
					int* types,		// input parameter: array of atom types (n of integer numbers)
					int* ids,		// input parameter: array of atom indices (n of integer numbers). Required to reorder numbers when their order in the other arrays is wrong
					double& en,		// output parameter: energy of configuration 
					double** f,		// output parameter: forces on atoms (cartesian, n x 3 double numbers)
					double* stresses)	// output parameter: stresses in eV (9 double numbers)
{
	// set size, new id
	if (n != comm_conf.size()) {
		comm_conf.destroy();
		comm_conf.resize(n);
	}
	else
		comm_conf.set_new_id();

	// set lattice 
	int foo = 0;
	for (int a=0; a<3; a++)
		for (int b=0; b<3; b++)
			comm_conf.lattice[a][b] = lat[foo++];

	// set atoms types and positions
	if (reorder_atoms)
	{
		map<int, int*> ordering;
		for (int i=0; i<n; i++)
			ordering.emplace(ids[i], &types[i]);

		if (ordering.size() != n)
			ERROR("Atoms leaking/inflow while ordering");

		int cntr = 0;
		for (auto& item : ordering)
		{
			int offset = (int)(item.second - types);
			comm_conf.type(cntr) = types[offset];
			memcpy(&comm_conf.pos(cntr), &x[offset][0], 3*sizeof(double));
			cntr++;
		}
	}
	else
	{
		memcpy(&comm_conf.pos(0, 0), &x[0][0], 3 * n*sizeof(double));
		for (int i=0; i<n; i++)
			comm_conf.type(i) = types[i]-1;
	}

	try
	{
		MLIP_wrp->CalcEFS(comm_conf);
	}
	catch (MlipException& exception)
	{
		Message(exception.What());
		exit(9992);
	}

	en += comm_conf.energy;
	//memcpy(&x[0][0], &comm_conf.pos(0, 0), 3 * n*sizeof(double));
        for(int i=0; i<n; i++)
		for (int a=0; a<3; a++)
			f[i][a] += comm_conf.force(i, a);
	foo = 0;
	for (int a=0; a<3; a++)
		for (int b=0; b<3; b++)
			stresses[foo++] += comm_conf.stresses[a][b];
}

void MLIP_calc_nbh(int inum,           // input parameter: number of neighborhoods
                   int* ilist,         // input parameter: 
                   int* numneigh,      // input parameter: number of neighbors in each neighborhood (inum integer numbers)
                   int** firstneigh,   // input parameter: pointer to the first neighbor
                   int n_local_atoms,  // input parameter: number of local atoms
                   int n_ghost_atoms,  // input parameter: number of ghost atoms
                   double** x,         // input parameter: array of coordinates of atoms
                   int* types,         // input parameter: array of atom types (inum of integer numbers)
                   double** f,                    // output parameter: forces on atoms (cartesian, n x 3 double numbers)
                   double& en,                    // output parameter: summ of site energies 
                   double* site_en=nullptr,       // output parameter: array of site energies (inum double numbers). if =nullptr while call no site energy calculation is done
                   double** site_virial=nullptr)  // output parameter: array of site energies (inum double numbers). if =nullptr while call no virial-stress-per-atom calculation is done
{
	Neighborhood nbh;

	for (int ii = 0; ii < inum; ii++) 
	{
		int i = ilist[ii];
		double xtmp = x[i][0];
		double ytmp = x[i][1];
		double ztmp = x[i][2];
		int* jlist = firstneigh[i];
		int jnum = numneigh[i];

		// 1. Construct neighborgood
		nbh.count = 0;
		nbh.my_type = types[i]-1;
		nbh.types.clear();
		nbh.inds.clear();
		nbh.vecs.clear();
		nbh.dists.clear();

		for (int jj=0; jj<jnum; jj++) 
		{
			int j = jlist[jj];
			j &= NEIGHMASK;

			double delx = x[j][0] - xtmp;
			double dely = x[j][1] - ytmp;
			double delz = x[j][2] - ztmp;
			double r = sqrt(delx*delx + dely*dely + delz*delz);

			if (r < cutoff) 
			{
				nbh.count++;
				nbh.inds.emplace_back(j);
				nbh.vecs.emplace_back(delx,dely,delz);
				nbh.dists.emplace_back(r);
				nbh.types.emplace_back(types[j]-1);
			}
		}

		// 2. Calculate site energy and their derivatives
		try
		{
			p_mlip->CalcSiteEnergyDers(nbh);
		}
		catch (MlipException& excp)
		{
			Message(excp.What());
			exit(9993);
		}
		double* p_site_energy_ders = &p_mlip->buff_site_energy_ders_[0][0];
		en += p_mlip->buff_site_energy_;
		if (site_en != nullptr)
			site_en[i] = p_mlip->buff_site_energy_;

		// 3. Add site energy derivatives to force array
		for (int jj=0; jj<nbh.count; jj++)
		{
			int j = nbh.inds[jj];

			f[i][0] += p_site_energy_ders[3*jj+0];
			f[i][1] += p_site_energy_ders[3*jj+1];
			f[i][2] += p_site_energy_ders[3*jj+2];
			
			f[j][0] -= p_site_energy_ders[3*jj+0];
			f[j][1] -= p_site_energy_ders[3*jj+1];
			f[j][2] -= p_site_energy_ders[3*jj+2];
		}

		// 4. Calculate virial stresses per atom (if required)
		if (site_virial != nullptr) 
			for (int jj = 0; jj < nbh.count; jj++)
			{
				site_virial[i][0] -= p_site_energy_ders[3*jj+0] * nbh.vecs[jj][0];
				site_virial[i][1] -= p_site_energy_ders[3*jj+1] * nbh.vecs[jj][1];
				site_virial[i][2] -= p_site_energy_ders[3*jj+2] * nbh.vecs[jj][2];
				site_virial[i][3] -= 0.5 * (p_site_energy_ders[3*jj+1] * nbh.vecs[jj][0] +
											p_site_energy_ders[3*jj+0] * nbh.vecs[jj][1]);
				site_virial[i][4] -= 0.5 * (p_site_energy_ders[3*jj+2] * nbh.vecs[jj][0] +
											p_site_energy_ders[3*jj+0] * nbh.vecs[jj][2]);
				site_virial[i][5] -= 0.5 * (p_site_energy_ders[3*jj+2] * nbh.vecs[jj][1] +
											p_site_energy_ders[3*jj+1] * nbh.vecs[jj][2]);
			}
	}
}

// destroys MLIP object
void MLIP_finalize()
{
	try
	{
		delete MLIP_wrp;
	}
	catch (MlipException& excp)
	{
		Message(excp.What());
		exit(9994);
	}
	MLIP_wrp = nullptr;
	comm_conf.destroy();

	//Message("MLIP link terminated\n");
	
	if (logfilestream.is_open())
		logfilestream.close();
}

