/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */

#include "molecular_dynamics.h"
#include <set>

using namespace std;


void MolecularDynamics::InitVels(double _temp, int _seed_init)
{
	std::default_random_engine generator(_seed_init);
	std::normal_distribution<double> distribution(0.0, sqrt(kBoltz*_temp));
	Vector3 sumV(0.0, 0.0, 0.0);
	for (int i = 0; i < count; i++)
	{
		sumV[0] += vels[i][0] = distribution(generator) / sqrt(masses[i]);
		sumV[1] += vels[i][1] = distribution(generator) / sqrt(masses[i]);
		sumV[2] += vels[i][2] = distribution(generator) / sqrt(masses[i]);
	}
	for (int i = 0; i < count; i++)
		vels[i] -= sumV / count;
	double _kin_en = KineticEnergy();
	target_kin_en = (3.0 / 2.0)*count*kBoltz*_temp;
	double scale_factor = sqrt(target_kin_en / _kin_en);
	for (int i = 0; i < count; i++)
		vels[i] *= scale_factor;
}

void MolecularDynamics::Reset(	AnyPotential * _potential,
								Configuration& _init_cfg,
								Array1D _masses,
								double _temp,
								int _step_limit,
								Thermostat _thermostat,
								double _dt,
								double _NH_thermostat_power,
								int _seed_init,
								int _OL_thermostat_seed)
{
	p_potential = _potential;
	init_cfg =_init_cfg;
	masses=_masses;
	temp=_temp;
	step_limit=_step_limit;
	thermostat=_thermostat;
	dt=_dt;
	NH_thermostat_power=_NH_thermostat_power;
	seed_init=_seed_init;
	OL_thermostat_seed=_OL_thermostat_seed;
}

MolecularDynamics::MolecularDynamics(	AnyPotential * _potential,
										Configuration& _init_cfg,
										Array1D _masses,
										double _temp,
										int _step_limit,
										Thermostat _thermostat,
										double _dt,
										double _NH_thermostat_power,
										int _seed_init,
										int _OL_thermostat_seed) 
{
	Reset(_potential, _init_cfg, _masses, _temp, _step_limit,
		_thermostat, _dt, _NH_thermostat_power, _seed_init, _OL_thermostat_seed);
}

void MolecularDynamics::Init()
{
	count = cfg.size();
	volume = cfg.lattice.det();

	int max_type_id = 0;
	for (int i=0; i<cfg.size(); i++)
		max_type_id = __max(max_type_id, cfg.type(i));
	if (masses.size() < max_type_id) // setting masses from features
	{
		for (int i=0; i<max_type_id; i++)
			if (cfg.features.count("mass_" + to_string(i)) == 0)
				ERROR("MD: masses in initial configuration is reuired to start MD. \
						Feature \"mass_" + to_string(i) + " is not found");
			else
			{
				masses[i] = stod(cfg.features["mass_" + to_string(i)]);
				Message("mass for atom of " + to_string(i) + "-type is " + to_string(masses[i]));
			}
	}

	if (OL_thermostat_seed == 0)
		OL_thermostat_seed = rand();
	p_generator = new std::default_random_engine(OL_thermostat_seed);

	step = 0;
	t = 0.0;
	cfg = init_cfg;
	aver_cntr = 0;
	ksi = 0.0;

	vels.resize(count);
	std::random_device rand;
	if (seed_init == 0)
		seed_init = rand();
	std::default_random_engine generator(seed_init);
	InitVels(temp, seed_init);

	Update_F();

	CalcObservables();

	logstrm << "MD: initialization complete";
}

MolecularDynamics::MolecularDynamics(	AnyPotential * _p_potential,
										Configuration & _init_cfg, 
										std::map<std::string, std::string> settings)
{
	Message("Using internal MD driver");

	// Extracting atomic masses from configuration features
	set<int> types;
	for (int i=0; i<_init_cfg.size(); i++)
		types.insert(_init_cfg.type(i));
	Array1D _masses;
	for (int i=0; i<types.size(); i++)
		if (_init_cfg.features.count("mass"+to_string(i)) != 0)
			_masses.push_back(stoi(_init_cfg.features["mass"+to_string(i)]));

	Reset(_p_potential, _init_cfg, _masses, 0);

	InitSettings();
	ApplySettings(settings);
	PrintSettings();
	
	SetLogStream(log_output);
	thermostat = (trmstt == 0) ? Thermostat::NoseHoover : Thermostat::Langevin;

}

MolecularDynamics::~MolecularDynamics()
{
	delete p_generator;
}

void MolecularDynamics::Update_X()
{
	for (int i = 0; i < count; i++)
		cfg.pos(i) += dt * (1.0-dt*ksi/2) * vels[i] + 0.5 * dt*dt * cfg.force(i) / masses[i];
}

void MolecularDynamics::Update_Vtmp()
{
	for (int i = 0; i < count; i++)
	{
		vels[i] *= (1.0 - dt*ksi / 2);
		vels[i] += 0.5*dt * cfg.force(i) / masses[i];
	}
}

void MolecularDynamics::Update_F()
{
	p_potential->CalcEFS(cfg);
}

void MolecularDynamics::Update_V()
{
	for (int i = 0; i < count; i++)
		vels[i] = (vels[i] + 0.5*dt*cfg.force(i)/masses[i]) / (1.0 + dt*ksi/2); 
}

double MolecularDynamics::KineticEnergy()
{
	double summ=0.0;
	for (int i = 0; i < count; i++)
		summ += (vels[i][0]*vels[i][0] + vels[i][1]*vels[i][1] + vels[i][2]*vels[i][2]) 
				*masses[i];
	return summ/2;
}

Vector3 MolecularDynamics::CenterMass(Configuration & conf)
{
	Vector3 CM(0);
	for (int i = 0; i < count; i++)
		CM += conf.pos(i);
	CM /= count;
	return CM;
}

void MolecularDynamics::PrintConf(const string& filename)
{
	ofstream ofs(filename);
	for (int i = 0; i < count; i++)
		ofs << cfg.pos(i, 0) << '\t'
			<< cfg.pos(i, 1) << '\t'
			<< cfg.pos(i, 2) << '\n';
	ofs.close();
}

void MolecularDynamics::PrintVels(const string& filename)
{
	ofstream ofs(filename);
	for (int i = 0; i < count; i++)
		ofs << vels[i][0] << '\t'
			<< vels[i][1] << '\t'
			<< vels[i][2] << '\n';
	ofs.close();
}

void MolecularDynamics::NextTimeStep_NoseHoover()
{
	Update_X();
	Update_Vtmp();
	Update_F();
	ksi += dt * 2*(KineticEnergy() - target_kin_en) * NH_thermostat_power;
	if ((ksi >= 1.0/dt) || (ksi <= -1.0/dt))
		ERROR("MolecularDynamics::NextTimeStep(): Calculation fails due to instability");
	Update_V();
	t += dt;
}

void MolecularDynamics::NextTimeStep_Langevin()
{
	std::normal_distribution<double> normal(0.0, 1.0);

	for (int ind=0; ind<cfg.size(); ind++)
		for (int a=0; a<3; a++)
			cfg.pos(ind, a) += dt*cfg.force(ind, a) 
									+ sqrt(2 * kBoltz * temp * dt) * normal(*p_generator);
	
	p_potential->CalcEFS(cfg);
	t += dt;
}

void MolecularDynamics::CalcObservables()
{
	mindist = cfg.MinDist();
	kin_en = KineticEnergy();
	pot_en = cfg.energy;
	tot_en = kin_en + pot_en;
	kin_temp = ((2.0 / 3.0) * kin_en / count) / kBoltz;
	stress_xx = eVA3_to_GPa * cfg.stresses[0][0] / volume;
	stress_yy = eVA3_to_GPa * cfg.stresses[1][1] / volume;
	stress_zz = eVA3_to_GPa * cfg.stresses[2][2] / volume;
	stress_xy = eVA3_to_GPa * 0.5* (cfg.stresses[0][1] + cfg.stresses[1][0]) / volume;
	stress_xz = eVA3_to_GPa * 0.5* (cfg.stresses[0][2] + cfg.stresses[2][0]) / volume;
	stress_yz = eVA3_to_GPa * 0.5* (cfg.stresses[2][1] + cfg.stresses[1][2]) / volume;
	press = (stress_xx + stress_yy + stress_zz) / 3;
/*
	vibr_amp = 0.0;
	for (int i = 0; i < cfg.size(); i++)
	{
		Vector3 atom_shift = cfg.pos(i) - cfg_start.pos(i);
		vibr_amp += atom_shift*atom_shift;
	}
	vibr_amp = sqrt(vibr_amp / cfg.size());
*/
	if (step == ave_startstep)
		Message("MD: starting averaging\n");
	if (step >= ave_startstep && (step - ave_startstep) % (ave_skipsteps + 1) == 0)
	{
		sum_kin_en += kin_en;
		sum_pot_en += pot_en;
		sum_tot_en += tot_en;
		sum_kin_temp += kin_temp;
		sum_press += press;
		sum_stress_xx += stress_xx;
		sum_stress_yy += stress_yy;
		sum_stress_zz += stress_zz;
		sum_stress_xy += stress_xy;
		sum_stress_xz += stress_xz;
		sum_stress_yz += stress_yz;
		sum_mindist += mindist;
//		sum_vibramp += vibramp;
		aver_cntr++;
	}
}

void MolecularDynamics::Run()
{
	Init();

	while (step < step_limit)
	{
		step++;

		if (thermostat == Thermostat::NoseHoover)
			NextTimeStep_NoseHoover();
		else
			NextTimeStep_Langevin();
		cfg.set_new_id();

		CalcObservables();

		logstrm << "MD: step#" << step 
				<< "\ttime " << t
				<< "\tmindist " << mindist
				//	<< CenterMass(cfg_next)[0] - CenterMass0[0] << '\t' << CenterMass(cfg_next)[1] - CenterMass0[1] << '\t' << CenterMass(cfg_next)[2] - CenterMass0[2]
				<< "\ttrmstt " << ksi
				<< "\ttemp " << kin_temp
				<< "\tpoten " << cfg.energy
				<< "\tkinen " << kin_en
				<< endl;

		//PrintConf("ConfImage.txt", cfg_curr);
	}
}

