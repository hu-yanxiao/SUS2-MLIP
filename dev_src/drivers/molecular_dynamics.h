/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */

#ifndef MLIP_MOLECULAR_DYNAMICS_H
#define MLIP_MOLECULAR_DYNAMICS_H

#pragma once

#include <random>
#include "../../src/common/multidimensional_arrays.h"
#include "../../src/drivers/basic_drivers.h"


//! Implements the molecular dynamics driver.
//! The potential is supplied as a reference -- means that if you modify the potential
//! outside MolecularDynamics, it will affect it
class MolecularDynamics : public AnyDriver
{
  public:
	enum class Thermostat { NoseHoover, Langevin };

  private:
	const double kBoltz = 8.61733238e-5;
	const double eVA3_to_GPa = 160.2176487;
	const double AtomicMass_to_eVpsA = 1.03642768e-4;

	double ksi;
	Thermostat thermostat;
	double target_kin_en;
	std::default_random_engine* p_generator;
	std::vector<Vector3> vels;
	double t;

	double sum_kin_en;
	double sum_pot_en;
	double sum_tot_en;
	double sum_kin_temp;
	double sum_press;
	double sum_stress_xx;
	double sum_stress_yy;
	double sum_stress_zz;
	double sum_stress_xy;
	double sum_stress_xz;
	double sum_stress_yz;
	double sum_mindist;
	double sum_vibr_amp;

	void InitVels(double temp, int seed_init);
	void Update_X();
	void Update_F();
	void Update_V();
	void Update_Vtmp();
	double KineticEnergy();

	void InitSettings()
	{
		MakeSetting(step_limit, "Driver:MD:Max_steps");
		MakeSetting(dt, "Driver:MD:Time_step");
		MakeSetting(temp, "Driver:MD:Temp");
		MakeSetting(default_atom_mass, "DRVR_MD_AtomMass");
		MakeSetting(trmstt, "Driver:MD:Thermostat");
		MakeSetting(NH_thermostat_power, "Driver:MD:Thermostat:NH:Power");
		MakeSetting(OL_thermostat_seed, "Driver:MD:Thermostat:OL:Seed");
		MakeSetting(seed_init, "Driver:MD:Seed_init");
		MakeSetting(ave_startstep, "Driver:MD:Averaging:Start_step");
		MakeSetting(ave_skipsteps, "Driver:MD:Averaging:Steps_to_skip");
		MakeSetting(log_output, "Driver:MD:Log");
	}

  public:

	// settings
	int step_limit = HUGE_INT;
	double dt = 0.001;
	double temp = 0;
	double default_atom_mass = 1.0;
	int trmstt = 0;
	double NH_thermostat_power = 0.0;
	int OL_thermostat_seed = 0;
	int seed_init = 0;
	int ave_startstep = 0;
	int ave_skipsteps = 0;
	std::string log_output = "";

	Configuration cfg;
	Configuration init_cfg;
	int count;
	double volume;
	Array1D masses;


	int step;

	int aver_cntr;

	double kin_en;
	double pot_en;
	double tot_en;
	double kin_temp;
	double press;
	double stress_xx;
	double stress_yy;
	double stress_zz;
	double stress_xy;
	double stress_xz;
	double stress_yz;
	double mindist;
	double vibr_amp;
//	Vector3 CenterMass0;
	inline double GetAve_kin_en() { return sum_kin_en / aver_cntr; };
	inline double GetAve_pot_en() { return sum_pot_en / aver_cntr; };
	inline double GetAve_tot_en() { return sum_tot_en / aver_cntr; };
	inline double GetAve_kin_temp() { return sum_kin_temp / aver_cntr; };
	inline double GetAve_press() { return sum_press / aver_cntr; };
	inline double GetAve_stress_xx() { return sum_stress_xx / aver_cntr; };
	inline double GetAve_stress_yy() { return sum_stress_yy / aver_cntr; };
	inline double GetAve_stress_zz() { return sum_stress_zz / aver_cntr; };
	inline double GetAve_stress_xy() { return sum_stress_xy / aver_cntr; };
	inline double GetAve_stress_xz() { return sum_stress_xz / aver_cntr; };
	inline double GetAve_stress_yz() { return sum_stress_yz / aver_cntr; };
	inline double GetAve_MinDist() { return sum_mindist / aver_cntr; };
	inline double GetAve_vibr_amp() { return sum_vibr_amp / aver_cntr; };

	void Reset(	AnyPotential * _potential,
				Configuration& _init_cfg, 
				Array1D _masses,
				double _temp,
				int _step_limit = HUGE_INT,
				Thermostat _termostat = Thermostat::NoseHoover,
				double _dt = 0.001,
				double NH_thermostat_power = 0.0,
				int _seed_init = 0,
				int OL_thermostat_seed = 0);
	MolecularDynamics(	AnyPotential * _potential,
						Configuration& _init_cfg,
						Array1D _masses,
						double _temp,
						int _step_limit = HUGE_INT,
						Thermostat _termostat = Thermostat::NoseHoover,
						double _dt = 0.001,
						double NH_thermostat_power = 0.0,
						int _seed_init = 0,
						int OL_thermostat_seed = 0);
	MolecularDynamics(	AnyPotential * _potential,
						Configuration& cfg_init, 
						std::map<std::string, std::string> _settings);
	~MolecularDynamics();

	Vector3 CenterMass(Configuration& conf);

	void PrintConf(const std::string& filename);
	void PrintVels(const std::string& filename);

	void Init();
	void NextTimeStep_NoseHoover();
	void NextTimeStep_Langevin();
	void CalcObservables();

	void Run() override;
};

#endif //#ifndef MLIP_MOLECULAR_DYNAMICS_H
