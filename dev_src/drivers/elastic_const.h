/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */

#ifndef MLIP_ELASTIC_CONST_H
#define MLIP_ELASTIC_CONST_H

#include <iostream>
#include <string>
#include "../../src/drivers/basic_drivers.h"


class ElasticConst : public AnyDriver
{
private:
	const double eVA3_to_GPa = 160.2176487;

	Configuration cfg;
	double sp[6];
	double sn[6];
	std::vector<Matrix3> strains;

	inline void GetStressPos(Configuration& conf)
	{
		double volume = fabs(cfg.lattice.det());
		sp[0] = eVA3_to_GPa*conf.stresses[0][0]/volume;
		sp[1] = eVA3_to_GPa*conf.stresses[1][1]/volume;
		sp[2] = eVA3_to_GPa*conf.stresses[2][2]/volume;
		sp[3] = eVA3_to_GPa*0.5*(conf.stresses[1][2]+conf.stresses[2][1])/volume;
		sp[4] = eVA3_to_GPa*0.5*(conf.stresses[0][2]+conf.stresses[2][0])/volume;
		sp[5] = eVA3_to_GPa*0.5*(conf.stresses[1][0]+conf.stresses[0][1])/volume;
	};
	inline void GetStressNeg(Configuration& conf)
	{
		double volume = fabs(cfg.lattice.det());
		sn[0] = eVA3_to_GPa*conf.stresses[0][0]/volume;
		sn[1] = eVA3_to_GPa*conf.stresses[1][1]/volume;
		sn[2] = eVA3_to_GPa*conf.stresses[2][2]/volume;
		sn[3] = eVA3_to_GPa*0.5*(conf.stresses[1][2]+conf.stresses[2][1])/volume;
		sn[4] = eVA3_to_GPa*0.5*(conf.stresses[0][2]+conf.stresses[2][0])/volume;
		sn[5] = eVA3_to_GPa*0.5*(conf.stresses[1][0]+conf.stresses[0][1])/volume;
	};
	double C[6][6];

public:
	double eps;

	ElasticConst(AnyPotential * _p_potential, Configuration& _cfg, double _eps=0.005) :
		eps(_eps)
	{
		cfg = _cfg;
		p_potential = _p_potential;	
		strains.emplace_back(eps,0,0, 0,0,0, 0,0,0);
		strains.emplace_back(0,0,0, 0,eps,0, 0,0,0);
		strains.emplace_back(0,0,0, 0,0,0, 0,0,eps);
		strains.emplace_back(0,0,0, 0,0,eps/2, 0,eps/2,0);
		strains.emplace_back(0,0,eps/2, 0,0,0, eps/2,0,0);
		strains.emplace_back(0,eps/2,0, eps/2,0,0, 0,0,0);
	};
	~ElasticConst() {};

	void Run(int MaxStepsCnt)
	{
		if (MaxStepsCnt < 25)
		{
			Warning("Elastic: steps count too small");
			return;
		}

		Configuration cfg0 = cfg;
		double vol = fabs(cfg.lattice.det());

		p_potential->CalcEFS(cfg);
		GetStressPos(cfg);
		Message("px = " + to_string(sp[0] * eVA3_to_GPa * (1/vol)));
		Message("py = " + to_string(sp[1] * eVA3_to_GPa * (1/vol)));
		Message("pz = " + to_string(sp[2] * eVA3_to_GPa * (1/vol)));

		for (int i=0; i<6; i++)
		{
			cfg = cfg0;
			Matrix3 deform = Matrix3::Id() + strains[i];
			cfg.deform(deform);
			p_potential->CalcEFS(cfg);
			GetStressPos(cfg);

			cfg = cfg0;
			deform = Matrix3::Id() - strains[i];
			cfg.deform(deform);
			p_potential->CalcEFS(cfg);
			GetStressNeg(cfg);

			for (int j=0; j<6; j++)
				C[i][j] = (sp[j]-sn[j]) / (2*eps);
		}

		for (int i=0; i<6; i++)
			for (int j=i; j<6; j++)
			{
				if (fabs((C[i][j]-C[j][i]) / (C[i][j]+C[j][i])) > 1.0e-3 && 
					__max(fabs(C[i][j]), fabs(C[j][i])) > 1.0e-6)
					Warning((std::string)"Elastic: Elastic tensor is not symmetric: " +
							"C" + to_string(i+1) + to_string(j+1) + " = " + to_string(C[i][j]) +
							", C" + to_string(j+1) + to_string(i+1) + " = " + to_string(C[j][i]));
 				C[i][j] = (C[i][j]+C[j][i])/2;
			}

		if (fabs(C[0][3])>1.0e-6 ||
			fabs(C[1][3])>1.0e-6 ||
			fabs(C[2][3])>1.0e-6 ||
			fabs(C[0][4])>1.0e-6 ||
			fabs(C[1][4])>1.0e-6 ||
			fabs(C[2][4])>1.0e-6 ||
			fabs(C[0][5])>1.0e-6 ||
			fabs(C[1][5])>1.0e-6 ||
			fabs(C[2][5])>1.0e-6)
			Warning("Elastic: Shear-elongation elastic components are large");
		if (fabs(C[3][4])>1.0e-6 ||
			fabs(C[3][5])>1.0e-6 ||
			fabs(C[4][5])>1.0e-6)
			Warning("Elastic: Shear-shear elastic components are large");

		Message("C11 = " + to_string(C[0][0]));
		Message("C22 = " + to_string(C[1][1]));
		Message("C33 = " + to_string(C[2][2]));
		Message("C12 = " + to_string(C[0][1]));
		Message("C13 = " + to_string(C[0][2]));
		Message("C23 = " + to_string(C[1][2]));
		Message("C44 = " + to_string(C[3][3]));
		Message("C55 = " + to_string(C[4][4]));
		Message("C66 = " + to_string(C[5][5]));
	};

};

#endif //#ifndef MLIP_ELASTIC_CONST_H
