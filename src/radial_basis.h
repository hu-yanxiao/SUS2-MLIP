/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Alexander Shapeev, Evgeny Podryabinkin, Konstantin Gubaev
 */

#ifndef MLIP_RADIAL_BASIS_H
#define MLIP_RADIAL_BASIS_H

#include "mlip.h"

class AnyRadialBasis
{	
public:
	int rb_size;

	double min_dist;
	double max_dist;
	double scaling = 1.0; // all functions are multiplied by scaling 

	// values and derivatives, set by calc(r)
	std::vector<double> rb_vals;
	std::vector<double> rb_ders;

	virtual std::string GetRBTypeString()
	{
		return "RBAny";
	}

	void ReadRadialBasis(std::ifstream& ifs);
	void WriteRadialBasis(std::ofstream& ofs);

	AnyRadialBasis(double _min_dist, double _max_dist, int _size);;
	AnyRadialBasis(std::ifstream& ifs);
	virtual ~AnyRadialBasis() {};
	
	virtual void RB_Calc(double r,double scal=0.1, double s= 0.1) = 0;	// calculates values and derivatives
};



class RadialBasis_Shapeev : public AnyRadialBasis
{
private:
	static const int ALLOCATED_DEGREE = 11;
	double rb_coeffs[ALLOCATED_DEGREE + 1][ALLOCATED_DEGREE + 3];
	void InitShapeevRB();
	
public:	
	std::string GetRBTypeString() override
	{
		return "RBShapeev";
	}

	RadialBasis_Shapeev(double _min_dist, double _max_dist, int _size);
	RadialBasis_Shapeev(std::ifstream& ifs);

	void RB_Calc(double r, double scal=0.1, double s= 0.1) override;
};



class RadialBasis_Chebyshev : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBChebyshev";
	}

	RadialBasis_Chebyshev(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Chebyshev(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r, double scal=0.1, double s=0.1) override;
};

class RadialBasis_Chebyshev_s : public AnyRadialBasis
{
public:
        std::string GetRBTypeString() override
        {
                return "RBChebyshev_s";
        }

        RadialBasis_Chebyshev_s(double _min_dist, double _max_dist, int _size)
                : AnyRadialBasis(_min_dist, _max_dist, _size) {};
        RadialBasis_Chebyshev_s(std::ifstream& ifs)
                : AnyRadialBasis(ifs) {};

        void RB_Calc(double r, double scal=0.1, double s=0.1) override;
};

class RadialBasis_Chebyshev_ss : public AnyRadialBasis
{
public:
        std::string GetRBTypeString() override
        {
                return "RBChebyshev_ss";
        }

        RadialBasis_Chebyshev_ss(double _min_dist, double _max_dist, int _size)
                : AnyRadialBasis(_min_dist, _max_dist, _size) {};
        RadialBasis_Chebyshev_ss(std::ifstream& ifs)
                : AnyRadialBasis(ifs) {};

        void RB_Calc(double r, double scal=0.1, double s=0.1) override;
};


class RadialBasis_Chebyshev_sss : public AnyRadialBasis
{
public:
        std::string GetRBTypeString() override
        {
                return "RBChebyshev_sss";
        }

        RadialBasis_Chebyshev_sss(double _min_dist, double _max_dist, int _size)
                : AnyRadialBasis(_min_dist, _max_dist, _size) {};
        RadialBasis_Chebyshev_sss(std::ifstream& ifs)
                : AnyRadialBasis(ifs) {};

        void RB_Calc(double r, double scal=0.1, double s=0.1) override;
};


class RadialBasis_Chebyshev_sss_lmp : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBChebyshev_ssss";
	}

	RadialBasis_Chebyshev_sss_lmp(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Chebyshev_sss_lmp(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r, double scal = 0.1, double s = 0.1) override;
};

class RadialBasis_Chebyshev_ssss : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBChebyshev_ssss";
	}

	RadialBasis_Chebyshev_ssss(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Chebyshev_ssss(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r, double scal = 0.1, double s = 0.1) override;
};


class RadialBasis_Chebyshev_repuls : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBChebyshev_repuls";
	}

	RadialBasis_Chebyshev_repuls(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Chebyshev_repuls(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r, double scal=0.1, double s=0.1) override;
};



class RadialBasis_Taylor : public AnyRadialBasis
{
public:	
	std::string GetRBTypeString() override
	{
		return "RBTaylor";
	}

	RadialBasis_Taylor(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Taylor(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r, double scal=0.1, double s=0.1) override;
};

#endif // MLIP_RADIAL_BASIS_H
