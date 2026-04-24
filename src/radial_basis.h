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
	
	virtual void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) = 0;	// calculates values and derivatives
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

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
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

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
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

        void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
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

        void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};



class RadialBasis_Chebyshev_ssw : public AnyRadialBasis
{

private:
	static const std::vector<double> arr;


public:
        std::string GetRBTypeString() override
        {
                return "RBChebyshev_ssw";
        }

        RadialBasis_Chebyshev_ssw(double _min_dist, double _max_dist, int _size)
                : AnyRadialBasis(_min_dist, _max_dist, _size) {};
        RadialBasis_Chebyshev_ssw(std::ifstream& ifs)
                : AnyRadialBasis(ifs) {};

        void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
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

        void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};


class RadialBasis_Chebyshev_sssw : public AnyRadialBasis
{

private:
        static const std::vector<double> arr;

public:
        std::string GetRBTypeString() override
        {
                return "RBChebyshev_sssw";
        }

        RadialBasis_Chebyshev_sssw(double _min_dist, double _max_dist, int _size)
                : AnyRadialBasis(_min_dist, _max_dist, _size) {};
        RadialBasis_Chebyshev_sssw(std::ifstream& ifs)
                : AnyRadialBasis(ifs) {};

        void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};


class RadialBasis_Chebyshev_sss_lmp : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBChebyshev_sss_lmp";
	}

	RadialBasis_Chebyshev_sss_lmp(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Chebyshev_sss_lmp(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};


class RadialBasis_Chebyshev_sssw_lmp : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBChebyshev_sssw_lmp";
	}

	RadialBasis_Chebyshev_sssw_lmp(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Chebyshev_sssw_lmp(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};



class RadialBasis_Chebyshev_ss_lmp : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBChebyshev_ss_lmp";
	}

	RadialBasis_Chebyshev_ss_lmp(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Chebyshev_ss_lmp(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};

class RadialBasis_Chebyshev_ssw_lmp : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBChebyshev_ssw_lmp";
	}

	RadialBasis_Chebyshev_ssw_lmp(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Chebyshev_ssw_lmp(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};




class RadialBasis_Chebyshev_s_lmp : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBChebyshev_s_lmp";
	}

	RadialBasis_Chebyshev_s_lmp(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Chebyshev_s_lmp(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
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

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};


class RadialBasis_Chebyshev_sssss : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBChebyshev_sssss";
	}

	RadialBasis_Chebyshev_sssss(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Chebyshev_sssss(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};



class RadialBasis_Chebyshev_sigma : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBChebyshev_sigma";
	}

	RadialBasis_Chebyshev_sigma(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Chebyshev_sigma(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};


class RadialBasis_Laguerre_log1p : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBLaguerre_log1p";
	}

	RadialBasis_Laguerre_log1p(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Laguerre_log1p(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};

class RadialBasis_Laguerre_log1p_lmp : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBLaguerre_log1p_lmp";
	}

	RadialBasis_Laguerre_log1p_lmp(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Laguerre_log1p_lmp(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};

class RadialBasis_Laguerre_log1p_pos : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBLaguerre_log1p_pos";
	}

	RadialBasis_Laguerre_log1p_pos(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Laguerre_log1p_pos(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};

class RadialBasis_Laguerre_log1p_pos_lmp : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBLaguerre_log1p_pos_lmp";
	}

	RadialBasis_Laguerre_log1p_pos_lmp(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Laguerre_log1p_pos_lmp(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};

class RadialBasis_Laguerre_log1p_noenv : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBLaguerre_log1p_noenv";
	}

	RadialBasis_Laguerre_log1p_noenv(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Laguerre_log1p_noenv(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};

class RadialBasis_Laguerre_log1p_noenv_lmp : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBLaguerre_log1p_noenv_lmp";
	}

	RadialBasis_Laguerre_log1p_noenv_lmp(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Laguerre_log1p_noenv_lmp(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};

class RadialBasis_Jacobi_sss : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBJacobi_sss";
	}

	RadialBasis_Jacobi_sss(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Jacobi_sss(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};

class RadialBasis_Jacobi_sss_lmp : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBJacobi_sss_lmp";
	}

	RadialBasis_Jacobi_sss_lmp(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Jacobi_sss_lmp(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};

class RadialBasis_Jacobi_sss_noweight : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBJacobi_sss_noweight";
	}

	RadialBasis_Jacobi_sss_noweight(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Jacobi_sss_noweight(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};

class RadialBasis_Jacobi_sss_noweight_lmp : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBJacobi_sss_noweight_lmp";
	}

	RadialBasis_Jacobi_sss_noweight_lmp(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Jacobi_sss_noweight_lmp(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};




class RadialBasis_Bessel : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBBessel";
	}

	RadialBasis_Bessel(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Bessel(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};

class RadialBasis_Besselw : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBBesselw";
	}

	RadialBasis_Besselw(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Besselw(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};


class RadialBasis_Chebyshev_Tri : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBChebyshev_Tri";
	}

	RadialBasis_Chebyshev_Tri(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Chebyshev_Tri(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};



class RadialBasis_Bessel_sss : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBBessel_sss";
	}

	RadialBasis_Bessel_sss(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Bessel_sss(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};


class RadialBasis_Bessel_sssw : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBBessel_sssw";
	}

	RadialBasis_Bessel_sssw(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Bessel_sssw(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};


class RadialBasis_Chebyshev_tanhexp : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBChebyshev_tanhexp";
	}

	RadialBasis_Chebyshev_tanhexp(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Chebyshev_tanhexp(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};

class RadialBasis_Chebyshev_tanhexp_w : public AnyRadialBasis
{
public:
	std::string GetRBTypeString() override
	{
		return "RBChebyshev_tanhexp_w";
	}

	RadialBasis_Chebyshev_tanhexp_w(double _min_dist, double _max_dist, int _size)
		: AnyRadialBasis(_min_dist, _max_dist, _size) {};
	RadialBasis_Chebyshev_tanhexp_w(std::ifstream& ifs)
		: AnyRadialBasis(ifs) {};

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
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

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
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

	void RB_Calc(double r,double scal=0.1, double s= 0.1,int k=0) override;
};

#endif // MLIP_RADIAL_BASIS_H
