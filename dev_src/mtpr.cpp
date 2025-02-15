/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */

#include <algorithm>
#include "mtpr.h"

#ifdef MLIP_MPI
#	include "mpi.h"
#endif

using namespace std;


void MLMTPR::Load(const string& filename)
{
	alpha_count = 0;
	energy_cmpnts = NULL;
	stress_cmpnts = NULL;

	ifstream ifs(filename);
	if (!ifs.is_open())
		ERROR((string)"Cannot open " + filename);


	char tmpline[1000];
	string tmpstr;

	ifs.getline(tmpline, 1000);
	int len = (int)((string)tmpline).length();
	if (tmpline[len - 1] == '\r')	// Ensures compatibility between Linux and Windows line endings
		tmpline[len - 1] = '\0';

	if ((string)tmpline != "MTP")
		ERROR("Can read only MTP format potentials");

		// version reading block
		ifs.getline(tmpline, 1000);
		len = (int)((string)tmpline).length();
		if (tmpline[len - 1] == '\r')	// Ensures compatibility between Linux and Windows line endings
			tmpline[len - 1] = '\0';
		if ((string)tmpline != "version = 1.1.0")
			ERROR("MTP file must have version \"1.1.0\"");

		// name/description reading block
		ifs >> tmpstr;
		if (tmpstr == "potential_name") // optional 
		{
			ifs.ignore(2);
			ifs >> pot_desc;
			ifs >> tmpstr;
		}

		if (tmpstr == "scaling") // optional 
		{
			ifs.ignore(2);
			ifs >> scaling;
			ifs >> tmpstr;
		}
		if (tmpstr != "L") // optional 
		{
			ERROR("Error reading .mtp file");
		}
		ifs.ignore(2);
		ifs >> L;
		ifs >> tmpstr;
		if (tmpstr != "scaling_map")
		{
			ERROR("Error reading .mtp file");
		}
			ifs.ignore(2);
			ifs >> scaling_map;
			ifs >> tmpstr;
		L += 1;   // L=0,1 ...... L_max

	if (tmpstr != "species_count")
		ERROR("Error reading .mtp file");
	ifs.ignore(2);
	ifs >> species_count;

	ifs >> tmpstr;
	if (tmpstr == "potential_tag")
	{
		getline(ifs, tmpstr);
		ifs >> tmpstr;
	}

	if (tmpstr != "radial_basis_type")
		ERROR("Error reading .mtp file");
	ifs.ignore(2);
	ifs >> rbasis_type;
	
	if (rbasis_type == "RBChebyshev")
		p_RadialBasis = new RadialBasis_Chebyshev(ifs);
	else if (rbasis_type == "RBChebyshev_repuls")
		p_RadialBasis = new RadialBasis_Chebyshev_repuls(ifs);
        else if (rbasis_type == "RBChebyshev_s")
                p_RadialBasis = new RadialBasis_Chebyshev_s(ifs);
        else if (rbasis_type == "RBChebyshev_ss")
                p_RadialBasis = new RadialBasis_Chebyshev_ss(ifs);
        else if (rbasis_type == "RBChebyshev_sss")
                p_RadialBasis = new RadialBasis_Chebyshev_sss(ifs);
		else if (rbasis_type == "RBChebyshev_sss_lmp")
                p_RadialBasis = new RadialBasis_Chebyshev_sss_lmp(ifs);
        else if (rbasis_type == "RBChebyshev_ssss")
                p_RadialBasis = new RadialBasis_Chebyshev_ssss(ifs);
	else if (rbasis_type == "RBShapeev")
		p_RadialBasis = new RadialBasis_Shapeev(ifs);
	else if (rbasis_type == "RBTaylor")
		p_RadialBasis = new RadialBasis_Taylor(ifs);
	else
		ERROR("Wrong radial basis type");

	// We do not need double scaling
	if (p_RadialBasis->scaling != 1.0) {
		scaling *= p_RadialBasis->scaling;
		p_RadialBasis->scaling = 1.0;
	}


	ifs >> tmpstr;
	if (tmpstr != "radial_funcs_count")
		ERROR("Error reading .mtp file");
	ifs.ignore(2);
	ifs >> radial_func_count;
	mu_to_K.resize(radial_func_count);
	if (scaling_map=="K")
	{
		K_ = radial_func_count / L;
	
	for (int n = 0; n < radial_func_count;n++)
	{
		mu_to_K[n] = n / L;
	}
	}
	else if (scaling_map=="L")
	{
		K_ = L;
		for (int n = 0; n < radial_func_count;n++)
	{
		mu_to_K[n] = n % L;
	}

	}
	else if (scaling_map=="LK")
	{
		K_ = radial_func_count;
		for (int n = 0; n < radial_func_count;n++)
	{
		mu_to_K[n] = n;
	}

	}
	else 
	{
		ERROR("Wrong scaling map");
	}



	//Radial coeffs initialization
	int pairs_count = species_count*species_count;           //number of species pairs

	char foo = ' ';
	int foo_int = 0;

	regression_coeffs.resize(species_count +2*pairs_count* K_ +radial_func_count*(p_RadialBasis->rb_size + species_count));

	inited = true;
        
        ifs >> tmpstr;
        if (tmpstr != "shift_coeffs")
        {

                for (int i = 0; i <species_count; i++)
                regression_coeffs[i] = -1.0;

                }
        else
        {ifs.ignore(4);

                for (int i = 0; i < species_count; i++){
                        ifs >> regression_coeffs[i] >> foo;}
                        ifs >> tmpstr; }
    
        
//	ifs >> tmpstr;
        if (tmpstr != "scal_coeffs")
        {
           double s2=(p_RadialBasis->min_dist+p_RadialBasis->max_dist)/2;
	   double s1=3.3*2/(-p_RadialBasis->min_dist+p_RadialBasis->max_dist);
			for (int i = 0; i < pairs_count; i++) {
				for (int j = 0; j < K_;j++) {
					regression_coeffs[species_count + j * 2 * pairs_count + pairs_count + i] = s2;
					regression_coeffs[species_count + j * 2 * pairs_count + i] = s1;
				}
			}
                }
        else
        {ifs.ignore(4);
                
                for (int i = 0; i < 2*pairs_count* K_; i++){
                        ifs >> regression_coeffs[species_count+i] >> foo;}
                        ifs >> tmpstr; }
                
	if (tmpstr == "radial_coeffs")
	{

		for (int s1 = 0; s1 < 1; s1++)
			for (int s2 = 0; s2 < 1; s2++)
			{
				ifs >> foo_int >> foo >> foo_int;

				double t;

				for (int i = 0; i < radial_func_count; i++)
				{
					ifs >> foo;
					for (int j = 0; j < p_RadialBasis->rb_size + species_count ; j++)
					{
						ifs >> t >> foo;
						regression_coeffs[species_count+2*pairs_count* K_ +i*(p_RadialBasis->rb_size + species_count) + j] = t;

					}

				}

			}

		ifs >> tmpstr;

	}
	else
	{
		//cout << "Radial coeffs not found, initializing defaults" << endl;
		inited = false;

		regression_coeffs.resize(species_count+2*species_count*species_count* K_ +radial_func_count*(p_RadialBasis->rb_size + species_count));


		for (pairs_count = 0; pairs_count < 1; pairs_count++)
			for (int i = 0; i < radial_func_count; i++)
			{

				for (int j = 0; j < p_RadialBasis->rb_size + species_count; j++)
				{	regression_coeffs[species_count+2*species_count*species_count* K_ +pairs_count*radial_func_count*(p_RadialBasis->rb_size + species_count) +
					i*(p_RadialBasis->rb_size + species_count) + j] = 1e-2;
                                        if (j >= p_RadialBasis->rb_size) {regression_coeffs[species_count+2*species_count*species_count* K_ +pairs_count*radial_func_count*(p_RadialBasis->rb_size + species_count) +
                                        i*(p_RadialBasis->rb_size + species_count) + j] = 1.1; }
                                }             
			//	regression_coeffs[pairs_count*radial_func_count*(p_RadialBasis->rb_size) +
			//		i*(p_RadialBasis->rb_size) + min(i, p_RadialBasis->rb_size)] = 1e-3;


			}
	}

	if (tmpstr != "alpha_moments_count")
		ERROR("Error reading .mtp file");
	ifs.ignore(2);
	ifs >> alpha_moments_count;
	if (ifs.fail())
		ERROR("Error reading .mtp file");

	ifs >> tmpstr;
	if (tmpstr != "alpha_index_basic_count")
		ERROR("Error reading .mtp file");
	ifs.ignore(2);
	ifs >> alpha_index_basic_count;
	if (ifs.fail())
		ERROR("Error reading .mtp file");

	ifs >> tmpstr;
	if (tmpstr != "alpha_index_basic")
		ERROR("Error reading .mtp file");
	ifs.ignore(4);

	alpha_index_basic = new int[alpha_index_basic_count][4];	
	if (alpha_index_basic == nullptr)
		ERROR("Memory allocation error");

	int radial_func_max = -1;
	for (int i = 0; i < alpha_index_basic_count; i++)
	{
		char tmpch;
		ifs.ignore(1000, '{');
		ifs >> alpha_index_basic[i][0] >> tmpch >> alpha_index_basic[i][1] >> tmpch >> alpha_index_basic[i][2] >> tmpch >> alpha_index_basic[i][3];
		if (ifs.fail())
			ERROR("Error reading .mtp file");

		if (alpha_index_basic[i][0]>radial_func_max)
			radial_func_max = alpha_index_basic[i][0];
	}
	

	//cout << radial_func_count << endl;

	if (radial_func_max!=radial_func_count-1)
		ERROR("Wrong number of radial functions specified");

	ifs.ignore(1000, '\n');

	ifs >> tmpstr;
	if (tmpstr != "alpha_index_times_count")
		ERROR("Error reading .mtp file");
	ifs.ignore(2);
	ifs >> alpha_index_times_count;
	if (ifs.fail())
		ERROR("Error reading .mtp file");

	ifs >> tmpstr;
	if (tmpstr != "alpha_index_times")
		ERROR("Error reading .mtp file");
	ifs.ignore(4);

	alpha_index_times = new int[alpha_index_times_count][4];	
	if (alpha_index_times == nullptr)
		ERROR("Memory allocation error");

	for (int i = 0; i < alpha_index_times_count; i++)
	{
		char tmpch;
		ifs.ignore(1000, '{');
		ifs >> alpha_index_times[i][0] >> tmpch >> alpha_index_times[i][1] >> tmpch >> alpha_index_times[i][2] >> tmpch >> alpha_index_times[i][3];
		if (ifs.fail())
			ERROR("Error reading .mtp file");
	}

	ifs.ignore(1000, '\n');

	ifs >> tmpstr;
	if (tmpstr != "alpha_scalar_moments")
		ERROR("Error reading .mtp file");
	ifs.ignore(2);
	ifs >> alpha_scalar_moments;
	if (alpha_index_times_count < 0)
		ERROR("Error reading .mtp file");

	alpha_moment_mapping = new int[alpha_scalar_moments];
	if (alpha_moment_mapping == nullptr)
		ERROR("Memory allocation error");

	ifs >> tmpstr;
	if (tmpstr != "alpha_moment_mapping")
		ERROR("Error reading .mtp file");
	ifs.ignore(4);
	for (int i = 0; i < alpha_scalar_moments; i++)
	{
		char tmpch = ' ';
		ifs >> alpha_moment_mapping[i] >> tmpch;
		if (ifs.fail())
			ERROR("Error reading .mtp file");
	}
	ifs.ignore(1000, '\n');

	alpha_count = alpha_scalar_moments + 1;


	//Reading linear coeffs
	ifs >> tmpstr;
	if (tmpstr != "species_coeffs")
	{
		inited = false;
		//cout << "Linear coeffs not found, initializing defaults, species_count = " << species_count << endl;
		linear_coeffs.resize(alpha_count + species_count - 1);
		for (int i = 0; i < alpha_count + species_count - 1; i++)
			linear_coeffs[i] = 1e-3;
	}
	else
	{
		ifs.ignore(4);

		linear_coeffs.resize(species_count);
		for (int i = 0; i < species_count; i++)
			ifs >> linear_coeffs[i] >> foo;


		ifs >> tmpstr;

		if (tmpstr != "moment_coeffs")
			ERROR("Cannot read linear coeffs");

		ifs.ignore(2);

		linear_coeffs.resize(alpha_count + species_count - 1);

		ifs.ignore(10, '{');


		for (int i = 0; i < alpha_count - 1; i++)
			ifs >> linear_coeffs[i + species_count] >> foo;

	}
		MemAlloc();
		DistributeCoeffs();
}



void MLMTPR::CalcDescriptors(Configuration& cfg, ofstream& ofs)
{
	int n = alpha_count + species_count - 1;

	Neighborhoods neighborhoods(cfg,p_RadialBasis->max_dist);
        ofs << "#start  ";
	ofs << cfg.size() << endl;
//	ofs <<  "cell_vec1 " << cfg.lattice[0][0] << " " << cfg.lattice[0][1] << " " << cfg.lattice[0][2]
//	    << " cell_vec2 " << cfg.lattice[1][0] << " " << cfg.lattice[1][1] << " " << cfg.lattice[1][2]
//	    << " cell_vec3 " << cfg.lattice[2][0] << " " << cfg.lattice[2][1] << " " << cfg.lattice[2][2]
//	    << " pbc 1 1 1 " << "alpha_scalar_moments " << alpha_scalar_moments << endl;
	for (int ind = 0; ind < cfg.size(); ind++) {
		Neighborhood& nbh = neighborhoods[ind];
		CalcBasisFuncs(nbh, basis_vals);

		ofs << nbh.my_type;
//		ofs << '\t' << cfg.pos(ind, 0)
//		    << '\t' << cfg.pos(ind, 1)
//		    << '\t' << cfg.pos(ind, 2);
		for (int i = 0; i < alpha_scalar_moments; i++) {
			ofs << '\t' << basis_vals[1 + i] ;
		}
//		ofs << '\t' << nbh.count;
//		for (int i = 0; i < alpha_scalar_moments; i++) {
//			for (int j = 0; j < nbh.count; j++) {
//				ofs << '\t' << basis_ders(1 + i, j, 0) / linear_coeffs[1 + i]
//				    << '\t' << basis_ders(1 + i, j, 1) / linear_coeffs[1 + i]
//				    << '\t' << basis_ders(1 + i, j, 2) / linear_coeffs[1 + i];
//			}
//		}
		ofs << endl;

		if (nbh.my_type>=species_count)
			throw MlipException("Too few species count in the MTP potential!");
	}
}

void MLMTPR::CalcpartialE(Configuration& cfg, ofstream& ofs)
{
	int n = alpha_count + species_count - 1;

	Neighborhoods neighborhoods(cfg,p_RadialBasis->max_dist);
        ofs << "#start  ";
	ofs << cfg.size() << endl;
//	ofs <<  "cell_vec1 " << cfg.lattice[0][0] << " " << cfg.lattice[0][1] << " " << cfg.lattice[0][2]
//	    << " cell_vec2 " << cfg.lattice[1][0] << " " << cfg.lattice[1][1] << " " << cfg.lattice[1][2]
//	    << " cell_vec3 " << cfg.lattice[2][0] << " " << cfg.lattice[2][1] << " " << cfg.lattice[2][2]
//	    << " pbc 1 1 1 " << "alpha_scalar_moments " << alpha_scalar_moments << endl;
	for (int ind = 0; ind < cfg.size(); ind++) {
		Neighborhood& nbh = neighborhoods[ind];
		CalcBasisFuncs(nbh, basis_vals);

		ofs << nbh.my_type;
//		ofs << '\t' << cfg.pos(ind, 0)
//		    << '\t' << cfg.pos(ind, 1)
//		    << '\t' << cfg.pos(ind, 2);
		for (int i = 0; i < alpha_scalar_moments; i++) {
			ofs << '\t' << linear_coeffs[1 + i]*basis_vals[1 + i] ;
		}
//		ofs << '\t' << nbh.count;
//		for (int i = 0; i < alpha_scalar_moments; i++) {
//			for (int j = 0; j < nbh.count; j++) {
//				ofs << '\t' << basis_ders(1 + i, j, 0) / linear_coeffs[1 + i]
//				    << '\t' << basis_ders(1 + i, j, 1) / linear_coeffs[1 + i]
//				    << '\t' << basis_ders(1 + i, j, 2) / linear_coeffs[1 + i];
//			}
//		}
		ofs << endl;

		if (nbh.my_type>=species_count)
			throw MlipException("Too few species count in the MTP potential!");
	}
}






void MLMTPR::Save(const string& filename)
{
	ofstream ofs(filename);
	ofs.setf(ios::scientific);
	ofs.precision(15);

	ofs << "MTP\n";
	ofs << "version = 1.1.0\n";
	ofs << "potential_name = " << pot_desc << endl;
	if(scaling != 1.0)
		ofs << "scaling = " << scaling << endl;
	ofs << "L = " << L-1 << endl;
	ofs << "scaling_map = " << scaling_map << endl;
	ofs << "species_count = " << species_count << endl;
	ofs << "potential_tag = " << "" << endl;
	p_RadialBasis->WriteRadialBasis(ofs);
	ofs << "\tradial_funcs_count = " << radial_func_count << '\n';
             
        ofs << "shift_coeffs = {";
        for (int i = 0; i < species_count; i++)
        {
                if (i != 0)
                        ofs << ", ";
                ofs << regression_coeffs[i];
        }
        ofs << '}' << endl;


//        ofs << "\tradial_coeffs" << '\n';


        ofs << "scal_coeffs = {";
        for (int i = 0; i < 2*species_count*species_count* K_; i++)
        {
                if (i != 0)
                        ofs << ", ";
                ofs << regression_coeffs[species_count+i];
        }
        ofs << '}' << endl;


	ofs << "\tradial_coeffs" << '\n';

	int q = species_count+2*species_count*species_count* K_;
	for (int i = 0; i < 1; i++)
		for (int j = 0; j < 1; j++)
		{
			ofs <<"\t\t"<< i << "-" << j << "\n";

			for (int k = 0; k < radial_func_count; k++)
			{
				ofs << "\t\t\t{";
				for (int l = 0; l < p_RadialBasis->rb_size + species_count; l++)
				{
					ofs << regression_coeffs[q++];
					if (l != p_RadialBasis->rb_size - 1 + species_count)
						ofs << ", ";
					else
						ofs << "}" << endl;
				}
			}
		}

	ofs << "alpha_moments_count = " << alpha_moments_count << '\n';
	ofs << "alpha_index_basic_count = " << alpha_index_basic_count << '\n';

	ofs << "alpha_index_basic = {";
	for (int i = 0; i < alpha_index_basic_count; i++)
	{
		ofs << '{'
			<< alpha_index_basic[i][0] << ", "
			<< alpha_index_basic[i][1] << ", "
			<< alpha_index_basic[i][2] << ", "
			<< alpha_index_basic[i][3] << "}";
		if (i < alpha_index_basic_count - 1)
			ofs << ", ";
	}
	ofs << "}\n";

	ofs << "alpha_index_times_count = " << alpha_index_times_count << '\n';

	ofs << "alpha_index_times = {";
	for (int i = 0; i < alpha_index_times_count; i++)
	{
		ofs << '{'
			<< alpha_index_times[i][0] << ", "
			<< alpha_index_times[i][1] << ", "
			<< alpha_index_times[i][2] << ", "
			<< alpha_index_times[i][3] << "}";
		if (i < alpha_index_times_count - 1)
			ofs << ", ";
	}
	ofs << "}\n";

	ofs << "alpha_scalar_moments = " << alpha_scalar_moments << '\n';

	ofs << "alpha_moment_mapping = {";
	for (int i = 0; i < alpha_scalar_moments - 1; i++)
		ofs << alpha_moment_mapping[i] << ", ";
	ofs << alpha_moment_mapping[alpha_scalar_moments - 1] << "}\n";

	ofs << "species_coeffs = {";
	for (int i = 0; i < species_count; i++)
	{
		if (i != 0)
			ofs << ", ";
		ofs << LinCoeff()[i];
	}
	ofs << '}' << endl;

	ofs << "moment_coeffs = {";
	for (int i = 0; i < alpha_count - 1; i++)
	{
		if (i != 0)
			ofs << ", ";
		ofs << LinCoeff()[i + species_count];
	}
	ofs << '}';

	ofs.close();
}


void MLMTPR::Save_2(const string& filename)
{
        ofstream ofs(filename);
        ofs.setf(ios::scientific);
        ofs.precision(15);

        ofs << "MTP\n";
        ofs << "version = 1.1.0\n";
        ofs << "potential_name = " << pot_desc << endl;
        if(scaling != 1.0)
                ofs << "scaling = " << scaling << endl;
        ofs << "species_count = " << species_count << endl;
        ofs << "potential_tag = " << "" << endl;
        p_RadialBasis->WriteRadialBasis(ofs);
        ofs << "\tradial_funcs_count = " << radial_func_count << '\n';

        ofs << "shift_coeffs = {";
        for (int i = 0; i < species_count; i++)
        {
                if (i != 0)
                        ofs << ", ";
                ofs << regression_coeffs[i];
        }
        ofs << '}' << endl;
  

        ofs << "scal_coeffs = {";
        for (int i = 0; i < 2*species_count*species_count; i++)
        {
                if (i != 0)
                        ofs << ", ";
                ofs << regression_coeffs[i+species_count];
        }
        ofs << '}' << endl;


        ofs << "\tradial_coeffs" << '\n';

        int q = species_count*species_count;

        for (int i = 0; i < species_count; i++)
                for (int j = 0; j < species_count; j++)
                {
                        ofs <<"\t\t"<< i << "-" << j << "\n";

                        for (int k = 0; k < radial_func_count; k++)
                        {
                                ofs << "\t\t\t{";
                                for (int l = 0; l < p_RadialBasis->rb_size ; l++)
                                {
                                        ofs << regression_coeffs[species_count+2*species_count*species_count+k*(p_RadialBasis->rb_size+species_count)+l]*regression_coeffs[species_count+2*species_count*species_count+k*(p_RadialBasis->rb_size+species_count)+p_RadialBasis->rb_size+i]
                                               *regression_coeffs[species_count+2*species_count*species_count+k*(p_RadialBasis->rb_size+species_count)+p_RadialBasis->rb_size+j] ;
                                        if (l != p_RadialBasis->rb_size - 1)
                                                ofs << ", ";
                                        else
                                                ofs << "}" << endl;
                                }
                        }
                }

        ofs << "alpha_moments_count = " << alpha_moments_count << '\n';
        ofs << "alpha_index_basic_count = " << alpha_index_basic_count << '\n';

        ofs << "alpha_index_basic = {";
        for (int i = 0; i < alpha_index_basic_count; i++)
        {
                ofs << '{'
                        << alpha_index_basic[i][0] << ", "
                        << alpha_index_basic[i][1] << ", "
                        << alpha_index_basic[i][2] << ", "
                        << alpha_index_basic[i][3] << "}";
                if (i < alpha_index_basic_count - 1)
                        ofs << ", ";
        }
        ofs << "}\n";

        ofs << "alpha_index_times_count = " << alpha_index_times_count << '\n';

        ofs << "alpha_index_times = {";
        for (int i = 0; i < alpha_index_times_count; i++)
        {
                ofs << '{'
                        << alpha_index_times[i][0] << ", "
                        << alpha_index_times[i][1] << ", "
                        << alpha_index_times[i][2] << ", "
                        << alpha_index_times[i][3] << "}";
                if (i < alpha_index_times_count - 1)
                        ofs << ", ";
        }
        ofs << "}\n";

        ofs << "alpha_scalar_moments = " << alpha_scalar_moments << '\n';

        ofs << "alpha_moment_mapping = {";
        for (int i = 0; i < alpha_scalar_moments - 1; i++)
                ofs << alpha_moment_mapping[i] << ", ";
        ofs << alpha_moment_mapping[alpha_scalar_moments - 1] << "}\n";

        ofs << "species_coeffs = {";
        for (int i = 0; i < species_count; i++)
        {
                if (i != 0)
                        ofs << ", ";
                ofs << LinCoeff()[i];
        }
        ofs << '}' << endl;

        ofs << "moment_coeffs = {";
        for (int i = 0; i < alpha_count - 1; i++)
        {
                if (i != 0)
                        ofs << ", ";
                ofs << LinCoeff()[i + species_count];
        }
        ofs << '}';

        ofs.close();
}





void MLMTPR::CalcEFSComponents(Configuration& cfg)
{
	int n = alpha_count + species_count - 1;

	Neighborhoods neighborhoods(cfg, p_RadialBasis->max_dist);

	if (cfg.size() != forces_cmpnts.size1)
		forces_cmpnts.resize(cfg.size(), n, 3);

	memset(energy_cmpnts, 0, n * sizeof(energy_cmpnts[0]));
	forces_cmpnts.set(0);
	memset(stress_cmpnts, 0, n * sizeof(stress_cmpnts[0]));

	for (int ind = 0; ind < cfg.size(); ind++) {
		Neighborhood& nbh = neighborhoods[ind];
		CalcBasisFuncsDers(nbh);

		if (nbh.my_type>=species_count)
			throw MlipException("Too few species count in the MTP potential!");

		energy_cmpnts[nbh.my_type] += basis_vals[0];

		for (int k = species_count; k < n; k++) {
			int i = k - species_count + 1;


			energy_cmpnts[k] += basis_vals[i];

			for (int j = 0; j < nbh.count; j++)
				for (int a = 0; a < 3; a++) {
					forces_cmpnts(ind, k, a) += basis_ders(i, j, a);
					forces_cmpnts(nbh.inds[j], k, a) -= basis_ders(i, j, a);
				}

			for (int j = 0; j < nbh.count; j++)
				for (int a = 0; a < 3; a++)
					for (int b = 0; b < 3; b++)
						stress_cmpnts[k][a][b] -= basis_ders(i, j, a) * nbh.vecs[j][b];
		}
	}
}

void MLMTPR::CalcEComponents(Configuration& cfg)
{
	int n = alpha_count + species_count - 1;

	Neighborhoods neighborhoods(cfg,p_RadialBasis->max_dist); 

	memset(energy_cmpnts, 0, n * sizeof(energy_cmpnts[0]));

	for (int ind = 0; ind < cfg.size(); ind++) {
		Neighborhood& nbh = neighborhoods[ind];
		CalcBasisFuncs(nbh, basis_vals);

		if (nbh.my_type>=species_count)
			throw MlipException("Too few species count in the MTP potential!");

		energy_cmpnts[nbh.my_type] += basis_vals[0];

		for (int k = species_count; k < n; k++)
		{
			int i = k - species_count + 1;
			energy_cmpnts[k] += basis_vals[i];
		}
	}
}

void MLMTPR::CalcBasisFuncs(Neighborhood& Neighborhood, double* bf_vals)
{
	int C = species_count;						//number of different species in current potential
	int K = radial_func_count;						//number of radial functions in current potential
	int R = p_RadialBasis->rb_size;				//number of Chebyshev polynomials constituting one radial function

	for (int i = 0; i < alpha_moments_count; i++)
		moment_vals[i] = 0;

	// max_alpha_index_basic calculation
	int max_alpha_index_basic = 0;
	for (int i = 0; i < alpha_index_basic_count; i++)
		max_alpha_index_basic = max(max_alpha_index_basic,
			alpha_index_basic[i][1] + alpha_index_basic[i][2] + alpha_index_basic[i][3]);
	max_alpha_index_basic++;
	dist_powers_.resize(max_alpha_index_basic);
	coords_powers_.resize(max_alpha_index_basic);

	int type_central = Neighborhood.my_type;

	if (type_central>=species_count)
			throw MlipException("Too few species count in the MTP potential!");

	for (int j = 0; j < Neighborhood.count; j++) {
		const Vector3& NeighbVect_j = Neighborhood.vecs[j];
                int type_outer = Neighborhood.types[j];
		
                
//		int type_outer = Neighborhood.types[j];/

		dist_powers_[0] = 1;
		coords_powers_[0] = Vector3(1, 1, 1);
		for (int k = 1; k < max_alpha_index_basic; k++) {
			dist_powers_[k] = dist_powers_[k - 1] * Neighborhood.dists[j];
			for (int a = 0; a < 3; a++)
				coords_powers_[k][a] = coords_powers_[k - 1][a] * NeighbVect_j[a];
		}
		std::vector<double> val_;
		std::vector<double> der_;
		val_.resize(K_*R);
		FillWithZero(val_);
		FillWithZero(der_);
		for (int k_=0; k_<K_;k_++)
		{
			p_RadialBasis->RB_Calc(Neighborhood.dists[j], 1.0 * regression_coeffs[C +2* k_ *C*C+ C * type_central + type_outer], 1.0 * regression_coeffs[C + 2 * k_ * C * C + C * C + C * type_central + type_outer]);
			for (int xi = 0; xi < p_RadialBasis->rb_size; xi++)
				val_[k_*R+xi]=p_RadialBasis->rb_vals[xi] * scaling;
	//		for (int xi = 0; xi < p_RadialBasis->rb_size; xi++)
		//		der_[k_*R+xi]=p_RadialBasis->rb_ders[xi] * scaling;

		}

		for (int i = 0; i < alpha_index_basic_count; i++) {
			double val = 0;
			int mu = alpha_index_basic[i][0];
			int k_=mu_to_K[mu];
			/* p_RadialBasis->RB_Calc(Neighborhood.dists[j], 1.0 * regression_coeffs[C +2* k_ *C*C+ C * type_central + type_outer], 1.0 * regression_coeffs[C + 2 * k_ * C * C + C * C + C * type_central + type_outer]);
			for (int xi = 0; xi < p_RadialBasis->rb_size; xi++)
				p_RadialBasis->rb_vals[xi] *= scaling;
			for (int xi = 0; xi < p_RadialBasis->rb_size * 5; xi++)
				p_RadialBasis->rb_ders[xi] *= scaling; */


			for (int xi = 0; xi < p_RadialBasis->rb_size; xi++)
				val += regression_coeffs[C+2*C*C*K_+ mu * (R+C) + xi] *regression_coeffs[C+2*C*C*K_+0 * (R+C)+ R + type_central] * regression_coeffs[C+2*C*C*K_+0 * (R+C)+ R + type_outer]* val_[k_*R+xi];
		
			int k = alpha_index_basic[i][1] + alpha_index_basic[i][2] + alpha_index_basic[i][3];
			double powk = 1.0 / dist_powers_[k];
			val *= powk;		

			double pow0 = coords_powers_[alpha_index_basic[i][1]][0];
			double pow1 = coords_powers_[alpha_index_basic[i][2]][1];
			double pow2 = coords_powers_[alpha_index_basic[i][3]][2];

			double mult0 = pow0*pow1*pow2;

			moment_vals[i] += val * mult0;
		}
	}

	// Next: calculating non-elementary b_i
	for (int i = 0; i < alpha_index_times_count; i++) {
		double val0 = moment_vals[alpha_index_times[i][0]];
		double val1 = moment_vals[alpha_index_times[i][1]];
		int val2 = alpha_index_times[i][2];
		moment_vals[alpha_index_times[i][3]] += val2 * val0 * val1;
	}

	// Next: copying all b_i corresponding to scalars into separate arrays,
	// basis_vals and basis_ders
	bf_vals[0] = 1.0;  // setting the constant basis function

	for (int i = 0; i < alpha_scalar_moments; i++) 
		bf_vals[1 + i] = moment_vals[alpha_moment_mapping[i]];
	
}

void MLMTPR::CalcBasisFuncsDers(const Neighborhood& Neighborhood)
{
	int C = species_count;						//number of different species in current potential
	int K = radial_func_count;						//number of radial functions in current potential
	int R = p_RadialBasis->rb_size;				//number of Chebyshev polynomials constituting one radial function

	if (Neighborhood.count != moment_ders.size2)
		moment_ders.resize(alpha_moments_count, Neighborhood.count, 3);

	for (int i = 0; i < alpha_moments_count; i++)
		moment_vals[i] = 0;

	moment_ders.set(0);

	// max_alpha_index_basic calculation
	int max_alpha_index_basic = 0;
	for (int i = 0; i < alpha_index_basic_count; i++)
		max_alpha_index_basic = max(max_alpha_index_basic,
			alpha_index_basic[i][1] + alpha_index_basic[i][2] + alpha_index_basic[i][3]);
	max_alpha_index_basic++;
	dist_powers_.resize(max_alpha_index_basic);
	coords_powers_.resize(max_alpha_index_basic);

	int type_central = Neighborhood.my_type;

	if (type_central>=species_count)
			throw MlipException("Too few species count in the MTP potential!");


	for (int j = 0; j < Neighborhood.count; j++) {
		const Vector3& NeighbVect_j = Neighborhood.vecs[j];
               // int type_outer = Neighborhood.types[j];
		// calculates vals and ders for j-th atom in the neighborhood
	//	p_RadialBasis->RB_Calc(Neighborhood.dists[j]);
	        int type_outer = Neighborhood.types[j];
	        
                
	//	int type_outer = Neighborhood.types[j];

		dist_powers_[0] = 1;
		coords_powers_[0] = Vector3(1, 1, 1);
		for (int k = 1; k < max_alpha_index_basic; k++) {
			dist_powers_[k] = dist_powers_[k - 1] * Neighborhood.dists[j];
			for (int a = 0; a < 3; a++)
				coords_powers_[k][a] = coords_powers_[k - 1][a] * NeighbVect_j[a];
		}
		std::vector<double> val_;
		std::vector<double> der_;
		val_.resize(K_*R);
		der_.resize(K_*R);
		FillWithZero(val_);
		FillWithZero(der_);
		
		for (int k_=0; k_<K_;k_++)
		{
			p_RadialBasis->RB_Calc(Neighborhood.dists[j], 1.0 * regression_coeffs[C +2* k_ *C*C+ C * type_central + type_outer], 1.0 * regression_coeffs[C + 2 * k_ * C * C + C * C + C * type_central + type_outer]);
			for (int xi = 0; xi < p_RadialBasis->rb_size; xi++){
				val_[k_*R+xi]=p_RadialBasis->rb_vals[xi] * scaling;
				der_[k_*R+xi]=p_RadialBasis->rb_ders[xi] * scaling;
			}
		}
		for (int i = 0; i < alpha_index_basic_count; i++) {

			double val = 0;
			double der = 0;
			int mu = alpha_index_basic[i][0];
			int k_ = mu_to_K[mu];
			/* p_RadialBasis->RB_Calc(Neighborhood.dists[j], 1.0 * regression_coeffs[C + 2 * k_ * C * C + C * type_central + type_outer], 1.0 * regression_coeffs[C + 2 * k_ * C * C + C * C + C * type_central + type_outer]);
			for (int xi = 0; xi < p_RadialBasis->rb_size; xi++)
				p_RadialBasis->rb_vals[xi] *= scaling;
			for (int xi = 0; xi < p_RadialBasis->rb_size * 5; xi++)
				p_RadialBasis->rb_ders[xi] *= scaling; */

			for (int xi = 0; xi < p_RadialBasis->rb_size; xi++)
			{

				// here \phi_xi(r) is RadialBasis::vals[xi]
				val += regression_coeffs[C+2*C*C* K_ + mu * (R+C) + xi] *regression_coeffs[C+2*C*C* K_ +0 * (R+C)+ R + type_central] * regression_coeffs[C+2*C*C* K_ +0 * (R+C)+ R + type_outer] * val_[k_*R+xi];
				der += regression_coeffs[C+2*C*C* K_ + mu * (R+C) + xi] *regression_coeffs[C+2*C*C* K_ +0 * (R+C)+ R + type_central] * regression_coeffs[C+2*C*C* K_ +0 * (R+C)+ R + type_outer] * der_[k_*R+xi];


			}

			int k = alpha_index_basic[i][1] + alpha_index_basic[i][2] + alpha_index_basic[i][3];
			double powk = 1.0 / dist_powers_[k];
			val *= powk;
			der = der * powk - k * val / Neighborhood.dists[j];

			double pow0 = coords_powers_[alpha_index_basic[i][1]][0];
			double pow1 = coords_powers_[alpha_index_basic[i][2]][1];
			double pow2 = coords_powers_[alpha_index_basic[i][3]][2];

			double mult0 = pow0*pow1*pow2;

			moment_vals[i] += val * mult0;

			mult0 *= der / Neighborhood.dists[j];
			moment_ders(i, j, 0) += mult0 * NeighbVect_j[0];
			moment_ders(i, j, 1) += mult0 * NeighbVect_j[1];
			moment_ders(i, j, 2) += mult0 * NeighbVect_j[2];



			if (alpha_index_basic[i][1] != 0) {
				moment_ders(i, j, 0) += val * alpha_index_basic[i][1]
					* coords_powers_[alpha_index_basic[i][1] - 1][0]
					* pow1
					* pow2;
			}
			if (alpha_index_basic[i][2] != 0) {
				moment_ders(i, j, 1) += val * alpha_index_basic[i][2]
					* pow0
					* coords_powers_[alpha_index_basic[i][2] - 1][1]
					* pow2;
			}
			if (alpha_index_basic[i][3] != 0) {
				moment_ders(i, j, 2) += val * alpha_index_basic[i][3]
					* pow0
					* pow1
					* coords_powers_[alpha_index_basic[i][3] - 1][2];
			}
		}
	}



	// Next: calculating non-elementary b_i
	for (int i = 0; i < alpha_index_times_count; i++) {
		double val0 = moment_vals[alpha_index_times[i][0]];
		double val1 = moment_vals[alpha_index_times[i][1]];
		int val2 = alpha_index_times[i][2];
		moment_vals[alpha_index_times[i][3]] += val2 * val0 * val1;

		for (int j = 0; j < Neighborhood.count; j++) {
			for (int a = 0; a < 3; a++) {
				moment_ders(alpha_index_times[i][3], j, a) += val2 * (moment_ders(alpha_index_times[i][0], j, a) * val1 + val0 * moment_ders(alpha_index_times[i][1], j, a));
			}
		}
	}

	// Next: copying all b_i corresponding to scalars into separate arrays,
	// basis_vals and basis_ders
	basis_vals[0] = 1.0;  // setting the constant basis function

	if (basis_ders.size2 != Neighborhood.count) // TODO: remove this check?
		basis_ders.resize(alpha_count, Neighborhood.count, 3);
	memset(&basis_ders(0, 0, 0), 0, 3 * Neighborhood.count*sizeof(double));

	for (int i = 0; i < alpha_scalar_moments; i++) {
		basis_vals[1 + i] = moment_vals[alpha_moment_mapping[i]];
		memcpy(&basis_ders(1 + i, 0, 0), &moment_ders(alpha_moment_mapping[i], 0, 0), 3 * Neighborhood.count*sizeof(double));
	}
}

void MLMTPR::MemAlloc()
{
	int n = alpha_count - 1 + species_count;

	energy_cmpnts = new double[n];
	forces_cmpnts.reserve(n * 3);
	stress_cmpnts = (double(*)[3][3])malloc(n * sizeof(stress_cmpnts[0]));

	moment_vals = new double[alpha_moments_count];
	basis_vals = new double[alpha_count];
	site_energy_ders_wrt_moments_.resize(alpha_moments_count);

}

MLMTPR::~MLMTPR()
{
	if (moment_vals != NULL) delete[] moment_vals;
	if (basis_vals != NULL) delete[] basis_vals;

	if (alpha_moment_mapping != NULL) delete[] alpha_moment_mapping;
	if (alpha_index_times != NULL) delete[] alpha_index_times;
	if (alpha_index_basic != NULL) delete[] alpha_index_basic;

	moment_vals = NULL;
	basis_vals = NULL;
	alpha_moment_mapping = NULL;
	alpha_index_times = NULL;
	alpha_index_basic = NULL;

	if (energy_cmpnts != NULL) delete[] energy_cmpnts;
	if (stress_cmpnts != NULL) free(stress_cmpnts);

	if (p_RadialBasis!=NULL)
	delete p_RadialBasis;


}

void MLMTPR::RadialCoeffsInit(ifstream& ifs_rad)
{
	if (!ifs_rad.is_open())
		ERROR("Radial functions file is not open");

	string types, type1, type2;

	int pairs_count = 0;           //number of species pairs

	types = "";
	type1 = "";
	type2 = "";
	char foo = ' ';

	ifs_rad >> types;

	if (types == "radial_coeffs")
	{

		ifs_rad >> types;

		while (!ifs_rad.eof())
		{
			int i = 0;
			pairs_count++;
			regression_coeffs.resize(pairs_count*radial_func_count*(p_RadialBasis->rb_size));

			while (types.at(i) != '-')
				type1 += types.at(i++);

			while (i != types.length() - 1)
			{
				if (types.at(i + 1) != '+')
					type2 += types.at(++i);
				else
					i++;
			}


			double t;
			for (i = 0; i < radial_func_count; i++) 
			{
				ifs_rad >> foo;
				for (int j = 0; j <= p_RadialBasis->rb_size; j++)
				{
					ifs_rad >> t >> foo;
					regression_coeffs[(pairs_count - 1)*radial_func_count*(p_RadialBasis->rb_size) +
						i*(p_RadialBasis->rb_size) + j] = t;


				}


			}

			types = "";
			type1 = "";
			type2 = "";

			ifs_rad >> types;

		};

		if (species_count == 0)
			species_count = (int)sqrt(pairs_count);

		if (abs(species_count*species_count - pairs_count) > 1e-8)
			ERROR("WRONG NUMBER OF RADIAL FUNCTION BLOCKS, MUST BE species_count^2");
	}
	else
	{

		if (species_count == 0)
			species_count = 2;


		//cout << "Radial coeffs not found, initializing defaults" << endl;

		regression_coeffs.resize(species_count*species_count*radial_func_count*(p_RadialBasis->rb_size));


		for (pairs_count = 0; pairs_count < species_count*species_count; pairs_count++)//!!! pairs_count \xEF\xE5\xF0\xE5\xEE\xEF\xF0\xE5\xE4\xE5\xEB\xB8\xED
			for (int i = 0; i < radial_func_count; i++)
			{

				for (int j = 0; j < p_RadialBasis->rb_size; j++)
					regression_coeffs[pairs_count*radial_func_count*(p_RadialBasis->rb_size) +
					i*(p_RadialBasis->rb_size) + j] = 1e-6;

				regression_coeffs[pairs_count*radial_func_count*(p_RadialBasis->rb_size) +
					i*(p_RadialBasis->rb_size) + min(i, p_RadialBasis->rb_size-1)] = 1e-3;


			}

	}


}

void MLMTPR::CalcSiteEnergyDers(const Neighborhood& nbh)
{
	buff_site_energy_ = 0.0;
        buff_site_energy_0 = 0.0;
	buff_site_energy_ders_.resize(nbh.count);
	FillWithZero(buff_site_energy_ders_);

	int C = species_count;						//number of different species in current potential
	int K = radial_func_count;						//number of radial functions in current potential
	int R = p_RadialBasis->rb_size;  //number of Chebyshev polynomials constituting one radial function


	linear_coeffs = LinCoeff();


	if (nbh.count != moment_jacobian_.size2)
		moment_jacobian_.resize(alpha_index_basic_count, nbh.count, 3);

	memset(moment_vals, 0, alpha_moments_count * sizeof(moment_vals[0]));
	moment_jacobian_.set(0);


	// max_alpha_index_basic calculation
	int max_alpha_index_basic = 0;
	for (int i = 0; i < alpha_index_basic_count; i++)
		max_alpha_index_basic = max(max_alpha_index_basic,
			alpha_index_basic[i][1] + alpha_index_basic[i][2] + alpha_index_basic[i][3]);
	max_alpha_index_basic++;
	dist_powers_.resize(max_alpha_index_basic);
	coords_powers_.resize(max_alpha_index_basic);

	int type_central = nbh.my_type;

	if (type_central>=species_count)
			throw MlipException("Too few species count in the MTP potential!");


	for (int j = 0; j < nbh.count; j++) {
		const Vector3& NeighbVect_j = nbh.vecs[j];

		// calculates vals and ders for j-th atom in the neighborhood
//		p_RadialBasis->RB_Calc(nbh.dists[j]);
//		for (int xi = 0; xi < p_RadialBasis->rb_size; xi++)
//			p_RadialBasis->rb_vals[xi] *= scaling;
//		for (int xi = 0; xi < p_RadialBasis->rb_size; xi++)
//			p_RadialBasis->rb_ders[xi] *= scaling;
                int type_outer = nbh.types[j];
                
                
		dist_powers_[0] = 1;
		coords_powers_[0] = Vector3(1, 1, 1);
		for (int k = 1; k < max_alpha_index_basic; k++) {
			dist_powers_[k] = dist_powers_[k - 1] * nbh.dists[j];
			for (int a = 0; a < 3; a++)
				coords_powers_[k][a] = coords_powers_[k - 1][a] * NeighbVect_j[a];
		}
		std::vector<double> val_;
		std::vector<double> der_;
		val_.resize(K_*R);
		der_.resize(K_*R*5);
		FillWithZero(val_);
		FillWithZero(der_);
		for (int k_=0; k_<K_;k_++)
		{
			p_RadialBasis->RB_Calc(nbh.dists[j], 1.0 * regression_coeffs[C +2* k_ *C*C+ C * type_central + type_outer], 1.0 * regression_coeffs[C + 2 * k_ * C * C + C * C + C * type_central + type_outer]);
			for (int xi = 0; xi < p_RadialBasis->rb_size; xi++)
				val_[k_*R+xi]=p_RadialBasis->rb_vals[xi] * scaling;
			for (int xi = 0; xi < p_RadialBasis->rb_size * 5; xi++)
				der_[k_*R*5+xi]=p_RadialBasis->rb_ders[xi] * scaling;

		}
//		int type_outer = nbh.types[j];

		for (int i = 0; i < alpha_index_basic_count; i++) {

			double val = 0, der = 0;
			int mu = alpha_index_basic[i][0];
			int k_ = mu_to_K[mu];
			/* p_RadialBasis->RB_Calc(nbh.dists[j], 1.0 * regression_coeffs[C + 2 * k_ * C * C + C * type_central + type_outer], 1.0 * regression_coeffs[C + 2 * k_ * C * C + C * C + C * type_central + type_outer]);
			for (int xi = 0; xi < p_RadialBasis->rb_size; xi++)
				p_RadialBasis->rb_vals[xi] *= scaling;
			for (int xi = 0; xi < p_RadialBasis->rb_size * 5; xi++)
				p_RadialBasis->rb_ders[xi] *= scaling; */

			for (int xi = 0; xi < R; xi++)
			{
				val += regression_coeffs[C+2*C*C*K_+ mu * (R+C) + xi] *regression_coeffs[C+2*C*C * K_ +0 * (R+C)+ R + type_central] * regression_coeffs[C+2*C*C * K_ +0 * (R+C)+ R + type_outer]* val_[k_*R+xi];
				der += regression_coeffs[C+2*C*C * K_ + mu * (R+C) + xi] *regression_coeffs[C+2*C*C * K_ +0 * (R+C)+ R + type_central] * regression_coeffs[C+2*C*C * K_ +0 * (R+C)+ R + type_outer] * der_[k_*R*5+xi];

			}



			int k = alpha_index_basic[i][1] + alpha_index_basic[i][2] + alpha_index_basic[i][3];
			double powk = 1.0 / dist_powers_[k];
			val *= powk;
			der = der * powk - k * val / nbh.dists[j];

			double pow0 = coords_powers_[alpha_index_basic[i][1]][0];
			double pow1 = coords_powers_[alpha_index_basic[i][2]][1];
			double pow2 = coords_powers_[alpha_index_basic[i][3]][2];

			double mult0 = pow0*pow1*pow2;

			moment_vals[i] += val * mult0;
			mult0 *= der / nbh.dists[j];
			moment_jacobian_(i, j, 0) += mult0 * NeighbVect_j[0];
			moment_jacobian_(i, j, 1) += mult0 * NeighbVect_j[1];
			moment_jacobian_(i, j, 2) += mult0 * NeighbVect_j[2];

			if (alpha_index_basic[i][1] != 0) {
				moment_jacobian_(i, j, 0) += val * alpha_index_basic[i][1]
					* coords_powers_[alpha_index_basic[i][1] - 1][0]
					* pow1
					* pow2;
			}
			if (alpha_index_basic[i][2] != 0) {
				moment_jacobian_(i, j, 1) += val * alpha_index_basic[i][2]
					* pow0
					* coords_powers_[alpha_index_basic[i][2] - 1][1]
					* pow2;
			}
			if (alpha_index_basic[i][3] != 0) {
				moment_jacobian_(i, j, 2) += val * alpha_index_basic[i][3]
					* pow0
					* pow1
					* coords_powers_[alpha_index_basic[i][3] - 1][2];
			}
		}
	
		//Repulsive term
		if (p_RadialBasis->GetRBTypeString() == "RBChebyshev_repuls")
		if (nbh.dists[j] < p_RadialBasis->min_dist)
		{
			double multiplier = 10000;
			buff_site_energy_ += multiplier*(exp(-10*(nbh.dists[j]-1)) - exp(-10*(p_RadialBasis->min_dist-1)));
			for (int a = 0; a < 3; a++)
				buff_site_energy_ders_[j][a] += -10 * multiplier*(exp(-10 * (nbh.dists[j] - 1))/ nbh.dists[j])*nbh.vecs[j][a];
		}
	}

	// Next: calculating non-elementary b_i
	for (int i = 0; i < alpha_index_times_count; i++) {
		double val0 = moment_vals[alpha_index_times[i][0]];
		double val1 = moment_vals[alpha_index_times[i][1]];
		int val2 = alpha_index_times[i][2];
		moment_vals[alpha_index_times[i][3]] += val2 * val0 * val1;
	}

	// renewing maximum absolute values
		for (int i = 0; i < alpha_scalar_moments; i++)
			max_linear[i] = max(max_linear[i],abs(linear_coeffs[species_count + i]*moment_vals[alpha_moment_mapping[i]]));


	// convolving with coefficients
	buff_site_energy_ +=  regression_coeffs[nbh.my_type]+ linear_coeffs[nbh.my_type];


	for (int i = 0; i < alpha_scalar_moments; i++)
	{	buff_site_energy_ +=  linear_coeffs[species_count + i]*linear_mults[i] * moment_vals[alpha_moment_mapping[i]]* linear_coeffs[nbh.my_type];
         buff_site_energy_0 +=  linear_coeffs[species_count + i]*linear_mults[i] * moment_vals[alpha_moment_mapping[i]]* linear_coeffs[nbh.my_type];
}

	//if (wgt_eqtn_forces != 0) //!!! CHECK IT!!!
	{
		// Backpropagation starts

		// Backpropagation step 1: site energy derivative is the corresponding linear combination
		memset(&site_energy_ders_wrt_moments_[0], 0, alpha_moments_count * sizeof(site_energy_ders_wrt_moments_[0]));

		for (int i = 0; i < alpha_scalar_moments; i++)
			site_energy_ders_wrt_moments_[alpha_moment_mapping[i]] = linear_coeffs[species_count + i]*linear_mults[i];

		// SAME BUT UNSAFE:
		// memcpy(&site_energy_ders_wrt_moments_[0], &basis_coeffs[1],
		//		alpha_scalar_moments * sizeof(site_energy_ders_wrt_moments_[0]));

		// Backpropagation step 2: expressing through basic moments:
		for (int i = alpha_index_times_count - 1; i >= 0; i--) {
			double val0 = moment_vals[alpha_index_times[i][0]];
			double val1 = moment_vals[alpha_index_times[i][1]];
			int val2 = alpha_index_times[i][2];

			site_energy_ders_wrt_moments_[alpha_index_times[i][1]] +=
				site_energy_ders_wrt_moments_[alpha_index_times[i][3]]
				* val2 * val0;
			site_energy_ders_wrt_moments_[alpha_index_times[i][0]] +=
				site_energy_ders_wrt_moments_[alpha_index_times[i][3]]
				* val2 * val1;
		}

		// Backpropagation step 3: multiply by the Jacobian:
		for (int i = 0; i < alpha_index_basic_count; i++)
			for (int j = 0; j < nbh.count; j++)
				for (int a = 0; a < 3; a++)
					buff_site_energy_ders_[j][a] += site_energy_ders_wrt_moments_[i] * moment_jacobian_(i, j, a) * linear_coeffs[nbh.my_type];

	}
}

void MLMTPR::AccumulateCombinationGrad(	const Neighborhood& nbh,
										std::vector<double>& out_grad_accumulator,
										const double se_weight,
										const Vector3* se_ders_weights) 
{
	int C = species_count;						//number of different species in current potential
	int K = radial_func_count;						//number of radial functions in current potential
	int R = p_RadialBasis->rb_size;  //number of Chebyshev polynomials constituting one radial function

	int coeff_count = C+2*C*C * K_ +(R+C)*K;

	buff_site_energy_ders_.resize(nbh.count);
	out_grad_accumulator.resize(CoeffCount());

	buff_site_energy_ = 0.0;
	FillWithZero(buff_site_energy_ders_);

	{
		linear_coeffs = LinCoeff();

		site_energy_ders_wrt_moments_.resize(alpha_moments_count);

		std::vector<double> mom_val(alpha_moments_count);

		for (int i = 0; i < alpha_moments_count; i++)
		{
			mom_val[i] = 0;
			site_energy_ders_wrt_moments_[i] = 0;
		}

		moment_ders.set(0);

		Array3D mom_jacobian_(alpha_index_basic_count, nbh.count, 3);
		mom_jacobian_.set(0);

		Array3D mom_rad_jacobian_(alpha_index_basic_count, C, R*K);
		mom_rad_jacobian_.set(0);
                Array3D mom_rad_jacobian_s(alpha_index_basic_count, C, R*K);
                mom_rad_jacobian_s.set(0);
                Array3D mom_rad_jacobian_ss(alpha_index_basic_count, C, R*K);
                mom_rad_jacobian_ss.set(0);
		Array3D mom_rad_coord_jacobian_(alpha_index_basic_count, C, R*K);
		mom_rad_coord_jacobian_.set(0);
                Array3D mom_rad_coord_jacobian_s(alpha_index_basic_count, C, R*K);
                mom_rad_coord_jacobian_s.set(0);
                Array3D mom_rad_coord_jacobian_ss(alpha_index_basic_count, C, R*K);
                mom_rad_coord_jacobian_ss.set(0);


		vector<double> dloss_dbas(alpha_moments_count);
		memset(&dloss_dbas[0], 0, alpha_moments_count*sizeof(double));

		vector<double> dloss_dsenders(alpha_moments_count);
		memset(&dloss_dsenders[0], 0, alpha_moments_count*sizeof(double));

		Array3D dloss_dmomjac(alpha_index_basic_count, nbh.count, 3);
		dloss_dmomjac.set(0);

		vector<double>dloss_dmom(alpha_moments_count);
		memset(&dloss_dmom[0], 0, alpha_moments_count*sizeof(double));

		// max_alpha_index_basic calculation
		int max_alpha_index_basic = 0;
		for (int i = 0; i < alpha_index_basic_count; i++)
			max_alpha_index_basic = max(max_alpha_index_basic,
				alpha_index_basic[i][1] + alpha_index_basic[i][2] + alpha_index_basic[i][3]);

		max_alpha_index_basic++;
		std::vector<double> auto_dist_powers_(max_alpha_index_basic);
		std::vector<Vector3> auto_coords_powers_(max_alpha_index_basic);


		int type_central = nbh.my_type;

		if (type_central>=species_count)
			throw MlipException("Too few species count in the MTP potential!");


		for (int j = 0; j < nbh.count; j++) {
			const Vector3& NeighbVect_j = nbh.vecs[j];

			// calculates vals and ders for j-th atom in the neighborhood
//			p_RadialBasis->RB_Calc(nbh.dists[j]);
//			for (int xi = 0; xi < p_RadialBasis->rb_size; xi++)
//				p_RadialBasis->rb_vals[xi] *= scaling;
//			for (int xi = 0; xi < p_RadialBasis->rb_size; xi++)
//				p_RadialBasis->rb_ders[xi] *= scaling;
                        int type_outer = nbh.types[j];
                        
                        
			auto_dist_powers_[0] = 1;
			auto_coords_powers_[0] = Vector3(1, 1, 1);
			for (int k = 1; k < max_alpha_index_basic; k++) {
				auto_dist_powers_[k] = auto_dist_powers_[k - 1] * nbh.dists[j];
				for (int a = 0; a < 3; a++)
					auto_coords_powers_[k][a] = auto_coords_powers_[k - 1][a] * NeighbVect_j[a];
			}
		std::vector<double> val_;
		std::vector<double> der_;
		val_.resize(K_*R);
		der_.resize(K_*R*5);
		FillWithZero(val_);
		FillWithZero(der_);
		for (int k_=0; k_<K_;k_++)
		{
			p_RadialBasis->RB_Calc(nbh.dists[j], 1.0 * regression_coeffs[C +2* k_ *C*C+ C * type_central + type_outer], 1.0 * regression_coeffs[C + 2 * k_ * C * C + C * C + C * type_central + type_outer]);
			for (int xi = 0; xi < p_RadialBasis->rb_size; xi++)
				val_[k_*R+xi]=p_RadialBasis->rb_vals[xi] * scaling;
			for (int xi = 0; xi < p_RadialBasis->rb_size * 5; xi++)
				der_[k_*R*5+xi]=p_RadialBasis->rb_ders[xi] * scaling;

		}
		//	int type_outer = nbh.types[j];

			for (int i = 0; i < alpha_index_basic_count; i++) {

				double val = 0;
				double der = 0;

				int mu = alpha_index_basic[i][0];
				int k = alpha_index_basic[i][1] + alpha_index_basic[i][2] + alpha_index_basic[i][3];
				int k_ = mu_to_K[mu];
				/* p_RadialBasis->RB_Calc(nbh.dists[j], 1.0 * regression_coeffs[C + 2 * k_ * C * C + C * type_central + type_outer], 1.0 * regression_coeffs[C + 2 * k_ * C * C + C * C + C * type_central + type_outer]);
				for (int xi = 0; xi < p_RadialBasis->rb_size; xi++)
					p_RadialBasis->rb_vals[xi] *= scaling;
				for (int xi = 0; xi < p_RadialBasis->rb_size * 5; xi++)
					p_RadialBasis->rb_ders[xi] *= scaling; */

				double powk = 1.0 / auto_dist_powers_[k];

				double pow0 = auto_coords_powers_[alpha_index_basic[i][1]][0];
				double pow1 = auto_coords_powers_[alpha_index_basic[i][2]][1];
				double pow2 = auto_coords_powers_[alpha_index_basic[i][3]][2];

				double mult0 = pow0*pow1*pow2;

				for (int xi = 0; xi < R; xi++)
				{
					double c_ = regression_coeffs[C + 2 * C * C*K_ + mu * (R + C) + xi] * regression_coeffs[C + 2 * C * C * K_ + 0 * (R + C) + R + type_central] * regression_coeffs[C + 2 * C * C * K_ + 0 * (R + C) + R + type_outer];
					val += regression_coeffs[C+2*C*C * K_ + mu * (R+C) + xi] *regression_coeffs[C+2*C*C * K_ +0* (R+C)+ R + type_central] * regression_coeffs[C+2*C*C * K_ +0 * (R+C)+ R + type_outer] * val_[k_*R+xi] * powk;
					der += regression_coeffs[C+2*C*C * K_ + mu * (R+C) + xi] *regression_coeffs[C+2*C*C * K_ +0 * (R+C)+ R + type_central] * regression_coeffs[C+2*C*C * K_ +0 * (R+C)+ R + type_outer] * der_[5*k_*R+xi];

					mom_val[i] += regression_coeffs[C+2*C*C * K_ + mu * (R+C) + xi] *regression_coeffs[C+2*C*C * K_ +0 * (R+C)+ R + type_central] * regression_coeffs[C+2*C*C * K_ +0 * (R+C)+ R + type_outer] * val_[k_*R+xi] * powk*mult0;

					mom_rad_jacobian_(i, type_outer, mu * R + xi) += val_[k_*R+xi] * powk*mult0;
                                        mom_rad_jacobian_s(i, type_outer, mu * R + xi) += der_[5*k_*R+xi + 1 * R] * powk*mult0*c_;
                                        mom_rad_jacobian_ss(i, type_outer, mu * R + xi) += der_[5*k_*R+xi + 3 * R] * powk*mult0* c_;
                                        //double c_ = regression_coeffs[C+2*C*C+ mu * (R+C) + xi] *regression_coeffs[C+2*C*C+mu * (R+C)+ R + type_central] * regression_coeffs[C+2*C*C+mu * (R+C)+ R + type_outer];
					if (se_ders_weights != nullptr)
					{
						double derx = (NeighbVect_j[0] / nbh.dists[j])*(der_[5*k_*R+xi] * powk*mult0 - val_[k_*R+xi] * k*powk*mult0 / nbh.dists[j]);
						double dery = (NeighbVect_j[1] / nbh.dists[j])*(der_[5*k_*R+xi] * powk*mult0 - val_[k_*R+xi] * k*powk*mult0 / nbh.dists[j]);
						double derz = (NeighbVect_j[2] / nbh.dists[j])*(der_[5*k_*R+xi] * powk*mult0 - val_[k_*R+xi] * k*powk*mult0 / nbh.dists[j]);
                                              
						double derx_s = (NeighbVect_j[0] / nbh.dists[j]) * (der_[5*k_*R+xi + 2 * R] * powk * mult0 - der_[5*k_*R+xi + 1 * R] * k * powk * mult0 / nbh.dists[j]);
						double dery_s = (NeighbVect_j[1] / nbh.dists[j]) * (der_[5*k_*R+xi + 2 * R] * powk * mult0 - der_[5*k_*R+xi+ 1 * R] * k * powk * mult0 / nbh.dists[j]);
						double derz_s = (NeighbVect_j[2] / nbh.dists[j]) * (der_[5*k_*R+xi + 2 * R] * powk * mult0 - der_[5*k_*R+xi + 1 * R] * k * powk * mult0 / nbh.dists[j]);
						double derx_ss = (NeighbVect_j[0] / nbh.dists[j]) * (der_[5*k_*R+xi + 4 * R] * powk * mult0 - der_[5*k_*R+xi + 3 * R] * k * powk * mult0 / nbh.dists[j]);
						double dery_ss = (NeighbVect_j[1] / nbh.dists[j]) * (der_[5*k_*R+xi + 4 * R] * powk * mult0 - der_[5*k_*R+xi + 3 * R] * k * powk * mult0 / nbh.dists[j]);
						double derz_ss = (NeighbVect_j[2] / nbh.dists[j]) * (der_[5*k_*R+xi + 4 * R] * powk * mult0 - der_[5*k_*R+xi + 3 * R] * k * powk * mult0 / nbh.dists[j]);

						if (alpha_index_basic[i][1] != 0) {
							derx += val_[k_*R+xi] * powk * alpha_index_basic[i][1]
								* auto_coords_powers_[alpha_index_basic[i][1] - 1][0]
								* pow1
								* pow2;
							derx_s += der_[5*k_*R+xi + 1 * R] * powk * alpha_index_basic[i][1]
								* auto_coords_powers_[alpha_index_basic[i][1] - 1][0]
								* pow1
								* pow2;
							derx_ss += der_[5*k_*R+xi + 3 * R] * powk * alpha_index_basic[i][1]
								* auto_coords_powers_[alpha_index_basic[i][1] - 1][0]
								* pow1
								* pow2;

						}

						if (alpha_index_basic[i][2] != 0) {
							dery += val_[k_*R+xi] * powk * alpha_index_basic[i][2]
								* pow0
								* auto_coords_powers_[alpha_index_basic[i][2] - 1][1]
								* pow2;
							dery_s += der_[5*k_*R+xi + 1 * R] * powk * alpha_index_basic[i][2]
								* pow0
								* auto_coords_powers_[alpha_index_basic[i][2] - 1][1]
								* pow2;
							dery_ss += der_[5*k_*R+xi + 3 * R] * powk * alpha_index_basic[i][2]
								* pow0
								* auto_coords_powers_[alpha_index_basic[i][2] - 1][1]
								* pow2;
						}

						if (alpha_index_basic[i][3] != 0) {
							derz += val_[k_*R+xi] * powk * alpha_index_basic[i][3]
								* pow0
								* pow1
								* auto_coords_powers_[alpha_index_basic[i][3] - 1][2];
							derz_s += der_[5*k_*R+xi + 1 * R] * powk * alpha_index_basic[i][3]
								* pow0
								* pow1
								* auto_coords_powers_[alpha_index_basic[i][3] - 1][2];
							derz_ss += der_[5*k_*R+xi + 3 * R] * powk * alpha_index_basic[i][3]
								* pow0
								* pow1
								* auto_coords_powers_[alpha_index_basic[i][3] - 1][2];
						}


						mom_rad_coord_jacobian_(i, type_outer, mu * R + xi) += se_ders_weights[j][0] * derx;
						mom_rad_coord_jacobian_(i, type_outer, mu * R + xi) += se_ders_weights[j][1] * dery;
						mom_rad_coord_jacobian_(i, type_outer, mu * R + xi) += se_ders_weights[j][2] * derz;

                                                mom_rad_coord_jacobian_s(i, type_outer, mu * R + xi) += se_ders_weights[j][0] * derx_s*c_;
                                                mom_rad_coord_jacobian_s(i, type_outer, mu * R + xi) += se_ders_weights[j][1] * dery_s*c_;
                                                mom_rad_coord_jacobian_s(i, type_outer, mu * R + xi) += se_ders_weights[j][2] * derz_s*c_;

                                                mom_rad_coord_jacobian_ss(i, type_outer, mu * R + xi) += se_ders_weights[j][0] * derx_ss*c_;
                                                mom_rad_coord_jacobian_ss(i, type_outer, mu * R + xi) += se_ders_weights[j][1] * dery_ss*c_;
                                                mom_rad_coord_jacobian_ss(i, type_outer, mu * R + xi) += se_ders_weights[j][2] * derz_ss*c_;
					}
				}

				//max_radial[(type_central*C + type_outer)*K + mu] = max(abs(max_radial[(type_central*C + type_outer)*K + mu]),abs(val));


				der = der * powk - k * val / nbh.dists[j];
				mult0 *= der / nbh.dists[j];

				mom_jacobian_(i, j, 0) += mult0 * NeighbVect_j[0];
				mom_jacobian_(i, j, 1) += mult0 * NeighbVect_j[1];
				mom_jacobian_(i, j, 2) += mult0 * NeighbVect_j[2];


				if (alpha_index_basic[i][1] != 0) {
					mom_jacobian_(i, j, 0) += val * alpha_index_basic[i][1]
						* auto_coords_powers_[alpha_index_basic[i][1] - 1][0]
						* pow1
						* pow2;
				}
				if (alpha_index_basic[i][2] != 0) {
					mom_jacobian_(i, j, 1) += val * alpha_index_basic[i][2]
						* pow0
						* auto_coords_powers_[alpha_index_basic[i][2] - 1][1]
						* pow2;
				}
				if (alpha_index_basic[i][3] != 0) {
					mom_jacobian_(i, j, 2) += val * alpha_index_basic[i][3]
						* pow0
						* pow1
						* auto_coords_powers_[alpha_index_basic[i][3] - 1][2];
				}

			}

			//Repulsive term
			if (p_RadialBasis->GetRBTypeString()=="RBChebyshev_repuls")
			if (nbh.dists[j] < p_RadialBasis->min_dist)
			{
                double multiplier = 10000;
                buff_site_energy_ += multiplier*(exp(-10 * (nbh.dists[j] - 1)) - exp(-10 * (p_RadialBasis->min_dist - 1)));
                for (int a = 0; a < 3; a++)
                    buff_site_energy_ders_[j][a] += -10 * multiplier*(exp(-10 * (nbh.dists[j] - 1)) / nbh.dists[j])*nbh.vecs[j][a];
            }

		}



		// Next: calculating non-elementary b_i
		for (int i = 0; i < alpha_index_times_count; i++) {
			double val0 = mom_val[alpha_index_times[i][0]];
			double val1 = mom_val[alpha_index_times[i][1]];
			double val2 = alpha_index_times[i][2];
			mom_val[alpha_index_times[i][3]] += val2 * val0 * val1;
		}
	
		// renewing maximum absolute values
		for (int i = 0; i < alpha_scalar_moments; i++)
			max_linear[i] = max(max_linear[i],abs(linear_coeffs[species_count + i]*mom_val[alpha_moment_mapping[i]]));
			

		// convolving with coefficients
		buff_site_energy_ +=   regression_coeffs[nbh.my_type]+ linear_coeffs[nbh.my_type];
		for (int i = 0; i < alpha_scalar_moments; i++)
			buff_site_energy_ += linear_coeffs[species_count + i]*linear_mults[i] * mom_val[alpha_moment_mapping[i]] * linear_coeffs[nbh.my_type];

		// Backpropagation starts

		// Backpropagation step 1: site energy derivative is the corresponding linear combination
		for (int i = 0; i < alpha_scalar_moments; i++)
			site_energy_ders_wrt_moments_[alpha_moment_mapping[i]] = linear_coeffs[species_count + i]*linear_mults[i];


		// Backpropagation step 2: expressing through basic moments:
		for (int i = alpha_index_times_count - 1; i >= 0; i--) {
			double val0 = mom_val[alpha_index_times[i][0]];
			double val1 = mom_val[alpha_index_times[i][1]];
			double val2 = alpha_index_times[i][2];

			site_energy_ders_wrt_moments_[alpha_index_times[i][1]] +=
				site_energy_ders_wrt_moments_[alpha_index_times[i][3]]
				* val2 * val0;
			site_energy_ders_wrt_moments_[alpha_index_times[i][0]] +=
				site_energy_ders_wrt_moments_[alpha_index_times[i][3]]
				* val2 * val1;
		}

		// Backpropagation step 3: multiply by the Jacobian:
		//if (se_ders_weights != nullptr)
			for (int i = 0; i < alpha_index_basic_count; i++)
				for (int j = 0; j < nbh.count; j++)
					for (int a = 0; a < 3; a++) {
						buff_site_energy_ders_[j][a] += site_energy_ders_wrt_moments_[i] * mom_jacobian_(i, j, a) * linear_coeffs[nbh.my_type];

						if (se_ders_weights != nullptr)
						dloss_dsenders[i] += se_ders_weights[j][a] * mom_jacobian_(i, j, a);
					}

		for (int i = 0; i < alpha_index_basic_count; i++)
			dloss_dmom[i] = se_weight * site_energy_ders_wrt_moments_[i];


		if (se_ders_weights)
		{

			for (int i = 0; i < alpha_index_times_count; i++) {
				double val0 = mom_val[alpha_index_times[i][0]];
				double val1 = mom_val[alpha_index_times[i][1]];
				double val2 = alpha_index_times[i][2];

				dloss_dsenders[alpha_index_times[i][3]] += dloss_dsenders[alpha_index_times[i][1]] * val2*val0;
				dloss_dsenders[alpha_index_times[i][3]] += dloss_dsenders[alpha_index_times[i][0]] * val2*val1;


			}


			for (int i = 0; i < alpha_index_times_count; i++) {
				//				double val0 = mom_val[alpha_index_times[i][0]];
				//				double val1 = mom_val[alpha_index_times[i][1]];
				double val2 = alpha_index_times[i][2];

				dloss_dmom[alpha_index_times[i][1]] += dloss_dsenders[alpha_index_times[i][0]] *
					site_energy_ders_wrt_moments_[alpha_index_times[i][3]] * val2;

				dloss_dmom[alpha_index_times[i][0]] += dloss_dsenders[alpha_index_times[i][1]] *
					site_energy_ders_wrt_moments_[alpha_index_times[i][3]] * val2;


			}

			for (int i = alpha_index_times_count - 1; i >= 0; i--) {
				double val0 = mom_val[alpha_index_times[i][0]];
				double val1 = mom_val[alpha_index_times[i][1]];
				double val2 = alpha_index_times[i][2];

				dloss_dmom[alpha_index_times[i][0]] += dloss_dmom[alpha_index_times[i][3]] * val2*val1;
				dloss_dmom[alpha_index_times[i][1]] += dloss_dmom[alpha_index_times[i][3]] * val2*val0;

			}

		}
              //  for (int i=0; i< C*C; i++){out_grad_accumulator[i]+=0.0; }
		for (int i = 0; i < alpha_index_basic_count; i++)
			for (int j = 0; j < C; j++)
				for (int l = 0; l < K; l++)
				{  for (int m = 0; m < R; m++) {
					out_grad_accumulator[C+2*C*C * K_ +l*(R+C)+m] += linear_coeffs[nbh.my_type]*dloss_dmom[i] * mom_rad_jacobian_(i, j, l*R+m)*
                                        regression_coeffs[C+2*C*C * K_ +0 * (R+C)+ R + type_central] * regression_coeffs[C+2*C*C * K_ +0 * (R+C)+ R + j]  ; //from basic moments
					//out_grad_accumulator[(C * j + type_central) * R * K + k] += 0.5*dloss_dmom[i] * mom_rad_jacobian_(i, j, k);
                                        out_grad_accumulator[C+2*C*C * K_ +0*(R+C)+R+type_central] += linear_coeffs[nbh.my_type] * dloss_dmom[i] * mom_rad_jacobian_(i, j, l*R+m)*regression_coeffs[C+2*C*C * K_ +0 * (R+C)+ R + j]
                                        *regression_coeffs[C+2*C*C * K_ +l * (R+C)+ m]  ;
                                        out_grad_accumulator[C+2*C*C * K_ +0*(R+C)+R+j] += linear_coeffs[nbh.my_type] * dloss_dmom[i] * mom_rad_jacobian_(i, j, l*R+m)*regression_coeffs[C+2*C*C * K_ +0 * (R+C)+ R + type_central]
                                        *regression_coeffs[C+2*C*C * K_ +l * (R+C)+ m]  ;
                                        out_grad_accumulator[C+2*C*C*mu_to_K[l] + type_central * C + j] += linear_coeffs[nbh.my_type] * mom_rad_jacobian_s(i, j, l * R + m) * dloss_dmom[i] * 1.0;
                                        out_grad_accumulator[C+ 2 * C * C * mu_to_K[l] +C*C+type_central*C+j]+= linear_coeffs[nbh.my_type] * mom_rad_jacobian_ss(i, j, l*R+m)*dloss_dmom[i]*1.0;
                                        
                                     //   out_grad_accumulator[l*(R+C)+R] =0;
					if (se_ders_weights != nullptr){																							  //if (wgt_eqtn_forces != 0)		/// CHECK IT CAREFULLY!!!!
					out_grad_accumulator[C+2*C*C * K_ +l*(R+C)+m] += linear_coeffs[nbh.my_type] * site_energy_ders_wrt_moments_[i] * mom_rad_coord_jacobian_(i, j, l*R+m)
                                        *regression_coeffs[C+2*C*C * K_ +0 * (R+C)+ R + type_central] * regression_coeffs[C+2*C*C * K_ +0 * (R+C)+ R + j]    ; //from basic moment ders 
                                        out_grad_accumulator[C+2*C*C * K_ +0*(R+C)+R+type_central] += linear_coeffs[nbh.my_type] * site_energy_ders_wrt_moments_[i] * mom_rad_coord_jacobian_(i, j, l*R+m)
                                        *regression_coeffs[C+2*C*C * K_ +0 * (R+C)+ R + j]*regression_coeffs[C+2*C*C * K_ +l * (R+C)+ m] ;
                                        out_grad_accumulator[C+2*C*C * K_ +0*(R+C)+R+j] += linear_coeffs[nbh.my_type] * site_energy_ders_wrt_moments_[i] * mom_rad_coord_jacobian_(i, j, l*R+m)
                                        *regression_coeffs[C+2*C*C * K_ +0 * (R+C)+ R + type_central ]*regression_coeffs[C+2*C*C * K_ +l * (R+C)+ m] ;
                                        out_grad_accumulator[C+ 2 * C * C * mu_to_K[l] + type_central*C+j]+= linear_coeffs[nbh.my_type] * site_energy_ders_wrt_moments_[i] * mom_rad_coord_jacobian_s(i, j, l*R+m)*1.0;
                                        out_grad_accumulator[C+ 2 * C * C * mu_to_K[l] + C*C+type_central*C+j]+= linear_coeffs[nbh.my_type] * site_energy_ders_wrt_moments_[i] * mom_rad_coord_jacobian_ss(i, j, l*R+m)*1.0;
					//out_grad_accumulator[(C * j + type_central) * R * K + k] += 0.5 * site_energy_ders_wrt_moments_[i] * mom_rad_coord_jacobian_(i, j, k);
		                }
                                }	
                      	     }
                if (shift_){
                out_grad_accumulator[type_central]+= se_weight;}
				out_grad_accumulator[coeff_count + type_central] += se_weight * (  (buff_site_energy_ - regression_coeffs[nbh.my_type]) / linear_coeffs[nbh.my_type]);
				for (int i = 0; i < alpha_scalar_moments; i++)
				{
					out_grad_accumulator[coeff_count + type_central] += dloss_dsenders[alpha_moment_mapping[i]] * linear_mults[i] * linear_coeffs[species_count + i];
					//out_grad_accumulator[coeff_count + type_central] += se_weight * (linear_coeffs[species_count + i] * linear_mults[i] * mom_val[alpha_moment_mapping[i]]);
				}
                
                

		for (int i = 0; i < alpha_scalar_moments; i++)
		{
			out_grad_accumulator[i + coeff_count + species_count] += linear_coeffs[nbh.my_type] * se_weight*mom_val[alpha_moment_mapping[i]]*linear_mults[i];		//from site energy

			if (se_ders_weights != nullptr)
				out_grad_accumulator[i + coeff_count + species_count] += linear_coeffs[nbh.my_type] * dloss_dsenders[alpha_moment_mapping[i]]*linear_mults[i];			//from site forces

		}

	}

}

void MLMTPR::AddPenaltyGrad(const double coeff,													// Must calculate add penalty and optionally (if out_penalty_grad_accumulator != nullptr) its gradient w.r.t. coefficients multiplied by coeff to out_penalty
							double& out_penalty_accumulator,
							Array1D* out_penalty_grad_accumulator)
{
	const int C = species_count;
	const int K = radial_func_count;
	const int R = p_RadialBasis->rb_size;

	if (out_penalty_grad_accumulator != nullptr)
		out_penalty_grad_accumulator->resize(CoeffCount());

	for (int k = 0; k < K; k++)
	{
		double norm = 0;

		for (int i = 0; i < 1; i++)
			for (int j = 0; j < 1; j++)
				for (int l = 0; l < R; l++)
					norm += regression_coeffs[C+2*C*C * K_ +(i*C + j)*K*R + k*(R+C) + l] * regression_coeffs[C+2*C*C * K_ +(i*C + j)*K*R + k*(R+C) + l];


		out_penalty_accumulator += coeff * (norm - 1) * (norm - 1);

		if (out_penalty_grad_accumulator != nullptr)
			for (int i = 0; i < 1; i++)
				for (int j = 0; j < 1; j++)
					for (int l = 0; l < R; l++)
						(*out_penalty_grad_accumulator)[C+2*C*C * K_ +(i*C + j)*K*R + k*(R+C) + l] += coeff * 4 * (norm - 1) * regression_coeffs[C+2*C*C * K_ +(i*C + j)*K*R + k*(R+C) + l];
	}



	for (int k1 = 0; k1 < K; k1++)
		for (int k2 = k1 + 1; k2 < K; k2++)
		{
			double scal = 0;

			for (int i = 0; i < 1; i++)
				for (int j = 0; j < 1; j++)
					for (int l = 0; l < R; l++)
						scal += regression_coeffs[C+2*C*C * K_ +(i*C + j)*K*R + k1*(R+C) + l] * regression_coeffs[C+2*C*C * K_ +(i*C + j)*K*(R+C) + k2*(R+C) + l];

			out_penalty_accumulator += coeff * scal*scal;

			if (out_penalty_grad_accumulator != nullptr)
				for (int i = 0; i < 1; i++)
					for (int j = 0; j < 1; j++)
						for (int l = 0; l < R; l++)
						{
							(*out_penalty_grad_accumulator)[C+2*C*C * K_ +(i*C + j)*K*R + k1*(R+C) + l] += coeff * 2 * scal*regression_coeffs[C+2*C*C * K_ +(i*C + j)*K*R + k2*(R+C) + l];
							(*out_penalty_grad_accumulator)[C+2*C*C * K_ +(i*C + j)*K*R + k2*(R+C) + l] += coeff * 2 * scal*regression_coeffs[C+2*C*C * K_ +(i*C + j)*K*R + k1*(R+C) + l];
						}
		}
}

void MLMTPR::Orthogonalize()
{
	const int C = species_count;
	const int K = radial_func_count;
	const int R = p_RadialBasis->rb_size;

	for (int k = 0; k < K; k++)
	{
//		for (int k2 = 0; k2 < k; k2++) {
//			double norm = 0;
//
//			for (int i = 0; i < 1; i++)
//				for (int j = 0; j < 1; j++)
//					for (int l = 0; l < R; l++)
//						norm += regression_coeffs[  k*(R+C) + l] * regression_coeffs[ k*(R+C) + l];
//
//			double scal = 0;
//			for (int i = 0; i < 1; i++)
//				for (int j = 0; j < 1; j++)
//					for (int l = 0; l < R; l++)
//						scal += regression_coeffs[ k*(R+C) + l] * regression_coeffs[ k2*(R+C) + l];
//			for (int i = 0; i < 1; i++)
//				for (int j = 0; j < 1; j++)
//					for (int l = 0; l < R; l++)
//						regression_coeffs[ k*(R+C) + l] -= regression_coeffs[ k2*(R+C) + l] * scal;
//		}


		{
			double norm = 0;

			for (int i = 0; i < 1; i++)
				for (int j = 0; j < 1; j++)
					for (int l = 0; l < R; l++)
						norm += regression_coeffs[ C+2*C*C * K_ +k*(R+C) + l] * regression_coeffs[ C+2*C*C * K_ +k*(R+C) + l]+1e-30;

			norm = sqrt(norm);
			for (int i = 0; i < 1; i++)
				for (int j = 0; j < 1; j++)
					for (int l = 0; l < R; l++)
						regression_coeffs[C+2*C*C * K_ + k*(R+C) + l] /= norm;
		}
	}
}

void MLMTPR::Perform_scaling()
{
	//#ifdef MLIP_MPI
	//#else
	//#endif

	int C = species_count;
	int K = radial_func_count;
	int R = p_RadialBasis->rb_size;

	std::vector<double> total_max(alpha_scalar_moments);
	total_max.clear();

	for (int j = 0; j < alpha_scalar_moments; j++)
	{

#ifdef MLIP_MPI
		MPI_Reduce(&max_linear[j], &total_max[j], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Bcast(&total_max[j], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		max_linear[j] = total_max[j];
#endif
		regression_coeffs[(R+C)*K + j + species_count] *= linear_mults[j];

		if (max_linear[j] > 0)
			linear_mults[j] = 1e10 / max_linear[j];
		else
			linear_mults[j] = 1e10;

		linear_mults[j] = max(linear_mults[j], 1e-14);
		linear_mults[j] = min(linear_mults[j], 1e14);

		//cout << "mult=" << linear_mults[j] << endl;

		regression_coeffs[R*K + j + species_count] /= linear_mults[j];
		max_linear[j] = 1e-10;
	}
}
