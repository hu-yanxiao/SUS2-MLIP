/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Alexander Shapeev, Evgeny Podryabinkin, Ivan Novikov
 */

#include <sstream>
#include <fstream>
#include <iomanip>
#include <set>
#include <iostream>
#include <algorithm>
#include "configuration.h"


using namespace std;

int MLIP_gcd(int a, int b) {
	return b == 0 ? a : MLIP_gcd(b, a % b);
}

long Configuration::id_cntr = 0;

void Configuration::destroy()
{
	lattice[0][0] = lattice[0][1] = lattice[0][2]
		= lattice[1][0] = lattice[1][1] = lattice[1][2]
		= lattice[2][0] = lattice[2][1] = lattice[2][2] = 0.0;
	types_.clear();
	pos_.clear();
	forces_.clear();
	site_energies_.clear();
	charges_.clear();
	has_energy_ = false;
	has_stresses_ = false;
	features.clear();

	set_new_id();	// new id will be set if configuration is reinitialized
}

bool Configuration::operator==(const Configuration& cfg) const
{
	if (lattice != cfg.lattice) return false;
	if (pos_ != cfg.pos_) return false;

	if (has_energy() != cfg.has_energy()) return false;
	if (has_energy() && energy != cfg.energy) return false;

	if (has_stresses() != cfg.has_stresses()) return false;
	if (has_stresses() && stresses != cfg.stresses) return false;

	if (has_forces() != cfg.has_forces()) return false;
	if (has_forces() && forces_ != cfg.forces_) return false;

	if (has_site_energies() != cfg.has_site_energies()) return false;
	if (has_site_energies() && site_energies_ != cfg.site_energies_) return false;

	if (has_charges() != cfg.has_charges()) return false;
	if (has_charges() && charges_ != cfg.charges_) return false;

	if (types_ != cfg.types_) return false;

	return true;
}

bool Configuration::operator!=(const Configuration& cfg) const
{
	return !operator==(cfg);
}

void Configuration::Deform(const Matrix3& mapping)
{
	lattice *= mapping;

	for (Vector3& coord : pos_)
		coord = coord * mapping;
}

void Configuration::MoveAtomsIntoCell()
{
	const Matrix3 map_to_lattice_basis = lattice.inverse();

	for (int i=0; i<(int)size(); i++)
	{
		Vector3 pos_in_lattice_basis = pos_[i] * map_to_lattice_basis;
		
		for (int a=0; a<3; a++)
			pos_in_lattice_basis[a] -= floor(pos_in_lattice_basis[a]);

		pos_[i] = pos_in_lattice_basis * lattice;
	}
}

void Configuration::CorrectSupercell()
{
	// 1. Make the lattice as orthogonal as possible
	bool proceed=true;
	while (proceed)
	{
		proceed = false;
		for (int i=0; i<3; i++)
		{
			Vector3 l_i = Vector3(lattice[i][0], lattice[i][1], lattice[i][2]);
			for (int j=0; j<3; j++)
				if (i != j)
				{
					Vector3 l_j = Vector3(lattice[j][0], lattice[j][1], lattice[j][2]);
					double orig_l_j_norm2 = l_j.NormSq();
					double new_l_j_norm2 = (l_j + l_i).NormSq();
					for (int itr=1; new_l_j_norm2 < orig_l_j_norm2; itr++)
					{
						l_j += l_i;
						new_l_j_norm2 = (l_j + l_i).NormSq();
						proceed = true;
					}
					new_l_j_norm2 = (l_j - l_i).NormSq();
					for (int itr=1; new_l_j_norm2 < orig_l_j_norm2; itr++)
					{
						l_j -= l_i;
						new_l_j_norm2 = (l_j - l_i).NormSq();
						proceed = true;
					}

					lattice[j][0] = l_j[0];
					lattice[j][1] = l_j[1];
					lattice[j][2] = l_j[2];
				}
		}
	}

	// 2. Update atom positions
	MoveAtomsIntoCell();

	// 3. Make lattice matrix lower triangle
	Matrix3 Q;
	Matrix3 L;
	lattice.transpose().QRdecomp(Q, L);
	Deform(Q);

	// 4. Set zero triangle instead of small numbers
	lattice[0][1] = 0.0;
	lattice[0][2] = 0.0;
	lattice[1][2] = 0.0;

	// 5. Rotate forces if present
	if (has_forces())
		for (Vector3& force : forces_)
			force = force * Q;

	// 6. Rotate stresses if present
	if (has_stresses())
		stresses = Q.transpose() * stresses * Q;
}

void Configuration::ReplicateUnitCell(int nx, int ny, int nz)
{
	int size_rep = size()*nx*ny*nz;
	Matrix3 lattice_rep;
	std::vector<int> types_rep(size_rep);
	std::vector<Vector3> pos_rep(size_rep);
	std::vector<Vector3> force_rep(size_rep);

	has_energy(true);
	has_site_energies(false);
	has_forces(true);
	has_stresses(true);
	has_charges(false);

	energy *= nx*ny*nz;
	stresses *= nx*ny*nz;

	for (int a = 0; a < 3; a++) {
		lattice_rep[0][a] = lattice[0][a] * nx; 
		lattice_rep[1][a] = lattice[1][a] * ny; 
		lattice_rep[2][a] = lattice[2][a] * nz; 
	}

	int idx = 0;

	for (int i = 0; i < size(); i++) {
		for (int ii = 0; ii < nx; ii++) {
			for (int jj = 0; jj < ny; jj++) {
				for (int kk = 0; kk < nz; kk++) {
					for (int l = 0; l < 3; l++) {
						pos_rep[idx][l] = pos_[i][l] + ii*lattice[0][l] + jj*lattice[1][l] + kk*lattice[2][l]; 
						force_rep[idx][l] = forces_[i][l];
					}
					types_rep[idx] = types_[i];
					idx++;
				}
			}
		}
	}

	resize(idx);

	lattice = lattice_rep;
	for (int i = 0; i < idx; i++) {
		types_[i] = types_rep[i];
		for (int l = 0; l < 3; l++) {
			pos_[i][l] = pos_rep[i][l];	
			forces_[i][l] = force_rep[i][l];
		}
	}

}

//! Replicates the configuration by making the unit cell L -> AL
void Configuration::ReplicateUnitCell(templateMatrix3<int> A)
{
	has_energy(false);
	has_site_energies(false);
	has_forces(false);
	has_stresses(false);
	has_charges(false);

	// calculating detA and making sure detA>0
	int detA = A.det();
	if (detA < 0) {
		detA *= -1; A[0][0] *= -1; A[1][0] *= -1; A[2][0] *= -1;
		lattice[0][0] *= -1; lattice[0][1] *= -1; lattice[0][2] *= -1;
	}

	vector<templateVector3<int>> replicate_vectors;
	replicate_vectors.reserve(detA);

	// We need to find all (i,j,k) such that 0 <= (i,j,k)*A^{-1} < 1
	// and add those (i,j,k) into replicate_vectors.
	// Doing a silly loop

	templateVector3<int> lmn;
	for (lmn[0] = 0; lmn[0] < detA; lmn[0]++)
		for (lmn[1] = 0; lmn[1] < detA; lmn[1]++)
			for (lmn[2] = 0; lmn[2] < detA; lmn[2]++) {
				templateVector3<int> ijk = lmn*A;
				if (ijk[0] % detA == 0 && ijk[1] % detA == 0 && ijk[2] % detA == 0)
					replicate_vectors.push_back(ijk / detA);
			}
	if (replicate_vectors.size() != detA)
		ERROR("replicate_vectors.size() != detA; something went wrong");

		// we use that replicate_vectors[0] == (0,0,0)

	int old_size;
	old_size = (int)pos_.size();
	pos_.resize(old_size * detA);
	for (int j = 1; j < detA; j++)
		for (int i = 0; i < old_size; i++)
			for (int a = 0; a < 3; a++) {
				pos_[i + old_size*j][a] = pos_[i][a]
					+ replicate_vectors[j][0] * lattice[0][a]
					+ replicate_vectors[j][1] * lattice[1][a]
					+ replicate_vectors[j][2] * lattice[2][a];
			}

	types_.resize(old_size * detA);
	for (int j = 1; j < detA; j++)
		for (int i = 0; i < old_size; i++)
			types_[i + j*old_size] = types_[i];

	Matrix3 A_double(
		(double)A[0][0], (double)A[0][1], (double)A[0][2],
		(double)A[1][0], (double)A[1][1], (double)A[1][2],
		(double)A[2][0], (double)A[2][1], (double)A[2][2]);
	lattice = A_double*lattice;

	MoveAtomsIntoCell();
}

Configuration::Configuration() :
	has_energy_(false),
	has_stresses_(false),
	lattice(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
	stresses(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
{
	set_new_id();
}

Configuration::Configuration(	CrystalSystemType crystal_system, 
								double lattice_parameter, 
								int replication_times)
								// TODO: basic_lattice -> deformation mapping
{
	set_new_id();

	if (crystal_system == CrystalSystemType::SC)
	{
		Matrix3 basic_lattice = Matrix3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0); 
		basic_lattice *= lattice_parameter;
		lattice = basic_lattice * replication_times;

		for (int i = 0; i < replication_times; i++)
			for (int j = 0; j < replication_times; j++)
				for (int k = 0; k < replication_times; k++)
				{
					pos_.emplace_back();

					pos_.back()[0] = i * basic_lattice[0][0] + j * basic_lattice[1][0] + k * basic_lattice[2][0];
					pos_.back()[1] = i * basic_lattice[0][1] + j * basic_lattice[1][1] + k * basic_lattice[2][1];
					pos_.back()[2] = i * basic_lattice[0][2] + j * basic_lattice[1][2] + k * basic_lattice[2][2];
				}
	}
	else if (crystal_system == CrystalSystemType::BCC)
	{
		Matrix3 basic_lattice = Matrix3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
		basic_lattice *= lattice_parameter;
		lattice = basic_lattice * replication_times;

		for (int i = 0; i < replication_times; i++)
			for (int j = 0; j < replication_times; j++)
				for (int k = 0; k < replication_times; k++)
				{
					pos_.emplace_back();

					pos_.back()[0] = i * basic_lattice[0][0] + j * basic_lattice[1][0] + k * basic_lattice[2][0];
					pos_.back()[1] = i * basic_lattice[0][1] + j * basic_lattice[1][1] + k * basic_lattice[2][1];
					pos_.back()[2] = i * basic_lattice[0][2] + j * basic_lattice[1][2] + k * basic_lattice[2][2];

					pos_.emplace_back();

					pos_.back()[0] = 0.5 * (basic_lattice[0][0] + basic_lattice[1][0] + basic_lattice[2][0]);
					pos_.back()[1] = 0.5 * (basic_lattice[0][1] + basic_lattice[1][1] + basic_lattice[2][1]);
					pos_.back()[2] = 0.5 * (basic_lattice[0][2] + basic_lattice[1][2] + basic_lattice[2][2]);

					pos_.back()[0] += i * basic_lattice[0][0] + j * basic_lattice[1][0] + k * basic_lattice[2][0];
					pos_.back()[1] += i * basic_lattice[0][1] + j * basic_lattice[1][1] + k * basic_lattice[2][1];
					pos_.back()[2] += i * basic_lattice[0][2] + j * basic_lattice[1][2] + k * basic_lattice[2][2];
				}
	}
	else if (crystal_system == CrystalSystemType::FCC)
	{
		Matrix3 basic_lattice = Matrix3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
		basic_lattice *= lattice_parameter;
		lattice = basic_lattice * replication_times;

		for (int i = 0; i < replication_times; i++)
			for (int j = 0; j < replication_times; j++)
				for (int k = 0; k < replication_times; k++)
				{
					pos_.emplace_back();

					pos_.back()[0] = i * basic_lattice[0][0] + j * basic_lattice[1][0] + k * basic_lattice[2][0];
					pos_.back()[1] = i * basic_lattice[0][1] + j * basic_lattice[1][1] + k * basic_lattice[2][1];
					pos_.back()[2] = i * basic_lattice[0][2] + j * basic_lattice[1][2] + k * basic_lattice[2][2];

					pos_.emplace_back();

					pos_.back()[0] = 0.5 * (0.0					+ basic_lattice[1][0] + basic_lattice[2][0]);
					pos_.back()[1] = 0.5 * (basic_lattice[0][1] + basic_lattice[1][1] + basic_lattice[2][1]);
					pos_.back()[2] = 0.5 * (basic_lattice[0][2] + basic_lattice[1][2] + basic_lattice[2][2]);

					pos_.back()[0] += i * basic_lattice[0][0] + j * basic_lattice[1][0] + k * basic_lattice[2][0];
					pos_.back()[1] += i * basic_lattice[0][1] + j * basic_lattice[1][1] + k * basic_lattice[2][1];
					pos_.back()[2] += i * basic_lattice[0][2] + j * basic_lattice[1][2] + k * basic_lattice[2][2];

					pos_.emplace_back();

					pos_.back()[0] = 0.5 * (basic_lattice[0][0] + basic_lattice[1][0] + basic_lattice[2][0]);
					pos_.back()[1] = 0.5 * (basic_lattice[0][1] + 0.0				  + basic_lattice[2][1]);
					pos_.back()[2] = 0.5 * (basic_lattice[0][2] + basic_lattice[1][2] + basic_lattice[2][2]);

					pos_.back()[0] += i * basic_lattice[0][0] + j * basic_lattice[1][0] + k * basic_lattice[2][0];
					pos_.back()[1] += i * basic_lattice[0][1] + j * basic_lattice[1][1] + k * basic_lattice[2][1];
					pos_.back()[2] += i * basic_lattice[0][2] + j * basic_lattice[1][2] + k * basic_lattice[2][2];

					pos_.emplace_back();

					pos_.back()[0] = 0.5 * (basic_lattice[0][0] + basic_lattice[1][0] + basic_lattice[2][0]);
					pos_.back()[1] = 0.5 * (basic_lattice[0][1] + basic_lattice[1][1] + basic_lattice[2][1]);
					pos_.back()[2] = 0.5 * (basic_lattice[0][2]	+ basic_lattice[1][2] + 0.0);

					pos_.back()[0] += i * basic_lattice[0][0] + j * basic_lattice[1][0] + k * basic_lattice[2][0];
					pos_.back()[1] += i * basic_lattice[0][1] + j * basic_lattice[1][1] + k * basic_lattice[2][1];
					pos_.back()[2] += i * basic_lattice[0][2] + j * basic_lattice[1][2] + k * basic_lattice[2][2];
				}
	}
	else if (crystal_system == CrystalSystemType::HCP)
	{
		Matrix3 basic_lattice = Matrix3(1.0, 0.0, 0.0, -0.5, sqrt(3)/2.0, 0.0, 0.0, 0.0, 2*sqrt(2.0/3.0));
		basic_lattice *= lattice_parameter;
		lattice = basic_lattice * replication_times;

		for (int i = 0; i < replication_times; i++)
			for (int j = 0; j < replication_times; j++)
				for (int k = 0; k < replication_times; k++)
				{
					pos_.emplace_back();

					pos_.back()[0] = i * basic_lattice[0][0] + j * basic_lattice[1][0] + k * basic_lattice[2][0];
					pos_.back()[1] = i * basic_lattice[0][1] + j * basic_lattice[1][1] + k * basic_lattice[2][1];
					pos_.back()[2] = i * basic_lattice[0][2] + j * basic_lattice[1][2] + k * basic_lattice[2][2];

					pos_.emplace_back();

					pos_.back()[0] = 2.0/3.0 * basic_lattice[0][0] + 1.0/3.0 * basic_lattice[1][0] + 0.5 * basic_lattice[2][0];
					pos_.back()[1] = 2.0/3.0 * basic_lattice[0][1] + 1.0/3.0 * basic_lattice[1][1] + 0.5 * basic_lattice[2][1];
					pos_.back()[2] = 2.0/3.0 * basic_lattice[0][2] + 1.0/3.0 * basic_lattice[1][2] + 0.5 * basic_lattice[2][2];

					pos_.back()[0] += i * basic_lattice[0][0] + j * basic_lattice[1][0] + k * basic_lattice[2][0];
					pos_.back()[1] += i * basic_lattice[0][1] + j * basic_lattice[1][1] + k * basic_lattice[2][1];
					pos_.back()[2] += i * basic_lattice[0][2] + j * basic_lattice[1][2] + k * basic_lattice[2][2];
				}
	}
	types_.resize(size());

	has_energy_ = has_stresses_ = false;
}

Configuration::Configuration(	CrystalSystemType crystal_system, 
								double basic_lattice_parameter1, 
								double basic_lattice_parameter2, 
								double basic_lattice_parameter3, 
								int replication_times)
	: Configuration(crystal_system, 1, replication_times)
{
	Matrix3 stretch(	basic_lattice_parameter1, 0, 0,
						0, basic_lattice_parameter2, 0,
						0, 0, basic_lattice_parameter3);
	Deform(stretch);
}

Configuration::~Configuration()
{
//	destroy();
}

bool Configuration::Load(std::ifstream& ifs)
{
	if (!ifs.is_open()) ERROR("Input stream is not open");
	if (ifs.fail()) ERROR("Input stream is open but not good");
	if (ifs.eof()) return false;

	char tag = 0;

	streampos cfg_beg_pos = ifs.tellg();
	if (ifs.fail()) ERROR("tellg failed");
	ifs >> tag;
	if (ifs.eof()) return false;
	if (ifs.fail()) ERROR("cannot read even one symbol!");
	ifs.seekg(cfg_beg_pos);
	if (ifs.fail()) ERROR("seekg failed");

	// checks if this is the binary file format
	if (tag == 'n')
		return LoadBin(ifs);

	string tmp_str;

	ifs >> tmp_str;

	// checks if this is the old file format
	if (tmp_str == "BEGIN") {
		ifs.seekg(cfg_beg_pos);
		if (ifs.fail()) ERROR("seekg failed");
		return LoadOld(ifs);
	}

	// checks if this is the new file format
	if (tmp_str != "BEGIN_CFG") ERROR("unknown token " + tmp_str);

	destroy();

	has_energy_ = has_stresses_ = false;

	// these are internal variables, if they remain false then this is an error
	bool has_atom_pos = false;
	int _size = -1;
	bool is_stress_virial = false;
	bool is_stress_negative = false;
	bool frac_pos = false;

	while (!ifs.eof()) {
		ifs >> tmp_str;

		if (!ifs.good() && tmp_str != "END_CFG")
			ERROR("Unable to read field name");

		if (tmp_str == "Size") {
			int new_size;
			ifs >> new_size;
			if (_size != -1 && _size != new_size)
				ERROR("Size given, but does not match earlier size");
			_size = new_size;
			if (!ifs.good()) ERROR("Unable to read configuration size");
			if (_size < 0)	ERROR("Incorrect configuration size");
			resize(_size);
		}
		else if (tmp_str == "Supercell" || tmp_str == "SuperCell") {
			ifs >> lattice[0][0] >> lattice[0][1] >> lattice[0][2]
				>> lattice[1][0] >> lattice[1][1] >> lattice[1][2]
				>> lattice[2][0] >> lattice[2][1] >> lattice[2][2];
			ifs.clear();
		} else if (tmp_str == "AtomData:" || tmp_str == "Atomic_data:") {
			if (_size == 0)
				Warning("Configuration with no atoms has been read");

			if (_size == -1)
				ERROR("Reading AtomData without Size given is not implemented.");

			enum class DataType { ID, TYPE, XC, YC, ZC, XD, YD, ZD, FX, FY, FZ, SITE_EN, CHARGE };
			vector<DataType> datamap;

			int control_summ_pos[3] = { 0,0,0 };
			int control_summ_frc[3] = { 0,0,0 };

			// in ids, '.first' is the id that is read and '.second' is the line number
			vector<int> ids;

			getline(ifs, tmp_str);
			istringstream keywords(tmp_str);
			do {
				string keyword;
				keywords >> keyword;

				if (keyword == "id") {
					ids.resize(_size);
					datamap.push_back(DataType::ID);
				} else if (keyword == "type") {
					types_.resize(_size);
					datamap.push_back(DataType::TYPE);
				} else if (keyword == "cartes_x") {
					pos_.resize(_size);
					datamap.push_back(DataType::XC);
					control_summ_pos[0]++;
				} else if (keyword == "cartes_y") {
					pos_.resize(_size);
					datamap.push_back(DataType::YC);
					control_summ_pos[1]++;
				} else if (keyword == "cartes_z") {
					pos_.resize(_size);
					datamap.push_back(DataType::ZC);
					control_summ_pos[2]++;
				} else if (keyword == "direct_x") {
					pos_.resize(_size);
					datamap.push_back(DataType::XD);
					control_summ_pos[0]--;
				} else if (keyword == "direct_y") {
					pos_.resize(_size);
					datamap.push_back(DataType::YD);
					control_summ_pos[1]--;
				} else if (keyword == "direct_z") {
					pos_.resize(_size);
					datamap.push_back(DataType::ZD);
					control_summ_pos[2]--;
				} else if (keyword == "site_en") {
					has_site_energies(true);
					datamap.push_back(DataType::SITE_EN);
				} else if (keyword == "fx") {
					has_forces(true);
					datamap.push_back(DataType::FX);
					control_summ_frc[0]++;
				} else if (keyword == "fy") {
					has_forces(true);
					datamap.push_back(DataType::FY);
					control_summ_frc[1]++;
				} else if (keyword == "fz") {
					has_forces(true);
					datamap.push_back(DataType::FZ);
					control_summ_frc[2]++;
				} else if (keyword == "charge") {
					has_charges(true);
					datamap.push_back(DataType::CHARGE);
				} else if (keyword == "")
					break;
				else
					ERROR("Unknown AtomData field");
			} while (keywords);

			// control_summ_pos should either be all 1 or -1 or all 0
			if (control_summ_pos[0] != control_summ_pos[1] ||
				control_summ_pos[0] != control_summ_pos[2] ||
				abs(control_summ_pos[0]) > 1)
				ERROR("Configuration reading eror. Wrong atom position information");
			if (control_summ_frc[0] != control_summ_frc[1] ||
				control_summ_frc[0] != control_summ_frc[2] ||
				control_summ_frc[0] > 1)
				ERROR("Configuration reading eror. Wrong force information");

			has_atom_pos = (abs(control_summ_pos[0]) == 1);
			frac_pos = (control_summ_pos[0] == -1);

			if (_size > 0) {
				for (int i = 0; i < size(); i++)
					for (int j = 0; j < (int)datamap.size(); j++)
						if (datamap[j] == DataType::TYPE)
							ifs >> types_[i];
						else if (datamap[j] == DataType::ID)
							ifs >> ids[i];
						else if (datamap[j] == DataType::XC)
							ifs >> pos_[i][0];
						else if (datamap[j] == DataType::YC)
							ifs >> pos_[i][1];
						else if (datamap[j] == DataType::ZC)
							ifs >> pos_[i][2];
						else if (datamap[j] == DataType::XD)
							ifs >> pos_[i][0];
						else if (datamap[j] == DataType::YD)
							ifs >> pos_[i][1];
						else if (datamap[j] == DataType::ZD)
							ifs >> pos_[i][2];
						else if (datamap[j] == DataType::FX)
							ifs >> forces_[i][0];
						else if (datamap[j] == DataType::FY)
							ifs >> forces_[i][1];
						else if (datamap[j] == DataType::FZ)
							ifs >> forces_[i][2];
						else if (datamap[j] == DataType::SITE_EN)
							ifs >> site_energies_[i];
						else if (datamap[j] == DataType::CHARGE)
							ifs >> charges_[i];
						if (!ifs.good())
							ERROR("Unable to read AtomData. Wrong Size specified?");
			} else {
				// HERE GOES THE CODE OF READING ATOMDATA WITHOUT SIZE GIVEN
			}

			// if ids are given (i.e., their size != 0) then check that they are 1, 2, 3,...
			for (int i = 0; i < (int)ids.size(); i++)
				if (ids[i] != i + 1)
					ERROR("Invalid file format: wrong order of id\'s");

			if (!has_atom_pos) ERROR("Invalid file format: no atom positions");

			// if types are not given, then set all to zero
			if (types_.size() == 0) {
				types_.resize(size());
				FillWithZero(types_);
			}
			std::for_each(types_.begin(), types_.end(), [&](int elem) {
				type_num[elem]=0;
				unique_elems.emplace(elem);
				});

			std::for_each(types_.begin(), types_.end(), [&](int elem) {
				type_num[elem]++;
				unique_elems.emplace(elem);
				});
		

		} else if (tmp_str == "Energy") {
			ifs >> energy;
			if (!ifs.good())
				ERROR("Configuration reading eror. Reading energy failed");
			has_energy_ = true;
		}
		else if (tmp_str == "Stress:" || tmp_str == "Virial:" || tmp_str == "PlusStress:") {
			if (tmp_str == "Virial:") is_stress_virial = true;
			if (tmp_str == "Stress:") {
				is_stress_negative = true;
#ifndef MLIP_NO_WRONG_STRESS_ERROR
				ERROR( "\n"
				"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
				"!!!          ERROR         ERROR        ERROR         !!!\n"
				"!!!      Read a configuration with (negative) Stress. !!!\n"
                                "!!!      This feature will be removed soon!           !!!\n"
                                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
#else
				Warning( "\n"
				"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
				"!!!         WARNING       WARNING      WARNING        !!!\n"
				"!!!      Read a configuration with (negative) Stress. !!!\n"
                                "!!!      This feature will be removed soon!           !!!\n"
                                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
#endif
			}

			vector<double*> datamap;

			getline(ifs, tmp_str);
			istringstream keywords(tmp_str);

			do {
				string keyword;
				keywords >> keyword;
				if (keyword == "xx")
					datamap.push_back(&stresses[0][0]);
				else if (keyword == "yy")
					datamap.push_back(&stresses[1][1]);
				else if (keyword == "zz")
					datamap.push_back(&stresses[2][2]);
				else if (keyword == "xy")
					datamap.push_back(&stresses[0][1]);
				else if (keyword == "xz")
					datamap.push_back(&stresses[0][2]);
				else if (keyword == "yz")
					datamap.push_back(&stresses[1][2]);
			} while (keywords);

			for (int j = 0; j < (int)datamap.size(); j++)
				ifs >> *datamap[j];
			if (!ifs.good())
				ERROR("Configuration reading error. Failed to read stresses");

			stresses[1][0] = stresses[0][1];
			stresses[2][1] = stresses[1][2];
			stresses[2][0] = stresses[0][2];
			has_stresses_ = true;
		}
		else if (tmp_str == "Features:") {
			getline(ifs, tmp_str);
			istringstream keywords(tmp_str);

			do {
				string keyword;
				keywords >> keyword;
				if (keyword != "")
					ifs >> features[keyword];
			} while (keywords);

			if (!ifs.good())
				ERROR("Configuration reading error (Broken features)");
		} else if (tmp_str == "Feature") {
			string key, value;
			ifs >> key;
			ifs.ignore(1);
			key = mlip_string_unescape(key);
			getline(ifs, value);
			if (!ifs.good())
				ERROR("Configuration reading error (Broken feature)");

			if (features.count(key) == 0)
				features[key] = value;
			else // multiline feature value
				features[key] += '\n' + value;
		} else if (tmp_str == "END_CFG")
			break;
		else
			ERROR("Configuration reading error. Unknown field \"" + tmp_str + "\"");
	}

	if (is_stress_virial)
		stresses *= 1.0 / fabs(lattice.det());
	if (is_stress_negative)
		stresses *= -1;
	if (frac_pos)
		if (lattice.det() == 0.0)
			ERROR("Non-periodic unitcell and fractional coordinates detected (incompatible case)");
		else
			for (int i=0; i<size(); i++)
				pos_[i] = pos_[i] * lattice;

	if (!has_atom_pos && size() == 0)
		return false;
	else
		return true;
}

bool Configuration::LoadOld(ifstream& ifs)
{
	destroy();

	if (!ifs.is_open())
		ERROR("Input stream not open");

	has_energy_ = has_stresses_ = false;

	bool _has_lattice = false;
	bool _has_atom_pos = false;
	bool _has_atom_types = false;
	int _size = 0;

	string tmp_str = "";

	ifs >> tmp_str;
	if (tmp_str != "BEGIN")
		return false;

	while (!ifs.eof())
	{
		ifs >> tmp_str;
		if (ifs.fail()) ERROR("Unable to read field name");

		if (tmp_str == "Size")
		{
			ifs >> _size;
		}
		else if (tmp_str == "MinusStress")
		{
			ifs >> stresses[0][0]
				>> stresses[1][1]
				>> stresses[2][2]
				>> stresses[0][1]
				>> stresses[1][2]
				>> stresses[0][2];

			stresses[1][0] = stresses[0][1];
			stresses[2][1] = stresses[1][2];
			stresses[2][0] = stresses[0][2];

			has_stresses_ = true;
		}
		else if (tmp_str == "MinusStress9")
		{
			ifs >> stresses[0][0]
				>> stresses[0][1]
				>> stresses[0][2]
				>> stresses[1][0]
				>> stresses[1][1]
				>> stresses[1][2]
				>> stresses[2][0]
				>> stresses[2][1]
				>> stresses[2][2];

			has_stresses_ = true;
		}
		else if ((tmp_str == "VirialStress") || (tmp_str == "Stress"))
		{
			ifs >> stresses[0][0]
				>> stresses[1][1]
				>> stresses[2][2]
				>> stresses[0][1]
				>> stresses[1][2]
				>> stresses[0][2];

			stresses[1][0] = stresses[0][1];
			stresses[2][1] = stresses[1][2];
			stresses[2][0] = stresses[0][2];

			stresses *= -1;

			has_stresses_ = true;
		}
		else if (tmp_str == "Lattice")
		{
			ifs >> lattice[0][0] >> lattice[0][1] >> lattice[0][2]
				>> lattice[1][0] >> lattice[1][1] >> lattice[1][2]
				>> lattice[2][0] >> lattice[2][1] >> lattice[2][2];
			_has_lattice = true;
		}
		else if ((tmp_str == "AtmData") || (tmp_str == "AtomData"))
		{
			double x, y, z;
			// loop while we keep reading numbers
			for (ifs >> x; ifs.good(); ifs >> x) {
				ifs >> y >> z;
				if (ifs.good()) {
					pos_.emplace_back();
					pos_.back()[0] = x;
					pos_.back()[1] = y;
					pos_.back()[2] = z;
				}
				else ERROR("Error reading coordinates");

				ifs >> x >> y >> z;
				if (ifs.good()) {
					forces_.emplace_back();
					forces_.back()[0] = x;
					forces_.back()[1] = y;
					forces_.back()[2] = z;
				}
				else ERROR("Error reading forces");
			}
			ifs.clear();
			_has_atom_pos = true;
		}
		else if (tmp_str == "AtomPos")
		{
			double x, y, z;
			// loop while we keep reading numbers
			for (ifs >> x; ifs.good(); ifs >> x) {
				ifs >> y >> z;
				if (ifs.good()) {
					pos_.emplace_back();
					pos_.back()[0] = x;
					pos_.back()[1] = y;
					pos_.back()[2] = z;
				}
				else ERROR("Error reading coordinates");
			}
			ifs.clear();
			_has_atom_pos = true;
		}
		else if (tmp_str == "AtomTypes") 
		{
			int type;
			// loop while we keep reading numbers
			for (ifs >> type; ifs.good(); ifs >> type)
				types_.push_back(type);
			ifs.clear();
			_has_atom_types = true;
		} 
		else if (tmp_str == "SiteEnergies")
		{
			double foo;
			// loop while we keep reading numbers
			ifs >> foo;
			while (ifs.good())
			{
				site_energies_.push_back(foo);
				ifs >> foo;
			}
			ifs.clear();
		}
		else if (tmp_str == "AtomForces")
		{
			double x, y, z;
			// loop while we keep reading numbers
			for (ifs >> x; ifs.good(); ifs >> x) {
				ifs >> y >> z;
				if (ifs.good()) {
					forces_.emplace_back();
					forces_.back()[0] = x;
					forces_.back()[1] = y;
					forces_.back()[2] = z;
				}
				else ERROR("Error reading forces");
			}
			ifs.clear();
		}
		else if (tmp_str == "Charges")
		{
			double foo;
			// loop while we keep reading numbers
			ifs >> foo;
			while (!ifs.fail())
			{
				charges_.push_back(foo);
				ifs >> foo;
			}
			ifs.clear();
		}
		else if (tmp_str == "Energy")
		{
			ifs >> energy;
			if (!ifs.good()) ERROR("Error reading energy");
			has_energy_ = true;
		}
		else if (tmp_str == "Feature")
		{
			std::string name, value;
			ifs >> name >> value;
			features[name] = value;
		}
		else if (tmp_str == "END")
			break;
		else 
			ERROR("Unknown configuration field \"" + tmp_str + "\"");

		if (!ifs.good()) ERROR("error reading");
	}
	
	if (!(_has_atom_pos && _has_lattice)) ERROR("No lattice or atom positions");
	if (_size == 0)
		_size = (int)pos_.size();
	if (_has_atom_types && types_.size() != _size) ERROR("Problem with configuration size");
	if (_size != (int)pos_.size()) ERROR("Problem with configuration size");
	if (has_forces() && (_size != (int)forces_.size())) ERROR("Problem with configuration size");
	if (has_site_energies() && (_size != (int)site_energies_.size())) ERROR("Problem with configuration size");
	if (has_charges() && (_size != (int)charges_.size())) ERROR("Problem with configuration size");

	if (!_has_atom_types) types_.resize(_size, 0);
	_has_atom_types = true;

	std::for_each(types_.begin(), types_.end(), [&](int elem) {
		type_num[elem]++;
		unique_elems.emplace(elem);
		});

	return true;
}

bool Configuration::LoadBin(std::ifstream & ifs)
{
	destroy();

	if (!ifs.is_open())
		ERROR("Input stream not open");

	char tag = 0;
	has_energy_ = has_stresses_ = false;
	bool has_pos = false;
	bool has_lattice = false;
	bool has_types = false;
	int _size;

	ifs >> tag;
	if (tag != 'n') {
		if (ifs.eof()) return false; // empty configuration
		ERROR("Corrupted binary configuration file");
	}
	else
		ifs.read((char*) &_size, sizeof(_size));

	if (_size < 0)
		ERROR("Corrupted binary configuration file");

	while (!ifs.eof())
	{
		ifs >> tag;

		if (tag == 'l') {
			for (int a = 0; a < 3; a++)
				for (int b = 0; b < 3; b++)
					ifs.read((char*)&lattice[a][b], sizeof(lattice[0][0]));
			has_lattice = true;
		}
		else if (tag == 'p') {
			pos_.resize(_size);
			ifs.read((char*)&pos_[0], _size*sizeof(pos_[0]));
			has_pos = true;
		}
		else if (tag == 't') {
			types_.resize(_size);
			ifs.read((char*)&types_[0], _size*sizeof(types_[0]));
			has_types = true;
		} 
		else if (tag == 'f') {
			forces_.resize(_size);
			ifs.read((char*)&forces_[0], _size*sizeof(forces_[0]));
		} 
		else if (tag == 's') {
			ifs.read((char*)&stresses[0][0], sizeof(stresses[0][0]));
			ifs.read((char*)&stresses[1][1], sizeof(stresses[0][0]));
			ifs.read((char*)&stresses[2][2], sizeof(stresses[0][0]));
			ifs.read((char*)&stresses[0][1], sizeof(stresses[0][0]));
			ifs.read((char*)&stresses[1][2], sizeof(stresses[0][0]));
			ifs.read((char*)&stresses[0][2], sizeof(stresses[0][0]));
			stresses[1][0] = stresses[0][1];
			stresses[2][1] = stresses[1][2];
			stresses[2][0] = stresses[0][2];
			has_stresses_ = true;
		}
		else if (tag == 'v') {
			site_energies_.resize(_size);
			ifs.read((char*)&site_energies_[0], _size*sizeof(site_energies_[0]));
		}
		else if (tag == 'c') {
			charges_.resize(_size);
			ifs.read((char*)&charges_[0], _size*sizeof(charges_[0]));
		}
		else if (tag == 'e') {
			ifs.read((char*)&energy, sizeof(energy));
			has_energy_ = true;
		} 
		else if (tag == 'F') {
			std::string name, value;
			int l;
			ifs.read((char*)&l, sizeof(l));
			name.resize(l);
			ifs.read(&name[0], l * sizeof(char));
			ifs.read((char*)&l, sizeof(l));
			value.resize(l);
			ifs.read(&value[0], l * sizeof(char));
			features[name] = value;
		} 
		else if (tag == '#')
			break;
		else
			ERROR("Unknown tag \'" + tag + "\'");
	}

	if (!has_types) types_.resize(_size, 0);
	has_types = true;

	if (!(has_lattice && has_pos)) ERROR("Corrupted binary configuration file");

	return true;
}

// Configuration file format definitions
#define LAT_FORMATTING right << setw(LAT_WIDTH) << setprecision(LAT_PRECISION)
#define IND_FORMATTING right << setw(IND_WIDTH) << setprecision(IND_PRECISION)
#define TYP_FORMATTING right << setw(TYP_WIDTH) << setprecision(TYP_PRECISION)
#define POS_FORMATTING right << setw(POS_WIDTH) << setprecision(POS_PRECISION)
#define SEN_FORMATTING right << setw(SEN_WIDTH) << setprecision(SEN_PRECISION)
#define FRC_FORMATTING right << setw(FRC_WIDTH) << setprecision(FRC_PRECISION)
#define CHG_FORMATTING right << setw(CHG_WIDTH) << setprecision(CHG_PRECISION)
#define ENE_FORMATTING right << setw(ENE_WIDTH) << setprecision(ENE_PRECISION)
#define STR_FORMATTING right << setw(STR_WIDTH) << setprecision(STR_PRECISION)

void Configuration::Save(ofstream& ofs, unsigned int flags) const
{
	int LAT_WIDTH; int LAT_PRECISION;
	int IND_WIDTH; int IND_PRECISION;
	int TYP_WIDTH; int TYP_PRECISION;
	int POS_WIDTH; int POS_PRECISION;
	int SEN_WIDTH; int SEN_PRECISION;
	int FRC_WIDTH; int FRC_PRECISION;
	int CHG_WIDTH; int CHG_PRECISION;
	int ENE_WIDTH; int ENE_PRECISION;
	int STR_WIDTH; int STR_PRECISION;

	unsigned int save_binary = flags & SAVE_BINARY;
	unsigned int without_loss = flags & SAVE_NO_LOSS;
#ifdef MLIP_LOSSLESS_CFG
	without_loss = SAVE_NO_LOSS;
#endif
	unsigned int direct_atom_coord = flags & SAVE_DIRECT_COORDS;

	if (!ofs.is_open())
		ERROR("Output stream not open");

	if (save_binary) {
		SaveBin(ofs);
		return;
	}

	ofs.precision(16);
	if (without_loss) {
		ofs << scientific;
		LAT_WIDTH = 1; LAT_PRECISION = 16;
		IND_WIDTH = 1; IND_PRECISION = 0;
		TYP_WIDTH = 1; TYP_PRECISION = 0;
		POS_WIDTH = 1; POS_PRECISION = 16;
		SEN_WIDTH = 1; SEN_PRECISION = 16;
		FRC_WIDTH = 1; FRC_PRECISION = 16;
		CHG_WIDTH = 1; CHG_PRECISION = 16;
		ENE_WIDTH = 1; ENE_PRECISION = 16;
		STR_WIDTH = 1; STR_PRECISION = 16;
	}
	else {
		ofs << fixed;
		LAT_WIDTH = 13; LAT_PRECISION =  6;
		IND_WIDTH = 10; IND_PRECISION =  0;
		TYP_WIDTH =  4; TYP_PRECISION =  0;
		POS_WIDTH = 13; POS_PRECISION =  6;
		SEN_WIDTH = 17; SEN_PRECISION = 12;
		FRC_WIDTH = 11; FRC_PRECISION =  6;
		CHG_WIDTH =  8; CHG_PRECISION =  4;
		ENE_WIDTH = 20; ENE_PRECISION = 12;
		STR_WIDTH = 11; STR_PRECISION =  5;
	}

	int _size = size();

	ofs << "BEGIN_CFG\n";

	ofs << " Size\n    " << _size << '\n';
	
	{
		int count = 0;
		for (int a = 0; a < 3; a++)
			if (lattice[a][0] != 0 || lattice[a][1] != 0 || lattice[a][2] != 0) count = a+1;
		
		if (count != 0) {
			ofs << " Supercell\n";
			for (int a = 0; a < count; a++)
				ofs << "    " << LAT_FORMATTING << lattice[a][0]
				<< ' ' << LAT_FORMATTING << lattice[a][1]
				<< ' ' << LAT_FORMATTING << lattice[a][2] << '\n';
		}
	}

	{
		ofs << " AtomData:  id type";
		
			if (direct_atom_coord)
				ofs << "       direct_x      direct_y      direct_z";
			else	
				ofs << "       cartes_x      cartes_y      cartes_z";
		if (has_site_energies())	ofs << "    site_en";
		if (has_forces())			ofs << "           fx          fy          fz";
		if (has_charges())			ofs << "      charge";
		ofs << '\n';

		for (int i=0; i<size(); i++)
		{
			ofs << "    " << IND_FORMATTING << i+1;
			ofs << ' ' << TYP_FORMATTING << types_[i];
				if (!direct_atom_coord)
					ofs << "  " << POS_FORMATTING << pos_[i][0]
						<< ' ' << POS_FORMATTING << pos_[i][1]
						<< ' ' << POS_FORMATTING << pos_[i][2];
				else
				{
					Vector3 dpos = direct_pos(i);
					ofs << "  " << POS_FORMATTING << dpos[0]
						<< ' ' << POS_FORMATTING << dpos[1]
						<< ' ' << POS_FORMATTING << dpos[2];
				}
			if (has_site_energies())	
				ofs << "  " << SEN_FORMATTING << cal_se0[i];
			if (has_forces())	
				ofs << "  " << FRC_FORMATTING << forces_[i][0]
					<< ' ' << FRC_FORMATTING << forces_[i][1]
					<< ' ' << FRC_FORMATTING << forces_[i][2];
			if (has_charges())	
				ofs << "    " << CHG_FORMATTING << charges(i);
			ofs << '\n';
		}
	}

	if (has_energy()) 
		ofs << " Energy\n    " << ENE_FORMATTING << energy << '\n';

	if (has_stresses())
	{
		ofs << " PlusStress:  xx          yy          zz"
			<< "          yz          xz          xy\n";

			ofs << "     " << STR_FORMATTING << stresses[0][0]
				<< ' ' << STR_FORMATTING << stresses[1][1]
				<< ' ' << STR_FORMATTING << stresses[2][2]
				<< ' ' << STR_FORMATTING << stresses[1][2]
				<< ' ' << STR_FORMATTING << stresses[0][2]
				<< ' ' << STR_FORMATTING << stresses[0][1];
		ofs << '\n';
	}

	if (features.size()>0)
	{
		for (auto& feature : features) 
		{
			std::string key = std::string(feature.first);
			std::string val = std::string(feature.second);
			while (!val.empty() && !key.empty())
			{
				size_t cut = val.find_first_of('\n');
				if (cut > val.size())
				{
					ofs << " Feature   " << mlip_string_escape(key) << '\t' << val << '\n';
					break;
				}
				else
				{
					ofs << " Feature   " << mlip_string_escape(key)
						<< '\t' << val.substr(0, cut) << '\n';
					val = val.substr(cut+1);
					if (val.empty())
						ofs << " Feature   " << mlip_string_escape(key)
							<< "\t\n";
				}
			}
		}
	}

	ofs << "END_CFG\n" << endl;
}

void Configuration::WriteJsonCfg(ofstream& ofs) const
{
	int LAT_WIDTH; int LAT_PRECISION;
	int IND_WIDTH; int IND_PRECISION;
	int TYP_WIDTH; int TYP_PRECISION;
	int POS_WIDTH; int POS_PRECISION;
	int SEN_WIDTH; int SEN_PRECISION;
	int FRC_WIDTH; int FRC_PRECISION;
	int CHG_WIDTH; int CHG_PRECISION;
	int ENE_WIDTH; int ENE_PRECISION;
	int STR_WIDTH; int STR_PRECISION;

	if (!ofs.is_open())
		ERROR("Output stream not open");

	ofs.precision(16);
	ofs << scientific;
	LAT_WIDTH = 1; LAT_PRECISION = 16;
	IND_WIDTH = 1; IND_PRECISION = 0;
	TYP_WIDTH = 1; TYP_PRECISION = 0;
	POS_WIDTH = 1; POS_PRECISION = 16;
	SEN_WIDTH = 1; SEN_PRECISION = 16;
	FRC_WIDTH = 1; FRC_PRECISION = 16;
	CHG_WIDTH = 1; CHG_PRECISION = 16;
	ENE_WIDTH = 1; ENE_PRECISION = 16;
	STR_WIDTH = 1; STR_PRECISION = 16;

	int _size = size();

	ofs << "{\n"; // BEGIN_CFG

	ofs << "  \"size\": " << _size;
	
	{
		int count = 0;
		for (int a = 0; a < 3; a++)
			if (lattice[a][0] != 0 || lattice[a][1] != 0 || lattice[a][2] != 0) count = a+1;
		
		if (count != 0) {
			ofs << ",\n  \"cell\": [";
			for (int a = 0; a < count; a++) {
				if(a) ofs << ',';
				ofs << "[" << LAT_FORMATTING << lattice[a][0]
				<< ',' << LAT_FORMATTING << lattice[a][1]
				<< ',' << LAT_FORMATTING << lattice[a][2] << ']';
			}
			ofs << "]";
		}
	}

	{
		ofs << ",\n  \"pos\": [";
		for (int i=0; i<size(); i++) {
			if(i) ofs << ", ";
			ofs << "[" << POS_FORMATTING << pos_[i][0]
				<< ',' << POS_FORMATTING << pos_[i][1]
				<< ',' << POS_FORMATTING << pos_[i][2] << ']';
		}
		ofs << "]";

		ofs << ",\n  \"types\": [";
		for (int i=0; i<size(); i++) {
			if(i) ofs << ", ";
			ofs << TYP_FORMATTING << types_[i];
		}
		ofs << "]";
		if (has_site_energies()) {
			ofs << ",\n  \"site_en\": [";
			for (int i=0; i<size(); i++) {
				if(i) ofs << ", ";
				ofs << SEN_FORMATTING << site_energy(i);
			}
			ofs << "]";
		}
		if (has_forces()) {
			ofs << ",\n  \"forces\": [";
			for (int i=0; i<size(); i++) {
				if(i) ofs << ", ";
				ofs << "[" << FRC_FORMATTING << forces_[i][0]
					<< ',' << FRC_FORMATTING << forces_[i][1]
					<< ',' << FRC_FORMATTING << forces_[i][2] << ']';
			}
			ofs << "]";
		}
		if (has_charges()) {
			ofs << ",\n  \"charges\": [";
			for (int i=0; i<size(); i++) {
				if(i) ofs << ", ";
				ofs << CHG_FORMATTING << charges(i);
			}
			ofs << "]";
		}
	}

	if (has_energy()) 
		ofs << ",\n  \"energy\": " << ENE_FORMATTING << energy;

	if (has_stresses())
	{
		ofs << ",\n  \"stress\": ";
		ofs << "[" << STR_FORMATTING << stresses[0][0]
			<< ',' << STR_FORMATTING << stresses[1][1]
			<< ',' << STR_FORMATTING << stresses[2][2]
			<< ',' << STR_FORMATTING << stresses[1][2]
			<< ',' << STR_FORMATTING << stresses[0][2]
			<< ',' << STR_FORMATTING << stresses[0][1];
		ofs << ']';
	}

	if (features.size()>0)
	{
		ofs << ",\n  \"features\": {";
                bool first_written = false;
		for (auto& feature : features) 
		{
			std::string key = std::string(feature.first);
			std::string val = std::string(feature.second);
                        
			if(first_written) ofs << ", ";
			ofs << '\"' << json_string_escape(key) << "\": \"" << json_string_escape(val) << "\"";
                        first_written = true;
		}
		ofs << "}"; // end features
	}

	ofs << "\n}"; // END_CFG
}

void Configuration::AppendToFile(const std::string & filename, unsigned int flags) const
{
	ofstream ofs(filename, ios::app | ios::binary );
	Save(ofs, flags);
}

void Configuration::SaveBin(std::ofstream & ofs) const
{
	if (!ofs.is_open())
		ERROR("Output stream not open");

	int _size = size();

	ofs << 'n';
	ofs.write((char*) &_size, sizeof(_size));

	ofs << 'l';
	for (int a = 0; a < 3; a++)
		for (int b = 0; b < 3; b++)
			ofs.write((char*) &lattice[a][b], sizeof(double));

	ofs << 'p';
	if ((_size > 0) && (!pos_.empty()))
		ofs.write((char*)&pos_[0], _size*sizeof(pos_[0]));

	ofs << 't';
	if ((_size > 0) && (!types_.empty()))
		ofs.write((char*)&types_[0], _size*sizeof(types_[0]));

	if (has_forces())
	{
		ofs << 'f';
		ofs.write((char*)&forces_[0], _size*sizeof(forces_[0]));
	}

	if (has_site_energies())
	{
		ofs << 'v';
		ofs.write((char*)&site_energies_[0], _size*sizeof(site_energies_[0]));
	}

	if (has_charges())
	{
		ofs << 'c';
		ofs.write((char*)&charges_[0], _size*sizeof(charges_[0]));
	}

	if (has_stresses())
	{
		ofs << 's';
		ofs.write((char*)&stresses[0][0], sizeof(stresses[0][0]));
		ofs.write((char*)&stresses[1][1], sizeof(stresses[0][0]));
		ofs.write((char*)&stresses[2][2], sizeof(stresses[0][0]));
		ofs.write((char*)&stresses[0][1], sizeof(stresses[0][0]));
		ofs.write((char*)&stresses[1][2], sizeof(stresses[0][0]));
		ofs.write((char*)&stresses[0][2], sizeof(stresses[0][0]));
	}

	if (has_energy())
	{
		ofs << 'e';
		ofs.write((char*)&energy, sizeof(energy));
	}
	
	for (auto& feature : features) 
	{
		ofs << 'F';
		int l;
		l = (int)feature.first.length();
		ofs.write((char*)&l, sizeof(l));
		ofs.write((char*)&feature.first[0], l*sizeof(char));
		l = (int)feature.second.length();
		ofs.write((char*)&l, sizeof(l));
		ofs.write((char*)&feature.second[0], l*sizeof(char));
	}

	ofs << '#';
	ofs.flush();
}

bool Configuration::ReadEFSFromProfessOutput(const std::string&  filename)
{
	std::ifstream ifs(filename);
	if (!ifs.is_open())
		ERROR("Can't open PROFESS output file \"" + filename + "\" for input");

	has_energy_ = has_stresses_ = false;

	int _size = size();

	string tmp_str = "";
	char tmp_buf[1000] = "";

	while (strncmp(tmp_buf, "#   TOTAL POTENTIAL ENERGY = ", 28))
	{
		ifs.getline(tmp_buf, 1000);
		if (ifs.eof())
			ERROR("Can't find energy in PROFESS output file\n");
	}
	for (int i = 0; i < 4; i++)
		ifs >> tmp_str;
	ifs >> energy;
	if (ifs.fail())
		ERROR("Error reading energy");
	else
		has_energy_ = true;

	for (int i = 0; i < 7; i++)
		ifs.getline(tmp_buf, 1000);
	forces_.resize(size());
	for (int i = 0; i < _size; i++)
	{
		ifs.ignore(12);
		ifs >> forces_[i][0] >> forces_[i][1] >> forces_[i][2];
		ifs.ignore(100, '\n');
	}
	if (ifs.fail() || ifs.eof())
		ERROR("Error reading");

	while (strncmp(tmp_buf, "                                STRESS (GPa)", 44))
	{
		ifs.getline(tmp_buf, 1000);
		if (ifs.eof())
			ERROR("Can't find energy in PROFESS output file\n");
	}
	ifs.ignore(15);
	ifs >> stresses[0][0] >> stresses[0][1] >> stresses[0][2];
	ifs.ignore(15);
	ifs >> stresses[1][0] >> stresses[1][1] >> stresses[1][2];
	ifs.ignore(15);
	ifs >> stresses[2][0] >> stresses[2][1] >> stresses[2][2];
	stresses[0][1] = stresses[1][0] = (stresses[0][1] + stresses[1][0]) / 2;
	stresses[0][2] = stresses[2][0] = (stresses[0][2] + stresses[2][0]) / 2;
	stresses[2][1] = stresses[1][2] = (stresses[2][1] + stresses[1][2]) / 2;

	const double eVA3_to_GPa = 160.2176487;
	stresses *= -fabs(lattice.det()) / eVA3_to_GPa;
	
	if (ifs.fail() || ifs.eof())
		ERROR("Error reading energy");
	else
		has_stresses_ = true;

	ifs.close();

	return true;
}

void Configuration::WriteProfessIon(const std::string&  filename, std::string* species_table_tag) const
{
	std::ofstream ofs(filename);
	if (!ofs.is_open())
		ERROR("Can't open PROFESS input file \"" + filename + "\" for writing");

	int _size = size();

	ofs << "%BLOCK LATTICE_CART\n";
	ofs << '\t' << lattice[0][0] << '\t' << lattice[0][1] << '\t' << lattice[0][2] << '\n'
		<< '\t' << lattice[1][0] << '\t' << lattice[1][1] << '\t' << lattice[1][2] << '\n'
		<< '\t' << lattice[2][0] << '\t' << lattice[2][1] << '\t' << lattice[2][2] << '\n';
	ofs << "%END BLOCK LATTICE_CART\n";
	ofs << "%BLOCK POSITIONS_CART\n";
	for (int i = 0; i < _size; i++)
		ofs << '\t' << species_table_tag[types_[i]] << '\t' << pos_[i][0] << '\t' << pos_[i][1] << '\t' << pos_[i][2] << '\n';
	ofs << "%END BLOCK POSITIONS_CART\n";
	ofs << "%BLOCK SPECIES_POT\n";
	set<int> atom_types;
	for (int i = 0; i < _size; i++)
		atom_types.insert(types_[i]);
	for (auto p_type = atom_types.begin(); p_type != atom_types.end(); ++p_type)
		ofs << '\t' << species_table_tag[*p_type] << ' ' << species_table_tag[*p_type] << "_HC.lda.recpot\n";
	ofs << "%END BLOCK SPECIES_POT\n";

	ofs.close();
}

bool Configuration::LoadNextFromOUTCAR(std::ifstream& ifs, int maxiter, int ISMEAR)
{
	std::string line;


	int iter = -1;
	

	// Skipping to the end of the SCF loop
	while ((line.substr(0, 104) !=
		"------------------------ aborting loop because EDIFF is reached ----------------------------------------")
		&&
		(line.substr(0, 43) != "                  Total CPU time used (sec)")
		) {
		std::getline(ifs, line);

		if (ifs.eof())
			ERROR((string)"Outcar-file parsing error: Can't find string \"aborting loop because EDIFF is reached\""); // remove: 3
		//Reading number of ionic iterations
		else if (line.substr(0, 51) == "----------------------------------------- Iteration") // VASP 5.4.1
		{
			line.erase(0, 51);

			std::stringstream stream(line);

			string s = "";
			char t=(char)0;

			while (t != '(')
				t = (char)stream.get();

			stream >> iter;
			
		}
		else if (line.substr(0, 49) == "--------------------------------------- Iteration") // VASP 5.4.4
		{
			line.erase(0, 49);

			std::stringstream stream(line);

			string s = "";
			char t=(char)0;

			while (t != '(')
				t = (char)stream.get();

			stream >> iter;
			
		}
	}

	if (maxiter != 0 && iter >= maxiter) {
		features["EFS_by"] = "VASP_not_converged";
		Warning("Loading configuration from OUTCAR: non-converged calculation detected. " + 
				 to_string(iter) + " iterations done");
	}
	else
		features["EFS_by"] = "VASP";

	// No "Total CPU time used (sec)" = likely broken outcar
	if (line.substr(0, 43) == "                  Total CPU time used (sec)")
		return false;

	// We now expect a configuration, so we can erase the current one
	has_energy(false);
	has_forces(false);
	has_stresses(false);
	pos_.clear();
	forces_.clear();

	// stresses reading block
	{
		// read until we find stresses (sometimes absent in the file) or lattice (always present)
		while (
			(line.substr(0, 24) != "  FORCE on cell =-STRESS") &&
			(line.substr(0, 71) != "      direct lattice vectors                 reciprocal lattice vectors")
			) {
			std::getline(ifs, line);
			if (ifs.eof())
				ERROR((string)"Outcar-file parsing error: Can't find stresses");
		}

		// check if found the stresses
		if (line.substr(0, 24) == "  FORCE on cell =-STRESS") {

			while (line.substr(0, 7) != "  Total") {
				std::getline(ifs, line);
				if (ifs.eof())
					ERROR((string)"Outcar-file parsing error: Can't find the stresses\"");
			}

			// now parsing the line
			std::stringstream stream(line);

			has_stresses(true);
			std::string tempstring;
			stream >> tempstring;

			stream >> stresses[0][0]
				>> stresses[1][1]
				>> stresses[2][2]
				>> stresses[0][1]
				>> stresses[1][2]
				>> stresses[0][2];

			stresses[1][0] = stresses[0][1];
			stresses[2][1] = stresses[1][2];
			stresses[2][0] = stresses[0][2];
		}
	}

	{// lattice reading block
		while (line.substr(0, 71) != "      direct lattice vectors                 reciprocal lattice vectors") {
			std::getline(ifs, line);
			if (ifs.eof())
				ERROR((string)"Outcar-file parsing error: Can't find lattice data");
		}
		double foo;
		ifs >> lattice[0][0] >> lattice[0][1] >> lattice[0][2] >> foo >> foo >> foo;
		ifs >> lattice[1][0] >> lattice[1][1] >> lattice[1][2] >> foo >> foo >> foo;
		ifs >> lattice[2][0] >> lattice[2][1] >> lattice[2][2] >> foo >> foo >> foo;
	}

	{// positions and forces reading block
									   
		while (line.substr(0, 70) != " POSITION                                       TOTAL-FORCE (eV/Angst)") {
			std::getline(ifs, line);

			if (ifs.eof())
				ERROR((string)"Outcar-file parsing error: Can't find forces");
				

		}

		std::getline(ifs, line);
		while ((!ifs.fail()) && (!ifs.eof())) {
			pos_.emplace_back();
			forces_.emplace_back();
			ifs >> pos_.back()[0] >> pos_.back()[1] >> pos_.back()[2]
				>> forces_.back()[0] >> forces_.back()[1] >> forces_.back()[2];
		}

		if (ifs.eof())
			ERROR("error reading");
		ifs.clear();

		pos_.pop_back();
		forces_.pop_back();
	}

	{// energy reading block
		while (line.substr(0, 46) != "  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)") {
			std::getline(ifs, line);
			if (ifs.eof())
				ERROR((string)"Outcar-file parsing error: Can't find energy");
		}
		std::getline(ifs, line);
		string str1 = "";
		string str2 = "";
		string str3 = "";
		string str4 = "";
		ifs >> str1 >> str2 >> str3 >> str4;
		if (str1 != "free" && str2 != "energy" && str3 != "TOTEN" && str4 != "=")
			ERROR((string)"Outcar-file parsing error: Can't read energy");
		ifs >> energy;
		if(ISMEAR>0) {
			std::getline(ifs, line);
			ifs >> str1 >> str2 >> str3;
//std::cerr << str1 << "," << str2 << ","<<str3;
			if (str1 != "energy" && str2 != "without" && str3 != "entropy=")
				ERROR((string)"Outcar-file parsing error: Can't read energy");
			ifs >> energy;
			ifs >> str1 >> str4;
			if (str1 != "energy(sigma->0)" && str4 != "=")
				ERROR((string)"Outcar-file parsing error: Can't read energy");
			ifs >> energy;
		}
	}

	if (ifs.fail() || ifs.eof()) {
		ifs.close();
		ERROR("error reading");
	} else {
		has_energy_ = true;
		site_energies_.clear();
	}

	return true;
}


bool Configuration::LoadNextFromOUTCARold(std::ifstream& ifs, int maxiter, int ISMEAR)
{
	std::string line;


	int iter = -1;
	

	// Skipping to the end of the SCF loop
	while ((line.substr(0, 104) !=
		"------------------------ aborting loop because EDIFF is reached ----------------------------------------")
		&&
		(line.substr(0, 43) != "                  Total CPU time used (sec)")
		) {
		std::getline(ifs, line);

		if (ifs.eof())
			ERROR((string)"Outcar-file parsing error: Can't find string \"aborting loop because EDIFF is reached\""); // remove: 3
		//Reading number of ionic iterations
		else if (line.substr(0, 51) == "----------------------------------------- Iteration")
		{
			line.erase(0, 51);

			std::stringstream stream(line);

			string s = "";
			char t=(char)0;

			while (t != '(')
				t = (char)stream.get();

			stream >> iter;
			
		}
	}

	if (maxiter != 0 && iter >= maxiter) {
		features["EFS_by"] = "VASP_not_converged";
		Warning("Loading configuration from OUTCAR: non-converged calculation detected. " + 
				 to_string(iter) + " iterations done");
	}
	else
		features["EFS_by"] = "VASP";

	// No "Total CPU time used (sec)" = likely broken outcar
	if (line.substr(0, 43) == "                  Total CPU time used (sec)")
		return false;

	// We now expect a configuration, so we can erase the current one
	has_energy(false);
	has_forces(false);
	has_stresses(false);
	pos_.clear();
	forces_.clear();

	{// energy reading block
		while (line.substr(0, 46) != "  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)") {
			std::getline(ifs, line);
			if (ifs.eof())
				ERROR((string)"Outcar-file parsing error: Can't find energy");
		}
		std::getline(ifs, line);
		string str1 = "";
		string str2 = "";
		string str3 = "";
		string str4 = "";
		ifs >> str1 >> str2 >> str3 >> str4;
		if (str1 != "free" && str2 != "energy" && str3 != "TOTEN" && str4 != "=")
			ERROR((string)"Outcar-file parsing error: Can't read energy");
		ifs >> energy;
		if(ISMEAR>0) {
			ifs >> str1 >> str2 >> str3 >> str4;
			if (str1 != "energy" && str2 != "without" && str3 != "entropy" && str4 != "=")
				ERROR((string)"Outcar-file parsing error: Can't read energy");
			ifs >> energy;
			ifs >> str1 >> str4;
			if (str1 != "energy(sigma->0)" && str4 != "=")
				ERROR((string)"Outcar-file parsing error: Can't read energy");
			ifs >> energy;
		}
	}

	// stresses reading block
	{
		// read until we find stresses (sometimes absent in the file) or lattice (always present)
		while (
			(line.substr(0, 24) != "  FORCE on cell =-STRESS") &&
			(line.substr(0, 71) != "      direct lattice vectors                 reciprocal lattice vectors")
			) {
			std::getline(ifs, line);
			if (ifs.eof())
				ERROR((string)"Outcar-file parsing error: Can't find stresses");
		}

		// check if found the stresses
		if (line.substr(0, 24) == "  FORCE on cell =-STRESS") {

			while (line.substr(0, 7) != "  Total") {
				std::getline(ifs, line);
				if (ifs.eof())
					ERROR((string)"Outcar-file parsing error: Can't find the stresses\"");
			}

			// now parsing the line
			std::stringstream stream(line);

			has_stresses(true);
			std::string tempstring;
			stream >> tempstring;

			stream >> stresses[0][0]
				>> stresses[1][1]
				>> stresses[2][2]
				>> stresses[0][1]
				>> stresses[1][2]
				>> stresses[0][2];

			stresses[1][0] = stresses[0][1];
			stresses[2][1] = stresses[1][2];
			stresses[2][0] = stresses[0][2];
		}
	}

	{// lattice reading block
		while (line.substr(0, 71) != "      direct lattice vectors                 reciprocal lattice vectors") {
			std::getline(ifs, line);
			if (ifs.eof())
				ERROR((string)"Outcar-file parsing error: Can't find lattice data");
		}
		double foo;
		ifs >> lattice[0][0] >> lattice[0][1] >> lattice[0][2] >> foo >> foo >> foo;
		ifs >> lattice[1][0] >> lattice[1][1] >> lattice[1][2] >> foo >> foo >> foo;
		ifs >> lattice[2][0] >> lattice[2][1] >> lattice[2][2] >> foo >> foo >> foo;
	}

	{// positions and forces reading block
									   
		while (line.substr(0, 70) != " POSITION                                       TOTAL-FORCE (eV/Angst)") {
			std::getline(ifs, line);

			if (ifs.eof())
				ERROR((string)"Outcar-file parsing error: Can't find forces");
				

		}

		std::getline(ifs, line);
		while ((!ifs.fail()) && (!ifs.eof())) {
			pos_.emplace_back();
			forces_.emplace_back();
			ifs >> pos_.back()[0] >> pos_.back()[1] >> pos_.back()[2]
				>> forces_.back()[0] >> forces_.back()[1] >> forces_.back()[2];
		}

		if (ifs.eof())
			ERROR("error reading");
		ifs.clear();

		pos_.pop_back();
		forces_.pop_back();
	}

	if (ifs.fail() || ifs.eof()) {
		ifs.close();
		ERROR("error reading");
	} else {
		has_energy_ = true;
		site_energies_.clear();
	}

	return true;
}

int LoadPreambleFromOUTCAR(std::ifstream& ifs, std::vector<int>& ions_per_type, bool& is_old_vasp, int& ISMEAR)
{
	std::string line;

	// Finding ions per type data by keywords
	while (line.substr(0, 18) !=
		"   ions per type =") {
		std::getline(ifs, line);
        if (line.substr(0,7) == " vasp.2" || line.substr(0,7) == " vasp.3" || line.substr(0,7) == " vasp.4") {
            is_old_vasp = true;
        }
        if (line.substr(0,7) == " vasp.5" || line.substr(0,7) == " vasp.6" || line.substr(0,7) == " vasp.7") {
            is_old_vasp = false;
        }
		if (ifs.eof())
			ERROR((string)"\"ions per type\" not given");
	}

	// now parsing the line, extracing ions per type
	{
		line.erase(0, 18);
		std::stringstream stream(line);
		int num;
		stream >> num;
		while (stream) {
			ions_per_type.emplace_back(num);
			stream >> num;
		}
	}

	//Reading max. number of ionic iterations
	while (line.substr(0, 11) !=
		"   NELM   =") {
		std::getline(ifs, line);
		if (ifs.eof())
			ERROR((string)"\"Number of electronic iterations\" not given");
	}

	int maxiter = -1;
	{
		line.erase(0, 11);
		std::stringstream stream(line);
		stream >> maxiter;
	}

	while (line.substr(0, 11) !=
		"   ISMEAR =") {
		std::getline(ifs, line);
		if (ifs.eof())
			ERROR((string)"\"ISMEAR\" not given");
	}

	{
		line.erase(0, 11);
		std::stringstream stream(line);
		stream >> ISMEAR;
	}
	return maxiter;
}

void Configuration::LoadFromOUTCAR(const std::string& filename)
{

	std::ifstream ifs(filename);
	if (!ifs.is_open())
		ERROR((string)"Can't open OUTCAR file \"" + filename + "\" for reading");

	std::vector<int> ions_per_type;
    bool is_old_vasp = false;
	int ISMEAR = 0;
	int maxiter = LoadPreambleFromOUTCAR(ifs, ions_per_type, is_old_vasp, ISMEAR);

    if (is_old_vasp == false) {

	if (!LoadNextFromOUTCAR(ifs, maxiter, ISMEAR))
		ERROR("No configuration read from OUTCAR");

    }
    else {

    if (!LoadNextFromOUTCARold(ifs, maxiter, ISMEAR))
        ERROR("No configuration read from OUTCAR");

    }

	std::string line;
	while (line.substr(0, 36) != " total amount of memory used by VASP") {
		std::getline(ifs, line);
		if (ifs.eof())
			ERROR((string)"OUTCAR does not end with \"total amount of memory used by VASP\"");
	}

	// filling types from ions_per_type
	types_.resize(size(), 0);
	int i = 0;
	for (int k = 0; k < (int)ions_per_type.size(); k++)
		for (int j = 0; j < ions_per_type[k]; j++)
			types_[i++] = k;
	if (i != size())
		ERROR((string)"Outcar-file parsing error: ions_per_type do not match the atoms count");
}

//! Assumes atom types 0, 1, ... correspond to those in POTCAR.
void Configuration::WriteVaspPOSCAR(const std::string& filename) const
{
#ifdef MLIP_LOSSLESS_CFG
	const int LAT_WIDTH = 23; const int LAT_PRECISION = 16;
	const int POS_WIDTH = 23; const int POS_PRECISION = 16;
#else
	const int LAT_WIDTH = 23; const int LAT_PRECISION = 16;
	const int POS_WIDTH = 23; const int POS_PRECISION = 16;
#endif

	std::ofstream ofs(filename);
	if (!ofs.is_open())
		ERROR((string)"Can't open file \"" + filename + "\" for output");

	ofs.precision(16);

	ofs << "MLIP output to VASP\n";

	int _size = size();

	// calculating the maximal atom type (minimal is 0)
	int max_type = 0;
	for (int i = 0; i < _size; i++)
		if (max_type < type(i)) max_type = type(i);

	// calculating of number of atoms per type
	std::vector<int> type_count;
	type_count.resize(1+max_type);
	type_count.assign(1+max_type, 0);
	for (int i = 0; i < _size; i++)
		type_count[type(i)] ++;

	double scale_factor = 1.0;	

	ofs << scale_factor << '\n';
	for	(int i = 0; i < 3; i++)
		ofs << "    " << LAT_FORMATTING << lattice[i][0] / scale_factor
			<< ' ' << LAT_FORMATTING << lattice[i][1] / scale_factor
			<< ' ' << LAT_FORMATTING << lattice[i][2] / scale_factor << '\n';

	for (int t = 0; t <= max_type; t++)
		ofs << ' ' << type_count[t];
	ofs << '\n';
	ofs << "cart\n";
	for (int t = 0; t <= max_type; t++)
		for (int i = 0; i < _size; i++)
			if(type(i) == t)
				ofs << "    " << POS_FORMATTING << pos_[i][0] / scale_factor
					<< ' ' << POS_FORMATTING << pos_[i][1] / scale_factor
					<< ' ' << POS_FORMATTING << pos_[i][2] / scale_factor << '\n';

	ofs.close();
}

bool LoadDynamicsFromOUTCAR(std::vector<Configuration> &db, const std::string& filename)
{
	std::ifstream ifs(filename);
	if (!ifs.is_open())
		ERROR((string)"Can't open OUTCAR file \"" + filename + "\" for reading");

	std::string line;
	std::vector<int> ions_per_type;
    bool is_old_vasp = false;

	int ISMEAR = 0;
	int maxiter = LoadPreambleFromOUTCAR(ifs, ions_per_type, is_old_vasp, ISMEAR);
	MlipException err("");
	bool caught = false;
	bool status = false;
    if (is_old_vasp == false) {
	for (Configuration cfg; ; db.push_back(cfg)) {
		try { status = cfg.LoadNextFromOUTCAR(ifs, maxiter, ISMEAR); }
		catch (const MlipException &e) {
			err = e; caught = true;
		}
		if (caught || !status) break;
		// filling types from ions_per_type
		cfg.types_.resize(cfg.size(), 0);
		int i = 0;
		for (int k = 0; k < (int)ions_per_type.size(); k++)
			for (int j = 0; j < ions_per_type[k]; j++)
				cfg.types_[i++] = k;
		if (i != cfg.size())
			ERROR((string)"OUTCAR file parsing error: ions_per_type (" + to_string(i) + ") do not match the atoms count (" + to_string(cfg.size()) + ")");
	}

    }
    else {
    for (Configuration cfg; ; db.push_back(cfg)) {
        try { status = cfg.LoadNextFromOUTCARold(ifs); }
        catch (const MlipException &e) {
            err = e; caught = true;
        }
        if (caught || !status) break;
        // filling types from ions_per_type
        cfg.types_.resize(cfg.size(), 0);
        int i = 0; 
        for (int k = 0; k < (int)ions_per_type.size(); k++)
            for (int j = 0; j < ions_per_type[k]; j++)
                cfg.types_[i++] = k;
        if (i != cfg.size())
            ERROR((string)"OUTCAR file parsing error: ions_per_type (" + to_string(i) + ") do not match the atoms count (" + to_string(cfg.size()) + ")");
    }

    }

	return !caught;
}

//! loads the last configuration from OUTCAR dynamics
//! returns false if the OUTCAR is broken, but at least one configuration was read successfully
bool Configuration::LoadLastFromOUTCAR(const std::string& filename)
{
	std::ifstream ifs(filename);
	if (!ifs.is_open())
		ERROR((string)"Can't open OUTCAR file \"" + filename + "\" for reading");

	std::string line;
	std::vector<int> ions_per_type;
    bool is_old_vasp = false;

	int ISMEAR = 0;
	int maxiter = LoadPreambleFromOUTCAR(ifs, ions_per_type, is_old_vasp, ISMEAR);

	MlipException err("");
	int i = 0;
	bool caught = false;
	try 
	{
        if (is_old_vasp == false) {

		while (LoadNextFromOUTCAR(ifs))
			;
		// filling types from ions_per_type
		types_.resize(size(), 0);
		for (int k = 0; k < (int)ions_per_type.size(); k++)
			for (int j = 0; j < ions_per_type[k]; j++)
				types_[i++] = k;

        }
        else {

        while (LoadNextFromOUTCARold(ifs))
            ;
        // filling types from ions_per_type
        types_.resize(size(), 0);
        for (int k = 0; k < (int)ions_per_type.size(); k++)
            for (int j = 0; j < ions_per_type[k]; j++)
                types_[i++] = k;

        }
	}
	catch (const MlipException &e) {
		err = e; caught = true;
	}

	if (i != size()) {
		if(caught)
			ERROR((std::string)err.What() + "\nAlso: tried to return the latest good config,\n  but ions_per_type (" + to_string(i) + ") do not match the atoms count (" + to_string(size()) + ")\\\n");
		ERROR((string)"OUTCAR file parsing error: ions_per_type (" + to_string(i) + ") do not match the atoms count (" + to_string(size()) + ")");
	}
	return !caught;
}

//! stupid O(N^2) algorithm
double Configuration::MinDist()
{
	// is the configuration periodic in any dimension?
	bool is_periodic[3];
	for (int a = 0; a < 3; a++)
		is_periodic[a] = lattice[a][0] != 0.0
		|| lattice[a][1] != 0.0
		|| lattice[a][2] != 0.0;

	// length of extension in 3 directions
	int ll[3];
	for (int a = 0; a < 3; a++)
		ll[a] = is_periodic[a] ? 1 : 0;

	double mindist = 9.9e99;
	for (int ind2 = 0; ind2 < size(); ind2++)
		for (int ind = ind2; ind < size(); ind++)
			for (int i = -ll[0]; i <= ll[0]; i++)
				for (int j = -ll[1]; j <= ll[1]; j++)
					for (int k = -ll[2]; k <= ll[2]; k++)
						if ((ind2 != ind) || (i != 0) || (j != 0) || (k != 0)) {
							Vector3 candidate = pos(ind);
							for (int a = 0; a < 3; a++)
								candidate[a] += i*lattice[0][a] + j*lattice[1][a] + k*lattice[2][a];
							mindist = __min(mindist, distance(pos(ind2), candidate));
						}
	return mindist;
}

vector<Configuration> LoadCfgs(const string& filename, const int max_count, const int proc_rank, const int proc_size)
{
	vector<Configuration> cfgs;
	ifstream ifs(filename, ios::binary);
	if (!ifs.is_open())
		ERROR("Can't open file \"" + filename + "\" for reading configurations");
	int cntr = 0;
	for (Configuration cfg; cntr < max_count && cfg.Load(ifs); cntr++)
		if (cntr%proc_size==proc_rank)
			cfgs.emplace_back(cfg);
	return cfgs;
}


void Configuration::ToAtomicUnits()
{
	const double length_unit = 1.889725989; // (becomes Bohr)
	const double energy_unit = 1.0/27.211385;   // (becomes Hartree)
	lattice *= length_unit;
	int size = (int)pos_.size();
	for (int i = 0; i < size; i++)
		pos(i) *= length_unit;
	energy *= energy_unit;
	if (has_forces()) {
		for (int i = 0; i < size; i++)
			force(i) *= energy_unit/length_unit;
	}
	if (has_stresses())
		stresses *= energy_unit;
}

void Configuration::FromAtomicUnits()
{
	const double length_unit = 1.0/1.889725989;
	const double energy_unit = 27.211385;
	lattice *= length_unit;
	int size = (int)pos_.size();
	for (int i = 0; i < size; i++)
		pos(i) *= length_unit;
	energy *= energy_unit;
	if (has_forces()) {
		for (int i = 0; i < size; i++)
			force(i) *= energy_unit / length_unit;
	}
	if (has_stresses())
		stresses *= energy_unit;
}

void Configuration::WriteLammpsDatafile(const std::string& filename) const
{
	Configuration cfg(*this);

	Matrix3 Q;
	Matrix3 L;
	cfg.lattice.transpose().QRdecomp(Q, L);
	cfg.Deform(Q);

	cfg.lattice[0][1] = 0.0;
	cfg.lattice[0][2] = 0.0;
	cfg.lattice[1][2] = 0.0;

	cfg.MoveAtomsIntoCell();

	ofstream ofs(filename);

	int n_at_types = 0; // number of atom types
	for (int i = 0; i < cfg.size(); i++)
		if (cfg.type(i) > n_at_types) n_at_types = cfg.type(i);
	n_at_types++;

	ofs << "\n";
	ofs << std::fixed << std::setprecision(9) << 0.0;
	ofs << " " << cfg.lattice[0][0] << " xlo xhi\n";

	ofs << std::fixed << std::setprecision(9) << 0.0;
	ofs << " " << cfg.lattice[1][1] << " ylo yhi\n";

	ofs << std::fixed << std::setprecision(9) << 0.0;
	ofs << " " << cfg.lattice[2][2] << " zlo zhi\n";

	ofs << std::fixed << std::setprecision(9);
	ofs << cfg.lattice[1][0] << " ";
	ofs << std::fixed << std::setprecision(9);
	ofs << cfg.lattice[2][0] << " ";
	ofs << std::fixed << std::setprecision(9);
	ofs << cfg.lattice[2][1] << " xy xz yz\n\n";

	ofs << cfg.size() << " atoms\n\n";
	ofs << n_at_types << " atom types\n\n";

	ofs << "Atoms\n\n";

	for (int i = 0; i < cfg.size(); i++) {
		ofs << std::right << std::setw(6) << i + 1;
		ofs << std::right << std::setw(6) << cfg.type(i) + 1;
		ofs << std::right << std::setw(20) << std::fixed << std::setprecision(8) << cfg.pos(i, 0);
		ofs << std::right << std::setw(20) << cfg.pos(i, 1);
		ofs << std::right << std::setw(20) << cfg.pos(i, 2) << "\n";
	}

	ofs.close();
}
