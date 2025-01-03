/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Alexander Shapeev, Evgeny Podryabinkin, Ivan Novikov
 */

#ifndef MLIP_CONFIGURATION_H
#define MLIP_CONFIGURATION_H

#include "mlip.h"
#include "common/matrix3.h"
#include <random>
#include <set>
//#include <map>


enum class CrystalSystemType{SC, BCC, FCC, HCP};


//! Atomic periodic configuration.
//! Configuration can be in either of the two states: uninitialized, when its size=0,
//! and initialized when its size()>0. The size()==pos_.size()==types_.size() always.
//! If the configuration is initialized, it may or may not have energy, forces, site_energies, etc.
//! To check whether it has energy, do `if(has_energy()) ...` .
//! To declare it does have energy, do `has_energy(true)`.
//! `has_energy(false)` does the opposite.
//! If forces (same as site_energies) are initialized, then forces_.size() == size().
//! Otherwise, forces_.size() == 0.
class Configuration
{
private:
	static long id_cntr;				//!< index counter	

protected:
	long id_;							//!< Configuration index

	//std::vector<int> types_;			//!< Array of atom types (0 default).
	std::vector<Vector3> pos_;			//!< Array of atom coordinates
	std::vector<Vector3> forces_;		//!< Array of forces on atoms
	std::vector<double> site_energies_;	//!< Array of site energies
	std::vector<double> charges_;

	bool has_energy_;
	bool has_stresses_;

	//!	Loading old-format configuration from the filestream. Returns true if success, false if end-of-file.
	bool LoadOld(std::ifstream& ifs);
	//! Loading one configuration from binary filestream. Returns true if read, false if end-of-file.
	bool LoadBin(std::ifstream& ifs);				

	//! reading one ionic iteration from VASP OUTCAR. Ions_per_type should be read before calling LoadNext.
	//! IMPORTANT: if there is no next configuration in the file, it should not overwrite the existing configuration!
	//! Returns true if success, false if end-of-file.
	bool LoadNextFromOUTCAR(std::ifstream&,int maxiter=0, int ISMEAR = 0);
	bool LoadNextFromOUTCARold(std::ifstream&,int maxiter=0, int ISMEAR = 0);
	int friend LoadPreambleFromOUTCAR(std::ifstream&, std::vector<int>& ions_per_type, bool& is_old_vasp, int& ISMEAR);	//!< Fills ions_per_type & max. number of iterations
public:

	//! Latice vectors in rows (i.e., lattice[0], lattice[1], lattice[2] are the 3 vectors). Required for periodical extension of the configuration. If lattice[3]==(0,0,0) then it is not periodic w.r.t lattice[3]. If all lattice==0 then it is an open-shell molecule, no periodicity whatsoever
	Matrix3 lattice;
	std::vector<int> types_;
	double energy; //!< Energy of the configuration
	std::vector<double>  cal_se;
	std::vector<double>  cal_se0;
	std::vector<double>  type_mean;
	
	std::set<int> unique_elems;

	Matrix3 stresses; //!< Stress tensor = derivative of the energy wrt the lattice (almost)
	
	//! An std::map that contains all nonstandard features
	std::map<std::string, std::string> features;
	std::map<int, int> type_num;
	
	inline long id() const;									//!< returns the configuration id, mainly for testing purposes
	inline long set_new_id();								//!< sets a new id

	inline int size() const;								//!< returns the number of atoms
	
	//! resizes the configuration (pos always, forces if present, site_energies if present)
	inline void resize(int new_size);

	inline bool has_energy() const; 						//!< checks whether there is energy
	inline bool has_stresses() const; 						//!< checks whether there are stresses
	inline bool has_forces() const;							//!< checks whether there are forces
	inline bool has_site_energies() const;					//!< checks whether there are site energies
	inline bool has_charges() const;						//!< checks whether there are site energies
	inline void has_energy(bool has_energy);				//!< declares whether or not there is energy
	inline void has_stresses(bool has_stresses);			//!< declares whether or not there are stresses
	inline void has_forces(bool has_forces);				//!< declares whether or not there are forces
	inline void has_site_energies(bool has_site_energies);	//!< declares whether or not there are site energies
	inline void has_charges(bool has_charges);				//!< declares whether or not there are site energies

	// Functions providing access to main fields of this class
	inline Vector3&        pos(int i);
	inline const Vector3&  pos(int i) const;
	inline double&         pos(int i, int a);
	inline const double&   pos(int i, int a) const;
	inline Vector3&	       force(int i);
	inline const Vector3&  force(int i) const;
	inline double&         force(int i, int a);
	inline const double&   force(int i, int a) const;
	inline int&            type(int i);
	inline const int&      type(int i) const;
	inline double&         site_energy(int i);
	inline const double&   site_energy(int i) const;
	inline double&         charges(int i);
	inline const double&   charges(int i) const;
	inline Vector3         direct_pos(int i) const;
	inline Matrix3         virial() const;

	Configuration();
	//! Construct a perfect crystal by repeating `replication_times` in each coordinate
	Configuration(	CrystalSystemType crystal_system,
					double basic_lattice_parameter,
					int replication_times = 1);
	//! Construct a perfect crystal by repeating `replication_times` in each coordinate
	Configuration(	CrystalSystemType crystal_system,
					double basic_lattice_parameter1,
					double basic_lattice_parameter2,
					double basic_lattice_parameter3,
					int replication_times = 1);
	~Configuration();
	void destroy();									//!< Clears all configuration data and neighborhoods

	//! compares two configurations. The features vector is allowed to be different
	bool operator==(const Configuration&) const;
	bool operator!=(const Configuration& cfg) const; //!< compares two configurations. The features vector is allowed to be different

	void Deform(const Matrix3& mapping);   //!< Apply linear transformation 'mapping' to the configuration
	void MoveAtomsIntoCell();              //!< Make fractional coords between 0 and 1 by periodically equivalent transformations
	void CorrectSupercell();               //!< Does following: 1. Construct an equivalent configuration by making it more orthogonal and make the supercell lower triangle; 2. Returns atoms from outside supercell to interior; 3. Make latice matrix lower triangle; 4. Forces and stresses preserved while transformations
	void ReplicateUnitCell(int, int, int); //!<Replicates configuration
	void ReplicateUnitCell(templateMatrix3<int>); //!<Replicates configuration
	double MinDist();								//!< Calculates minimal distance between atoms in configuration
	void ToAtomicUnits();							//!< Converts coordinates, forces, etc., into atomic units
	void FromAtomicUnits();							//!< Converts coordinates, forces, etc., into atomic units

	// I/O functions
	bool Load(std::ifstream& ifs);					//!< Loading one configuration from text filestream. Returns true if read, false if end-of-file.

	// Flags for Save()
	static const unsigned int SAVE_BINARY = 0x1;
	static const unsigned int SAVE_NO_LOSS = 0x2;
	static const unsigned int SAVE_DIRECT_COORDS = 0x4;

	void Save(std::ofstream& ofs, unsigned int flags = 0) const;					//!< Saving to a file stream
	void AppendToFile(const std::string& filename, unsigned int flags = 0) const;	//!< Append configuration to a file
	void SaveBin(std::ofstream& ofs) const;		//!< Saving, binary format

	// Import/Export functions to the other formats
	bool ReadEFSFromProfessOutput(const std::string& filename);				//!<	reading PROFESS output
	void WriteProfessIon(const std::string& filename, std::string*) const;	//!<	writing input for PROFESS
	void LoadFromOUTCAR(const std::string& filename);						//!<	reading VASP output
	bool LoadLastFromOUTCAR(const std::string& filename);					//!<	reading VASP output
	void WriteVaspPOSCAR(const std::string& filename) const;				//!<	writing POSCAR for VASP
	void WriteLammpsDatafile(const std::string& filename) const;			//!<	writing cfg for LAMMPS
	void WriteJsonCfg(std::ofstream& ofs) const;					//!< Saving as a json
	//! Load all configurations in OUTCAR while relaxation or dynamics, saves into db
	//! Returns Configuration::VALID on success
	friend bool LoadDynamicsFromOUTCAR(std::vector<Configuration> &db, const std::string& filename);
};



inline long Configuration::id() const
{ 
	return id_; 
} //!< returns configuration id, mainly for testing purposes

inline long Configuration::set_new_id()
{
	return id_ = id_cntr++;
} //!< returns configuration id, mainly for testing purposes

inline int Configuration::size() const
{ 
	return (int)pos_.size();
} //!< returns the number of atoms

inline void Configuration::resize(int new_size)
{
	if (new_size != size()) {
		pos_.resize(new_size);
		types_.resize(new_size);
		if (has_forces()) forces_.resize(new_size);
		if (has_site_energies()) site_energies_.resize(new_size);
		if (has_charges()) charges_.resize(new_size);
	}
}

inline Vector3& Configuration::pos(int i) {
#ifdef MLIP_DEBUG
	if (i > (int)pos_.size())
		ERROR("i > pos_.size()");
#endif // MLIP_DEBUG
	return pos_[i];
}
inline Vector3 Configuration::direct_pos(int i) const {
#ifdef MLIP_DEBUG
	if (i > (int)pos_.size())
		ERROR("i > pos_.size()");
#endif // MLIP_DEBUG
	return pos_[i] * lattice.inverse();
}
inline const Vector3& Configuration::pos(int i) const {
#ifdef MLIP_DEBUG
	if (i > (int)pos_.size())
		ERROR("i > pos_.size()");
#endif // MLIP_DEBUG
	return pos_[i];
}
inline Vector3& Configuration::force(int i) {
#ifdef MLIP_DEBUG
	if (i > (int)forces_.size())
		ERROR("i > forces_.size()");
#endif // MLIP_DEBUG
	return forces_[i];
}
inline const Vector3& Configuration::force(int i) const {
#ifdef MLIP_DEBUG
	if (i > (int)forces_.size())
		ERROR("i > forces_.size()");
#endif // MLIP_DEBUG
	return forces_[i];
}
inline int& Configuration::type(int i) {
#ifdef MLIP_DEBUG
	if (i > (int)types_.size())
		ERROR("i > types_.size()");
#endif // MLIP_DEBUG
	return types_[i];
}
inline const int& Configuration::type(int i) const {
#ifdef MLIP_DEBUG
	if (i > (int)types_.size())
		ERROR("i > types_.size()");
#endif // MLIP_DEBUG
	return types_[i];
}
inline double& Configuration::site_energy(int i) {
#ifdef MLIP_DEBUG
	if (i > (int)site_energies_.size())
		ERROR("i > site_energies_.size()");
#endif // MLIP_DEBUG
	return site_energies_[i];
}
inline const double& Configuration::site_energy(int i) const {
#ifdef MLIP_DEBUG
	if (i > (int)site_energies_.size())
		ERROR("i > site_energies_.size()");
#endif // MLIP_DEBUG
	return site_energies_[i];
}
inline double& Configuration::charges(int i) {
#ifdef MLIP_DEBUG
	if (i > (int)charges_.size())
		ERROR("i > charges_.size()");
#endif // MLIP_DEBUG
	return charges_[i];
}
inline const double& Configuration::charges(int i) const {
#ifdef MLIP_DEBUG
	if (i > (int)charges_.size())
		ERROR("i > charges_.size()");
#endif // MLIP_DEBUG
	return charges_[i];
}
inline double& Configuration::pos(int i, int a) {
#ifdef MLIP_DEBUG
	if (i > (int)pos_.size())
		ERROR("i > pos_.size()");
#endif // MLIP_DEBUG
	return pos_[i][a];
}
inline const double& Configuration::pos(int i, int a) const {
#ifdef MLIP_DEBUG
	if (i > (int)pos_.size())
		ERROR("i > pos_.size()");
#endif // MLIP_DEBUG
	return pos_[i][a];
}
inline double& Configuration::force(int i, int a) {
#ifdef MLIP_DEBUG
	if (i > (int)forces_.size())
		ERROR("i > forces_.size()");
#endif // MLIP_DEBUG
	return forces_[i][a];
}
inline const double& Configuration::force(int i, int a) const {
#ifdef MLIP_DEBUG
	if (i > (int)forces_.size())
		ERROR("i > forces_.size()");
#endif // MLIP_DEBUG
	return forces_[i][a];
}
inline Matrix3 Configuration::virial() const {
	double vol = fabs(lattice.det());
	return stresses * (1/vol);
}

inline bool Configuration::has_energy() const			//!< checks whether there is energy
{ 
	return has_energy_; 
}								
inline void Configuration::has_energy(bool has_energy)	//!< declares whether or not there is energy
{ 
	has_energy_ = has_energy; 
}				

inline bool Configuration::has_stresses() const 
{ 
	return has_stresses_; 
}							//!< checks whether there are stresses
inline void Configuration::has_stresses(bool has_stresses) 
{ 
	has_stresses_ = has_stresses; 
}		//!< declares whether or not there are stresses

inline bool Configuration::has_forces() const 
{ 
	return !forces_.empty(); 
}								//!< checks whether there are forces
inline void Configuration::has_forces(bool has_forces) 
{ 
	forces_.resize(has_forces ? size() : 0); 
}								//!< declares whether or not there are forces

inline bool Configuration::has_site_energies() const 
{ 
	return !site_energies_.empty(); 
}								//!< checks whether there are site energies
inline void Configuration::has_site_energies(bool has_site_energies)
{
	site_energies_.resize(has_site_energies ? size() : 0);
}								//!< declares whether or not there are site energies

inline bool Configuration::has_charges() const 
{ 
	return !charges_.empty(); 
}								//!< checks whether there are site energies
inline void Configuration::has_charges(bool has_charges) 
{ 
	charges_.resize(has_charges ? size() : 0); 
}								//!< declares whether or not there are site energies


// Global function loads configurations from file and returns vector of loaded configurations
std::vector<Configuration> LoadCfgs(const std::string& filename, const int max_count=HUGE_INT, const int proc_rank=0, const int proc_size=1); //two last arguments stand for parallel implementation

#endif // MLIP_CONFIGURATION_H
