/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Alexander Shapeev, Evgeny Podryabinkin
 */

#ifndef MLIP_NEIGHBORHOODS_H
#define MLIP_NEIGHBORHOODS_H

#include "configuration.h"

//! Structure describing a neighborhood of an atom
struct Neighborhood
{
	int count;                      //  number of neighbors
	std::vector<int> inds;          //	array of indices of neighbor atoms
	std::vector<Vector3> vecs;     //	array of relative positions of the neighbor atoms
	std::vector<double> dists;      //	array of distances to the neighbor atoms	
	std::vector<int> types;
	int my_type;
};

class Neighborhoods
{
  private:
	//!	Indices of original atoms corresponding to original and ghost atoms
	std::vector<int> orig_atom_inds;
	std::vector<Vector3> pos; //!< ghost atoms, temporary variable

	//!	Procedure adds new atoms to the configuration corresponding to periodical extention of the configuration. It doesn't change count-property. The number of atoms in extended configuration is available via pos.size().
	void InitNbhs_AddGhostAtoms(const Configuration& cfg, const double cutoff);
	//!	Calculates indicies of neighboring atoms. For atoms from periodical extensions indices of their atom-origins are used
	void InitNbhs_ConstructNbhs(const Configuration& cfg, const double cutoff);
	void InitNbhs_RemoveGhostAtoms();
	void InitNbhs(const Configuration& cfg, const double _cutoff);		//!< neighborhood initialization procedure

  protected:
	std::vector<Neighborhood> nbhs; //!< list of neighborhoods

  public:
	inline Neighborhood& operator[](int i)							{ return nbhs[i]; }				//!< returns nbhs[i], for a more readable code
	inline const Neighborhood& operator[](int i) const				{ return nbhs[i]; }				//!< returns nbhs[i], for a more readable code
	inline int size() const											{ return (int)nbhs.size(); }	//!< number of neighborhoods
	inline void resize(int new_size)								{ nbhs.resize(new_size); }	//!< number of neighborhoods
	inline std::vector<Neighborhood>::iterator begin()				{ return nbhs.begin(); }
	inline std::vector<Neighborhood>::const_iterator begin() const	{ return nbhs.begin(); }
	inline std::vector<Neighborhood>::iterator end()				{ return nbhs.end(); }
	inline std::vector<Neighborhood>::const_iterator end() const	{ return nbhs.end(); }
	double cutoff;					//	Cutoff radius neighborhoods initialized by. If neighborhoods not inited then cutoff = 0 
	Neighborhoods(const Configuration& cfg, const double _cutoff);
	Neighborhoods() { cutoff = 0.0; };
	~Neighborhoods();
	void Replace(const Configuration& cfg, const double _cutoff);
};

#endif // MLIP_NEIGHBORHOODS_H
