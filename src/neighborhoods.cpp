/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Alexander Shapeev, Evgeny Podryabinkin
 */

#include <algorithm>
#include <memory.h>

#include "neighborhoods.h"


using namespace std;

//!	Procedure adds ghost atoms to the configuration according to the periodic extention.
//!	The number of atoms in extended configuration is available via pos_.size().
void Neighborhoods::InitNbhs_AddGhostAtoms(const Configuration& cfg, const double cut_off)
{
	const int size = cfg.size();
	pos.resize(size);
	memcpy(&pos[0][0], &cfg.pos(0,0), size*sizeof(Vector3));

	// computing the bounding box enclosing all the atoms
	Vector3 min_pos = pos[0];
	Vector3 max_pos = pos[0];
	for (int i = 1; i < size; i++) {
		for (int a = 0; a < 3; a++) {
			if (min_pos[a] > pos[i][a])
				min_pos[a] = pos[i][a];
			if (max_pos[a] < pos[i][a])
				max_pos[a] = pos[i][a];
		}
	}

	// is the configuration periodic in any dimension?
	bool is_periodic[3];
	for (int a = 0; a < 3; a++)
		is_periodic[a] = cfg.lattice[a][0] != 0.0
		              || cfg.lattice[a][1] != 0.0
		              || cfg.lattice[a][2] != 0.0;

	// periodically extends the box until no new ghost atoms are added
	bool added_new_points = true;
	for (int l = 1; added_new_points; l++) {
		added_new_points = false;

		// length of extension in 3 directions
		int ll[3];
		for (int a = 0; a < 3; a++)
			ll[a] = is_periodic[a] ? l : 0;

		for (int i = -ll[0]; i <= ll[0]; i++) {
			for (int j = -ll[1]; j <= ll[1]; j++) {
				for (int k = -ll[2]; k <= ll[2]; k++) {
					if (std::max(std::max(abs(i), abs(j)), abs(k)) == l) {
						for (int ind = 0; ind < size; ind++)
						{
							Vector3 candidate = pos[ind];
							for (int a = 0; a < 3; a++)
								candidate[a] += i*cfg.lattice[0][a] + j*cfg.lattice[1][a]
										+ k*cfg.lattice[2][a];

							if ((candidate[0] < min_pos[0] - cut_off)
								|| (candidate[1] < min_pos[1] - cut_off)
								|| (candidate[2] < min_pos[2] - cut_off)
								|| (candidate[0] > max_pos[0] + cut_off)
								|| (candidate[1] > max_pos[1] + cut_off)
								|| (candidate[2] > max_pos[2] + cut_off))
								continue;

							int jnd;
							for (jnd = 0;
								(jnd < size) && distance(candidate, pos[jnd]) > cut_off; jnd++)
								;

							if (jnd < size) { // found a point to add
								pos.push_back(candidate);
								orig_atom_inds.push_back(ind);
								added_new_points = true;
							}
						}
					}
				}
			}
		}
	}

}

//!	Fills vects, dists, etc., for the neighborhoods
//!	For atoms from periodical extensions indices of their atom-origins are used
void Neighborhoods::InitNbhs_ConstructNbhs(const Configuration& cfg, const double cut_off)
{
	int size = cfg.size();

	int MaxNeighbCount = 0;

	if (!nbhs.empty())
		nbhs.clear();
	nbhs.resize(size);

	// Loop over all pairs of atoms in configuration
	for (int i = 0; i < size; i++) {
		for (int j = i + 1; j < (int)pos.size(); j++)
		{	// pos_.size() != size after configuration extension
			// TODO: use Vector3
			double dx_ij = pos[j][0] - pos[i][0];
			double dy_ij = pos[j][1] - pos[i][1];
			double dz_ij = pos[j][2] - pos[i][2];

			double dist_ij = sqrt(dx_ij*dx_ij + dy_ij*dy_ij + dz_ij*dz_ij);

			if ((dist_ij < cut_off) && (i <= orig_atom_inds[j]))
			{
				// for (i > orig_atom_inds[j]) this connection should be already treated
				Neighborhood& neighborhood_i = nbhs[orig_atom_inds[i]];
				// Neighborhood& neighborhood_i = nbhs[i] - is the same
				Neighborhood& neighborhood_j = nbhs[orig_atom_inds[j]];

				neighborhood_i.count++;
				neighborhood_i.inds.push_back(orig_atom_inds[j]);
				neighborhood_i.vecs.emplace_back();
				neighborhood_i.vecs.back()[0] = dx_ij;
				neighborhood_i.vecs.back()[1] = dy_ij;
				neighborhood_i.vecs.back()[2] = dz_ij;
				neighborhood_i.dists.push_back(dist_ij);
				neighborhood_i.types.push_back(cfg.type(orig_atom_inds[j]));

				if (i != orig_atom_inds[j])
				{	//(i == orig_atom_inds[j]) if distance between atom and its image < CuttOffRad
					neighborhood_j.count++;
					neighborhood_j.inds.push_back(i);
					neighborhood_j.vecs.emplace_back();
					neighborhood_j.vecs.back()[0] = -dx_ij;
					neighborhood_j.vecs.back()[1] = -dy_ij;
					neighborhood_j.vecs.back()[2] = -dz_ij;
					neighborhood_j.dists.push_back(dist_ij);
					neighborhood_j.types.push_back(cfg.type(i));
				}

				MaxNeighbCount = __max(MaxNeighbCount,
					__max(neighborhood_i.count, neighborhood_j.count));
			}
		}
		nbhs[i].my_type = cfg.type(i);
	}
}

void Neighborhoods::InitNbhs_RemoveGhostAtoms()
{
	pos.clear();
	orig_atom_inds.clear();
}

void Neighborhoods::InitNbhs(const Configuration& cfg, const double _cutoff)
{
	if (nbhs.size() != 0)
		nbhs.clear();

	if (cfg.size() != 0) {
		orig_atom_inds.resize(cfg.size());
		for (int i = 0; i < cfg.size(); i++)
			orig_atom_inds[i] = i;

		InitNbhs_AddGhostAtoms(cfg, _cutoff);
		InitNbhs_ConstructNbhs(cfg, _cutoff);
		InitNbhs_RemoveGhostAtoms();
	}

	cutoff = _cutoff;
}

//!	Cut off radius neighborhoods initiated by. if neighborhoods not inited nbs_init_cutoff = 0 
Neighborhoods::Neighborhoods(const Configuration& cfg, const double _cutoff)
{
	Replace(cfg, _cutoff);
}

Neighborhoods::~Neighborhoods()
{
	nbhs.clear();
}

//!	Rewrites the existing neighborhood information
void Neighborhoods::Replace(const Configuration& cfg, const double _cutoff)
{
	if (nbhs.size() != cfg.size())
		nbhs.resize(cfg.size());
	InitNbhs(cfg, _cutoff);
	cutoff = _cutoff;
}

