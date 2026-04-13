/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Alexander Shapeev, Evgeny Podryabinkin
 */

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iostream>
#include <memory.h>

#include "neighborhoods.h"


using namespace std;

namespace {

std::string CurrentTimestamp()
{
	std::time_t now = std::time(nullptr);
	char buf[32];
	std::strftime(buf, sizeof(buf), "%F %T", std::localtime(&now));
	return std::string(buf);
}

struct AxisAlignedCellGrid
{
	Vector3 min_pos;
	double cell_size;
	double inv_cell_size;
	int dims[3];
	std::vector<std::vector<int> > cell_atoms;

	AxisAlignedCellGrid(const std::vector<Vector3>& positions, double cutoff)
		: min_pos(positions[0]),
		  cell_size(cutoff),
		  inv_cell_size(1.0 / cutoff)
	{
		Vector3 max_pos = positions[0];
		for (size_t i = 1; i < positions.size(); ++i) {
			for (int a = 0; a < 3; ++a) {
				min_pos[a] = std::min(min_pos[a], positions[i][a]);
				max_pos[a] = std::max(max_pos[a], positions[i][a]);
			}
		}

		for (int a = 0; a < 3; ++a) {
			dims[a] = std::max(1, static_cast<int>(std::floor((max_pos[a] - min_pos[a]) * inv_cell_size)) + 1);
		}

		const size_t cell_count =
			static_cast<size_t>(dims[0]) *
			static_cast<size_t>(dims[1]) *
			static_cast<size_t>(dims[2]);
		cell_atoms.resize(cell_count);

		for (int i = 0; i < static_cast<int>(positions.size()); ++i) {
			const Vector3int cell = Locate(positions[i]);
			cell_atoms[Flatten(cell[0], cell[1], cell[2])].push_back(i);
		}
	}

	inline int ClampCoord(double coord, int axis) const
	{
		int cell = static_cast<int>((coord - min_pos[axis]) * inv_cell_size);
		if (cell < 0)
			cell = 0;
		else if (cell >= dims[axis])
			cell = dims[axis] - 1;
		return cell;
	}

	inline Vector3int Locate(const Vector3& pos) const
	{
		return Vector3int(
			ClampCoord(pos[0], 0),
			ClampCoord(pos[1], 1),
			ClampCoord(pos[2], 2));
	}

	inline size_t Flatten(int ix, int iy, int iz) const
	{
		return
			(static_cast<size_t>(ix) * static_cast<size_t>(dims[1]) +
			 static_cast<size_t>(iy)) * static_cast<size_t>(dims[2]) +
			static_cast<size_t>(iz);
	}
};

inline void AppendNeighbor(Neighborhood& neighborhood,
						   int ind,
						   const Vector3& vec,
						   double dist,
						   int type)
{
	neighborhood.count++;
	neighborhood.inds.push_back(ind);
	neighborhood.vecs.push_back(vec);
	neighborhood.dists.push_back(dist);
	neighborhood.types.push_back(type);
}

bool HasOriginalAtomWithinCutoff(const AxisAlignedCellGrid& cell_grid,
								 const std::vector<Vector3>& positions,
								 const Vector3& candidate,
								 double cut_off_sq)
{
	const Vector3int origin_cell = cell_grid.Locate(candidate);
	for (int dx = -1; dx <= 1; ++dx) {
		const int cx = origin_cell[0] + dx;
		if (cx < 0 || cx >= cell_grid.dims[0])
			continue;

		for (int dy = -1; dy <= 1; ++dy) {
			const int cy = origin_cell[1] + dy;
			if (cy < 0 || cy >= cell_grid.dims[1])
				continue;

			for (int dz = -1; dz <= 1; ++dz) {
				const int cz = origin_cell[2] + dz;
				if (cz < 0 || cz >= cell_grid.dims[2])
					continue;

				const std::vector<int>& bucket =
					cell_grid.cell_atoms[cell_grid.Flatten(cx, cy, cz)];
				for (size_t idx = 0; idx < bucket.size(); ++idx) {
					const Vector3& origin = positions[bucket[idx]];
					const double dx_ij = candidate[0] - origin[0];
					const double dy_ij = candidate[1] - origin[1];
					const double dz_ij = candidate[2] - origin[2];
					const double dist_sq = dx_ij * dx_ij + dy_ij * dy_ij + dz_ij * dz_ij;
					if (dist_sq <= cut_off_sq)
						return true;
				}
			}
		}
	}
	return false;
}

bool CanUseOrthorhombicMinimumImage(const Configuration& cfg, double cut_off)
{
	const double eps = 1.0e-12;
	if (cfg.size() == 0)
		return false;
	if (std::abs(cfg.lattice[0][1]) > eps || std::abs(cfg.lattice[0][2]) > eps ||
		std::abs(cfg.lattice[1][0]) > eps || std::abs(cfg.lattice[1][2]) > eps ||
		std::abs(cfg.lattice[2][0]) > eps || std::abs(cfg.lattice[2][1]) > eps)
		return false;
	const double lx = std::abs(cfg.lattice[0][0]);
	const double ly = std::abs(cfg.lattice[1][1]);
	const double lz = std::abs(cfg.lattice[2][2]);
	if (lx <= eps || ly <= eps || lz <= eps)
		return false;
	return std::min(lx, std::min(ly, lz)) > 2.0 * cut_off;
}

void InitNbhs_ConstructNbhs_OrthorhombicMinImage(std::vector<Neighborhood>& nbhs,
												 const Configuration& cfg,
												 double cut_off)
{
	const int size = cfg.size();
	const double cut_off_sq = cut_off * cut_off;
	const double lx = cfg.lattice[0][0];
	const double ly = cfg.lattice[1][1];
	const double lz = cfg.lattice[2][2];

	nbhs.assign(size, Neighborhood());
	for (int i = 0; i < size; ++i)
		nbhs[i].my_type = cfg.type(i);

	for (int i = 0; i < size; ++i) {
		for (int j = i + 1; j < size; ++j) {
			double dx = cfg.pos(j, 0) - cfg.pos(i, 0);
			double dy = cfg.pos(j, 1) - cfg.pos(i, 1);
			double dz = cfg.pos(j, 2) - cfg.pos(i, 2);

			dx -= std::round(dx / lx) * lx;
			dy -= std::round(dy / ly) * ly;
			dz -= std::round(dz / lz) * lz;

			const double dist_sq = dx * dx + dy * dy + dz * dz;
			if (dist_sq >= cut_off_sq)
				continue;

			const double dist_ij = std::sqrt(dist_sq);
			const Vector3 vec_ij(dx, dy, dz);
			AppendNeighbor(nbhs[i], j, vec_ij, dist_ij, cfg.type(j));
			AppendNeighbor(nbhs[j], i, -vec_ij, dist_ij, cfg.type(i));
		}
	}
}

} // namespace

//!	Procedure adds ghost atoms to the configuration according to the periodic extention.
//!	The number of atoms in extended configuration is available via pos_.size().
void Neighborhoods::InitNbhs_AddGhostAtoms(const Configuration& cfg, const double cut_off)
{
	const int size = cfg.size();
	const double cut_off_sq = cut_off * cut_off;
	pos.resize(size);
	memcpy(&pos[0][0], &cfg.pos(0,0), size*sizeof(Vector3));
	const std::vector<Vector3> orig_positions(pos.begin(), pos.begin() + size);

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

	AxisAlignedCellGrid original_grid(orig_positions, cut_off);

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

							if (HasOriginalAtomWithinCutoff(original_grid, orig_positions, candidate, cut_off_sq)) {
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
	const int size = cfg.size();
	const double cut_off_sq = cut_off * cut_off;

	nbhs.assign(size, Neighborhood());
	for (int i = 0; i < size; ++i)
		nbhs[i].my_type = cfg.type(i);

	if (pos.empty())
		return;

	AxisAlignedCellGrid cell_grid(pos, cut_off);
	std::vector<int> candidate_indices;
	candidate_indices.reserve(128);

	// Loop over pairs of atoms using a Cartesian cell grid to avoid full scans.
	for (int i = 0; i < size; ++i) {
		candidate_indices.clear();

		const Vector3int origin_cell = cell_grid.Locate(pos[i]);
		for (int dx = -1; dx <= 1; ++dx) {
			const int cx = origin_cell[0] + dx;
			if (cx < 0 || cx >= cell_grid.dims[0])
				continue;

			for (int dy = -1; dy <= 1; ++dy) {
				const int cy = origin_cell[1] + dy;
				if (cy < 0 || cy >= cell_grid.dims[1])
					continue;

				for (int dz = -1; dz <= 1; ++dz) {
					const int cz = origin_cell[2] + dz;
					if (cz < 0 || cz >= cell_grid.dims[2])
						continue;

					const std::vector<int>& bucket =
						cell_grid.cell_atoms[cell_grid.Flatten(cx, cy, cz)];
					candidate_indices.insert(candidate_indices.end(), bucket.begin(), bucket.end());
				}
			}
		}

		std::sort(candidate_indices.begin(), candidate_indices.end());

		for (size_t idx = 0; idx < candidate_indices.size(); ++idx) {
			const int j = candidate_indices[idx];
			if (j <= i)
				continue;

			const int orig_j = orig_atom_inds[j];
			if (i > orig_j)
				continue;

			const double dx_ij = pos[j][0] - pos[i][0];
			const double dy_ij = pos[j][1] - pos[i][1];
			const double dz_ij = pos[j][2] - pos[i][2];
			const double dist_sq = dx_ij * dx_ij + dy_ij * dy_ij + dz_ij * dz_ij;
			if (dist_sq >= cut_off_sq)
				continue;

			const double dist_ij = std::sqrt(dist_sq);
			const Vector3 vec_ij(dx_ij, dy_ij, dz_ij);

			Neighborhood& neighborhood_i = nbhs[i];
			Neighborhood& neighborhood_j = nbhs[orig_j];

			AppendNeighbor(neighborhood_i, orig_j, vec_ij, dist_ij, cfg.type(orig_j));

			if (i != orig_j) {
				AppendNeighbor(neighborhood_j, i, -vec_ij, dist_ij, cfg.type(i));
			}
		}
	}
}

void Neighborhoods::InitNbhs_RemoveGhostAtoms()
{
	pos.clear();
	orig_atom_inds.clear();
}

void Neighborhoods::InitNbhs(const Configuration& cfg, const double _cutoff)
{
	const auto start_time = std::chrono::steady_clock::now();
	if (nbhs.size() != 0)
		nbhs.clear();

	int ghost_atoms = 0;
	if (cfg.size() != 0) {
		orig_atom_inds.resize(cfg.size());
		for (int i = 0; i < cfg.size(); i++)
			orig_atom_inds[i] = i;

		if (CanUseOrthorhombicMinimumImage(cfg, _cutoff)) {
			InitNbhs_ConstructNbhs_OrthorhombicMinImage(nbhs, cfg, _cutoff);
			ghost_atoms = 0;
			pos.clear();
			orig_atom_inds.clear();
		} else {
			InitNbhs_AddGhostAtoms(cfg, _cutoff);
			InitNbhs_ConstructNbhs(cfg, _cutoff);
			ghost_atoms = static_cast<int>(pos.size()) - cfg.size();
			InitNbhs_RemoveGhostAtoms();
		}
	}

	cutoff = _cutoff;

	const double elapsed_seconds = std::chrono::duration<double>(
		std::chrono::steady_clock::now() - start_time).count();
	if (elapsed_seconds > 2.0) {
		double min_lattice_len = 0.0;
		double max_lattice_len = 0.0;
		if (cfg.lattice.det() != 0.0) {
			min_lattice_len = max_lattice_len = std::sqrt(
				cfg.lattice[0][0] * cfg.lattice[0][0] +
				cfg.lattice[0][1] * cfg.lattice[0][1] +
				cfg.lattice[0][2] * cfg.lattice[0][2]);
			for (int a = 0; a < 3; ++a) {
				const double len = std::sqrt(
					cfg.lattice[a][0] * cfg.lattice[a][0] +
					cfg.lattice[a][1] * cfg.lattice[a][1] +
					cfg.lattice[a][2] * cfg.lattice[a][2]);
				min_lattice_len = std::min(min_lattice_len, len);
				max_lattice_len = std::max(max_lattice_len, len);
			}
		}
		std::cout << "[" << CurrentTimestamp() << "] "
		          << "Neighborhoods slow cfg"
		          << " atoms=" << cfg.size()
		          << " ghosts=" << ghost_atoms
		          << " cutoff=" << _cutoff
		          << " lattice_min=" << min_lattice_len
		          << " lattice_max=" << max_lattice_len
		          << " elapsed_s=" << elapsed_seconds
		          << std::endl;
	}
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
