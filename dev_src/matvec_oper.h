/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */

#include "../src/common/stdafx.h"

void orthoproject(int, double *);
double vecvec(const int, const double *, const double *);
void matvec(const int, const double *, const double *, double *);
void matmat(const int, const double *, const double *, double *);
void gen_matr(const int, const double *, const double *, double *);

