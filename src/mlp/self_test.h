/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Alexander Shapeev, Evgeny Podryabinkin, Ivan Novikov, Konstantin Gubaev
 */

#ifndef MLIP_UTILS_SELF_TEST_H
#define MLIP_UTILS_SELF_TEST_H

//	Uncomment for generation of reference results
//#define GEN_TESTS

bool RunAllTests(bool is_parallel);

#endif // SELF_TEST
