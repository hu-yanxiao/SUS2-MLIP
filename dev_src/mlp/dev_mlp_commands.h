/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Alexander Shapeev, Evgeny Podryabinkin, Ivan Novikov
 */

#ifndef MLIP_DEV_MLP_COMMANDS_H
#define MLIP_DEV_MLP_COMMANDS_H

bool DevCommands(const std::string& command, std::vector<std::string>& args, std::map<std::string, std::string>& opts);

#endif // MLIP_DEV_MLP_COMMANDS_H