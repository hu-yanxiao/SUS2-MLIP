/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Alexander Shapeev, Evgeny Podryabinkin
 */

#ifndef MLIP_LINEAR_REGRESSION_H
#define MLIP_LINEAR_REGRESSION_H


#include "mtp.h"
#include "basic_trainer.h"


// class for training linear potentials
class LinearRegression : public AnyTrainer
{
  private:
    double* p_matrix1;                      // The first of two replicas of SLAE matrix. One of two SLAE is required to SLAE storing, the second one - for SLAE solving procedure which modify it
    double* p_matrix2;                      // The second of two replicas of SLAE matrix. One of two SLAE is required to SLAE storing, the second one - for SLAE solving procedure which modify it
    double* p_vector1;                      // The first of two replicas of SLAE right parts. One of two SLAE is required to SLAE storing, the second one - for SLAE solving procedure which modify it
     double* p_vector2;                     // The second of two replicas of SLAE right parts. One of two SLAE is required to SLAE storing, the second one - for SLAE solving procedure which modify it

  protected:
    static const char* tagname;             // tag name of object
  public:
    double* quad_opt_matr;                  // Pointer to "active" (either equals p_matrix1 or p_matrix2) SLAE matrix (which is modified and being solved) of least squres minimization problem A*A^T = b*A^T
    double* quad_opt_vec;                   // Pointer to "active" (either equals p_vector1 or p_vector2) SLAE right part (which is modified and being solved) of least squres minimization problem A*A^T = b*A^T
    double quad_opt_scalar;                 // Scalar value in least squares minimization problem A*A^T = b*A^T (Not needed in mos scenarios)
    int quad_opt_eqn_count;                 // Number of equation in overdetermined system

    Array1D energy_cmpnts;                  // CoeffCount() array
    Array3D forces_cmpnts;                  // cfg.size()x3xCoeffCount() array
    Array3D stress_cmpnts;                  // 3x3xCoeffCount() array

    MTP* p_mtp;                             // = p_mlip

  protected:
    void CopySLAE();                        // Copying of SLAE Matrix and right part betweenMtrx1, p_vector1 and p_matrix2, p_vector2 before it will be destroyed during solving
    void RestoreSLAE();                     // Switches quad_opt_matr and quad_opt_vec to betweenMtrx1, p_vector1 or Mtrx2, p_vector2 in order to restore broken SLAE after solving
    void SymmetrizeSLAE();                  // Symmetrization of the SLAE matrix before solving (only upper right part is filled during adding or removing configuration to regression)
    void SolveSLAE();

  private:
    LinearRegression() {}
  public:
    LinearRegression(AnyLocalMLIP* _p_mtp,  // Constructor requires MTP basis
                     double _wgt_eqtn_energy = 1.0,//    Optional parameters are the weights coeficients of energy, forces and stresses equations in minimization problem
                     double _wgt_eqtn_forces = 0.001,
                     double _wgt_eqtn_stress = 0.1,
                     double _wgt_rel_forces = 0.0);
    ~LinearRegression();

    void ClearSLAE();                       // Setting SLAE Matrix and right part to zero
    void AddToSLAE(Configuration& cfg, double weight = 1); // Adds configuration to regression SLAE. If weight = -1 removes from regression
    void RemoveFromSLAE(Configuration& cfg); // Removes configuration from SLAE
    void JoinSLAE(LinearRegression& to_add); // Adds SLAE of another object with the same p_mlip

    void Train(); // SLAE solving with restoring after solving. Returns true if SLAE is of full rank, otherwise returns false and doesn't solve SLAE
    void Train(std::vector<Configuration>& training_set) override; // Finds regression coefficients minimizing r.m.s. residual for all configuration in TrainingSet. Returns true if SLAE is of full rank, otherwise returns false and doesn't solve SLAE
};

#endif // MLIP_LINEAR_REGRESSION
