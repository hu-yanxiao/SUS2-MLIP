/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin
 */

#ifndef MLIP_MAXVOL_H
#define MLIP_MAXVOL_H

#include "common/stdafx.h"
#include "common/multidimensional_arrays.h"


// Object for maximization of |det(A)| (volume) by swaping its rows with the rows of matrix B. Can maximize volume multiple times for a number of different B
class MaxVol  //: public LogWriting
{
private:
	Array2D invA;				// column-ordered 
	Array2D BinvA;				// row-ordered 

	// temporal arrays
	Array1D bufVect1;		
	Array1D bufVect2;
	Array1D bufVect3;
	Array2D bufMtrx;

	int i0=-1;					// row index of maxmod element in B (0<=i0<=m)
	int j0=-1;					// column index of maxmod element in B (0<=j0<=n)
	double swap_grade;			// temporal for swap grade storage

	double init_treshold;		// value for A initilization (A = IdentityMatrix * init_treshold)
	
	unsigned int eval_cntr = 0;	// counts BinvA calculations

protected:
  static const char* tagname;                           // tag name of object
public:
	int n = 0;					// size of A is n x n. Read only

	Array2D A;					// Matrix of maximal volume	
	Array2D B;					// Matrix which rows to maximize volume

	std::vector<std::pair<int, int>> swap_tracker;	// stores swap history for the last call of MaximizeVol(). The first is row index in A, the second is row index in B

	double log_n_det;			// stores logarithm of |det(A)| (of base n)

private:
	void UpdateInvA();
	void UpdateBinvA(int m);
	void SwapRows();

	double FindMaxElem(Array2D& Mtrx_, int rCount = 0);
	double Iterate(double& MaxModElem, int m);

public:
	MaxVol() {};
	MaxVol(int _n, double _init_treshold = 1.0e-5);
	~MaxVol();
	void Reset(int _n, double _init_treshold = 1.0e-5);					//!< Resets maxvol object 

	int InitedRowsCount();												//!< Returns the number of inited rows in A
	//void RecalculateInvA();											//!< Not actually required

	double CalcSwapGrade(int m);										//!< Calculates BinvA and finds maximal in modules element in it (in the first m rows). BinvA is sotored and can be used in MaximizeVol until MaximizeVol is called.
	int MaximizeVol(double threshold, int m, int swap_limit=HUGE_INT);	//!< Main function maximizing |det(A)| by swapping its rows with rows of B. Does not calculate BinvA if CalcSwapGrade() called before (use BinvA calulated in CalcSwapGrade()).

	void WriteData(std::ofstream& ofs);									//!< Writes A and invA in binary format. Required for Saving active learning
	void ReadData(std::ifstream& ifs);									//!< Reads A and invA in binary format. Required for Saving active learning
};

#endif //MLIP_MAXVOL_H
