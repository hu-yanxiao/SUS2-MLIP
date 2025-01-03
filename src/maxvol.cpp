/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin
 */

#include "maxvol.h"

#ifdef MLIP_INTEL_MKL
#include <mkl_cblas.h>
#else 
#include <cblas.h>
#endif

#include <sstream>

using namespace std;

const char* MaxVol::tagname = {"MaxVol"};

void MaxVol::Reset(int _n, double _init_treshold) 
{
	n = _n;
	init_treshold = _init_treshold;

	A.resize(n, n);
	A.set(0);
	for (int i = 0; i < n; i++)
		A(i, i) = init_treshold;

	invA.resize(n, n);
	invA.set(0);
	for (int i=0; i<n; i++)
		invA(i, i) = 1.0/init_treshold;

	B.resize(0, n);
	BinvA.resize(1, n);

	bufVect1.resize(n);
	bufVect2.resize(n);
	bufVect3.resize(n);
	//bufMtrx.resize(n, n);
	
	log_n_det = n*log(init_treshold)/log(n);
}

MaxVol::MaxVol(int _n, double _init_treshold)
{
	Reset(_n, _init_treshold);
}

MaxVol::~MaxVol()
{
}

double MaxVol::FindMaxElem(Array2D& Mtrx_, int r_count)
{
	int ind = (int)cblas_idamax(r_count*Mtrx_.size2, &Mtrx_(0, 0), 1);
	i0 = ind / Mtrx_.size2;
	j0 = ind % Mtrx_.size2;
	return fabs(Mtrx_(i0, j0));
}

void MaxVol::UpdateInvA()
{
	// 0. bufVect1 = B[i0][:] - A[j0][:]
	memcpy(&bufVect1[0], &B(i0, 0), n*sizeof(double));
	cblas_daxpy(n, -1, &A(j0, 0), 1, &bufVect1[0], 1);

	// SMW - update:
	// 1. tmp = (1 + v^T * invA * q) = 1.0/BinvA[i0][j0]
	double tmp = 1.0/BinvA(i0, j0);
	// 2. &invA[j0][0] - j0-th column of A^-1
	memcpy(&bufVect2[0], &invA(j0, 0), n*sizeof(double));
	// 3. tmp * bufVect1^T * invA
	cblas_dgemv(CBLAS_ORDER::CblasRowMajor/*sic!*/, CBLAS_TRANSPOSE::CblasNoTrans, 
				n, n, tmp, &invA(0, 0), n, &bufVect1[0], 1, 0, &bufVect3[0], 1);
	// 4. update invA
	cblas_dger(CBLAS_ORDER::CblasColMajor, n, n, -1.0, 
				&bufVect2[0], 1, &bufVect3[0], 1, &invA(0, 0), n);
}

void MaxVol::UpdateBinvA(int m)
{
	// 0. remember BinvA[i0][j0];
	double b_i0j0 = BinvA(i0, j0);
	// 1. vector-column formation
	bufVect1.resize(__max(n, m));
	FillWithZero(bufVect1);
	cblas_daxpy(m, 1.0/b_i0j0, &BinvA(0, j0), n, &bufVect1[0], 1);
	bufVect1[i0] += 1.0/b_i0j0;
	// 2. vector-row formation
	memcpy(&bufVect2[0], &BinvA(i0, 0), n*sizeof(double));
	bufVect2[j0] -= 1.0;
	// 3. BinvA update
	cblas_dger(CBLAS_ORDER::CblasRowMajor, m, n, -1.0,
		&bufVect1[0], 1, &bufVect2[0], 1, &BinvA(0, 0), n);
}

void MaxVol::SwapRows()
{
	cblas_dswap(n, &A(j0, 0), 1, &B(i0, 0), 1);
}

//void MaxVol::RecalculateInvA()
//{
//	int report;
//	int nxn = n*n;
//	std::vector<int> ipiv(n);
//
//	cblas_domatcopy(CBLAS_ORDER::CblasRowMajor, CBLAS_TRANSPOSE::CblasTrans, 
//		n, n, 1, &A(0,0), n, &invA(0,0), n);
//	LAPACK_dgetrf(&n, &n, &invA(0, 0), &n, ipiv.data(), &report);
//	LAPACK_dgetri(&n, &invA(0, 0), &n, ipiv.data(), &bufMtrx(0, 0), &nxn, &report);
//
//	if (report != 0)
//		ERROR("Maxvol: failed to inverse matrix");
//}

double MaxVol::Iterate(double& MaxModElem, int m)
{
	UpdateInvA();
	UpdateBinvA(m);
	SwapRows();
	swap_tracker.emplace_back(i0, j0);

	MaxModElem = FindMaxElem(BinvA, m);

	return MaxModElem;
}

int MaxVol::InitedRowsCount()
{
	if (init_treshold < 0)	// init_treshold = -1 if all rows inited
		return n;

	int cntr = 0;
	for (int i=0; i<n; i++)
		if (A(i, i) != init_treshold)	// only diagonal elements checked. It is not honest but saves performance
			cntr++;

	if (cntr == n)
		init_treshold = -1;	// It means that all rows inited

	return cntr;
}

double MaxVol::CalcSwapGrade(int m)
{
	if (BinvA.size1 < m)
		BinvA.resize(m, n);

	swap_tracker.clear();

	cblas_dgemm(CBLAS_ORDER::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasTrans,
				m, n, n, 1, &B(0, 0), n, &invA(0, 0), n, 0, &BinvA(0, 0), n);

	swap_grade = FindMaxElem(BinvA, m);

// Obsolete: to be removed
//	if (InitedRowsCount() != n)
//	if (A(j0, j0)==init_treshold && swap_grade>1.0)	// only diagonal elements checked. It is not honest but saves performance
//			swap_grade = 9.9e99;						// this means that it is initialization swap and must be done regardless of selection treshold value

  std::stringstream logstrm1;
	logstrm1 << "\rMaxVol: call# " << ++eval_cntr
			<< '\t' << m << 'x' << n 
			<< " matrix. First swap grade " << swap_grade << endl;
  MLP_LOG(tagname,logstrm1.str());

	return swap_grade;
}

int MaxVol::MaximizeVol(double threshold, int m, int swap_limit)
{
	if (i0==-1 || j0==-1)
		CalcSwapGrade(m);

  std::stringstream logstrm1;
	for (int swap_cntr = 0; swap_grade > threshold && swap_cntr < swap_limit; swap_cntr++)
	{
		log_n_det += log(swap_grade)/log(n);

		Iterate(swap_grade, m);

		logstrm1 << "\rMaxVol: " << swap_cntr+1 << " swaps. Inited "
				<< InitedRowsCount() << "/" << n << ". log(det): "
				<< log_n_det << "\t";
    MLP_LOG(tagname,logstrm1.str()); logstrm1.str("");
	}

	if (!swap_tracker.empty())
		logstrm1 << endl;
    MLP_LOG(tagname,logstrm1.str());

	i0=-1, j0=-1;
	swap_grade = 0.0;

	return (int)swap_tracker.size();
}

void MaxVol::WriteData(std::ofstream & ofs)
{
	ofs.write((char*)&A(0, 0), A.size1*A.size2*sizeof(double));
	ofs.write((char*)&invA(0, 0), invA.size1*invA.size2*sizeof(double));
}

void MaxVol::ReadData(std::ifstream & ifs)
{
	ifs.read((char*)&A(0, 0), A.size1*A.size2*sizeof(double));
	ifs.read((char*)&invA(0, 0), invA.size1*invA.size2*sizeof(double));
	if (ifs.fail())
		ERROR("MaxVol::ReadData(): Error loading matrices");
}


