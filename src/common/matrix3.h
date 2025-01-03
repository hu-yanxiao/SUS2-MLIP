/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Alexander Shapeev, Evgeny Podryabinkin, Ivan Novikov
 */

// TODO: can be optimized through constexpr

#ifndef MLIP_MATRIX3_H
#define MLIP_MATRIX3_H


#include "vector3.h"


template<typename F>
class templateMatrix3 {
protected:
	F val[3][3];
public:
	templateMatrix3() {};
	templateMatrix3(const F _val[][3]) {
		for (unsigned int i = 0; i < 3; i++)
			for (unsigned int j = 0; j < 3; j++)
				val[i][j] = _val[i][j];
	}

	templateMatrix3(F v00, F v01, F v02, F v10, F v11, F v12, F v20, F v21, F v22)
	{
		val[0][0] = v00; val[0][1] = v01; val[0][2] = v02;
		val[1][0] = v10; val[1][1] = v11; val[1][2] = v12;
		val[2][0] = v20; val[2][1] = v21; val[2][2] = v22;
	}

	templateMatrix3(const templateMatrix3& m) {
		for (unsigned int i = 0; i < 3; i++)
			for (unsigned int j = 0; j < 3; j++)
				val[i][j] = m.val[i][j];
	}

	~templateMatrix3(void) {};

	bool operator==(const templateMatrix3& m) const {
		for (unsigned int i = 0; i < 3; i++)
			for (unsigned int j = 0; j < 3; j++)
				if (val[i][j] != m.val[i][j]) return false;
		return true;
	};

	bool operator!=(const templateMatrix3& m) const {
		return !operator==(m);
	};

	templateMatrix3& operator=(const templateMatrix3& m) {
		for (unsigned int i = 0; i < 3; i++)
			for (unsigned int j = 0; j < 3; j++)
				val[i][j] = m.val[i][j];
		return *this;
	};

	templateMatrix3 operator+(const templateMatrix3& m) const {
		templateMatrix3 out_matrix;
		for (unsigned int i = 0; i < 3; i++)
			for (unsigned int j = 0; j < 3; j++)
				out_matrix.val[i][j] = val[i][j] + m.val[i][j];

		return out_matrix;
	};

	templateMatrix3& operator+=(const templateMatrix3& m) {
		for (unsigned int i = 0; i < 3; i++)
			for (unsigned int j = 0; j < 3; j++)
				val[i][j] += m.val[i][j];
		return *this;
	};

	templateMatrix3 operator-() const {
		templateMatrix3 out_matrix;
		for (unsigned int i = 0; i < 3; i++)
			for (unsigned int j = 0; j < 3; j++)
				out_matrix.val[i][j] = - val[i][j];

		return out_matrix;
	};

	templateMatrix3 operator-(const templateMatrix3& m) const {
		templateMatrix3 out_matrix;
		for (unsigned int i = 0; i < 3; i++)
			for (unsigned int j = 0; j < 3; j++)
				out_matrix.val[i][j] = val[i][j] - m.val[i][j];

		return out_matrix;
	};

	templateMatrix3& operator-=(const templateMatrix3& m) {
		for (unsigned int i = 0; i < 3; i++)
			for (unsigned int j = 0; j < 3; j++)
				val[i][j] -= m.val[i][j];
		return *this;
	};

	templateMatrix3& operator*=(const templateMatrix3& m) {
		for (unsigned int i = 0; i < 3; i++) {
			F temp_array[3] = { val[i][0], val[i][1], val[i][2]};
			for (unsigned int k = 0; k < 3; k++) {
				val[i][k] = 0;
				for (unsigned int j = 0; j < 3; j++)
					val[i][k] += temp_array[j] * m.val[j][k];
			}
		}
		return *this;
	};

	templateMatrix3 operator*(const F mult) const {
		templateMatrix3 out_matrix;
		for (unsigned int i = 0; i < 3; i++)
			for (unsigned int j = 0; j < 3; j++)
				out_matrix.val[i][j] = val[i][j] * mult;
		return out_matrix;
	};
	friend templateMatrix3 operator*(const double mult, const templateMatrix3 mat) {
		templateMatrix3 out_matrix;
		for (unsigned int i = 0; i < 3; i++)
			for (unsigned int j = 0; j < 3; j++)
				out_matrix.val[i][j] = mat.val[i][j] * mult;
		return out_matrix;
	};

	templateVector3<F> operator*(const templateVector3<F>& vec) const {
		templateVector3<F> out_vect(0, 0, 0);
		for (unsigned int j = 0; j < 3; j++)
			for (unsigned int i = 0; i < 3; i++)
				out_vect[j] += val[j][i] * vec[i];
		return out_vect;
	};

	templateMatrix3 operator*(const templateMatrix3& m) const {
		templateMatrix3 out_matrix(*this);
		return out_matrix *= m;
	};

	friend templateVector3<F> operator*(templateVector3<F> vec, templateMatrix3 mtrx)
	{
		templateVector3<F> out_vect(0,0,0);
		for (unsigned int i = 0; i < 3; i++)
			for (unsigned int j = 0; j < 3; j++)
				out_vect[i] += vec[j] * mtrx.val[j][i];
		return out_vect;
	}

	templateMatrix3& operator*=(const double mult) {
		for (unsigned int i = 0; i < 3; i++)
			for (unsigned int j = 0; j < 3; j++)
					val[i][j] *= mult;
		return *this;
	};

	F* operator[](const int ind) {
		return val[ind];
	};

	const F* operator[](const int ind) const {
		return val[ind];
	};

	constexpr F det() const
	{
		return
			val[0][0] * (val[1][1] * val[2][2] - val[1][2] * val[2][1]) -
			val[0][1] * (val[1][0] * val[2][2] - val[1][2] * val[2][0]) +
			val[0][2] * (val[1][0] * val[2][1] - val[1][1] * val[2][0]);
	};

	//!	Frobenius norm of a matrix
	friend inline double norm(const templateMatrix3& x) {
		return sqrt(
			  x.val[0][0] * x.val[0][0] + x.val[0][1] * x.val[0][1] + x.val[0][2] * x.val[0][2]
			+ x.val[1][0] * x.val[1][0] + x.val[1][1] * x.val[1][1] + x.val[1][2] * x.val[1][2]
			+ x.val[2][0] * x.val[2][0] + x.val[2][1] * x.val[2][1] + x.val[2][2] * x.val[2][2]);
	};

	templateMatrix3 cofactor() const
	{
		return templateMatrix3(
			-(val[1][2] * val[2][1]) + val[1][1] * val[2][2], val[1][2] * val[2][0] - val[1][0] * val[2][2], -(val[1][1] * val[2][0]) + val[1][0] * val[2][1],
			val[0][2] * val[2][1] - val[0][1] * val[2][2], -(val[0][2] * val[2][0]) + val[0][0] * val[2][2], val[0][1] * val[2][0] - val[0][0] * val[2][1],
			-(val[0][2] * val[1][1]) + val[0][1] * val[1][2], val[0][2] * val[1][0] - val[0][0] * val[1][2], -(val[0][1] * val[1][0]) + val[0][0] * val[1][1]
		);
	}

	templateMatrix3 inverse() const
	{
		templateMatrix3 retmtrx(
			val[1][1]*val[2][2] - val[1][2]*val[2][1],
			val[0][2]*val[2][1] - val[0][1]*val[2][2],
			val[0][1]*val[1][2] - val[1][1]*val[0][2],
			val[1][2]*val[2][0] - val[1][0]*val[2][2],
			val[0][0]*val[2][2] - val[0][2]*val[2][0],
			val[0][2]*val[1][0] - val[0][0]*val[1][2],
			val[1][0]*val[2][1] - val[1][1]*val[2][0],
			val[0][1]*val[2][0] - val[0][0]*val[2][1],
			val[0][0]*val[1][1] - val[0][1]*val[1][0]);
		retmtrx*=(1.0/det());
		return retmtrx;
	}

	templateMatrix3 transpose() const
	{
		return templateMatrix3(val[0][0], val[1][0], val[2][0],
						val[0][1], val[1][1], val[2][1],
						val[0][2], val[1][2], val[2][2] );
	}

	void QRdecomp(templateMatrix3& Q, templateMatrix3& R) const
	{
		R[0][0] = sqrt(val[0][0]*val[0][0]+val[1][0]*val[1][0]+val[2][0]*val[2][0]);
		R[1][0] = 0.0;
		R[2][0] = 0.0;
		Q[0][0] = val[0][0]/R[0][0];
		Q[1][0] = val[1][0]/R[0][0];
		Q[2][0] = val[2][0]/R[0][0];

		R[0][1] = val[0][1]*Q[0][0]+val[1][1]*Q[1][0]+val[2][1]*Q[2][0];
		Q[0][1] = val[0][1]-Q[0][0]*R[0][1];
		Q[1][1] = val[1][1]-Q[1][0]*R[0][1];
		Q[2][1] = val[2][1]-Q[2][0]*R[0][1];
		R[1][1] = sqrt(Q[0][1]*Q[0][1]+Q[1][1]*Q[1][1]+Q[2][1]*Q[2][1]);
		R[2][1] = 0.0;
		Q[0][1] /= R[1][1];
		Q[1][1] /= R[1][1];
		Q[2][1] /= R[1][1];

		R[0][2] = val[0][2]*Q[0][0]+val[1][2]*Q[1][0]+val[2][2]*Q[2][0];
		R[1][2] = val[0][2]*Q[0][1]+val[1][2]*Q[1][1]+val[2][2]*Q[2][1];
		Q[0][2] = val[0][2]-Q[0][0]*R[0][2]-Q[0][1]*R[1][2];
		Q[1][2] = val[1][2]-Q[1][0]*R[0][2]-Q[1][1]*R[1][2];
		Q[2][2] = val[2][2]-Q[2][0]*R[0][2]-Q[2][1]*R[1][2];
		R[2][2] = sqrt(Q[0][2]*Q[0][2]+Q[1][2]*Q[1][2]+Q[2][2]*Q[2][2]);
		Q[0][2] /= R[2][2];
		Q[1][2] /= R[2][2];
		Q[2][2] /= R[2][2];
	}

	static const templateMatrix3 Id() { return templateMatrix3(1, 0, 0, 0, 1, 0, 0, 0, 1); };
	double NormFrobeniusSq() const {
		return val[0][0] * val[0][0] + val[0][1] * val[0][1] + val[0][2] * val[0][2]
			+ val[1][0] * val[1][0] + val[1][1] * val[1][1] + val[1][2] * val[1][2]
			+ val[2][0] * val[2][0] + val[2][1] * val[2][1] + val[2][2] * val[2][2];
	};
	double MaxAbs() const {
		double ret = 0;
		for (int a = 0; a < 3; a++)
			for (int b = 0; b < 3; b++)
				ret = __max(ret, std::abs(val[a][b]));
		return ret;
	};
};


typedef templateMatrix3<double> Matrix3;

#endif //MLIP_MATRIX3_H
