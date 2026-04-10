/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Alexander Shapeev, Evgeny Podryabinkin
 */

 // TODO: can be optimized through constexpr

#ifndef MLIP_VECTOR3_H
#define MLIP_VECTOR3_H


#include <cmath>
 

template <typename F>
class templateVector3
{
protected:
	F val[3];
public:
	templateVector3() {
		val[0] = val[1] = val[2] = 0;
	};

	template<typename G>
	templateVector3(G x, G y, G z) {
		val[0] = x;
		val[1] = y;
		val[2] = z;
	};

	template<typename G>
	templateVector3(const templateVector3<G>& v) {
		val[0] = v[0];
		val[1] = v[1];
		val[2] = v[2];
	};

	templateVector3(const F _val[3]) {
		for (unsigned int i = 0; i < 3; i++)
				val[i] = _val[i];
	};

	templateVector3(const templateVector3& m) {
		for (unsigned int i = 0; i < 3; i++)
				val[i] = m.val[i];
	};

	~templateVector3(void) {};

	templateVector3& operator=(const templateVector3& m) {
		for (unsigned int i = 0; i < 3; i++)
				val[i] = m.val[i];
		return *this;
	};

	constexpr bool operator==(const templateVector3& m) const {
		return (val[0] == m.val[0]) && (val[1] == m.val[1]) && (val[2] == m.val[2]);
	};

	constexpr bool operator!=(const templateVector3& m) const {
		return !operator==(m);
	};

	templateVector3 operator+(const templateVector3& m) const {
		templateVector3 out_vector;
		for (unsigned int i = 0; i < 3; i++)
			out_vector.val[i] = val[i] + m.val[i];

		return out_vector;
	};

	templateVector3& operator+=(const templateVector3& m) {
		for (unsigned int i = 0; i < 3; i++)
				val[i] += m.val[i];
		return *this;
	};

	templateVector3 operator-() const {
		templateVector3 out_vector;
		for (unsigned int i = 0; i < 3; i++)
			out_vector.val[i] = - val[i];

		return out_vector;
	};

	templateVector3 operator-(const templateVector3& m) const {
		templateVector3 out_vector;
		for (unsigned int i = 0; i < 3; i++)
			out_vector.val[i] = val[i] - m.val[i];

		return out_vector;
	};

	templateVector3& operator-=(const templateVector3& m) {
		for (unsigned int i = 0; i < 3; i++)
				val[i] -= m.val[i];
		return *this;
	};

	templateVector3& operator*=(const F mult) {
		for (unsigned int i = 0; i < 3; i++)
			val[i] *= mult;
		return *this;
	};

	templateVector3 operator*(const F mult) const {
		templateVector3 out_vector;
		for (unsigned int i = 0; i < 3; i++)
			out_vector.val[i] = val[i] * mult;
		return out_vector;
	};

	templateVector3& operator/=(const F mult) {
		for (unsigned int i = 0; i < 3; i++)
			val[i] /= mult;
		return *this;
	};

	templateVector3 operator/(const F mult) const {
		templateVector3 out_vector;
		for (unsigned int i = 0; i < 3; i++)
			out_vector.val[i] = val[i] / mult;
		return out_vector;
	};

	templateVector3& set(const F scalar) {
		for (unsigned int i = 0; i < 3; i++)
			val[i] = scalar;
		return *this;
	};

	F dot(const templateVector3& another) const {
		F scalar_product = 0.0;
		for (unsigned int i = 0; i < 3; i++)
			scalar_product += val[i] * another.val[i];
		return scalar_product;
	};

	friend templateVector3 operator*(const F mult, const templateVector3& vec) {
		templateVector3 out_vector;
		for (unsigned int i = 0; i < 3; i++)
			out_vector.val[i] = vec.val[i] * mult;
		return out_vector;
	};
	
	F& operator[](int ind) {
		return val[ind];
	};

	const F& operator[](int ind) const {
		return val[ind];
	};

	F Norm() const {
		F sum_sq = 0.0;
		for (unsigned int i = 0; i < 3; i++)
			sum_sq += val[i] * val[i];
		return sqrt(sum_sq);
	};

	F NormSq() const {
		F sum_sq = 0.0;
		for (unsigned int i = 0; i < 3; i++)
			sum_sq += val[i] * val[i];
		return sum_sq;
	};

	static templateVector3 VectProd(templateVector3& v1, templateVector3& v2)
	{
		return templateVector3(v1[1]*v2[2]-v1[2]*v2[1], v2[0]*v1[2]-v1[0]*v2[2], v1[0]*v2[1]-v2[0]*v1[1]);
	}
};

//!	Distance between two points
template <typename F>
inline F distance(const templateVector3<F>& x, const templateVector3<F>& y)
{
	return sqrt(pow(x[0] - y[0], 2) + pow(x[1] - y[1], 2) + pow(x[2] - y[2], 2));
}

//!	2-norm of a vector
template <typename F>
inline F norm(const templateVector3<F>& x)
{
	return sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
}



typedef templateVector3<double> Vector3;
typedef templateVector3<int> Vector3int;

#endif //MLIP_VECTOR3_H
