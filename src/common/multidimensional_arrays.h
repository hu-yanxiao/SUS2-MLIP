/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin
 */

#ifndef MLIP_MULTIDIMENSIONALARRAYS_H
#define MLIP_MULTIDIMENSIONALARRAYS_H


#include "utils.h"


template<typename F>
class templateArray2D
{
private:
	std::vector<F> holder;

public:
	int size1;
	int size2;

	templateArray2D() : size1(0), size2(0) {};
	templateArray2D(int _size1, int _size2)
	{
		resize(_size1, _size2);
	};
	~templateArray2D()
	{
		holder.clear();
	};
	void reserve(int max_length)
	{
		holder.reserve(max_length);
	};
	void resize(int _size1, int _size2)
	{
#ifdef MLIP_DEBUG
		if (_size1 < 0 || _size2 < 0)
			ERROR("Array2D::resize(): invalid size");
#endif
		size1 = _size1;
		size2 = _size2;

		holder.resize(size1*size2);

		//set(0);
	};
	templateArray2D<F>& operator=(templateArray2D<F>& another)
	{
		if (size1!=another.size1 || size2!=another.size2)
			resize(another.size1, another.size2);
		memcpy(holder.data(), another.holder.data(), holder.size()*sizeof(F));
		return (*this);
	}
	inline F& operator() (int i, int j)            ///!! should be F instead of B<double>
	{
#ifdef MLIP_DEBUG
		if (i < 0 || j < 0 || i >= size1 || j >= size2)
			ERROR("Array2D::operator(): error index value(s)");
		if (i*size2 + j > holder.size())
			ERROR("Array2D::operator(): inconsistent state detected");
#endif
		return holder[i*size2 + j];
	};
	inline void set(F val)
	{
		if (val == 0)
			FillWithZero(holder);
		else
			for (int i = 0; i < size1*size2; i++)
				holder[i] = val;
	};

};

template<typename F>
class templateArray3D
{
private:
	std::vector<F> holder;
	int size2_size3;

public:
	int size1;
	int size2;
	int size3;

	templateArray3D() : size1(0), size2(0), size3(0) {};
	templateArray3D(int _size1, int _size2, int _size3)
	{
		resize(_size1, _size2, _size3);
	};
	~templateArray3D()
	{
		holder.clear();

	};
	void reserve(int max_length)
	{
		holder.reserve(max_length);
	};
	void resize(int _size1, int _size2, int _size3)
	{
#ifdef MLIP_DEBUG
		if (_size1<0 || _size2<0 || _size3<0)
			ERROR("Array3D::resize(): invalid size");
#endif
		size1 = _size1;
		size2 = _size2;
		size3 = _size3;

		size2_size3 = size2*size3;
		holder.resize(size1*size2*size3);

		//set(0);
	};
	templateArray3D<F>& operator=(templateArray3D<F>& another)
	{
		if (size1!=another.size1 || size2!=another.size2 || size3!=another.size3)
			resize(another.size1, another.size2, another.size3);
		memcpy(holder.data(), another.holder.data(), holder.size()*sizeof(F));
		return (*this);
	}
	inline F& operator() (int i, int j, int k)            ///!! should be F instead of B<double>
	{
#ifdef MLIP_DEBUG
		if (i<0 || j<0 || k<0 || i>=size1 || j>=size2 || k>=size3)
			ERROR("Array3D::operator(): error index value(s)");
		if (i*size2_size3 + j*size3 + k > holder.size())
			ERROR("Array3D::operator(): inconsistent state detected");
#endif
		return holder[i*size2_size3 + j*size3 + k];
	};
	inline void set(F val)
	{
		if (val == 0)
			FillWithZero(holder);
		else
			for (int i=0; i<size1*size2_size3; i++)
				holder[i] = val;
	};

};

template<typename F>
class templateArray4D
{
private:
	std::vector<F> holder;
	int size3_size4;
	int size2_size3_size4;

public:
	int size1;
	int size2;
	int size3;
	int size4;

	templateArray4D() : size1(0), size2(0), size3(0), size4(0) {};
	templateArray4D(int _size1, int _size2, int _size3, int _size4)
	{
		resize(_size1, _size2, _size3, _size4);
	};
	~templateArray4D()
	{
		holder.clear();

	};
	void reserve(int max_length)
	{
		holder.reserve(max_length);
	};
	void resize(int _size1, int _size2, int _size3, int _size4)
	{
#ifdef MLIP_DEBUG
		if (_size1<0 || _size2<0 || _size3<0 || _size4<0 || 
			(_size1 + _size2 + _size3 + _size4 < 1))
			ERROR("Array4D::resize(): invalid size");
#endif
		size1 = _size1;
		size2 = _size2;
		size3 = _size3;
		size4 = _size4;

		size3_size4 = size3*size4;
		size2_size3_size4 = size2*size3_size4;
		holder.resize(size1*size2_size3_size4);

		//set(0);
	};
	templateArray4D<F>& operator=(templateArray4D<F>& another)
	{
		if (size1!=another.size1 || 
			size2!=another.size2 || 
			size3!=another.size3 || 
			size4!=another.size4)
			resize(another.size1, another.size2, another.size3, another.size4);
		memcpy(holder.data(), another.holder.data(), holder.size()*sizeof(F));
		return (*this);
	}
	inline F& operator() (int i, int j, int k, int l)           
	{
#ifdef MLIP_DEBUG
		if (i<0 || j<0 || k<0 || l<0 || 
			i>=size1 || j>=size2 || k>=size3 || l>=size4)
			ERROR("Array4D::operator(): error index value(s)");
		if (i*size2_size3_size4 + j*size3_size4 + k*size4 + l > holder.size())
			ERROR("Array4D::operator(): inconsistent state detected");
#endif
		return holder[i*size2_size3_size4 + j*size3_size4 + k*size4 + l];
	};
	inline void set(F val)
	{
		if (val == 0)
			FillWithZero(holder);
		else
			for (int i=0; i<size1*size2_size3_size4; i++)
				holder[i] = val;
	};

};

typedef std::vector<double>		Array1D;
typedef templateArray2D<double> Array2D;
typedef templateArray3D<double> Array3D;
typedef templateArray4D<double> Array4D;

#endif //MLIP_MULTIDIMENSIONALARRAYS_H
