/*
 * Vectors.h
 *
 *  Created on: Mar 20, 2013
 *      Author: ilya
 */

#ifndef VECTORS_H_
#define VECTORS_H_

#include "Abstract.h"
#include "Utils.h"
#include <vector>
#include <exception>
#include <cmath>
#include <iostream>

//#define TESTING

namespace HMMPI
{
//------------------------------------------------------------------------------------------
// template for 2D vectors (= 2D arrays)
template <class T>
class Vector2
{
protected:
	std::vector<T> data;				// row-major storage: (i, j) <-> data[i*jcount + j]
	size_t icount, jcount;
public:
	Vector2(){icount = jcount = 0;};	// default constructor - empty std::vector
	Vector2(size_t I, size_t J);		// initialize I x J std::vector, allocate memory
	Vector2(size_t I, size_t J, const T &val);				// initialize I x J std::vector, allocate memory, set all elements to 'val'
	Vector2(std::vector<T> v, size_t I, size_t J);			// initialize I x J std::vector, copy/move data from "v"
	Vector2(const Vector2<T> &V);		// copy constructor
	Vector2(Vector2<T> &&V) noexcept;	// move constructor
	virtual ~Vector2(){};
	const Vector2 &operator=(const Vector2<T> &V);			// copy =
	const Vector2 &operator=(Vector2<T> &&V) noexcept;		// move =
	virtual T &operator()(size_t i, size_t j);				// element (i, j)
	virtual const T &operator()(size_t i, size_t j) const;	// const element (i, j)
	size_t ICount() const {return icount;};
	size_t JCount() const {return jcount;};
	size_t Length() const {return icount*jcount;};
	const T *Serialize() const;				// returns underlying native array
	virtual void Deserialize(const T *v);	// inverse operation to above; current "icount", "jcount" are used for size
	static void Copy(const Vector2<T> &Src, size_t i1, size_t i2, size_t j1, size_t j2,
					      Vector2<T> &Dest, size_t x1, size_t x2, size_t y1, size_t y2);
				// copy from subset of one std::vector to subset of another: [i1, i2)x[j1, j2) -> [x1, x2)x[y1, y2)
};
//------------------------------------------------------------------------------------------
// implementation of TEMPLATE FUNCTIONS
//------------------------------------------------------------------------------------------
template <class T>
Vector2<T>::Vector2(size_t I, size_t J) : icount(I), jcount(J)
{
	data = std::vector<T>(I*J);
}
//------------------------------------------------------------------------------------------
template <class T>
Vector2<T>::Vector2(size_t I, size_t J, const T &val) : Vector2<T>(I, J)	// C++11 delegating constructors
{
	size_t SZ = I*J;
	for (size_t i = 0; i < SZ; i++)
		data[i] = val;
}
//------------------------------------------------------------------------------------------
template <class T>
Vector2<T>::Vector2(std::vector<T> v, size_t I, size_t J)
{
	if (v.size() != I*J)
		throw HMMPI::Exception("Неправильные размеры векторов в конструкторе Vector2",
							   "Incorrect size in Vector2 constructor");

	icount = I;
	jcount = J;

	data = std::move(v);

#ifdef TESTING
	std::cout << "Vector2<T>::Vector2(std::vector<T> v, size_t I, size_t J), std::move(v)" << std::endl;
#endif
}
//------------------------------------------------------------------------------------------
template <class T>
Vector2<T>::Vector2(const Vector2<T> &V)
{
	*this = V;
}
//------------------------------------------------------------------------------------------
template <class T>
Vector2<T>::Vector2(Vector2<T> &&V) noexcept
{
	*this = std::move(V);
}
//------------------------------------------------------------------------------------------
template <class T>
const Vector2<T> &Vector2<T>::operator=(const Vector2<T> &V)
{
	if (this != &V)
	{
		icount = V.icount;
		jcount = V.jcount;
		data = V.data;
	}
	return *this;
}
//------------------------------------------------------------------------------------------
template <class T>
const Vector2<T> &Vector2<T>::operator=(Vector2<T> &&V) noexcept
{
	if (this != &V)
	{
		icount = V.icount;
		jcount = V.jcount;
		data = std::move(V.data);

		V.icount = 0;
		V.jcount = 0;
	}
	return *this;
}
//------------------------------------------------------------------------------------------
template <class T>
T &Vector2<T>::operator()(size_t i, size_t j)
{
	return data[i*jcount + j];
}
//------------------------------------------------------------------------------------------
template <class T>
const T &Vector2<T>::operator()(size_t i, size_t j) const
{
	return data[i*jcount + j];
}
//------------------------------------------------------------------------------------------
template <class T>
const T *Vector2<T>::Serialize() const
{
	return data.data();
}
//------------------------------------------------------------------------------------------
template <class T>
void Vector2<T>::Deserialize(const T *v)
{
	data = std::vector<T>(v, v + icount*jcount);
}
//------------------------------------------------------------------------------------------
template <class T>
void Vector2<T>::Copy(const Vector2<T> &Src, size_t i1, size_t i2, size_t j1, size_t j2,
				           Vector2<T> &Dest, size_t x1, size_t x2, size_t y1, size_t y2)
{
	if (i1 >= i2 || j1 >= j2 || x1 >= x2 || y1 >= y2)
		throw HMMPI::Exception("Индексы в диапазоне (1,2) должны увеличиваться в Vector2::Copy",
							   "Range indices (1,2) should be increasing in Vector2::Copy");
	if (i1 < 0 || i2 > Src.icount || j1 < 0 || j2 > Src.jcount ||
		x1 < 0 || x2 > Dest.icount || y1 < 0 || y2 > Dest.jcount)
		throw HMMPI::Exception("Индексы вне диапазона в 2-мерном массиве в Vector2::Copy",
							   "Range indices out of 2D array index range in Vector2::Copy");
	if (i2-i1 != x2-x1 || j2-j1 != y2-y1)
		throw HMMPI::Exception("Исходный и конечный диапазоны имеют разный размер в Vector2::Copy",
							   "Source and destination ranges have different size in Vector2::Copy");
	for (size_t I = i1; I < i2; I++)
		for (size_t J = j1; J < j2; J++)
			Dest.data[(I-i1+x1)*Dest.jcount + J-j1+y1] = Src.data[I*Src.jcount + J];
}
//------------------------------------------------------------------------------------------
}	// namespace HMMPI

#endif /* VECTORS_H_ */
