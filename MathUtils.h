/*
 * MathUtils.h
 *
 *  Created on: 16 May 2016
 *      Author: ilya fursov
 */

#ifndef MATHUTILS_H_
#define MATHUTILS_H_

#include <cassert>
#include <string>
#include <vector>
#include <cstdio>
#include <functional>
#include <algorithm>
#include <numeric>
#include <set>
#include "Vectors.h"
#include "mpi.h"

namespace tlib
{
	template<class value_t>
	class tensor;
}

namespace HMMPI
{

enum OH {OH2, OH4};
//------------------------------------------------------------------------------------------
// some functions
inline double Min(double x, double y){return (x < y) ? x : y;};
inline double Max(double x, double y){return (x > y) ? x : y;};
double NumD(const std::function<double(double)> &f, double x, double h = 1e-4, OH oh = OH2);		// numerical derivative df/dx; h - increment, oh - precision
double NumD2(const std::function<double(double)> &f, double x, double h = 1e-4, OH oh = OH2);		// numerical derivative d2f/dx2; h - increment, oh - precision
double NumD3(const std::function<double(double)> &f, double x, double h = 1e-4, OH oh = OH2);		// numerical derivative d3f/dx3; h - increment, oh - precision

class Func1D_CDF;
double integr_Gauss(const std::function<double(double)> &g, int n, double x0, double mu, double sigma);		// calculate int_{x0...+inf} g(x)p(x)dx, where p = PDF Normal(mu, sigma^2), using "n" integration intervals with trapezoid rule
double integr_Gauss(const std::function<double(double)> &g, int n, double x0, double mu, double sigma, const Func1D_CDF &F);	// similar to above, with user-defined CDF F, employing Normal score transform:
																											// int_{invP0(F(x0))...+inf} g(invF(P0(y)))p(y)dy, where p = PDF Normal(mu, sigma^2), P0 is Standard Normal CDF
bool IsNaN(double d);
double _sqrt(double d);
void Sobol(long long int &seed, std::vector<double> &vec);	// generates a new quasirandom Sobol vector with each call; it's a wrapper for "i8_sobol"; dimension is taken according to the vec.size(); 'seed' is incremented with each call
void CholDecomp(const double *A, double *res, int sz);		// Cholesky decomp. A[sz,sz] -> res[sz,sz], A should be symmetric positive definite; for row-major storage of A, only lower triangular part of A is used
															// array 'res' should have been allocated
void Bcast_string(std::string &s, int root, MPI_Comm comm);			// MPI broadcast std::string from 'root' rank; memory allocation is done if needed
void Bcast_vector(std::vector<double> &v, int root, MPI_Comm comm);		// MPI broadcast vector<double> from 'root' rank; memory allocation is done if needed
void Bcast_vector(std::vector<int> &v, int root, MPI_Comm comm);		// MPI broadcast vector<int> from 'root' rank; memory allocation is done if needed
void Bcast_vector(std::vector<size_t> &v, int root, MPI_Comm comm);		// MPI broadcast vector<size_t> from 'root' rank; memory allocation is done if needed
void Bcast_vector(std::vector<char> &v, int root, MPI_Comm comm);		// MPI broadcast vector<char> from 'root' rank; memory allocation is done if needed
void Bcast_vector(std::vector<std::vector<double>> &v, int root, MPI_Comm comm);	// MPI broadcast vector of vectors from 'root' rank; memory allocation is done if needed
void Bcast_vector(std::vector<std::vector<std::vector<double>>> &v, int root, MPI_Comm comm);	// MPI broadcast vector^3 from 'root' rank; memory allocation is done if needed
void Bcast_vector(double **v, int len1, int len2, int root, MPI_Comm comm);			// MPI broadcast vector of vectors v[len1][len2] from 'root' rank; memory should be allocated IN ADVANCE
void Bcast_vector(std::vector<std::string> &v, int root, MPI_Comm comm);			// MPI broadcast vector<string> from 'root' rank; memory allocation is done if needed

namespace ManualMath
{
	double InnerProd(const std::vector<double> &a, const std::vector<double> &b);				// scalar product (a, b), manual
}
double InnerProd(const std::vector<double> &a, const std::vector<double> &b);					// scalar product (a, b), BLAS

std::vector<double> Vec_x_ay(std::vector<double> x, const std::vector<double> &y, double a);	// x + ay
std::vector<double> Vec_ax(std::vector<double> x, double a);									// ax

double Vec_pow_multiind(const std::vector<double> &v, const std::vector<int> &mi, int di = -1, int dj = -1);	// for di == dj == -1, returns v^mi, or PROD(v[mi[k]]), where "mi" is a multi-index
																					// in general, returns d2/(di*dj)[v^mi], or PROD(v[mi[k\(di,dj)]]) -- derivatives (without coefficients) of order 0, 1, 2
																					// mi[k] are indices for "v"; di, dj are indices for "mi", with '-1' meaning no differentiation
//------------------------------------------------------------------------------------------
// some template functions
template <class RandomAccessIterator>
std::vector<size_t> SortPermutation(RandomAccessIterator first, RandomAccessIterator last);	// sorts [first, last) - not via modification, but by returning the indices of permutation

template <class FwdIterator, class T>
FwdIterator FindBinary(FwdIterator first, FwdIterator last, const T &val);			// binary search of "val" in a SORTED range [first, last); returns iterator to the first found element == "val", returns "last" if not found

template <class T>
std::vector<std::vector<T>> SqueezeVec(std::vector<std::vector<T>> v);				// returns a copy of "v" with empty v[i]'s removed

template <class T>
std::vector<T> Reorder(const std::vector<T> &v, const std::vector<size_t> &ord);	// creates vector from elements of "v" with indices from "ord" (indices may be repeated)

template <class T>																		// treats "v" as row-major storage of M x N matrix, extracts ordi.size() x ordj.size() sub-matrix with indices "ordi", "ordj" (indices may be repeated)
std::vector<T> Reorder(const std::vector<T> &v, size_t M, size_t N, const std::vector<size_t> &ordi, const std::vector<size_t> &ordj, bool skipneg = false, T defval = T());	// returns its row-major storage vector
																						// if 'skipneg' == true, indices ordi, ordj equal to -1 will be populated with 'defval'
template <class T>																					// returns vector of indices of "subvec" elements within "mainvec" elements (only first encounter);
std::vector<size_t> GetSubvecInd(const std::vector<T> &mainvec, const std::vector<T> &subvec);	 	// index is set to -1 if element is not found in "mainvec"; result.size = subvec.size

template <class T>
std::vector<size_t> GetSubvecIndSorted(const std::vector<T> &mainvec_sorted, const std::vector<T> &subvec);	 	// same as GetSubvecInd, but "mainvec_sorted" should be a sorted vector; uses binary search

template <class T>																						// returns subvector of "subvec" composed of elements with corresponding subvec_ind[i] == -1 (lengths of "subvec", "subvec_ind" should be the same)
std::vector<T> SubvecNotFound(const std::vector<T> &subvec, const std::vector<size_t> &subvec_ind);		// if subvec_ind := GetSubvecInd[Sorted](mainvec, subvec), then the returned vector is the not-found part of "subvec"

template <class T>
void VecAssign(std::vector<T> &vec, const std::vector<size_t> &ind, const std::vector<T> &rhs);	// vec[ind] = rhs; sizes of 'ind' and 'rhs' should be the same; 'ind' are indices in 'vec'
																								// for the remaining indices (those not in 'ind'), 'vec' values are not changed
template <class T>
std::vector<std::vector<T>> VecTranspose(const std::vector<std::vector<T>> &arr2d);		// transposes 2D array: res[i][j] = arr2d[j][i]

template <class T>
std::vector<T> VecTranspose(const std::vector<T> &arr_row_maj, size_t Nrow);			// transposes a row-major 2D array 'arr_row_maj' which originally has Nrow rows

template <class T>
bool FindDuplicate(std::vector<T> vec, T &dup);									// 'true' if "vec" has duplicate elements, in this case 'dup' is set to the found duplicate

template <class T>
std::vector<T> Unique(const std::vector<T> &vec);								// returns a vector of unique elements of 'vec' (uses std::set)

template <class FwdIterator>
bool is_strictly_sorted(FwdIterator first, FwdIterator last);					// 'true' if [first, last) contains strictly ascending elements

template <class T>
std::string ToString(const std::vector<T> &v, const std::string fmt = "%12.8g", const std::string delim = "\t");	// convert to string, applying format "fmt" to each element, separating them by "delim", adding '\n' in the end
template <class T>
void SaveASCII(FILE *f, const T *Data, size_t len, std::string fmt = "%12.8g");	// save Data[len] to "f" with given format
template <class T>
void SaveASCII(FILE *f, const T* const *Data, size_t len1, size_t len2, std::string fmt = "%12.8g");	// save Data[len1][len2] to "f" with given format; len1 numbers rows, len2 - columns
template <class T>
void VecAppend(std::vector<T> &a, const std::vector<T> &b);						// append to the vector end: a += b

//------------------------------------------------------------------------------------------
// This class represents matrices or vectors (= N x 1 matrices).
// Data are stored in row-major format (as in Vector2<>): (i, j) <-> data[i*jcount + j].
// Some I/O and arithmetic operations are implemented.
// DEVELOPER: call reset_chol_spo_cache(), reset_dsytrf_cache() every time the object is modified (including from the base class).
class Mat : public Vector2<double>
{
private:
	int op_switch;								// 1 - use hand-written algorithms for some functions (e.g. operator*), 2 - use BLAS/LAPACK algorithms

	mutable double *chol_spo_cache;				// cached values of Cholesky decomposition (upper triangle (lower = 0)); erased on every matrix change

	mutable double *dsytrf_cache;				// cached values of DSYTRF decomposition (upper triangle (lower = 0)); erased on every matrix change
	mutable int *dsytrf_ipiv;

	void reset_chol_spo_cache() const;			// delete & set zero
	void reset_dsytrf_cache() const;			// delete & set zero
protected:
	std::string delim = " \t\r\n";				// delimiters for parsing ASCII files
	const std::string debug_file = "DEBUG_Output_Mat_r%d.txt";		// output to this file if LAPACK procedures fail; %d is for rank number

	const double *chol_spo() const;				// [DPOTRF] performs and caches Cholesky decomposition U'*U of symmetric positive definite matrix [the upper triangle of *this is used]; uses DPOTRF
	void cache_msg_to_file(const std::string &msg) const;			// debug output of 'msg' to file, SIMILAR to Cache<T>::MsgToFile()
	void dsytrf(const double **A, const int **ipiv) const;			// performs and caches DSYTRF decomposition of symmetric matrix [the upper triangle of *this is used]

	void debug_output(const std::string &msg, const Mat *m) const;	// output "msg" and "m" to debug_file
public:
	// constructors and assignments
	Mat();										// default constructor - empty matrix
	Mat(size_t N);								// N x N unity matrix
	Mat(size_t I0, size_t J0, double val);		// I0 x J0 matrix, all elements set to 'val'
	Mat(std::vector<double> v, size_t I0, size_t J0);		// initialize I0 x J0 matrix, copy/move sequential data from vector "v"
	Mat(const std::vector<double> &v, bool IsDiag = false);	// initialize N x 1 "matrix" equal to vector "v" _OR_ N x N diagonal matrix with diagonal "v"
	Mat(const Mat &m);							// copy constructor
	Mat(Mat &&m) noexcept;						// move constructor
	virtual ~Mat();
	const Mat &operator=(const Mat &m);			// copy =
	const Mat &operator=(Mat &&m) noexcept;		// move =

	// I/O and conversions
	void LoadASCII(FILE *f, size_t num = -1);		// load matrix from file "f" by reading "num" non-empty lines (or until EOF); if num == -1, reads until EOF
													// any data previously stored in *this will be overridden
	void SaveASCII(FILE *f, std::string fmt = "%12.8g") const;		// save matrix to file "f", applying format "fmt" to each element
	std::string ToString(std::string fmt = "%12.8g") const;			// convert matrix to string (table), applying format "fmt" to each element
	void Bcast(int root, MPI_Comm comm);			// MPI broadcast matrix from 'root' rank; memory allocation is done if needed
	void Reshape(size_t i, size_t j);				// changes icount, jcount
	const std::vector<double> &ToVector() const {return data;};		// return the underlying 'data' vector (const reference)
	std::vector<double> &ToVectorMutable();							// return the underlying 'data' vector (reference)
	virtual void Deserialize(const double *v);		// fills values from "v"; current "icount", "jcount" are used for size
	void SetOpSwitch(int s);						// sets op_switch
	int GetOpSwitch() const {return op_switch;};	// gets op_switch

	// math
	// when many operations are used in one line, try to maximize the number of rvalues to minimize copying
	Mat Tr() const;							// transpose, using Manual | OpenBLAS depending on 'op_switch'
	double Trace() const;					// square matrix trace
	double Sum() const;						// sum of all elements
	double Max(size_t &i, size_t &j) const;		// returns the max element and its index (i, j)
	double Min(size_t &i, size_t &j) const;		// returns the min element and its index (i, j)
	double Norm1() const;					// 1-norm of a vector (matrix gets extended to the vector)
	double Norm2() const;					// 2-norm of a vector (matrix gets extended to the vector), Manual | BLAS depending on 'op_switch'
	double NormInf() const;					// inf-norm of a vector (matrix gets extended to the vector)
	void Func(const std::function<double (double)> &f);					// apply function f: R -> R to all matrix elements, i.e. Mat(i, j) = f(Mat(i, j))
	void FuncInd(const std::function<double (size_t, size_t, double)> &f);	// apply function f: (N, N, R) -> R to all matrix elements, i.e. Mat(i, j) = f(i, j, Mat(i, j))
	virtual double &operator()(size_t i, size_t j);			// element (i, j) - overrides the operator from Vector2<double>, adding reset_chol_spo_cache(), reset_dsytrf_cache()
	virtual const double &operator()(size_t i, size_t j) const;
	friend Mat operator&&(const Mat &m1, const Mat &m2);	// returns extended matrix by appending "m2" to the right of "m1": [m1, m2]
	friend Mat operator||(Mat m1, const Mat &m2);			// returns extended matrix by appending "m2" below "m1": [m1; m2]
	Mat Reorder(const std::vector<size_t> &ordi, const std::vector<size_t> &ordj) const;	// creates matrix with indices from "ordi" and "ordj" (indices may be repeated)
	Mat Reorder(const std::vector<size_t> &ord, int dim) const;			// creates submatrix with row or column indices from "ord"; use dim = 0 for applying "ord" to rows, dim = 1 for columns
	Mat Reorder(size_t i0, size_t i1, size_t j0, size_t j1) const;		// creates a submatrix with indices [i0, i1)*[j0, j1)
	Mat operator+(Mat m) const;				// *this + m; explicit RL associativity allows avoiding unnecessary copying, e.g. (a + (b + (c + d))) instead of a + b + c + d
	Mat operator-(Mat m) const;				// *this - m
	void operator+=(const Mat &m);			// *this += m
	void operator-=(const Mat &m);			// *this -= m
	Mat operator*(const Mat &m) const;		// *this * m, using Manual | BLAS depending on 'op_switch'
	std::vector<double> MultvecR(const std::vector<double> &v) const;		// *this * v, using BLAS dgemm
	std::vector<double> MultvecL(const std::vector<double> &v) const;		// *this * v, using BLAS dgemm; left multiplication with transposition is employed, i.e. (x'*A')'
	Mat Autocorr() const;					// calculates vector of the same size as input, its values at [k] = sample autocorrelations at lag k (*this should be a vector)
	int Ess(double &res) const;				// calculates effective sample size (res) using initial monotone sequence estimator (*this should be a vector); returns lag at which the estimator stopped
	Mat Chol() const;						// Cholesky decomposition, uses some simple handwritten code; the lower triangular part is referenced (and returned)
	Mat CholSPO() const;					// Cholesky decomposition of Symmetric POsitive definite matrix [the upper triangle of *this is used]; the upper triangle is returned (lower = 0); uses DPOTRF
	Mat InvSPO() const;						// inverse of Symmetric POsitive definite matrix [the upper triangle of *this is used]; uses DPOTRF, DPOTRI
	Mat InvU() const;						// inverse of the upper triangular matrix [the upper triangle of *this is used]; uses DTRTRI
	double LnDetSPO() const;				// ln(det) of Symmetric POsitive definite matrix [the upper triangle of *this is used]; uses DPOTRF
	Mat InvSY() const;						// inverse of SYmmetric matrix [the upper triangle of *this is used]; uses DSYTRF, DSYTRI
	double LnDetSY(int &sign) const;		// ln|det| of SYmmetric matrix [the upper triangle of *this is used]; sign(det) is also returned; uses DSYTRF
	Mat SymSqrt() const;					// symmetric square root, uses EigVal
	std::vector<double> EigVal(size_t I0, size_t I1) const;	// finds eigenvalues [increasing order] with indices [I0, I1) [0-based] of symmetric matrix [the upper triangle of *this is used]; uses LAPACK's DSYEVR
	std::vector<double> EigVal(size_t I0, size_t I1, Mat &EigVec) const;	// finds eigenvalues and eigenvectors of symmetric matrix [the upper triangle of *this is used]; uses LAPACK's DSYEVR
											// eigenvalues [in increasing order] with indices [I0, I1) [0-based] are returned
											// corresponding eigenvectors are saved as columns to 'EigVec'
	std::vector<double> SgVal() const;		// finds singular values of a rectangular matrix, in decreasing order; uses DGESDD
	double ICond1SPO() const;				// reciprocal condition number (in 1-norm) of symmetric positive definite matrix [the upper triangle of *this is used]; uses DPOCON, DLANGE, DPOTRF
	Mat BFGS_update_B(const Mat &dk, const Mat &gk) const;	// taking "this" as symmetric matrix B(k), returns BFGS update B(k+1) based on coordinate difference "dk" and gradient difference "gk"
	Mat BFGS_update_B(const std::vector<Mat> &Xk, const std::vector<Mat> &Gk) const;  // makes a series of BFGS updates, "Xk" - coordinate vectors, "Gk" - gradient vectors; in total (Xk.size - 1) updates are done
	friend double InnerProd(const Mat &a, const Mat &b);	// (a, b), inner product of two vectors; using Manual | BLAS depending on 'a.op_switch'
	friend Mat OuterProd(const Mat &a, const Mat &b);		// a * b', outer product of two vectors
	friend Mat VecProd(const Mat &a, const Mat &b);			// a (x) b, vector product of two 3-dim vectors
	friend Mat operator*(double d, Mat m);	// d * m
	friend Mat operator%(const std::vector<double> &v, Mat m);	// diag(v) * m -- multiplication by a diagonal matrix from left
	friend Mat operator%(Mat m, const std::vector<double> &v);	// m * diag(v) -- multiplication by a diagonal matrix from right
	friend std::vector<double> operator*(const Mat &m, const std::vector<double> &v);		// m * v, where v and result are columns, using Manual | BLAS depending on 'op_switch'
	friend std::vector<double> operator*(const std::vector<double> &v, const Mat &m);		// v * m, where v and result are rows, using Manual | BLAS depending on 'op_switch'
	friend Mat operator/(Mat A, Mat b);		// A^(-1) * b - i.e. solution of A*x = b, uses Gaussian elimination, with pivoting; 'b' may contain multiple right hand sides
};
//------------------------------------------------------------------------------------------
// arbitrary order tensor, based on tlib::tensor<double>
// data is stored contiguously
// indexing (i, j, k, ... ): i - fastest, k - slower, ...
//
// NB although ttv() calls openblas_set_num_threads(std::thread::hardware_concurrency()), it does not seem to launch all these threads...
// given the openblas configured as single-threaded
class TensorTTV
{
protected:
	tlib::tensor<double> *T;
	std::vector<size_t> Shape;

public:
	TensorTTV();								// creates 'empty' tensor; although under the hood 'T' will contain 1 element
	TensorTTV(std::vector<size_t> shape);		// creates tensor of the specified shape; order = |shape|
	TensorTTV(const TensorTTV &x);				// copy ctor
	const TensorTTV &operator=(const TensorTTV &x);		// assignment
	~TensorTTV();								// dtor

	const std::vector<double> &data() const;			// access the data, read-only
	const std::vector<size_t> &shape() const;			// access the shape, read-only
	void fill_from(const std::vector<double> &src);		// fetch the data from the provided 'src' vector
	Mat MultVec(const std::vector<double> &v, size_t mode, std::string slicing, std::string fusion) const;	// multiplication by vector 'v', contraction mode = 'mode' (zero-based)
																											// only works for order-3 tensors; slicing = "SMALL", "LARGE"; fusion = "NONE", "OUTER", "ALL"
};
//------------------------------------------------------------------------------------------
// rank-3 tensor with manual T*vec operation - mostly for timing and debug purposes
// data is stored contiguously
// indexing (i, j, k): i - fastest, k - slowest
class Tensor3
{
protected:
	size_t N0, N1, N2;			// dimensions
	size_t N0N1;				// stride for 'k'
	std::vector<double> data;

public:
	Tensor3(size_t n0, size_t n1, size_t n2, const std::vector<double> &v);	// initialize tensor using its shape and data array
	Tensor3() : Tensor3(0, 0, 0, std::vector<double>()){};					// empty tensor

	double &operator()(size_t i, size_t j, size_t k);				// element (i, j, k)
	const double &operator()(size_t i, size_t j, size_t k) const;	// const element (i, j, k)
	Mat MultVec(const std::vector<double> &v, size_t mode) const;	// multiplication by vector 'v', contraction mode = 'mode' (zero-based)
};
//------------------------------------------------------------------------------------------
// a bit legacy class for Standard Normal r.v. generation; no MPI;
// uses rand(); initialize the seed with srand() elsewhere
//
// THERE IS also a similar class HMMPI::Rand in MonteCarlo.h
//------------------------------------------------------------------------------------------
class RandNormal
{
protected:
	bool hold;
	double tmp;
public:
	RandNormal(){hold = false; tmp = 0;};
	double get();						// one Standard Normal r.v.
	std::vector<double> get(size_t n);	// a vector of 'n' Standard Normal r.v.
};
//------------------------------------------------------------------------------------------
// *** classes for 1D functions
//------------------------------------------------------------------------------------------
// the very base 1D function class
class Func1D
{
public:
	virtual ~Func1D(){};
	virtual double val(double x) const = 0;
	virtual double inv(double y) const;			// inverse function value
};
//------------------------------------------------------------------------------------------
// base class for piecewise linear functions
class Func1D_pwlin : public Func1D
{
protected:
	std::vector<double> xi, yi;
	std::vector<double> dri;						// same size as 'xi', 'yi'; dri[k] = (yi[k] - yi[k-1])/(xi[k] - xi[k-1]);

	mutable size_t locate_cache;					// cache for the locate_point()
	size_t locate_point(const std::vector<double> &vec, const double x) const;	// returns a "rough index" of 'x' in array 'vec' (which should be SORTED in increasing order), namely:
																				// for x <= vec[0] returns 0, for vec[last] < x returns vec.size()
public:																			// for vec[i-1] < x <= vec[i] returns 'i'

	Func1D_pwlin(std::vector<double> x, std::vector<double> y);
	virtual double val(double x) const;
};
//------------------------------------------------------------------------------------------
// class for piecewise-quadratic CDF (_automatically_ normalized to 1.0) based on piecewise-linear pdf
class Func1D_CDF : public Func1D_pwlin
{
protected:
	std::vector<double> Fi;							// normalized CDF at nodes

public:
	Func1D_CDF(std::vector<double> x, std::vector<double> y);	// the input is a pdf; the resulting CDF should be strictly increasing
	virtual double val(double x) const;
	virtual double inv(double y) const;							// inverse CDF
};
//------------------------------------------------------------------------------------------
// base class for 1D correlation functions (and their derivatives) - or radial basis functions
class Func1D_corr : public Func1D
{
protected:
	double nugget = 0;					// at the moment only used in Gauss and Matern

public:
	virtual ~Func1D_corr(){};
	virtual double val(double x) const {return f(x);};
	virtual double f(double x, bool smooth_at_nugget = false) const = 0;	// if smooth_at_nugget = true, there will be no discontinuity at x = 0; otherwise (default), discontinuity exists
	virtual double df(double x) const = 0;			// f'
	virtual double d2f(double x) const = 0;			// f''
	virtual double d3f(double x) const;				// f'''
	virtual double lim_df(double y) const;			// f'(y)/y, where y = x/R should be used in kriging	--	the names "lim" are historical; originally they were the limits
	virtual double lim_d2f(double y) const;			// [f''(y) - f'(y)/y]/(y^2)
	virtual double lim_d3f(double y) const;			// [3*f''/y - 3*f'/(y^2) - f''']/(y^3)
	virtual Func1D_corr* Copy() const {return 0;};		// some derived classes may override this; *** delete *** the returned pointer in the end
	void SetNugget(double n){nugget = n;};
};
//------------------------------------------------------------------------------------------
class CorrDummyConst : public Func1D_corr	// Dummy placeholder class
{
public:
	virtual double f(double x, bool smooth_at_nugget = false) const;
	virtual double df(double x) const;
	virtual double d2f(double x) const;
};
//------------------------------------------------------------------------------------------
class CorrGauss : public Func1D_corr		// Gaussian correlation
{
public:
	virtual double f(double x, bool smooth_at_nugget) const;
	virtual double df(double x) const;
	virtual double d2f(double x) const;
	virtual double d3f(double x) const;
	virtual double lim_df(double y) const;
	virtual double lim_d2f(double y) const;
	virtual double lim_d3f(double y) const;
	virtual Func1D_corr* Copy() const;					// *** delete *** the returned pointer in the end
};
//------------------------------------------------------------------------------------------
class CorrSpher : public Func1D_corr		// Spherical correlation
{
public:
	virtual double f(double x, bool smooth_at_nugget) const;
	virtual double df(double x) const;
	virtual double d2f(double x) const;
};
//------------------------------------------------------------------------------------------
class CorrExp : public Func1D_corr			// Exponential correlation
{
public:
	virtual double f(double x, bool smooth_at_nugget) const;
	virtual double df(double x) const;
	virtual double d2f(double x) const;
};
//------------------------------------------------------------------------------------------
class VarGauss : public Func1D_corr			// Gaussian variogram
{
public:
	virtual double f(double x, bool smooth_at_nugget) const;
	virtual double df(double x) const;
	virtual double d2f(double x) const;
};
//------------------------------------------------------------------------------------------
class BesselMod2k : public Func1D_corr		// modified Bessel function of second kind K_nu -- not to be used as correlation function
{
protected:
	double Kn(double Nu, double x) const;

public:
	double nu;		// set at any moment before usage

	BesselMod2k(double Nu = 2) : nu(Nu){};
	virtual double f(double x, bool smooth_at_nugget) const;
	virtual double df(double x) const;
	virtual double d2f(double x) const;
	virtual double d3f(double x) const;		// f'''
};
//------------------------------------------------------------------------------------------
class LnBesselMod2k : public Func1D_corr	// logarithm of modified Bessel function of second kind ln(K_nu) -- not to be used as correlation function
{
public:
	double nu;		// set at any moment before usage

	LnBesselMod2k(double Nu = 3.5) : nu(Nu){};
	virtual double f(double x, bool smooth_at_nugget) const;
	virtual double df(double x) const;
	virtual double d2f(double x) const;
	virtual double d3f(double x) const;		// f'''

	static double scaledKn(double Nu, double x);		// exp(x)*Kn(Nu, x) -- library function
	static double lnKn(double Nu, double x);			// ln(Kn(Nu, x)) -- library function
};
//------------------------------------------------------------------------------------------
class CorrMatern : public Func1D_corr		// Matern correlation
{
protected:
	LnBesselMod2k lnbess;

	const double tol0 = 1e-8;			// for x < tol, Taylor expansion will be used for f
	const double tol1 = 1e-8;			// for x < tol, Taylor expansion will be used for df
	const double tol2 = 1e-8;			// for x < tol, Taylor expansion will be used for d2f
	const double tol3 = 1e-8;			// for x < tol, Taylor expansion will be used for d3f

	const double limtol1 = 1e-8;		// for y < limtol, Taylor expansion is used for lim_df
	const double limtol2 = 1e-8;		// --"--										lim_d2f
	const double limtol3 = 1e-8;		// --"--										lim_d3f

public:
	virtual double f(double x, bool smooth_at_nugget = false) const;	// NOTE the default value is the same as for Func1D_corr::f()
	virtual double df(double x) const;
	virtual double d2f(double x) const;
	virtual double d3f(double x) const;
	virtual double lim_df(double y) const;
	virtual double lim_d2f(double y) const;
	virtual double lim_d3f(double y) const;
	virtual Func1D_corr* Copy() const;		// *** delete *** the returned pointer in the end
	void SetNu(double n);					// sets lnbess::nu
	double GetNu() const;
};
//------------------------------------------------------------------------------------------
class Func1D_corr_factory
{
public:
	static Func1D_corr *New(std::string type);		// produces 1D correlation function according to type: GAUSS, SPHER, EXP, VARGAUSS, MATERN, CONST
													// in the end, *** delete *** the returned pointer
};
//------------------------------------------------------------------------------------------
// *** classes for block-diagonal matrix
//------------------------------------------------------------------------------------------
// base class for a single block in block-diagonal matrix
class DiagBlock
{
protected:
	mutable bool holding_chol = false;		// 'true' = Cholesky decomposition was done, is stored, and can be accessed

public:
	virtual ~DiagBlock();
	virtual void mult(const std::vector<double> &vec1, int start, std::vector<double> &vec2) const = 0;			// vec2 = BLOCK * vec1, sub-vectors [start, start + block_size) are used
	virtual void div(const std::vector<double> &vec1, int start, std::vector<double> &vec2) const = 0;			// vec2 = BLOCK / vec1, sub-vectors [start, start + block_size) are used
	virtual void div(const Mat &m1, int start, Mat &m2) const = 0;												// m2 = BLOCK / m1, sub-matrices with rows [start, start + block_size) are used
	virtual void chol_mult(const std::vector<double> &vec1, int start, std::vector<double> &vec2) const = 0;	// vec2 = chol(BLOCK) * vec1, sub-vectors [start, start + block_size) are used
	virtual int size() const = 0;
	virtual std::string ToString() const = 0;			// for reporting
	virtual DiagBlock *Copy() const = 0; 				// creates a copy of 'this'; should be DELETED manually in the end!
};
//------------------------------------------------------------------------------------------
// block = diagonal matrix
class DiagBlockNum : public DiagBlock
{
protected:
	std::vector<double> d;					// block matrix (diag)
	mutable std::vector<double> chol_d;		// its Cholesky decomposition (= sqrt(d))

public:
	virtual void mult(const std::vector<double> &vec1, int start, std::vector<double> &vec2) const;
	virtual void div(const std::vector<double> &vec1, int start, std::vector<double> &vec2) const;
	virtual void div(const Mat &m1, int start, Mat &m2) const;
	virtual void chol_mult(const std::vector<double> &vec1, int start, std::vector<double> &vec2) const;
	DiagBlockNum(std::vector<double> v) : d(std::move(v)){};
	virtual int size() const {return (int) d.size();};
	virtual std::string ToString() const {return Mat(d).ToString();};
	virtual DiagBlock *Copy() const {return new DiagBlockNum(*this);}; 			// creates a copy of 'this'; should be DELETED manually in the end!
};
//------------------------------------------------------------------------------------------
// block = dense matrix
class DiagBlockMat : public DiagBlock
{
protected:
	Mat M;					// block matrix
	mutable Mat chol_M;		// its Cholesky decomposition

public:
	virtual void mult(const std::vector<double> &vec1, int start, std::vector<double> &vec2) const;
	virtual void div(const std::vector<double> &vec1, int start, std::vector<double> &vec2) const;
	virtual void div(const Mat &m1, int start, Mat &m2) const;
	virtual void chol_mult(const std::vector<double> &vec1, int start, std::vector<double> &vec2) const;
	DiagBlockMat(Mat m);
	virtual int size() const {return (int) M.ICount();};
	virtual std::string ToString() const {return M.ToString();};
	virtual DiagBlock *Copy() const {return new DiagBlockMat(*this);}; 			// creates a copy of 'this'; should be DELETED manually in the end!
};
//------------------------------------------------------------------------------------------
// Class for block-diagonal matrix.
// The matrix is MPI-distributed (block-wise), but all the arithmetic member-functions work independently on all ranks, no result collection is done.
// HOW TO WORK: 1) construct-1, 2) AddBlock's, 3) Finalize, 4) anything else
// OR:          1) construct-2 (easy), 2) anything else
// All functions should be called on ALL RANKS in communicator.
class BlockDiagMat : public ManagedObject
{
private:
	int last_r;						// rank to which last block was added; sync between ranks
	const std::string file_rpt = "BDC_report_%.100s_rank%d.txt";		// file name template for reporting

	void write_test_tor(std::string msg) const;		// reports from ctors and dtors
protected:
	std::vector<DiagBlock*> Blocks;	// MPI-distributed
	std::vector<int> data_ind;		// vector of size = MPI_Size + 1, stores the beginnings and ends of all "data points" on each rank; sync between ranks
	MPI_Comm comm;
	int sz;							// local size (different for each rank)
	bool finalized;					// sync between ranks (comm)

public:
	BlockDiagMat(MPI_Comm c);
	BlockDiagMat(MPI_Comm c, const HMMPI::CorrelCreator *Corr, const HMMPI::StdCreator *Std);	// easy constructor: it creates a fully prepared object
	BlockDiagMat(const HMMPI::Mat &Pr, const BlockDiagMat &b);	// this CTOR prepends the existing object 'b' by the new block 'Pr' (a full matrix, or diagonal) on rank-0
	~BlockDiagMat();
	BlockDiagMat(const BlockDiagMat &b) = delete;
	const BlockDiagMat &operator=(const BlockDiagMat &b) = delete;
	int size() const;								// matrix size for the current rank
	std::vector<int> Data_ind() const;				// returns 'data_ind'; this is sync between ranks (comm)
	MPI_Comm GetComm() const {return comm;};		// communicator

													// In sequential (contiguous) calls to AddBlock(...), 'r' (sync!) should be incremented by 0 or 1
	void AddBlock(std::vector<double> v, int r);	// block = diag(v) is added to rank 'r'; 'v' should exist at least on rank 'r'
	void AddBlock(HMMPI::Mat m, int r);				// block =     'm' is added to rank 'r'; 'm' should exist at least on rank 'r'

	void Finalize();												// (does some checks, fills 'data_ind') should be called after adding all blocks, prior to main work
	std::vector<double> operator*(const std::vector<double> &v) const;		// MAT * v, each rank does its own part, the full output vector is not created; 'v' is MPI-distributed
	std::vector<double> operator/(const std::vector<double> &v) const;		// MAT / v, each rank does its own part, the full output vector is not created; 'v' is MPI-distributed
	Mat operator/(const Mat &m) const;										// MAT / m, each rank does its own part, the full output matrix is not created; 'm' is row-wise MPI-distributed; columns of 'm' are multiple right hand sides
	std::vector<double> operator%(const std::vector<double> &v) const;		// chol(MAT) * v, each rank does its own part, the full output vector is not created; 'v' is MPI-distributed
	double InvTwoSideVecMult(const std::vector<double> &v) const;			// v' * MAT^(-1) * v, the MPI-local part of the "result" is returned; if necessary, it should be MPI_Reduced afterwards
	void PrintToFile(std::string tag) const;		// print the contents to files, each rank outputs to its own file; 'tag' is a suffix
};
//------------------------------------------------------------------------------------------
// *** classes for solving linear equations
//------------------------------------------------------------------------------------------
class Solver										// abstract class - base for all solvers
{
protected:
	virtual Mat _solve(Mat A, Mat b) const = 0;		// solves A*x = b

public:
	mutable int rank;								// some solvers may occasionally fill the matrix rank

	Solver(){rank = -1;};
	virtual ~Solver(){};
	Mat Solve(Mat A, Mat b) const;					// solves A*x = b, on error outputs A, b to file
};
//------------------------------------------------------------------------------------------
class SolverGauss : public Solver
{
protected:
	virtual Mat _solve(Mat A, Mat b) const;			// hand-written Gaussian elimination - i.e. Mat operator/
};
//------------------------------------------------------------------------------------------
class SolverDGESV : public Solver
{
protected:
	virtual Mat _solve(Mat A, Mat b) const;			// DGESV (general) solver for square matrix
};
//------------------------------------------------------------------------------------------
class SolverDGELS : public Solver
{
protected:
	virtual Mat _solve(Mat A, Mat b) const;			// DGELS (least squares) solver for full rank rectangular matrix
};
//------------------------------------------------------------------------------------------
class SolverDGELSD : public Solver
{
protected:
	virtual Mat _solve(Mat A, Mat b) const;			// DGELSD (least squares) solver for matrix of any rank; uses bidiagonal least squares, and divide and conquer approach
};
//------------------------------------------------------------------------------------------
class SolverDGELSS : public Solver
{
protected:
	virtual Mat _solve(Mat A, Mat b) const;			// DGELSS (least squares) solver for matrix of any rank; uses the singular value decomposition
};
//------------------------------------------------------------------------------------------
class SolverDGELSY : public Solver
{
protected:
	virtual Mat _solve(Mat A, Mat b) const;			// DGELSY (least squares) solver for matrix of any rank; uses complete orthogonal factorization
};
//------------------------------------------------------------------------------------------
class Rand;
// this class is for constraints that are bounds [min, max]
class BoundConstr : public Constraints
{
protected:
	std::vector<double> min;								// min & max - INTERNAL representation
	std::vector<double> max;

	virtual std::string par_name(int i) const;				// name for parameter "i" - used in reporting
	virtual double minrpt(int i) const {return min[i];};	// used in BoundConstr::Check for reporting, may return either internal or external representation in derived classes
	virtual double maxrpt(int i) const {return max[i];};
public:

	void OverrideBounds(const std::vector<double> &newmin, const std::vector<double> &newmax);	// overrides min, max after checking that dimensions are the same
	virtual std::vector<double> fullmin() const {return min;};	// min & max - INTERNAL FULLDIM parameters
	virtual std::vector<double> fullmax() const {return max;};
	virtual std::vector<double> actmin() const {return min;};	// min & max - INTERNAL ACTIVE parameters
	virtual std::vector<double> actmax() const {return max;};
																		// functions below deal with vectors in INTERNAL representation
	virtual std::string Check(const std::vector<double> &p) const;		// "", if all constraints are satisfied for 'p'; or a message, if not
	virtual std::string CheckEps(std::vector<double> &p, const double eps) const;		// same as Check(); violations 'eps' are allowed, and where they take place, 'p' is adjusted
	virtual bool FindIntersect(const std::vector<double> &x0, const std::vector<double> &x1, std::vector<double> &xint, double &alpha, int &i) const;	// 'true' if all constraints are satisfied for 'x1'
																		// otherwise ('false') finds intersection with x0 + t*(x1-x0)
	virtual std::vector<double> SobolSeq(long long int &seed) const;	// generate a Sobol point in [min, max], based on 'seed'
	virtual std::vector<double> RandU(Rand *rctx) const;				// generate a uniform random point in [min, max], based on 'rctx' (whose state will change)
	virtual void Write_params_log(const std::vector<double> &p, std::string fname) const = 0;		// writes "p" (and other appropriate data) to "fname"
	virtual void Push_point(double Init, double Min, double Max, std::string AN, std::string Name);	// adds one more point with its attributes (Init, Min, Max - INTERNAL values, AN - active flag, Name - as in KW_parameters)
																									// other attributes defined in derived classes take some default values
	void AdjustInitSpherical(std::vector<double> &p) const;				// changes 'p' - initial point for optimization (spherical coordinates) so that min <= p <= max
																		// trying p[i] +/- 2*pi, and if it doesn't help, taking p[i] = min[i] or max[i]
};
//------------------------------------------------------------------------------------------
// class for transforming between R^n Cartesian coordinates "x" and (n-1)-sphere coordinates "p"
// The sphere is: |x - c| = R
// The derivatives of the transform are also found: dx/dp, d/dp(dx/dp_k)
class SpherCoord
{
protected:
	double arccot(double a, double b) const;	// arccot(a/b), also handling the case b == 0

	std::vector<double> calc_cos(const std::vector<double> &v) const;		// calculates cos(v_i)
	std::vector<double> calc_sin(const std::vector<double> &v) const;		// calculates sin(v_i)

	Cache<SpherCoord, std::vector<double>, std::vector<double>> Cos;		// trigonometric caches
	Cache<SpherCoord, std::vector<double>, std::vector<double>> Sin;

public:
	const std::vector<double> c;				// sphere center (size = dim)
	const int dim;
	const double R;
	mutable double radius;		 				// filled by cart_to_spher()

	const double pi;
	const double pi2;

	SpherCoord(double R0, const std::vector<double> &c0);
	std::vector<double> spher_to_cart(const std::vector<double> &p) const;
	std::vector<double> cart_to_spher(std::vector<double> x) const;	// this function also calculates 'radius'
	Mat dxdp(const std::vector<double> &p) const;					// dx/dp
	Mat dxdp_k(const std::vector<double> &p, int k) const;			// d/dp(dx/dp_k)
	Mat dxdp_k_upper(const std::vector<double> &p, int k) const;	// d/dp(dx/dp_k) with elements (i,j) where only elements j<=k are filled (using symmetry for speed)

	bool periodic_swap(const HMMPI::BoundConstr *bc, std::vector<double> &p) const;		// if the range for the last component is [0, 2*pi] and p.last is on the boundary, swapping 0 <-> 2*pi is done for p.last
																						// returns 'true' if swapping was done
};
//------------------------------------------------------------------------------------------
/* Note on using BLAS & LAPACK
 * required header files: lapacke_select.h
 * project should link (-l) the libraries: blas, lapack, lapacke
 * library path (-L) should point to directory with libblas.dll, liblapack.dll, liblapacke.dll
 * dll's (mingw & lapack) should be put to some directory, then add that directory to PATH in cygwin
 */
//------------------------------------------------------------------------------------------
// implementation of some TEMPLATE FUNCTIONS
//------------------------------------------------------------------------------------------
template <class RandomAccessIterator>
std::vector<size_t> SortPermutation(RandomAccessIterator first, RandomAccessIterator last)
{
	assert(last - first >= 0);
	std::vector<size_t> res(last - first);
	std::iota(res.begin(), res.end(), 0);	// [0, N)

	std::sort(res.begin(), res.end(), [&](size_t a, size_t b){return *(first+a) < *(first+b);});
	return res;
}
//------------------------------------------------------------------------------------------
template <class FwdIterator, class T>
FwdIterator FindBinary(FwdIterator first, FwdIterator last, const T &val)
{
	FwdIterator i = std::lower_bound(first, last, val);
	if (i != last && !(val < *i))
		return i;		// found
	else
		return last;	// not found
}
//------------------------------------------------------------------------------------------
template <class T>
std::vector<std::vector<T>> SqueezeVec(std::vector<std::vector<T>> v)		// returns a copy of "v" with empty v[i]'s removed
{
	std::vector<std::vector<T>> res;
	res.reserve(v.size());

	for (size_t i = 0; i < v.size(); i++)
		if (v[i].size() > 0) res.push_back(std::move(v[i]));

	return res;
}
//------------------------------------------------------------------------------------------
template <class T>
std::vector<T> Reorder(const std::vector<T> &v, const std::vector<size_t> &ord)
{
	std::vector<T> res;

	size_t LEN = ord.size();
	res.reserve(LEN);
	for (size_t i = 0; i < LEN; i++)
	{
		size_t k = ord[i];
		if (k >= v.size())
			throw Exception(stringFormatArr("Index out of range in Reorder(), index k = {0:%zu} is not in vector of size = {1:%zu}", std::vector<size_t>{k, v.size()}));

		res.push_back(v[k]);
	}

	return res;
}
//------------------------------------------------------------------------------------------
template <class T>
std::vector<T> Reorder(const std::vector<T> &v, size_t M, size_t N, const std::vector<size_t> &ordi, const std::vector<size_t> &ordj, bool skipneg, T defval)
{
	if (M*N != v.size())
		throw Exception(stringFormatArr("M*N ({0:%zu}) != v.size ({1:%zu}) in Reorder()", std::vector<size_t>{M*N, v.size()}));

	size_t m = ordi.size();
	size_t n = ordj.size();
	std::vector<T> res(m*n);

	for (size_t i = 0; i < m; i++)
		for (size_t j = 0; j < n; j++)
		{
			size_t k = ordi[i];
			size_t l = ordj[j];
			if (skipneg && (k == (size_t)-1 || l == (size_t)-1))		// if 'skipneg' == true, indices ordi, ordj equal to -1 will be populated with 'defval'
				res[i*n + j] = defval;
			else
			{
				if (k >= M || l >= N)
					throw Exception(stringFormatArr("Index [{0:%zu}, {1:%zu}] is out of range in Reorder()", std::vector<size_t>{k, l}));

				res[i*n + j] = v[k*N + l];
			}
		}

	return res;
}
//------------------------------------------------------------------------------------------
template <class T>
std::vector<size_t> GetSubvecInd(const std::vector<T> &mainvec, const std::vector<T> &subvec)
{
	std::vector<size_t> res(subvec.size());
	for (size_t i = 0; i < subvec.size(); i++)
	{
		size_t ind = std::find(mainvec.begin(), mainvec.end(), subvec[i]) - mainvec.begin();
		if (ind != mainvec.size())
			res[i] = ind;
		else
			res[i] = -1;
	}

	return res;
}
//------------------------------------------------------------------------------------------
template <class T>
std::vector<size_t> GetSubvecIndSorted(const std::vector<T> &mainvec_sorted, const std::vector<T> &subvec)
{
	std::vector<size_t> res(subvec.size());
	for (size_t i = 0; i < subvec.size(); i++)
	{
		size_t ind = FindBinary(mainvec_sorted.begin(), mainvec_sorted.end(), subvec[i]) - mainvec_sorted.begin();
		if (ind != mainvec_sorted.size())
			res[i] = ind;
		else
			res[i] = -1;
	}

	return res;
}
//------------------------------------------------------------------------------------------
template <class T>
std::vector<T> SubvecNotFound(const std::vector<T> &subvec, const std::vector<size_t> &subvec_ind)
{
	if (subvec.size() != subvec_ind.size())
		throw Exception("subvec.size() != subvec_ind.size() in SubvecNotFound()");

	std::vector<T> res;
	for (size_t i = 0; i < subvec.size(); i++)
		if (subvec_ind[i] == (size_t)-1)
			res.push_back(subvec[i]);

	return res;
}
//------------------------------------------------------------------------------------------
template <class T>
void VecAssign(std::vector<T> &vec, const std::vector<size_t> &ind, const std::vector<T> &rhs)
{
	if (ind.size() != rhs.size())
		throw Exception("Sizes of ind and rhs do not match in VecAssign<T>");
	for (size_t i = 0; i < ind.size(); i++)
	{
		if (ind[i] >= vec.size())
			throw Exception("Index ind[i] is out of range in VecAssign<T>");
		vec[ind[i]] = rhs[i];
	}
}
//------------------------------------------------------------------------------------------
template <class T>
std::vector<std::vector<T>> VecTranspose(const std::vector<std::vector<T>> &arr2d)		// transposes 2D array: res[i][j] = arr2d[j][i]
{
	std::vector<std::vector<T>> res;
	if (arr2d.size() > 0)
	{
		const size_t M = arr2d.size();
		const size_t N = arr2d[0].size();
		res = std::vector<std::vector<T>>(N, std::vector<T>(M));
		for (size_t i = 0; i < M; i++)
		{
			if (arr2d[i].size() != N)
				throw Exception(stringFormatArr("Input in VecTranspose is not a 2D array, lengths: {0:%zu} (i=0), {1:%zu} (i={2:%zu})", std::vector<size_t>{N, arr2d[i].size(), i}));

			for (size_t j = 0; j < N; j++)
				res[j][i] = arr2d[i][j];
		}
	}

	return res;
}
//------------------------------------------------------------------------------------------
template <class T>
std::vector<T> VecTranspose(const std::vector<T> &arr_row_maj, size_t Nrow)				// transposes a row-major 2D array 'arr_row_maj' which originally has Nrow rows
{
	const size_t N = arr_row_maj.size();
	std::vector<T> res(N);

	if (N == 0)
		return res;

	assert(N > 0 && Nrow > 0);
	assert(N % Nrow == 0);
	const size_t Ncol = N / Nrow;

	for (size_t i = 0; i < Nrow; i++)
		for (size_t j = 0; j < Ncol; j++)
			res[Nrow*j + i] = arr_row_maj[Ncol*i + j];

	return res;
}
//------------------------------------------------------------------------------------------
template <class T>
bool FindDuplicate(std::vector<T> vec, T &dup)				// 'true' if "vec" has duplicate elements, in this case 'dup' is set to the found duplicate
{
	std::sort(vec.begin(), vec.end());						// note: the copy "vec" is sorted
	auto rpt = std::adjacent_find(vec.begin(), vec.end());	// find repeats
	if (rpt != vec.end())
	{
		dup = *rpt;
		return true;
	}
	else
		return false;
}
//------------------------------------------------------------------------------------------
template <class T>
std::vector<T> Unique(const std::vector<T> &vec)			// returns a vector of unique elements of 'vec' (uses std::set)
{
	std::set<T> work(vec.begin(), vec.end());
	return std::vector<T>(work.begin(), work.end());
}
//------------------------------------------------------------------------------------------
template <class FwdIterator>
bool is_strictly_sorted(FwdIterator first, FwdIterator last)					// 'true' if [first, last) contains strictly increasing elements
{
	return std::adjacent_find(first, last, std::greater_equal<typename std::iterator_traits<FwdIterator>::value_type>()) == last;
}
//------------------------------------------------------------------------------------------
template <>
std::string ToString<std::string>(const std::vector<std::string> &v, const std::string fmt, const std::string delim);
//------------------------------------------------------------------------------------------
template <class T>
std::string ToString(const std::vector<T> &v, const std::string fmt, const std::string delim)	// convert to string, applying format "fmt" to each element, separating them by "delim", adding '\n' in the end
{
	char buff[BUFFSIZE];
	std::string res;

	for (size_t i = 0; i < v.size(); i++)
	{
		sprintf(buff, fmt.c_str(), v[i]);
		res += buff;
		if (i < v.size()-1)
			res += delim;
		else
			res += "\n";
	}

	return res;
}
//------------------------------------------------------------------------------------------
template <class T>
void SaveASCII(FILE *f, const T *Data, size_t len, std::string fmt)
{
	fputs(ToString(std::vector<T>(Data, Data + len), fmt).c_str(), f);
}
//------------------------------------------------------------------------------------------
template <class T>
void SaveASCII(FILE *f, const T* const *Data, size_t len1, size_t len2, std::string fmt)
{
	for (size_t i = 0; i < len1; i++)
		for (size_t j = 0; j < len2; j++)
		{
			fprintf(f, fmt.c_str(), Data[i][j]);
			if (j+1 < len2)
				fprintf(f, "\t");
			else
				fprintf(f, "\n");
		}
}
//------------------------------------------------------------------------------------------
template <class T>
void VecAppend(std::vector<T> &a, const std::vector<T> &b)
{
	a.reserve(a.size() + b.size());
	a.insert(a.end(), b.begin(), b.end());
}
//------------------------------------------------------------------------------------------

}	// namespace HistMatMPI

#endif /* MATHUTILS_H_ */
