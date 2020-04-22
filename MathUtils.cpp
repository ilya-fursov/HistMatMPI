#include "Abstract.h"
#include "MathUtils.h"
#include "MonteCarlo.h"
#include "lapacke.h"
#include "cblas.h"
#include "sobol.hpp"
#include <cstring>
#include <iostream>
#include <numeric>
#include <limits>
#include <cmath>
#include <cassert>
#include <chrono>
#include <algorithm>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>

//#define TESTING
//#define TESTBCAST
//#define TESTBLOCK
//#define TESTSOLVER

const double PI = acos(-1.0);

namespace HMMPI
{

//------------------------------------------------------------------------------------------
// some functions
//------------------------------------------------------------------------------------------
size_t LINEBUFF = 4096;
//------------------------------------------------------------------------------------------
double NumD(const std::function<double(double)> &f, double x, double h, OH oh)
{
	if (oh == OH2)
		return (f(x+h) - f(x-h))/(2*h);
	else if (oh == OH4)
		return ((f(x-2*h) - f(x+2*h))/12 + (f(x+h) - f(x-h))*2/3)/h;
	else
		throw Exception("Unrecognised OH type in NumD");
}
//------------------------------------------------------------------------------------------
double NumD2(const std::function<double(double)> &f, double x, double h, OH oh)
{
	if (oh == OH2)
		return ((f(x+h) + f(x-h)) - 2*f(x))/(h*h);
	else if (oh == OH4)
		return ((f(x+h) + f(x-h))*4/3 - f(x)*5/2 - (f(x+2*h) + f(x-2*h))/12)/(h*h);
	else
		throw Exception("Unrecognised OH type in NumD2");
}
//------------------------------------------------------------------------------------------
double NumD3(const std::function<double(double)> &f, double x, double h, OH oh)		// numerical derivative d3f/dx3; h - increment, oh - precision
{
	if (oh == OH2)
		return ((f(x-h) - f(x+h)) + 0.5*(f(x+2*h) - f(x-2*h)))/(h*h*h);
	else if (oh == OH4)
		return ((f(x-h) - f(x+h))*13/8 + (f(x+2*h) - f(x-2*h)) + (f(x-3*h) - f(x+3*h))/8)/(h*h*h);
	else
		throw Exception("Unrecognised OH type in NumD3");
}
//------------------------------------------------------------------------------------------
double integr_Gauss(const std::function<double(double)> &g, int n, double x0, double mu, double sigma)		// calculate int_{x0...+inf} g(x)p(x)dx, where p = PDF Normal(mu, sigma^2), using "n" integration intervals with trapezoid rule
{
	assert(n > 1);

	if (sigma == 0)				// degenerate case
	{
		if (x0 <= mu)
			return g(mu);
		else
			return 0;
	}

	double res = 0;
	const double p0 = gsl_cdf_gaussian_P(x0 - mu, sigma);	// first integration point (corresp. to x0)
	double f0 = gsl_ran_gaussian_pdf(x0 - mu, sigma);		// f at the first point
	double G0 = g(x0);										// g at the first point

	const int i0 = floor(p0*(n+2)) + 1;						// first index of the next integration point
	for (int i = i0; i <= n+1; i++)			// i is the index of the next integration point
	{
		double x = mu + gsl_cdf_gaussian_Pinv((double)i/(n+2), sigma);	// next integration point
		double f = gsl_ran_gaussian_pdf(x - mu, sigma);		// f at the next point
		double G = g(x);									// g at the next point

		res += (f0*G0 + f*G)/2*(x - x0);	// trapezoid term

		x0 = x;								// update the quantities
		f0 = f;
		G0 = G;
	}

	return res;
}
//------------------------------------------------------------------------------------------
double integr_Gauss(const std::function<double(double)> &g, int n, double x0, double mu, double sigma, const Func1D_CDF &F)	// similar to above, with user-defined CDF F, employing Normal score transform:
{																															// int_{invP0(F(x0))...+inf} g(invF(P0(y)))p(y)dy, where p = PDF Normal(mu, sigma^2), P0 is Standard Normal CDF
	double y0 = gsl_cdf_gaussian_Pinv(F.val(x0), 1.0);		// P0 = N(0, 1)
	auto gnew = [&F, &g](double y) -> double
	{
		double p = gsl_cdf_gaussian_P(y, 1.0);				// P0 = N(0, 1)
		return g(F.inv(p));
	};

	return integr_Gauss(gnew, n, y0, mu, sigma);
}
//------------------------------------------------------------------------------------------
bool IsNaN(double d)
{
	return std::isnan(d) || std::isinf(d);
}
//------------------------------------------------------------------------------------------
double _sqrt(double d)
{
	return sqrt(d);
};
//------------------------------------------------------------------------------------------
void Sobol(long long int &seed, std::vector<double> &vec)
{
	i8_sobol ((int)vec.size(), &seed, vec.data());
}
//------------------------------------------------------------------------------------------
void CholDecomp(const double *A, double *res, int sz)
{
	for (int j = 0; j < sz; j++)
		for (int i = 0; i < sz; i++)
			res[j*sz + i] = 0;

	for (int j = 0; j < sz; j++)
	{
		double sum = 0;
		for (int k = 0; k <= j-1; k++)
			sum += res[j*sz + k]*res[j*sz + k];
		if (A[j*sz + j] - sum < 0)
			throw Exception("sqrt of negative value in CholDecomp(), input matrix A is not positive definite");
		res[j*sz + j] = sqrt(A[j*sz + j] - sum);

		for (int i = j+1; i < sz; i++)
		{
			sum = 0;
			for (int k = 0; k <= j-1; k++)
				sum += res[i*sz + k]*res[j*sz + k];
			res[i*sz + j] = (A[i*sz + j] - sum)/res[j*sz + j];
		}
	}
}
//------------------------------------------------------------------------------------------
void Bcast_string(std::string &s, int root, MPI_Comm comm)			// MPI broadcast std::string from 'root' rank; memory allocation is done if needed
{
	int rank;
	MPI_Comm_rank(comm, &rank);
	if (rank != root)
		s = "";

	std::vector<char> work(s.c_str(), s.c_str()+s.size()+1);
	Bcast_vector(work, root, comm);

	if (rank != root)
		s = work.data();
}
//------------------------------------------------------------------------------------------
void Bcast_vector(std::vector<double> &v, int root, MPI_Comm comm)
{
	if (comm == MPI_COMM_NULL)
		return;

	int sz = v.size();							// vector size to be broadcasted from 'root'
	MPI_Bcast(&sz, 1, MPI_INT, root, comm);
	if ((int)v.size() != sz)
		v = std::vector<double>(sz);			// reallocate array

	MPI_Bcast(v.data(), sz, MPI_DOUBLE, root, comm);

#ifdef TESTBCAST
	int rank00;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank00);
	std::cout << "rank " << rank00 << "\tBcast_vector<double>\n";
#endif
}
//------------------------------------------------------------------------------------------
void Bcast_vector(std::vector<int> &v, int root, MPI_Comm comm)
{
	if (comm == MPI_COMM_NULL)
		return;

	int sz = v.size();							// vector size to be broadcasted from 'root'
	MPI_Bcast(&sz, 1, MPI_INT, root, comm);
	if ((int)v.size() != sz)
		v = std::vector<int>(sz);				// reallocate array

	MPI_Bcast(v.data(), sz, MPI_INT, root, comm);

#ifdef TESTBCAST
	int rank00;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank00);
	std::cout << "rank " << rank00 << "\tBcast_vector<int>\n";
#endif
}
//------------------------------------------------------------------------------------------
void Bcast_vector(std::vector<char> &v, int root, MPI_Comm comm)
{
	if (comm == MPI_COMM_NULL)
		return;

	int sz = v.size();							// vector size to be broadcasted from 'root'
	MPI_Bcast(&sz, 1, MPI_INT, root, comm);
	if ((int)v.size() != sz)
		v = std::vector<char>(sz);				// reallocate array

	MPI_Bcast(v.data(), sz, MPI_CHAR, root, comm);

#ifdef TESTBCAST
	int rank00;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank00);
	std::cout << "rank " << rank00 << "\tBcast_vector<char>\n";
#endif
}
//------------------------------------------------------------------------------------------
void Bcast_vector(std::vector<std::vector<double>> &v, int root, MPI_Comm comm)
{
	if (comm == MPI_COMM_NULL)
		return;

	int sz = v.size();									// vector<vector> size to be broadcasted from 'root'
	MPI_Bcast(&sz, 1, MPI_INT, root, comm);
	if ((int)v.size() != sz)
		v = std::vector<std::vector<double>>(sz);		// reallocate array

	for (int i = 0; i < sz; i++)
		Bcast_vector(v[i], root, comm);

#ifdef TESTBCAST
	int rank00;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank00);
	std::cout << "rank " << rank00 << "\tBcast_vector<vector<double>>\n";
#endif
}
//------------------------------------------------------------------------------------------
void Bcast_vector(std::vector<std::vector<std::vector<double>>> &v, int root, MPI_Comm comm)
{
	if (comm == MPI_COMM_NULL)
		return;

	int sz = v.size();												// vector<vector<vector>> size to be broadcasted from 'root'
	MPI_Bcast(&sz, 1, MPI_INT, root, comm);
	if ((int)v.size() != sz)
		v = std::vector<std::vector<std::vector<double>>>(sz);		// reallocate array

	for (int i = 0; i < sz; i++)
		Bcast_vector(v[i], root, comm);

#ifdef TESTBCAST
	int rank00;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank00);
	std::cout << "rank " << rank00 << "\tBcast_vector<vector<vector<double>>>\n";
#endif
}
//------------------------------------------------------------------------------------------
void Bcast_vector(double **v, int len1, int len2, int root, MPI_Comm comm)
{
	if (comm == MPI_COMM_NULL)
		return;

	for (int i = 0; i < len1; i++)
		MPI_Bcast(v[i], len2, MPI_DOUBLE, root, comm);

#ifdef TESTBCAST
	int rank00;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank00);
	std::cout << "rank " << rank00 << "\tBcast_vector(double**)\n";
#endif
}
//------------------------------------------------------------------------------------------
namespace ManualMath
{
double InnerProd(const std::vector<double> &a, const std::vector<double> &b)
{
	size_t len = a.size();
	if (b.size() != len)
		throw Exception(stringFormatArr("Vector sizes do not match in InnerProd(vector({0:%zu}), vector({1:%zu}))", std::vector<size_t>{len, b.size()}));

	double res = 0;
	for (size_t i = 0; i < len; i++)
		res += a[i]*b[i];

	return res;
}
}
//------------------------------------------------------------------------------------------
double InnerProd(const std::vector<double> &a, const std::vector<double> &b)	// BLAS version
{
	size_t len = a.size();
	if (b.size() != len)
		throw Exception(stringFormatArr("Vector sizes do not match in InnerProd(vector({0:%zu}), vector({1:%zu}))", std::vector<size_t>{len, b.size()}));

	return cblas_ddot(len, a.data(), 1, b.data(), 1);
}
//------------------------------------------------------------------------------------------
std::vector<double> Vec_x_ay(std::vector<double> x, const std::vector<double> &y, double a)
{
	size_t len = x.size();
	assert(y.size() == len);

	std::vector<double> res = std::move(x);
	for (size_t i = 0; i < len; i++)
		res[i] += a*y[i];

	return res;
}
//------------------------------------------------------------------------------------------
std::vector<double> Vec_ax(std::vector<double> x, double a)
{
	std::vector<double> res = std::move(x);
	for (auto &i : res)
		i *= a;

	return res;
}
//------------------------------------------------------------------------------------------
double Vec_pow_multiind(const std::vector<double> &v, const std::vector<int> &mi, int di, int dj)
{
	int len = mi.size();

	if (di != -1 && di == dj)
		throw Exception("Одинаковые индексы дифференцирования di, dj в Vec_pow_multiind", "Same differentiation indices di, dj in Vec_pow_multiind");
	if (di >= len || dj >= len)
		throw Exception("di, dj вне допустимого диапазона в Vec_pow_multiind", "di, dj out of range in Vec_pow_multiind");

	double res = 1;
	for (int i = 0; i < len; i++)
	{
		int ind = mi[i];
		if (ind < 0 || ind >= (int)v.size())
			throw Exception("Индекс вне диапазона в Vec_pow_multiind()", "Index out of range in Vec_pow_multiind");
		if ((di == -1 || i != di)&&(dj == -1 || i != dj))	// differentiation indices di, dj are skipped (if != -1)
			res *= v[ind];
	}

	return res;
}
//------------------------------------------------------------------------------------------
template <>
std::string ToString<std::string>(const std::vector<std::string> &v, const std::string fmt, const std::string delim)
{
	std::string res;
	for (size_t i = 0; i < v.size(); i++)
	{
		size_t DYNBUFF = BUFFSIZE*2;
		while (v[i].length() + BUFFSIZE > DYNBUFF-1)
			DYNBUFF *= 2;

		char *buff = new char[DYNBUFF];

		int n = sprintf(buff, fmt.c_str(), v[i].c_str());
		if (n < 0 || n >= (int)DYNBUFF)
		{
			delete [] buff;
			throw Exception("Formatted output not successful in ToString<string>");
		}

		res += buff;
		if (i < v.size()-1)
			res += delim;
		else
			res += "\n";

		delete [] buff;
	}

	return res;
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
// class Mat
//------------------------------------------------------------------------------------------
void Mat::reset_chol_spo_cache() const
{
	delete [] chol_spo_cache;
	chol_spo_cache = 0;
}
//------------------------------------------------------------------------------------------
void Mat::reset_dsytrf_cache() const
{
	delete [] dsytrf_cache;
	delete [] dsytrf_ipiv;

	dsytrf_cache = 0;
	dsytrf_ipiv = 0;
}
//------------------------------------------------------------------------------------------
const double *Mat::chol_spo() const
{
	if (chol_spo_cache == 0)		// empty cache -> recalculate
	{
		if (icount != jcount)
			throw Exception("Non-square matrix in Mat::chol_spo");

		chol_spo_cache = new double[icount*icount];
		memcpy(chol_spo_cache, data.data(), icount*icount*sizeof(double));		// copy the matrix data

		int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'U', icount, chol_spo_cache, icount);
		if (info != 0)
		{
			reset_chol_spo_cache();
			std::string msg = stringFormatArr("DPOTRF завершилась с info {0:%d}", "DPOTRF exited with info {0:%d}", info);
			debug_output(msg + "\n", this);
			throw Exception(msg);
		}

		// make the lower triangular part zero
		for (size_t i = 1; i < icount; i++)
			for (size_t j = 0; j < i; j++)
				chol_spo_cache[i*jcount + j] = 0;

		cache_msg_to_file("chol_spo_cache: recalculating.....\n");
	}
	else
		cache_msg_to_file("chol_spo_cache: USING_CACHE!\n");

	return chol_spo_cache;
}
//------------------------------------------------------------------------------------------
void Mat::cache_msg_to_file(const std::string &msg) const			// debug output of 'msg' to file, SIMILAR to Cache<T>::MsgToFile()
{
#ifdef TEST_CACHE
	char fname[500];
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	sprintf(fname, TEST_CACHE, rank);
	FILE *f = fopen(fname, "a");
	if (f != NULL)
	{
		fputs(msg.c_str(), f);
		fclose(f);
	}
#endif
}
//------------------------------------------------------------------------------------------
void Mat::dsytrf(const double **A, const int **ipiv) const
{
	if (dsytrf_cache == 0)			// empty cache -> recalculate
	{
		if (icount != jcount)
			throw Exception("Non-square matrix in Mat::dsytrf");

		dsytrf_cache = new double[icount*icount];
		dsytrf_ipiv = new int[icount];
		memcpy(dsytrf_cache, data.data(), icount*icount*sizeof(double));		// copy the matrix data

		int info = LAPACKE_dsytrf(LAPACK_ROW_MAJOR, 'U', icount, dsytrf_cache, icount, dsytrf_ipiv);
		if (info != 0)
		{
			reset_dsytrf_cache();
			std::string msg = stringFormatArr("DSYTRF завершилась с info {0:%d}", "DSYTRF exited with info {0:%d}", info);
			debug_output(msg + "\n", this);
			throw Exception(msg);
		}

		// make the lower triangular part zero
		for (size_t i = 1; i < icount; i++)
			for (size_t j = 0; j < i; j++)
				dsytrf_cache[i*jcount + j] = 0;

		cache_msg_to_file("dsytrf_cache: recalculating.....\n");
	}
	else
		cache_msg_to_file("dsytrf_cache: USING_CACHE!\n");

	*A = dsytrf_cache;
	*ipiv = dsytrf_ipiv;
}
//------------------------------------------------------------------------------------------
void Mat::debug_output(const std::string &msg, const Mat *m) const
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	char fname[BUFFSIZE];
	sprintf(fname, debug_file.c_str(), rank);
	FILE *debug = fopen(fname, "w");

	if (debug != 0)			// write to file
	{
		fputs(msg.c_str(), debug);
		m->SaveASCII(debug, "%20.16g");
		fclose(debug);
	}
}
//------------------------------------------------------------------------------------------
Mat::Mat() : Vector2<double>(), op_switch(2), chol_spo_cache(0), dsytrf_cache(0), dsytrf_ipiv(0)
{
#ifdef TESTING
	std::cout << "Mat::Mat()" << std::endl;
#endif
}
//------------------------------------------------------------------------------------------
Mat::Mat(size_t N) : Vector2<double>(N, N, 0), op_switch(2), chol_spo_cache(0), dsytrf_cache(0), dsytrf_ipiv(0)
{
	for (size_t i = 0; i < N; i++)
		data[i*N + i] = 1;

#ifdef TESTING
	std::cout << "Mat::Mat(size_t N) - unity matrix" << std::endl;
#endif
}
//------------------------------------------------------------------------------------------
Mat::Mat(size_t I0, size_t J0, double val) : Vector2<double>(I0, J0, val), op_switch(2), chol_spo_cache(0), dsytrf_cache(0), dsytrf_ipiv(0)
{
#ifdef TESTING
	std::cout << "Mat::Mat(size_t I0, size_t J0, double val)" << std::endl;
#endif
}
//------------------------------------------------------------------------------------------
Mat::Mat(std::vector<double> v, size_t I0, size_t J0) : Vector2<double>(std::move(v), I0, J0), op_switch(2), chol_spo_cache(0), dsytrf_cache(0), dsytrf_ipiv(0)
{
#ifdef TESTING
	std::cout << "Mat::Mat(std::vector<double> v, size_t I0, size_t J0), std::move(v)" << std::endl;
#endif
}
//------------------------------------------------------------------------------------------
// initialize N x 1 "matrix" equal to vector "v" _OR_ N x N diagonal matrix with diagonal "v"
Mat::Mat(const std::vector<double> &v, bool IsDiag) : Vector2<double>(v, v.size(), 1), op_switch(2), chol_spo_cache(0), dsytrf_cache(0), dsytrf_ipiv(0)
{
	if (IsDiag)
	{
		const size_t N = v.size();
		*this = Mat(N, N, 0.0);
		for (size_t i = 0; i < N; i++)
			data[i*N + i] = v[i];
	}
#ifdef TESTING
	std::cout << "Mat::Mat(const std::vector<double> &v, bool IsDiag)" << std::endl;
#endif
}
//------------------------------------------------------------------------------------------
Mat::Mat(const Mat &m) : Vector2<double>(m), op_switch(m.op_switch), chol_spo_cache(0), dsytrf_cache(0), dsytrf_ipiv(0), delim(m.delim)
{
	if (m.chol_spo_cache != 0)
	{
		assert(icount == jcount);
		chol_spo_cache = new double[icount*icount];
		memcpy(chol_spo_cache, m.chol_spo_cache, icount*icount*sizeof(double));
	}

	if (m.dsytrf_cache != 0)
	{
		assert(icount == jcount);
		dsytrf_cache = new double[icount*icount];
		dsytrf_ipiv = new int[icount];
		memcpy(dsytrf_cache, m.dsytrf_cache, icount*icount*sizeof(double));
		memcpy(dsytrf_ipiv, m.dsytrf_ipiv, icount*sizeof(int));
	}

#ifdef TESTING
	std::cout << "Mat::Mat(const Mat &m)" << std::endl;
#endif
}
//------------------------------------------------------------------------------------------
Mat::Mat(Mat &&m) noexcept : Vector2<double>(std::move(m)), op_switch(m.op_switch), delim(m.delim)
{
	chol_spo_cache = m.chol_spo_cache;
	m.chol_spo_cache = 0;

	dsytrf_cache = m.dsytrf_cache;
	m.dsytrf_cache = 0;

	dsytrf_ipiv = m.dsytrf_ipiv;
	m.dsytrf_ipiv = 0;

#ifdef TESTING
	std::cout << "Mat::Mat(Mat &&m)" << std::endl;
#endif
}
//------------------------------------------------------------------------------------------
Mat::~Mat()
{
	reset_chol_spo_cache();
	reset_dsytrf_cache();

#ifdef TESTING
	std::cout << "Mat::~Mat()" << std::endl;
#endif
}
//------------------------------------------------------------------------------------------
const Mat &Mat::operator=(const Mat &m)
{
#ifdef TESTING
	std::cout << "Mat::operator=(const Mat &m)" << std::endl;
#endif

	Vector2<double>::operator=(m);
	op_switch = m.op_switch;
	delim = m.delim;

	reset_chol_spo_cache();
	if (m.chol_spo_cache != 0)
	{
		assert(icount == jcount);
		chol_spo_cache = new double[icount*icount];
		memcpy(chol_spo_cache, m.chol_spo_cache, icount*icount*sizeof(double));
	}
	else
		chol_spo_cache = 0;

	reset_dsytrf_cache();
	if (m.dsytrf_cache != 0)
	{
		assert(icount == jcount);
		dsytrf_cache = new double[icount*icount];
		dsytrf_ipiv = new int[icount];
		memcpy(dsytrf_cache, m.dsytrf_cache, icount*icount*sizeof(double));
		memcpy(dsytrf_ipiv, m.dsytrf_ipiv, icount*sizeof(int));
	}
	else
	{
		dsytrf_cache = 0;
		dsytrf_ipiv = 0;
	}

	return *this;
}
//------------------------------------------------------------------------------------------
const Mat &Mat::operator=(Mat &&m) noexcept	// TODO not much sure here
{
#ifdef TESTING
	std::cout << "Mat::operator=(Mat &&m)" << std::endl;
#endif

	Vector2<double>::operator=(std::move(m));
	op_switch = m.op_switch;
	delim = m.delim;

	chol_spo_cache = m.chol_spo_cache;
	m.chol_spo_cache = 0;

	dsytrf_cache = m.dsytrf_cache;
	m.dsytrf_cache = 0;

	dsytrf_ipiv = m.dsytrf_ipiv;
	m.dsytrf_ipiv = 0;

	return *this;
}
//------------------------------------------------------------------------------------------
void Mat::Reshape(size_t i, size_t j)
{
	reset_chol_spo_cache();
	reset_dsytrf_cache();

	if (icount*jcount != i*j)
		throw Exception("New dimensions in Mat::Reshape are not consistent with data length");

	icount = i;
	jcount = j;
}
//------------------------------------------------------------------------------------------
std::vector<double> &Mat::ToVectorMutable()
{
	reset_chol_spo_cache();
	reset_dsytrf_cache();
	return data;
}
//------------------------------------------------------------------------------------------
void Mat::Deserialize(const double *v)
{
	reset_chol_spo_cache();
	reset_dsytrf_cache();
	data = std::vector<double>(v, v + icount*jcount);
}
//------------------------------------------------------------------------------------------
void Mat::SetOpSwitch(int s)						// sets op_switch
{
	if (s != 1 && s != 2)
		throw Exception("Mat::SetOpSwitch requires s = 1 or 2");

	op_switch = s;
}
//------------------------------------------------------------------------------------------
void Mat::LoadASCII(FILE *f, int num)
{
	reset_chol_spo_cache();
	reset_dsytrf_cache();

	char line[LINEBUFF];
	std::vector<std::string> toks;
	size_t ci = 0, cj = 0;		// size of the loaded matrix
	std::vector<double> vec;	// data for the loaded matrix

	while (!feof(f) && ci < (size_t)num)		// size_t(-1) is a very big number
	{
		if (!fgets(line, LINEBUFF, f))
			break;				// may happen when empty line with EOF was read
		tokenize(line, toks, delim, true);
		if (toks.size() > 0)	// only take non-empty lines
		{
			if (cj == 0)
				cj = toks.size();				// first line sets the number of columns
			else if (cj != toks.size())
				throw Exception("Varying number of items per line in Mat::LoadASCII");	// lines after the first: check consistency
			for (size_t j = 0; j < cj; j++)
				vec.push_back(StoD(toks[j]));
			ci++;
		}
	}

	*this = Mat(vec, ci, cj);
}
//------------------------------------------------------------------------------------------
void Mat::SaveASCII(FILE *f, std::string fmt) const
{
	fputs(ToString(fmt).c_str(), f);
}
//------------------------------------------------------------------------------------------
std::string Mat::ToString(std::string fmt) const
{
	char buff[BUFFSIZE];
	std::string res;

	for (size_t i = 0; i < icount; i++)
		for (size_t j = 0;  j < jcount; j++)
		{
			sprintf(buff, fmt.c_str(), (*this)(i, j));
			res += buff;
			if (j < jcount-1)
				res += "\t";
			else
				res += "\n";
		}

	return res;
}
//------------------------------------------------------------------------------------------
void Mat::Bcast(int root, MPI_Comm comm)
{
	reset_chol_spo_cache();
	reset_dsytrf_cache();

	if (comm == MPI_COMM_NULL)
		return;

#ifdef TESTBCAST
	int rank;
	MPI_Comm_rank(comm, &rank);
	std::cout << "rank " << rank << "\tMat::Bcast, initial data.data() " << data.data() << ", icount = " << icount << ", jcount = " << jcount << "\n";
#endif

	int previous_size = icount*jcount;			// save the previous array size on each rank
	MPI_Bcast(&icount, 1, MPI_UNSIGNED_LONG, root, comm);
	MPI_Bcast(&jcount, 1, MPI_UNSIGNED_LONG, root, comm);

	if (previous_size != int(icount*jcount))
		*this = Mat(icount, jcount, 0.0);		// reallocate array

#ifdef TESTBCAST
	std::cout << "rank " << rank << "\tMat::Bcast, final data.data() " << data.data() << ", icount = " << icount << ", jcount = " << jcount << "\n";
#endif

	MPI_Bcast(data.data(), icount*jcount, MPI_DOUBLE, root, comm);
}
//------------------------------------------------------------------------------------------
Mat Mat::Tr() const				// transpose
{
	Mat res(jcount, icount, 0.0);
	for (size_t i = 0; i < jcount; i++)
		for (size_t j = 0; j < icount; j++)
			res.data[i*icount + j] = data[j*jcount + i];

	return res;
}
//------------------------------------------------------------------------------------------
double Mat::Trace() const
{
	if (icount != jcount)
		throw Exception("Вызов Mat::Trace для неквадратной матрицы", "Mat::Trace called for a non-square matrix");

	double res = 0;
	const double *p = data.data();
	for (size_t i = 0; i < icount; i++)
		res += p[i*jcount + i];

	return res;
}
//------------------------------------------------------------------------------------------
double Mat::Sum() const
{
	double res = 0;
	const double *p = data.data();
	size_t SZ = icount*jcount;
	for (size_t i = 0; i < SZ; i++)
		res += p[i];

	return res;
}
//------------------------------------------------------------------------------------------
double Mat::Max(int &i, int &j) const
{
	double max = std::numeric_limits<double>::lowest();
	int SZ = icount*jcount, ind = -1;
	for (int x = 0; x < SZ; x++)
		if (data[x] > max)
		{
			max = data[x];
			ind = x;
		}

	if (ind != -1)
	{
		i = ind / jcount;
		j = ind % jcount;
		return data[ind];
	}
	else
	{
		i = -1;
		j = -1;
		return std::numeric_limits<double>::quiet_NaN();
	}
}
//------------------------------------------------------------------------------------------
double Mat::Min(int &i, int &j) const
{
	double min = std::numeric_limits<double>::max();
	int SZ = icount*jcount, ind = -1;
	for (int x = 0; x < SZ; x++)
		if (data[x] < min)
		{
			min = data[x];
			ind = x;
		}

	if (ind != -1)
	{
		i = ind / jcount;
		j = ind % jcount;
		return data[ind];
	}
	else
	{
		i = -1;
		j = -1;
		return std::numeric_limits<double>::quiet_NaN();
	}
}
//------------------------------------------------------------------------------------------
double Mat::Norm1() const
{
	double res = 0;
	for (const auto &d : data)
		res += fabs(d);

	return res;
}
//------------------------------------------------------------------------------------------
double Mat::Norm2() const
{
	double res = 0;
	for (const auto &d : data)
		res += d*d;

	return sqrt(res);
}
//------------------------------------------------------------------------------------------
double Mat::NormInf() const
{
	double res = 0;
	for (const auto &d : data)
		if (fabs(d) > res)
			res = fabs(d);

	return res;
}
//------------------------------------------------------------------------------------------
void Mat::Func(const std::function<double (double)> &f)
{
	reset_chol_spo_cache();
	reset_dsytrf_cache();

	size_t SZ = icount*jcount;
	for (size_t i = 0; i < SZ; i++)
		data[i] = f(data[i]);
}
//------------------------------------------------------------------------------------------
void Mat::FuncInd(const std::function<double (int, int, double)> &f)
{
	reset_chol_spo_cache();
	reset_dsytrf_cache();

	for (size_t i = 0; i < icount; i++)
		for (size_t j = 0; j < jcount; j++)
			data[i*jcount + j] = f(i, j, data[i*jcount + j]);
}
//------------------------------------------------------------------------------------------
double &Mat::operator()(size_t i, size_t j)
{
	reset_chol_spo_cache();
	reset_dsytrf_cache();

	return data[i*jcount + j];
}
//------------------------------------------------------------------------------------------
const double &Mat::operator()(size_t i, size_t j) const
{
	return data[i*jcount + j];
}
//------------------------------------------------------------------------------------------
Mat operator&&(const Mat &m1, const Mat &m2)
{
	if (m1.icount == 0 || m1.jcount == 0)
		return m2;
	if (m2.icount == 0 || m2.jcount == 0)
		return m1;

	if (m1.icount != m2.icount)
		throw Exception(stringFormatArr("Inconsistent icount in operator&&(Mat, Mat): {0:%zu}, {1:%zu}", std::vector<size_t>{m1.icount, m2.icount}));

	Mat res(m1.icount, m1.jcount + m2.jcount, 0.0);
	double *pres = res.data.data();
	const double *pm1 = m1.data.data();
	const double *pm2 = m2.data.data();

	for (size_t i = 0; i < m1.icount; i++)
	{
		memcpy(&pres[i*res.jcount], &pm1[i*m1.jcount], m1.jcount * sizeof(double));
		memcpy(&pres[i*res.jcount + m1.jcount], &pm2[i*m2.jcount], m2.jcount * sizeof(double));
	}

	return res;
}
//------------------------------------------------------------------------------------------
Mat operator||(Mat m1, const Mat &m2)
{
	m1.reset_chol_spo_cache();
	m1.reset_dsytrf_cache();

	if (m1.icount == 0 || m1.jcount == 0)
		return m2;
	if (m2.icount == 0 || m2.jcount == 0)
		return m1;

	// m1 will be the result to which m2 is appended

	if (m1.jcount != m2.jcount)
		throw Exception(stringFormatArr("Inconsistent jcount in operator||(Mat, Mat): {0:%zu}, {1:%zu}", std::vector<size_t>{m1.jcount, m2.jcount}));

	m1.icount += m2.icount;
	m1.data.reserve(m1.icount * m1.jcount);
	m1.data.insert(m1.data.end(), m2.data.begin(), m2.data.end());	// take advantage of row major storage

	return m1;
}
//------------------------------------------------------------------------------------------
Mat Mat::Reorder(const std::vector<int> &ordi, const std::vector<int> &ordj) const
{
	return Mat(HMMPI::Reorder(data, icount, jcount, ordi, ordj), ordi.size(), ordj.size());
}
//------------------------------------------------------------------------------------------
Mat Mat::Reorder(int i0, int i1, int j0, int j1) const		// creates a submatrix with indices [i0, i1)*[j0, j1)
{
	if (i0 < 0 || i1 > (int)icount || j0 < 0 || j1 > (int)jcount)
		throw Exception(stringFormatArr("In Mat::Reorder indices [{0:%d}, {1:%d})*[{2:%d}, {3:%d}) are inconsistent with matrix dimension {4:%d} * {5:%d}",
						std::vector<int>{i0, i1, j0, j1, (int)icount, (int)jcount}));
	if (i0 >= i1 || j0 >= j1)
		throw Exception(stringFormatArr("In Mat::Reorder I-indices ({0:%d}, {1:%d}) and J-indices ({2:%d}, {3:%d}) should be strictly increasing",
						std::vector<int>{i0, i1, j0, j1}));

	std::vector<int> ordi(i1-i0);
	std::vector<int> ordj(j1-j0);
	std::iota(ordi.begin(), ordi.end(), i0);
	std::iota(ordj.begin(), ordj.end(), j0);

	return Reorder(ordi, ordj);
}
//------------------------------------------------------------------------------------------
Mat Mat::operator+(Mat m) const
{
	// m is a copy or rvalue

	m.reset_chol_spo_cache();
	m.reset_dsytrf_cache();
	if (icount != m.icount || jcount != m.jcount)
		throw Exception("Inconsistent dimensions in Mat::operator+");

	size_t SZ = icount*jcount;
	double *pres = m.data.data();
	for (size_t i = 0; i < SZ; i++)
		pres[i] += data[i];

	return m;
}
//------------------------------------------------------------------------------------------
Mat Mat::operator-(Mat m) const
{
	// m is a copy or rvalue

	m.reset_chol_spo_cache();
	m.reset_dsytrf_cache();
	if (icount != m.icount || jcount != m.jcount)
		throw Exception(stringFormatArr("Inconsistent dimensions in Mat::operator- ({0:%zu} x {1:%zu}), ({2:%zu} x {3:%zu})", std::vector<size_t>{icount, jcount, m.icount, m.jcount}));

	size_t SZ = icount*jcount;
	double *pres = m.data.data();
	for (size_t i = 0; i < SZ; i++)
		pres[i] = data[i] - pres[i];

	return m;
}
//------------------------------------------------------------------------------------------
void Mat::operator+=(const Mat &m)
{
	reset_chol_spo_cache();
	reset_dsytrf_cache();

	if (icount != m.icount || jcount != m.jcount)
		throw Exception("Inconsistent dimensions in Mat::operator+=");

	size_t SZ = icount*jcount;
	const double *pm = m.data.data();
	for (size_t i = 0; i < SZ; i++)
		data[i] += pm[i];
}
//------------------------------------------------------------------------------------------
void Mat::operator-=(const Mat &m)
{
	reset_chol_spo_cache();
	reset_dsytrf_cache();

	if (icount != m.icount || jcount != m.jcount)
		throw Exception("Inconsistent dimensions in Mat::operator-=");

	size_t SZ = icount*jcount;
	const double *pm = m.data.data();
	for (size_t i = 0; i < SZ; i++)
		data[i] -= pm[i];
}
//------------------------------------------------------------------------------------------
double InnerProd(const Mat &a, const Mat &b)	// (a, b), inner product of two vectors; using Manual | BLAS depending on 'a.op_switch'
{
	if (a.jcount != 1 || b.jcount != 1 || a.icount != b.icount)
		throw Exception(stringFormatArr("Inner product should be applied to vectors of equal size ({0:%zu} != 1 || {1:%zu} != 1 || {2:%zu} != {3:%zu})", std::vector<size_t>{a.jcount, b.jcount, a.icount, b.icount}));

	const int swtch = a.GetOpSwitch();
	if (swtch == 1)
		return ManualMath::InnerProd(a.data, b.data);
	else if (swtch == 2)
		return InnerProd(a.data, b.data);
	else
		throw Exception("Bad a.op_switch in InnerProd(Mat, Mat)");
}
//------------------------------------------------------------------------------------------
Mat OuterProd(const Mat &a, const Mat &b)
{
	if (a.jcount != 1 || b.jcount != 1)
		throw Exception(stringFormatArr("Outer product should be applied to column-vectors ({0:%zu} != 1 || {1:%zu} != 1)", std::vector<size_t>{a.jcount, b.jcount}));

	Mat res(a.icount, b.icount, 0);
	double *pres = res.data.data();
	const double *pa = a.data.data();
	const double *pb = b.data.data();

	for (size_t i = 0; i < res.icount; i++)
		for (size_t j = 0; j < res.jcount; j++)
			pres[res.jcount*i + j] = pa[i] * pb[j];

	return res;
}
//------------------------------------------------------------------------------------------
Mat VecProd(const Mat &a, const Mat &b)			// a (x) b, vector product of two 3-dim vectors
{
	if (a.icount != 3 || b.icount != 3 || a.jcount != 1 || b.jcount != 1)
		throw Exception(stringFormatArr("Vector product should be applied to 3-dim column-vectors ({0:%zu} != 3 || {1:%zu} != 3 || {2:%zu} != 1 || {3:%zu} != 1)",
										std::vector<size_t>{a.icount, b.icount, a.jcount, b.jcount}));
	Mat res(3, 1, 0.0);
	const double *A = a.data.data();
	const double *B = b.data.data();
	double *dat_res = res.data.data();

	dat_res[2] = A[0]*B[1] - A[1]*B[0];

	if (A[2] != 0 || B[2] != 0)					// vectors have z-components
	{
		dat_res[1] = -(A[0]*B[2] - A[2]*B[0]);
		dat_res[0] = A[1]*B[2] - A[2]*B[1];
	}

	return res;
}
//------------------------------------------------------------------------------------------
Mat operator*(double d, Mat m)		// number * Mat
{
	// m is a copy or rvalue

	m.reset_chol_spo_cache();
	m.reset_dsytrf_cache();
	size_t SZ = m.icount*m.jcount;
	double *pm = m.data.data();
	for (size_t i = 0; i < SZ; i++)
		pm[i] *= d;

	return m;
}
//------------------------------------------------------------------------------------------
Mat operator%(const std::vector<double> &v, Mat m)	// diag * Mat
{
	m.reset_chol_spo_cache();
	m.reset_dsytrf_cache();

	if (v.size() != m.icount)
		throw Exception("Inconsistent dimensions in diag(vector) % Mat");

	for (size_t i = 0; i < m.icount; i++)
		for (size_t j = 0; j < m.jcount; j++)
			m.data[i*m.jcount + j] *= v[i];

	return m;
}
//------------------------------------------------------------------------------------------
Mat operator%(Mat m, const std::vector<double> &v)	// Mat * diag
{
	m.reset_chol_spo_cache();
	m.reset_dsytrf_cache();

	if (v.size() != m.jcount)
		throw Exception("Inconsistent dimensions in Mat % diag(vector)");

	for (size_t i = 0; i < m.icount; i++)
		for (size_t j = 0; j < m.jcount; j++)
			m.data[i*m.jcount + j] *= v[j];

	return m;
}
//------------------------------------------------------------------------------------------
Mat Mat::operator*(const Mat &m) const		// *this * Mat
{
	if (jcount != m.icount)
		throw Exception("Inconsistent dimensions in Mat::operator*(Mat)");

	size_t sz_I = icount;
	size_t sz_J = m.jcount;
	size_t sz = jcount;

	Mat res(sz_I, sz_J, 0.0);
	const double *p = data.data();
	const double *pm = m.data.data();
	double *pres = res.data.data();

	for (size_t i = 0; i < sz_I; i++)
		for (size_t j = 0; j < sz_J; j++)
		{
			double sum = 0;
			for (size_t v = 0; v < sz; v++)
				sum += p[sz*i + v]*pm[sz_J*v + j];	// row-major storage!

			pres[sz_J*i + j] = sum;
		}

	return res;
}
//------------------------------------------------------------------------------------------
std::vector<double> Mat::operator*(const std::vector<double> &v) const		// *this * v, using Manual | BLAS depending on 'op_switch'
{
	if (jcount != v.size())
		throw Exception("Inconsistent dimensions in Mat::operator*(vector)");

	if (op_switch == 1)
	{
		Mat res = (*this)*Mat(v);
		return res.ToVector();
	}
	else if (op_switch == 2)
	{
		std::vector<double> res(icount);
		const int lda = jcount;
		const double alpha = 1;
		const double beta = 0;
		cblas_dgemv(CblasRowMajor, CblasNoTrans, icount, jcount, alpha, data.data(), lda, v.data(), 1, beta, res.data(), 1);

		return res;
	}
	else
		throw Exception("Bad op_switch in Mat::operator*(vector)");
}
//------------------------------------------------------------------------------------------
Mat operator/(Mat A, Mat b)
{
	// A and b are changed in this function
	// but they are copies or rvalues

	A.reset_chol_spo_cache();
	b.reset_chol_spo_cache();
	A.reset_dsytrf_cache();
	b.reset_dsytrf_cache();

	size_t icount = A.icount;
	if (A.icount != A.jcount)
		throw Exception("Non-square matrix in operator/(Mat, Mat)");
	size_t rhscount = b.jcount;
	if (icount != b.icount)
		throw Exception(stringFormatArr("Dimensions mismatch in operator/(Mat, Mat), A[{0:%ld} x {1:%ld}], b[{2:%ld} x {3:%ld}]", std::vector<size_t>{A.icount, A.jcount, b.icount, b.jcount}));

	Mat res(icount, rhscount, 0.0);
	double *pwork = A.data.data();
	double *pm = b.data.data();
	double *pres = res.data.data();

	std::vector<int> piv(icount);			// pivots reordering
	std::iota(piv.begin(), piv.end(), 0);	// fill with 0, 1, 2,...

	try
	{
		for (size_t i = 0; i < icount; i++)
		{
			size_t max_i = i;		// newly found pivot
			double max = fabs(pwork[piv[max_i]*icount + i]);
			for (size_t j = i+1; j < icount; j++)
				if (fabs(pwork[piv[j]*icount + i]) > max)
				{
					max_i = j;
					max = fabs(pwork[piv[j]*icount + i]);
				}

			std::swap(piv[i], piv[max_i]);

	#ifdef TESTING
			if (i != max_i)
				std::cout << "operator/(Mat, Mat) swapping rows " << i << " and " << max_i << std::endl;
	#endif

			// for the lines below piv is fixed
			int pivi = piv[i];
			if (pwork[pivi*icount + i] == 0)
				throw Exception("Determinant = 0, no solution exists in operator/(Mat, Mat)");

			for (size_t j = i+1; j < icount; j++)
			{
				int pivj = piv[j];
				if (pwork[pivj*icount + i] != 0)
				{
					double mult = -pwork[pivj*icount + i] / pwork[pivi*icount + i];
					for (size_t k = i+1; k < icount; k++)
						pwork[pivj*icount + k] += pwork[pivi*icount + k] * mult;

					pwork[pivj*icount + i] = 0;

					// pm[pivj] += pm[pivi] * mult;		-- old version - for 1 RHS
					for (size_t k = 0; k < rhscount; k++)
						pm[pivj*rhscount + k] += pm[pivi*rhscount + k] * mult;
				}
			}
		}

		for (int i = icount-1; i >= 0; i--)
		{
			int pivi = piv[i];
			if (pwork[pivi*icount + i] == 0)
				throw Exception("Determinant = 0, no solution exists in operator/(Mat, Mat)");

			for (size_t k = 0; k < rhscount; k++)
			{
				double aux = pm[pivi*rhscount + k];
				for (size_t j = i+1; j < icount; j++)
					aux -= pwork[pivi*icount + j] * pres[j*rhscount + k];

				pres[i*rhscount + k] = aux / pwork[pivi*icount + i];
			}
		}
	}
	catch (const Exception &e)
	{
		std::string msg = e.what() + std::string("\nTransformed matrix 'A' which caused the exception\n");
		A.debug_output(msg, &A);

		throw e;
	}

	return res;
}
//------------------------------------------------------------------------------------------
Mat Mat::Autocorr() const
{
	if (icount <= 1 || jcount != 1)
		throw Exception("Autocorr() should be applied to N x 1 vectors, N > 1");

	Mat res(icount, 1, 0.0);
	double *pres = res.data.data();
	const double *p = data.data();
	double mean = Sum()/icount;

	for (size_t k = 0; k < icount; k++)
	{
		double d = 0;
		for (size_t t = 0; t < icount-k; t++)
			d += (p[t] - mean)*(p[t+k] - mean);
		pres[k] = d/(icount-1);
		if (k > 0)
			pres[k] /= pres[0];		// normalization
	}
	pres[0] /= pres[0];		// normalization

	return res;
}
//------------------------------------------------------------------------------------------
int Mat::Ess(double &res) const
{
	Mat ac = Autocorr();
	size_t N = icount/2;
	const double *pac = ac.data.data();

    res = 0;
    int lag = 0;
    double G_prev = std::numeric_limits<double>::max();
    for (size_t m = 0; m < N; m++)
    {
    	double G = pac[2*m] + pac[2*m+1];	// Gamma_m
		if (G > 0 && G < G_prev)
		{
			res += G;
			lag = 2*m + 1;
		}
		else
			break;

    	G_prev = G;
    }

    res = (double)icount/(2*res - 1);
    return lag;
}
//------------------------------------------------------------------------------------------
Mat Mat::Chol() const
{
	if (icount != jcount)
		throw Exception("Non-square matrix in Mat::Chol");

	Mat res(icount, jcount, 0.0);
	CholDecomp(data.data(), res.data.data(), icount);

	return res;
}
//------------------------------------------------------------------------------------------
Mat Mat::CholSPO() const
{
	if (icount != jcount)
		throw Exception("Non-square matrix in Mat::CholSPO");

	const double *A = chol_spo();
	Mat res(std::vector<double>(A, A + icount*icount), icount, icount);

	return res;
}
//------------------------------------------------------------------------------------------
Mat Mat::InvSPO() const
{
	if (icount != jcount)
		throw Exception("Non-square matrix in Mat::InvSPO");

	std::vector<double> UtU(chol_spo(), chol_spo() + icount*icount);
	double *A = UtU.data();
	int info = LAPACKE_dpotri(LAPACK_ROW_MAJOR, 'U', icount, A, icount);
	if (info != 0)
	{
		std::string msg = stringFormatArr("DPOTRI завершилась с info {0:%d}", "DPOTRI exited with info {0:%d}", info);
		Mat temp(std::vector<double>(chol_spo(), chol_spo() + icount*icount), icount, icount);
		debug_output(msg + "\n", &temp);
		throw Exception(msg);
	}

	// symmetrically fill the lower triangular part
	for (size_t i = 1; i < icount; i++)
		for (size_t j = 0; j < i; j++)
			A[i*jcount + j] = A[j*jcount + i];

	Mat res(std::move(UtU), icount, icount);
	return res;
}
//------------------------------------------------------------------------------------------
Mat Mat::InvU() const						// inverse of the upper triangular matrix [the upper triangle of *this is used]
{
	if (icount != jcount)
		throw Exception("Non-square matrix in Mat::InvU");

	Mat res = *this;
	int info = LAPACKE_dtrtri(LAPACK_ROW_MAJOR, 'U', 'N', icount, res.data.data(), jcount);
	if (info != 0)
	{
		std::string msg = stringFormatArr("DTRTRI завершилась с info {0:%d}", "DTRTRI exited with info {0:%d}", info);
		debug_output(msg + "\n", this);
		throw Exception(msg);
	}

	// fill the lower triangular part with zeros
	for (size_t i = 1; i < icount; i++)
		for (size_t j = 0; j < i; j++)
			res(i, j) = 0;

	return res;
}
//------------------------------------------------------------------------------------------
double Mat::LnDetSPO() const
{
	if (icount != jcount)
		throw Exception("Non-square matrix in Mat::LnDetSPO");

	const double *A = chol_spo();
	double res = 0;
	for (size_t i = 0; i < icount; i++)
		res += log(A[i*jcount + i]);

	return 2*res;
}
//------------------------------------------------------------------------------------------
Mat Mat::InvSY() const
{
	if (icount != jcount)
		throw Exception("Non-square matrix in Mat::InvSY");

	const double *A;		// DSYTRF decomposition results
	const int *ipiv;
	dsytrf(&A, &ipiv);

	std::vector<double> res(A, A + icount*icount);		// A is copied to "res"
	int info = LAPACKE_dsytri(LAPACK_ROW_MAJOR, 'U', icount, res.data(), icount, ipiv);
	if (info != 0)
	{
		std::string msg = stringFormatArr("DSYTRI завершилась с info {0:%d}, исходная матрица:", "DSYTRI exited with info {0:%d}, original matrix:", info);
		debug_output(msg + "\n", this);
		throw Exception(msg);
	}

	// symmetrically fill the lower triangular part
	for (size_t i = 1; i < icount; i++)
		for (size_t j = 0; j < i; j++)
			res[i*jcount + j] = res[j*jcount + i];

	return Mat(std::move(res), icount, icount);
}
//------------------------------------------------------------------------------------------
double Mat::LnDetSY(int &sign) const
{
	if (icount != jcount)
		throw Exception("Non-square matrix in Mat::LnDetSY");

	const double *A;		// DSYTRF decomposition results
	const int *ipiv;		// ipiv[i] are 1-based indices
	dsytrf(&A, &ipiv);

	double res = 0;
	sign = 1;
	for (size_t i = 0; i < icount; i++)
		if (ipiv[i] > 0 )	// 1x1 block
		{
			double d = A[i*jcount + i];
			if (d < 0)
			{
				d = -d;
				sign = -sign;
			}
			res += log(d);
		}
		else				// 2x2 block
		{
			assert(i < icount-1);
			double d = A[i*jcount + i]*A[(i+1)*jcount + i+1] - A[i*jcount + i+1]*A[i*jcount + i+1];
			if (d < 0)
			{
				d = -d;
				sign = -sign;
			}
			res += log(d);
			i++;
		}

	return res;
}
//------------------------------------------------------------------------------------------
Mat Mat::SymSqrt() const
{
	Mat Vec;
	std::vector<double> val = EigVal(0, ICount(), Vec);		// *this = Vec*diag(val)*Vec'

	for (size_t i = 0; i < val.size(); i++)		// find square root of diagonal
	{
		if (val[i] < 0)
			throw Exception("Negative eigenvalue in Mat::SymSqrt");
		val[i] = sqrt(val[i]);
	}

	Mat Res = val % Vec.Tr();
	return Vec * Res;
}
//------------------------------------------------------------------------------------------
std::vector<double> Mat::EigVal(size_t I0, size_t I1) const
{
	if (icount != jcount)
		throw Exception("Non-square matrix in Mat::EigVal");
	if (I0 >= I1 || I1 > icount)
		throw Exception("Wrong indices I0, I1 in Mat::EigVal");

	double *W = new double[icount];						// storage for eigenvalues
	double *A = new double[icount*icount];				// copy of the matrix data (it will be destroyed by the procedure)
	memcpy(A, data.data(), icount*icount*sizeof(double));

	int lda = icount;
	int M;												// number of eigenvalues found
	double abstol = LAPACKE_dlamch('S');
	int info = LAPACKE_dsyevr(LAPACK_ROW_MAJOR, 'N', 'I', 'U', icount, A, lda, 0, 0, I0+1, I1, abstol, &M, W, 0, icount, 0);
	if (info != 0)
	{
		delete [] W;
		delete [] A;

		std::string msg = stringFormatArr("DSYEVR завершилась с info {0:%d}", "DSYEVR exited with info {0:%d}", info);
		debug_output(msg + "\n", this);
		throw Exception(msg);
	}

	std::vector<double> res(W, W + M);

	delete [] W;
	delete [] A;

#ifdef TESTING
	std::cout << "Mat::EigVal found " << M << " eigenvalues" << std::endl;
#endif

	return res;
}
//------------------------------------------------------------------------------------------
std::vector<double> Mat::EigVal(size_t I0, size_t I1, Mat &EigVec) const
{
	EigVec.reset_chol_spo_cache();
	EigVec.reset_dsytrf_cache();

	if (icount != jcount)
		throw Exception("Non-square matrix in Mat::EigVal");
	if (I0 >= I1 || I1 > icount)
		throw Exception("Wrong indices I0, I1 in Mat::EigVal");

	int M = (int)I1 - (int)I0;							// number of eigenvalues found (well, its value is known from indices)

	double *W = new double[icount];						// storage for eigenvalues
	double *A = new double[icount*icount];				// copy of the matrix data (it will be destroyed by the procedure)
	double *Z = new double[icount*M];					// storage for eigenvectors
	int *isuppz = new int[2*M];
	memcpy(A, data.data(), icount*icount*sizeof(double));

	int lda = icount;
	int ldz = M;		// ldz = icount for column-major

	double abstol = LAPACKE_dlamch('S');
	int info = LAPACKE_dsyevr(LAPACK_ROW_MAJOR, 'V', 'I', 'U', icount, A, lda, 0, 0, I0+1, I1, abstol, &M, W, Z, ldz, isuppz);
	if (info != 0)
	{
		delete [] W;
		delete [] A;
		delete [] Z;
		delete [] isuppz;

		std::string msg = stringFormatArr("DSYEVR завершилась с info {0:%d}", "DSYEVR exited with info {0:%d}", info);
		debug_output(msg + "\n", this);
		throw Exception(msg);
	}
	assert(M == (int)I1 - (int)I0);		// M should not have changed!

	std::vector<double> res(W, W + M);
	EigVec = HMMPI::Mat(std::vector<double>(Z, Z + icount*M), icount, M);

	delete [] W;
	delete [] A;
	delete [] Z;

#ifdef TESTING
	std::cout << "Mat::EigVal found " << M << " eigenvalues" << std::endl;
	std::cout << "isuppz = " << HMMPI::ToString(std::vector<int>(isuppz, isuppz + 2*M), "%d");
#endif

	delete [] isuppz;

	return res;
}
//------------------------------------------------------------------------------------------
std::vector<double> Mat::SgVal() const
{
	double *A = new double[icount*jcount];					// copy of the matrix data (it will be destroyed by the procedure)
	memcpy(A, data.data(), icount*jcount*sizeof(double));

	int res_size = (icount < jcount) ? icount : jcount;		// min
	std::vector<double> res(res_size);

	int lda = jcount;
	int info = LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'N', icount, jcount, A, lda, res.data(), NULL, icount, NULL, jcount);
	if (info != 0)
	{
		delete [] A;
		std::string msg = stringFormatArr("DGESDD завершилась с info {0:%d}", "DGESDD exited with info {0:%d}", info);
		debug_output(msg + "\n", this);
		throw Exception(msg);
	}

	delete [] A;
	return res;
}
//------------------------------------------------------------------------------------------
double Mat::ICond1SPO() const
{
	if (icount != jcount)
		throw Exception("Non-square matrix in Mat::ICond1SPO");

	int lda = jcount;
	double norm1 = LAPACKE_dlange(LAPACK_ROW_MAJOR, '1', icount, jcount, data.data(), lda);

	double res;
	const Mat chol = CholSPO();

	int info = LAPACKE_dpocon(LAPACK_ROW_MAJOR, 'U', icount, chol.data.data(), lda, norm1, &res);
	if (info != 0)
	{
		std::string msg = stringFormatArr("DPOCON завершилась с info {0:%d}", "DPOCON exited with info {0:%d}", info);
		debug_output(msg + "\n", &chol);
		throw Exception(msg);
	}
	return res;
}
//------------------------------------------------------------------------------------------
Mat Mat::BFGS_update_B(const Mat &dk, const Mat &gk) const
{
	Mat Bdelta = (*this)*dk;
	double gamma_delta = InnerProd(gk, dk);
	double delta_B_delta = InnerProd(dk, Bdelta);

	if (gamma_delta <= 0)	// DEBUG check -- to reject some points
	{
		int rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);	// DEBUG
		if (rank == 0)
			std::cout << "rejected dk*gk " << gamma_delta << "\n";	// DEBUG
		return *this;		// DEBUG
	}

	return (*this) + (1/gamma_delta)*OuterProd(gk, gk) + (-1/delta_B_delta)*OuterProd(Bdelta, Bdelta);
}
//------------------------------------------------------------------------------------------
Mat Mat::BFGS_update_B(const std::vector<Mat> &Xk, const std::vector<Mat> &Gk) const
{
	if (Xk.size() != Gk.size())
		throw Exception("Xk.size() != Gk.size() in Mat::BFGS_update_B");
	if (Xk.size() < 2)
		throw Exception("Xk.size() < 2 in Mat::BFGS_update_B");

	// fill delta and gamma - coordinate and gradient changes
	std::vector<Mat> dk(Xk.size()-1);
	std::vector<Mat> gk(Xk.size()-1);
	for (size_t i = 0; i < Xk.size()-1; i++)
	{
		dk[i] = Xk[i+1] - Xk[i];
		gk[i] = Gk[i+1] - Gk[i];
	}

	// make a series of BFGS updates
	Mat res = BFGS_update_B(dk[0], gk[0]);
	for (size_t i = 1; i < dk.size(); i++)
		res = res.BFGS_update_B(dk[i], gk[i]);

	return res;
}
//------------------------------------------------------------------------------------------
// RandNormal
//------------------------------------------------------------------------------------------
double RandNormal::get()
{
	if (hold)
	{
		hold = false;
		return tmp;
	}
	else
	{
		hold = true;
		double a1 = double(rand())/(double(RAND_MAX) + 1);
		double a2 = double(rand())/(double(RAND_MAX) + 1);
		double aux = sqrt(-2*log(a1));

		tmp = aux*cos(2*PI*a2);
		return aux*sin(2*PI*a2);
	}
}
//------------------------------------------------------------------------------------------
std::vector<double> RandNormal::get(int n)
{
	std::vector<double> res(n);
	for (auto &i : res)
		i = get();

	return res;
}
//------------------------------------------------------------------------------------------
// Func1D
//------------------------------------------------------------------------------------------
double Func1D::inv(double y) const					// inverse function value
{
	throw Exception("Illegal call to Func1D::inv");
}
//------------------------------------------------------------------------------------------
// Func1D_pwlin
//------------------------------------------------------------------------------------------
size_t Func1D_pwlin::locate_point(const std::vector<double> &vec, const double x) const		// returns a "rough index" of 'x' in array 'vec' (which should be SORTED in increasing order), namely:
{																							// for x <= vec[0] returns 0, for vec[last] < x returns vec.size()
	assert(vec.size() > 0);																	// for vec[i-1] < x <= vec[i] returns 'i'
	if (locate_cache > 0 && locate_cache < vec.size() && vec[locate_cache-1] < x && x <= vec[locate_cache])
		return locate_cache;
	else if (locate_cache == vec.size() && vec[locate_cache-1] < x)
		return locate_cache;
	else if (locate_cache == 0 && x <= vec[locate_cache])
		return locate_cache;
	else
	{
		locate_cache = std::lower_bound(vec.begin(), vec.end(), x) - vec.begin();
		return locate_cache;
	}
}
//------------------------------------------------------------------------------------------
Func1D_pwlin::Func1D_pwlin(std::vector<double> x, std::vector<double> y) : xi(std::move(x)), yi(std::move(y)), locate_cache(0)
{
	// check the sizes
	if (xi.size() != yi.size())
		throw Exception("Input arrays differ in size in Func1D_pwlin::Func1D_pwlin");

	// check monotonicity of 'xi'
	if (!is_strictly_sorted(xi.begin(), xi.end()))
		throw Exception("Array 'xi' should be strictly increasing in Func1D_pwlin::Func1D_pwlin");

	dri = std::vector<double>(xi.size(), 0.0);		// fill the auxiliary array
	for (size_t k = 1; k < xi.size(); k++)
		dri[k] = (yi[k] - yi[k-1])/(xi[k] - xi[k-1]);
}
//------------------------------------------------------------------------------------------
double Func1D_pwlin::val(double x) const
{
	size_t ind = locate_point(xi, x);
	if (ind == 0 || ind == xi.size())
		return 0;
	else
		return yi[ind-1] + dri[ind]*(x - xi[ind-1]);
}
//------------------------------------------------------------------------------------------
// Func1D_CDF
//------------------------------------------------------------------------------------------
Func1D_CDF::Func1D_CDF(std::vector<double> x, std::vector<double> y) : Func1D_pwlin(std::move(x), std::move(y))	// the input is a pdf; the resulting CDF should be strictly increasing
{
	Fi = std::vector<double>(xi.size(), 0.0);

	for (size_t k = 1; k < Fi.size(); k++)
		Fi[k] = Fi[k-1] + (yi[k] + yi[k-1])*(xi[k] - xi[k-1])/2;

	if (!is_strictly_sorted(Fi.begin(), Fi.end()))
		throw Exception("CDF (Fi) should be strictly increasing in Func1D_CDF::Func1D_CDF");

	for (size_t k = 0; k < yi.size(); k++)
		if (yi[k] < 0)
			throw Exception("All 'yi' should be non-negative in Func1D_CDF::Func1D_CDF");

	const double norm = *--Fi.end(); 			// normalize to 1.0
	for (size_t k = 0; k < Fi.size(); k++)
	{
		yi[k] /= norm;
		dri[k] /= norm;
		Fi[k] /= norm;
	}
}
//------------------------------------------------------------------------------------------
double Func1D_CDF::val(double x) const
{
	size_t ind = locate_point(xi, x);
	if (ind == 0)
		return 0;
	else if (ind == xi.size())
		return 1;
	else
		return Fi[ind-1] + (yi[ind-1] + dri[ind]*(x - xi[ind-1])/2)*(x - xi[ind-1]);
}
//------------------------------------------------------------------------------------------
double Func1D_CDF::inv(double y) const							// inverse CDF
{
	if (y < 0 || y > 1)
		throw Exception("'y' out of range [0, 1] in Func1D_CDF::inv");

	size_t ind = locate_point(Fi, y);
	if (ind == 0)
		return xi[0];
	else if (ind == Fi.size())
		return *--xi.end();
	else
		return xi[ind-1] + 2*(y - Fi[ind-1])/(sqrt(yi[ind-1]*yi[ind-1] + 2*dri[ind]*(y - Fi[ind-1])) + yi[ind-1]);
}
//------------------------------------------------------------------------------------------
// Func1D_corr
//------------------------------------------------------------------------------------------
double Func1D_corr::d3f(double x) const
{
	throw Exception("Illegal call to Func1D_corr::d3f");
}
//------------------------------------------------------------------------------------------
double Func1D_corr::lim_df(double y) const			// f'(y)/y
{
	if (y == 0)
		return std::numeric_limits<double>::infinity();
	else
		return df(y)/y;
}
//------------------------------------------------------------------------------------------
double Func1D_corr::lim_d2f(double y) const			// [f''(y) - f'(y)/y]/(y^2)
{
	if (y == 0)
		return std::numeric_limits<double>::infinity();
	else
		return (d2f(y) - df(y)/y)/(y*y);
}
//------------------------------------------------------------------------------------------
double Func1D_corr::lim_d3f(double y) const			// [3*f''/y - 3*f'/(y^2) - f''']/(y^3)
{
	if (y == 0)
		return std::numeric_limits<double>::infinity();
	else
	{
		double y2 = y*y;
		return (3*(d2f(y)/y - df(y)/y2) - d3f(y))/(y*y2);
	}
}
//------------------------------------------------------------------------------------------
// CorrGauss
//------------------------------------------------------------------------------------------
double CorrGauss::f(double x, bool smooth_at_nugget) const
{
	double mult = 1-nugget;
	if (x == 0 && !smooth_at_nugget)
		mult = 1;

	return mult*exp(-3*x*x);
}
//------------------------------------------------------------------------------------------
double CorrGauss::df(double x) const
{
	double mult = 1-nugget;
	return -mult*6*x*exp(-3*x*x);
}
//------------------------------------------------------------------------------------------
double CorrGauss::d2f(double x) const
{
	double mult = 1-nugget;
	return mult*(36*x*x - 6)*exp(-3*x*x);
}
//------------------------------------------------------------------------------------------
double CorrGauss::d3f(double x) const
{
	double mult = 1-nugget;
	return mult*108*x*(1 - 2*x*x)*exp(-3*x*x);
}
//------------------------------------------------------------------------------------------
double CorrGauss::lim_df(double y) const
{
	return (1-nugget)*(-6)*exp(-3*y*y);		// NOTE (1-nugget) is used even for y == 0
}
//------------------------------------------------------------------------------------------
double CorrGauss::lim_d2f(double y) const
{
	return (1-nugget)*36*exp(-3*y*y);
}
//------------------------------------------------------------------------------------------
double CorrGauss::lim_d3f(double y) const
{
	return (1-nugget)*216*exp(-3*y*y);
}
//------------------------------------------------------------------------------------------
Func1D_corr* CorrGauss::Copy() const
{
	return new CorrGauss(*this);
}
//------------------------------------------------------------------------------------------
// CorrSpher
//------------------------------------------------------------------------------------------
double CorrSpher::f(double x, bool smooth_at_nugget) const
{
    if (x <= 1)
        return 1 - 1.5*x + 0.5*x*x*x;
    else
        return 0;
}
//------------------------------------------------------------------------------------------
double CorrSpher::df(double x) const
{
    if (x <= 1)
        return -1.5 + 1.5*x*x;
    else
        return 0;
}
//------------------------------------------------------------------------------------------
double CorrSpher::d2f(double x) const
{
    if (x <= 1)
        return 3*x;
    else
        return 0;
}
//------------------------------------------------------------------------------------------
// CorrExp
//------------------------------------------------------------------------------------------
double CorrExp::f(double x, bool smooth_at_nugget) const
{
	return exp(-3*x);
}
//------------------------------------------------------------------------------------------
double CorrExp::df(double x) const
{
	return -3*exp(-3*x);
}
//------------------------------------------------------------------------------------------
double CorrExp::d2f(double x) const
{
	return 9*exp(-3*x);
}
//------------------------------------------------------------------------------------------
// VarGauss
//------------------------------------------------------------------------------------------
double VarGauss::f(double x, bool smooth_at_nugget) const
{
	return -expm1(-3*x*x);
}
//------------------------------------------------------------------------------------------
double VarGauss::df(double x) const
{
	return 6*x*exp(-3*x*x);
}
//------------------------------------------------------------------------------------------
double VarGauss::d2f(double x) const
{
	return (6 - 36*x*x)*exp(-3*x*x);
}
//------------------------------------------------------------------------------------------
// BesselMod2k
//------------------------------------------------------------------------------------------
double BesselMod2k::Kn(double Nu, double x) const
{
	gsl_sf_result res;
	int e = gsl_sf_bessel_Knu_e(Nu, x, &res);
	if (e != GSL_SUCCESS)
		throw Exception(stringFormatArr("gsl_sf_bessel_Knu_e завершилась со статусом {0:%d}", "gsl_sf_bessel_Knu_e returned with status {0:%d}", e));

	return res.val;
	// error: res.err
}
//------------------------------------------------------------------------------------------
double BesselMod2k::f(double x, bool smooth_at_nugget) const
{
	return Kn(nu, x);
}
//------------------------------------------------------------------------------------------
double BesselMod2k::df(double x) const
{
	return Kn(nu, x)*nu/x - Kn(nu+1, x);
}
//------------------------------------------------------------------------------------------
double BesselMod2k::d2f(double x) const
{
	return (nu*nu - nu)/(x*x)*Kn(nu, x) - (2*nu + 1)/x*Kn(nu+1, x) + Kn(nu+2, x);
}
//------------------------------------------------------------------------------------------
double BesselMod2k::d3f(double x) const
{
	double nu2 = nu*nu;
	return (nu2 - 3*nu + 2)*nu/(x*x*x)*Kn(nu, x) - (3*nu2)/(x*x)*Kn(nu+1, x) + 3*(nu+1)/x*Kn(nu+2, x) - Kn(nu+3,x);
}
//------------------------------------------------------------------------------------------
// LnBesselMod2k
//------------------------------------------------------------------------------------------
double LnBesselMod2k::f(double x, bool smooth_at_nugget) const
{
	return lnKn(nu, x);
}
//------------------------------------------------------------------------------------------
double LnBesselMod2k::df(double x) const
{
	return nu/x - scaledKn(nu+1, x)/scaledKn(nu, x);
}
//------------------------------------------------------------------------------------------
double LnBesselMod2k::d2f(double x) const
{
	double aux = scaledKn(nu, x);
	double r1 = scaledKn(nu+1, x)/aux;
	double r2 = scaledKn(nu+2, x)/aux;
	return -nu/(x*x) - r1/x - r1*r1 + r2;
}
//------------------------------------------------------------------------------------------
double LnBesselMod2k::d3f(double x) const
{
	double aux = scaledKn(nu, x);
	double r1 = scaledKn(nu+1, x)/aux;
	double r2 = scaledKn(nu+2, x)/aux;
	double r3 = scaledKn(nu+3, x)/aux;
	double r1_2 = r1*r1;

	return 2*nu/(x*x*x) - r3 + r1*r1_2 + (3/x + 3*r1)*(r2 - r1_2);
}
//------------------------------------------------------------------------------------------
double LnBesselMod2k::scaledKn(double Nu, double x)		// exp(x)*Kn(Nu, x) -- library function
{
	gsl_sf_result res;
	int e = gsl_sf_bessel_Knu_scaled_e(Nu, x, &res);
	if (e != GSL_SUCCESS)
		throw Exception(stringFormatArr("gsl_sf_bessel_Knu_scaled_e завершилась со статусом {0:%d}", "gsl_sf_bessel_Knu_scaled_e returned with status {0:%d}", e));

	return res.val;
	// error: res.err
}
//------------------------------------------------------------------------------------------
double LnBesselMod2k::lnKn(double Nu, double x)			// ln(Kn(Nu, x)) -- library function
{
	gsl_sf_result res;
	int e = gsl_sf_bessel_lnKnu_e(Nu, x, &res);
	if (e != GSL_SUCCESS)
		throw Exception(stringFormatArr("gsl_sf_bessel_lnKnu_e завершилась со статусом {0:%d}", "gsl_sf_bessel_lnKnu_e returned with status {0:%d}", e));

	return res.val;
	// error: res.err
}
//------------------------------------------------------------------------------------------
// CorrMatern
//------------------------------------------------------------------------------------------
double CorrMatern::f(double x, bool smooth_at_nugget) const
{
	double mult = 1-nugget;
	if (x == 0 && !smooth_at_nugget)
		mult = 1;

	if (x < tol0)
	{
		double nu2 = lnbess.nu*lnbess.nu;
		double x2 = x*x;
		double x4 = x2*x2;
		double multnu2 = (1 - lnbess.nu)*(2 - lnbess.nu);
		double multnu3 = multnu2*(3 - lnbess.nu);
		return mult*(1 + 3*lnbess.nu/(1 - lnbess.nu)*x2 + 4.5*nu2/multnu2*x4 + 4.5*nu2*lnbess.nu/multnu3*x2*x4);
	}
	else
	{
		double sq = sqrt(12 * lnbess.nu);
		double y = sq*x;
		double lg = (1 - lnbess.nu)*log(2.0) + lnbess.nu*log(y) - lgamma(lnbess.nu);		// use logarithms to avoid overflow

		return mult*exp(lg + LnBesselMod2k::lnKn(lnbess.nu, y));
	}
}
//------------------------------------------------------------------------------------------
double CorrMatern::df(double x) const
{
	double mult = 1-nugget;
	if (lnbess.nu <= 1)
		throw Exception("'nu' should be > 1 in CorrMatern::df");

	if (x < tol1)
	{
		double nu2 = lnbess.nu*lnbess.nu;
		double x3 = x*x*x;
		double multnu2 = (1 - lnbess.nu)*(2 - lnbess.nu);
		double multnu3 = multnu2*(3 - lnbess.nu);
		return mult*(6*lnbess.nu/(1 - lnbess.nu)*x + 18*nu2/multnu2*x3 + 27*nu2*lnbess.nu/multnu3*(x3*x*x));
	}
	else
	{
		double sq = sqrt(12 * lnbess.nu);
		double y = sq*x;
		double lg = (1 - lnbess.nu)*log(2.0) + lnbess.nu*log(y) - lgamma(lnbess.nu);		// use logarithms to avoid overflow

//		return (1-nugget)*exp(lg + lnbess.f(y))*sq * (lnbess.nu/y + lnbess.df(y));	// old version

		return mult*exp(lg + LnBesselMod2k::lnKn(lnbess.nu-1, y))*(-sq);
	}
}
//------------------------------------------------------------------------------------------
double CorrMatern::d2f(double x) const
{
	double mult = 1-nugget;
	if (lnbess.nu <= 2)
		throw Exception("'nu' should be > 2 in CorrMatern::d2f");

	if (x < tol2)
	{
		double nu2 = lnbess.nu*lnbess.nu;
		double x2 = x*x;
		double multnu2 = (1 - lnbess.nu)*(2 - lnbess.nu);
		double multnu3 = multnu2*(3 - lnbess.nu);
		return mult*(6*lnbess.nu/(1 - lnbess.nu) + 54*nu2/multnu2*x2 + 135*nu2*lnbess.nu/multnu3*(x2*x2));
	}
	else
	{
		double sq = sqrt(12 * lnbess.nu);
		double y = sq*x;

//		double lg = (1 - lnbess.nu)*log(2.0) + lnbess.nu*log(y) - lgamma(lnbess.nu);		// use logarithms to avoid overflow
//		double dlnKn = lnbess.df(y);
//		return (1-nugget)*exp(lg + lnbess.f(y))*sq*sq * ((lnbess.nu*lnbess.nu-lnbess.nu)/(y*y) + 2*lnbess.nu/y*dlnKn + lnbess.d2f(y) + dlnKn*dlnKn);	// old version

		double lg = (1 - lnbess.nu)*log(2.0) + (lnbess.nu - 1)*log(y) - lgamma(lnbess.nu);		// use logarithms to avoid overflow
		return mult*exp(lg + LnBesselMod2k::lnKn(lnbess.nu-2, y))*(y - LnBesselMod2k::scaledKn(lnbess.nu-1, y)/LnBesselMod2k::scaledKn(lnbess.nu-2, y))*(sq*sq);
	}
}
//------------------------------------------------------------------------------------------
double CorrMatern::d3f(double x) const
{
	double mult = 1-nugget;
	if (lnbess.nu <= 3)
		throw Exception("'nu' should be > 3 in CorrMatern::d3f");

	if (x < tol3)
	{
		double nu2 = lnbess.nu*lnbess.nu;
		double multnu2 = (1 - lnbess.nu)*(2 - lnbess.nu);
		double multnu3 = multnu2*(3 - lnbess.nu);
		return mult*(108*nu2/multnu2*x + 540*nu2*lnbess.nu/multnu3*(x*x*x));
	}
	else
	{
		double sq = sqrt(12 * lnbess.nu);
		double y = sq*x;

//		double lg = (1 - lnbess.nu)*log(2.0) + lnbess.nu*log(y) - lgamma(lnbess.nu);		// use logarithms to avoid overflow
//		double dlnKn = lnbess.df(y);
//		double ddlnKn = lnbess.d2f(y);
//		double nu2 = lnbess.nu*lnbess.nu;
//		double y2 = y*y;
//		double dlnKn2 = dlnKn*dlnKn;

	//	return (1-nugget)*exp(lg + lnbess.f(y))*sq*sq*sq * ((lnbess.nu*nu2 - 3*nu2 + 2*lnbess.nu)/(y*y2) +		// very old version
	//														3*(nu2 - lnbess.nu)/y2*dlnKn +
	//														3*lnbess.nu/y*(ddlnKn + dlnKn2) +
	//														lnbess.d3f(y) + 3*dlnKn*ddlnKn + dlnKn*dlnKn2);

//		return (1-nugget)*exp(lg + lnbess.f(y))*sq*sq*sq * (3*(lnbess.nu/y + dlnKn)*(ddlnKn + dlnKn2 + (nu2-lnbess.nu)/y2)	// old version
//															- 2*lnbess.nu*(nu2 - 1)/(y*y2)
//															+ lnbess.d3f(y)
//															- 2*dlnKn*dlnKn2);

		double lg = (1 - lnbess.nu)*log(2.0) + (lnbess.nu - 1)*log(y) - lgamma(lnbess.nu);		// use logarithms to avoid overflow
		return mult*exp(lg + LnBesselMod2k::lnKn(lnbess.nu-3, y))*(3*LnBesselMod2k::scaledKn(lnbess.nu-2, y)/LnBesselMod2k::scaledKn(lnbess.nu-3, y) - y)*(sq*sq*sq);
	}
}
//------------------------------------------------------------------------------------------
double CorrMatern::lim_df(double y) const
{
	if (lnbess.nu <= 1)
		throw Exception("'nu' should be > 1 in CorrMatern::lim_df");

	if (y < limtol1)				// Taylor
	{
		double nu2 = lnbess.nu*lnbess.nu;
		double y2 = y*y;
		double multnu2 = (1 - lnbess.nu)*(2 - lnbess.nu);
		double multnu3 = multnu2*(3 - lnbess.nu);
		return (1-nugget)*(6*lnbess.nu/(1 - lnbess.nu) + 18*nu2/multnu2*y2 + 27*nu2*lnbess.nu/multnu3*(y2*y2));
	}
	else
	{
		double n = 12*lnbess.nu;
		double z = y*sqrt(n);
		double lg = (1 - lnbess.nu)*log(2.0) + (lnbess.nu - 1)*log(z) - lgamma(lnbess.nu);		// use logarithms to avoid overflow
		return (1-nugget)*exp(lg + LnBesselMod2k::lnKn(lnbess.nu-1, z))*(-n);
	}
}
//------------------------------------------------------------------------------------------
double CorrMatern::lim_d2f(double y) const
{
	if (lnbess.nu <= 2)
		throw Exception("'nu' should be > 2 in CorrMatern::lim_d2f");

	if (y < limtol2)				// Taylor
	{
		double nu2 = lnbess.nu*lnbess.nu;
		double multnu2 = (1 - lnbess.nu)*(2 - lnbess.nu);
		double multnu3 = multnu2*(3 - lnbess.nu);
		return (1-nugget)*(36*nu2/multnu2 + 108*nu2*lnbess.nu/multnu3*(y*y));
	}
	else
	{
		double n = 12*lnbess.nu;
		double z = y*sqrt(n);
		double lg = (1 - lnbess.nu)*log(2.0) + (lnbess.nu - 2)*log(z) - lgamma(lnbess.nu);		// use logarithms to avoid overflow
		return (1-nugget)*exp(lg + LnBesselMod2k::lnKn(lnbess.nu-2, z))*(n*n);
	}
}
//------------------------------------------------------------------------------------------
double CorrMatern::lim_d3f(double y) const
{
	if (lnbess.nu <= 3)
		throw Exception("'nu' should be > 3 in CorrMatern::lim_d3f");

	if (y < limtol3)				// Taylor
	{
		double nu3 = lnbess.nu*lnbess.nu*lnbess.nu;
		double multnu2 = (1 - lnbess.nu)*(2 - lnbess.nu);
		double multnu3 = multnu2*(3 - lnbess.nu);
		double multnu4 = multnu3*(4 - lnbess.nu);
		if (lnbess.nu > 4)
			return (1-nugget)*(-216*nu3/multnu3 - 648*nu3*lnbess.nu/multnu4*(y*y));				// for a smoother Matern correlation - take longer Taylor series
		else
			return (1-nugget)*(-216*nu3/multnu3);
	}
	else
	{
		double n = 12*lnbess.nu;
		double z = y*sqrt(n);
		double lg = (1 - lnbess.nu)*log(2.0) + (lnbess.nu - 3)*log(z) - lgamma(lnbess.nu);		// use logarithms to avoid overflow
		return (1-nugget)*exp(lg + LnBesselMod2k::lnKn(lnbess.nu-3, z))*(n*n*n);
	}
}
//------------------------------------------------------------------------------------------
Func1D_corr* CorrMatern::Copy() const
{
	CorrMatern *res = new CorrMatern(*this);
	return res;
}
//------------------------------------------------------------------------------------------
void CorrMatern::SetNu(double n)
{
	lnbess.nu = n;
}
//------------------------------------------------------------------------------------------
double CorrMatern::GetNu() const
{
	return lnbess.nu;
}
//------------------------------------------------------------------------------------------
// Func1D_corr_factory
//------------------------------------------------------------------------------------------
Func1D_corr *Func1D_corr_factory::New(std::string type)
{
	if (type == "GAUSS")
		return new CorrGauss();
	else if (type == "SPHER")
		return new CorrSpher();
	else if (type == "EXP")
		return new CorrExp();
	else if (type == "VARGAUSS")
		return new VarGauss();
	else if (type == "MATERN")
		return new CorrMatern();
	else
		throw Exception(stringFormatArr("Неопознанный тип функции {0:%s}", "Unrecognized function type {0:%s}", type));
}
//------------------------------------------------------------------------------------------
// DiagBlock
//------------------------------------------------------------------------------------------
DiagBlock::~DiagBlock()
{
#ifdef TESTBLOCK
	int rank00;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank00);
	std::cout << "rank " << rank00 << "\t--- DiagBlock destructor\n";
#endif
}
//------------------------------------------------------------------------------------------
// DiagBlockNum
//------------------------------------------------------------------------------------------
void DiagBlockNum::mult(const std::vector<double> &vec1, int start, std::vector<double> &vec2) const
{
	int end = start + d.size();
	if (start < 0 || end > (int)vec1.size() || end > (int)vec2.size())
		throw Exception("Index 'start' and the block size are not appropriate in DiagBlockNum::mult");

	for (int i = start; i < end; i++)
		vec2[i] = vec1[i] * d[i-start];
}
//------------------------------------------------------------------------------------------
void DiagBlockNum::div(const std::vector<double> &vec1, int start, std::vector<double> &vec2) const
{
	int end = start + d.size();
	if (start < 0 || end > (int)vec1.size() || end > (int)vec2.size())
		throw Exception("Index 'start' and the block size are not appropriate in DiagBlockNum::div");

	for (int i = start; i < end; i++)
		vec2[i] = vec1[i] / d[i-start];
}
//------------------------------------------------------------------------------------------
void DiagBlockNum::div(const Mat &m1, int start, Mat &m2) const
{
	int end = start + d.size();
	if (start < 0 || end > (int)m1.ICount() || end > (int)m2.ICount())
		throw Exception("Index 'start' and the block size are not appropriate in DiagBlockNum::div(2)");
	if (m1.JCount() != m2.JCount())
		throw Exception("Number of columns in m1, m2 do not match in DiagBlockNum::div(2)");

	std::vector<double> dinv(d.size());
	std::transform(d.begin(), d.end(), dinv.begin(), [](double x){return 1/x;});	// invert the diagonal

	Mat m1_loc(d.size(), m1.JCount(), 0);
	memcpy(m1_loc.ToVectorMutable().data(), m1.ToVector().data() + start*m1.JCount(), d.size()*m1.JCount()*sizeof(double));		// m1_loc = m1(start:start+block, :)

	Mat m2_loc = dinv % std::move(m1_loc);		// multiply by diagonal from left
	memcpy(m2.ToVectorMutable().data() + start*m1.JCount(), m2_loc.ToVector().data(), d.size()*m1.JCount()*sizeof(double));		// m2(start:start+block, :) = m2_loc
}
//------------------------------------------------------------------------------------------
void DiagBlockNum::chol_mult(const std::vector<double> &vec1, int start, std::vector<double> &vec2) const
{
	if (!holding_chol)
	{
		chol_d = d;
		std::transform(chol_d.begin(), chol_d.end(), chol_d.begin(), _sqrt);
		holding_chol = true;
	}

	int end = start + chol_d.size();
	if (start < 0 || end > (int)vec1.size() || end > (int)vec2.size())
		throw Exception("Index 'start' and the block size are not appropriate in DiagBlockNum::chol_mult");

	for (int i = start; i < end; i++)
		vec2[i] = vec1[i] * chol_d[i-start];
}
//------------------------------------------------------------------------------------------
// DiagBlockMat
//------------------------------------------------------------------------------------------
void DiagBlockMat::mult(const std::vector<double> &vec1, int start, std::vector<double> &vec2) const
{
	int end = start + size();
	if (start < 0 || end > (int)vec1.size() || end > (int)vec2.size())
		throw Exception("Index 'start' and the block size are not appropriate in DiagBlockMat::mult");

	Mat v1(std::vector<double>(vec1.begin()+start, vec1.begin()+end));
	Mat v2 = M * v1;
	memcpy(vec2.data() + start, v2.ToVector().data(), (end-start)*sizeof(double));
}
//------------------------------------------------------------------------------------------
void DiagBlockMat::div(const std::vector<double> &vec1, int start, std::vector<double> &vec2) const
{
	int end = start + size();
	if (start < 0 || end > (int)vec1.size() || end > (int)vec2.size())
		throw Exception("Index 'start' and the block size are not appropriate in DiagBlockMat::div");

	Mat v1(std::vector<double>(vec1.begin()+start, vec1.begin()+end));
	Mat v2 = M / v1;
	memcpy(vec2.data() + start, v2.ToVector().data(), (end-start)*sizeof(double));
}
//------------------------------------------------------------------------------------------
void DiagBlockMat::div(const Mat &m1, int start, Mat &m2) const
{
	int end = start + size();
	if (start < 0 || end > (int)m1.ICount() || end > (int)m2.ICount())
		throw Exception("Index 'start' and the block size are not appropriate in DiagBlockMat::div(2)");
	if (m1.JCount() != m2.JCount())
		throw Exception("Number of columns in m1, m2 do not match in DiagBlockMat::div(2)");

	Mat m1_loc(size(), m1.JCount(), 0);
	memcpy(m1_loc.ToVectorMutable().data(), m1.ToVector().data() + start*m1.JCount(), size()*m1.JCount()*sizeof(double));		// m1_loc = m1(start:start+block, :)

	Mat m2_loc = M / std::move(m1_loc);
	memcpy(m2.ToVectorMutable().data() + start*m1.JCount(), m2_loc.ToVector().data(), size()*m1.JCount()*sizeof(double));		// m2(start:start+block, :) = m2_loc
}
//------------------------------------------------------------------------------------------
void DiagBlockMat::chol_mult(const std::vector<double> &vec1, int start, std::vector<double> &vec2) const
{
	if (!holding_chol)
	{
		chol_M = M.Chol();
		holding_chol = true;
	}

	int end = start + size();
	if (start < 0 || end > (int)vec1.size() || end > (int)vec2.size())
		throw Exception("Index 'start' and the block size are not appropriate in DiagBlockMat::chol_mult");

	Mat v1(std::vector<double>(vec1.begin()+start, vec1.begin()+end));
	Mat v2 = chol_M * v1;
	memcpy(vec2.data() + start, v2.ToVector().data(), (end-start)*sizeof(double));
}
//------------------------------------------------------------------------------------------
DiagBlockMat::DiagBlockMat(Mat m) : M(std::move(m))
{
	if (M.ICount() != M.JCount())
		throw Exception(stringFormatArr(MessageRE("Неквадратная матрица {0:%ld} x {1:%ld} в DiagBlockMat::DiagBlockMat",
												  "Non-square matrix {0:%ld} x {1:%ld} in DiagBlockMat::DiagBlockMat"), std::vector<size_t>{M.ICount(), M.JCount()}));
};
//------------------------------------------------------------------------------------------
// BlockDiagMat
//------------------------------------------------------------------------------------------
BlockDiagMat::BlockDiagMat(MPI_Comm c, const HMMPI::CorrelCreator *Corr, const HMMPI::StdCreator *Std) : last_r(0), comm(c), sz(0), finalized(false)
{
	// make BlockDiagMat covariance matrix : both diagonal blocks and dense blocks are added
	// blocks are MPI-distributed
	// communicator 'c' is used

	std::vector<Mat> Blocks = Corr->CorrBlocks();
	std::vector<double> Sigma = Std->Std();		// sqrt(diag)

	size_t full_sz = 0;				// full size of covariance matrix
	for (auto i : Blocks)
		full_sz += i.ICount();

	int start = 0;					// start data index for the given block
	int cblock = 0;					// index of the block to be processed
	int Nblocks = Blocks.size();	// number of blocks in corr. matrix
	if (Sigma.size() != full_sz)
		throw HMMPI::Exception(HMMPI::stringFormatArr("Number of std's ({0:%ld}) does not match the size of correlation matrix ({1:%ld}) in BlockDiagMat::Create", std::vector<size_t>{Sigma.size(), full_sz}));

	int rank = -1, size = -1;
	if (c != MPI_COMM_NULL)
	{
		MPI_Comm_rank(c, &rank);
		MPI_Comm_size(c, &size);
		for (int i = 0; i < size; i++)		// loop through all ranks in 'c'
		{
			int Nblocks_rank = Nblocks/size + (i < Nblocks % size ? 1 : 0);		// number of blocks in rank "i"
			for (int j = cblock; j < cblock+Nblocks_rank; j++)					// loop through blocks of rank "i"; "j" is the block index
			{
				int end = start + Blocks[j].ICount();							// end data index for the given block
				std::vector<double> loc(Sigma.begin() + start, Sigma.begin() + end);		// 'loc' = sqrt(diagonal of covariance matrix)

				if (Blocks[j].JCount() == 1)
				{
					std::transform(loc.begin(), loc.end(), loc.begin(), [](double x){return x*x;});		// 'loc' = diag (back transform from sqrt)
					AddBlock(loc, i);												// add diagonal block
				}
				else
				{
					HMMPI::Mat cov_block;
					if (rank == i)
						cov_block = loc % Blocks[j] % loc;							// cov_block = sqrt(diag) * R * sqrt(diag)

					AddBlock(cov_block, i);											// add dense block
				}

//				if (rank == i)		// DEBUG
//				{
//					int rank_w;		// DEBUG
//					MPI_Comm_rank(MPI_COMM_WORLD, &rank_w);		// DEBUG
//					std::cout << "rank_w " << rank_w << ", block " << j << ", indices " << start << " -- " << end << "\n";	// DEBUG
//				}

				start = end;
			}
			if (Nblocks_rank == 0)
				AddBlock(std::vector<double>(), i);		// empty block to fill the remaining ranks

			cblock += Nblocks_rank;
		}
	}
	Finalize();
}
//------------------------------------------------------------------------------------------
BlockDiagMat::~BlockDiagMat()
{
	for (auto &i : Blocks)
		delete i;
}
//------------------------------------------------------------------------------------------
int BlockDiagMat::size() const
{
	if (!finalized)
		throw Exception("Calling size() on a non-finalized BlockDiagMat");

	return sz;
}
//------------------------------------------------------------------------------------------
std::vector<int> BlockDiagMat::Data_ind() const
{
	if (!finalized)
		throw Exception("Calling Data_ind() on a non-finalized BlockDiagMat");

	return data_ind;
}
//------------------------------------------------------------------------------------------
void BlockDiagMat::AddBlock(std::vector<double> v, int r)
{
	if (comm == MPI_COMM_NULL)
		return;

	int rank = 0, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	if (r < 0 || r >= size)
		throw Exception(stringFormatArr("Attempt to add to rank {0:%d} in a communicator of size {1:%d} in BlockDiagMat::AddBlock", std::vector<int>{r, size}));
	if (r - last_r > 1 || r - last_r < 0)
		throw Exception(stringFormatArr("Attempt to add to rank {1:%d} immediately after rank {0:%d} in BlockDiagMat::AddBlock", std::vector<int>{last_r, r}));

	if (rank == r)
	{
		DiagBlockNum *block = new DiagBlockNum(v);
		Blocks.push_back(block);
		sz += block->size();

#ifdef TESTBLOCK
	std::cout << "rank " << rank << "\t+++ BlockDiagMat::AddBlock (diagonal) of size " << block->size() << "\n";
#endif
	}

	last_r = r;
	finalized = false;
}
//------------------------------------------------------------------------------------------
void BlockDiagMat::AddBlock(HMMPI::Mat m, int r)
{
	if (comm == MPI_COMM_NULL)
		return;

	int rank = 0, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	if (r < 0 || r >= size)
		throw Exception(stringFormatArr("Attempt to add to rank {0:%d} in a communicator of size {1:%d} in BlockDiagMat::AddBlock", std::vector<int>{r, size}));
	if (r - last_r > 1 || r - last_r < 0)
		throw Exception(stringFormatArr("Attempt to add to rank {1:%d} immediately after rank {0:%d} in BlockDiagMat::AddBlock", std::vector<int>{last_r, r}));

	if (rank == r)
	{
		DiagBlockMat *block = new DiagBlockMat(m);
		Blocks.push_back(block);
		sz += block->size();

#ifdef TESTBLOCK
	std::cout << "rank " << rank << "\t+++ BlockDiagMat::AddBlock (dense) of size " << block->size() << "\n";
#endif
	}

	last_r = r;
	finalized = false;
}
//------------------------------------------------------------------------------------------
void BlockDiagMat::Finalize()
{
	if (comm == MPI_COMM_NULL)
	{
		finalized = true;
		return;
	}

	int rank = 0, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	if (last_r != size-1)
		throw Exception(stringFormatArr("Highest filled rank is {0:%d} (too small) in a communicator of size {1:%d} in BlockDiagMat::Finalize", std::vector<int>{last_r, size}));

	std::vector<int> vecsz(size);
	MPI_Gather(&sz, 1, MPI_INT, vecsz.data(), 1, MPI_INT, 0, comm);
	HMMPI::Bcast_vector(vecsz, 0, comm);

	data_ind = std::vector<int>(size + 1, 0);
	for (int i = 0; i < size; i++)
		data_ind[i+1] = data_ind[i] + vecsz[i];

	finalized = true;
}
//------------------------------------------------------------------------------------------
std::vector<double> BlockDiagMat::operator*(const std::vector<double> &v) const
{
	if (!finalized)
		throw Exception("Calling operator* on a non-finalized BlockDiagMat");
	if ((int)v.size() != sz)
		throw Exception("Vector size mismatch in BlockDiagMat::operator*");

	std::vector<double> res(sz);
	double start = 0;
	for (size_t i = 0; i < Blocks.size(); i++)
	{
		Blocks[i]->mult(v, start, res);
		start += Blocks[i]->size();
	}

	return res;
}
//------------------------------------------------------------------------------------------
std::vector<double> BlockDiagMat::operator/(const std::vector<double> &v) const
{
	if (!finalized)
		throw Exception("Calling operator/ on a non-finalized BlockDiagMat");
	if ((int)v.size() != sz)
		throw Exception("Vector size mismatch in BlockDiagMat::operator/");

	std::vector<double> res(sz);
	double start = 0;
	for (size_t i = 0; i < Blocks.size(); i++)
	{
		Blocks[i]->div(v, start, res);
		start += Blocks[i]->size();
	}

	return res;
}
//------------------------------------------------------------------------------------------
Mat BlockDiagMat::operator/(const Mat &m) const
{
	if (!finalized)
		throw Exception("Calling operator/(2) on a non-finalized BlockDiagMat");
	if ((int)m.ICount() != sz)
		throw Exception("Matrix row-size mismatch in BlockDiagMat::operator/(2)");

	Mat res(sz, m.JCount(), 0);
	double start = 0;
	for (size_t i = 0; i < Blocks.size(); i++)
	{
		Blocks[i]->div(m, start, res);
		start += Blocks[i]->size();
	}

	return res;
}
//------------------------------------------------------------------------------------------
std::vector<double> BlockDiagMat::operator%(const std::vector<double> &v) const
{
	if (!finalized)
		throw Exception("Calling operator% on a non-finalized BlockDiagMat");
	if ((int)v.size() != sz)
		throw Exception("Vector size mismatch in BlockDiagMat::operator%");

	std::vector<double> res(sz);
	double start = 0;
	for (size_t i = 0; i < Blocks.size(); i++)
	{
		Blocks[i]->chol_mult(v, start, res);
		start += Blocks[i]->size();
	}

	return res;
}
//------------------------------------------------------------------------------------------
double BlockDiagMat::InvTwoSideVecMult(const std::vector<double> &v) const
{
	std::vector<double> div = *this / v;		// MAT^(-1) * v
	return InnerProd(v, div);					// v' * MAT^(-1) * v
}
//------------------------------------------------------------------------------------------
// SolverGauss
//------------------------------------------------------------------------------------------
Mat SolverGauss::Solve(Mat A, Mat b) const
{
#ifdef TESTSOLVER
	int rank00, size00;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank00);
	MPI_Comm_size(MPI_COMM_WORLD, &size00);
	std::cout << "rank " << rank00 << ", size " << size00 << "\tSolverGauss\n";
#endif

	return std::move(A) / std::move(b);
}
//------------------------------------------------------------------------------------------
// SolverDGESV
//------------------------------------------------------------------------------------------
Mat SolverDGESV::Solve(Mat A, Mat b) const
{
#ifdef TESTSOLVER
	int rank00, size00;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank00);
	MPI_Comm_size(MPI_COMM_WORLD, &size00);
	std::cout << "rank " << rank00 << ", size " << size00 << "\tSolverDGESV\n";
#endif

	int icount = A.ICount();
	int jcount = A.JCount();
	int szA = icount*jcount;

	int nrhs = b.JCount();
	int szb = icount*nrhs;

	assert ((int)A.ToVector().size() == szA && (int)b.ToVector().size() == szb);
	if (A.ICount() != b.ICount())
		throw Exception("A and b size mismatch");
	if (icount != jcount)
		throw Exception("Матрица A неквадратная", "Matrix A is non-square");

	const double *cA = A.ToVector().data();		// input raw arrays
	const double *cb = b.ToVector().data();

	double *pA = new double[szA];				// working raw arrays
	double *pb = new double[szb];
	int *ipiv = new int[icount];

	memcpy(pA, cA, szA * sizeof(double));
	memcpy(pb, cb, szb * sizeof(double));

    int lda = jcount;
    int ldb = nrhs;
	int info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, icount, nrhs, pA, lda, ipiv, pb, ldb);

	if (info != 0)
	{
		delete [] pA;
		delete [] pb;
		delete [] ipiv;

		std::string msg = stringFormatArr("DGESV завершилась с info {0:%d}", "DGESV exited with info {0:%d}", info);
		throw Exception(msg);
	}

	Mat res(std::vector<double>(pb, pb + szb), icount, nrhs);

	delete [] pA;
	delete [] pb;
	delete [] ipiv;

	return res;
}
//------------------------------------------------------------------------------------------
// SolverDGELS
//------------------------------------------------------------------------------------------
Mat SolverDGELS::Solve(Mat A, Mat b) const
{
#ifdef TESTSOLVER
	int rank00, size00;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank00);
	MPI_Comm_size(MPI_COMM_WORLD, &size00);
	std::cout << "rank " << rank00 << ", size " << size00 << "\tSolverDGELS\n";
#endif

	int icount = A.ICount();
	int jcount = A.JCount();
	int szA = icount*jcount;

	int nrhs = b.JCount();
	int icount_bnew = icount > jcount ? icount : jcount;		// i-size of the working 'b', MAX(icount, jcount)
	int szb = icount_bnew*nrhs;									// full length of the working 'b'

	assert ((int)A.ToVector().size() == szA && (int)b.ToVector().size() == icount*nrhs);
	if (A.ICount() != b.ICount())
		throw Exception("A and b size mismatch");

	const double *cA = A.ToVector().data();		// input raw arrays
	const double *cb = b.ToVector().data();

	double *pA = new double[szA];				// working raw arrays
	double *pb = new double[szb];

	memcpy(pA, cA, szA * sizeof(double));
	memcpy(pb, cb, icount*nrhs * sizeof(double));

    int lda = jcount;
    int ldb = nrhs;
	int info = LAPACKE_dgels(LAPACK_ROW_MAJOR, 'N', icount, jcount, nrhs, pA, lda, pb, ldb);

	if (info != 0)
	{
		delete [] pA;
		delete [] pb;
		throw Exception(stringFormatArr("DGELS завершилась с info {0:%d}", "DGELS exited with info {0:%d}", info));
	}

	Mat res(std::vector<double>(pb, pb + jcount*nrhs), jcount, nrhs);

	delete [] pA;
	delete [] pb;

	return res;
}
//------------------------------------------------------------------------------------------
// SolverDGELSD
//------------------------------------------------------------------------------------------
Mat SolverDGELSD::Solve(Mat A, Mat b) const
{
#ifdef TESTSOLVER
	int rank00, size00;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank00);
	MPI_Comm_size(MPI_COMM_WORLD, &size00);
	std::cout << "rank " << rank00 << ", size " << size00 << "\tSolverDGELSD\n";
#endif

	int icount = A.ICount();
	int jcount = A.JCount();
	int szA = icount*jcount;

	int nrhs = b.JCount();
	int icount_bnew = icount > jcount ? icount : jcount;		// i-size of the working 'b', MAX(icount, jcount)
	int szb = icount_bnew*nrhs;									// full length of the working 'b'

	int lensv = icount < jcount ? icount : jcount;				// length of array of singular values of A, MIN(icount, jcount)

	assert ((int)A.ToVector().size() == szA && (int)b.ToVector().size() == icount*nrhs);
	if (A.ICount() != b.ICount())
		throw Exception("A and b size mismatch");

	const double *cA = A.ToVector().data();		// input raw arrays
	const double *cb = b.ToVector().data();

	double *pA = new double[szA];				// working raw arrays
	double *pb = new double[szb];
	double *S = new double[lensv];				// singular values of A

	memcpy(pA, cA, szA * sizeof(double));
	memcpy(pb, cb, icount*nrhs * sizeof(double));

    int lda = jcount;
    int ldb = nrhs;
    double rcond = -1;			// negative rcond -> machine precision will be used instead; TODO: change if necessary!

	int info = LAPACKE_dgelsd(LAPACK_ROW_MAJOR, icount, jcount, nrhs, pA, lda, pb, ldb, S, rcond, &rank);

	if (info != 0)
	{
		delete [] pA;
		delete [] pb;
		delete [] S;
		throw Exception(stringFormatArr("DGELSD завершилась с info {0:%d}", "DGELSD exited with info {0:%d}", info));
	}

	Mat res(std::vector<double>(pb, pb + jcount*nrhs), jcount, nrhs);

	// S - unused array of singular values of A; cond(A) in 2-norm is S[0]/S[lensv-1];
	// rank - effective rank

#ifdef TESTSOLVER
	std::cout << "cond(A) = " << S[0]/S[lensv-1] << std::endl;
	std::cout << "rank(A) = " << rank << std::endl;
#endif

	delete [] pA;
	delete [] pb;
	delete [] S;

	return res;
}
//------------------------------------------------------------------------------------------
// SolverDGELSS
//------------------------------------------------------------------------------------------
Mat SolverDGELSS::Solve(Mat A, Mat b) const
{
#ifdef TESTSOLVER
	int rank00, size00;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank00);
	MPI_Comm_size(MPI_COMM_WORLD, &size00);
	std::cout << "rank " << rank00 << ", size " << size00 << "\tSolverDGELSS\n";
#endif

	int icount = A.ICount();
	int jcount = A.JCount();
	int szA = icount*jcount;

	int nrhs = b.JCount();
	int icount_bnew = icount > jcount ? icount : jcount;		// i-size of the working 'b', MAX(icount, jcount)
	int szb = icount_bnew*nrhs;									// full length of the working 'b'

	int lensv = icount < jcount ? icount : jcount;				// length of array of singular values of A, MIN(icount, jcount)

	if (A.ICount() != b.ICount())
		throw Exception(stringFormatArr("A and b size mismatch in SolverDGELSS::Solve; A.ICount = {0:%zu}, b.ICount = {1:%zu}", std::vector<size_t>{A.ICount(), b.ICount()}));

	if ((int)A.ToVector().size() != szA || (int)b.ToVector().size() != icount*nrhs)
		throw Exception(stringFormatArr("Internal error in SolverDGELSS::Solve, A.size ({0:%d}) != icount*jcount ({1:%d}*{2:%d}) || b.size ({3:%d}) != icount*nrhs ({4:%d}*{5:%d})",
														  std::vector<int>{(int)A.ToVector().size(), icount, jcount, (int)b.ToVector().size(), icount, nrhs}));

	const double *cA = A.ToVector().data();		// input raw arrays
	const double *cb = b.ToVector().data();

	double *pA = new double[szA];				// working raw arrays
	double *pb = new double[szb];
	double *S = new double[lensv];				// singular values of A

	memcpy(pA, cA, szA * sizeof(double));
	memcpy(pb, cb, icount*nrhs * sizeof(double));

    int lda = jcount;
    int ldb = nrhs;
    double rcond = -1;			// negative rcond -> machine precision will be used instead; TODO: change if necessary!

	int info = LAPACKE_dgelss(LAPACK_ROW_MAJOR, icount, jcount, nrhs, pA, lda, pb, ldb, S, rcond, &rank);
	if (info != 0)
	{
		delete [] pA;
		delete [] pb;
		delete [] S;
		throw Exception(stringFormatArr("DGELSS завершилась с info {0:%d}", "DGELSS exited with info {0:%d}", info));
	}

	Mat res(std::vector<double>(pb, pb + jcount*nrhs), jcount, nrhs);

	// S - unused array of singular values of A; cond(A) in 2-norm is S[0]/S[lensv-1];
	// rank - effective rank

#ifdef TESTSOLVER
	std::cout << "cond(A) = " << S[0]/S[lensv-1] << std::endl;
	std::cout << "rank(A) = " << rank << std::endl;
#endif

	delete [] pA;
	delete [] pb;
	delete [] S;

	return res;
}
//------------------------------------------------------------------------------------------
// SolverDGELSY
//------------------------------------------------------------------------------------------
Mat SolverDGELSY::Solve(Mat A, Mat b) const
{
#ifdef TESTSOLVER
	int rank00, size00;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank00);
	MPI_Comm_size(MPI_COMM_WORLD, &size00);
	std::cout << "rank " << rank00 << ", size " << size00 << "\tSolverDGELSY\n";
#endif

	int icount = A.ICount();
	int jcount = A.JCount();
	int szA = icount*jcount;

	int nrhs = b.JCount();
	int icount_bnew = icount > jcount ? icount : jcount;		// i-size of the working 'b', MAX(icount, jcount)
	int szb = icount_bnew*nrhs;									// full length of the working 'b'

	assert ((int)A.ToVector().size() == szA && (int)b.ToVector().size() == icount*nrhs);
	if (A.ICount() != b.ICount())
		throw Exception("A and b size mismatch");

	const double *cA = A.ToVector().data();		// input raw arrays
	const double *cb = b.ToVector().data();

	double *pA = new double[szA];				// working raw arrays
	double *pb = new double[szb];
	int *jpvt = new int[jcount];
	for (int i = 0; i < jcount; i++)
		jpvt[i] = 0;

	memcpy(pA, cA, szA * sizeof(double));
	memcpy(pb, cb, icount*nrhs * sizeof(double));

    int lda = jcount;
    int ldb = nrhs;
    double rcond = -1;			// negative rcond -> machine precision will be used instead; TODO: change if necessary!

	int info = LAPACKE_dgelsy(LAPACK_ROW_MAJOR, icount, jcount, nrhs, pA, lda, pb, ldb, jpvt, rcond, &rank);

	if (info != 0)
	{
		delete [] pA;
		delete [] pb;
		delete [] jpvt;
		throw Exception(stringFormatArr("DGELSY завершилась с info {0:%d}", "DGELSY exited with info {0:%d}", info));
	}

	Mat res(std::vector<double>(pb, pb + jcount*nrhs), jcount, nrhs);

	// jpvt - unused array of permutation indices
	// rank - effective rank

#ifdef TESTSOLVER
	std::cout << "jpvt = " << ToString(std::vector<int>(jpvt, jpvt + jcount), "%d");
	std::cout << "rank(A) = " << rank << std::endl;
#endif

	delete [] pA;
	delete [] pb;
	delete [] jpvt;

	return res;
}
//------------------------------------------------------------------------------------------
// BoundConstr
//------------------------------------------------------------------------------------------
std::string BoundConstr::par_name(int i) const
{
	return HMMPI::stringFormatArr("{0:%d}", std::vector<int>{i+1});
}
//------------------------------------------------------------------------------------------
void BoundConstr::OverrideBounds(const std::vector<double> &newmin, const std::vector<double> &newmax)	// overrides min, max after checking that dimensions are the same
{
	assert(min.size() == newmin.size() && max.size() == newmax.size());
	min = newmin;
	max = newmax;
}
//------------------------------------------------------------------------------------------
std::string BoundConstr::Check(const std::vector<double> &p) const		// "", if all constraints are satisfied for 'p'; or a message, if not
{
	size_t len = p.size();
	if (len != min.size() || len != max.size())
		throw HMMPI::Exception("Не совпадает число параметров в векторе и общее число параметров в min/max (BoundConstr::Check)",
							   "Number of parameters in the vector and in min/max do not match (BoundConstr::Check)");
	std::string res = "";
	for (size_t i = 0; i < len; i++)
	{
		if (p[i] < min[i] || p[i] > max[i])		// TODO this might be not the safest check (consider when inputs are nan)
		{
			res += HMMPI::stringFormatArr("для параметра {0:%s} ", "for parameter {0:%s} ", par_name(i));
			res += HMMPI::stringFormatArr(HMMPI::MessageRE("нарушены границы [{0}, {1}]\n",
														   "bounds [{0}, {1}] are violated\n"), std::vector<double>{minrpt(i), maxrpt(i)});
			break;
		}
	}

	return res;
}
//------------------------------------------------------------------------------------------
std::string BoundConstr::CheckEps(std::vector<double> &p, const double eps) const
{
	size_t len = p.size();
	if (eps < 0)
		throw HMMPI::Exception("eps < 0 in BoundConstr::CheckEps");
	if (len != min.size() || len != max.size())
		throw HMMPI::Exception("Не совпадает число параметров в векторе и общее число параметров в min/max (BoundConstr::CheckEps)",
							   "Number of parameters in the vector and in min/max do not match (BoundConstr::CheckEps)");
	std::string res = "";
	for (size_t i = 0; i < len; i++)
	{
		if (p[i] < min[i]-eps || p[i] > max[i]+eps)		// TODO same comment as above
		{
			res += HMMPI::stringFormatArr("для параметра {0:%s} ", "for parameter {0:%s} ", par_name(i));
			res += HMMPI::stringFormatArr(HMMPI::MessageRE("нарушены границы [{0}, {1}]\n",
														   "bounds [{0}, {1}] are violated\n"), std::vector<double>{minrpt(i), maxrpt(i)});
			break;
		}
		else
		{
			if (p[i] < min[i])
				p[i] = min[i];
			if (p[i] > max[i])
				p[i] = max[i];
		}
	}

	return res;
}
//------------------------------------------------------------------------------------------
bool BoundConstr::FindIntersect(const std::vector<double> &x0, const std::vector<double> &x1, std::vector<double> &xint, double &alpha, int &i) const	// 'true' if all constraints are satisfied for 'x1'
{
	assert(x0.size() == x1.size());
	size_t len = x1.size();
	if (len != min.size() || len != max.size())
		throw HMMPI::Exception("Не совпадает число параметров в векторе и общее число параметров в min/max (BoundConstr::FindIntersect)",
							   "Number of parameters in the vector and in min/max do not match (BoundConstr::FindIntersect)");

	bool res = true;
	double amin = std::numeric_limits<double>::max();
	double i_bound;				// curr_bound for the found index
	int imin = -1;
	for (size_t j = 0; j < len; j++)
	{
		double a = std::numeric_limits<double>::max();
		double curr_bound = std::numeric_limits<double>::quiet_NaN();		// concrete min[j] or max[j]
		if (x1[j] < min[j])
		{
			a = (min[j] - x0[j])/(x1[j] - x0[j]);
			curr_bound = min[j];
			if (a < 0 || a > 1 || x1[j] - x0[j] == 0)
				throw HMMPI::Exception("In BoundConstr::FindIntersect 'x0' is not within the bounds");
		}
		if (x1[j] > max[j])
		{
			a = (max[j] - x0[j])/(x1[j] - x0[j]);
			curr_bound = max[j];
			if (a < 0 || a > 1 || x1[j] - x0[j] == 0)
				throw HMMPI::Exception("In BoundConstr::FindIntersect 'x0' is not within the bounds");
		}

		if (a < amin)
		{
			amin = a;
			imin = j;
			i_bound = curr_bound;
			res = false;
		}
	}

	if (!res)
	{
		alpha = amin;
		i = imin;
		xint = x0;
		for (size_t j = 0; j < len; j++)
			xint[j] += alpha * (x1[j] - x0[j]);
		xint[imin] = i_bound;		// directly assign min or max value to avoid roundoff errors
	}

	return res;
}
//------------------------------------------------------------------------------------------
std::vector<double> BoundConstr::SobolSeq(long long int &seed) const
{
	assert(min.size() == max.size());
	std::vector<double> vec(min.size());	// raw point
	std::vector<double> res(min.size());	// scaled point
	HMMPI::Sobol(seed, vec);

	for (size_t i = 0; i < res.size(); i++)
		res[i] = vec[i]*(max[i] - min[i]) + min[i];

	return res;
}
//------------------------------------------------------------------------------------------
std::vector<double> BoundConstr::RandU(Rand *rctx) const
{
	std::vector<double> vec = rctx->RandU(min.size(), 1).ToVector();
	assert(min.size() == max.size() && min.size() == vec.size());

	for (size_t i = 0; i < vec.size(); i++)
		vec[i] = vec[i]*(max[i] - min[i]) + min[i];

	return vec;
}
//------------------------------------------------------------------------------------------
void BoundConstr::Push_point(double Init, double Min, double Max, std::string AN, std::string Name)
{
	min.push_back(Min);
	max.push_back(Max);
}
//------------------------------------------------------------------------------------------
void BoundConstr::AdjustInitSpherical(std::vector<double> &p) const
{
	assert(p.size() == min.size());
	const double pi2 = 2*acos(-1);
	for (size_t i = 0; i < p.size(); i++)
	{
		if (min[i] <= p[i] && p[i] <= max[i])
			continue;
		else
			p[i] += pi2;		// +2*pi

		if (min[i] <= p[i] && p[i] <= max[i])
			continue;
		else
			p[i] -= 2*pi2;		// -2*pi

		if (min[i] <= p[i] && p[i] <= max[i])
			continue;
		else
		{
			p[i] += pi2;		// restored initial
			if (p[i] < min[i])
				p[i] = min[i];
			if (p[i] > max[i])
				p[i] = max[i];
		}
	}
}
//------------------------------------------------------------------------------------------
// SpherCoord
//------------------------------------------------------------------------------------------
double SpherCoord::arccot(double a, double b) const
{
	if (b == 0)
	{
		if (a >= 0)
			return 0;
		else
			return pi;
	}

	return pi/2 - atan(a/b);
}
//------------------------------------------------------------------------------------------
std::vector<double> SpherCoord::spher_to_cart(const std::vector<double> &p) const
{
	assert(dim >= 2);
	if (p.size()+1 != (size_t)dim)
		throw Exception("Inconsistent dimensions in SpherCoord::spher_to_cart");

	std::vector<double> res(dim);
	double prod = 1;
	for (int i = 0; i < dim-1; i++)
	{
		res[i] = R*prod*cos(p[i]) + c[i];
		prod *= sin(p[i]);
	}
	res[dim-1] = R*prod + c[dim-1];

	return res;
}
//------------------------------------------------------------------------------------------
std::vector<double> SpherCoord::cart_to_spher(std::vector<double> x) const
{
	assert(dim >= 2);
	if (x.size() != (size_t)dim)
		throw Exception(stringFormatArr("Inconsistent dimensions in SpherCoord::cart_to_spher, input Cartesian dim = {0:%d}, spherical dim = {1:%d}", std::vector<int>{(int)x.size(), dim-1}));

	for (int i = 0; i < dim; i++)	// shift to the center
		x[i] -= c[i];

	std::vector<double> res(dim-1);
	double sumsq = x[dim-1]*x[dim-1] + x[dim-2]*x[dim-2];

	if (x[dim-1] != 0)
		res[dim-2] = 2*arccot(x[dim-2] + sqrt(sumsq), x[dim-1]);
	else
	{
		if (x[dim-2] < 0)
			res[dim-2] = pi;
		else
			res[dim-2] = 0;
	}

	for (int i = dim-3; i >= 0; i--)
	{
		res[i] = arccot(x[i], sqrt(sumsq));
		sumsq += x[i]*x[i];
	}
	radius = sqrt(sumsq);	// radius could be potentially used later

	return res;
}
//------------------------------------------------------------------------------------------
Mat SpherCoord::dxdp(const std::vector<double> &p) const
{
	assert(dim >= 2);
	if (p.size()+1 != (size_t)dim)
		throw Exception("Inconsistent dimensions in SpherCoord::dxdp");

	Mat res(dim, dim-1, 0.0);
	double prodsin = 1;						// accumulates only product of sines
	std::vector<double> prod(dim-1, 0.0);	// will accumulate product for each column
	for (int i = 0; i < dim-1; i++)			// row
	{
		double sini = sin(p[i]);
		double cosi = cos(p[i]);
		for (int j = 0; j < i; j++)			// column
		{
			res(i, j) = R*prod[j]*cosi;
			prod[j] *= sini;
		}

		prod[i] = prodsin;					// now, for j = i
		res(i, i) = -R*prod[i]*sini;
		prod[i] *= cosi;
		prodsin *= sini;
	}

	for (int j = 0; j < dim-1; j++)			// last row, i = dim-1
		res(dim-1, j) = R*prod[j];

	return res;
}
//------------------------------------------------------------------------------------------
Mat SpherCoord::dxdp_k(const std::vector<double> &p, int k) const
{
	assert(dim >= 2);
	if (p.size()+1 != (size_t)dim)
		throw Exception("Inconsistent dimensions in SpherCoord::dxdp_k");
	if (k < 0 || k >= dim-1)
		throw Exception("Index 'k' out of range in SpherCoord::dxdp_k");

	Mat res(dim, dim-1, 0.0);
	double prodsin = 1;						// accumulates product of sines, and probably one cosine
	std::vector<double> prod(dim-1, 0.0);	// will accumulate product for each column
	for (int i = 0; i < dim-1; i++)			// row
	{
		double sini = sin(p[i]);
		double cosi = cos(p[i]);
		for (int j = 0; j < i; j++)			// column
		{
			if (i > k)
				res(i, j) = R*prod[j]*cosi;
			else if (i == k)
				res(i, j) = -R*prod[j]*sini;

			if (i != k)
				prod[j] *= sini;
			else
				prod[j] *= cosi;
		}

		prod[i] = prodsin;					// now, for j = i
		if (i > k)
			res(i, i) = -R*prod[i]*sini;
		else if (i == k)
			res(i, i) = -R*prod[i]*cosi;

		if (i != k)
		{
			prod[i] *= cosi;
			prodsin *= sini;
		}
		else
		{
			prod[i] *= -sini;
			prodsin *= cosi;
		}
	}

	for (int j = 0; j < dim-1; j++)			// last row, i = dim-1
		res(dim-1, j) = R*prod[j];

	return res;
}
//------------------------------------------------------------------------------------------
bool SpherCoord::periodic_swap(const HMMPI::BoundConstr *bc, std::vector<double> &p) const
{
	const int sdim = p.size();
	const std::vector<double> min = bc->fullmin();
	const std::vector<double> max = bc->fullmax();

	if ((int)min.size() != sdim)
		throw Exception("'bc' and 'p' sizes do not match in SpherCoord::periodic_swap");

	if (min[sdim-1] == 0 && max[sdim-1] == pi2)
	{
		if (p[sdim-1] == 0)							// check if boundary is reached for the last coordinate
		{
			p[sdim-1] = pi2;
			return true;
		}
		if (p[sdim-1] == pi2)
		{
			p[sdim-1] = 0;
			return true;
		}
	}
	return false;
}
//------------------------------------------------------------------------------------------
}	// namespace HMMPI
