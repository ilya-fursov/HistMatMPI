#include "Abstract.h"
#include "MathUtils.h"
#include "MonteCarlo.h"
#include "lapacke.h"
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
void Bcast_vector(std::vector<std::string> &v, int root, MPI_Comm comm)			// MPI broadcast vector<string> from 'root' rank; memory allocation is done if needed
{
	if (comm == MPI_COMM_NULL)
		return;

	int sz = v.size();							// vector size to be broadcasted from 'root'
	MPI_Bcast(&sz, 1, MPI_INT, root, comm);
	if ((int)v.size() != sz)
		v = std::vector<std::string>(sz);		// reallocate array

	for (size_t i = 0; i < v.size(); i++)
		Bcast_string(v[i], root, comm);

#ifdef TESTBCAST
	int rank00;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank00);
	std::cout << "rank " << rank00 << "\tBcast_vector<string>\n";
#endif
}
//------------------------------------------------------------------------------------------
// see InnerProd() in MathUtilsMat.cpp
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

	if (lnbess.nu <= 0)
		throw Exception("'nu' should be > 0 in CorrMatern::f");

	if (x == 0)
		return mult;
	else if (x < tol0)
	{
		double nu2 = lnbess.nu*lnbess.nu;
		double x2 = x*x;
		double x4 = x2*x2;
		double multnu2 = (1 - lnbess.nu)*(2 - lnbess.nu);		// for nu > 2
		double multnu3 = multnu2*(3 - lnbess.nu);				// for nu > 3
		double res = 1;
		if (lnbess.nu > 1)
			res += 3*lnbess.nu/(1 - lnbess.nu)*x2;
		if (lnbess.nu > 2)
			res += 4.5*nu2/multnu2*x4;
		if (lnbess.nu > 3)
			res += 4.5*nu2*lnbess.nu/multnu3*x2*x4;

		return mult*res;
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

	if (x == 0)
		return 0;
	else if (x < tol1)
	{
		double nu2 = lnbess.nu*lnbess.nu;
		double x3 = x*x*x;
		double multnu2 = (1 - lnbess.nu)*(2 - lnbess.nu);	// for nu > 2
		double multnu3 = multnu2*(3 - lnbess.nu);			// for nu > 3
		double res = 6*lnbess.nu/(1 - lnbess.nu)*x;
		if (lnbess.nu > 2)
			res += 18*nu2/multnu2*x3;
		if (lnbess.nu > 3)
			res += 27*nu2*lnbess.nu/multnu3*(x3*x*x);

		return mult*res;
	}
	else
	{
		double sq = sqrt(12 * lnbess.nu);
		double y = sq*x;
		double lg = (1 - lnbess.nu)*log(2.0) + lnbess.nu*log(y) - lgamma(lnbess.nu);		// use logarithms to avoid overflow
		return mult*exp(lg + LnBesselMod2k::lnKn(lnbess.nu-1, y))*(-sq);
	}
}
//------------------------------------------------------------------------------------------
double CorrMatern::d2f(double x) const
{
	double mult = 1-nugget;
	if (lnbess.nu <= 2)
		throw Exception("'nu' should be > 2 in CorrMatern::d2f");

	if (x == 0)
		return mult*6*lnbess.nu/(1 - lnbess.nu);
	else if (x < tol2)
	{
		double nu2 = lnbess.nu*lnbess.nu;
		double x2 = x*x;
		double multnu2 = (1 - lnbess.nu)*(2 - lnbess.nu);	// for nu > 2
		double multnu3 = multnu2*(3 - lnbess.nu);			// for nu > 3
		double res = 6*lnbess.nu/(1 - lnbess.nu) + 54*nu2/multnu2*x2;
		if (lnbess.nu > 3)
			res += 135*nu2*lnbess.nu/multnu3*(x2*x2);

		return mult*res;
	}
	else
	{
		double sq = sqrt(12 * lnbess.nu);
		double y = sq*x;
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

	if (x == 0)
		return 0;
	else if (x < tol3)
	{
		double nu2 = lnbess.nu*lnbess.nu;
		double multnu2 = (1 - lnbess.nu)*(2 - lnbess.nu);	// for nu > 2
		double multnu3 = multnu2*(3 - lnbess.nu);			// for nu > 3
		return mult*(108*nu2/multnu2*x + 540*nu2*lnbess.nu/multnu3*(x*x*x));
	}
	else
	{
		double sq = sqrt(12 * lnbess.nu);
		double y = sq*x;
		double lg = (1 - lnbess.nu)*log(2.0) + (lnbess.nu - 1)*log(y) - lgamma(lnbess.nu);		// use logarithms to avoid overflow
		return mult*exp(lg + LnBesselMod2k::lnKn(lnbess.nu-3, y))*(3*LnBesselMod2k::scaledKn(lnbess.nu-2, y)/LnBesselMod2k::scaledKn(lnbess.nu-3, y) - y)*(sq*sq*sq);
	}
}
//------------------------------------------------------------------------------------------
double CorrMatern::lim_df(double y) const
{
	if (lnbess.nu <= 1)
		throw Exception("'nu' should be > 1 in CorrMatern::lim_df");

	if (y == 0)
		return (1-nugget)*6*lnbess.nu/(1 - lnbess.nu);
	else if (y < limtol1)									// Taylor
	{
		double nu2 = lnbess.nu*lnbess.nu;
		double y2 = y*y;
		double multnu2 = (1 - lnbess.nu)*(2 - lnbess.nu);	// for nu > 2
		double multnu3 = multnu2*(3 - lnbess.nu);			// for nu > 3
		double res = 6*lnbess.nu/(1 - lnbess.nu);
		if (lnbess.nu > 2)
			res += 18*nu2/multnu2*y2;
		if (lnbess.nu > 3)
			res += 27*nu2*lnbess.nu/multnu3*(y2*y2);

		return (1-nugget)*res;
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

	double nu2 = lnbess.nu*lnbess.nu;
	double multnu2 = (1 - lnbess.nu)*(2 - lnbess.nu);		// for nu > 2

	if (y == 0)
		return (1-nugget)*36*nu2/multnu2;
	else if (y < limtol2)									// Taylor
	{
		double multnu3 = multnu2*(3 - lnbess.nu);			// for nu > 3
		double res = 36*nu2/multnu2;
		if (lnbess.nu > 3)
			res += 108*nu2*lnbess.nu/multnu3*(y*y);

		return (1-nugget)*res;
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

	double nu3 = lnbess.nu*lnbess.nu*lnbess.nu;
	double multnu2 = (1 - lnbess.nu)*(2 - lnbess.nu);	// for nu > 2
	double multnu3 = multnu2*(3 - lnbess.nu);			// for nu > 3

	if (y == 0)
		return (1-nugget)*(-216)*nu3/multnu3;
	else if (y < limtol3)								// Taylor
	{
		double multnu4 = multnu3*(4 - lnbess.nu);		// for nu > 4
		double res = -216*nu3/multnu3;
		if (lnbess.nu > 4)
			res -= 648*nu3*lnbess.nu/multnu4*(y*y);

		return (1-nugget)*res;
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
