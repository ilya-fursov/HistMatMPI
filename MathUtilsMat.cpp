#include <cstring>
#include "Utils.h"
#include "MathUtils.h"
#include "lapacke_select.h"
#include "cblas_select.h"

namespace HMMPI
{
//------------------------------------------------------------------------------------------
const size_t LINEBUFF = 4096;
//------------------------------------------------------------------------------------------
namespace ManualMath
{
double InnerProd(const std::vector<double>& a, const std::vector<double>& b)
{
	size_t len = a.size();
	if (b.size() != len)
		throw Exception(stringFormatArr("Vector sizes do not match in InnerProd(vector({0:%zu}), vector({1:%zu}))", std::vector<size_t>{len, b.size()}));

	double res = 0;
	for (size_t i = 0; i < len; i++)
		res += a[i] * b[i];

	return res;
}
}		// namespace ManualMath
//------------------------------------------------------------------------------------------
double InnerProd(const std::vector<double>& a, const std::vector<double>& b)	// BLAS version
{
	size_t len = a.size();
	if (b.size() != len)
		throw Exception(stringFormatArr("Vector sizes do not match in InnerProd(vector({0:%zu}), vector({1:%zu}))", std::vector<size_t>{len, b.size()}));

	return cblas_ddot(len, a.data(), 1, b.data(), 1);
}
//------------------------------------------------------------------------------------------
// class Mat
//------------------------------------------------------------------------------------------
void Mat::reset_chol_spo_cache() const
{
	delete[] chol_spo_cache;
	chol_spo_cache = 0;
}
//------------------------------------------------------------------------------------------
void Mat::reset_dsytrf_cache() const
{
	delete[] dsytrf_cache;
	delete[] dsytrf_ipiv;

	dsytrf_cache = 0;
	dsytrf_ipiv = 0;
}
//------------------------------------------------------------------------------------------
const double* Mat::chol_spo() const
{
	if (chol_spo_cache == 0)		// empty cache -> recalculate
	{
		if (icount != jcount)
			throw Exception("Non-square matrix in Mat::chol_spo");

		chol_spo_cache = new double[icount * icount];
		memcpy(chol_spo_cache, data.data(), icount * icount * sizeof(double));		// copy the matrix data

		int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'U', icount, chol_spo_cache, icount);
		if (info != 0)
		{
			reset_chol_spo_cache();
			std::string msg = stringFormatArr("DPOTRF ����������� � info {0:%d}", "DPOTRF exited with info {0:%d}", info);
			debug_output(msg + "\n", this);
			throw Exception(msg);
		}

		// make the lower triangular part zero
		for (size_t i = 1; i < icount; i++)
			for (size_t j = 0; j < i; j++)
				chol_spo_cache[i * jcount + j] = 0;

		cache_msg_to_file("chol_spo_cache: recalculating.....\n");
	}
	else
		cache_msg_to_file("chol_spo_cache: USING_CACHE!\n");

	return chol_spo_cache;
}
//------------------------------------------------------------------------------------------
void Mat::cache_msg_to_file(const std::string& msg) const			// debug output of 'msg' to file, SIMILAR to Cache<T>::MsgToFile()
{
#ifdef TEST_CACHE
	char fname[500];
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	sprintf(fname, TEST_CACHE, rank);
	FILE* f = fopen(fname, "a");
	if (f != NULL)
	{
		fputs(msg.c_str(), f);
		fclose(f);
	}
#endif
}
//------------------------------------------------------------------------------------------
void Mat::dsytrf(const double** A, const int** ipiv) const
{
	if (dsytrf_cache == 0)			// empty cache -> recalculate
	{
		if (icount != jcount)
			throw Exception("Non-square matrix in Mat::dsytrf");

		dsytrf_cache = new double[icount * icount];
		dsytrf_ipiv = new int[icount];
		memcpy(dsytrf_cache, data.data(), icount * icount * sizeof(double));		// copy the matrix data

		int info = LAPACKE_dsytrf(LAPACK_ROW_MAJOR, 'U', icount, dsytrf_cache, icount, dsytrf_ipiv);
		if (info != 0)
		{
			reset_dsytrf_cache();
			std::string msg = stringFormatArr("DSYTRF ����������� � info {0:%d}", "DSYTRF exited with info {0:%d}", info);
			debug_output(msg + "\n", this);
			throw Exception(msg);
		}

		// make the lower triangular part zero
		for (size_t i = 1; i < icount; i++)
			for (size_t j = 0; j < i; j++)
				dsytrf_cache[i * jcount + j] = 0;

		cache_msg_to_file("dsytrf_cache: recalculating.....\n");
	}
	else
		cache_msg_to_file("dsytrf_cache: USING_CACHE!\n");

	*A = dsytrf_cache;
	*ipiv = dsytrf_ipiv;
}
//------------------------------------------------------------------------------------------
void Mat::debug_output(const std::string& msg, const Mat* m) const
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	char fname[BUFFSIZE];
	sprintf(fname, debug_file.c_str(), rank);
	FILE* debug = fopen(fname, "w");

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
		data[i * N + i] = 1;

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
Mat::Mat(const std::vector<double>& v, bool IsDiag) : Vector2<double>(v, v.size(), 1), op_switch(2), chol_spo_cache(0), dsytrf_cache(0), dsytrf_ipiv(0)
{
	if (IsDiag)
	{
		const size_t N = v.size();
		*this = Mat(N, N, 0.0);
		for (size_t i = 0; i < N; i++)
			data[i * N + i] = v[i];
	}
#ifdef TESTING
	std::cout << "Mat::Mat(const std::vector<double> &v, bool IsDiag)" << std::endl;
#endif
}
//------------------------------------------------------------------------------------------
Mat::Mat(const Mat& m) : Vector2<double>(m), op_switch(m.op_switch), chol_spo_cache(0), dsytrf_cache(0), dsytrf_ipiv(0), delim(m.delim)
{
	if (m.chol_spo_cache != 0)
	{
		assert(icount == jcount);
		chol_spo_cache = new double[icount * icount];
		memcpy(chol_spo_cache, m.chol_spo_cache, icount * icount * sizeof(double));
	}

	if (m.dsytrf_cache != 0)
	{
		assert(icount == jcount);
		dsytrf_cache = new double[icount * icount];
		dsytrf_ipiv = new int[icount];
		memcpy(dsytrf_cache, m.dsytrf_cache, icount * icount * sizeof(double));
		memcpy(dsytrf_ipiv, m.dsytrf_ipiv, icount * sizeof(int));
	}

#ifdef TESTING
	std::cout << "Mat::Mat(const Mat &m)" << std::endl;
#endif
}
//------------------------------------------------------------------------------------------
Mat::Mat(Mat&& m) noexcept : Vector2<double>(std::move(m)), op_switch(m.op_switch), delim(m.delim)
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
const Mat& Mat::operator=(const Mat& m)
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
		chol_spo_cache = new double[icount * icount];
		memcpy(chol_spo_cache, m.chol_spo_cache, icount * icount * sizeof(double));
	}
	else
		chol_spo_cache = 0;

	reset_dsytrf_cache();
	if (m.dsytrf_cache != 0)
	{
		assert(icount == jcount);
		dsytrf_cache = new double[icount * icount];
		dsytrf_ipiv = new int[icount];
		memcpy(dsytrf_cache, m.dsytrf_cache, icount * icount * sizeof(double));
		memcpy(dsytrf_ipiv, m.dsytrf_ipiv, icount * sizeof(int));
	}
	else
	{
		dsytrf_cache = 0;
		dsytrf_ipiv = 0;
	}

	return *this;
}
//------------------------------------------------------------------------------------------
const Mat& Mat::operator=(Mat&& m) noexcept	// TODO not much sure here
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
void Mat::LoadASCII(FILE* f, int num)
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
void Mat::SaveASCII(FILE* f, std::string fmt) const
{
	fputs(ToString(fmt).c_str(), f);
}
//------------------------------------------------------------------------------------------
std::string Mat::ToString(std::string fmt) const
{
	char buff[BUFFSIZE];
	std::string res;

	for (size_t i = 0; i < icount; i++)
		for (size_t j = 0; j < jcount; j++)
		{
			sprintf(buff, fmt.c_str(), (*this)(i, j));
			res += buff;
			if (j < jcount - 1)
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

	int previous_size = icount * jcount;			// save the previous array size on each rank
	MPI_Bcast(&icount, 1, MPI_LONG_LONG, root, comm);
	MPI_Bcast(&jcount, 1, MPI_LONG_LONG, root, comm);

	if (previous_size != int(icount * jcount))
		*this = Mat(icount, jcount, 0.0);		// reallocate array

#ifdef TESTBCAST
	std::cout << "rank " << rank << "\tMat::Bcast, final data.data() " << data.data() << ", icount = " << icount << ", jcount = " << jcount << "\n";
#endif

	MPI_Bcast(data.data(), icount * jcount, MPI_DOUBLE, root, comm);
}
//------------------------------------------------------------------------------------------
void Mat::Reshape(size_t i, size_t j)
{
	reset_chol_spo_cache();
	reset_dsytrf_cache();

	if (icount * jcount != i * j)
		throw Exception("New dimensions in Mat::Reshape are not consistent with data length");

	icount = i;
	jcount = j;
}
//------------------------------------------------------------------------------------------
std::vector<double>& Mat::ToVectorMutable()
{
	reset_chol_spo_cache();
	reset_dsytrf_cache();
	return data;
}
//------------------------------------------------------------------------------------------
void Mat::Deserialize(const double* v)
{
	reset_chol_spo_cache();
	reset_dsytrf_cache();
	data = std::vector<double>(v, v + icount * jcount);
}
//------------------------------------------------------------------------------------------
void Mat::SetOpSwitch(int s)						// sets op_switch
{
	if (s != 1 && s != 2)
		throw Exception("Mat::SetOpSwitch requires s = 1 or 2");

	op_switch = s;
}
//------------------------------------------------------------------------------------------
Mat Mat::Tr() const				// transpose
{
	Mat res(jcount, icount, 0.0);
	for (size_t i = 0; i < jcount; i++)
		for (size_t j = 0; j < icount; j++)
			res.data[i * icount + j] = data[j * jcount + i];

	return res;
}
//------------------------------------------------------------------------------------------
double Mat::Trace() const
{
	if (icount != jcount)
		throw Exception("����� Mat::Trace ��� ������������ �������", "Mat::Trace called for a non-square matrix");

	double res = 0;
	const double* p = data.data();
	for (size_t i = 0; i < icount; i++)
		res += p[i * jcount + i];

	return res;
}
//------------------------------------------------------------------------------------------
double Mat::Sum() const
{
	double res = 0;
	const double* p = data.data();
	size_t SZ = icount * jcount;
	for (size_t i = 0; i < SZ; i++)
		res += p[i];

	return res;
}
//------------------------------------------------------------------------------------------
double Mat::Max(int& i, int& j) const
{
	double max = std::numeric_limits<double>::lowest();
	int SZ = icount * jcount, ind = -1;
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
double Mat::Min(int& i, int& j) const
{
	double min = std::numeric_limits<double>::max();
	int SZ = icount * jcount, ind = -1;
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
	for (const auto& d : data)
		res += fabs(d);

	return res;
}
//------------------------------------------------------------------------------------------
double Mat::Norm2() const	// 2-norm of a vector (matrix gets extended to the vector), Manual | BLAS depending on 'op_switch'
{
	if (op_switch == 1)
	{
		double res = 0;
		for (const auto& d : data)
			res += d * d;

		return sqrt(res);
	}
	else if (op_switch == 2)
		return cblas_dnrm2(data.size(), data.data(), 1);
	else
		throw Exception("Bad op_switch in Mat::Norm2()");
}
//------------------------------------------------------------------------------------------
double Mat::NormInf() const
{
	double res = 0;
	for (const auto& d : data)
		if (fabs(d) > res)
			res = fabs(d);

	return res;
}
//------------------------------------------------------------------------------------------
void Mat::Func(const std::function<double(double)>& f)
{
	reset_chol_spo_cache();
	reset_dsytrf_cache();

	size_t SZ = icount * jcount;
	for (size_t i = 0; i < SZ; i++)
		data[i] = f(data[i]);
}
//------------------------------------------------------------------------------------------
void Mat::FuncInd(const std::function<double(int, int, double)>& f)
{
	reset_chol_spo_cache();
	reset_dsytrf_cache();

	for (size_t i = 0; i < icount; i++)
		for (size_t j = 0; j < jcount; j++)
			data[i * jcount + j] = f(i, j, data[i * jcount + j]);
}
//------------------------------------------------------------------------------------------
double& Mat::operator()(size_t i, size_t j)
{
	reset_chol_spo_cache();
	reset_dsytrf_cache();

	return data[i * jcount + j];
}
//------------------------------------------------------------------------------------------
const double& Mat::operator()(size_t i, size_t j) const
{
	return data[i * jcount + j];
}
//------------------------------------------------------------------------------------------
Mat operator&&(const Mat& m1, const Mat& m2)
{
	if (m1.icount == 0 || m1.jcount == 0)
		return m2;
	if (m2.icount == 0 || m2.jcount == 0)
		return m1;

	if (m1.icount != m2.icount)
		throw Exception(stringFormatArr("Inconsistent icount in operator&&(Mat, Mat): {0:%zu}, {1:%zu}", std::vector<size_t>{m1.icount, m2.icount}));

	Mat res(m1.icount, m1.jcount + m2.jcount, 0.0);
	double* pres = res.data.data();
	const double* pm1 = m1.data.data();
	const double* pm2 = m2.data.data();

	for (size_t i = 0; i < m1.icount; i++)
	{
		memcpy(&pres[i * res.jcount], &pm1[i * m1.jcount], m1.jcount * sizeof(double));
		memcpy(&pres[i * res.jcount + m1.jcount], &pm2[i * m2.jcount], m2.jcount * sizeof(double));
	}

	return res;
}
//------------------------------------------------------------------------------------------
Mat operator||(Mat m1, const Mat& m2)
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
Mat Mat::Reorder(const std::vector<int>& ordi, const std::vector<int>& ordj) const
{
	return Mat(HMMPI::Reorder(data, icount, jcount, ordi, ordj), ordi.size(), ordj.size());
}
//------------------------------------------------------------------------------------------
Mat Mat::Reorder(int i0, int i1, int j0, int j1) const		// creates a submatrix with indices [i0, i1)*[j0, j1)
{
	if (i0 < 0 || i1 >(int)icount || j0 < 0 || j1 >(int)jcount)
		throw Exception(stringFormatArr("In Mat::Reorder indices [{0:%d}, {1:%d})*[{2:%d}, {3:%d}) are inconsistent with matrix dimension {4:%d} * {5:%d}",
			std::vector<int>{i0, i1, j0, j1, (int)icount, (int)jcount}));
	if (i0 >= i1 || j0 >= j1)
		throw Exception(stringFormatArr("In Mat::Reorder I-indices ({0:%d}, {1:%d}) and J-indices ({2:%d}, {3:%d}) should be strictly increasing",
			std::vector<int>{i0, i1, j0, j1}));

	std::vector<int> ordi(i1 - i0);
	std::vector<int> ordj(j1 - j0);
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

	size_t SZ = icount * jcount;
	double* pres = m.data.data();
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

	size_t SZ = icount * jcount;
	double* pres = m.data.data();
	for (size_t i = 0; i < SZ; i++)
		pres[i] = data[i] - pres[i];

	return m;
}
//------------------------------------------------------------------------------------------
void Mat::operator+=(const Mat& m)
{
	reset_chol_spo_cache();
	reset_dsytrf_cache();

	if (icount != m.icount || jcount != m.jcount)
		throw Exception("Inconsistent dimensions in Mat::operator+=");

	size_t SZ = icount * jcount;
	const double* pm = m.data.data();
	for (size_t i = 0; i < SZ; i++)
		data[i] += pm[i];
}
//------------------------------------------------------------------------------------------
void Mat::operator-=(const Mat& m)
{
	reset_chol_spo_cache();
	reset_dsytrf_cache();

	if (icount != m.icount || jcount != m.jcount)
		throw Exception("Inconsistent dimensions in Mat::operator-=");

	size_t SZ = icount * jcount;
	const double* pm = m.data.data();
	for (size_t i = 0; i < SZ; i++)
		data[i] -= pm[i];
}
//------------------------------------------------------------------------------------------
Mat Mat::operator*(const Mat& m) const		// *this * m, using Manual | BLAS depending on 'op_switch'
{
	if (jcount != m.icount)
		throw Exception("Inconsistent dimensions in Mat::operator*(Mat)");

	size_t sz_I = icount;
	size_t sz_J = m.jcount;
	size_t sz = jcount;

	Mat res(sz_I, sz_J, 0.0);
	const double* p = data.data();
	const double* pm = m.data.data();
	double* pres = res.data.data();

	if (op_switch == 1)
	{
		for (size_t i = 0; i < sz_I; i++)
			for (size_t j = 0; j < sz_J; j++)
			{
				double sum = 0;
				for (size_t v = 0; v < sz; v++)
					sum += p[sz * i + v] * pm[sz_J * v + j];	// row-major storage!

				pres[sz_J * i + j] = sum;
			}
	}
	else if (op_switch == 2)
	{
		const int lda = jcount;
		const int ldb = sz_J;
		const int ldc = sz_J;
		if (sz_I > 0 && sz_J > 0 && sz > 0)
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, icount, sz_J, jcount, 1.0, p, lda, pm, ldb, 0.0, pres, ldc);
	}
	else
		throw Exception("Bad op_switch in Mat::operator*(Mat)");

	return res;
}
//------------------------------------------------------------------------------------------
std::vector<double> Mat::operator*(const std::vector<double>& v) const		// *this * v, using Manual | BLAS depending on 'op_switch'
{
	if (jcount != v.size())
		throw Exception("Inconsistent dimensions in Mat::operator*(vector)");

	if (op_switch == 1)
	{
		Mat res = (*this) * Mat(v);
		return res.ToVector();
	}
	else if (op_switch == 2)
	{
		std::vector<double> res(icount, 0.0);
		const int lda = jcount;
		const double alpha = 1;
		const double beta = 0;
		if (icount > 0 && jcount > 0)
			cblas_dgemv(CblasRowMajor, CblasNoTrans, icount, jcount, alpha, data.data(), lda, v.data(), 1, beta, res.data(), 1);

		return res;
	}
	else
		throw Exception("Bad op_switch in Mat::operator*(vector)");
}
//------------------------------------------------------------------------------------------
Mat Mat::Autocorr() const
{
	if (icount <= 1 || jcount != 1)
		throw Exception("Autocorr() should be applied to N x 1 vectors, N > 1");

	Mat res(icount, 1, 0.0);
	double* pres = res.data.data();
	const double* p = data.data();
	double mean = Sum() / icount;

	for (size_t k = 0; k < icount; k++)
	{
		double d = 0;
		for (size_t t = 0; t < icount - k; t++)
			d += (p[t] - mean) * (p[t + k] - mean);
		pres[k] = d / (icount - 1);
		if (k > 0)
			pres[k] /= pres[0];		// normalization
	}
	pres[0] /= pres[0];		// normalization

	return res;
}
//------------------------------------------------------------------------------------------
int Mat::Ess(double& res) const
{
	Mat ac = Autocorr();
	size_t N = icount / 2;
	const double* pac = ac.data.data();

	res = 0;
	int lag = 0;
	double G_prev = std::numeric_limits<double>::max();
	for (size_t m = 0; m < N; m++)
	{
		double G = pac[2 * m] + pac[2 * m + 1];	// Gamma_m
		if (G > 0 && G < G_prev)
		{
			res += G;
			lag = 2 * m + 1;
		}
		else
			break;

		G_prev = G;
	}

	res = (double)icount / (2 * res - 1);
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

	const double* A = chol_spo();
	Mat res(std::vector<double>(A, A + icount * icount), icount, icount);

	return res;
}
//------------------------------------------------------------------------------------------
Mat Mat::InvSPO() const
{
	if (icount != jcount)
		throw Exception("Non-square matrix in Mat::InvSPO");

	std::vector<double> UtU(chol_spo(), chol_spo() + icount * icount);
	double* A = UtU.data();
	int info = LAPACKE_dpotri(LAPACK_ROW_MAJOR, 'U', icount, A, icount);
	if (info != 0)
	{
		std::string msg = stringFormatArr("DPOTRI ����������� � info {0:%d}", "DPOTRI exited with info {0:%d}", info);
		Mat temp(std::vector<double>(chol_spo(), chol_spo() + icount * icount), icount, icount);
		debug_output(msg + "\n", &temp);
		throw Exception(msg);
	}

	// symmetrically fill the lower triangular part
	for (size_t i = 1; i < icount; i++)
		for (size_t j = 0; j < i; j++)
			A[i * jcount + j] = A[j * jcount + i];

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
		std::string msg = stringFormatArr("DTRTRI ����������� � info {0:%d}", "DTRTRI exited with info {0:%d}", info);
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

	const double* A = chol_spo();
	double res = 0;
	for (size_t i = 0; i < icount; i++)
		res += log(A[i * jcount + i]);

	return 2 * res;
}
//------------------------------------------------------------------------------------------
Mat Mat::InvSY() const
{
	if (icount != jcount)
		throw Exception("Non-square matrix in Mat::InvSY");

	const double* A;		// DSYTRF decomposition results
	const int* ipiv;
	dsytrf(&A, &ipiv);

	std::vector<double> res(A, A + icount * icount);		// A is copied to "res"
	int info = LAPACKE_dsytri(LAPACK_ROW_MAJOR, 'U', icount, res.data(), icount, ipiv);
	if (info != 0)
	{
		std::string msg = stringFormatArr("DSYTRI ����������� � info {0:%d}, �������� �������:", "DSYTRI exited with info {0:%d}, original matrix:", info);
		debug_output(msg + "\n", this);
		throw Exception(msg);
	}

	// symmetrically fill the lower triangular part
	for (size_t i = 1; i < icount; i++)
		for (size_t j = 0; j < i; j++)
			res[i * jcount + j] = res[j * jcount + i];

	return Mat(std::move(res), icount, icount);
}
//------------------------------------------------------------------------------------------
double Mat::LnDetSY(int& sign) const
{
	if (icount != jcount)
		throw Exception("Non-square matrix in Mat::LnDetSY");

	const double* A;		// DSYTRF decomposition results
	const int* ipiv;		// ipiv[i] are 1-based indices
	dsytrf(&A, &ipiv);

	double res = 0;
	sign = 1;
	for (size_t i = 0; i < icount; i++)
		if (ipiv[i] > 0)	// 1x1 block
		{
			double d = A[i * jcount + i];
			if (d < 0)
			{
				d = -d;
				sign = -sign;
			}
			res += log(d);
		}
		else				// 2x2 block
		{
			assert(i < icount - 1);
			double d = A[i * jcount + i] * A[(i + 1) * jcount + i + 1] - A[i * jcount + i + 1] * A[i * jcount + i + 1];
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

	double* W = new double[icount];						// storage for eigenvalues
	double* A = new double[icount * icount];				// copy of the matrix data (it will be destroyed by the procedure)
	memcpy(A, data.data(), icount * icount * sizeof(double));

	int lda = icount;
	int M;												// number of eigenvalues found
	double abstol = LAPACKE_dlamch('S');
	int info = LAPACKE_dsyevr(LAPACK_ROW_MAJOR, 'N', 'I', 'U', icount, A, lda, 0, 0, I0 + 1, I1, abstol, &M, W, 0, icount, 0);
	if (info != 0)
	{
		delete[] W;
		delete[] A;

		std::string msg = stringFormatArr("DSYEVR ����������� � info {0:%d}", "DSYEVR exited with info {0:%d}", info);
		debug_output(msg + "\n", this);
		throw Exception(msg);
	}

	std::vector<double> res(W, W + M);

	delete[] W;
	delete[] A;

#ifdef TESTING
	std::cout << "Mat::EigVal found " << M << " eigenvalues" << std::endl;
#endif

	return res;
}
//------------------------------------------------------------------------------------------
std::vector<double> Mat::EigVal(size_t I0, size_t I1, Mat& EigVec) const
{
	EigVec.reset_chol_spo_cache();
	EigVec.reset_dsytrf_cache();

	if (icount != jcount)
		throw Exception("Non-square matrix in Mat::EigVal");
	if (I0 >= I1 || I1 > icount)
		throw Exception("Wrong indices I0, I1 in Mat::EigVal");

	int M = (int)I1 - (int)I0;							// number of eigenvalues found (well, its value is known from indices)

	double* W = new double[icount];						// storage for eigenvalues
	double* A = new double[icount * icount];				// copy of the matrix data (it will be destroyed by the procedure)
	double* Z = new double[icount * M];					// storage for eigenvectors
	int* isuppz = new int[2 * M];
	memcpy(A, data.data(), icount * icount * sizeof(double));

	int lda = icount;
	int ldz = M;		// ldz = icount for column-major

	double abstol = LAPACKE_dlamch('S');
	int info = LAPACKE_dsyevr(LAPACK_ROW_MAJOR, 'V', 'I', 'U', icount, A, lda, 0, 0, I0 + 1, I1, abstol, &M, W, Z, ldz, isuppz);
	if (info != 0)
	{
		delete[] W;
		delete[] A;
		delete[] Z;
		delete[] isuppz;

		std::string msg = stringFormatArr("DSYEVR ����������� � info {0:%d}", "DSYEVR exited with info {0:%d}", info);
		debug_output(msg + "\n", this);
		throw Exception(msg);
	}
	assert(M == (int)I1 - (int)I0);		// M should not have changed!

	std::vector<double> res(W, W + M);
	EigVec = HMMPI::Mat(std::vector<double>(Z, Z + icount * M), icount, M);

	delete[] W;
	delete[] A;
	delete[] Z;

#ifdef TESTING
	std::cout << "Mat::EigVal found " << M << " eigenvalues" << std::endl;
	std::cout << "isuppz = " << HMMPI::ToString(std::vector<int>(isuppz, isuppz + 2 * M), "%d");
#endif

	delete[] isuppz;

	return res;
}
//------------------------------------------------------------------------------------------
std::vector<double> Mat::SgVal() const
{
	double* A = new double[icount * jcount];					// copy of the matrix data (it will be destroyed by the procedure)
	memcpy(A, data.data(), icount * jcount * sizeof(double));

	int res_size = (icount < jcount) ? icount : jcount;		// min
	std::vector<double> res(res_size);

	int lda = jcount;
	int info = LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'N', icount, jcount, A, lda, res.data(), NULL, icount, NULL, jcount);
	if (info != 0)
	{
		delete[] A;
		std::string msg = stringFormatArr("DGESDD ����������� � info {0:%d}", "DGESDD exited with info {0:%d}", info);
		debug_output(msg + "\n", this);
		throw Exception(msg);
	}

	delete[] A;
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
		std::string msg = stringFormatArr("DPOCON ����������� � info {0:%d}", "DPOCON exited with info {0:%d}", info);
		debug_output(msg + "\n", &chol);
		throw Exception(msg);
	}
	return res;
}
//------------------------------------------------------------------------------------------
Mat Mat::BFGS_update_B(const Mat& dk, const Mat& gk) const
{
	Mat Bdelta = (*this) * dk;
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

	return (*this) + (1 / gamma_delta) * OuterProd(gk, gk) + (-1 / delta_B_delta) * OuterProd(Bdelta, Bdelta);
}
//------------------------------------------------------------------------------------------
Mat Mat::BFGS_update_B(const std::vector<Mat>& Xk, const std::vector<Mat>& Gk) const
{
	if (Xk.size() != Gk.size())
		throw Exception("Xk.size() != Gk.size() in Mat::BFGS_update_B");
	if (Xk.size() < 2)
		throw Exception("Xk.size() < 2 in Mat::BFGS_update_B");

	// fill delta and gamma - coordinate and gradient changes
	std::vector<Mat> dk(Xk.size() - 1);
	std::vector<Mat> gk(Xk.size() - 1);
	for (size_t i = 0; i < Xk.size() - 1; i++)
	{
		dk[i] = Xk[i + 1] - Xk[i];
		gk[i] = Gk[i + 1] - Gk[i];
	}

	// make a series of BFGS updates
	Mat res = BFGS_update_B(dk[0], gk[0]);
	for (size_t i = 1; i < dk.size(); i++)
		res = res.BFGS_update_B(dk[i], gk[i]);

	return res;
}
//------------------------------------------------------------------------------------------
double InnerProd(const Mat& a, const Mat& b)	// (a, b), inner product of two vectors; using Manual | BLAS depending on 'a.op_switch'
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
Mat OuterProd(const Mat& a, const Mat& b)
{
	if (a.jcount != 1 || b.jcount != 1)
		throw Exception(stringFormatArr("Outer product should be applied to column-vectors ({0:%zu} != 1 || {1:%zu} != 1)", std::vector<size_t>{a.jcount, b.jcount}));

	Mat res(a.icount, b.icount, 0);
	double* pres = res.data.data();
	const double* pa = a.data.data();
	const double* pb = b.data.data();

	for (size_t i = 0; i < res.icount; i++)
		for (size_t j = 0; j < res.jcount; j++)
			pres[res.jcount * i + j] = pa[i] * pb[j];

	return res;
}
//------------------------------------------------------------------------------------------
Mat VecProd(const Mat& a, const Mat& b)			// a (x) b, vector product of two 3-dim vectors
{
	if (a.icount != 3 || b.icount != 3 || a.jcount != 1 || b.jcount != 1)
		throw Exception(stringFormatArr("Vector product should be applied to 3-dim column-vectors ({0:%zu} != 3 || {1:%zu} != 3 || {2:%zu} != 1 || {3:%zu} != 1)",
			std::vector<size_t>{a.icount, b.icount, a.jcount, b.jcount}));
	Mat res(3, 1, 0.0);
	const double* A = a.data.data();
	const double* B = b.data.data();
	double* dat_res = res.data.data();

	dat_res[2] = A[0] * B[1] - A[1] * B[0];

	if (A[2] != 0 || B[2] != 0)					// vectors have z-components
	{
		dat_res[1] = -(A[0] * B[2] - A[2] * B[0]);
		dat_res[0] = A[1] * B[2] - A[2] * B[1];
	}

	return res;
}
//------------------------------------------------------------------------------------------
Mat operator*(double d, Mat m)		// number * Mat
{
	// m is a copy or rvalue

	m.reset_chol_spo_cache();
	m.reset_dsytrf_cache();
	size_t SZ = m.icount * m.jcount;
	double* pm = m.data.data();
	for (size_t i = 0; i < SZ; i++)
		pm[i] *= d;

	return m;
}
//------------------------------------------------------------------------------------------
Mat operator%(const std::vector<double>& v, Mat m)	// diag * Mat
{
	m.reset_chol_spo_cache();
	m.reset_dsytrf_cache();

	if (v.size() != m.icount)
		throw Exception("Inconsistent dimensions in diag(vector) % Mat");

	for (size_t i = 0; i < m.icount; i++)
		for (size_t j = 0; j < m.jcount; j++)
			m.data[i * m.jcount + j] *= v[i];

	return m;
}
//------------------------------------------------------------------------------------------
Mat operator%(Mat m, const std::vector<double>& v)	// Mat * diag
{
	m.reset_chol_spo_cache();
	m.reset_dsytrf_cache();

	if (v.size() != m.jcount)
		throw Exception("Inconsistent dimensions in Mat % diag(vector)");

	for (size_t i = 0; i < m.icount; i++)
		for (size_t j = 0; j < m.jcount; j++)
			m.data[i * m.jcount + j] *= v[j];

	return m;
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
	double* pwork = A.data.data();
	double* pm = b.data.data();
	double* pres = res.data.data();

	std::vector<int> piv(icount);			// pivots reordering
	std::iota(piv.begin(), piv.end(), 0);	// fill with 0, 1, 2,...

	try
	{
		for (size_t i = 0; i < icount; i++)
		{
			size_t max_i = i;		// newly found pivot
			double max = fabs(pwork[piv[max_i] * icount + i]);
			for (size_t j = i + 1; j < icount; j++)
				if (fabs(pwork[piv[j] * icount + i]) > max)
				{
					max_i = j;
					max = fabs(pwork[piv[j] * icount + i]);
				}

			std::swap(piv[i], piv[max_i]);

#ifdef TESTING
			if (i != max_i)
				std::cout << "operator/(Mat, Mat) swapping rows " << i << " and " << max_i << std::endl;
#endif

			// for the lines below piv is fixed
			int pivi = piv[i];
			if (pwork[pivi * icount + i] == 0)
				throw Exception("Determinant = 0, no solution exists in operator/(Mat, Mat)");

			for (size_t j = i + 1; j < icount; j++)
			{
				int pivj = piv[j];
				if (pwork[pivj * icount + i] != 0)
				{
					double mult = -pwork[pivj * icount + i] / pwork[pivi * icount + i];
					for (size_t k = i + 1; k < icount; k++)
						pwork[pivj * icount + k] += pwork[pivi * icount + k] * mult;

					pwork[pivj * icount + i] = 0;

					// pm[pivj] += pm[pivi] * mult;		-- old version - for 1 RHS
					for (size_t k = 0; k < rhscount; k++)
						pm[pivj * rhscount + k] += pm[pivi * rhscount + k] * mult;
				}
			}
		}

		for (int i = icount - 1; i >= 0; i--)
		{
			int pivi = piv[i];
			if (pwork[pivi * icount + i] == 0)
				throw Exception("Determinant = 0, no solution exists in operator/(Mat, Mat)");

			for (size_t k = 0; k < rhscount; k++)
			{
				double aux = pm[pivi * rhscount + k];
				for (size_t j = i + 1; j < icount; j++)
					aux -= pwork[pivi * icount + j] * pres[j * rhscount + k];

				pres[i * rhscount + k] = aux / pwork[pivi * icount + i];
			}
		}
	}
	catch (const Exception& e)
	{
		std::string msg = e.what() + std::string("\nTransformed matrix 'A' which caused the exception\n");
		A.debug_output(msg, &A);

		throw e;
	}

	return res;
}
//------------------------------------------------------------------------------------------

}		// namespace HMMPI
