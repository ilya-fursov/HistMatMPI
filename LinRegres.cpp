/*
 * LinRegres.cpp
 *
 *  Created on: Mar 28, 2013
 *      Author: ilya
 */

#include "MathUtils.h"
#include "LinRegress.h"
#include "Parsing.h"
#include "Parsing2.h"
#include <cmath>
#include <limits>
#include <iostream>
#include <fstream>
#include "mpi.h"
#include "Tracking.h"

double LinRegress::eps;
KW_regressConstr *RegEntryConstr::regcon;
KW_regressConstr *LinRegressConstr::regcon;
std::string RegEntry::CWD;
const double PI = acos(-1.0);
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
std::vector<double> LinRegress::Norm2()
{
	countPTS = Xi.ICount();
	countIV = Xi.JCount();

	std::vector<double> res(countIV);
	for (size_t i = 0; i < countIV; i++)
	{
		double d = 0;
		for (size_t j = 0; j < countPTS; j++)
			d += Xi(j, i)*Xi(j, i);

		res[i] = sqrt(d);
	}

	return res;
}
//------------------------------------------------------------------------------------------
void LinRegress::UpdateXi()
{
	std::vector<double> norm2 = Norm2();

	normCoeff = norm2;
	for (size_t i = 0; i < countPTS; i++)
		for (size_t j = 0; j < countIV; j++)
			Xi(i, j) /= normCoeff[j];

	double max = norm2[0];
	for (size_t i = 0; i < countIV; i++)
	{
		if (norm2[i] > max)
			max = norm2[i];
	}
	if (max > 0)
	{
		for (size_t i = 0; i < countIV; i++)
			norm2[i] /= max;

		act_var = std::vector<int>(countIV);
		int c = 0;
		for (size_t i = 0; i < countIV; i++)
		{
			if (norm2[i] < eps)
				act_var[i] = -1;
			else
			{
				act_var[i] = c;
				c++;
			}
		}

		HMMPI::Vector2<double> Xinew(countPTS, c);
		for (size_t i = 0; i < countPTS; i++)
		{
			for (size_t j = 0; j < countIV; j++)
				if (act_var[j] != -1)
					Xinew(i, act_var[j]) = Xi(i, j);
		}

		Xi = Xinew;
		countIV = c;
	}
	else
	{
		// óáèðàåì âñå ñòîëáöû
		Xi = HMMPI::Vector2<double>(countPTS, 0);
		act_var = std::vector<int>(countIV);
		for (size_t i = 0; i < countIV; i++)
			act_var[i] = -1;

		countIV = 0;
	}
}
//------------------------------------------------------------------------------------------
void LinRegress::UpdateRes()
{
	size_t count = act_var.size();
	std::vector<double> res(count);
	for (size_t i = 0; i < count; i++)
	{
		if (act_var[i] != -1)
			res[i] = sol[act_var[i]];
		else
			res[i] = std::numeric_limits<double>::quiet_NaN();

		res[i] /= normCoeff[i];
	}

	sol = res;
}
//------------------------------------------------------------------------------------------
double LinRegress::ScalProdXX(int i, int j)
{
	double res = 0;
	for (size_t k = 0; k < countPTS; k++)
		res += Xi(k, i) * Xi(k, j);

	return res;
}
//------------------------------------------------------------------------------------------
double LinRegress::ScalProdXf(int i)
{
	double res = 0;
	for (size_t k = 0; k < countPTS; k++)
		res += Xi(k, i) * f[k];

	return res;
}
//------------------------------------------------------------------------------------------
void LinRegress::Swap(int cur)
{
	int max_i;
	size_t SIZE = M.ICount();
	double max, aux;

	max_i = cur;
	max = fabs(M(cur, cur));
	for (size_t i = cur+1; i < SIZE; i++)
	{
		if (fabs(M(i, cur)) > max)
		{
			max_i = i;
			max = fabs(M(i, cur));
		}
	}

	if (max_i != cur)
	{
		for (size_t i = 0; i < SIZE; i++)
		{
			aux = M(max_i, i);
			M(max_i, i) = M(cur, i);
			M(cur, i) = aux;
		}
		aux = RHS[max_i];
		RHS[max_i] = RHS[cur];
		RHS[cur] = aux;
	}
}
//------------------------------------------------------------------------------------------
void LinRegress::FillMatr()
{
	countPTS = Xi.ICount();
	countIV = Xi.JCount();

	if (countPTS != f.size())
		throw HMMPI::EObjFunc("(eng) LinRegress::FillMatr",
							  "Xi and f sizes do not match in LinRegress::FillMatr");

	RHS = std::vector<double>(countIV);
	M = HMMPI::Vector2<double>(countIV, countIV);

	for (size_t i = 0; i < countIV; i++)
	{
		RHS[i] = ScalProdXf(i);
		for (size_t j = 0; j < countIV; j++)
			M(i, j) = ScalProdXX(i, j);
	}
}
//------------------------------------------------------------------------------------------
void LinRegress::CalcR2()
{
	if (countIV == 0)
		R2 = std::numeric_limits<double>::quiet_NaN();
	else
	{
		double barf = 0;
		for (size_t i = 0; i < countPTS; i++)
			barf += f[i];

		barf /= countPTS;

		double SStot = 0, SSerr = 0;
		for (size_t i = 0; i < countPTS; i++)
		{
			SStot += (f[i] - barf)*(f[i] - barf);
			double modfi = 0;
			for (size_t j = 0; j < countIV; j++)
				modfi += sol[j] * Xi(i, j);

			SSerr += (f[i] - modfi)*(f[i] - modfi);
		}
		R2 = 1 - SSerr/SStot;
	}
}
//------------------------------------------------------------------------------------------
void LinRegress::SolveGauss()
{
	size_t SIZE = M.ICount();
	std::vector<double> res(SIZE);

	for (size_t i = 0; i < SIZE; i++)
	{
		Swap(i);
		if (M(i, i) == 0)
			throw HMMPI::EObjFunc("(eng) LinRegress::SolveGauss",
									 "Determinant = 0, no solution exists in LinRegress::SolveGauss");

		for (size_t j = i+1; j < SIZE; j++)
		{
			double mult = -M(j, i) / M(i, i);
			for (size_t k = i+1; k < SIZE; k++)
				M(j, k) += M(i, k) * mult;

			M(j, i) = 0;
			RHS[j] += RHS[i] * mult;
		}
	}

	for (int i = SIZE-1; i >= 0; i--)
	{
		double aux = RHS[i];
		for (size_t j = i+1; j < SIZE; j++)
			aux -= M(i, j) * res[j];

		if (M(i, i) == 0)
			throw HMMPI::EObjFunc("Îïðåäåëèòåëü = 0, ðåøåíèé íåò â LinRegress::SolveGauss",
									 "Determinant = 0, no solution exists in LinRegress::SolveGauss");

		res[i] = aux / M(i, i);
	}

	sol = res;
}
//------------------------------------------------------------------------------------------
LinRegress::LinRegress() : TOLbeta(-1e-18)
{
	eps = 1e-20;
	R2 = 0;
	avgA0 = 0;
	countPTS = 0;
	countIV = 0;

	TOLcur = 0;
	minAx = 0;
	minAxind = -1;
	normAi = 0;
	normxcur = 0;
	func_val = 0;

	//c_Cov = c_L = 0;
	RLS = 0;
	//cT = sz = 0;
	RLS_ind = -1;
	w5 = 0;
}
//------------------------------------------------------------------------------------------
std::vector<double> LinRegress::Solve0()
{
	UpdateXi();
	FillMatr();
	SolveGauss();
	CalcR2();

	UpdateRes();
	return sol;
}
//------------------------------------------------------------------------------------------
void LinRegress::CalcAvgA0()
{
	double res = 0;
	for (size_t i = 0; i < A0.size(); i++)
		res += A0[i];

	avgA0 = res/A0.size();
}
//------------------------------------------------------------------------------------------
void LinRegress::LoadMatr(const RegListSpat *rls, int rls_ind, double W5)
{
	RLS = rls;
	RLS_ind = rls_ind;
	w5 = W5;
}
//------------------------------------------------------------------------------------------
// LinRegressConstr
//------------------------------------------------------------------------------------------
std::vector<double> LinRegressConstr::vect_Vi(const HMMPI::Vector2<double> &A, int col, double &alpha)
{
	size_t n = A.ICount();

	double norm = 0;
	for (size_t i = col; i < n; i++)
		norm += A(i, col)*A(i, col);

	norm = sqrt(norm);

	alpha = norm;
	if (A(col, col) >= 0)
		alpha = -alpha;

	std::vector<double> u(n-col);

	u[0] = A(col, col) + alpha;
	norm = u[0]*u[0];
	for (size_t i = 1; i < n-col; i++)
	{
		u[i] = A(i+col, col);
		norm += u[i]*u[i];
	}
	norm = sqrt(norm);

	if (norm != 0)
	{
		for (size_t i = 0; i < n-col; i++)
			u[i] /= norm;
	}

	return u;
}
//------------------------------------------------------------------------------------------
void LinRegressConstr::updateM(HMMPI::Vector2<double> &M, int col, double alpha, const std::vector<double> &v)
{
	size_t n = M.ICount();
	size_t m = M.JCount();

	M(col, col) = -alpha;
	for (size_t i = col+1; i < n; i++)
		M(i, col) = 0;

	for (size_t j = col+1; j < m; j++)
	{
		// <v, mj>
		double sp = 0;
		for (size_t k = col; k < n; k++)
			sp += M(k, j) * v[k-col];

		sp *= -2;

		for (size_t k = col; k < n; k++)
			M(k, j) += sp * v[k-col];
	}
}
//------------------------------------------------------------------------------------------
void LinRegressConstr::CalcGg()
{
	// new version adds some code here
	const RegEntryConstr *REC = RLS->GetReg(RLS_ind);

	int sz = REC->sz;
	int cT = REC->cT;

	if (sz*cT != (int)countPTS)
		throw HMMPI::EObjFunc(HMMPI::stringFormatArr(HMMPI::MessageRE("При вычислении матрицы для квадратичного программирования найдено точек данных: {0:%d}, "
				"что не соответствует размеру ковариационной матрицы {1:%d} и числу временных шагов {2:%d} в LinRegressConstr::CalcGg. "
				"Следует правильно задать неактивные ячейки в гриде EKMAPREG, т.к. этот грид определит размер ковариационной матрицы",
				"When calculating matrix for quadratic programming encountered {0:%d} data points, "
				"which doesn't match the size of covariance matrix {1:%d} and number of time steps {2:%d} in LinRegressConstr::CalcGg. "
				"Make sure inactive cells are properly defined in EKMAPREG grid, as this grid controls the covariance matrix size"),
				std::vector<int>{(int)countPTS, sz, cT}));

	// матрицы создаются поблочно для каждого временного шага
	HMMPI::Vector2<double> Xicopy = Xi;
	std::vector<double> fcopy = f;
	HMMPI::Vector2<double> Psingle(sz, countIV);	// подвектор для одного врем. шага
	HMMPI::Vector2<double> Ps_single(sz, countIV);
	std::vector<double> fs_single(sz);

	for (int t = 0; t < cT; t++)	// loop through time steps
	{
		HMMPI::Vector2<double>::Copy(Xicopy, t*sz, (t+1)*sz, 0, countIV, Psingle, 0, sz, 0, countIV);	// submatrix
		std::vector<double> fsingle(fcopy.begin() + t*sz, fcopy.begin() + (t+1)*sz);						// subvector
		RLS->Linv_P(Psingle, Ps_single, RLS_ind);		// multiply by 1/L
		RLS->Linv_v(fsingle, fs_single, RLS_ind);
		HMMPI::Vector2<double>::Copy(Ps_single, 0, sz, 0, countIV, Xicopy, t*sz, (t+1)*sz, 0, countIV);	// bigger matrix
		for (int i = 0; i < sz; i++)
			fcopy[t*sz + i] = fs_single[i];															// bigger std::vector
	}

	f2 = 0;
	for (size_t i = 0; i < countPTS; i++)
		f2 += fcopy[i]*fcopy[i];
	f2 *= w5;

	// old version starts here
	g = std::vector<double>(countIV);
	G = HMMPI::Vector2<double>(countIV, countIV);

	for (size_t i = 0; i < countIV; i++)
	{
		double res = 0;		// ScalProdXf(i)
		for (size_t k = 0; k < countPTS; k++)
			res += Xicopy(k, i) * fcopy[k];

		g[i] = -res*w5;

		for (size_t j = 0; j < countIV; j++)
		{
			double res = 0;	// ScalProdXX(i, j)
			for (size_t k = 0; k < countPTS; k++)
				res += Xicopy(k, i) * Xicopy(k, j);

			G(i, j) = res*w5;
		}
	}

#ifdef WRITE_LINREGRESS_FILES
		std::ofstream fileG;
		fileG.open(HMMPI::stringFormatArr(RegEntry::CWD + "/correl_Gg_{0:%d}.txt", std::vector<int>{(int)RLS_ind}));
		fileG << "f2\nG | g\n" << f2 << std::endl;
		for (size_t i = 0; i < countIV; i++)
		{
			for (size_t j = 0; j < countIV; j++)
				fileG << G(i, j) << "\t";
			fileG << g[i] << std::endl;
		}
		fileG.close();

		std::ofstream fileXi;
		fileXi.open(HMMPI::stringFormatArr(RegEntry::CWD + "/correl_Xif_{0:%d}.txt", std::vector<int>{(int)RLS_ind}));
		fileXi << "Xi | f" << std::endl;
		for (size_t i = 0; i < countPTS; i++)
		{
			for (size_t j = 0; j < countIV; j++)
				fileXi << Xi(i, j) << "\t";
			fileXi << f[i] << std::endl;
		}
		fileXi.close();

#endif
}
//------------------------------------------------------------------------------------------
void LinRegressConstr::UpdateAxcur()
{
	size_t c = normCoeff.size();		// original countIV
	size_t count = A.JCount();

	std::vector<double> xnew(countIV);
	for (size_t i = 0; i < c; i++)
	{
		if (act_var[i] != -1)
			xnew[act_var[i]] = xcur[i] * normCoeff[i];
	}
	xcur = xnew;

	HMMPI::Vector2<double> Anew(countIV, count);
	for (size_t i = 0; i < c; i++)
	{
		for (size_t j = 0; j < count; j++)
			if (act_var[i] != -1)
				Anew(act_var[i], j) = A(i, j) / normCoeff[i];
	}

	A = Anew;
}
//------------------------------------------------------------------------------------------
void LinRegressConstr::actA_fromX1()
{
	size_t count = A.JCount();
	actind = std::vector<int>(count);
	for (size_t j = 0; j < count; j++)
		actind[j] = 0;

	size_t tot = 0;
	actA = HMMPI::Vector2<double>(countIV, 0);
	HMMPI::Vector2<double> prevA;
	for (size_t j = 0; j < count; j++)
	{
		double d = Ai_x(j, xcur);
		if (d == b[j])
		{
			actind[j] = 1;
			tot++;

			// make matrix actA for rank check
			actA = HMMPI::Vector2<double>(countIV, tot);
			int cur = 0;
			for (size_t m = 0; m < count; m++)
			{
				if (actind[m])
				{
					for (size_t i = 0; i < countIV; i++)
						actA(i, cur) = A(i, m);

					cur++;
				}
			}

			if (Rank(actA) < (int)tot)
			{
				actind[j] = 0;
				tot--;

				actA = prevA;
			}
			else
			{
				prevA = actA;
			}
		}
		else if (d < b[j])
			throw HMMPI::EObjFunc(HMMPI::MessageRE("(eng)", "Unfeasible x0"));

		if (tot >= countIV)	// don't take more constraints than variables
			break;
	}
}
//------------------------------------------------------------------------------------------
void LinRegressConstr::actA_fromind()
{
	size_t count = actind.size();
	int tot = 0;
	for (size_t i = 0; i < count; i++)
	{
		if (actind[i])
			tot++;
	}

	actA = HMMPI::Vector2<double>(countIV, tot);
	int cur = 0;
	for (size_t j = 0; j < count; j++)
	{
		if (actind[j])
		{
			for (size_t i = 0; i < countIV; i++)
				actA(i, cur) = A(i, j);

			cur++;
		}
	}
}
//------------------------------------------------------------------------------------------
void LinRegressConstr::qrDecomp()
{
	std::vector<HMMPI::Vector2<double>> QR = QRdecomp(actA);
	if (QR.size() == 0)
	{
		Q1t = HMMPI::Vector2<double>();
		Q2t = HMMPI::Vector2<double>();
		R = HMMPI::Vector2<double>();

		qr_decomp_flag = 0;
	}
	else
	{
		HMMPI::Vector2<double> Q = QR[0];

		size_t lenR = QR[1].JCount();
		R = HMMPI::Vector2<double>(lenR, lenR);
		for (size_t i = 0; i < lenR; i++)
			for (size_t j = 0; j < lenR; j++)
				R(i, j) = QR[1](i, j);

		size_t m = actA.JCount();
		Q1t = HMMPI::Vector2<double>(m, countIV);
		Q2t = HMMPI::Vector2<double>(countIV-m, countIV);
		for (size_t i = 0; i < countIV; i++)
		{
			for (size_t j = 0; j < countIV; j++)
			{
				if (j < m)
					Q1t(j, i) = Q(i, j);
				else
					Q2t(j-m, i) = Q(i, j);
			}
		}

		qr_decomp_flag = 1;
	}
}
//------------------------------------------------------------------------------------------
double LinRegressConstr::Ai_x(int i, const std::vector<double> &x)
{
	if (x.size() != countIV)
		throw HMMPI::EObjFunc(HMMPI::MessageRE("(eng)", "Inconsistent dimensions in LinRegressConstr::Ai_x"));

	double res = 0;
	for (size_t j = 0; j < countIV; j++)
		res += A(j, i) * x[j];

	return res;
}
//------------------------------------------------------------------------------------------
void LinRegressConstr::FillMatr()
{
	size_t count;
	if (qr_decomp_flag)
		count = Q2t.ICount();
	else
		count = countIV;

	M = HMMPI::Vector2<double>(count, count);

	// M
	if (qr_decomp_flag)
	{
		for (size_t i = 0; i < count; i++)
		{
			for (size_t n = 0; n < count; n++)
			{
				double v = 0;
				for (size_t j = 0; j < countIV; j++)
					for (size_t k = 0; k < countIV; k++)
						v += Q2t(i, j) * G(j, k) * Q2t(n, k);

				M(i, n) = v;
			}
		}
	}
	else
		M = G;

	// RHS
	if (qr_decomp_flag)
		RHS = MultMatrVect(Q2t, gcur);
	else
		RHS = gcur;

	for (size_t i = 0; i < count; i++)
		RHS[i] = -RHS[i];
}
//------------------------------------------------------------------------------------------
double LinRegressConstr::calcBeta(const std::vector<double> &sk, int &p)
{
	double TOL = TOLcur;
	size_t count = actind.size();

	double min = 1;
	p = -1;
	for (size_t i = 0; i < count; i++)
	{
		if (actind[i] == 0)
		{
			double ai_sk = Ai_x(i, sk);
			if (ai_sk < TOL)
			{
				double d = (b[i] - Ai_x(i, xcur)) / ai_sk;
				if (d <= 1 && d <= min)
				{
					min = d;
					p = i;
				}
			}
		}
	}

	return min;
}
//------------------------------------------------------------------------------------------
int LinRegressConstr::minLambda()
{
	size_t count = actind.size();
	int res = -1;
	int n = 0;
	double min = 0;
	for (size_t i = 0; i < count; i++)
	{
		if (actind[i])
		{
			if (lambdacur[n] < min)
			{
				res = i;
				min = lambdacur[n];
			}
			n++;
		}
	}

	return res;
}
//------------------------------------------------------------------------------------------
void LinRegressConstr::minAxcur(const std::vector<double> &xcur, int &ind, double &min)
{
	ind = -1;
	min = 0;
	size_t count = A.JCount();
	for (size_t i = 0; i < count; i++)
	{
		double d = Ai_x(i, xcur);
		if (d < min)
		{
			min = d;
			ind = i;
		}
	}
}
//------------------------------------------------------------------------------------------
double LinRegressConstr::NormAi(int i)
{
	if (i == -1)
		return 0;

	if (i >= (int)A.JCount())
		throw HMMPI::EObjFunc(HMMPI::MessageRE("(eng)", "Index out of range in LinRegressConstr::NormAi"));

	size_t len = A.ICount();
	double res = 0;
	for (size_t k = 0; k < len; k++)
		res += A(k, i)*A(k, i);

	return sqrt(res);
}
//------------------------------------------------------------------------------------------
double LinRegressConstr::Norm(const std::vector<double> &v)
{
	double res = 0;
	for (size_t i = 0; i < v.size(); i++)
		res += v[i]*v[i];

	return sqrt(res);
}
//------------------------------------------------------------------------------------------
std::vector<HMMPI::Vector2<double>> LinRegressConstr::QRdecomp(const HMMPI::Vector2<double> &A)
{
	size_t n = A.ICount();
	size_t m = A.JCount();
	size_t tmax;

	if (n == m)
		tmax = n-1;
	else if (n > m)
		tmax = m;
	else
		throw HMMPI::EObjFunc(HMMPI::MessageRE("(eng)", "In QR decomposition n >= m is expected"));

	if (m == 0)
		return std::vector<HMMPI::Vector2<double>>();

	HMMPI::Vector2<double> M = A;

	// Q1
	double alpha;
	std::vector<double> v = vect_Vi(M, 0, alpha);
	HMMPI::Vector2<double> P(n, n);
	for (size_t i = 0; i < n; i++)
	{
		for (size_t j = 0; j < n; j++)
			P(i, j) = -2*v[i]*v[j];

		P(i, i) += 1;
	}

	updateM(M, 0, alpha, v);

	// main cycle
	for (size_t t = 1; t < tmax; t++)
	{
		v = vect_Vi(M, t, alpha);

		// update P
		for (size_t i = 0; i < n; i++)
		{
			// <pi, v>
			double sp = 0;
			for (size_t j = t; j < n; j++)
				sp += P(i, j) * v[j-t];

			sp *= -2;

			for (size_t j = t; j < n; j++)
				P(i, j) += sp * v[j-t];
		}

		updateM(M, t, alpha, v);
	}
	std::vector<HMMPI::Vector2<double>> res(2);
	res[0] = std::move(P);
	res[1] = std::move(M);
	return res;
}
//------------------------------------------------------------------------------------------
int LinRegressConstr::Rank(const HMMPI::Vector2<double> &A)
{
	const double TOL = 1e-15;
	std::vector<HMMPI::Vector2<double>> QR = QRdecomp(A);
	HMMPI::Vector2<double> R = QR[1];

	int res = 0;
	size_t col = R.JCount();
	for (size_t i = 0; i < col; i++)
		if (fabs(R(i, i)) > TOL)
			res++;

	return res;
}
//------------------------------------------------------------------------------------------
std::vector<double> LinRegressConstr::MultMatrVect(const HMMPI::Vector2<double> &M, const std::vector<double> &v)
{
	size_t c0 = M.ICount();
	size_t c1 = M.JCount();
	if (v.size() != c1)
		throw HMMPI::EObjFunc(HMMPI::MessageRE("(eng)", "Dimensions mismatch in LinRegressConstr::MultMatrVect"));

	std::vector<double> res(c0);
	for (size_t i = 0; i < c0; i++)
	{
		double d = 0;
		for (size_t j = 0; j < c1; j++)
			d += M(i, j)*v[j];

		res[i] = d;
	}

	return res;
}
//------------------------------------------------------------------------------------------
std::vector<double> LinRegressConstr::MultMatrTVect(const HMMPI::Vector2<double> &M, const std::vector<double> &v)
{
	size_t c0 = M.JCount();
	size_t c1 = M.ICount();
	if (v.size() != c1)
		throw HMMPI::EObjFunc(HMMPI::MessageRE("(eng)", "Dimensions mismatch in LinRegressConstr::MultMatrTVect"));

	std::vector<double> res(c0);
	for (size_t i = 0; i < c0; i++)
	{
		double d = 0;
		for (size_t j = 0; j < c1; j++)
			d += M(j, i)*v[j];

		res[i] = d;
	}

	return res;
}
//------------------------------------------------------------------------------------------
std::vector<double> LinRegressConstr::Rinv(const HMMPI::Vector2<double> &Rmatr, const std::vector<double> &v)
{
	size_t lenR = Rmatr.ICount();
	if (v.size() != lenR || Rmatr.JCount() != lenR)
		throw HMMPI::EObjFunc(HMMPI::MessageRE("(eng)", "Inconsistent dimensions in LinRegressConstr::Rinv"));

	std::vector<double> res(lenR);
	for (int i = lenR-1; i >= 0; i--)
	{
		double aux = v[i];
		for (size_t j = i+1; j < lenR; j++)
			aux -= Rmatr(i, j) * res[j];

		if (Rmatr(i, i) == 0)
			throw HMMPI::EObjFunc(HMMPI::MessageRE("(eng) LinRegressConstr::Rinv",
									  "Determinant = 0, no solution exists in LinRegressConstr::Rinv"));
		res[i] = aux / Rmatr(i, i);
	}

	return res;
}
//------------------------------------------------------------------------------------------
LinRegressConstr::LinRegressConstr(bool make_reg) : LinRegress()
{
	make_regress = make_reg;
	qr_decomp_flag = 0;
	c_EP = c_Lagr = 0;
	f2 = 0;
}
//------------------------------------------------------------------------------------------
std::vector<double> LinRegressConstr::Solve0()
{
	const size_t MAXMULT = 100;

	countIV = xcur.size();
	if (make_regress)
	{
		UpdateXi();
		UpdateAxcur();
		CalcGg();
	}

	double A0sign = (avgA0 > 0)?(1):(-1);
	xcur = regcon->getInitPoint(A0sign, act_var);

	c_Lagr = 0;
	c_EP = 0;
	std::vector<double> delta;

	actA_fromX1();				// a
	size_t MAXDEL = MAXMULT * A.JCount();
	size_t del_count = 0;
	bool finished = false;
	bool zero_solves = false;
	int lastp = -1;

	while (!finished)
	{
		// gcur
		gcur = MultMatrVect(G, xcur);
		for (size_t i = 0; i < countIV; i++)
		{
			gcur[i] += g[i];
		}

		if (zero_solves)		// b
		{
			c_Lagr++;			// c
			if (!qr_decomp_flag)
			{
				lambdacur = std::vector<double>();
				break;
			}
			else
			{
				std::vector<double> rhs = MultMatrVect(Q1t, gcur);
				lambdacur = Rinv(R, rhs);
				int q = minLambda();

				if (q != -1 && q == lastp)
					throw HMMPI::EObjFunc(HMMPI::MessageRE("(eng)", "Active set method added and removed the same constraint"));

				if (q == -1)
					break;
				else
				{
					actind[q] = 0;
					actA_fromind();
					lastp = -1;
					del_count++;
				}

				if (del_count > MAXDEL)
					throw HMMPI::EObjFunc(HMMPI::MessageRE("(eng)", "Active set method removed too many constraints, it is probably stuck"));
			}
		}

		qrDecomp();
		if (qr_decomp_flag)
		{
			if (Q2t.ICount() != 0)
			{
				FillMatr();
				SolveGauss();
				delta = MultMatrTVect(Q2t, sol);	// d
			}
			else	// actA is square
			{
				delta = std::vector<double>(countIV);
				for (size_t i = 0; i < countIV; i++)
					delta[i] = 0;
			}
		}
		else
		{
			FillMatr();
			SolveGauss();
			delta = sol;							// d
		}

		int p;
		double beta = calcBeta(delta, p);			// e
		if (p == -1)
			zero_solves = true;
		else
		{
			zero_solves = false;
			actind[p] = 1;
			actA_fromind();
			lastp = p;
		}

		for (size_t i = 0; i < countIV; i++)
			xcur[i] += beta*delta[i];

		c_EP++;
	}

	minAxcur(xcur, minAxind, minAx);
	normAi = NormAi(minAxind);
	normxcur = Norm(xcur);
	func_val = FuncVal();

	sol = xcur;

	if (make_regress)
	{
		CalcR2();
		UpdateRes();
	}

	return sol;
}
//------------------------------------------------------------------------------------------
double LinRegressConstr::FuncVal()
{
	double res = 0;
	for (size_t i = 0; i < countIV; i++)
		for (size_t j = 0; j < countIV; j++)
			res += xcur[i] * G(i, j) * xcur[j];

	for (size_t i = 0; i < countIV; i++)
		res += 2 * g[i] * xcur[i];

	return res + f2;
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
RegEntry::RegEntry(int rn)
{
	R2 = 0;
	regnum = rn;

	F = std::vector<double>();
	Xi = std::vector<std::vector<double>>();
	SC0 = std::vector<double>();
	I = std::vector<int>();
	J = std::vector<int>();

	TOLbeta = 0;
	minAx = 0;
	minAxind = -1;
	normAi = 0;
	normxcur = 0;
	func_val = 0;

	//c_Cov = c_L = 0;
	RLS = 0;
	RLS_ind = -1;
	w5 = 0;
}
//------------------------------------------------------------------------------------------
void RegEntry::Add(double f, const std::vector<double> &xi, double a0, int i, int j)
{
	F.push_back(f);
	Xi.push_back(xi);
	SC0.push_back(a0);
	I.push_back(i);
	J.push_back(j);
}
//------------------------------------------------------------------------------------------
void RegEntry::MakeArrays()
{
	f = F;
	sc0 = SC0;
	i = I;
	j = J;

	size_t countPTS = Xi.size();
	size_t countIV = Xi[0].size();

	xi = HMMPI::Vector2<double>(countPTS, countIV);
	for (size_t i0 = 0; i0 < countPTS; i0++)
		for (size_t j0 = 0; j0 < countIV; j0++)
			xi(i0, j0) = Xi[i0][j0];
}
//------------------------------------------------------------------------------------------
void RegEntry::AddConstraints(LinRegress *lr)
{
}
//------------------------------------------------------------------------------------------
LinRegress *RegEntry::InitLR()
{
	return new LinRegress();
}
//------------------------------------------------------------------------------------------
void RegEntry::Regression()
{
	LinRegress *LR = InitLR();
	LR->f = f;
	LR->Xi = xi;
	LR->A0 = sc0;
	LR->CalcAvgA0();

	bool finished = false;
	TOLbeta = LR->TOLbeta;
	while (!finished)
	{
		try
		{
			LR->TOLcur = TOLbeta;
			AddConstraints(LR);
			LR->LoadMatr(RLS, RLS_ind, w5); // 10.12.2013
			coeffs = LR->Solve0();
			R2 = LR->R2;
			minAx = LR->minAx;
			minAxind = LR->minAxind;
			normAi = LR->normAi;
			normxcur = LR->normxcur;
			func_val = LR->func_val;

			delete LR;
			finished = true;
		}
		catch (const HMMPI::Exception &e)
		{
			delete LR;
			if (TOLbeta <= -1)
			{
				coeffs = std::vector<double>();
				R2 = std::numeric_limits<double>::quiet_NaN();
				func_val = std::numeric_limits<double>::quiet_NaN();
				finished = false;
				throw;
			}
			else
			{
				TOLbeta *= 10;
				LR = InitLR();
				LR->f = f;
				LR->Xi = xi;
				LR->A0 = sc0;
				LR->CalcAvgA0();
			}
		}
	}
}
//------------------------------------------------------------------------------------------
void RegEntry::WriteToFile(std::string fn)
{
	std::ofstream sw;
	sw.exceptions(std::ios_base::badbit | std::ios_base::failbit);

	try
	{
		sw.open(fn, std::ios_base::app);
		sw << HMMPI::stringFormatArr("регион\t{0:%d}\n", "region\t{0:%d}\n", regnum);

		size_t cI = xi.ICount();
		size_t cJ = xi.JCount();
			
		sw << "i\tj\tA0\tdA\tvar-s\n";
		for (size_t vi = 0; vi < cI; vi++)
		{
			sw << HMMPI::stringFormatArr("{0:%d}\t{1:%d}\t", std::vector<int>{i[vi], j[vi]});
			sw << HMMPI::stringFormatArr("{0}\t", std::vector<double>{sc0[vi]});
			sw << HMMPI::stringFormatArr("{0}\t", std::vector<double>{f[vi]});
			for (size_t vj = 0; vj < cJ; vj++)
			{
				if (vj < cJ-1)
					sw << HMMPI::stringFormatArr("{0}\t", std::vector<double>{xi(vi, vj)});
				else
					sw << HMMPI::stringFormatArr("{0}\n", std::vector<double>{xi(vi, vj)});
			}
		}
		sw << "\n";
		sw.close();
	}
	catch (...)
	{
		if (sw.is_open())
			sw.close();
		throw;
	}
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
void RegEntryConstr::AddConstraints(LinRegress *lr)
{
	KW_regressConstr *regCon = regcon;
	LinRegressConstr *LRC = dynamic_cast<LinRegressConstr*>(lr);

	size_t PTS = LRC->Xi.ICount();
	size_t DIM = LRC->Xi.JCount();

	std::vector<std::vector<double>> auxA;
	std::vector<double> xi(DIM);

	std::vector<bool> to_be_processed(4);

	// only linear
	for (int i = 0; i < 4; i++)
		if (!regCon->hasQuadr(i))
		{
			to_be_processed[i] = false;
			std::vector<double> con = regCon->getConstrFinal(i, xi, LRC->avgA0);
			if (con.size() != 0)
				auxA.push_back(con);
		}
		else
			to_be_processed[i] = true;

	// only quadratic
	for (int i = 0; i < 4; i++)
		if (to_be_processed[i])
		{
			for (size_t p = 0; p < PTS; p++)
			{
				for (size_t j = 0; j < DIM; j++)
					xi[j] = LRC->Xi(p, j);

				std::vector<double> con = regCon->getConstrFinal(i, xi, LRC->A0[p]);
				if (con.size() != 0)
					auxA.push_back(con);
			}
		}

	size_t con_count = auxA.size();
	LRC->A = HMMPI::Vector2<double>(DIM, con_count);
	LRC->b = std::vector<double>(con_count);
	for (size_t i = 0; i < con_count; i++)
	{
		LRC->b[i] = 0;
		for (size_t j = 0; j < DIM; j++)
			LRC->A(j, i) = auxA[i][j];
	}

	LRC->xcur = std::vector<double>(DIM);
	for (size_t i = 0; i < DIM; i++)
		LRC->xcur[i] = 0;

#ifdef WRITE_REGENTRYCONSTR
	std::ofstream sw(CWD + "/Constraints_A.txt");
	for (size_t j = 0; j < LRC->A.JCount(); j++)
	{
		for (size_t i = 0; i < LRC->A.ICount(); i++)
			if (i < LRC->A.ICount()-1)
				sw << HMMPI::stringFormatArr("{0}\t", std::vector<double>{LRC->A(i, j)});
			else
				sw << HMMPI::stringFormatArr("{0}\n", std::vector<double>{LRC->A(i, j)});
	}
	sw.close();
#endif
}
//------------------------------------------------------------------------------------------
LinRegress *RegEntryConstr::InitLR()
{
	return new LinRegressConstr(true);
}
//------------------------------------------------------------------------------------------
double RegEntryConstr::ScalProd(const std::vector<double> &a, const std::vector<double> &b)
{
	size_t len = a.size();
	double res = 0;

	for (size_t i = 0; i < len; i++)
		res += a[i]*b[i];

	return res;
}
//------------------------------------------------------------------------------------------
RegEntryConstr::RegEntryConstr(int rn) : RegEntry(rn)
{
	Cov = L = 0;
	cT = sz = 0;
}
//------------------------------------------------------------------------------------------
RegEntryConstr::~RegEntryConstr()
{
	if (Cov != 0)
		delete [] Cov;
	if (L != 0)
		delete [] L;
}
//------------------------------------------------------------------------------------------
void RegEntryConstr::LoadMatr(RegEntry *RE, double W5)
{
	if (regnum != RE->regnum)
		throw HMMPI::Exception("Не совпадают индексы регионов в RegEntryConstr::LoadMatr",
						"Region indices don't match in RegEntryConstr::LoadMatr");

//	c_Cov = dynamic_cast<RegEntryConstr*>(RE)->Cov;
//	c_L = dynamic_cast<RegEntryConstr*>(RE)->L;
//	sz = dynamic_cast<RegEntryConstr*>(RE)->sz;
//	cT = dynamic_cast<RegEntryConstr*>(RE)->cT;
	RLS = RE->RLS;
	RLS_ind = RE->RLS_ind;
	w5 = W5;
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
int RegList::RegInd(int n)
{
	size_t count = data.size();
	for (size_t i = 0; i < count; i++)
	{
		if (data[i]->regnum == n)
			return i;
	}

	return -1;
}
//------------------------------------------------------------------------------------------
void RegList::Add(int rn, double f, const std::vector<double> &xi, double a0, int i, int j)
{
	int ind = RegInd(rn);
	if (ind != -1)
		data[ind]->Add(f, xi, a0, i, j);
	else
	{
#ifdef REGRESS_WITH_CONSTR
		RegEntry *RE = new RegEntryConstr(rn);
#else
		RegEntry *RE = new RegEntry(rn);
#endif
		RE->Add(f, xi, a0, i, j);
		data.push_back(RE);
	}
}
//------------------------------------------------------------------------------------------
RegList::RegList()
{
	composR2 = avgR2 = f5 = VA = SSerr = 0;
	data = std::vector<RegEntry*>();
}
//------------------------------------------------------------------------------------------
RegList::~RegList()
{
	for (size_t i = 0; i < data.size(); i++)
		delete data[i];
}
//------------------------------------------------------------------------------------------
void RegList::ReadAllData(const std::vector<Grid2D> &dA, const HMMPI::Vector2<Grid2D> &dRES, const Grid2D &reg, const Grid2D &A0, const RegListSpat *RLS, double W5)
{
	size_t cT = dRES.ICount();
	size_t cM = dRES.JCount();
	if (cT != dA.size())
		throw HMMPI::EObjFunc(HMMPI::MessageRE("Разное число шагов в dA, dRES в RegList::ReadAllData",
								 "Different number of time steps in dA, dRES in RegList::ReadAllData"));

	Grid2D REG = reg;
	Grid2D scale0 = A0;
	int cX = REG.CountX();
	int cY = REG.CountY();

	for (size_t t = 0; t < cT; t++)	// outer loop since 10.11.2013
	{
		for (int i = 0; i < cX; i++)
		{
			for (int j = 0; j < cY; j++)
			{
				if (REG.flag[i][j])
				{
					int rn = int(REG.data[i][j]);

					double F = dA[t].data[i][j];
					double a0 = scale0.data[i][j];
					std::vector<double> XI(cM);
					for (size_t m = 0; m < cM; m++)
						XI[m] = dRES(t, m).data[i][j];

					Add(rn, F, XI, a0, i, j);
				}
			}
		}
	}

	size_t count = data.size();
	if (RLS->data.size() != count)
		throw HMMPI::EObjFunc(HMMPI::MessageRE("Недопустимое число регионов в RLS в RegList::ReadAllData",
								 "Inacceptable number of regions in RLS in RegList::ReadAllData"));

	for (size_t i = 0; i < count; i++)
	{
		data[i]->MakeArrays();

		// 24.11.2013 считывание данных из RLS
		data[i]->LoadMatr(RLS->data[i], W5);
		data[i]->RLS = RLS;
		data[i]->RLS_ind = (int)i;
	}

#ifdef WRITE_LINREGRESS_FILES
	std::string fn = RegEntry::CWD + "/regions_data.txt";
	std::ofstream sw;
	sw.open(fn);
	sw.close();
	for (size_t i = 0; i < count; i++)
		data[i]->WriteToFile(fn);
#endif
}
//------------------------------------------------------------------------------------------
std::vector<Grid2D> RegList::Regression(const std::vector<Grid2D> &dA, const HMMPI::Vector2<Grid2D> &dRES, const Grid2D &reg)
{
	size_t cT = dRES.ICount();
	size_t cM = dRES.JCount();
	if (cT != dA.size())
		throw HMMPI::EObjFunc(HMMPI::MessageRE("Разное число временных шагов в dA, dRES в RegList::Regression",
								  "Different number of time steps in dA, dRES in RegList::Regression"));

	Grid2D REG = reg;

	size_t count = data.size();
	double FuncVal = 0;			// sum of each region's obj. func.
	for (size_t i = 0; i < count; i++)
	{
		data[i]->Regression();
		FuncVal += data[i]->func_val;
	}

	std::vector<Grid2D> res(cM+1 + cT);

	int cX = REG.CountX();
	int cY = REG.CountY();

	for (size_t m = 0; m < cM+1 + cT; m++)
		res[m] = REG;

	double tot_pts = 0;
	avgR2 = 0;
	for (size_t n = 0; n < count; n++)
	{
		size_t cpts = data[n]->i.size();
		if (!HMMPI::IsNaN(data[n]->R2))
		{
			avgR2 += data[n]->R2 * cpts;
			tot_pts += cpts;
		}

		for (size_t k = 0; k < cpts; k++)
		{
			int i = data[n]->i[k];
			int j = data[n]->j[k];

			res[0].data[i][j] = data[n]->R2;
			if (REG.flag[i][j] == 0)
				res[0].flag[i][j] = 0;

			for (size_t m = 1; m < cM+1; m++)
			{
				if (data[n]->coeffs.size() != 0)
					res[m].data[i][j] = data[n]->coeffs[m-1];
				else
					res[m].data[i][j] = std::numeric_limits<double>::quiet_NaN();

				if (REG.flag[i][j] == 0)
					res[m].flag[i][j] = 0;
			}

			for (size_t t = 0; t < cT; t++)
			{
				if (data[n]->coeffs.size() != 0)
				{
					double d = 0;
					for (size_t m = 0; m < cM; m++)
					{
						if (!HMMPI::IsNaN(data[n]->coeffs[m]))
							d += data[n]->coeffs[m] * dRES(t, m).data[i][j];
					}
					res[t+cM+1].data[i][j] = d;
				}
				else
					res[t+cM+1].data[i][j] = std::numeric_limits<double>::quiet_NaN();

				if (REG.flag[i][j] == 0)
					res[t+cM+1].flag[i][j] = 0;
			}
		}
	}
	avgR2 /= tot_pts;

	// âû÷èñëÿåì ñîñòàâíîé R2
	double barf = 0;
	int count_f = 0;
	for (size_t t = 0; t < cT; t++)
	{
		Grid2D aux = dA[t];
		Grid2D calc = res[t+cM+1];
		for (int i = 0; i < cX; i++)
		{
			for (int j = 0; j < cY; j++)
			{
				if ((aux.flag[i][j])&&(!HMMPI::IsNaN(aux.data[i][j]))&&
					(calc.flag[i][j])&&(!HMMPI::IsNaN(calc.data[i][j])))
				{
					barf += aux.data[i][j];
					count_f += 1;
				}
			}
		}
	}
	barf /= count_f;

	double SStot = 0;
	SSerr = 0;
	for (size_t t = 0; t < cT; t++)
	{
		Grid2D aux = dA[t];
		Grid2D calc = res[t+cM+1];
		for (int i = 0; i < cX; i++)
		{
			for (int j = 0; j < cY; j++)
			{
				if ((aux.flag[i][j])&&(!HMMPI::IsNaN(aux.data[i][j]))&&
					(calc.flag[i][j])&&(!HMMPI::IsNaN(calc.data[i][j])))
				{
					SStot += (aux.data[i][j] - barf)*(aux.data[i][j] - barf);
					SSerr += (aux.data[i][j] - calc.data[i][j])*(aux.data[i][j] - calc.data[i][j]);
				}
			}
		}
	}
	composR2 = 1 - SSerr/SStot;
	VA = SStot;
	f5 = FuncVal;

	return res;
}
//------------------------------------------------------------------------------------------
std::string RegList::ReportLog()
{
	std::string res = "";
	size_t count = data.size();
	for (size_t i = 0; i < count; i++)
	{
		res += HMMPI::stringFormatArr("region {0:%d}\tmin constraint\tind = {1:%d}\t", std::vector<int>{data[i]->regnum, data[i]->minAxind});
		res += HMMPI::stringFormatArr("Ai*x = {0:%-10g}\ttol = {1}\t|Ai| = {2:%-10g}\t|x| = {3}\tx = ", std::vector<double>{data[i]->minAx, data[i]->TOLbeta, data[i]->normAi, data[i]->normxcur});
		for (size_t j = 0; j < data[i]->coeffs.size(); j++)
			res += HMMPI::stringFormatArr("\t{0}", std::vector<double>{data[i]->coeffs[j]});
		res += "\n";
	}

	return res;
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
RegListSpat::RegListSpat()
{
	countT = 0;
	countReg = 0;
	ownerCount = 0;
}
//------------------------------------------------------------------------------------------
int RegListSpat::VecSize(int r) const
{
	return dynamic_cast<RegEntryConstr*>(data[r])->sz;
}
//------------------------------------------------------------------------------------------
int RegListSpat::RegNum(int r) const
{
	return data[r]->regnum;
}
//------------------------------------------------------------------------------------------
void RegListSpat::ReadSpatial(const Grid2D &REG, const KW_variogram_Cs *var, int t_steps)
{
	int cX = REG.CountX();
	int cY = REG.CountY();

	for (int i = 0; i < cX; i++)
	{
		for (int j = 0; j < cY; j++)
		{
			if (REG.flag[i][j])
			{
				int rn = int(REG.data[i][j]);
				std::vector<double> XI(1);	// dummy std::vector
				Add(rn, 0, XI, 0, i, j);
			}
		}
	}

	size_t count = data.size();

	countT = t_steps;
	countReg = count;

	for (size_t r = 0; r < count; r++)
	{
		RegEntryConstr *REC = dynamic_cast<RegEntryConstr*>(data[r]);
		REC->MakeArrays();

		REC->cT = t_steps;
		REC->ref = REG;
		REC->sz = REC->i.size();
		size_t LEN = size_t(REC->sz)*size_t(REC->sz);
		REC->Cov = new double[LEN];
		for (size_t k = 0; k < LEN; k++)
			REC->Cov[k] = 0;

		double sill = 1, nugget = 0, Vchi = var->chi/180*PI;
		for (int p1 = 0; p1 < REC->sz; p1++)
		{
			for (int p2 = 0; p2 <= p1; p2++)
			{
				double val;
				double h = Grid2D::EllipseTransform(REC->i[p1] - REC->i[p2], REC->j[p1] - REC->j[p2], Vchi, var->R, var->r);

				if (var->type == "SPHER")
					val = Grid2D::VarSpher(h, 1, sill, nugget);
				else if (var->type == "EXP")
					val = Grid2D::VarExp(h, 1, sill, nugget);
				else if (var->type == "GAUSS")
					val = Grid2D::VarGauss(h, 1, sill, nugget);
				else
					throw HMMPI::EObjFunc(HMMPI::MessageRE("Некорректный тип вариограммы RegListSpat::ReadSpatial",
							 	    						"Incorrect variogram type in RegListSpat::ReadSpatial"));
				REC->Cov[p1*REC->sz + p2] = 1 - val;
				REC->Cov[p2*REC->sz + p1] = 1 - val;
			}
		}

		REC->L = new double[LEN];
		HMMPI::CholDecomp(REC->Cov, REC->L, REC->sz);

#ifdef WRITE_LINREGRESS_FILES
		std::ofstream fileC, fileL;
		fileC.open(HMMPI::stringFormatArr(RegEntry::CWD + "/correl_Cov_{0:%d}.txt", std::vector<int>{(int)r}));
		fileL.open(HMMPI::stringFormatArr(RegEntry::CWD + "/correl_L_{0:%d}.txt", std::vector<int>{(int)r}));
		fileC << "Covariance matrix" << std::endl;
		fileL << "Choleski decomposition" << std::endl;
		for (int i = 0; i < REC->sz; i++)
		{
			for (int j = 0; j < REC->sz; j++)
			{
				if (j < REC->sz-1)
				{
					fileC << REC->Cov[i*REC->sz+j] << "\t";
					fileL << REC->L[i*REC->sz+j] << "\t";
				}
				else
				{
					fileC << REC->Cov[i*REC->sz+j] << "\n";
					fileL << REC->L[i*REC->sz+j] << "\n";
				}
			}
		}
		fileC.close();
		fileL.close();
#endif

	}
}
//------------------------------------------------------------------------------------------
void RegListSpat::Linv_v(const std::vector<double> &v, std::vector<double> &dest, int r) const
{
	RegEntryConstr *REC = dynamic_cast<RegEntryConstr*>(data[r]);

	size_t SZ = REC->sz; 	// size of Cov and L
	if (v.size() != dest.size() || v.size() != SZ)
		throw HMMPI::EObjFunc(HMMPI::MessageRE("Не совпадают размеры векторов/матриц в RegListSpat::Linv_v",
												"Vector/matrix sizes don't match in RegListSpat::Linv_v"));

	for (size_t i = 0; i < SZ; i++)
	{
		double sum = 0;
		for (size_t j = 0; j < i; j++)
			sum += REC->L[i*SZ+j] * dest[j];
		dest[i] = (v[i] - sum) / REC->L[i*SZ+i];
	}
}
//------------------------------------------------------------------------------------------
void RegListSpat::L_v(const std::vector<double> &v, std::vector<double> &dest, int r) const
{
	RegEntryConstr *REC = dynamic_cast<RegEntryConstr*>(data[r]);

	size_t SZ = REC->sz; 	// size of Cov and L
	if (v.size() != dest.size() || v.size() != SZ)
		throw HMMPI::EObjFunc(HMMPI::MessageRE("Не совпадают размеры векторов/матриц в RegListSpat::L_v",
												"Vector/matrix sizes don't match in RegListSpat::L_v"));

	for (size_t i = 0; i < SZ; i++)
	{
		double sum = 0;
		for (size_t j = 0; j <= i; j++)
			sum += REC->L[i*SZ+j] * v[j];
		dest[i] = sum;
	}
}
//------------------------------------------------------------------------------------------
void RegListSpat::vec2grid(const std::vector<double> &v, Grid2D &GR, int r) const
{
	RegEntryConstr *REC = dynamic_cast<RegEntryConstr*>(data[r]);

	size_t SZ = REC->sz; 	// size of Cov and L
	size_t cX = GR.CountX();
	size_t cY = GR.CountY();
	if (v.size() != SZ || (int)cX != REC->ref.CountX() || (int)cY != REC->ref.CountY())
		throw HMMPI::EObjFunc(HMMPI::MessageRE("Не совпадают размеры векторов/гридов в RegListSpat::vec2grid",
												"Vector/grid sizes don't match in RegListSpat::vec2grid"));

	int c = 0;
	for (size_t i = 0; i < cX; i++)
		for (size_t j = 0; j < cY; j++)
		{
			if (REC->ref.flag[i][j] && (int)REC->ref.data[i][j] == REC->regnum)
			{
				if (c >= (int)SZ)
					throw HMMPI::EObjFunc(HMMPI::stringFormatArr(HMMPI::MessageRE(
							"Регион грида {0:%d} имеет больше ячеек, чем имеется значений в векторе ({1:%d}) в RegListSpat::vec2grid",
							"Grid region {0:%d} has more cells than there are values in the vector ({1:%d}) in RegListSpat::vec2grid"),
							std::vector<int>{REC->regnum, (int)SZ}));
				GR.data[i][j] = v[c];
				c++;
			}
			if (!REC->ref.flag[i][j])
				GR.flag[i][j] = 0;
		}

	if (c < (int)SZ)
		throw HMMPI::EObjFunc(HMMPI::stringFormatArr(HMMPI::MessageRE(
				"Регион грида {0:%d} имеет меньше ячеек ({1:%d}), чем имеется значений в векторе ({2:%d}) в RegListSpat::vec2grid",
				"Grid region {0:%d} has less cells ({1:%d}) than there are values in the vector ({2:%d}) in RegListSpat::vec2grid"),
				std::vector<int>{REC->regnum, c, (int)SZ}));

}
//------------------------------------------------------------------------------------------
void RegListSpat::Linv_P(const HMMPI::Vector2<double> &P, HMMPI::Vector2<double> &Dest, int r) const
{
	RegEntryConstr *REC = dynamic_cast<RegEntryConstr*>(data[r]);

	size_t SZ = REC->sz; 	// size of Cov and L
	size_t count = P.JCount();
	if (P.ICount() != SZ || P.ICount() != Dest.ICount() || count != Dest.JCount())
		throw HMMPI::EObjFunc(HMMPI::MessageRE("Не совпадают размеры матриц в RegListSpat::Linv_P",
												"Matrix sizes don't match in RegListSpat::Linv_P"));

	for (size_t k = 0; k < count; k++)
		for (size_t i = 0; i < SZ; i++)
		{
			double sum = 0;
			for (size_t j = 0; j < i; j++)
				sum += REC->L[i*SZ+j] * Dest(j, k);
			Dest(i, k) = (P(i, k) - sum) / REC->L[i*SZ+i];
		}
}
//------------------------------------------------------------------------------------------
const RegEntryConstr *RegListSpat::GetReg(int r) const
{
	return const_cast<const RegEntryConstr*>(dynamic_cast<RegEntryConstr*>(data[r]));
}
//------------------------------------------------------------------------------------------
// VectCorrEntry
//---------------------------------------------------------------------------
VectCorrEntry::VectCorrEntry()
{
	sz = 0;
	C = L = 0;
	R0 = 0;
}
//---------------------------------------------------------------------------
VectCorrEntry::~VectCorrEntry()
{
	if (C != 0)
		delete [] C;
	if (L != 0)
		delete [] L;
}
//---------------------------------------------------------------------------
void VectCorrEntry::FillData(size_t ind, const HMMPI::Vector2<double> &textsmry, const std::vector<double> &tm, double R, HMMPI::Func1D_corr *func)
{
	R0 = R;

	size_t dcount = tm.size();
	if (dcount != textsmry.ICount())
		throw HMMPI::Exception("Длина списка дат и соответствующий размер textsmry не совпадают в VectCorrEntry::FillData",
						"Dates array length and the corresponding dimension of textsmry do not match in VectCorrEntry::FillData");

	size_t vcount = textsmry.JCount()/2;
	if (ind >= vcount)
		throw HMMPI::Exception("Индекс ind вне допустимого диапазона в VectCorrEntry::FillData",
						"Index ind out of range in VectCorrEntry::FillData");

	// count valid data points
	sz = 0;
	std::vector<double> tvalid;
	sigma = std::vector<double>();
	indvalid = std::vector<int>(dcount);
	for (size_t i = 0; i < dcount; i++)
	{
		indvalid[i] = 0;
		if (!HMMPI::IsNaN(textsmry(i, ind)) && textsmry(i, ind + vcount) != 0)
		{
			indvalid[i] = 1;
			tvalid.push_back(tm[i]);
			sigma.push_back(textsmry(i, ind + vcount));
			sz++;
		}
	}

	if (sz != 0)
	{
		size_t count = sz * sz;
		delete [] C;
		delete [] L;
		if (R0 <= R_threshold)
		{
			C = L = nullptr;			// diagonal covariance
		}
		else
		{
			C = new double[count];
			L = new double[count];
			for (size_t i = 0; i < sz; i++)
				for (size_t j = 0; j <= i; j++)
				{
					double dt = fabs(tvalid[i] - tvalid[j]) / R;
					C[i*sz + j] = func->f(dt);
					C[j*sz + i] = C[i*sz + j];
				}

			HMMPI::CholDecomp(C, L, sz);

#ifdef WRITE_WELLCOVAR_FILES
			int rank;
			MPI_Comm_rank(MPI_COMM_WORLD, &rank);
			if (rank == 0)
			{
				std::ofstream fileC, fileL;
				fileC.open(HMMPI::stringFormatArr("correl_Cov_vec{0:%d}.txt", std::vector<int>{(int)ind}));	// 12.04.2017 removed RegEntry::CWD
				fileL.open(HMMPI::stringFormatArr("correl_L_vec{0:%d}.txt", std::vector<int>{(int)ind}));
				fileC << "Correlation matrix" << std::endl;
				fileL << "Choleski decomposition" << std::endl;
				for (size_t i = 0; i < sz; i++)
				{
					for (size_t j = 0; j < sz; j++)
					{
						if (j < sz-1)
						{
							fileC << C[i*sz + j] << "\t";
							fileL << L[i*sz + j] << "\t";
						}
						else
						{
							fileC << C[i*sz + j] << "\n";
							fileL << L[i*sz + j] << "\n";
						}
					}
				}
				fileC.close();
				fileL.close();
			}
#endif
		}
	}
}
//---------------------------------------------------------------------------
void VectCorrEntry::Linv_v(const std::vector<double> &v, std::vector<double> &dest) const
{
	if (v.size() != sz || dest.size() != sz)
		throw HMMPI::EObjFunc("Неверные размеры векторов в VectCorrEntry::Linv_v", "Incorrect vector sizes in VectCorrEntry::Linv_v");

	if (L == 0)
		dest = v;
	else
	{
		for (size_t i = 0; i < sz; i++)
		{
			double sum = 0;
			for (size_t j = 0; j < i; j++)
				sum += L[i*sz + j] * dest[j];

			dest[i] = (v[i] - sum)/L[i*sz + i];
		}
	}
}
//---------------------------------------------------------------------------
void VectCorrEntry::L_v(const std::vector<double> &v, std::vector<double> &dest) const
{
	if (v.size() != sz || dest.size() != sz)
		throw HMMPI::Exception("Неверные размеры векторов в VectCorrEntry::L_v", "Incorrect vector sizes in VectCorrEntry::L_v");

	if (L == 0)
		dest = v;
	else
	{
		for (size_t i = 0; i < sz; i++)
		{
			double sum = 0;
			for (size_t j = 0; j <= i; j++)
				sum += L[i*sz + j] * v[j];

			dest[i] = sum;
		}
	}
}
//---------------------------------------------------------------------------
double VectCorrEntry::Perturbation(std::vector<double> &dest, HMMPI::RandNormal *rn, double sigma1) const
{
	if (dest.size() != sz)
		throw HMMPI::Exception("Неверный размер вектора в VectCorrEntry::Perturbation", "Incorrect vector size in VectCorrEntry::Perturbation");

	std::vector<double> v(sz);
	for (size_t i = 0; i < sz; i++)
		v[i] = rn->get();				// v = xi ~ N(0,I)

	L_v(v, dest);						// dest ~ N(0, L*L')
	for (size_t i = 0; i < sz; i++)
		dest[i] *= (sigma[i] * sigma1);	// dest ~ N(0, sigma1*S*L*L'*S'*sigma1)

	double sum = 0;
	std::vector<double> w(sz);
	for (size_t i = 0; i < sz; i++)
		w[i] = dest[i] / (sigma[i] * sigma1);
	Linv_v(w, v);
	for (size_t i = 0; i < sz; i++)
		sum += v[i]*v[i];

	return sum;
}
//---------------------------------------------------------------------------
void VectCorrEntry::v2vector(const std::vector<double> &v, HMMPI::Vector2<double> &smry, size_t ind)
{
	size_t vcount = smry.JCount()/2;
	size_t dcount = smry.ICount();
	if (v.size() != sz || dcount != indvalid.size())
		throw HMMPI::Exception("Неправильные размеры векторов в VectCorrEntry::v2vector",
						"Inconsistent vector sizes in VectCorrEntry::v2vector");
	if (ind >= vcount)
		throw HMMPI::Exception("Индекс ind вне допустимого диапазона в VectCorrEntry::v2vector",
						"Index ind out of range in VectCorrEntry::v2vector");

	int c = 0;
	for (size_t i = 0; i < dcount; i++)
		if (indvalid[i])
		{
			smry(i, ind) = v[c];
			c++;
		}
}
//---------------------------------------------------------------------------
void VectCorrEntry::v2vector_add(const std::vector<double> &v, HMMPI::Vector2<double> &smry, size_t ind) const
{
	size_t vcount = smry.JCount()/2;
	size_t dcount = smry.ICount();
	if (v.size() != sz || dcount != indvalid.size())
		throw HMMPI::Exception("Неправильные размеры векторов в VectCorrEntry::v2vector_add",
						"Inconsistent vector sizes in VectCorrEntry::v2vector_add");
	if (ind >= vcount)
		throw HMMPI::Exception("Индекс ind вне допустимого диапазона в VectCorrEntry::v2vector_add",
						"Index ind out of range in VectCorrEntry::v2vector_add");

	int c = 0;
	for (size_t i = 0; i < dcount; i++)
		if (indvalid[i])
		{
			smry(i, ind) += v[c];	// разница с VectCorrEntry::v2vector только в этой строке
			c++;
		}
}
//---------------------------------------------------------------------------
double VectCorrEntry::ObjFunc(const HMMPI::Vector2<double> &smryMod, const HMMPI::Vector2<double> &smryHist, size_t ind, const HMMPI::Vector2<double> *smrySens)
{
	size_t dcount = indvalid.size();
	if (dcount != smryMod.ICount() || dcount != smryHist.ICount() || (smrySens != 0 && dcount != smrySens->ICount()))
		throw HMMPI::EObjFunc("Неправильные размеры массивов в VectCorrEntry::ObjFunc",
							  "Inconsistent array sizes in VectCorrEntry::ObjFunc");

	std::vector<double> e(sz), e1(sz);
	std::vector<double> u(sz);
	size_t c = 0;
	for (size_t i = 0; i < dcount; i++)
		if (indvalid[i])
		{
			e[c] = (smryMod(i, ind) - smryHist(i, ind))/sigma[c];
			if (smrySens != 0)
				e1[c] = (*smrySens)(i, ind)/sigma[c];
			c++;
		}
	Linv_v(e, u);

	std::vector<double> u1 = u;
	if (smrySens != 0)
		Linv_v(e1, u1);

	double sum = 0;
	for (size_t i = 0; i < sz; i++)
		sum += u[i]*u1[i];

	return sum;
}
//---------------------------------------------------------------------------
// VectCorrList
//---------------------------------------------------------------------------
VectCorrList::VectCorrList()
{
	ownerCount = 0;
}
//---------------------------------------------------------------------------
void VectCorrList::LoadData(const HMMPI::Vector2<double> &textsmry, const std::vector<double> &tm, const std::vector<double> &R, std::vector<HMMPI::Func1D_corr*> F)
{
	size_t vcount = textsmry.JCount()/2;
	if (vcount != R.size())
		throw HMMPI::Exception("Размер вектора рангов корреляций R не совпадает с числом векторов в VectCorrList::LoadData",
						"The size of the correlation range vector R does not match the number of vectors in VectCorrList::LoadData");
	if (vcount != F.size())
		throw HMMPI::Exception("Size of array 'F' and number of vectors do not match in VectCorrList::LoadData");

	data = std::vector<VectCorrEntry>(vcount);

	for (size_t i = 0; i < vcount; i++)
		data[i].FillData(i, textsmry, tm, R[i], F[i]);
}
//---------------------------------------------------------------------------
double VectCorrList::PerturbData(HMMPI::Vector2<double> &smry, HMMPI::RandNormal *rn, double sigma1)
{
	size_t vcount = smry.JCount()/2;
	if (vcount != data.size())
		throw HMMPI::Exception("Не совпадают размерности массивов в VectCorrList::PerturbData", "Array sizes do not match in VectCorrList::PerturbData");

	double sum = 0;
	for (size_t i = 0; i < vcount; i++)
	{
		std::vector<double> noise(data[i].sz);
		sum += data[i].Perturbation(noise, rn, sigma1);
		data[i].v2vector_add(noise, smry, i);
	}

	return sum;
}
//---------------------------------------------------------------------------
double VectCorrList::ObjFunc(const HMMPI::Vector2<double> &smryMod, const HMMPI::Vector2<double> &smryHist, bool &cov_is_diag, const HMMPI::Vector2<double> *smrySens)
{
	size_t vcount = data.size();
	if (vcount != smryMod.JCount() || vcount != smryHist.JCount()/2 || (smrySens != 0 && vcount != smrySens->JCount()))
		throw HMMPI::EObjFunc("Не совпадают размерности массивов в VectCorrList::ObjFunc", "Array sizes do not match in VectCorrList::ObjFunc");

	of1 = std::vector<double>(vcount);
	double sum = 0;
	cov_is_diag = true;
	for (size_t i = 0; i < vcount; i++)
	{
		of1[i] = data[i].ObjFunc(smryMod, smryHist, i, smrySens);
		sum += of1[i];

		if (data[i].R0 > data[i].R_threshold)
			cov_is_diag = false;
	}

	return sum;
}
//---------------------------------------------------------------------------
int VectCorrList::countUndef()
{
	size_t vcount = data.size();
	int sum = 0;
	for (size_t i = 0; i < vcount; i++)
		sum += data[i].indvalid.size() - data[i].sz;

	return sum;
}
//---------------------------------------------------------------------------

