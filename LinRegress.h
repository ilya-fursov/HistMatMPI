/*
 * LinRegress.h
 *
 *  Created on: Mar 28, 2013
 *      Author: ilya
 */

#ifndef LINREGRESS_H_
#define LINREGRESS_H_

#include "Abstract.h"
#include "Vectors.h"
#include "Parsing.h"
#include "Tracking.h"
#include "MathUtils.h"
#include "MonteCarlo.h"
#include <vector>
#include <string>

class KW_regressConstr;
class Grid2D;
class KW_variogram_Cs;

class RegListSpat;
//------------------------------------------------------------------------------------------
class LinRegress
{
private:
	static double eps;	// åñëè |Xi|/max|Xj| < eps, ñòîëáåö i áóäåò óäàëåí, ñîîòâ. êîýôô. ðåãðåññèè áóäåò = NaN
protected:
	HMMPI::Vector2<double> M;
	std::vector<double> RHS;
	std::vector<double> sol;
	std::vector<int> act_var;		// íàïð. åñëè èñõîäíàÿ Xi èìåëà 4 ñòîëáöà, è 3-é ñòîëáåö áûë óäàëåí, act_var = {0, 1, -1, 2}
	std::vector<double> normCoeff;

	size_t countIV;
	size_t countPTS;

	std::vector<double> Norm2();		// åâêëèäîâà íîðìà âñåõ ñòîëáöîâ |Xi|
	void UpdateXi();			// íîðìèðóåò Xi, èñêëþ÷àåò ñòîëáöû Xi ñ î÷. ìàëåíüêîé îòíîñèòåëüíîé íîðìîé, çàïîëíÿåò act_var
	void UpdateRes();			// âñòàâëÿåò NaN ãäå íåîáõîäèìî, äåëàåò ïåðåíîðìèðîâêó, ðåçóëüòàò - â sol
	double ScalProdXX(int i, int j);
	double ScalProdXf(int i);
	void Swap(int cur);
	virtual void FillMatr();
	void CalcR2();
	void SolveGauss();

	//const double *c_Cov;	// sz x sz
	//const double *c_L;
	const RegListSpat *RLS;
	//int cT, sz;
	int RLS_ind;
	double w5;
public:
	std::vector<double> f;
	HMMPI::Vector2<double> Xi;
	std::vector<double> A0;
	double R2;
	double avgA0;

	const double TOLbeta;		// small negative
	double TOLcur;
	double minAx;
	int minAxind;
	double normAi;
	double normxcur;
	double func_val;			// final (quadratic) obj. func.

	LinRegress();
	virtual ~LinRegress(){};
	virtual std::vector<double> Solve0();
	void CalcAvgA0();
	void LoadMatr(const RegListSpat *rls, int rls_ind, double W5);
};
//------------------------------------------------------------------------------------------
class LinRegressConstr : public LinRegress
{
private:
	static std::vector<double> vect_Vi(const HMMPI::Vector2<double> &A, int col, double &alpha);	// (eng) std::vector Vi for QRdecomp, from given column (and row = colum)
	static void updateM(HMMPI::Vector2<double> &M, int col, double alpha, const std::vector<double> &v);	// M = QM

protected:
	bool make_regress;		// (eng) if false, only does QP from G, g, A, b, xcur; if true, does regression from Xi, f, A, b, xcur

	HMMPI::Vector2<double> actA;
	HMMPI::Vector2<double> Q1t;
	HMMPI::Vector2<double> Q2t;
	HMMPI::Vector2<double> R;

	std::vector<double> lambdacur;
	std::vector<double> gcur;
	std::vector<int> actind;		// (0, 1, 1, 0,...)
	int qr_decomp_flag;		// 0, 1

	void CalcGg();			// (eng) calc G, g from Xi, f, after UpdateXi!
	void UpdateAxcur();		// (eng) remove excessive rows accord. to act_var
	void actA_fromX1();		// (eng) calc actA from initial point xcur = x0
	void actA_fromind();	// actA = A(actind)
	void qrDecomp();		// actA = [Q1, Q2]*R
	double Ai_x(int i, const std::vector<double> &x);	// <Ai, x>, Ai - ith column of A (eng)
	virtual void FillMatr();				// Z'GZy = -Z'g_k, where Z = Q2 (eng)
	double calcBeta(const std::vector<double> &sk, int &p);	// (eng) p - index of constraint to be added, -1 if no constr. added
	int minLambda();		// (eng) returns active constraint index to be removed, -1 if all lambda_i >= 0 for active constraints
	void minAxcur(const std::vector<double> &xcur, int &ind, double &min);
	double NormAi(int i);
public:
	static KW_regressConstr *regcon;

	int c_Lagr, c_EP;		// (eng) counts calculations of Lagrangian multipliers, solutions of equality problem
	HMMPI::Vector2<double> G;
	std::vector<double> g;
	double f2;				// f'*Cs*f for obj func
	HMMPI::Vector2<double> A;		// A'x >= b
	std::vector<double> b;
	std::vector<double> xcur;

	static std::vector<HMMPI::Vector2<double>> QRdecomp(const HMMPI::Vector2<double> &A);	// res = {Q, R}, QR = A
	static int Rank(const HMMPI::Vector2<double> &A);
	static std::vector<double> MultMatrVect(const HMMPI::Vector2<double> &M, const std::vector<double> &v);	// res = Mv
	static std::vector<double> MultMatrTVect(const HMMPI::Vector2<double> &M, const std::vector<double> &v);	// res = M'v
	static std::vector<double> Rinv(const HMMPI::Vector2<double> &Rmatr, const std::vector<double> &v);	// res = Rmatr\v, Rmatr - upper triangular (eng)
	LinRegressConstr(bool make_reg);
	virtual std::vector<double> Solve0();	// (eng) sol is returned
	double FuncVal();		// xcur'Gxcur + 2g'xcur + f2
	static double Norm(const std::vector<double> &v);
};
//------------------------------------------------------------------------------------------
class RegEntry
{
private:
	std::vector<double> F;
	std::vector<std::vector<double>> Xi;	// äëèíà ìàññèâîâ = countIV
	std::vector<double> SC0;
	std::vector<int> I;
	std::vector<int> J;
protected:
	std::vector<double> f;
	HMMPI::Vector2<double> xi;
	std::vector<double> sc0;

	//const double *c_Cov;		// будут использованы в регрессии
	//const double *c_L;
	double w5;
	virtual void AddConstraints(LinRegress *lr);	// (eng) empty
	virtual LinRegress *InitLR();
public:
	const RegListSpat *RLS;
	int RLS_ind;

	int regnum;
	std::vector<int> i;
	std::vector<int> j;

	std::vector<double> coeffs;		// ðåçóëüòàòû ðåãðåññèè
	double R2;

	static std::string CWD;
	double TOLbeta;
	double minAx;
	int minAxind;
	double normAi;
	double normxcur;
	double func_val;			// final (quadratic) obj. func.

	RegEntry(int rn);
	virtual ~RegEntry(){};
	void Add(double f, const std::vector<double> &xi, double a0, int i, int j);
	virtual void LoadMatr(RegEntry *RE, double W5){};	// все действия - начиная с RegEntryConstr
	void MakeArrays();
	void Regression();
	void WriteToFile(std::string fn);
};
//------------------------------------------------------------------------------------------
class RegEntryConstr : public RegEntry
{
protected:
	virtual void AddConstraints(LinRegress *lr);	// (eng) define xcur, A, b
	virtual LinRegress *InitLR();
public:
	double *Cov;	// sz x sz матрица, индексация: [i][j] <-> [i*sz + j] - i.e. row major
	double *L;		// Cholesky decomp. | Разложение Холеского
	int sz;
	int cT;			// time steps
	Grid2D ref;		// for reference

	static KW_regressConstr *regcon;

	static double ScalProd(const std::vector<double> &a, const std::vector<double> &b);
	RegEntryConstr(int rn);
	~RegEntryConstr();
	virtual void LoadMatr(RegEntry *RE, double W5);
};
//------------------------------------------------------------------------------------------
class RegListSpat;
class RegList
{
protected:
	std::vector<RegEntry*> data;

	int RegInd(int n);		// âîçâðàùàåò èíäåêñ (â data) ðåãèîíà "n", -1 åñëè ðåãèîí íå íàéäåí
	void Add(int rn, double f, const std::vector<double> &xi, double a0, int i, int j);	// äîáàâëÿåò äàííûå èç íîâîé òî÷êè ãðèäà
public:
	double avgR2;			// ðåçóëüòàò
	double composR2;
	double VA;				// SStot - no covariance
	double SSerr;			// SSerr - no covariance
	double f5;				// covariance!

	RegList();
	~RegList();
	void ReadAllData(const std::vector<Grid2D> &dA, const HMMPI::Vector2<Grid2D> &dRES, const Grid2D &reg, const Grid2D &A0, const RegListSpat *RLS, double W5);
							// dA : {cT}, dRES : {cT}x{cM}, cT - кол-во врем. шагов, cM - êîë-âî êàðò (countIV)
							// RLS идет из PhysModelHM
	std::vector<Grid2D> Regression(const std::vector<Grid2D> &dA, const HMMPI::Vector2<Grid2D> &dRES, const Grid2D &reg);
							// return = {R2, a, b, c,.., calcA1, calcA2,...}
							// ò.å. R2, êàðòû êîýôôèöèåíòîâ, ðàñ÷èòàííûå êàðòû àòðèáóòîâ (ïî âñåì âðåì. øàãàì)
	std::string ReportLog();
};
//------------------------------------------------------------------------------------------
class RegListSpat : public RegList
{
protected:
	int countT;
	int countReg;
public:
	int ownerCount;	// показывает, сколько моделей владеют этим ресурсом

	RegListSpat();
	int CountT() const {return countT;};
	int CountReg() const {return countReg;};
	int VecSize(int r) const;					// data[r]->sz
	int RegNum(int r) const;					// data[r]->regnum
	void ReadSpatial(const Grid2D &REG, const KW_variogram_Cs *var, int t_steps);	// makes correlation matrix
					// перед вызовом надо синхронизировать акт. ячейки во всевозможных гридах
	void Linv_v(const std::vector<double> &v, std::vector<double> &dest, int r) const;		// dest = Lr\v, should have appropriate length
	void L_v(const std::vector<double> &v, std::vector<double> &dest, int r) const;			// dest = Lr*v, should have appropriate length
	void vec2grid(const std::vector<double> &v, Grid2D &GR, int r) const;				// v -> GR for region r
	void Linv_P(const HMMPI::Vector2<double> &P, HMMPI::Vector2<double> &Dest, int r) const;	// Dest = Lr\P, should have appropriate size
	const RegEntryConstr *GetReg(int r) const;
};
//---------------------------------------------------------------------------
// correlations for eclipse well vectors
//---------------------------------------------------------------------------
class VectCorrEntry		// covariance for a single Eclipse vector
{
protected:
	void FillConstCorr(double *C) const;	// fills the 'CONST' covariance C of size sz*sz, using the big eigenvalue 'R0' and eigenvector 1/sigma
											// i.e. C - the covariance that damps the const vectors
public:
	const double R_threshold = 0.01;		// when R0 <= R_threshold, correlation matrix will not be calculated, but "unity operator" will be used instead
	size_t sz;		// total defined historic vals with non-zero sigma in this well vector
	double *C;		// correlation from variogram, sz x sz, full covariance is {sigma}'*C*{sigma}; C = nullptr if R0 <= R_threshold, which means "using unity C"
	double *L;		// Choleski(C), sz x sz; L = nullptr if R0 <= R_threshold
	std::vector<double> sigma;	// sz
	std::vector<int> indvalid;	// 0 for unused data points, 1 for used data points
	double R0;		// correlation radius - stored for reference

	VectCorrEntry();
	~VectCorrEntry();
	void FillData(size_t ind, const HMMPI::Vector2<double> &textsmry, const std::vector<double> &tm, double R, const HMMPI::Func1D_corr *func);	// ind - vector index in ECLVECTORS, tm - zero-based time (days), R - variogram range in days, func - 1D corr. function
					// if func == CorrDummyConst, then R is the 'big number' used for the correlation matrix eigenvalue
					// sz, C, L, sigma, indvalid, R0 are filled
	void Linv_v(const std::vector<double> &v, std::vector<double> &dest) const;
	void L_v(const std::vector<double> &v, std::vector<double> &dest) const;
	double Perturbation(std::vector<double> &dest, HMMPI::RandNormal *rn, double sigma1) const;			// dest = sigma1*S*L*xi, returns (Linv*Sinv/sigma1*dest)^2 for Chi-2
	void v2vector(const std::vector<double> &v, HMMPI::Vector2<double> &smry, size_t ind);				// вектор #ind в smry принимает значения из v, нулевые сигмы пропускаются
	void v2vector_add(const std::vector<double> &v, HMMPI::Vector2<double> &smry, size_t ind) const;	// к вектору #ind в smry прибавляются значения из v, нулевые сигмы пропускаются
	double ObjFunc(const HMMPI::Vector2<double> &smryMod, const HMMPI::Vector2<double> &smryHist, size_t ind, const HMMPI::Vector2<double> *smrySens = 0);		// (m-h)'*Cinv*(m-h), or if smrySens != 0, (m-h)'*Cinv*Sens
};
//---------------------------------------------------------------------------
class VectCorrList
{
protected:
	std::vector<VectCorrEntry> data;
public:
	int ownerCount;
	std::vector<double> of1;				// objective function for each vector

	VectCorrList();
	void LoadData(const HMMPI::Vector2<double> &textsmry, const std::vector<double> &tm, const std::vector<double> &R, std::vector<const HMMPI::Func1D_corr*> F);
	double PerturbData(HMMPI::Vector2<double> &smry, HMMPI::RandNormal *rn, double sigma1);	// возмущения добавляются к истории smry, здесь надо использовать textsmry->pet_dat
																				// возвращает noise'*C*noise для критерия хи-2
																				// sigma1 - сомножитель для сигм, идущий из w1
	double ObjFunc(const HMMPI::Vector2<double> &smryMod, const HMMPI::Vector2<double> &smryHist, bool &cov_is_diag, const HMMPI::Vector2<double> *smrySens = 0);	// (m-h)'*Cinv*(m-h), or if smrySens != 0, (m-h)'*Cinv*Sens
																																									// also updates "cov_is_diag" ('true' if covariance is fully diagonal)
	int countUndef();
	const std::vector<VectCorrEntry> &vec_vce(){return data;};
};
//------------------------------------------------------------------------------------------

#endif /* LINREGRESS_H_ */
