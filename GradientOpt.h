/*
 * GradientOpt.h
 *
 *  Created on: 29.01.2014
 *      Author: FursovIV
 */

#ifndef GRADIENTOPT_H_
#define GRADIENTOPT_H_

#include "Abstract.h"
#include "Utils.h"
#include "PhysModels.h"
#include "alglib-3.10.0_cpp/optimization.h"
#include "gsl/gsl_multiroots.h"
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_dense.h>

namespace HMMPI
{
	class Mat;
}
//---------------------------------------------------------------------------
// GradPhysMod
//---------------------------------------------------------------------------
// abstract descendant of PhysModel which can be used in quasi-Newton optimization
class GradPhysMod : public PhysModel
{
protected:
	std::vector<double> x_as;
public:
	double f_0, f_1;
	std::vector<double> grad_0;
	std::vector<double> grad_1;

	std::vector<double> x;
	std::vector<double> s;

	virtual void GradObjFunc(int n) = 0;	// grad_n = g(x + as), n = 0, 1
	virtual void UpdateMu(double &mu) = 0;
	virtual void CalcX_aS(double a);
	virtual double CalcBeta();
	virtual double Calc_df0();				// (grad_0' * s)
	virtual double Calc_df1();				// (grad_1' * s)
	virtual double Calc_grad_df1();			// по умолчанию выч. grad_1, потом (grad_1' * s); при необходимости переделать на более эффективную
	virtual std::vector<double> LinComb(double a, const std::vector<double> &U, double b, const std::vector<double> &V);	// возвращает aU + bV
	virtual std::string StringX();				// x_as -> std::string
	std::vector<double> GetX_aS(){return x_as;};
};
//---------------------------------------------------------------------------
// NonlinearSystemSolver - abstract base class for solving non-linear systems of equations F(x) = 0
// _NOTE_ number of equations == number of unknowns
// Before using the solver, "Func" [and "Jac"] should be set up
// "Func" and "Jac" input/output should be sync on MPI_COMM_WORLD
//---------------------------------------------------------------------------
class NonlinearSystemSolver
{
protected:
	double eps;																	// tolerance of residual for stopping the iteration (residual may be in 1-norm, 2-norm, inf-norm, depending on solver)
	int maxit;																	// maximum iterations
	const double save_eps;														// keeps hard value of 'eps', used in rescaling

public:
	std::function<std::vector<double>(const std::vector<double>&)> Func;		// equation Func(x) = 0 is to be solved; Func() input and output should be sync on MPI_COMM_WORLD
	int iter;																	// iterations count, filled by Solve()
	std::string exc_msg;														// this message is added to the exception message
	bool flag;																	// flag for controlling e.g. debug_check_jac()

	NonlinearSystemSolver(double eps_, int maxit_) : eps(eps_), maxit(maxit_), save_eps(eps_), iter(0), exc_msg(""), flag(false) {};
	virtual void SetFuncFromVM(const VectorModel *vm);							// takes Func_ACT [, Jac_ACT] from "vm"; NOTE: "vm" object itself will be used in these function calls, so check it's properly set up;
	virtual ~NonlinearSystemSolver() {};
	virtual std::vector<double> Solve(std::vector<double> x) = 0;				// 'x' is the starting point; 'iter' gets filled; input and output should be sync on MPI_COMM_WORLD; Func & Jac are erased in the end
	virtual std::string msg(const std::vector<double> &x) const;				// message about iterations, etc; 'x' should be the found solution
	virtual void debug_check_jac(const gsl_vector *x) const {};					// prints analytical (if present) and fin-diff Jacobians
	void rescale_eps(double factor) {eps = save_eps*factor;};
	double atol() const {return save_eps;};
};
//---------------------------------------------------------------------------
// FixedPointIter - class for performing fixed-point iteration
//---------------------------------------------------------------------------
class FixedPointIter : public NonlinearSystemSolver
{
public:
	FixedPointIter(double eps_, int maxit_) : NonlinearSystemSolver(eps_, maxit_) {};
	virtual std::vector<double> Solve(std::vector<double> x);					// 1-norm is used as stopping criterion
};
//---------------------------------------------------------------------------
// NewtonIter - class for Newton solvers
//---------------------------------------------------------------------------
class NewtonIter : public NonlinearSystemSolver
{
private:
	static int func(const gsl_vector *x, void *params, gsl_vector *f);			// these three functions are essentially the wrappers; pass 'params' = this
	static int df(const gsl_vector *x, void *params, gsl_matrix *jac);
	static int fdf(const gsl_vector *x, void *params, gsl_vector *f, gsl_matrix *jac);

protected:
	std::string Type;															// NEWTON, GNEWTON, HYBRIDPOWELL
	std::function<HMMPI::Mat(const std::vector<double>&)> Jac;					// Jac() input and output should be sync on MPI_COMM_WORLD

public:
	NewtonIter(std::string type, double eps_, int maxit_) : NonlinearSystemSolver(eps_, maxit_), Type(type) {};
	virtual void SetFuncFromVM(const VectorModel *vm);							// takes Func_ACT, Jac_ACT from "vm"; NOTE: "vm" object itself will be used in these function calls, so check it's properly set up
	virtual std::vector<double> Solve(std::vector<double> x);					// 1-norm is used as stopping criterion
	virtual void debug_check_jac(const gsl_vector *x) const;					// prints analytical and fin-diff Jacobians
};
//---------------------------------------------------------------------------
// SUNIter - Newton/FixedPoint/Picard method from SUNDIALS
// in the code, 'int' is used in place of 'sunindextype'		TODO they have different size -> potential problems?
// 'double' is used in place of 'sunrealtype'
// Currently, "SUNContext" is created without any MPI communicator
//---------------------------------------------------------------------------
static_assert(sizeof(double) == sizeof(sunrealtype), "'sunrealtype' should be 'double'");
//---------------------------------------------------------------------------
class SUNIter : public NewtonIter
{
private:																								// 'ctx' should be = this
	static int func(N_Vector u, N_Vector f, void *ctx);													// returns 0 on success, throws Exception on failure
	static int df(N_Vector u, N_Vector fu, SUNMatrix J, void *ctx, N_Vector tmp1, N_Vector tmp2);		// returns 0 on success, throws Exception on failure
	static void check_flag(const void *flagvalue, const char *funcname, int opt);						// checks flag and throws Exception if necessary

protected:
	// Type: KIN_NEWTON (no Line Search), KIN_NEWTON_LS (with Line Search), KIN_FP (fixed point with Anderson Acceleration), KIN_PICARD (with Anderson Acceleration)
	// eps - residual tolerance - is in inf-norm
	double eps_x;			// tolerance for step in inf-norm for stopping the iteration
	int maxJacIters;		// max nonlinear iterations before Jacobian recalculation; "1" corresponds to exact Newton method
	long int maa;			// Anderson acceleration subspace size (for fixed point and Picard)

public:
	long int func_evals;			// filled by Solve()
	long int jac_evals;

	SUNIter(std::string type, double eps_, double eps_x_, int maxit_, int maxjacit, long int maa_) : NewtonIter(type, eps_, maxit_), eps_x(eps_x_), maxJacIters(maxjacit), maa(maa_), func_evals(0), jac_evals(0) {};
	virtual std::vector<double> Solve(std::vector<double> x);					// inf-norms for residual and step are used as stopping criterion; NO SCALING for 'x' and 'f' is used
	virtual std::string msg(const std::vector<double> &x) const;
};
//---------------------------------------------------------------------------
// [Base] classes for optimization:
// OptContext, OptCtxLM - user's context; another derived class from OptContext is Parser_1
// Optimizer - abstract optimizer; its derived classes are: OptLM, OptCMAES
//---------------------------------------------------------------------------
class OptContext
{
public:
	virtual ~OptContext(){};
};
//---------------------------------------------------------------------------
class OptCtxLM : public OptContext
{
public:
	int maxit;
	double epsG, epsF, epsX;

	OptCtxLM(int M, double eg, double ef = 0, double ex = 0) : maxit(M), epsG(eg), epsF(ef), epsX(ex){};
};
//---------------------------------------------------------------------------
class Optimizer
{
protected:
	std::vector<double> x_opt_mult;				// found optimum point (RunOptMult)
	double best_of;		// best o.f. value after RunOptMult
	int mult_count;		// number of optimizations in RunOptMult

	virtual void reset_mult_stats(){};			// to be called in the beginning of RunOptMult/RunOptRestrict
	virtual void gather_mult_stats(){};			// to be called after each optimization within RunOptMult()
public:
	int restrict_choice;						// index showing how restricted optimization finished: 1 if RunOptRestrict returned f1; 2 if RunOptRestrict returned f2

	Optimizer() : best_of(0), mult_count(0), restrict_choice(0) {};
	virtual ~Optimizer(){};
	static Optimizer *Make(std::string type);	// Factory for producing optimizer instances; currently supported 'types': LM, LMFI, LMFIMIX, CMAES; *** DELETE the pointer in the end! ***
	virtual std::vector<double> RunOpt(PhysModel *pm, std::vector<double> x0, const OptContext *ctx) = 0;	// Optimises "pm" starting from "x0" (full-dim); "ctx" - user's context
																											// MPI: inputs and outputs are sync on all pm-comm-ranks
																											// input and output vectors are full-dim, inner variables
	std::vector<double> RunOptMult(PhysModel *pm, std::vector<std::vector<double>> X0, const OptContext *ctx);
												// Performs optimization starting from different points X0[], the best found optimum is returned;
												// after each optimization, an extra call to ObjFunc() is made.
												// This function also fills best_of, x_opt_mult, mult_count.
												// MPI: inputs and outputs are sync on all pm-comm-ranks
	std::vector<double> RunOptRestrict(PhysModel *pm, const std::vector<double> &x0, const double R, const double delta, const OptContext *ctx, const OptContext *ctx_spher, std::string &msg);
												// Performs restricted optimization of "pm" starting from "x0" (full-dim, internal representation); Restriction is: |x - x0|_2 <= R (plus the usual const bounds)
												// Optimization is done in two steps (with selection of the best result): a) ordinary optimization with const bounds, b) optimization on the sphere (with gap 'delta' on poles)
												// Additional message about the two steps is put to 'msg'
												// After each optimization (step), an extra call to ObjFunc() is made.
												// Steps 1 and 2 of optimization may accept different convergence settings: ctx, ctx_spher.
												// This function also fills best_of, x_opt_mult, mult_count, restrict_choice.
												// MPI: inputs and outputs are sync on all pm-comm-ranks
	std::vector<double> RunOptRestrictCube(PhysModel *pm, const std::vector<double> &x0, const double R, const OptContext *ctx);
												// Performs restricted optimization of "pm" starting from "x0" (full-dim, internal representation); Restriction is the cube: |x - x0|_inf <= R (plus the usual const bounds)
												// After optimization, an extra call to ObjFunc() is made.
												// This function also fills best_of, x_opt_mult, mult_count, restrict_choice.
												// MPI: inputs and outputs are sync on all pm-comm-ranks
	virtual std::string ReportMsg() = 0;				// Get optimization report (call after RunOpt); call on all ranks!
	virtual std::string ReportMsgMult(){return "";};	// Get optimization report (call after RunOptMult); call on all ranks!
	double get_best_of() const {return best_of;};
};
//---------------------------------------------------------------------------
// OptLM - class for performing Levenberg-Marquardt optimization with bound-constraints.
// Requires PhysModel with ObjFunc, ObjFuncGrad and ObjFuncHess; the model also defines the constraints and A/N flags for parameters.
// Works in terms of inner variables. MPI: call synchronously on all ranks.
// Uses ALGLIB
//---------------------------------------------------------------------------
class OptLM : public Optimizer
{
protected:
	std::string name;			// class name, for reporting
	PhysModel *pm_work;			// used by callback functions, and ReportMsg()
	std::vector<double> x_opt;	// found optimum point

	void make_checks(int fulldim) const;
	static void func(const alglib::real_1d_array &x, double &f, void *ptr);					// callback functions, pass ptr = &OptLM_object
	static void grad(const alglib::real_1d_array &x, double &f, alglib::real_1d_array &g, void *ptr);
	static void hess(const alglib::real_1d_array &x, double &f, alglib::real_1d_array &g, alglib::real_2d_array &h, void *ptr);

																							// virtual function allowing the derived classes to behave differently
	virtual void (*get_hess() const)(const alglib::real_1d_array &x, double &f, alglib::real_1d_array &g, alglib::real_2d_array &h, void *ptr){return hess;};

	virtual void reset_mult_stats();	// resets "sum_..."
	virtual void gather_mult_stats();	// fills "sum_..."
public:						// some quantities reported after optimization
	int conv_reason;		// ALGLIB's "terminationtype"
	int iter_count;			// iterations count
	int nfunc;				// number of function/gradient/Hessian/Cholesky calls
	int ngrad;
	int nhess;
	int nchol;

	int sum_iter_count;		// cumulative quantities reset by reset_mult_stats(), filled by gather_mult_stats()
	int sum_nfunc;
	int sum_ngrad;
	int sum_nhess;
	int sum_nchol;

	mutable double tfunc;	// time (sec) for function evaluation, gradients and Hessians
	mutable double tgrad;
	mutable double thess;

	double sum_tfunc;
	double sum_tgrad;
	double sum_thess;

	OptLM() : name("OptLM"), pm_work(NULL), conv_reason(0), iter_count(0), nfunc(0), ngrad(0), nhess(0), nchol(0),
						 	 sum_iter_count(0), sum_nfunc(0), sum_ngrad(0), sum_nhess(0), sum_nchol(0),
						 	 tfunc(0), tgrad(0), thess(0), sum_tfunc(0), sum_tgrad(0), sum_thess(0){};
	virtual std::vector<double> RunOpt(PhysModel *pm, std::vector<double> x0, const OptContext *ctx);		// runs Levenberg-Marquardt optimization for "pm" starting from "x0" (full-dim),
																		// A/N parameters are taken according to "pm"; bounds are taken from "pm" (may be NULL)
																		// returns the solution vector (full-dim), and fills the 'int' quantities above (conv_reason ... nchol)
																		// solution vector is also saved to "x_opt", "pm" saved to "pm_work"
																		// "ctx" should be OptCtxLM*
																		// MPI: inputs and outputs are sync on all pm-comm-ranks
	virtual std::string ReportMsg();		// note: this makes an extra call to ObjFunc()
	virtual std::string ReportMsgMult();
};
//---------------------------------------------------------------------------
// Converged reasons:
//* -7    derivative correctness check failed; see rep.funcidx, rep.varidx for more information.
//* -3    constraints are inconsistent
//*  1    relative function improvement is no more than EpsF.
//*  2    relative step is no more than EpsX.
//*  4    gradient is no more than EpsG.
//*  5    MaxIts steps was taken
//*  7    stopping conditions are too stringent, further improvement is impossible
//*  8    terminated by user who called minlmrequesttermination().
//        X contains point which was "current accepted" when termination request was submitted.
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
// class OptLMFI - same as OptLM, except that it uses 2*FI (Gauss-Newton) instead of Hessian
//---------------------------------------------------------------------------
class OptLMFI : public OptLM
{
protected:
	static void fi(const alglib::real_1d_array &x, double &f, alglib::real_1d_array &g, alglib::real_2d_array &h, void *ptr);

	virtual void (*get_hess() const)(const alglib::real_1d_array &x, double &f, alglib::real_1d_array &g, alglib::real_2d_array &h, void *ptr){return fi;};
public:
	OptLMFI() : OptLM() {name = "OptLMFI";};
};
//---------------------------------------------------------------------------
// class OptLMFImix - same as OptLM, except that it uses FImix (~ Gauss-Newton) instead of Hessian
//---------------------------------------------------------------------------
class OptLMFImix : public OptLM
{
protected:
	static void fimix(const alglib::real_1d_array &x, double &f, alglib::real_1d_array &g, alglib::real_2d_array &h, void *ptr);

	virtual void (*get_hess() const)(const alglib::real_1d_array &x, double &f, alglib::real_1d_array &g, alglib::real_2d_array &h, void *ptr){return fimix;};
public:
	OptLMFImix() : OptLM() {name = "OptLMFImix";};
};
//---------------------------------------------------------------------------


#endif /* GRADIENTOPT_H_ */
