/*
 * GradientOpt.cpp
 *
 *  Created on: 29.01.2014
 *      Author: FursovIV
 */

#define GSL_RANGE_CHECK_OFF

#include <mpi.h>
#include "Abstract.h"
#include "Utils.h"
#include "GradientOpt.h"
#include "CMAES_interface.h"
#include "PhysModels.h"
#include "lapacke_select.h"
#include <cmath>
#include <cassert>
#include <limits>
#include "Parsing2.h"
#include <kinsol/kinsol.h>             /* access to KINSOL func., consts. */
#include <nvector/nvector_serial.h>    /* access to serial N_Vector       */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix       */
//#include <kinsol/kinsol_direct.h>      /* access to KINDls interface      */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver */
#include <sundials/sundials_types.h>   /* defs. of realtype, sunindextype */
#include <chrono>

using namespace alglib;
using namespace std::chrono;
//---------------------------------------------------------------------------
// GradPhysMod
//---------------------------------------------------------------------------
void GradPhysMod::CalcX_aS(double a)
{
	size_t len = x.size();
	std::vector<double> res = x;
	if (a != 0)
	{
		for (size_t i = 0; i < len; i++)
		{
			res[i] += a * s[i];
		}
	}
	x_as = res;
}
//---------------------------------------------------------------------------
double GradPhysMod::CalcBeta()
{
	size_t len = grad_0.size();

	double sum1 = 0, sum2 = 0;
	for (size_t i = 0; i < len; i++)
	{
		sum1 += (grad_1[i] - grad_0[i]) * grad_1[i];
		sum2 += grad_0[i] * grad_0[i];
	}

	return sum1 / sum2;
}
//---------------------------------------------------------------------------
double GradPhysMod::Calc_df0()
{
	size_t len = grad_0.size();

	double res = 0;
	for (size_t i = 0; i < len; i++)
	{
		res += grad_0[i] * s[i];
	}

	return res;
}
//---------------------------------------------------------------------------
double GradPhysMod::Calc_df1()
{
	size_t len = grad_1.size();

	double res = 0;
	for (size_t i = 0; i < len; i++)
	{
		res += grad_1[i] * s[i];
	}

	return res;
}
//---------------------------------------------------------------------------
double GradPhysMod::Calc_grad_df1()
{
	GradObjFunc(1);
    return Calc_df1();
}
//---------------------------------------------------------------------------
std::vector<double> GradPhysMod::LinComb(double a, const std::vector<double> &U, double b, const std::vector<double> &V)
{
	size_t len = U.size();
	std::vector<double> res(len);

	if (b == 0)
	{
		for (size_t i = 0; i < len; i++)
		{
			res[i] = a*U[i];
		}
	}
	else
	{
		for (size_t i = 0; i < len; i++)
		{
			res[i] = a*U[i] + b*V[i];
		}
	}

	return res;
}
//---------------------------------------------------------------------------
std::string GradPhysMod::StringX()
{
	std::string res = "";
	size_t len = x_as.size();

	for (size_t i = 0; i < len; i++)
	{
		res += HMMPI::stringFormatArr("\t{0}", std::vector<double>{x_as[i]});
	}

	return res;
}
//---------------------------------------------------------------------------
// NonlinearSystemSolver
//---------------------------------------------------------------------------
void NonlinearSystemSolver::SetFuncFromVM(const VectorModel *vm)
{
	Func = std::bind(&VectorModel::Func_ACT, vm, std::placeholders::_1);
}
//---------------------------------------------------------------------------
std::string NonlinearSystemSolver::msg(const std::vector<double> &x) const
{
	if (!Func)
		throw HMMPI::Exception("Func is not callable in NonlinearSystemSolver::msg");

	std::string res = HMMPI::stringFormatArr("Итераций: {0:%d}", "Iterations: {0:%d}", iter);
	HMMPI::Mat f = Func(x);
	res += HMMPI::stringFormatArr(", 1-норма невязки: {0:%g}\n", ", 1-norm of residual: {0:%g}\n", f.Norm1());

	return res;
}
//---------------------------------------------------------------------------
// FixedPointIter
//---------------------------------------------------------------------------
std::vector<double> FixedPointIter::Solve(std::vector<double> x)		// 'x' is the starting point; 'iter' gets filled; input and output should be sync on MPI_COMM_WORLD
{																		// 1-norm is used as stopping criterion
	if (!Func)
		throw HMMPI::Exception("Func is not callable in FixedPointIter::Solve");

	iter = 0;									// iterations count
	bool finished = false;
	const size_t dim = x.size();
	double norm1 = 0;

	while (!finished)
	{
		std::vector<double> x1 = Func(x);		// x1 = Func(x)

		if (x1.size() != dim)
			throw HMMPI::Exception("Func() input and output have different dimensions");

		norm1 = 0;
		for (size_t i = 0; i < dim; i++)
			norm1 += fabs(x1[i]);

		for (size_t i = 0; i < dim; i++)		// x1 = Func(x) + x
			x1[i] += x[i];

		x = std::move(x1);
		iter++;
		if (iter >= maxit || norm1 < eps)
			finished = true;
	}
	if (norm1 >= eps)
		throw HMMPI::Exception("Fixed point iteration has not converged");

	Func = std::function<std::vector<double>(const std::vector<double>&)>();		// erasing function
	return x;
}
//---------------------------------------------------------------------------
// NonlinearSystemSolver
//---------------------------------------------------------------------------
int NewtonIter::func(const gsl_vector *x, void *params, gsl_vector *f)
{
	const NewtonIter *this_obj = static_cast<const NewtonIter*>(params);

	std::vector<double> x0(x->size);
	for (size_t i = 0; i < x0.size(); i++)
		x0[i] = gsl_vector_get(x, i);

	std::vector<double> f0 = this_obj->Func(x0);
	if (f0.size() != x0.size())
		throw HMMPI::Exception("Func() input and output have different dimensions in NewtonIter::func");

	if (f->size != f0.size())
		throw HMMPI::Exception("f->size != f0.size() in NewtonIter::func");

	for (size_t i = 0; i < f0.size(); i++)
		gsl_vector_set(f, i, f0[i]);

	return GSL_SUCCESS;
}
//---------------------------------------------------------------------------
int NewtonIter::df(const gsl_vector *x, void *params, gsl_matrix *jac)
{
	const NewtonIter *this_obj = static_cast<const NewtonIter*>(params);

	std::vector<double> x0(x->size);
	for (size_t i = 0; i < x0.size(); i++)
		x0[i] = gsl_vector_get(x, i);

	HMMPI::Mat jac0 = this_obj->Jac(x0);
	if (jac0.ICount() != x0.size() || jac0.JCount() != x0.size())
		throw HMMPI::Exception("Jac() output dimensions are not the same as for input, in NewtonIter::df");

	if (jac0.ICount() != jac->size1 || jac0.JCount() != jac->size2)
		throw HMMPI::Exception("jac0.ICount() != jac->size1 || jac0.JCount() != jac->size2 in NewtonIter::df");

	for (size_t i = 0; i < jac0.ICount(); i++)
		for (size_t j = 0; j < jac0.JCount(); j++)
			gsl_matrix_set(jac, i, j, jac0(i, j));

	return GSL_SUCCESS;
}
//---------------------------------------------------------------------------
int NewtonIter::fdf(const gsl_vector *x, void *params, gsl_vector *f, gsl_matrix *jac)
{
	int res_f = func(x, params, f);
	if (res_f != GSL_SUCCESS)
		return res_f;

	int res_df = df(x, params, jac);
	if (res_df != GSL_SUCCESS)
		return res_df;

	return GSL_SUCCESS;
}
//---------------------------------------------------------------------------
void NewtonIter::SetFuncFromVM(const VectorModel *vm)
{
	Func = std::bind(&VectorModel::Func_ACT, vm, std::placeholders::_1);
	Jac = std::bind(&VectorModel::Jac_ACT, vm, std::placeholders::_1);
}
//---------------------------------------------------------------------------
std::vector<double> NewtonIter::Solve(std::vector<double> x)							// 1-norm is used as stopping criterion
{
	int err = 0, errloc = 0;
	const size_t dim = x.size();
	std::vector<double> res(dim);

	try
	{
		if (!Func)
			throw HMMPI::Exception("Func is not callable in NewtonIter::Solve");
		if (!Jac)
			throw HMMPI::Exception("Jac is not callable in NewtonIter::Solve");

		const gsl_multiroot_fdfsolver_type *T = 0;
		if (Type == "NEWTON")
			T = gsl_multiroot_fdfsolver_newton;			// Newton’s Method
		else if (Type == "GNEWTON")
			T = gsl_multiroot_fdfsolver_gnewton;		// modified version of Newton’s method
		else if (Type == "HYBRIDPOWELL")
			T = gsl_multiroot_fdfsolver_hybridsj;		// modified version of Powell’s Hybrid method
		else
			throw HMMPI::Exception("Solver type " + Type + " not recognized in NewtonIter::Solve");

		gsl_multiroot_fdfsolver *s = gsl_multiroot_fdfsolver_alloc(T, dim);
		if (s == NULL)
			errloc = 1;
		MPI_Allreduce(&errloc, &err, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		if (err)
			throw HMMPI::Exception("Failed to allocate solver in NewtonIter::Solve");

		gsl_vector *xgsl = gsl_vector_alloc(dim);
		if (xgsl == NULL)
			errloc = 1;
		MPI_Allreduce(&errloc, &err, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		if (err)
			throw HMMPI::Exception("Failed to allocate vector in NewtonIter::Solve");

		for (size_t i = 0; i < dim; i++)
			gsl_vector_set(xgsl, i, x[i]);

		gsl_multiroot_function_fdf FDF = {func, df, fdf, dim, this};
		gsl_multiroot_fdfsolver_set(s, &FDF, xgsl);

		if (flag)
			debug_check_jac(xgsl);

		int status;
		iter = 0;
		do
		{
			status = gsl_multiroot_fdfsolver_iterate(s);
			if (status)
				break;

			status = gsl_multiroot_test_residual(s->f, eps);	// 1-norm test
			iter++;

			// DEBUG
	//		{
	//			std::vector<double> sol_intermeidate(dim);
	//			for (size_t i = 0; i < dim; i++)
	//				sol_intermeidate[i] = gsl_vector_get(s->x, i);
	//			std::cout << "NewtonIter, x = " << HMMPI::ToString(sol_intermeidate);	// DEBUG
	//		}
			// DEBUG
		}
		while (status == GSL_CONTINUE && iter < maxit);

		if (s->x->size != dim)
			throw HMMPI::Exception("Incorrect solution dimension in NewtonIter::Solve");
		if (status != GSL_SUCCESS)
		{
			char msg[HMMPI::BUFFSIZE];
			sprintf(msg, "Solver finished with status '%s', completed %d iteration(s), eps_tol = %g", gsl_strerror(status), iter, eps);
			throw HMMPI::Exception(msg);
		}

		for (size_t i = 0; i < dim; i++)
			res[i] = gsl_vector_get(s->x, i);

		gsl_multiroot_fdfsolver_free(s);
		gsl_vector_free(xgsl);
	}
	catch (const std::exception &e)
	{
		std::string msg0 = e.what() + exc_msg;
		exc_msg = "";
		throw HMMPI::Exception(msg0);
	}

	Func = std::function<std::vector<double>(const std::vector<double>&)>();		// erasing function
	Jac = std::function<HMMPI::Mat(const std::vector<double>&)>();					// erasing Jacobian
	return res;
}
//---------------------------------------------------------------------------
void NewtonIter::debug_check_jac(const gsl_vector *x) const							// prints analytical and fin-diff Jacobians
{
	if (!Func)
		throw HMMPI::Exception("Func is not callable in NewtonIter::debug_check_jac");
	if (!Jac)
		throw HMMPI::Exception("Jac is not callable in NewtonIter::debug_check_jac");

	int RNK;
	MPI_Comm_rank(MPI_COMM_WORLD, &RNK);

	std::vector<double> x0(x->size);
	for (size_t i = 0; i < x0.size(); i++)
		x0[i] = gsl_vector_get(x, i);

	if (RNK == 0)
		std::cout << "x = " << HMMPI::ToString(x0);

	HMMPI::Mat jac0 = Jac(x0);							// analytical Jacobian
	if (jac0.ICount() != x0.size() || jac0.JCount() != x0.size())
		throw HMMPI::Exception("Jac() output dimensions are not the same as for input, in NewtonIter::debug_check_jac");

	HMMPI::Mat jacnum;									// numerical Jacobian, fixed dh is used, with central finite differences (OH2)
	const double dh = 1e-5;
	for (size_t i = 0; i < x0.size(); i++)
	{
		x0[i] += dh;
		HMMPI::Mat v = Func(x0);
		x0[i] -= 2*dh;
		v = (0.5/dh)*(std::move(v) - Func(x0));
		x0[i] += dh;

		v.Reshape(1, x0.size());
		jacnum = std::move(jacnum) || v;
	}
	jacnum = jacnum.Tr();

	if (RNK == 0)
	{
		std::cout << "Analytical Jacobian\n" << jac0.ToString();
		std::cout << "Numerical Jacobian\n" << jacnum.ToString();
		std::cout << "2-Norm of diff " << (jac0 - jacnum).Norm2() << "\n\n";
	}
}
//---------------------------------------------------------------------------
// SUNIter
//---------------------------------------------------------------------------
int SUNIter::func(N_Vector u, N_Vector f, void *ctx)												// returns 0 on success, throws Exception on failure
{
	const SUNIter *this_obj = static_cast<const SUNIter*>(ctx);

	int dim = N_VGetLength_Serial(u);							// big indices are not expected, so using 'int'
	if (dim != (int)N_VGetLength_Serial(f))
		throw HMMPI::Exception("Sizes of 'u' and 'f' do not match in SUNIter::func");

	const double *u_data = N_VGetArrayPointer(u);
	double *f_data = N_VGetArrayPointer(f);

	std::vector<double> u0(u_data, u_data + dim);
	std::vector<double> f0 = this_obj->Func(u0);
	if ((int)f0.size() != dim)
		throw HMMPI::Exception("'Func' input and output have different dimensions");

	memcpy(f_data, f0.data(), dim*sizeof(double));
	return 0;
}
//---------------------------------------------------------------------------
int SUNIter::df(N_Vector u, N_Vector fu, SUNMatrix J, void *ctx, N_Vector tmp1, N_Vector tmp2)		// returns 0 on success, throws Exception on failure
{
	const SUNIter *this_obj = static_cast<const SUNIter*>(ctx);

	int dim = N_VGetLength_Serial(u);
	if (dim != (int)SM_ROWS_D(J) || dim != (int)SM_COLUMNS_D(J))
		throw HMMPI::Exception("Sizes of 'J' and 'u' are not consistent in SUNIter::df");

	const double *u_data = N_VGetArrayPointer(u);
	double *J_data = SM_DATA_D(J);								// col-major storage

	std::vector<double> u0(u_data, u_data + dim);
	HMMPI::Mat J0 = this_obj->Jac(u0).Tr();						// transpose to get col-major
	if ((int)J0.ICount() != dim || (int)J0.JCount() != dim)
		throw HMMPI::Exception("'Jac' output dimensions are not suitable");

	memcpy(J_data, J0.ToVector().data(), size_t(dim)*size_t(dim)*sizeof(double));
	return 0;
}
//---------------------------------------------------------------------------
void SUNIter::check_flag(const void *flagvalue, const char *funcname, int opt)
{
	char msg[HMMPI::BUFFSIZE];
	if (opt == 0 && flagvalue == NULL)			// Check if SUNDIALS function returned NULL pointer - no memory allocated
	{
		sprintf(msg, "SUNDIALS_Error: %s() failed - returned NULL pointer", funcname);
		throw HMMPI::Exception(msg);
	}
	else if (opt == 1)							// Check if *flag < 0
	{
		const int *errflag = (const int*)flagvalue;
		if (*errflag < 0)
		{
			sprintf(msg, "SUNDIALS_Error: %s() failed with flag = %d", funcname, *errflag);
			throw HMMPI::Exception(msg);
		}
	}
	else if (opt == 2 && flagvalue == NULL)		// Check if function returned NULL pointer - no memory allocated
	{
		sprintf(msg, "MEMORY_Error: %s() failed - returned NULL pointer", funcname);
		throw HMMPI::Exception(msg);
	}
}
//---------------------------------------------------------------------------
std::vector<double> SUNIter::Solve(std::vector<double> x)
{
	int flag;
	SUNContext sunctx = 0;
	try
	{
		flag = SUNContext_Create(SUN_COMM_NULL, &sunctx);
		check_flag(&flag, "SUNContext_Create", 1);

		if (!Func)
			throw HMMPI::Exception("Func is not callable in SUNIter::Solve");
		if (!Jac)
			throw HMMPI::Exception("Jac is not callable in SUNIter::Solve");

		N_Vector u, scale;
		void *kin;
		SUNMatrix J;
		SUNLinearSolver LS;
		int dim = x.size(), glob_strat;

		if (maxJacIters < 0)
			throw HMMPI::Exception("maxJacIters should be >= 0 in SUNIter::Solve");

		if (Type == "KIN_NEWTON")
			glob_strat = KIN_NONE;
		else if (Type == "KIN_NEWTON_LS")
			glob_strat = KIN_LINESEARCH;
		else if (Type == "KIN_FP")
			glob_strat = KIN_FP;
		else if (Type == "KIN_PICARD")
			glob_strat = KIN_PICARD;
		else
			throw HMMPI::Exception("Incorrect type " + Type + " in SUNIter::Solve");

		u = N_VNew_Serial(dim, sunctx);
		check_flag((void *)u, "N_VNew_Serial", 0);
		memcpy(N_VGetArrayPointer(u), x.data(), dim*sizeof(double));

		scale = N_VNew_Serial(dim, sunctx);
		check_flag((void *)scale, "N_VNew_Serial", 0);
		N_VConst_Serial(1, scale); 								// no scaling

		kin = KINCreate(sunctx);
		check_flag((void *)kin, "KINCreate", 0);

		flag = KINSetUserData(kin, this);
		check_flag(&flag, "KINSetUserData", 1);

		flag = KINSetFuncNormTol(kin, eps);
		check_flag(&flag, "KINSetFuncNormTol", 1);

		flag = KINSetScaledStepTol(kin, eps_x);
		check_flag(&flag, "KINSetScaledStepTol", 1);

		flag = KINSetNumMaxIters(kin, maxit);
		check_flag(&flag, "KINSetNumMaxIters", 1);

		flag = KINSetMaxSetupCalls(kin, maxJacIters);
		check_flag(&flag, "KINSetMaxSetupCalls", 1);

		flag = KINSetMAA(kin, maa);
		check_flag(&flag, "KINSetMAA", 1);

		flag = KINInit(kin, func, u);
		check_flag(&flag, "KINInit", 1);

		J = SUNDenseMatrix(dim, dim, sunctx);					// Create dense SUNMatrix
		check_flag((void *)J, "SUNDenseMatrix", 0);

		LS = SUNLinSol_Dense(u, J, sunctx);						// Create dense SUNLinearSolver object
		check_flag((void *)LS, "SUNLinSol_Dense", 0);

		flag = KINSetLinearSolver(kin, LS, J);					// Attach the matrix and linear solver to KINSOL
		check_flag(&flag, "KINSetLinearSolver", 1);

		flag = KINSetJacFn(kin, df);
		check_flag(&flag, "KINSetJacFn", 1);

		flag = KINSol(kin, u, glob_strat, scale, scale);
		check_flag(&flag, "KINSol", 1);

		const double *u_data = N_VGetArrayPointer(u);			// Collect the results
		std::vector<double> res(u_data, u_data + dim);

		flag = KINGetNumFuncEvals(kin, &func_evals);
		check_flag(&flag, "KINGetNumFuncEvals", 1);

		long int niters;
		flag = KINGetNumNonlinSolvIters(kin, &niters);
		check_flag(&flag, "KINGetNumNonlinSolvIters", 1);
		iter = niters;

		flag = KINGetNumJacEvals(kin, &jac_evals);
		check_flag(&flag, "KINGetNumJacEvals", 1);

		N_VDestroy_Serial(u);									// Free memory
		N_VDestroy_Serial(scale);
		KINFree(&kin);

		flag = SUNLinSolFree(LS);
		check_flag(&flag, "SUNLinSolFree", 1);

		SUNMatDestroy(J);

		Func = std::function<std::vector<double>(const std::vector<double>&)>();		// erasing function
		Jac = std::function<HMMPI::Mat(const std::vector<double>&)>();					// erasing Jacobian

		if (sunctx != 0)
		{
			flag = SUNContext_Free(&sunctx);
			check_flag(&flag, "SUNContext_Free", 1);
		}

		return res;
	}
	catch (const std::exception &e)
	{
		if (sunctx != 0)
			SUNContext_Free(&sunctx);				// TODO other objects should also be freed

		std::string msg0 = e.what() + exc_msg;
		exc_msg = "";
		throw HMMPI::Exception(msg0);
	}
}
//---------------------------------------------------------------------------
std::string SUNIter::msg(const std::vector<double> &x) const
{
	if (!Func)
		throw HMMPI::Exception("Func is not callable in SUNIter::msg");

	char res[HMMPI::BUFFSIZE];
	char reseng[HMMPI::BUFFSIZE];

	HMMPI::Mat f = Func(x);
	double norm = f.NormInf();
	sprintf(res, "Итераций: %d, выч. функ. %ld, выч. Якоб. %ld, inf-норма невязки: %g\n", iter, func_evals, jac_evals, norm);
	sprintf(reseng, "Iterations: %d, func eval-s: %ld, Jac. eval-s %ld, inf-norm of residual: %g\n", iter, func_evals, jac_evals, norm);

	return HMMPI::MessageRE(res, reseng);
}
//---------------------------------------------------------------------------
// Optimizer
//---------------------------------------------------------------------------
Optimizer *Optimizer::Make(std::string type)	// Factory for producing optimizer instances; currently supported 'types': LM, LMFI, LMFIMIX, CMAES
{												// *** DELETE the pointer in the end! ***
	if (type == "CMAES")
		return new OptCMAES;
	else if (type == "LM")
		return new OptLM;
	else if (type == "LMFI")
		return new OptLMFI;
	else if (type == "LMFIMIX")
		return new OptLMFImix;
	else
		throw HMMPI::Exception(HMMPI::stringFormatArr("Неправильный тип алгоритма {0:%s} в Optimizer::Make",
													  "Incorrect algorithm type {0:%s} in Optimizer::Make", type));
}
//---------------------------------------------------------------------------
std::vector<double> Optimizer::RunOptMult(PhysModel *pm, std::vector<std::vector<double>> X0, const OptContext *ctx)
{
	MPI_Comm comm = pm->GetComm();
	HMMPI::Bcast_vector(X0, 0, comm);				// sync X0 just in case

	mult_count = X0.size();
	best_of = std::numeric_limits<double>::max();
	reset_mult_stats();

	for (int i = 0; i < mult_count; i++)
	{
		std::vector<double> x1 = RunOpt(pm, X0[i], ctx);
		gather_mult_stats();
		double of = pm->ObjFunc(x1);
		MPI_Bcast(&of, 1, MPI_DOUBLE, 0, comm);		// sync o.f.
		if (of < best_of)
		{
			best_of = of;
			x_opt_mult = x1;
		}
	}

	return x_opt_mult;
}
//---------------------------------------------------------------------------
std::vector<double> Optimizer::RunOptRestrict(PhysModel *pm, const std::vector<double> &x0, const double R, const double delta, const OptContext *ctx, const OptContext *ctx_spher, std::string &msg)
{
	const int Max_spher_changes = pm->ParamsDim_ACT();	// maximum number of changes 0 <-> 2*pi for the last coordinate during optimization on the sphere (some other value could be used here as well!)

	mult_count = 0;
	reset_mult_stats();
	bool take1 = false, take2 = false;					// show if results from steps 'a' and 'b' participate in comparison
	bool result2 = false;								// true if result from step 'b' is finally taken

	// *****
	// * A *	ordinary optimization, x0 -> x1
	// *****
	std::vector<double> x1 = RunOpt(pm, x0, ctx);
	mult_count++;
	gather_mult_stats();
	double of1 = pm->ObjFunc(x1);
	MPI_Bcast(&of1, 1, MPI_DOUBLE, 0, pm->GetComm());	// sync o.f.
	double dist1 = (HMMPI::Mat(x0) - HMMPI::Mat(x1)).Norm2();
	if (dist1 <= R)
		take1 = true;

	// *****
	// * B *	spherical optimization, x1 -> p1 -> p1' -> x2
	// *****
	const HMMPI::BlockDiagMat *bdc_del, *bdc_work;
	bdc_del = pm->GetBDC(&bdc_work);
	PM_Spherical Spher(pm, bdc_work, R, pm->act_par(x0), delta);

	bool finished = false;								// controls the meta-iterations
	int c = 0;											// counts meta-iterations
	std::vector<double> p1 = Spher.SC().cart_to_spher(pm->act_par(x1));
	Spher.GetConstr()->AdjustInitSpherical(p1);

	//Spher.SC().R = R;		- old version
	while (!finished)
	{
		p1 = RunOpt(&Spher, p1, ctx_spher);					// NOTE the same 'ctx' is used for spherical optimization
		mult_count++;
		gather_mult_stats();
		finished = !Spher.SC().periodic_swap(Spher.GetConstr(), p1);

		if (dynamic_cast<OptLM*>(this) != nullptr && dynamic_cast<OptLM*>(this)->iter_count == 0)
			finished = true;
		c++;
		if (c >= Max_spher_changes)
			finished = true;
	}
	std::vector<double> x2 = pm->tot_par(Spher.SC().spher_to_cart(p1));
	double of2 = pm->ObjFunc(x2);
	MPI_Bcast(&of2, 1, MPI_DOUBLE, 0, pm->GetComm());	// sync o.f.
	if (pm->CheckLimitsEps(x2, LAPACKE_dlamch('P')))
		take2 = true;
	else	// DEBUG output why x2 is not taken
	{
		int rank0;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank0);

		const std::vector<double> min = pm->GetConstr()->fullmin();
		const std::vector<double> max = pm->GetConstr()->fullmax();
		if (rank0 == 0)
		{
			std::cout << "********************* spher opt not taken, violated coords follow *********************\n";	// DEBUG
			for (size_t j = 0; j < x2.size(); j++)
				if (x2[j] < min[j] || x2[j] > max[j])
					printf("i = %d, min = %g, x2 = %g, max = %g\n", (int)j, min[j], x2[j], max[j]);	// DEBUG

			std::cout << "\n";	// DEBUG
		}	// DEBUG
	}

	// *****
	// * C *	decide which optimization result is finally taken
	// *****
	if (take1 && take2)			// x1, x2 participate in comparison
	{
		if (of2 < of1)
			result2 = true;
	}
	else if (take2)				// only x2 participates
		result2 = true;
	else						// only x1 participates, or no one participates
		result2 = false;

	if (!result2)
	{
		best_of = of1;
		x_opt_mult = x1;
		restrict_choice = 1;
	}
	else
	{
		best_of = of2;
		x_opt_mult = x2;
		restrict_choice = 2;
	}

	auto func_xv = [](bool take) -> std::string {return take ? "v" : "x";};
	msg = std::string(HMMPI::MessageRE("Границы: (", "Bounds: (")) + func_xv(take1) + ", " + func_xv(take2) + "), " +
					  HMMPI::stringFormatArr("(f1, f2) = ({0:%g}, {1:%g}), ", std::vector<double>{of1, of2}) +
					  HMMPI::stringFormatArr("результат: f{0:%d}\n", "result: f{0:%d}\n", restrict_choice);

	delete bdc_del;

	return x_opt_mult;
}
//---------------------------------------------------------------------------
std::vector<double> Optimizer::RunOptRestrictCube(PhysModel *pm, const std::vector<double> &x0, const double R, const OptContext *ctx)
{
	reset_mult_stats();

	const HMMPI::BlockDiagMat *bdc_del, *bdc_work;
	bdc_del = pm->GetBDC(&bdc_work);
	PM_CubeBounds Cube(pm, bdc_work, R, x0);

	x_opt_mult = RunOpt(&Cube, x0, ctx);
	mult_count = 1;
	gather_mult_stats();

	best_of = pm->ObjFunc(x_opt_mult);
	MPI_Bcast(&best_of, 1, MPI_DOUBLE, 0, pm->GetComm());	// sync o.f.

	// determine if the optimal point is on the boundary
	restrict_choice = 1;
	const double eps = LAPACKE_dlamch('P');
	const ParamsInterface *cube_con = dynamic_cast<const ParamsInterface*>(Cube.GetConstr());
	assert(cube_con != nullptr);
	const std::vector<double> min = cube_con->fullmin();
	const std::vector<double> max = cube_con->fullmax();
	assert(x_opt_mult.size() == min.size());

	for (size_t i = 0; i < x_opt_mult.size(); i++)
		if (cube_con->act[i] == "A" && (fabs(x_opt_mult[i] - min[i]) < eps || fabs(x_opt_mult[i] - max[i]) < eps))
		{
			restrict_choice = 2;
			break;
		}

	delete bdc_del;

	return x_opt_mult;
}
//---------------------------------------------------------------------------
// OptLM
//---------------------------------------------------------------------------
void OptLM::make_checks(int fulldim) const
{
	if (pm_work == NULL)
		throw HMMPI::Exception("pm_work == NULL in " + name);
	if (pm_work->ParamsDim() != fulldim)
		throw HMMPI::Exception("pm_work->ParamsDim() != fulldim in " + name);

	const HMMPI::BoundConstr *con = pm_work->GetConstr();
	if (con != NULL && (con->fullmin().size() != (size_t)fulldim || con->fullmax().size() != (size_t)fulldim))
		throw HMMPI::Exception("'pm->con' dimension != fulldim in " + name);
}
//---------------------------------------------------------------------------
void OptLM::func(const real_1d_array &x, double &f, void *ptr)
{
	const OptLM *this_obj = static_cast<const OptLM*>(ptr);
	int fulldim = x.length();
	this_obj->make_checks(fulldim);
	const double *xdata = x.getcontent();
	std::vector<double> xin = std::vector<double>(xdata, xdata + fulldim);

	high_resolution_clock::time_point time1 = high_resolution_clock::now(), time_of;

	f = this_obj->pm_work->ObjFunc(xin);
	time_of = high_resolution_clock::now();

	this_obj->tfunc += duration_cast<duration<double>>(time_of-time1).count();		// update the time stats

	MPI_Comm comm = this_obj->pm_work->GetComm();		// MPI-synchronisation
	if (comm != MPI_COMM_NULL)
		MPI_Bcast(&f, 1, MPI_DOUBLE, 0, comm);
}
//---------------------------------------------------------------------------
void OptLM::grad(const real_1d_array &x, double &f, real_1d_array &g, void *ptr)
{
	const OptLM *this_obj = static_cast<const OptLM*>(ptr);
	int fulldim = x.length();
	this_obj->make_checks(fulldim);
	const double *xdata = x.getcontent();
	std::vector<double> xin = std::vector<double>(xdata, xdata + fulldim);

	high_resolution_clock::time_point time1 = high_resolution_clock::now(), time_of, time_grad;

	f = this_obj->pm_work->ObjFunc(xin);
	time_of = high_resolution_clock::now();

	std::vector<double> gr = this_obj->pm_work->ObjFuncGrad(xin);
	time_grad = high_resolution_clock::now();

	this_obj->tfunc += duration_cast<duration<double>>(time_of-time1).count();		// update the time stats
	this_obj->tgrad += duration_cast<duration<double>>(time_grad-time_of).count();	// update the time stats

	MPI_Comm comm = this_obj->pm_work->GetComm();		// MPI-synchronisation
	if (comm != MPI_COMM_NULL)
	{
		MPI_Bcast(&f, 1, MPI_DOUBLE, 0, comm);
		HMMPI::Bcast_vector(gr, 0, comm);
	}

	assert(gr.size() == (size_t)fulldim);
	real_1d_array g_work;
	g_work.setcontent(fulldim, gr.data());
	g = g_work;
}
//---------------------------------------------------------------------------
void OptLM::hess(const real_1d_array &x, double &f, real_1d_array &g, real_2d_array &h, void *ptr)
{
	const OptLM *this_obj = static_cast<const OptLM*>(ptr);
	int fulldim = x.length();
	this_obj->make_checks(fulldim);
	const double *xdata = x.getcontent();
	std::vector<double> xin = std::vector<double>(xdata, xdata + fulldim);

	high_resolution_clock::time_point time1 = high_resolution_clock::now(), time_of, time_grad, time_hess;

	f = this_obj->pm_work->ObjFunc(xin);
	time_of = high_resolution_clock::now();

	std::vector<double> gr = this_obj->pm_work->ObjFuncGrad(xin);
	time_grad = high_resolution_clock::now();

	HMMPI::Mat Hess = this_obj->pm_work->ObjFuncHess(xin);
	time_hess = high_resolution_clock::now();

	this_obj->tfunc += duration_cast<duration<double>>(time_of-time1).count();			// update the time stats
	this_obj->tgrad += duration_cast<duration<double>>(time_grad-time_of).count();		// update the time stats
	this_obj->thess += duration_cast<duration<double>>(time_hess-time_grad).count();	// update the time stats

	MPI_Comm comm = this_obj->pm_work->GetComm();		// MPI-synchronisation
	if (comm != MPI_COMM_NULL)
	{
		MPI_Bcast(&f, 1, MPI_DOUBLE, 0, comm);
		HMMPI::Bcast_vector(gr, 0, comm);
		Hess.Bcast(0, comm);
	}

	assert(gr.size() == (size_t)fulldim);
	assert(Hess.ICount() == (size_t)fulldim && Hess.JCount() == (size_t)fulldim);

	real_1d_array g_work;
	g_work.setcontent(fulldim, gr.data());
	g = g_work;

	real_2d_array h_work;
	h_work.setcontent(fulldim, fulldim, Hess.Serialize());
	h = h_work;
}
//---------------------------------------------------------------------------
void OptLM::reset_mult_stats()
{
	sum_iter_count = sum_nfunc = sum_ngrad = sum_nhess = sum_nchol = 0;
	sum_tfunc = sum_tgrad = sum_thess = 0;
}
//---------------------------------------------------------------------------
void OptLM::gather_mult_stats()
{
	sum_iter_count += iter_count;
	sum_nfunc += nfunc;
	sum_ngrad += ngrad;
	sum_nhess += nhess;
	sum_nchol += nchol;

	sum_tfunc += tfunc;
	sum_tgrad += tgrad;
	sum_thess += thess;
}
//---------------------------------------------------------------------------
std::vector<double> OptLM::RunOpt(PhysModel *pm, std::vector<double> x0, const OptContext *ctx)
{
	const OptCtxLM *Ctx = dynamic_cast<const OptCtxLM*>(ctx);
	if (Ctx == nullptr)
		throw HMMPI::Exception("Cannot convert OptContext* to OptCtxLM* in " + name + "::RunOpt");

	pm_work = pm;
	x0 = pm_work->tot_par(pm_work->act_par(x0));		// replace inactive params by init values
	int fulldim = x0.size();
	make_checks(fulldim);

	if (pm_work->GetComm() == MPI_COMM_NULL)
		return std::vector<double>();					// empty vector is returned on ranks where "pm" is 'not defined'

	if (!pm_work->CheckLimits(x0))
		throw HMMPI::Exception("Initial point 'x0' violates the bounds in " + name + "::RunOpt, x0 =\n" + HMMPI::ToString(x0, "%20.16g", "\n"));

	tfunc = tgrad = thess = 0;
	try
	{
		minlmstate state;
		minlmreport rep;
		real_1d_array x;
		x.setlength(fulldim);
		x.setcontent(fulldim, x0.data());
		minlmcreatefgh(x, state);							// create optimizer

		minlmsetcond(state, Ctx->epsG, Ctx->epsF, Ctx->epsX, Ctx->maxit);		// set stopping criteria

		std::vector<double> min, max;						// auxiliary vectors for constraints
		const HMMPI::BoundConstr *con = pm_work->GetConstr();
		if (con == NULL)									// no constraints
		{
			min = std::vector<double>(fulldim, -std::numeric_limits<double>::infinity());
			max = std::vector<double>(fulldim, std::numeric_limits<double>::infinity());
		}
		else												// constraints are set
		{
			min = con->fullmin();
			max = con->fullmax();
		}

		min = pm_work->tot_par(pm_work->act_par(min));		// replace inactive params by init values
		max = pm_work->tot_par(pm_work->act_par(max));

		real_1d_array bndl, bndu;							// bound-constraints to be used by optimizer
		bndl.setlength(fulldim);
		bndu.setlength(fulldim);
		bndl.setcontent(fulldim, min.data());
		bndu.setcontent(fulldim, max.data());
		minlmsetbc(state, bndl, bndu);						// set bound-constraints

		minlmoptimize(state, func, grad, get_hess(), NULL, this);				// optimize and get results
		minlmresults(state, x, rep);

		conv_reason = rep.terminationtype;
		iter_count = rep.iterationscount;
		nfunc = rep.nfunc;
		ngrad = rep.ngrad;
		nhess = rep.nhess;
		nchol = rep.ncholesky;

		// tfunc, tgrad, thess are accumulated internally

//		{	// DEBUG
//			int Rank;
//			MPI_Comm_rank(MPI_COMM_WORLD, &Rank);
//			if (Rank == 0)
//				std::cout << "+++++++ DEBUG LM iter count " << iter_count << "\n";	// DEBUG
//		}	// DEBUG

		const double *data = x.getcontent();
		x_opt = std::vector<double>(data, data + fulldim);
		//pm_work = NULL;	keep "pm_work" to use it in ReportMsg()
	}
	catch (const ap_error &e)
	{
		throw HMMPI::Exception(e.msg);
	}

	return x_opt;
}
//---------------------------------------------------------------------------
std::string OptLM::ReportMsg()
{
	char res[HMMPI::BUFFSIZE];
	char reseng[HMMPI::BUFFSIZE];

	if (pm_work == NULL)
		throw HMMPI::Exception("Call to " + name + "::ReportMsg before " + name + "::RunOpt");

	double of = pm_work->ObjFunc(x_opt);
	sprintf(res, "Целевая функция = %g\n"
			"Тип завершения оптимизации: %d\n"
			"Итераций: %d\n"
			"Кол-во ц.ф. %d (%.4g сек.)\n"
			"Кол-во град. %d (%.4g сек.)\n"
			"Кол-во Гесс. %d (%.4g сек.)\n"
			"Кол-во разл. Хол. %d\n", of, conv_reason, iter_count, nfunc, tfunc, ngrad, tgrad, nhess, thess, nchol);

	sprintf(reseng, "Objective function = %g\n"
			"Converged reason: %d\n"
			"Iterations: %d\n"
			"# o.f. %d (%.4g sec.)\n"
			"# grad. %d (%.4g sec.)\n"
			"# Hess. %d (%.4g sec.)\n"
			"# Chol. %d\n", of, conv_reason, iter_count, nfunc, tfunc, ngrad, tgrad, nhess, thess, nchol);

	return HMMPI::MessageRE(res, reseng);
}
//---------------------------------------------------------------------------
std::string OptLM::ReportMsgMult()
{
	char res[HMMPI::BUFFSIZE];
	char reseng[HMMPI::BUFFSIZE];

	sprintf(res, "Целевая функция = %g\n"
			"Запусков оптимизации: %d, всего итераций: %d\n"
			"Количество расчетов ц.ф. %d (%.4g с), градиентов %d (%.4g с), Гессианов %d (%.4g с), разл. Хол. %d\n", best_of, mult_count, sum_iter_count, sum_nfunc, sum_tfunc, sum_ngrad, sum_tgrad, sum_nhess, sum_thess, sum_nchol);

	sprintf(reseng, "Objective function = %g\n"
			"Optimization runs: %d, total iterations: %d\n"
			"Number of evaluations of o.f. %d (%.4g s), gradient %d (%.4g s), Hessian %d (%.4g s), Chol. dec. %d\n", best_of, mult_count, sum_iter_count, sum_nfunc, sum_tfunc, sum_ngrad, sum_tgrad, sum_nhess, sum_thess, sum_nchol);

	return HMMPI::MessageRE(res, reseng);
}
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
void OptLMFI::fi(const alglib::real_1d_array &x, double &f, alglib::real_1d_array &g, alglib::real_2d_array &h, void *ptr)
{
	const OptLMFI *this_obj = static_cast<const OptLMFI*>(ptr);
	int fulldim = x.length();
	this_obj->make_checks(fulldim);
	const double *xdata = x.getcontent();
	std::vector<double> xin = std::vector<double>(xdata, xdata + fulldim);

	high_resolution_clock::time_point time1 = high_resolution_clock::now(), time_of, time_grad, time_hess;

	f = this_obj->pm_work->ObjFunc(xin);
	time_of = high_resolution_clock::now();

	std::vector<double> gr = this_obj->pm_work->ObjFuncGrad(xin);
	time_grad = high_resolution_clock::now();

	HMMPI::Mat FI = this_obj->pm_work->ObjFuncFisher(xin);				// === Using 2*FI instead of Hess ===
	FI = 2*std::move(FI);
	time_hess = high_resolution_clock::now();

	this_obj->tfunc += duration_cast<duration<double>>(time_of-time1).count();			// update the time stats
	this_obj->tgrad += duration_cast<duration<double>>(time_grad-time_of).count();		// update the time stats
	this_obj->thess += duration_cast<duration<double>>(time_hess-time_grad).count();	// update the time stats

	MPI_Comm comm = this_obj->pm_work->GetComm();		// MPI-synchronisation
	if (comm != MPI_COMM_NULL)
	{
		MPI_Bcast(&f, 1, MPI_DOUBLE, 0, comm);
		HMMPI::Bcast_vector(gr, 0, comm);
		FI.Bcast(0, comm);
	}

	assert(gr.size() == (size_t)fulldim);
	assert(FI.ICount() == (size_t)fulldim && FI.JCount() == (size_t)fulldim);

	real_1d_array g_work;
	g_work.setcontent(fulldim, gr.data());
	g = g_work;

	real_2d_array h_work;
	h_work.setcontent(fulldim, fulldim, FI.Serialize());
	h = h_work;
}
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
void OptLMFImix::fimix(const alglib::real_1d_array &x, double &f, alglib::real_1d_array &g, alglib::real_2d_array &h, void *ptr)
{
	const OptLMFImix *this_obj = static_cast<const OptLMFImix*>(ptr);
	int fulldim = x.length();
	this_obj->make_checks(fulldim);
	const double *xdata = x.getcontent();
	std::vector<double> xin = std::vector<double>(xdata, xdata + fulldim);

	high_resolution_clock::time_point time1 = high_resolution_clock::now(), time_of, time_grad, time_hess;

	f = this_obj->pm_work->ObjFunc(xin);
	time_of = high_resolution_clock::now();

	std::vector<double> gr = this_obj->pm_work->ObjFuncGrad(xin);
	time_grad = high_resolution_clock::now();

	HMMPI::Mat FImix = this_obj->pm_work->ObjFuncFisher_mix(xin);				// === Using FImix instead of Hess ===
	time_hess = high_resolution_clock::now();

	this_obj->tfunc += duration_cast<duration<double>>(time_of-time1).count();			// update the time stats
	this_obj->tgrad += duration_cast<duration<double>>(time_grad-time_of).count();		// update the time stats
	this_obj->thess += duration_cast<duration<double>>(time_hess-time_grad).count();	// update the time stats

	MPI_Comm comm = this_obj->pm_work->GetComm();		// MPI-synchronisation
	if (comm != MPI_COMM_NULL)
	{
		MPI_Bcast(&f, 1, MPI_DOUBLE, 0, comm);
		HMMPI::Bcast_vector(gr, 0, comm);
		FImix.Bcast(0, comm);
	}

	assert(gr.size() == (size_t)fulldim);
	assert(FImix.ICount() == (size_t)fulldim && FImix.JCount() == (size_t)fulldim);

	real_1d_array g_work;
	g_work.setcontent(fulldim, gr.data());
	g = g_work;

	real_2d_array h_work;
	h_work.setcontent(fulldim, fulldim, FImix.Serialize());
	h = h_work;
}
//---------------------------------------------------------------------------
