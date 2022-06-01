/*
 * MonteCarlo.cpp
 *
 *  Created on: 26 Jul 2016
 *      Author: ilya fursov
 */

#include "MonteCarlo.h"
#include "GradientOpt.h"
#include "lapacke_select.h"
#include "mpi.h"
#include <chrono>
#include <cmath>

#define TESTLF
#define TESTMASSMATR "MASS_MATRIX_initial.txt"		// output initial MM to the file; also enable eigenvalue calculation and SVD of MM in LeapFrog::Recalc (for reporting)
#define FULL_SAVE_MCMC_POINT						// if not defined, only params[0] is saved, modelled data, params_all are not saved

namespace HMMPI
{

//------------------------------------------------------------------------------------------
// Rand
//------------------------------------------------------------------------------------------
Rand::Rand(unsigned int s, double a, double b, double mu, double sigma, bool SyncSeed) : seed(s)	// seed,
{																									// parameters for uniform distribution, parameters for normal distribution
	if (seed == 0)																					// if seed == 0, it will be initialized by time
		seed = std::chrono::system_clock::now().time_since_epoch().count();							// if SyncSeed == true on RANK-0, seed gets sync over MPI_COMM_WORLD

	MPI_Bcast(&SyncSeed, 1, MPI_BYTE, 0, MPI_COMM_WORLD);
	if (SyncSeed)
		MPI_Bcast(&seed, 1, MPI_INT, 0, MPI_COMM_WORLD);

	gen = std::default_random_engine(seed);
	uni = std::uniform_real_distribution<double>(a, b);
	norm = std::normal_distribution<double>(mu, sigma);
}
//------------------------------------------------------------------------------------------
double Rand::RandU()
{
	return uni(gen);
}
//------------------------------------------------------------------------------------------
double Rand::RandN()
{
	return norm(gen);
}
//------------------------------------------------------------------------------------------
Mat Rand::RandU(size_t I0, size_t J0)
{
	Mat res(I0, J0, 0);
	std::vector<double> &data = res.ToVectorMutable();
	size_t len = res.Length();

	for (size_t i = 0; i < len; i++)
		data[i] = RandU();

	return res;
}
//------------------------------------------------------------------------------------------
Mat Rand::RandN(size_t I0, size_t J0)
{
	Mat res(I0, J0, 0);
	std::vector<double> &data = res.ToVectorMutable();
	size_t len = res.Length();

	for (size_t i = 0; i < len; i++)
		data[i] = RandN();

	return res;
}
//------------------------------------------------------------------------------------------
// LeapFrog
//------------------------------------------------------------------------------------------
void LeapFrog::BounceVel1(Mat &vel, int ci) const
{
	vel(ci, 0) = -vel(ci, 0);
}
//------------------------------------------------------------------------------------------
void LeapFrog::BounceVel2(Mat &vel, int ci, const Mat &factM) const
{
	Mat Lei(vel.ICount(), 1, 0);
	Lei(ci, 0) = 1;
	Lei = factM / std::move(Lei);		// Lei = L^(-1)*ei, normal for Householder transform

	Mat Lt = factM.Tr();				// L'; here L is the short notation for 'factM'
	Mat w = Lt * vel;					// w = L'*v
	w += ((-2)*InnerProd(w, Lei)/InnerProd(Lei, Lei)) * Lei;	// w = R*w, Householder transform
	vel = std::move(Lt) / std::move(w);		// result = L^(-T)*R*L^T*v
}
//------------------------------------------------------------------------------------------
void LeapFrog::BounceVel3(Mat &vel, int ci) const
{
	Mat Mei(M.ICount(), 1, 0.0);
	Mei(ci, 0) = 1;
	Mei = M / std::move(Mei);			// M^(-1)*ei -- ci-th column of M^(-1)

	vel += (-2*vel(ci, 0)/Mei(ci, 0)) * Mei;
}
//------------------------------------------------------------------------------------------
const Mat &LeapFrog::SymSqrtM() const
{
	if (!ssM_cached)
	{
		ssM = M.SymSqrt();
		ssM_cached = true;
	}

	return ssM;
}
//------------------------------------------------------------------------------------------
void LeapFrog::comm_compatible(MPI_Comm c, MPI_Comm pm_comm, std::string where)
{
	int size_w, rank_w;
	MPI_Comm_size(MPI_COMM_WORLD, &size_w);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank_w);

	int rank_c = -1, rank_pm = -1;
	if (c != MPI_COMM_NULL)
		MPI_Comm_rank(c, &rank_c);
	if (pm_comm != MPI_COMM_NULL)
		MPI_Comm_rank(pm_comm, &rank_pm);

	int err = 0;
	if (rank_c == 0 && rank_pm != 0)
		err = 1;
	if (rank_c == -1 && rank_pm != -1)
		err = 1;

	std::vector<int> Errors(size_w);
	MPI_Gather(&err, 1, MPI_INT, Errors.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);		// collect all error reports

	if (rank_w == 0)		// analyse error report
	{
		for (const auto &i : Errors)
			if (i == 1)
			{
				err = 1;
				break;
			}
	}

	MPI_Bcast(&err, 1, MPI_INT, 0, MPI_COMM_WORLD);		// Bcast error analysis
	if (err == 1)
	{
		std::string msg = MessageRE("Несовместимые коммуникаторы в ", "Incompatible communicators in ");
		msg += where + ":\n" + MPI_Ranks(std::vector<MPI_Comm>{c, pm_comm});
		throw Exception(msg);
	}
}
//------------------------------------------------------------------------------------------
LeapFrog::LeapFrog(MPI_Comm c, double n, double e, double gamma, std::string bounce, std::string mm, Mat m) :
	ssM_cached(false), is_dummy(false), comm(c), nu(n), bounce_type(bounce), MM_type(mm), const_mat(std::move(m)), dump_flag(-1), gammaBFGS(gamma), eps(e), eig_min(0), cond2(0), m_adj(0)
{
};
//------------------------------------------------------------------------------------------
void LeapFrog::Recalc(PhysModel *pm, const Mat &x)
{
	if (is_dummy)			// dummy object immediately quits
		return;

	comm_compatible(comm, pm->GetComm(), "LeapFrog");

	if (comm == MPI_COMM_NULL)
		return;

	int rank;
	MPI_Comm_rank(comm, &rank);

	Mat MM_work;
	if (MM_type == "HESS")
		MM_work = pm->ObjFuncHess_ACT(x.ToVector());
	else if (MM_type == "FI")
		MM_work = pm->ObjFuncFisher_ACT(x.ToVector());
	else if (MM_type == "UNITY")
		MM_work = Mat(x.ICount());
	else if (MM_type == "BFGS")
	{
		if (rank == 0)
		{
			if (Xk.size() == 0 || Gk.size() == 0)
			{
				//Hk = (1/gammaBFGS)*Mat(x.ICount());		// initial call
				Bk = gammaBFGS*Mat(x.ICount());
				//Sk = sqrt(1/gammaBFGS)*Mat(x.ICount());
				Ck = sqrt(gammaBFGS)*Mat(x.ICount());
			}
			else											// subsequent calls
			{

				// DEBUG: reset
				Bk = gammaBFGS*Mat(x.ICount());	// DEBUG
				Ck = sqrt(gammaBFGS)*Mat(x.ICount());	// DEBUG

//				Bk = gammaBFGS*Bk;	// DEBUG			-- damping!
//				Ck = sqrt(gammaBFGS)*Ck;	// DEBUG
				// DEBUG: reset



				if (Xk.size() != Gk.size())
					throw Exception("Xk.size() != Gk.size() in LeapFrog::Recalc");

				for (size_t i = 1; i < Xk.size(); i++)
				{
					Mat sk = Xk[i] - Xk[i-1];
					Mat yk = Gk[i] - Gk[i-1];

					double s_y = InnerProd(sk, yk);
					//if (s_y > 0)			// only take these steps to maintain Hk > 0			TODO original!
					if (s_y > 0.01*sk.Norm2()*yk.Norm2())	// DEBUG
					{
						Mat B_s = Bk*sk;
						double s_B_s = InnerProd(sk, B_s);

						//Mat pk = (1/s_y)*sk;
						Mat tk = (1/s_B_s)*sk;
						//Mat qk = sqrt(s_y/s_B_s)*B_s + yk;
						Mat uk = sqrt(s_B_s/s_y)*yk + B_s;

						//Sk -= OuterProd(pk, Sk.Tr()*qk);
						Ck -= OuterProd(uk, Ck.Tr()*tk);
						//Hk = Sk*(Sk.Tr());
						Bk = Ck*(Ck.Tr());
					}
				}
				resetBFGSVecs();
			}
			MM_work = Bk;
		}
	}
	else if (MM_type == "MAT")
	{
		MM_work = const_mat;
		if (MM_work.ICount() != x.ICount() || MM_work.JCount() != x.ICount())
			throw Exception(stringFormatArr(MessageRE("Размеры матрицы массы MAT {0:%d} x {1:%d} не соответствуют активному вектору {2:%d}\n",
													  "Size of mass matrix MAT {0:%d} x {1:%d} is not consistent with the active vector size {2:%d}\n"), std::vector<int>{(int)MM_work.ICount(), (int)MM_work.JCount(), (int)x.ICount()}));
	}
	else
		throw Exception("Unrecognized mass matrix type " + MM_type + " in LeapFrog::Recalc");

	// adjust spectrum of MM_work
	if (rank == 0)
	{
		m_adj = 0;			// used only for reporting
		if (nu > 0)
		{
			std::vector<double> eig = MM_work.EigVal(0, 1);	// one (smallest) eigenvalue
			double lmin = eig[0];
			if (lmin < nu)
			{
				double d = lmin - nu;
				if (d == lmin)								// fix numerical issue (in case 'lmin' is big, 'nu' is small)
					d = lmin*(1 - LAPACKE_dlamch('P'));

				M = MM_work - d * Mat(MM_work.ICount());			// M = MM_work - (lmin - nu)*I
				m_adj = -d;
			}
			else
				M = std::move(MM_work);
		}
		else
			M = std::move(MM_work);

		// 'eig_min' and 'cond2' are only used for reporting
#ifdef TESTMASSMATR			// eigenvalue calculation and SVD take time, turn them off if necessary
		std::vector<double> eig = M.EigVal(0, 1);
		eig_min = eig[0];
		std::vector<double> sg = M.SgVal();
		cond2 = sg[0] / (*--sg.end());
#endif

		if (MM_type == "BFGS" && nu == 0)
		{
			L = Ck;		// ORIGINAL
			//L = M.Chol();
			//L = M.SymSqrt();
		}
		else
			L = M.Chol();
		ssM_cached = false;

		//std::cout << "M\n" << M.ToString() << "\n";	// DEBUG
	}
}
//------------------------------------------------------------------------------------------
void LeapFrog::updateBFGSVecs(const Mat &x, const Mat &g) const
{
	if (comm == MPI_COMM_NULL)
		return;

	int rank;
	MPI_Comm_rank(comm, &rank);

	if (rank == 0 && MM_type == "BFGS")
	{
		Xk.push_back(x);
		Gk.push_back(g);
	}
}
//------------------------------------------------------------------------------------------
void LeapFrog::resetBFGSVecs() const
{
	std::cout << "Reset BFGS vectors of length " << Xk.size() << "\n";	// DEBUG

	Xk.clear();
	Gk.clear();
}
//------------------------------------------------------------------------------------------
int LeapFrog::Run1(PhysModel *pm, Mat &x, Mat &p, int N) const
{
	comm_compatible(comm, pm->GetComm(), "LeapFrog");

	if (comm == MPI_COMM_NULL)
		return 0;

	int rank, step_count = 0;
	MPI_Comm_rank(comm, &rank);

	Mat grad(pm->ObjFuncGrad_ACT(x.ToVector()));			// 'grad' may be defined only on pm->comm-RANKS-0 (hence, on comm-RANKS-0), but the function is called on ALL RANKS
	grad = 0.5 * std::move(grad);
	Mat x_prev = x;											// 'x' and 'x_prev' are sync between ranks
    for (int i = 0; i < N; i++)
    {
    	if (rank == 0)
    	{
			p -= (eps/2) * grad;
			x += M / (eps*p);
    	}

    	x.Bcast(0, comm);
        if (!pm->CheckLimits_ACT(x.ToVector()))		// check the bounds for the parameters
        {
        	x = x_prev;								// 'x_prev' is within the limits
        	break;
        }
        grad = pm->ObjFuncGrad_ACT(x.ToVector());
        grad = 0.5 * std::move(grad);

        if (rank == 0)
			p -= (eps/2) * grad;			// 'grad' will be reused

        step_count++;
        x_prev = x;
    }
    p.Bcast(0, comm);

    return step_count;
}
//------------------------------------------------------------------------------------------
int LeapFrog::Run2(PhysModel *pm, Mat &x, Mat &p, int N, double &dr) const
{
	comm_compatible(comm, pm->GetComm(), "LeapFrog");

	if (comm == MPI_COMM_NULL)
		return 0;

	int RNK;
	MPI_Comm_rank(MPI_COMM_WORLD, &RNK);
	FILE *f0 = NULL;						// used for debug dumping
	if (RNK == 0 && dump_flag != -1)
	{
		char fname[100];
		sprintf(fname, dump_file, dump_flag);
		f0 = fopen(fname, "w");
	}

	HMMPI::Mat x_init = x;					// sync on 'comm'
	double dist = 0;
	double max_dist = 0;

	int rank, tot_bounce_count = 0;
	MPI_Comm_rank(comm, &rank);

	Mat grad(pm->ObjFuncGrad_ACT(x.ToVector()));			// 'grad' may be defined only on pm->comm-RANKS-0 (hence, on comm-RANKS-0), but the function is called on ALL RANKS
	//updateVecs(x, grad);		// don't add this 'x' as it may coincide with the last 'x' from previous Run2()
	grad = 0.5 * std::move(grad);
	Mat x_prev = x;											// 'x' and 'x_prev' are sync between ranks; 'x_prev' is always within the bounds
	if (RNK == 0 && dump_flag != -1)
	{
		fprintf(f0, "x\t"); x.Tr().SaveASCII(f0, "%20.16g");
		fprintf(f0, "p\t"); p.Tr().SaveASCII(f0, "%20.16g");
		fprintf(f0, "grad/2\t"); grad.Tr().SaveASCII(f0, "%20.16g");
		fprintf(f0, "\n");
	}

    for (int i = 0; i < N; i++)
    {
    	int bounce_count = 0;				// counts bounces of 'x' from the boundaries for the current leapfrog step
    	if (rank == 0)
    	{
			p -= (eps/2) * grad;
			if (RNK == 0 && dump_flag != -1)
			{
				fprintf(f0, "p%d_a\t", i); p.Tr().SaveASCII(f0, "%20.16g");
			}

			bool bounces_done = false;		// will become 'true' after the potential bounces are complete, and 'x' is within the bounds
			double eps0 = eps;
			Mat vel = M/p;					// velocity vector M^(-1)*p
			if (RNK == 0 && dump_flag != -1)
			{
				fprintf(f0, "vel%d\t", i); vel.Tr().SaveASCII(f0, "%20.16g");
			}

			while (!bounces_done)
			{
				x += eps0 * vel;

				std::vector<double> xint;	// intersection point
				int con_ind;				// index of the constraint which gave the bounce
				double alpha;				// fraction of step "eps" made till the bounce
				if (!pm->FindIntersect_ACT(x_prev.ToVector(), x.ToVector(), xint, alpha, con_ind))	// bounds violated
				{
					x = xint;								// move to the intersection point
					x_prev = x;
					eps0 *= 1 - alpha;						// update the remaining step length

					if (bounce_type == "NEG")				// bounce the velocity vector
						BounceVel1(vel, con_ind);
					else if (bounce_type == "CHOL")
						BounceVel2(vel, con_ind, L);
					else if (bounce_type == "EIG")
						BounceVel2(vel, con_ind, SymSqrtM());
					else if (bounce_type == "HT")
						BounceVel3(vel, con_ind);
					else
					{
						if (f0 != NULL)
							fclose(f0);
						throw Exception("Unrecognized bounce_type '" + bounce_type + "' in LeapFrog::Run2");
					}

					p = M*vel;								// update the momentum vector after bounce
					bounce_count++;

					if (RNK == 0 && dump_flag != -1)
					{
						fprintf(f0, "x_bnc%d\t", i); x.Tr().SaveASCII(f0, "%20.16g");
						fprintf(f0, "vel_bnc%d\t", i); vel.Tr().SaveASCII(f0, "%20.16g");
						fprintf(f0, "p_bnc%d\t", i); p.Tr().SaveASCII(f0, "%20.16g");
					}
				}
				else						// bounds ok -- accept the move to 'x'
					bounces_done = true;

				if (RNK == 0 && dump_flag != -1)
				{
					fprintf(f0, "x_new%d\t", i); x.Tr().SaveASCII(f0, "%20.16g");
				}
			}
    	}

    	x.Bcast(0, comm);
    	MPI_Bcast(&bounce_count, 1, MPI_INT, 0, comm);

        grad = pm->ObjFuncGrad_ACT(x.ToVector());
#ifdef HMC_BFGS
        updateBFGSVecs(x, grad);
#endif

        grad = 0.5 * std::move(grad);
        if (rank == 0)
			p -= (eps/2) * grad;			// 'grad' will be reused

        // TODO if last (=output) x is accepted in the MH test, then its grad can be reused in the next Run2()

		if (RNK == 0 && dump_flag != -1)
		{
			fprintf(f0, "p%d_b\t", i); p.Tr().SaveASCII(f0, "%20.16g");
			fprintf(f0, "grad/2\t"); grad.Tr().SaveASCII(f0, "%20.16g");
			fprintf(f0, "\n");
		}

        tot_bounce_count += bounce_count;
        x_prev = x;

        dist = (x - x_init).Norm2();		// update the distances
        if (dist > max_dist)
        	max_dist = dist;
    }
    p.Bcast(0, comm);
    dr = dist / max_dist;

    if (f0 != NULL)
    	fclose(f0);

    return tot_bounce_count;
}
//------------------------------------------------------------------------------------------
int LeapFrog::Run_SOL(PhysModel *pm, Mat &x, Mat &v, int N, double &dr) const	// Hamiltonian integration for SOL-HMC (not a leap frog!).
{															// "pm" should be the proxy model; x - coordinate vector, v - velocity vector, N - number of steps (of size "eps"), dr - output distance ratio
															// "M" is used as covariance (see SOL-HMC definition)
															// All other comments from Run2() apply. The algorithm is allowed to leave the boundary.
	comm_compatible(comm, pm->GetComm(), "LeapFrog SOL-HMC");

	if (comm == MPI_COMM_NULL)
		return 0;

	int RNK;
	MPI_Comm_rank(MPI_COMM_WORLD, &RNK);
	FILE *f0 = NULL;						// used for debug dumping
	if (RNK == 0 && dump_flag != -1)
	{
		char fname[100];
		sprintf(fname, dump_file, dump_flag);
		f0 = fopen(fname, "w");
	}

	HMMPI::Mat x_init = x;					// sync on 'comm'
	double dist = 0;
	double max_dist = 0;
	const int dim = x.ICount();

	int rank;
	MPI_Comm_rank(comm, &rank);

	Mat grad(pm->ObjFuncGrad_ACT(x.ToVector()));			// 'grad' may be defined only on pm->comm-RANKS-0 (hence, on comm-RANKS-0), but the function is called on ALL RANKS
	grad = 0.5 * std::move(grad);
	if (RNK == 0 && dump_flag != -1)
	{
		fprintf(f0, "x\t"); x.Tr().SaveASCII(f0, "%20.16g");
		fprintf(f0, "v\t"); v.Tr().SaveASCII(f0, "%20.16g");
		fprintf(f0, "grad/2\t"); grad.Tr().SaveASCII(f0, "%20.16g");
		fprintf(f0, "\n");
	}

    for (int i = 0; i < N; i++)
    {
    	if (rank == 0)
    	{
			v -= (eps/2) * M * grad;						// M is the prior covariance here
			if (RNK == 0 && dump_flag != -1)
			{
				fprintf(f0, "v%d_a\t", i); v.Tr().SaveASCII(f0, "%20.16g");
			}

			// rotations
			for (int k = 0; k < dim; k++)
			{
				double xk = x(k,0);
				double vk = v(k,0);
				double rk = sqrt(xk*xk + vk*vk);
				double ak = atan2(-vk, xk);
				x(k,0) = rk*cos(ak + eps);
				v(k,0) = -rk*sin(ak + eps);
			}

			if (RNK == 0 && dump_flag != -1)
			{
				fprintf(f0, "x_rot%d\t", i); x.Tr().SaveASCII(f0, "%20.16g");
				fprintf(f0, "v_rot%d\t", i); v.Tr().SaveASCII(f0, "%20.16g");
			}
    	}

    	x.Bcast(0, comm);
        grad = pm->ObjFuncGrad_ACT(x.ToVector());
        grad = 0.5 * std::move(grad);
        if (rank == 0)
			v -= (eps/2) * M * grad;						// 'grad' will be reused

		if (RNK == 0 && dump_flag != -1)
		{
			fprintf(f0, "v%d_b\t", i); v.Tr().SaveASCII(f0, "%20.16g");
			fprintf(f0, "grad/2\t"); grad.Tr().SaveASCII(f0, "%20.16g");
			fprintf(f0, "\n");
		}

        dist = (x - x_init).Norm2();						// update the distances
        if (dist > max_dist)
        	max_dist = dist;
    }
    v.Bcast(0, comm);
    dr = dist / max_dist;

    if (f0 != NULL)
    	fclose(f0);

    return 0;
}
//------------------------------------------------------------------------------------------
int LeapFrog::Run_SOL2(PhysModel *pm, Mat &x, Mat &v, int N, double &dr) const	// Same as Run_SOL, but with bounces on the boundaries. Returns the number of bounces.
{
	comm_compatible(comm, pm->GetComm(), "LeapFrog SOL-HMC");

	const std::vector<double> min = pm->GetConstr()->actmin();		// bounds for quick access
	const std::vector<double> max = pm->GetConstr()->actmax();
	assert(x.ICount() == min.size() && x.ICount() == max.size());	// TODO check N/A

	int count_refl = 0;

	if (comm == MPI_COMM_NULL)
		return 0;

	int RNK;
	MPI_Comm_rank(MPI_COMM_WORLD, &RNK);
	FILE *f0 = NULL;						// used for debug dumping
	if (RNK == 0 && dump_flag != -1)
	{
		char fname[100];
		sprintf(fname, dump_file, dump_flag);
		f0 = fopen(fname, "w");
	}

	HMMPI::Mat x_init = x;					// sync on 'comm'
	double dist = 0;
	double max_dist = 0;
	const int dim = x.ICount();

	int rank;
	MPI_Comm_rank(comm, &rank);

	Mat grad(pm->ObjFuncGrad_ACT(x.ToVector()));			// 'grad' may be defined only on pm->comm-RANKS-0 (hence, on comm-RANKS-0), but the function is called on ALL RANKS
	grad = 0.5 * std::move(grad);
	if (RNK == 0 && dump_flag != -1)
	{
		fprintf(f0, "x\t"); x.Tr().SaveASCII(f0, "%20.16g");
		fprintf(f0, "v\t"); v.Tr().SaveASCII(f0, "%20.16g");
		fprintf(f0, "grad/2\t"); grad.Tr().SaveASCII(f0, "%20.16g");
		fprintf(f0, "\n");
	}

    for (int i = 0; i < N; i++)
    {
    	if (rank == 0)
    	{
			v -= (eps/2) * M * grad;
			if (RNK == 0 && dump_flag != -1)
			{
				fprintf(f0, "v%d_a\t", i); v.Tr().SaveASCII(f0, "%20.16g");
			}

			// rotations =======================================================-----------
			bool complete = false;
			double step = eps;				// remaining step
			while (!complete)
			{
				const Mat x_backup = x;
				const Mat v_backup = v;

				// trial step, possibly violating the bounds; {x, v} become updated
				double min_step_taken = step;
				int min_step_ind = -1;
				for (int k = 0; k < dim; k++)
				{
					const double xk = x(k,0);
					const double vk = v(k,0);
					double rk = sqrt(xk*xk + vk*vk);
					double ak = atan2(-vk, xk);
					x(k,0) = rk*cos(ak + step);
					v(k,0) = -rk*sin(ak + step);

					double step_taken_k = step;
					if (x(k,0) < min[k])	// min bound violation
					{
						double ac = acos(min[k]/rk);
						if (ac-ak < 0 || ac-ak > step)
							ac = -ac;		// pick the correct solution
						step_taken_k = ac - ak;
						assert(step_taken_k > 0 && step_taken_k <= step);
					}
					if (x(k,0) > max[k])	// max bound violation
					{
						double ac = acos(max[k]/rk);
						if (ac-ak < 0 || ac-ak > step)
							ac = -ac;		// pick the correct solution
						step_taken_k = ac - ak;
						assert(step_taken_k > 0 && step_taken_k <= step);
					}
					if (step_taken_k < min_step_taken)
					{
						min_step_taken = step_taken_k;
						min_step_ind = k;
					}
				}

				if (min_step_ind != -1)				// perform a shorter step (min_step_taken) and bounce from the boundary
				{
					x = x_backup;
					v = v_backup;

					for (int k = 0; k < dim; k++)
					{
						const double xk = x(k,0);
						const double vk = v(k,0);
						double rk = sqrt(xk*xk + vk*vk);
						double ak = atan2(-vk, xk);
						x(k,0) = rk*cos(ak + min_step_taken);
						v(k,0) = -rk*sin(ak + min_step_taken);
					}
					v(min_step_ind, 0) = -v(min_step_ind, 0);	// reflection
					count_refl++;

					step -= min_step_taken;			// continue with the shortened step length after reflection

					if (RNK == 0 && dump_flag != -1)
					{
						fprintf(f0, "** Made step %g out of %g, reflection for %d\n", min_step_taken, eps, min_step_ind);
						fprintf(f0, "x_refl%d\t", i); x.Tr().SaveASCII(f0, "%20.16g");
						fprintf(f0, "v_refl%d\t", i); v.Tr().SaveASCII(f0, "%20.16g");
					}
				}
				else
					complete = true;				// exit with x(k,0), v(k,0)
			}
			// end of rotations =======================================================-----------

			if (RNK == 0 && dump_flag != -1)
			{
				fprintf(f0, "x_rot%d\t", i); x.Tr().SaveASCII(f0, "%20.16g");
				fprintf(f0, "v_rot%d\t", i); v.Tr().SaveASCII(f0, "%20.16g");
			}
    	}

    	x.Bcast(0, comm);
        grad = pm->ObjFuncGrad_ACT(x.ToVector());
        grad = 0.5 * std::move(grad);
        if (rank == 0)
			v -= (eps/2) * M * grad;						// 'grad' will be reused

		if (RNK == 0 && dump_flag != -1)
		{
			fprintf(f0, "v%d_b\t", i); v.Tr().SaveASCII(f0, "%20.16g");
			fprintf(f0, "grad/2\t"); grad.Tr().SaveASCII(f0, "%20.16g");
			fprintf(f0, "\n");
		}

        dist = (x - x_init).Norm2();						// update the distances
        if (dist > max_dist)
        	max_dist = dist;
    }
    v.Bcast(0, comm);
    MPI_Bcast(&count_refl, 1, MPI_INT, 0, comm);
    dr = dist / max_dist;

    if (f0 != NULL)
    	fclose(f0);

    return count_refl;
}
//------------------------------------------------------------------------------------------
int LeapFrog::RunHor2(PhysModel *pm, Mat &x, Mat &p, int N, double &dr) const	// Same as Run2, but is designed for Horowitz-type sampler
{
	comm_compatible(comm, pm->GetComm(), "LeapFrog");

	if (comm == MPI_COMM_NULL)
		return 0;

	int RNK;
	MPI_Comm_rank(MPI_COMM_WORLD, &RNK);
	FILE *f0 = NULL;						// used for debug dumping
	if (RNK == 0 && dump_flag != -1)
	{
		char fname[100];
		sprintf(fname, dump_file, dump_flag);
		f0 = fopen(fname, "w");
	}

	HMMPI::Mat x_init = x;					// sync on 'comm'
	double dist = 0;
	double max_dist = 0;

	int rank, tot_bounce_count = 0;
	MPI_Comm_rank(comm, &rank);

	Mat grad(pm->ObjFuncGrad_ACT(x.ToVector()));			// 'grad' may be defined only on pm->comm-RANKS-0 (hence, on comm-RANKS-0), but the function is called on ALL RANKS
	grad = 0.5 * std::move(grad);
	Mat x_prev = x;											// 'x' and 'x_prev' are sync between ranks; 'x_prev' is always within the bounds
	if (RNK == 0 && dump_flag != -1)
	{
		fprintf(f0, "x\t"); x.Tr().SaveASCII(f0, "%20.16g");
		fprintf(f0, "p\t"); p.Tr().SaveASCII(f0, "%20.16g");
		fprintf(f0, "grad/2\t"); grad.Tr().SaveASCII(f0, "%20.16g");
		fprintf(f0, "\n");
	}

    for (int i = 0; i < N; i++)
    {
    	int bounce_count = 0;				// counts bounces of 'x' from the boundaries for the current leapfrog step
    	if (rank == 0)
    	{
			p -= (eps/2) * ((M*grad) + x);		// added M and x here
			if (RNK == 0 && dump_flag != -1)
			{
				fprintf(f0, "p%d_a\t", i); p.Tr().SaveASCII(f0, "%20.16g");
			}

			bool bounces_done = false;		// will become 'true' after the potential bounces are complete, and 'x' is within the bounds
			double eps0 = eps;

			while (!bounces_done)
			{
				x += eps0 * p;

				std::vector<double> xint;	// intersection point
				int con_ind;				// index of the constraint which gave the bounce
				double alpha;				// fraction of step "eps" made till the bounce
				if (!pm->FindIntersect_ACT(x_prev.ToVector(), x.ToVector(), xint, alpha, con_ind))	// bounds violated
				{
					x = xint;								// move to the intersection point
					x_prev = x;
					eps0 *= 1 - alpha;						// update the remaining step length

					BounceVel1(p, con_ind);
					bounce_count++;

					if (RNK == 0 && dump_flag != -1)
					{
						fprintf(f0, "x_bnc%d\t", i); x.Tr().SaveASCII(f0, "%20.16g");
						fprintf(f0, "p_bnc%d\t", i); p.Tr().SaveASCII(f0, "%20.16g");
					}
				}
				else						// bounds ok -- accept the move to 'x'
					bounces_done = true;

				if (RNK == 0 && dump_flag != -1)
				{
					fprintf(f0, "x_new%d\t", i); x.Tr().SaveASCII(f0, "%20.16g");
				}
			}
    	}

    	x.Bcast(0, comm);
    	MPI_Bcast(&bounce_count, 1, MPI_INT, 0, comm);

        grad = pm->ObjFuncGrad_ACT(x.ToVector());
        grad = 0.5 * std::move(grad);
        if (rank == 0)
			p -= (eps/2) * ((M*grad) + x);		// 'grad' will be reused; added M and x here


		if (RNK == 0 && dump_flag != -1)
		{
			fprintf(f0, "p%d_b\t", i); p.Tr().SaveASCII(f0, "%20.16g");
			fprintf(f0, "grad/2\t"); grad.Tr().SaveASCII(f0, "%20.16g");
			fprintf(f0, "\n");
		}

        tot_bounce_count += bounce_count;
        x_prev = x;

        dist = (x - x_init).Norm2();		// update the distances
        if (dist > max_dist)
        	max_dist = dist;
    }
    p.Bcast(0, comm);
    dr = dist / max_dist;

    if (f0 != NULL)
    	fclose(f0);

    return tot_bounce_count;
}
//------------------------------------------------------------------------------------------
//// TEST !
// testing whether it's possible to regularly replace MM - after paper by Fu, Luo, Zhang.
// Rosenbrock 2D showed -- no, not working!
//
//int LeapFrog::Run2(PhysModel *pm, Mat &x, Mat &p, int N, double &dr) const		// TEST VERSION !!!!!!!!!!!!!
//{
//	comm_compatible(comm, pm->GetComm(), "LeapFrog");
//	if (comm == MPI_COMM_NULL)
//		return 0;
//
//	const HMMPI::Mat C = pm->ObjFuncFisher_ACT(x.ToVector()).InvSY().SymSqrt();		// NEW!
//	//HMMPI::Mat C = pm->ObjFuncHess_ACT(x.ToVector());		// NEW!
//
//	HMMPI::Mat x_init = x;					// sync on 'comm'
//	double dist = 0;
//	double max_dist = 0;
//
//	int rank, tot_bounce_count = 0;
//	MPI_Comm_rank(comm, &rank);
//
//	Mat grad(pm->ObjFuncGrad_ACT(x.ToVector()));			// 'grad' may be defined only on pm->comm-RANKS-0 (hence, on comm-RANKS-0), but the function is called on ALL RANKS
//	grad = 0.5 * std::move(grad);
//	Mat x_prev = x;											// 'x' and 'x_prev' are sync between ranks; 'x_prev' is always within the bounds
//
//    for (int i = 0; i < N; i++)
//    {
//    	int bounce_count = 0;				// counts bounces of 'x' from the boundaries for the current leapfrog step
//    	if (rank == 0)
//    	{
//			p -= (eps/2) * (C*grad);
//
//			bool bounces_done = false;		// will become 'true' after the potential bounces are complete, and 'x' is within the bounds
//			double eps0 = eps;
//			Mat vel = C*(M/p);					// velocity vector M^(-1)*p
//
//			while (!bounces_done)
//			{
//				x += eps0 * vel;
//
//				std::vector<double> xint;	// intersection point
//				int con_ind;				// index of the constraint which gave the bounce
//				double alpha;				// fraction of step "eps" made till the bounce
//				if (!pm->FindIntersect_ACT(x_prev.ToVector(), x.ToVector(), xint, alpha, con_ind))	// bounds violated
//				{
//					x = xint;								// move to the intersection point
//					x_prev = x;
//					eps0 *= 1 - alpha;						// update the remaining step length
//
//					//BounceVel3(vel, con_ind);
//					//void LeapFrog::BounceVel3(Mat &vel, int ci) const
//					{
//						Mat Mei(M.ICount(), 1, 0.0);
//						Mei(con_ind, 0) = 1;
//						Mei = C*(M / std::move(Mei));			// M^(-1)*ei -- ci-th column of M^(-1)
//
//						vel += (-2*vel(con_ind, 0)/Mei(con_ind, 0)) * Mei;
//					}
//
//					p = M*(C/vel);								// update the momentum vector after bounce
//					bounce_count++;
//				}
//				else						// bounds ok -- accept the move to 'x'
//					bounces_done = true;
//			}
//    	}
//
//    	x.Bcast(0, comm);
//    	MPI_Bcast(&bounce_count, 1, MPI_INT, 0, comm);
//
//        grad = pm->ObjFuncGrad_ACT(x.ToVector());
//        grad = 0.5 * std::move(grad);
//        if (rank == 0)
//			p -= (eps/2) * (C*grad);			// 'grad' will be reused
//
//
//        tot_bounce_count += bounce_count;
//        x_prev = x;
//
//        dist = (x - x_init).Norm2();		// update the distances
//        if (dist > max_dist)
//        	max_dist = dist;
//    }
//    p.Bcast(0, comm);
//    dr = dist / max_dist;
//
//    return tot_bounce_count;
//}
//------------------------------------------------------------------------------------------
// LeapFrogGeneralized
//------------------------------------------------------------------------------------------
Mat LeapFrogGeneralized::make_step_p0(const Mat &x, const Mat &p, double eps, const Mat *init0)		// p(tau) -> p(tau + eps/2), implicit equation; if init0 != NULL, it is always used as initial point
{
	Ham0.G.MsgToFile("***make_step_p0***\n");
	VM_Ham_eq1 VM1(&Ham0);
	VM1.x = x.ToVector();
	VM1.p0 = p.ToVector();
	VM1.eps = eps;

	Mat init = p;
	if (init0 != NULL)
		init = *init0;
	else if (solver_init_pt == 2)
		init = Mat(VM1.Func_ACT(p.ToVector())) + p;

	solver->exc_msg = "\nmake_step_p0, eps = " + stringFormatArr("{0:%g}", std::vector<double>{VM1.eps}) +
					  "\nx = " + ToString(VM1.x) + "p0 = " + ToString(VM1.p0) + "init = " + ToString(init.ToVector());
	solver->SetFuncFromVM(&VM1);
	solver->rescale_eps(Min(1, eps/scale_factor_eps));
	std::vector<double> res = solver->Solve(init.ToVector());
	iter_count += solver->iter;

	return res;
}
//------------------------------------------------------------------------------------------
Mat LeapFrogGeneralized::make_step_p0_ref(const Mat &x, const Mat &p, double eps)	// p(tau) -> p(tau + eps/2), implicit equation; in case of failure init points from refined stepping are tried
{
	Ham0.G.MsgToFile("-----------LeapFrogGeneralized, make_step_p0_ref\n");
	int RNK;
	MPI_Comm_rank(MPI_COMM_WORLD, &RNK);

	const Mat *init0 = NULL;
	Mat init_ref;					// init point from refined stepping
	Mat res;

	bool sol_ok = false;			// main loop controller; main loop exits when solution is found, or when refinement limit is exceeded
	bool init_ok = true;			// TRUE if init point can be used
	int c = 0;						// refinement degree
	size_t Nref = 1;				// number of fine steps
	while (!sol_ok)
	{
		bool error = false;			// TRUE if solver fails to find appropriate point (which then leads to refinement)
		if (init_ok)				// init point - good, try solving
		{
			try
			{
				res = make_step_p0(x, p, eps, init0);			// p(tau) -> p(tau + eps/2), implicit equation with full step 'eps'
			}
			catch (const std::exception &e)
			{
				if (RNK == 0)
					std::cout << "[make_step_p0_ref] solver failure: " << e.what() << "\n";
				error = true;						// if solver fails, refine
			}
		}
		else
			error = true;			// init point not good

		if (error)					// no suitable solution found -> find another init point, using refinement
		{
			Nref *= 2;
			if (c >= max_refine)
				break;

			try
			{
				init_ref = make_Nstep_p0(x, p, eps/Nref, Nref);		// make a new init point; inner solver failures will result in rejecting the init point
				init0 = &init_ref;
				init_ok = true;
			}
			catch (const std::exception &e)
			{
				if (RNK == 0)
					std::cout << "[make_step_p0_ref] failure to find 'init' point with refinement " << Nref << "\n" << e.what() << "\n";
				init_ok = false;
			}
			c++;					// increment refinements count
		}
		else
			sol_ok = true;			// appropriate solution found!
	}

	if (!sol_ok)
		throw Exception("LeapFrogGeneralized::make_step_p0_ref failed");

	if (RNK == 0 && c > 0)
		std::cout << "[make_step_p0_ref] SOLUTION FOUND after " << Nref << " refinements, sol-n = " << ToString(res.ToVector()) << "\n";

	return res;
}
//------------------------------------------------------------------------------------------
Mat LeapFrogGeneralized::make_step_x(const Mat &x, const Mat &p, double eps, const Mat *init0)		// x(tau) -> x(tau + eps), implicit equation, no boundary bouncing; if init0 != NULL, it is always used as initial point
{
	Ham0.G.MsgToFile("---make_step_x---\n");
	VM_Ham_eq2 VM2(&Ham0, &Ham1);
	VM2.x0 = x.ToVector();
	VM2.p = p.ToVector();
	VM2.eps = eps;

	Mat init = x;
	if (init0 != NULL)
		init = *init0;
	else if (solver_init_pt == 2)
		init = Mat(VM2.Func_ACT(x.ToVector())) + x;

	solver->exc_msg = "\nmake_step_x, eps = " + stringFormatArr("{0:%g}", std::vector<double>{VM2.eps}) +
					  "\nx0 = " + ToString(VM2.x0) + "p = " + ToString(VM2.p) + "init = " + ToString(init.ToVector());
	solver->SetFuncFromVM(&VM2);
	solver->rescale_eps(Min(1, eps/scale_factor_eps));
	std::vector<double> res = solver->Solve(init.ToVector());
	iter_count += solver->iter;

	return res;
}
//------------------------------------------------------------------------------------------
Mat LeapFrogGeneralized::make_step_x_ref(const Mat &x, const Mat &p, double eps, int minref)		// x(tau) -> x(tau + eps), no boundary bouncing; in case of failure, init points from refined stepping are tried; 'minref' forces to make at least this refinement for init point
{
	Ham0.G.MsgToFile("-----------LeapFrogGeneralized, make_step_x_ref\n");
	int RNK;
	MPI_Comm_rank(MPI_COMM_WORLD, &RNK);

	Mat res;
	bool sol_ok = false;			// main loop exits when solution is found (sol_ok == true)
	size_t Nref = 1;				// number of fine steps

	for (int c = 0; c < minref; c++)	// Nref = 2^minref
		Nref *= 2;

	Mat init_ref;					// init point from refined stepping
	for (int c = minref; c < max_refine; c++)
	{
		bool init_ok = true;		// TRUE if init point can be used
		const Mat *init0 = NULL;

		if (c > 0)
		{
			try
			{
				init_ref = make_Nstep_x(x, p, eps/Nref, Nref);		// make a new init point; inner solver failures will result in rejecting the init point
				init0 = &init_ref;
				init_ok = true;
			}
			catch (const std::exception &e)
			{
				if (RNK == 0)
					std::cout << "[make_step_x_ref] failure to find 'init' point with refinement " << Nref << "\n" << e.what() << "\n";
				init_ok = false;
			}
		}

		if (init_ok)				// init point - good, try solving
		{
			try
			{
				res = make_step_x(x, p, eps, init0);			// x(tau) -> x(tau + eps), implicit equation with full step 'eps'
				sol_ok = true;		// successful solution
			}
			catch (const std::exception &e)
			{
				if (RNK == 0)
					std::cout << "[make_step_x_ref] solver failure: " << e.what() << "\n";
				sol_ok = false;		// solver failed, continue refining
			}
		}
		else
			sol_ok = false;			// init point not good, continue refining

		if (sol_ok)
			break;

		Nref *= 2;
	}

	if (!sol_ok)
		throw Exception("LeapFrogGeneralized::make_step_x_ref failed");

//	if (RNK == 0 && Nref != 1)
//		std::cout << "[make_step_x_ref] SOLUTION FOUND after " << Nref << " refinements, eps = " << eps << ", sol-n = " << ToString(res.ToVector()) <<
//					 "init = " << ToString(init_ref.ToVector()) << "x = " << ToString(x.ToVector()) << "p = " << ToString(p.ToVector()) << "\n";

	return res;
}
//------------------------------------------------------------------------------------------
void LeapFrogGeneralized::bounce_p(const Mat &x, Mat &p, int con_ind)									// perform velocity bounce at point 'x' w.r.t. coordinate 'con_ind', updating momentum 'p'
{
	Ham0.G.MsgToFile("-----------LeapFrogGeneralized, bounce_p, Mass_Matrix_for_velocity\n");
	int rank;
	MPI_Comm_rank(comm, &rank);

	Mat MM = Ham1.G.Get(&Ham1, x.ToVector());
	if (rank == 0)
	{
		Mat vel = MM/p;
		//std::cout << "vel before update " << ToString(vel.ToVector());	// DEBUG
		const Mat vel_prev = vel;				// for angle monitoring - debug
		BounceVel(MM, vel, con_ind);
		//std::cout << "vel after  update " << ToString(vel.ToVector());	// DEBUG
		//std::cout << "MM\n" << MM.ToString();	// DEBUG
		p = MM*vel;								// update the momentum vector after bounce

		const double ang = acos(InnerProd(vel_prev, vel)/(vel_prev.Norm2()*vel.Norm2())) / acos(-1.0)*180;
		std::cout << ", reflection angle (degrees) for vel = " << ang << "\n";			// DEBUG
	}
	p.Bcast(0, comm);							// sync p
}
//------------------------------------------------------------------------------------------
double LeapFrogGeneralized::make_step_x_bounce_1(Mat &x, Mat &p, const double eps, int &bounce_count)			// x(tau) -> x(tau + eps'), boundary bouncing, 0 < eps' <= eps, exits after first bounce (updating "p"), or after reaching the step size "eps" (no bounce); returns the remaining step eps - eps'
{
	// this is the first strategy for making bounces;
	// when nonlinear solver fails, all other bounds are tried for locating the intersection

	Ham0.G.MsgToFile("-----------LeapFrogGeneralized, make_step_x_bounce_1\n");
	int RNK;
	MPI_Comm_rank(MPI_COMM_WORLD, &RNK);

	const Mat xprev = x;
	x = make_step_x_ref(x, p, eps);		// x(tau) -> x(tau + eps), implicit equation

	std::vector<double> xint;			// intersection point
	int con_ind;						// index of the constraint which gave the bounce
	double alpha;						// fraction of step "eps" made till the bounce
	if (!Ham0.FindIntersect_ACT(xprev.ToVector(), x.ToVector(), xint, alpha, con_ind))	// bounds violated
	{
		size_t actdim = x.ICount();
		const BoundConstr *con = Ham0.GetConstr();
		const std::vector<double> min = con->actmin();
		const std::vector<double> max = con->actmax();

		std::vector<bound> bounds(2*actdim);		// array of bounds for finding the intersection
		bounds[0].first = con_ind;
		if (fabs(xint[con_ind] - min[con_ind]) < fabs(xint[con_ind] - max[con_ind]))
			bounds[0].second = min[con_ind];
		else
			bounds[0].second = max[con_ind];
		make_bounds(xprev, bounds);

		bool sol_ok = false;
		std::vector<double> res;
		for (size_t i = 0; i < 2*actdim; i++)		// try to find a good intersection for some coordinate
		{
			VM_Ham_eq2 VM2(&Ham0, &Ham1);
			VM2.x0 = xprev.ToVector();
			VM2.p = p.ToVector();

			VM_Ham_eq2_eps VM2eps(VM2, bounds[i].first, bounds[i].second);

			std::vector<double> init;
			if (i == 0)
				init = VM2eps.map_xfull_x(xint);	// for the first coordinate, initial point = "linear" intersection point
			else
				init = VM2eps.map_xfull_x(xprev.ToVector());	// for the other coordinates, initial point = current point

			std::string msg0 = "\nsolving VM_Ham_eq2_eps, " + stringFormatArr("bound-{0:%g}, coordinate-{1:%g} with value = {2:%g}\n", std::vector<double>{(double)i, (double)bounds[i].first, bounds[i].second}) +
							   "x0 = " + ToString(VM2eps.x0) + "x = " + ToString(x.ToVector()) + "p = " + ToString(VM2eps.p) + "init = " + ToString(init) + "full bounds list:\n" + bounds_to_str(bounds) + "\n";
			try
			{
				solver->exc_msg = msg0;
				solver->SetFuncFromVM(&VM2eps);
				solver->rescale_eps(1);
				res = solver->Solve(init);
				VM2eps.Func_ACT(res);				// call this only to properly update 'eps'
				iter_count += solver->iter;
			}
			catch (const std::exception &e)
			{
				if (RNK == 0)
					std::cout << "Solver failure: " << e.what() << "\n";
				continue;							// if solver fails, go to other coordinate
			}

			if (VM2eps.eps <= 0 || VM2eps.eps > eps)
			{
				const std::vector<double> resid = VM2eps.Func_ACT(res);		// for debug monitoring
				if (RNK == 0)
					std::cout << "Failure: found eps' = " << VM2eps.eps << " violates the range (0, " << eps << "]; residual = " << ToString(resid) <<
								"sol-n = " << ToString(res) << "after" + msg0;
				continue;							// the intersection found has inappropriate epsilon
			}

			res = VM2eps.map_x_xfull(res);			// return from ACTDIM-1 to ACTDIM
			if (!Ham0.CheckLimits_ACT(res))
			{
				if (RNK == 0)
					std::cout << "Failure: found point = " << ToString(res) << "violates the coordinate bounds after" + msg0;
				continue;							// the intersection found violates some other bounds
			}

			sol_ok = true;							// appropriate intersection found!
			con_ind = bounds[i].first;				// set the exact constraint index, which gave the bounce
			if (RNK == 0)		// DEBUG
				std::cout << "DEBUG bounce coord. " << con_ind << ", alpha approx. = " << alpha;	// DEBUG
			alpha = VM2eps.eps / eps;				// set the exact fraction of the step made
			if (RNK == 0)	// DEBUG
				std::cout << ", alpha exact. = " << alpha;	// DEBUG
			break;
		}

		if (!sol_ok)
			throw Exception("LeapFrogGeneralized::make_step_x_bounce_1 failed to find intersection point on any coordinate bound");

		// VELOCITY (MOMENTUM) BOUNCE
		x = std::move(res);							// update x
		bounce_p(x, p, con_ind);					// bounce the velocity
		bounce_count++;
		return eps*(1 - alpha);						// the remaining step length
	}
	else				// no bounce, x - updated, p - same
		return 0;
}
//------------------------------------------------------------------------------------------
double LeapFrogGeneralized::make_step_x_bounce_2(Mat &x, Mat &p, const double eps, int &bounce_count)		// x(tau) -> x(tau + eps'), -"-, employs the refinement strategy in case of solver failure
{
	// this is the second strategy for making bounces;
	// when nonlinear solver fails, another initial point for the solver is tried (found by several refined steps)

	const int ModelType = 2;		// 1 for VM_Ham_eq2_eps, 2 for VM_Ham_eq2_eps_full

	Ham0.G.MsgToFile("-----------LeapFrogGeneralized, make_step_x_bounce_2\n");
	int RNK;
	MPI_Comm_rank(MPI_COMM_WORLD, &RNK);

	const Mat xprev = x;
	x = make_step_x_ref(x, p, eps);		// x(tau) -> x(tau + eps), implicit equation

	std::vector<double> xint;			// "linear" intersection point
	int con_ind;						// index of the constraint which gave the bounce
	double alpha;						// fraction of step "eps" made till the bounce
	if (!Ham0.FindIntersect_ACT(xprev.ToVector(), x.ToVector(), xint, alpha, con_ind))	// bounds violated
	{
		const BoundConstr *con = Ham0.GetConstr();
		const std::vector<double> min = con->actmin();
		const std::vector<double> max = con->actmax();
		double Mi;						// value at the intersection bound
		if (fabs(xint[con_ind] - min[con_ind]) < fabs(xint[con_ind] - max[con_ind]))
			Mi = min[con_ind];
		else
			Mi = max[con_ind];

		std::vector<double> res;
		VM_Ham_eq2 VM2(&Ham0, &Ham1);
		VM2.x0 = xprev.ToVector();
		VM2.p = p.ToVector();

		VM_Ham_eq2_eps *VM2eps = 0;
		if (ModelType == 1)
			VM2eps = new VM_Ham_eq2_eps(VM2, con_ind, Mi);			// first, try the found "linear" intersection
		else
			VM2eps = new VM_Ham_eq2_eps_full(VM2, con_ind, Mi);

		VM2eps->eps = eps*alpha;
		std::vector<double> init = VM2eps->map_xfull_x(xint);		// take "linear" intersection point
		std::string msg0 = "\nsolving VM_Ham_eq2_eps, " + stringFormatArr("refine-{0:%g}, coordinate-{1:%g} with value = {2:%g}, init_eps = {3:%g}\n", std::vector<double>{1, (double)con_ind, Mi, VM2eps->eps}) +
						   "x0 = " + ToString(VM2eps->x0) + "x = " + ToString(x.ToVector()) + "p = " + ToString(VM2eps->p) + "init = " + ToString(init) + "\n";

		bool sol_ok = false;
		bool init_ok = true;	// TRUE if init point can be used
		int c = 0;				// refinement degree
		size_t Nref = 1;		// number of fine steps
		while (!sol_ok)
		{
			bool error = false;			// TRUE if solver fails to find appropriate point
			if (init_ok)				// init point - good, try solving
			{
				res = std::vector<double>(init.size(), 0);		// form some 'res' in case the solver fails
				std::vector<double> resid;						// for debug monitoring
				try
				{
					solver->exc_msg = msg0;
					solver->SetFuncFromVM(VM2eps);
					solver->rescale_eps(1);
					res = solver->Solve(init);
					resid = VM2eps->Func_ACT(res);		// call this to properly update 'eps'
					iter_count += solver->iter;
				}
				catch (const std::exception &e)
				{
					if (RNK == 0)
						std::cout << "Solver failure: " << e.what() << "\n";
					error = true;						// if solver fails, refine
				}

				res = VM2eps->map_x_xfull(res);			// return from "inner" to "outer" representation; fill 'eps'
				if (!error && (VM2eps->eps <= 0 || VM2eps->eps > eps))
				{
					if (RNK == 0)
						std::cout << "Failure: found eps' = " << VM2eps->eps << " violates the range (0, " << eps << "]; residual = " << ToString(resid) <<
									 "sol-n = " << ToString(res) << "after" + msg0;
					error = true;						// the intersection found has inappropriate epsilon
				}

				if (!error && !Ham0.CheckLimits_ACT(res))
				{
					if (RNK == 0)
						std::cout << "Failure: found point = " << ToString(res) << "violates the coordinate bounds after" + msg0;
					error = true;						// the intersection found violates some other bounds
				}
			}
			else						// init point not good
				error = true;

			if (error)					// no suitable solution found -> find another init point, using refinement
			{
				Nref *= 2;
				if (c >= max_refine)
					break;

				init = std::vector<double>(xprev.ICount(), 0);		// form some 'init' in case solver fails
				double eps_part;
				try
				{
					init = make_Nstep_xint(xprev, p, eps/Nref, Nref, con_ind, eps_part).ToVector();		// make a new init point; inner solver failures will result in rejecting the init point
					init_ok = true;
				}
				catch (...)
				{
					if (RNK == 0)
						std::cout << "[make_step_x_bounce_2] failure to find 'init' point with refinement " << Nref << "\n";

					con_ind = 0;		// just in case
					eps_part = 0;
					init_ok = false;
				}

				if (fabs(init[con_ind] - min[con_ind]) < fabs(init[con_ind] - max[con_ind]))
					Mi = min[con_ind];
				else
					Mi = max[con_ind];

				VM2eps->eps = eps_part;
				VM2eps->i0 = con_ind;
				VM2eps->M0 = Mi;
				init = VM2eps->map_xfull_x(init);

				msg0 = "\nsolving VM_Ham_eq2_eps, " + stringFormatArr("refine-{0:%g}, coordinate-{1:%g} with value = {2:%g}, eps_part = {3:%g}\n", std::vector<double>{(double)Nref, (double)con_ind, Mi, eps_part}) +
					   "x0 = " + ToString(VM2eps->x0) + "p = " + ToString(VM2eps->p) + "init = " + ToString(init) + "\n";

				c++;					// increment refinements count
			}
			else						// solution found
			{
				sol_ok = true;							// appropriate intersection found!
				con_ind = VM2eps->i0;					// set the exact constraint index, which gave the bounce
				if (RNK == 0)	// DEBUG
					std::cout << "DEBUG bounce coord. " << con_ind << ", alpha linear = " << alpha;	// DEBUG
				alpha = VM2eps->eps / eps;				// set the exact fraction of the step made
				if (RNK == 0)	// DEBUG
					std::cout << ", alpha exact. = " << alpha << "\n";	// DEBUG
			}
		}

		if (!sol_ok)
		{
			delete VM2eps;
			throw Exception("LeapFrogGeneralized::make_step_x_bounce_2 failed");
		}

		// VELOCITY (MOMENTUM) BOUNCE
		x = std::move(res);							// update x
		bounce_p(x, p, con_ind);					// bounce the velocity
		bounce_count++;

		if (RNK == 0 && c > 0)
			std::cout << "[make_step_x_bounce_2] SOLUTION FOUND after " << Nref << " refinements, sol-n = " << ToString(x.ToVector()) << "\n";

		delete VM2eps;
		return eps*(1 - alpha);						// the remaining step length
	}
	else								// no bounce, x - updated, p - same
		return 0;
}
//------------------------------------------------------------------------------------------
double LeapFrogGeneralized::make_step_x_bounce_3(Mat &x, Mat &p, const double eps, int &bounce_count)	// x(tau) -> x(tau + eps'), -"-, employs the bisection strategy in case of solver failure
{
	// this is the third strategy for making bounces;
	// when nonlinear solver fails, a step eps0 (found from bisection) is tried

	const int ModelType = 2;		// 1 for VM_Ham_eq2_eps, 2 for VM_Ham_eq2_eps_full

	Ham0.G.MsgToFile("-----------LeapFrogGeneralized, make_step_x_bounce_3\n");
	int RNK;
	MPI_Comm_rank(MPI_COMM_WORLD, &RNK);

	double a = 0;
	double b = eps;
	double eps0 = eps;
	const Mat xprev = x;

	const BoundConstr *con = Ham0.GetConstr();
	const std::vector<double> min = con->actmin();
	const std::vector<double> max = con->actmax();

	for (int c = 0; c < max_refine3; c++)
	{
		x = make_step_x_ref(xprev, p, eps0, (c/5 < max_eps_refine ? c/5 : max_eps_refine));	// x(tau) -> x(tau + eps0), implicit equation			TODO instead of c/5, can use: 0, c,...

		// DEBUG
//		if (c == max_refine-3)
//		{
//			//const double deps = 0.0002;
//			const double deps = 0.01;
//			const int NN = 300;
//			for (int i = 0; i < NN; i++)
//			{
//				//double eps1 = eps0 + deps*i/NN;
//				double eps1 = deps*i/NN;
//				Mat x2 = make_step_x_ref(xprev, p, eps1, 2);
//				Mat vel = Ham0.G.Get(&Ham0, x2.ToVector()) / p;
//				std::cout << eps1 << "\t" << x2(0,0) << "\t" << x2(1,0) << "\t" << ToString(vel.ToVector());
//			}
//		}
		// DEBUG

		// DEBUG
//		if (c == max_refine-3)
//		{
//			VM_Ham_eq2 VM2(&Ham0, &Ham1);
//			VM2.x0 = xprev.ToVector();
//			VM2.p = p.ToVector();
//			VM_Ham_eq2_eps VM2eps(VM2, 1, -2);
//
//			const int NN = 500;
//			for (int i = 0; i < NN; i++)
//			{
//				double d = 0.5 + double(0.4)*i/NN;
//				std::vector<double> xx(1), yy;
//				xx[0] = d;
//				yy = VM2eps.Func_ACT(xx);
//				std::cout << d << "\t" << VM2eps.eps << "\t" << yy[0] << "\n";
//			}
//		}
		// DEBUG

		std::vector<double> xint;				// projection point
		int con_ind;							// index of the constraint to search the bounce
		//if (!Ham0.FindIntersect_ACT(xprev.ToVector(), x.ToVector(), xint, alpha, con_ind))	// doesn't give desired behaviour if xprev itself is on the violated bound
		if (!limits_and_proj(x.ToVector(), xint, con_ind))		// bounds violated
		{
			double Mi;							// value at the intersection bound
			if (fabs(xint[con_ind] - min[con_ind]) < fabs(xint[con_ind] - max[con_ind]))
				Mi = min[con_ind];
			else
				Mi = max[con_ind];

			std::vector<double> res;
			VM_Ham_eq2 VM2(&Ham0, &Ham1);
			VM2.x0 = xprev.ToVector();
			VM2.p = p.ToVector();

			//std::vector<double> func_x = VM2.Func_ACT(x.ToVector());		// debug output
			//Mat jac_x = VM2.Jac_ACT(x.ToVector());						// debug output

			VM_Ham_eq2_eps *VM2eps = 0;
			if (ModelType == 1)
				VM2eps = new VM_Ham_eq2_eps(VM2, con_ind, Mi);
			else
				VM2eps = new VM_Ham_eq2_eps_full(VM2, con_ind, Mi);

			VM2eps->eps = eps0;
			std::vector<double> init = VM2eps->map_xfull_x(xint);			// take projection point
			std::vector<double> VM2eps_func_trial = VM2eps->Func_ACT(init);	// this approximate solution might be taken as the final one

			bool error = false;							// TRUE if solver fails to find appropriate point
			if (Mat(VM2eps_func_trial).Norm1() >= solver->atol())
			{
				res = std::vector<double>(init.size(), 0);	// form dummy 'res' in case the solver fails
				std::vector<double> resid;					// for debug monitoring
				std::string msg0 = "\nsolving VM_Ham_eq2_eps, " + stringFormatArr("bisection-{0:%g}, coordinate-{1:%g} with value = {2:%g}; eps0 = {3:%g}, b-a = {4:%g}, [a, b] = [{5:%g}, {6:%g}]\n",
									std::vector<double>{(double)c, (double)con_ind, Mi, eps0, b-a, a, b}) +
								   "x0 = " + ToString(VM2eps->x0) + "x = " + ToString(x.ToVector()) + "x projection = " + ToString(xint) + "func_trial = " + ToString(VM2eps_func_trial) +
								   "p = " + ToString(VM2eps->p) + "init (inner) = " + ToString(init) + "\n";
				try
				{
					assert(b > a);
					solver->exc_msg = msg0;
					solver->SetFuncFromVM(VM2eps);
					solver->rescale_eps((b-a)/eps);
					res = solver->Solve(init);
					resid = VM2eps->Func_ACT(res);			// update 'eps'
					iter_count += solver->iter;
				}
				catch (const std::exception &e)
				{
					if (RNK == 0)
						std::cout << "Solver failure: " << e.what() << "\n";
					error = true;							// if solver fails, refine
				}

				res = VM2eps->map_x_xfull(res);				// return from inner to outer, fill 'eps'
				if (!error && (VM2eps->eps <= 0 || VM2eps->eps > eps0))
				{
					if (RNK == 0)
						std::cout << "Failure: found eps' = " << VM2eps->eps << " violates the range (0, " << eps0 << "]; residual = " << ToString(resid) <<
									 "sol-n = " << ToString(res) << "after" + msg0;
					error = true;							// the intersection found has inappropriate epsilon
				}

				if (!error && !Ham0.CheckLimits_ACT(res))
				{
					if (RNK == 0)
						std::cout << "Failure: found point = " << ToString(res) << "violates the coordinate bounds after" + msg0;
					error = true;							// the intersection found violates some other bounds
				}
			}
			else
			{
				res = xint;						// take approximate solution
				VM2eps->eps = eps0;
				error = false;
			}

			if (error)							// no suitable solution found -> bisect, and do the next iteration
				b = eps0;
			else								// solution found -> take it and exit
			{
				if (RNK == 0)	// DEBUG
					std::cout << "DEBUG bounce coord. " << con_ind << ", alpha approx. = " << eps0/eps;	// DEBUG
				double alpha = VM2eps->eps / eps;				// set the exact fraction of the step made
				if (RNK == 0)	// DEBUG
					std::cout << ", alpha exact. = " << alpha;	// DEBUG

				// VELOCITY (MOMENTUM) BOUNCE
				x = std::move(res);						// update x
				bounce_p(x, p, con_ind);				// bounce the velocity
				bounce_count++;

				if (RNK == 0 && c > 0)
					std::cout << "[make_step_x_bounce_3] SOLUTION FOUND after " << c << " bisections, sol-n = " << ToString(x.ToVector()) << "\n";

				delete VM2eps;
				return eps*(1 - alpha);						// the remaining step length
			}

			delete VM2eps;
		}
		else					// bounds are not violated
		{
			if (c == 0)
				return 0;		// no bounce, x - updated, p - same
			else
				a = eps0;
		}

		eps0 = (a+b)/2;			// bisection
	}

	throw Exception("LeapFrogGeneralized::make_step_x_bounce_3 failed");
}
//------------------------------------------------------------------------------------------
Mat LeapFrogGeneralized::make_Nstep_xint(Mat x, const Mat &p, double eps0, size_t N, int &con_ind, double &eps_part)		// makes at most N steps (size = eps0) for 'x', returns the first found "linear" intersection
{	// (also filling 'con_ind' - intersected bound, and 'eps_part' - cumulative step length taken till intersection), or the final point; this function is for making the initial approximation candidates for make_step_x_bounce_2
	Ham0.G.MsgToFile("-----------LeapFrogGeneralized, make_Nstep_xint\n");
	int RNK;
	MPI_Comm_rank(MPI_COMM_WORLD, &RNK);

	eps_part = 0;
	for (size_t i = 0; i < N; i++)
	{
		const Mat xprev = x;
		x = make_step_x(x, p, eps0);		// x(tau) -> x(tau + eps0), implicit equation

		std::vector<double> xint;			// intersection point
		double alpha;						// fraction of step "eps0" made till the bounce
		if (!Ham0.FindIntersect_ACT(xprev.ToVector(), x.ToVector(), xint, alpha, con_ind))				// bounds violated -> intersection found
		{
			eps_part += alpha*eps0;
			return xint;
		}

		eps_part += eps0;
	}

	const BoundConstr *con = Ham0.GetConstr();
	const std::vector<double> min = con->actmin();
	const std::vector<double> max = con->actmax();

	size_t actdim = min.size();
	assert(actdim == x.ICount());
	double mindist = std::numeric_limits<double>::max();		// find the bound closest to the final point
	for (size_t i = 0; i < actdim; i++)
	{
		if (fabs(x(i, 0) - min[i]) < mindist)
		{
			mindist = fabs(x(i, 0) - min[i]);
			con_ind = i;
		}
		if (fabs(x(i, 0) - max[i]) < mindist)
		{
			mindist = fabs(x(i, 0) - max[i]);
			con_ind = i;
		}
	}

	if (RNK == 0)
		std::cout << ">>> make_Nstep_xint, " << N << " step(s), reaching the final point, con_ind = " << con_ind << "\n";
	return x;				// no intersection found, return the final point
}
//------------------------------------------------------------------------------------------
Mat LeapFrogGeneralized::make_Nstep_p0(const Mat &x, Mat p, double eps0, size_t N)				// makes N steps (size = eps0) for 'p'; this function is for making the initial approximation candidates for make_step_p0_ref
{
	Ham0.G.MsgToFile("-----------LeapFrogGeneralized, make_Nstep_p0\n");

	for (size_t i = 0; i < N; i++)
		p = make_step_p0(x, p, eps0);		// p(tau) -> p(tau + eps0/2), implicit equation

	return p;
}
//------------------------------------------------------------------------------------------
Mat LeapFrogGeneralized::make_Nstep_x(Mat x, const Mat &p, double eps0, size_t N)				// makes N steps (size = eps0) for 'x'; this function is for making the initial approximation candidates for make_step_x_ref
{
	Ham0.G.MsgToFile("-----------LeapFrogGeneralized, make_Nstep_x\n");

	for (size_t i = 0; i < N; i++)
	{
		x = make_step_x(x, p, eps0);		// x(tau) -> x(tau + eps0), implicit equation
		//std::cout << i << " " << eps0 << " " << ToString(x.ToVector());	// DEBUG
	}

	return x;
}
//------------------------------------------------------------------------------------------
void LeapFrogGeneralized::make_bounds(const Mat &xcurr, std::vector<bound> &bounds) const		// fills array of bounds (2*ACTDIM), ordered by increasing distance from 'xcurr' to the boundaries; bounds[0] is intact, and its value is not repeated in bounds[i>0]
{
	const BoundConstr *con = Ham0.GetConstr();
	const std::vector<double> min = con->actmin();
	const std::vector<double> max = con->actmax();

	size_t actdim = min.size();

	assert(bounds.size() == 2*actdim && xcurr.ICount() == actdim);
	std::vector<double> dist(2*actdim, 0);
	bound save = bounds[0];					// save for later use

	for (size_t i = 0; i < actdim; i++)
	{
		bounds[2*i].first = i;
		bounds[2*i].second = min[i];
		dist[2*i] = fabs(xcurr(i, 0) - min[i]);

		bounds[2*i + 1].first = i;
		bounds[2*i + 1].second = max[i];
		dist[2*i + 1] = fabs(xcurr(i, 0) - max[i]);
	}

	for (size_t i = 0; i < 2*actdim; i++)
		if (bounds[i] == save)
		{
			dist[i] = 0;					// to return the original bound[0] to the beginning
			break;
		}

	std::vector<size_t> inds = SortPermutation(dist.begin(), dist.end());
	bounds = Reorder(bounds, inds);
}
//------------------------------------------------------------------------------------------
Mat LeapFrogGeneralized::make_step_p1(const Mat &x, const Mat &p, double eps)			// p(tau + eps/2) -> p(tau + eps), explicit equation
{
	Ham0.G.MsgToFile("+++make_step_p1+++\n");
	Ham1.pact = p;
	Mat Hx = Ham1.ObjFuncGrad_ACT(x.ToVector());
	Hx.Bcast(0, comm);

	return p - (eps/2)*Hx;
}
//------------------------------------------------------------------------------------------
std::string LeapFrogGeneralized::bounds_to_str(const std::vector<bound> &bounds) const
{
	std::string res;
	char work[BUFFSIZE];
	for (size_t i = 0; i < bounds.size(); i++)
	{
		sprintf(work, "%zu:\ti = %d\tM = %g\n", i, bounds[i].first, bounds[i].second);
		res += work;
	}

	return res;
}
//------------------------------------------------------------------------------------------
void LeapFrogGeneralized::BounceVel(const Mat &MM, Mat &vel, int ci) const			// velocity bounce w.r.t. constraint 'ci', works with 'MM' directly (does not use decomposition)
{
	Mat Mei(MM.ICount(), 1, 0.0);
	Mei(ci, 0) = 1;
	Mei = MM / std::move(Mei);			// MM^(-1)*ei -- ci-th column of MM^(-1)

	vel += (-2*vel(ci, 0)/Mei(ci, 0)) * Mei;
}
//------------------------------------------------------------------------------------------
bool LeapFrogGeneralized::limits_and_proj(const std::vector<double> &x, std::vector<double> &x_proj, int &con_ind) const	// returns FALSE if "x" violates the bounds; in this case sets "con_ind" to the index of the bound with strongest violation
{																															// and sets "x_proj" to "x" with violated coordinates set to the corresponding exact bound values
	const BoundConstr *con = Ham0.GetConstr();
	const std::vector<double> min = con->actmin();
	const std::vector<double> max = con->actmax();

	if (x.size() != min.size())
		throw Exception(stringFormatArr("x.size [{0:%zu}] != min.size [{1:%zu}] in LeapFrogGeneralized::limits_and_proj", std::vector<size_t>{x.size(), min.size()}));

	x_proj = x;

	double max_v = 0;
	for (size_t i = 0; i < x.size(); i++)
	{
		double v = 0;				// violation value
		if (x[i] < min[i])
		{
			v = fabs(x[i] - min[i]);
			x_proj[i] = min[i];
		}
		if (v > max_v)
		{
			con_ind = i;
			max_v = v;
		}

		if (x[i] > max[i])
		{
			v = fabs(x[i] - max[i]);
			x_proj[i] = max[i];
		}
		if (v > max_v)
		{
			con_ind = i;
			max_v = v;
		}
	}

	if (max_v > 0)
		return false;
	else
		return true;
}
//------------------------------------------------------------------------------------------
LeapFrogGeneralized::LeapFrogGeneralized(PhysModel *pm, NonlinearSystemSolver *sol, int maxref, double MM_shift) :
		comm(pm->GetComm()), Ham0(pm, MM_shift), Ham1(pm, MM_shift), solver(sol), solver_init_pt(2), max_refine3(maxref), scale_factor_eps(1e-10), max_eps_refine(12), iter_count(0), dist_ratio(0)
{
	max_refine = (max_refine3 < max_eps_refine ? max_refine3 : max_eps_refine);

#ifdef LF_GENERALIZED_OUT_X
    	FILE *fwork = fopen(LF_GENERALIZED_OUT_X, "w");
    	fclose(fwork);
#endif
}
//------------------------------------------------------------------------------------------
int LeapFrogGeneralized::Run(Mat &x, Mat &p, int N, double eps)
{
	if (comm == MPI_COMM_NULL)
		return 0;

	const Mat x_init = x;					// sync on 'comm'
	double max_dist = 0;

	int tot_bounce_count = 0;
    for (int i = 0; i < N; i++)
    {
    	p = make_step_p0_ref(x, p, eps);	// p(tau) -> p(tau + eps/2), implicit equation

    	double eps_remaining = eps;
    	while (eps_remaining > 0)
    		eps_remaining = make_step_x_bounce_3(x, p, eps_remaining, tot_bounce_count);		// x(tau) -> x(tau + eps), with intermediate bounces

    	p = make_step_p1(x, p, eps);		// p(tau + eps/2) -> p(tau + eps), explicit equation

#ifdef LF_GENERALIZED_OUT_X
    	FILE *fwork = fopen(LF_GENERALIZED_OUT_X, "a");
    	fputs(x.Tr().ToString().c_str(), fwork);
    	fclose(fwork);
#endif

    	Ham0 = Ham1;

        const double dist = (x - x_init).Norm2();		// update the distances
        if (dist > max_dist)
        	max_dist = dist;
        dist_ratio = dist / max_dist;
    }

    return tot_bounce_count;
}
//------------------------------------------------------------------------------------------
double LeapFrogGeneralized::of_ham0_ACT(const std::vector<double> &x, const std::vector<double> &p)
{
	Ham0.pact = p;
	return Ham0.ObjFunc_ACT(x);
}
//------------------------------------------------------------------------------------------
double LeapFrogGeneralized::of_ham1_ACT(const std::vector<double> &x, const std::vector<double> &p)
{
	Ham1.pact = p;
	return Ham1.ObjFunc_ACT(x);
}
//------------------------------------------------------------------------------------------
// MC_point
//------------------------------------------------------------------------------------------
void MC_point::save_point_2()
{
	int rnk = -1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rnk);

	if (rnk == 0)
	{
		FILE *file = fopen(etc_file, "a");					// o.f. value, eps, acc_rate
		if (file != NULL)
		{
			fprintf(file, "%12.8g\t%12.8g\t%12.8g\n", y, eps, acc_rate);
			fclose(file);
		}
	}
}
//------------------------------------------------------------------------------------------
void MC_point::ResetFiles()
{
	int rnk = -1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rnk);

	if (rnk == 0)
	{
		FILE *file = fopen(par_file, "w");
		if (file != NULL)
			fclose(file);

		file = fopen(par_all_file, "w");
		if (file != NULL)
			fclose(file);

		file = fopen(mod_data_file, "w");
		if (file != NULL)
			fclose(file);

		file = fopen(etc_file, "w");
		if (file != NULL)
			fclose(file);
	}
}
//------------------------------------------------------------------------------------------
void MC_point::SavePoint()
{
	int rnk = -1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rnk);

	if (rnk == 0)
	{
#ifdef FULL_SAVE_MCMC_POINT
		FILE *file = fopen(par_file, "a");				// model parameters
		if (file != NULL)
		{
			fputs(ToString(X).c_str(), file);
			fclose(file);
		}

		file = fopen(par_all_file, "a");				// model parameters for all points (incl. rejected)
		if (file != NULL)
		{
			fputs(ToString(Xall).c_str(), file);
			fclose(file);
		}

		file = fopen(mod_data_file, "a");				// modelled data
		if (file != NULL)
		{
			fputs(ToString(Mod).c_str(), file);
			fclose(file);
		}
#else
		FILE *file = fopen(par_file, "a");				// model parameter[0]
		if (file != NULL)
		{
			fprintf(file, "%12.8g\n", X[0]);
			fclose(file);
		}
#endif
	}

	save_point_2();		// o.f. value, etc
}
//------------------------------------------------------------------------------------------
// HMC_point
//------------------------------------------------------------------------------------------
void HMC_point::save_point_2()
{
	int rnk = -1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rnk);

	if (rnk == 0)
	{
		FILE *file = fopen(etc_file, "a");				// o.f. value, etc
		if (file != NULL)
		{
			fprintf(file, "%12.8g\t%12.8g\t%12.8g\t%d\t%12.8g\t%12.8g\t%12.8g\t%12.8g\t%12.8g\t%12.8g\t%12.8g\t%12.8g\t%12.8g\t%12.8g\t%12.8g\t%12.8g\t%12.8g\n",
						y, eps, acc_rate, lf_bounces, dist_ratio, U0, U0_aux, U1, U1_aux, Kin0, Kin1, dE, M_emin, M_cond2, m_adj, val0, val1);

			fclose(file);
		}
	}
}
//------------------------------------------------------------------------------------------
// EpsUpdate1
//------------------------------------------------------------------------------------------
double EpsUpdate1::EpsMult(int block_size, double acc_mult)	// calculates acc_rate, acc_rate_inner, returns multiplier for epsilon (depending on 'acc_from_inner');
															// sets 'acc_block', 'acc_block_inner' to 0 (to work with the next block)
															// 'block_size' is the size of block which has been processed
															// 'acc_rate' is multiplied by 'acc_mult'; 'acc_rate_inner' is not multiplied
{
	// this function returns beta^res, where res = res(acc_rate), res(0) = -1, and res(1) = alpha
	const int _option_ = 2;		// option 1: [0, acc_targ) - linear 'res', [acc_targ, 1) - quadratic 'res', and the first derivative is continuous @ acc_rate = acc_targ; however when acc_targ is small, the quadratic part may have a peak larger than 'alpha'
								// option 2: [0, acc_targ) - linear 'res', [acc_targ, 1) - another linear 'res', derivative is discontinuous, but the whole function is monotonic and continuous

	acc_rate = double(acc_block)/block_size * acc_mult;
	acc_rate_inner = acc_block_inner/block_size;
	const double rate = acc_from_inner ?  acc_rate_inner : acc_rate;

	double res = 1;
	double a0 = (alpha+1 - 1/acc_targ)/((acc_targ-1)*(acc_targ-1));
	double b0 = 1/acc_targ - 2*a0*acc_targ;
	double c0 = alpha - a0 - b0;

	if (rate < acc_targ)				// first part
		res = rate/acc_targ - 1;
	else								// second part
	{
		if (_option_ == 1)
			res = a0*rate*rate + b0*rate + c0;
		else
			res = alpha*(rate - acc_targ)/(1 - acc_targ);
	}

	acc_block = 0;
	acc_block_inner = 0;
    return pow(beta, res);
}
//------------------------------------------------------------------------------------------
// MCMC
//------------------------------------------------------------------------------------------
MCMC::MCMC(PhysModel *p, Rand g, EpsUpdate1 eu, int bi, int ufreq, int Nadd) : gen(std::move(g)), iter_counter(0), of0(0), of1(0), dE(0), pm(p), EUpd(eu), burn_in(bi), upd_freq(ufreq), Nadd_fval_pts(Nadd)
{
	MPI_Comm_rank(MPI_COMM_WORLD, &RNK);
	MPI_Comm_rank(pm->GetComm(), &rank_pm);
}
//------------------------------------------------------------------------------------------
int MCMC::Run(int count, Mat &x0, std::string *msg_time)		//TODO ...
{
	std::chrono::high_resolution_clock::time_point time1, time2, time3;
	time1 = time2 = std::chrono::high_resolution_clock::now();

	of0 = pm->ObjFunc_ACT(x0.ToVector());
	if (rank_pm == 0)
		ModelledData0 = pm->ModelledData();

	preprocess(x0);			// initial preparations
	EUpd.acc_total = 0;		// reset total accepted count

	for (int i = 0; i < count; i++)
	{
		iter_counter = i;
		x0.Bcast(0, MPI_COMM_WORLD);
		proposal(x0);		// propose move x0 -> x1

		x1.Bcast(0, MPI_COMM_WORLD);
		if (pm->CheckLimits_ACT(x1.ToVector()))			// check that the new point is in the box
		{
			of1 = pm->ObjFunc_ACT(x1.ToVector());		// pm->comm-RANKS-0 have 'of1'
			if (rank_pm == 0)
				ModelledData1 = pm->ModelledData();
		}
		else
		{
			of1 = of0;
			ModelledData1 = ModelledData0;
		}

		process_new_point(x0);							// find dE

		int accept = 0;
		if (RNK == 0 && log(gen.RandU()) < dE)			// Metropolis-Hastings test: only on RNK-0
		{
			accept = 1;
			x0 = x1;
			of0 = of1;
			ModelledData0 = ModelledData1;
			EUpd.AccModel();
		}

		save_output(x0);

		MPI_Bcast(&accept, 1, MPI_INT, 0, MPI_COMM_WORLD);
		if (accept)
			accept_new_point();
		else
			reject_new_point();

		make_updates(x0);								// update proxy etc

		if (i == burn_in-1)
			time2 = std::chrono::high_resolution_clock::now();
	}

	time3 = std::chrono::high_resolution_clock::now();
	if (msg_time != nullptr)
	{
		*msg_time = HMMPI::stringFormatArr("Burn-in time {0:%.3f} sec, main time {1:%.3f} sec\n", std::vector<double>{
										  std::chrono::duration_cast<std::chrono::duration<double>>(time2-time1).count(),
										  std::chrono::duration_cast<std::chrono::duration<double>>(time3-time2).count()});
		Bcast_string(*msg_time, 0, MPI_COMM_WORLD);
	}

	// sync the output
	MPI_Bcast(&(EUpd.acc_total), 1, MPI_INT, 0, MPI_COMM_WORLD);
	x0.Bcast(0, MPI_COMM_WORLD);

	return EUpd.acc_total;
}
//------------------------------------------------------------------------------------------
// RWM1
//------------------------------------------------------------------------------------------
void RWM1::proposal(const Mat &x)
{
	if (RNK == 0)
	{
		if (L.Length() != 0)
			x1 = x + eps * (L * gen.RandN(x.ICount(), 1));
		else
			x1 = x + eps * gen.RandN(x.ICount(), 1);		// <-> unity covariance

		if (!pm->CheckLimits_ACT(x1.ToVector()))			// if point violates bounds, reject it
		{
			x1 = x;
			of1 = of0;
			ModelledData1 = ModelledData0;
			EUpd.decr_count();		// decrease count now, because MH test will formally "accept" the rejected point (x_new - x_old = 0)
		}
	}
}
//------------------------------------------------------------------------------------------
void RWM1::process_new_point(const Mat &x)		// take x = x0 !
{
	if (RNK == 0)
		dE = -of1/2 + of0/2;
}
//------------------------------------------------------------------------------------------
void RWM1::save_output(const Mat &x)
{
	// x0 = x (RNK-0) now stores the current point (either old or updated)
	if (RNK == 0)
	{
		MC_point point(x.ToVector(), x1.ToVector(), ModelledData0, of0, eps, EUpd.acc_rate);
		point.SavePoint();
	}
}
//------------------------------------------------------------------------------------------
void RWM1::make_updates(Mat &x)
{
	if ((iter_counter+1) % upd_freq == 0)
	{
		double e_mult = EUpd.EpsMult(upd_freq);
		MPI_Bcast(&e_mult, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		if (iter_counter < burn_in)
			eps *= e_mult;					// update epsilon
	}
}
//------------------------------------------------------------------------------------------
RWM1::RWM1(PhysModel *p, Rand g, EpsUpdate1 eu, int bi, int ufreq, int Nadd, double e, Mat cov) : MCMC(p, std::move(g), eu, bi, ufreq, Nadd), eps(e), C(std::move(cov))
{
	LeapFrog::comm_compatible(MPI_COMM_WORLD, pm->GetComm(), "RWM1");

	if (C.Length() != 0)
		L = C.CholSPO().Tr();		// the upper triangle of C is used
	else
		L = HMMPI::Mat();			// for cov == empty (-> use unity matrix) don't do Cholesky decomposition

	// reset the output files
	MC_point point;
	point.ResetFiles();
}
//------------------------------------------------------------------------------------------
// pCN1
//------------------------------------------------------------------------------------------
void pCN1::proposal(const Mat &x)
{
	if (RNK == 0)
	{
		//x1 = sqrt(1 - eps*eps) * x + eps * (L * gen.RandN(x.ICount(), 1) + dpr);
		x1 = sqrt(1 - eps*eps) * x + eps * (L * gen.RandN(x.ICount(), 1));

		if (!pm->CheckLimits_ACT(x1.ToVector()))			// if point violates bounds, reject it
		{
			x1 = x;
			of1 = of0;
			ModelledData1 = ModelledData0;
			EUpd.decr_count();		// decrease count now, because MH test will formally "accept" the rejected point (x_new - x_old = 0)
		}
	}
}
//------------------------------------------------------------------------------------------
void pCN1::process_new_point(const Mat &x)		// take x = x0 !
{
	const PM_Posterior *post = dynamic_cast<const PM_Posterior*>(pm);
	double p0, p1; 					// 0-centred prior

	p0 = InnerProd(x, post->cov_prior()/x);
	p1 = InnerProd(x1, post->cov_prior()/x1);

	if (RNK == 0)
		dE = -of1/2 + of0/2 + p1/2 - p0/2;
}
//------------------------------------------------------------------------------------------
pCN1::pCN1(PhysModel *p, Rand g, EpsUpdate1 eu, int bi, int ufreq, int Nadd, double e) : RWM1(p, g, eu, bi, ufreq, Nadd, e, HMMPI::Mat())
{
	const PM_Posterior *post = dynamic_cast<const PM_Posterior*>(pm);
	if (post == nullptr)
		throw Exception("For pCN sampler, model should be of type POSTERIOR");

	L = post->cov_prior().CholSPO().Tr();					// the upper triangle of "Cpr" is used
}
//------------------------------------------------------------------------------------------
// HMC1
//------------------------------------------------------------------------------------------
void HMC1::resetVecs(bool is_initial)
{
	if (update_type == 0 || is_initial)
	{
		Xnew.clear();
		ynew.clear();
		Datanew.clear();
		Gradnew.clear();

		if (pm_aux->is_proxy())
		{
			Xnew.reserve(upd_freq);				// when update_type == 1, this reserve may be insufficient, however, this will not affect validity of the algorithm
			ynew.reserve(upd_freq);
			Gradnew.reserve(upd_freq);
			if (pm_aux->is_dataproxy())
				Datanew.reserve(upd_freq);
		}
	}
}
//------------------------------------------------------------------------------------------
void HMC1::updateVecs(std::vector<double> x, double y, std::vector<double> d, std::vector<double> grad)
{
	if (pm_aux->is_proxy())
	{
		Bcast_vector(x, 0, MPI_COMM_WORLD);
		x = pm->tot_par(x);

		PM_Posterior *post_work = dynamic_cast<PM_Posterior*>(pm_aux);
		if (post_work != nullptr && dynamic_cast<const PM_Proxy*>(post_work->get_PM()) != nullptr)		// if pm_aux is POSTERIOR <- PROXY, remove the PRIOR part
			post_work->correct_of_grad(x, y, grad);

		MPI_Bcast(&y, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		Xnew.push_back(x);
		ynew.push_back(y);
		if (!pm_aux->is_dataproxy())		// simple proxy
		{
			Bcast_vector(grad, 0, MPI_COMM_WORLD);
			if (grad.size() > 0)
				assert(grad.size() == x.size());
			Gradnew.push_back(grad);
		}
		else
			Gradnew.push_back(std::vector<double>());		// empty "gradient" added for data proxy

		if (pm_aux->is_dataproxy())			// data proxy
		{
			Bcast_vector(d, 0, MPI_COMM_WORLD);
			Datanew.push_back(d);
		}
	}
}
//------------------------------------------------------------------------------------------
void HMC1::trainProxy(bool is_initial)
{
	if (pm_aux->is_proxy() && Xnew.size() != 0)
	{
		// 0. Decide how many points to select
		int Nselect = Nadd_fval_pts;				// number of points (func. vals) that will be selected
		const int u = Xnew.size() / upd_freq;		// update number (1, 2, ...), with 0 <-> initial single-point update
		if (!is_initial && update_type == 1)
		{
			assert(Xnew.size() % upd_freq == 0);
			Nselect = u*Nadd_fval_pts;
		}

		// 0a. Fall back to the base proxy if necessary
		if (update_type == 1 && !is_initial)					// get the "base proxy"
		{
			const int flag = proxy_int->GetDumpFlag();

			assert(proxy_int_base != nullptr);
			delete proxy_int;
			proxy_int = proxy_int_base->Copy();					// copy
			proxy_int->SetDumpFlag(flag);

			pm_aux = dynamic_cast<PhysModel*>(proxy_int);		// borrow
			assert(pm_aux != nullptr);
		}

		// 1. Prepare indices for gradient points
		std::vector<size_t> inds_grad;
		if (calc_grads())
		{
			if (!is_initial)
				assert(Xnew.size() >= (size_t)Nselect);
			std::vector<size_t> inds_sel = proxy_int->PointsSubset(Xnew, Nselect);			// inds_sel points will be selected from Xnew (func. vals); inds_sel[i] is index for "Xnew"
			HMMPI::Bcast_vector(inds_sel, 0, pm_aux->GetComm());

			if (is_initial)
			{
				assert(Xnew.size() == 1);
				if (inds_sel.size() > 0 && ind_grad_add_pts.size() > 0 && ind_grad_add_pts[0] == 0)
					inds_grad.push_back(0);
			}
			else
			{
				std::vector<size_t> ind_grad_extended(ind_grad_add_pts.begin(), ind_grad_add_pts.end());
				if (update_type == 1)													// e.g. for Nadd_fval_pts = 5, ind_grad_extended => 0,1,4 | 5,6,9 | 10,11,14 |...
				{
					assert(u > 0);
					const size_t bsize = ind_grad_add_pts.size();
					ind_grad_extended.resize(u*bsize);
					for (int i = 0; i < u; i++)
						for (size_t j = 0; j < bsize; j++)
							ind_grad_extended[i*bsize + j] = ind_grad_add_pts[j] + i*Nadd_fval_pts;
				}
				inds_grad = HMMPI::Reorder(inds_sel, ind_grad_extended);				// inds_grad points will be selected from Xnew (grads); inds_grad[i] is index for "Xnew"
			}

//			std::cout << "DEBUG proxy train, select " << inds_sel.size() << " points with func. vals: " << HMMPI::ToString(inds_sel, "%d");
//			std::cout << "DEBUG proxy train, select " << inds_grad.size() << " points with gradients: " << HMMPI::ToString(inds_grad, "%d");
		}

		// 2. Prepare ValCont and the full list of points
		ValCont *VC = 0;
		std::vector<std::vector<double>> Xfull = Xnew;												// Xfull will = points for func. vals + points for grads
		if (pm_aux->is_dataproxy())			// dataproxy/simproxy and simple proxy are updated differently
		{
			if (dynamic_cast<PM_SimProxy*>(pm_aux) == nullptr)
				VC = new ValContVecDouble(pm_aux->GetComm(), proxy_int->Data_ind(), Datanew);		// DATAPROXY; func. vals points
			else
				VC = new ValContSimProxy(pm_aux->GetComm(), proxy_int->Data_ind(), HMMPI::VecTranspose(Datanew));		// SIMPROXY; the last argument is vals[smry_len][N_points] (from RANKS-0)

			if (calc_grads())
				throw HMMPI::Exception("Currently data proxy training in HMC does not take gradients");
		}
		else
		{
			VC = new ValContDouble(pm_aux->GetComm(), ynew, std::vector<std::vector<double>>());	// simple PROXY; func. vals points

			if (calc_grads())
			{
				assert(Xnew.size() == Gradnew.size());
				HMMPI::VecAppend(Xfull, HMMPI::Reorder(Xnew, inds_grad));

				ValContDouble VCgrad(pm_aux->GetComm(), std::vector<double>(), HMMPI::Reorder(Gradnew, inds_grad));		// grads points
				dynamic_cast<ValContDouble*>(VC)->Add(VCgrad);
			}
		}

		// 3. Add new data to the proxy
		std::string msg = proxy_int->AddData(Xfull, VC, Nselect);
		resetVecs(is_initial);	// clear vectors
		delete VC;				// clear value container

		std::string proc_msg = pm_aux->proc_msg();				// proxy_int = pm_aux
		if (RNK == 0)			// note: this messaging will not go to the report file, only stdout
		{
			std::cout << "update type = " << update_type << "\n";
			std::cout << msg;
			std::cout << proc_msg;
		}

		// 4. Save the base proxy if necessary
		if (update_type == 1 && is_initial)
		{
			assert(proxy_int_base == nullptr);
			proxy_int_base = proxy_int->Copy();
			proxy_int_base->SetDumpFlag(-1);
		}
	}
}
//------------------------------------------------------------------------------------------
void HMC1::preprocess(const Mat &x)
{
	MPI_Bcast(&LF_steps, 1, MPI_INT, 0, MPI_COMM_WORLD);

	//if (proxy != nullptr) proxy->SetDumpFlag(1);	// DEBUG

	// Re-train proxy adding the staring point x0 (= x)
	std::vector<double> grad;
	if (calc_grads())
		grad = pm->ObjFuncGrad(pm->tot_par(x.ToVector())); 		// NOTE the fulldim gradient is taken

	updateVecs(x.ToVector(), of0, ModelledData0, grad);			// only RNK-0 needs input params
	trainProxy(true);			// initial single-point training

	if (pm_aux->is_proxy()) proxy_int->SetDumpFlag(-1);	// DEBUG

	LF.Recalc(pm_aux, x);		// recalculate mass matrix

#ifdef TESTMASSMATR
	FILE *massmatr_file = fopen(TESTMASSMATR, "w");
	if (massmatr_file != NULL)
	{
		LF.M.SaveASCII(massmatr_file, "%20.16g");
		fclose(massmatr_file);
	}
#endif
}
//------------------------------------------------------------------------------------------
void HMC1::proposal(const Mat &x)
{
	of0_aux = pm_aux->ObjFunc_ACT(x.ToVector());	// store for monitoring purposes

	if (RNK == 0)
	{
		p0 = LF.L * gen.RandN(x.ICount(), 1);		// randomize the momentum
		Kin0 = InnerProd(p0, LF.M / p0)/2;			// p0^t * M^(-1) * p0
		x1 = x;
		p1 = std::move(p0);
	}

	x1.Bcast(0, MPI_COMM_WORLD);
	p1.Bcast(0, MPI_COMM_WORLD);
	dist_rat = 0;
	lf_bounces = LF.Run2(pm_aux, x1, p1, LF_steps, dist_rat);	// *LEAPFROG* integration; input and output is LF-comm-sync;			<ORIGINAL>
																// don't use LF.Run1 here -- it has output with different implications in the following code

	//lf_bounces = LF.Run_SOL2(pm_aux, x1, p1, LF_steps, dist_rat);	// *quasi-LEAPFROG* integration; input and output is LF-comm-sync; 	<TEMP>

	//lf_bounces = LF.Run2(pm_grads, x1, p1, LF_steps, dist_rat); // DEBUG -- using a separate model for gradients
}
//------------------------------------------------------------------------------------------
void HMC1::process_new_point(const Mat &x)
{
	U0 = of0/2;
	of1_aux = pm_aux->ObjFunc_ACT(x1.ToVector());		// for monitoring purposes, and for inner acc. rate

	std::vector<double> grad;
	if (calc_grads())
		grad = pm->ObjFuncGrad(pm->tot_par(x1.ToVector())); 		// NOTE the fulldim gradient is taken

	updateVecs(x1.ToVector(), of1, ModelledData1, grad);			// only RNK-0 needs input params

	double dE_proxy;
	if (RNK == 0)
	{
		Kin1 = InnerProd(p1, LF.M / p1)/2;
		dE = -(of1/2+Kin1) + of0/2+Kin0;				// total energy change (1-0)
		dE_proxy = -(of1_aux/2+Kin1) + of0_aux/2+Kin0;	// will be used to determine inner acc. rate
	}

	MPI_Bcast(&dE_proxy, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	EUpd.incr_inner(HMMPI::Min(1, exp(dE_proxy)));
}
//------------------------------------------------------------------------------------------
void HMC1::save_output(const Mat &x)
{
	// x0 = x (RNK-0) now stores the current point (either old or updated)
	if (RNK == 0)
	{
		HMC_point point(x.ToVector(), x1.ToVector(), ModelledData0, of0, LF.eps, EUpd.acc_rate, lf_bounces, dist_rat, U0, of0_aux/2, of1/2, of1_aux/2, Kin0, Kin1, dE, LF.eig_min, LF.cond2, LF.m_adj, EUpd.acc_rate_inner, 0);	// dummy value 0
		point.SavePoint();
	}
}
//------------------------------------------------------------------------------------------
void HMC1::make_updates(Mat &x)
{
	//DEBUG	-============- Leap frog trajectory dump
//	if (i >= 1604 && i <= 1613)
//		LF.SetDumpFlag(i);
//	else
//		LF.SetDumpFlag(-1);
	//DEBUG

	// update proxy, mass matrix and epsilon
	if ((iter_counter+1) % upd_freq == 0)
	{
		if (std::find(proxy_dump_inds.begin(), proxy_dump_inds.end(), iter_counter) != proxy_dump_inds.end() && pm_aux->is_proxy())
		{
			if (RNK == 0)
				std::cout << "-------------------- Saving proxy-" << iter_counter << " to a dump file --------------------\n";		// this only goes to stdout, not report file
			proxy_int->SetDumpFlag(iter_counter);
		}

		if (iter_counter < burn_in)
			trainProxy(false);					// update the proxy

		if (pm_aux->is_proxy())
			proxy_int->SetDumpFlag(-1);

		x.Bcast(0, MPI_COMM_WORLD);

		if (iter_counter < burn_in)
			LF.Recalc(pm_aux, x);				// recalculate mass matrix only during burn-in
		else
			LF.resetBFGSVecs();

		double e_mult = EUpd.EpsMult(upd_freq);
		MPI_Bcast(&e_mult, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		if (iter_counter < burn_in)
			LF.eps *= e_mult;					// update epsilon
		if (LF.eps > max_step_eps)
			LF.eps = max_step_eps;
	}
}
//------------------------------------------------------------------------------------------
HMC1::HMC1(PhysModel *p, PhysModel *aux, Rand g, LeapFrog lf, EpsUpdate1 eu, int bi, int lf_steps, double maxstep, int upd, int Nadd, std::vector<int> ind_gpts, std::vector<int> ind_gcomp, std::vector<int> dump_inds, int upd_type) :
		MCMC(p, std::move(g), eu, bi, upd, Nadd),
		update_type(upd_type), proxy_int_base(nullptr), U0(0), of0_aux(0), of1_aux(0), Kin0(0), Kin1(0), dist_rat(0), lf_bounces(0), pm_aux(aux), LF(std::move(lf)), LF_steps(lf_steps), max_step_eps(maxstep),
		ind_grad_add_pts(std::move(ind_gpts)), ind_grad_comps(std::move(ind_gcomp)), proxy_dump_inds(std::move(dump_inds))
{
	LeapFrog::comm_compatible(MPI_COMM_WORLD, pm->GetComm(), "HMC1");
	LeapFrog::comm_compatible(MPI_COMM_WORLD, pm_aux->GetComm(), "HMC1");
	LeapFrog::comm_compatible(MPI_COMM_WORLD, LF.GetComm(), "HMC1");

	if (update_type != 0 && update_type != 1)
		throw HMMPI::Exception("update_type should be 0 or 1 in HMC1::HMC1");

	proxy_int = nullptr;
	if (pm_aux->is_proxy())				// including POSTERIOR<PROXY>
	{
		Proxy_train_interface *aux = dynamic_cast<Proxy_train_interface*>(pm_aux);
		if (aux == nullptr)
			throw HMMPI::Exception("Cannot convert 'pm_aux' which is '" + pm_aux->name + "' to Proxy_train_interface in HMC1::HMC1");
		if (update_type == 0)
			proxy_int = aux;			// borrow pointer
		else
		{
			proxy_int = aux->Copy();	// copy (to be deleted!)
			pm_aux = dynamic_cast<PhysModel*>(proxy_int);		// borrow
			assert(pm_aux != nullptr);
		}
	}

	// reset the output files
	MC_point point;
	point.ResetFiles();

	std::sort(ind_grad_add_pts.begin(), ind_grad_add_pts.end());			// [0, Nadd_fval_pts)
	std::sort(ind_grad_comps.begin(), ind_grad_comps.end());				// [0, fulldim)
	if (ind_grad_add_pts.size() > 0 && (ind_grad_add_pts[0] < 0 || *--ind_grad_add_pts.end() >= Nadd_fval_pts))
		throw HMMPI::Exception("ind_grad_add_pts out of range in HMC1::HMC1");
	if (ind_grad_comps.size() > 0 && (ind_grad_comps[0] < 0 || *--ind_grad_comps.end() >= pm_aux->ParamsDim()))
		throw HMMPI::Exception("ind_grad_comps out of range in HMC1::HMC1");
};
//------------------------------------------------------------------------------------------
HMC1::~HMC1()
{
	if (update_type == 1)
	{
		delete proxy_int;
		delete proxy_int_base;
	}
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
// HMCrej
//------------------------------------------------------------------------------------------
void HMCrej::process_new_point(const Mat &x)
{
	HMC1::process_new_point(x);

	MPI_Bcast(&lf_bounces, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (lf_bounces > 0)			// reject the point
	{
		dE = 0;
		x1 = x;
		of1 = of0;
		ModelledData1 = ModelledData0;
		EUpd.decr_count();		// decrease count now, because MH test will formally "accept" the rejected point (dE = 0)
	}
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
// SOL_HMC
//------------------------------------------------------------------------------------------
void SOL_HMC::preprocess(const Mat &x)
{
	HMC1::preprocess(x);

	if (RNK == 0)
		p0 = LF.L * gen.RandN(x.ICount(), 1);		// set initial velocity
	p0.Bcast(0, MPI_COMM_WORLD);
}
//------------------------------------------------------------------------------------------
void SOL_HMC::proposal(const Mat &x)
{
	pm_aux->SetIntTag(iter_counter);				// the iteration counter will be available to the simulator
	of0_aux = pm_aux->ObjFunc_ACT(x.ToVector());	// x is "x0", the initial point, i.e. a legal sample
	pm_aux->SetIntTag(-1);							// all subsequent calculations in proposal don't use smpl_tag

	// advance velocity p0 by OU process
	if (RNK == 0)
		p0 = sqrt(1 - ii*ii)*p0 + ii*(LF.L*gen.RandN(x.ICount(), 1));
	p0.Bcast(0, MPI_COMM_WORLD);

	if (RNK == 0)
	{
		Kin0 = InnerProd(p0, LF.M / p0)/2;			// p0^t * C^(-1) * p0
		KinQ0 = InnerProd(x, LF.M / x)/2;			// x0^t * C^(-1) * x0
		x1 = x;
		p1 = p0;
	}

	x1.Bcast(0, MPI_COMM_WORLD);
	p1.Bcast(0, MPI_COMM_WORLD);
	dist_rat = 0;
	//lf_bounces = LF.Run_SOL2(pm_aux, x1, p1, LF_steps, dist_rat);	// *quasi-LEAPFROG* integration; input and output is LF-comm-sync; (ORIGINAL)

	//lf_bounces = LF.Run_SOL(pm_aux, x1, p1, LF_steps, dist_rat);	// No bounces, only rejections (TEMP)

	lf_bounces = LF.RunHor2(pm_aux, x1, p1, LF_steps, dist_rat);	// Horowitz-type (TEMP)
}
//------------------------------------------------------------------------------------------
void SOL_HMC::process_new_point(const Mat &x)
{
	U0 = -99;
	double dE_proxy;
	std::vector<double> grad(pm->ParamsDim());				// fulldim gradient

	if (!pm->CheckLimits_ACT(x1.ToVector()))				// point has left the box: reject by setting very low dE
	{
		dE = dE_proxy = -1e6;
	}
	else
	{
		of1_aux = pm_aux->ObjFunc_ACT(x1.ToVector());		// for monitoring purposes, and for inner acc. rate
		if (calc_grads())
			grad = pm->ObjFuncGrad(pm->tot_par(x1.ToVector())); 		// NOTE the fulldim gradient is taken

		if (RNK == 0)
		{
			Kin1 = InnerProd(p1, LF.M / p1)/2;
			KinQ1 = InnerProd(x1, LF.M / x1)/2;
			dE = -(of1/2+Kin1+KinQ1) + of0/2+Kin0+KinQ0;				// total energy change (1-0)
			dE_proxy = -(of1_aux/2+Kin1+KinQ1) + of0_aux/2+Kin0+KinQ0;	// will be used to determine inner acc. rate
		}
	}

	updateVecs(x1.ToVector(), of1, ModelledData1, grad);				// only RNK-0 needs input params

	MPI_Bcast(&dE_proxy, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	EUpd.incr_inner(HMMPI::Min(1, exp(dE_proxy)));
}
//------------------------------------------------------------------------------------------
void SOL_HMC::accept_new_point()
{
	of0_aux = of1_aux;						// rank-0
	KinQ0 = KinQ1;
	p0 = p1;
}
//------------------------------------------------------------------------------------------
void SOL_HMC::reject_new_point()
{
	p0 = (-1)*p0;
}
//------------------------------------------------------------------------------------------
SOL_HMC::SOL_HMC(PhysModel *p, PhysModel *aux, Rand g, LeapFrog lf, EpsUpdate1 eu, int bi, int lf_steps, double maxstep, int upd, int Nadd, std::vector<int> ind_gpts, std::vector<int> ind_gcomp, std::vector<int> dump_inds, int upd_type, double i0) :
		HMC1(p, aux, g, lf, eu, bi, lf_steps, maxstep, upd, Nadd, ind_gpts, ind_gcomp, dump_inds, upd_type), ii(i0), KinQ0(-1), KinQ1(-1)
{
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
// RHMC1
//------------------------------------------------------------------------------------------
void RHMC1::preprocess(const Mat &x)	// re-train proxy adding the staring point x0 (= x)
{
	MPI_Bcast(&LF_steps, 1, MPI_INT, 0, MPI_COMM_WORLD);

	std::vector<double> grad;
	if (calc_grads())
		grad = pm->ObjFuncGrad(pm->tot_par(x.ToVector())); 		// NOTE the fulldim gradient is taken

	//if (proxy != nullptr) proxy->SetDumpFlag(1);	// DEBUG
	updateVecs(x.ToVector(), of0, ModelledData0, grad);			// only RNK-0 needs input params
	trainProxy(true);
	if (pm_aux->is_proxy()) proxy_int->SetDumpFlag(-1);	// DEBUG
}
//------------------------------------------------------------------------------------------
void RHMC1::proposal(const Mat &x)
{
	of0_aux = pm_aux->ObjFunc_ACT(x.ToVector());	// pm_aux-rank-0
	Lfg.ham0_msg("-----------RHMC1::proposal, calculate Mass Matrix, Ham0\n");
	HMMPI::Mat MMinit = Lfg.MM0(x.ToVector());		// this Mass Matrix is valid on rank_aux-0
	if (RNK == 0)
	{
		p0 = MMinit.Chol() * gen.RandN(x.ICount(), 1);			// randomize the momentum
		x1 = x;
		p1 = std::move(p0);

		// 'eig_min' and 'cond2' are only used for reporting; this calculation takes time, TURN OFF if necessary!
#ifdef TESTMASSMATR
		std::vector<double> eig = MMinit.EigVal(0, 1);
		eig_min = eig[0];
		std::vector<double> sg = MMinit.SgVal();
		cond2 = sg[0] / (*--sg.end());
#endif
	}

	x1.Bcast(0, MPI_COMM_WORLD);
	p1.Bcast(0, MPI_COMM_WORLD);

	Lfg.ham0_msg("-----------RHMC1::proposal, calculate Hamiltonian-0\n");
	Kin0 =  Lfg.of_ham0_ACT(x1.ToVector(), p1.ToVector()) - of0_aux/2;		// "kinetic energy"
	lf_bounces = Lfg.Run(x1, p1, LF_steps, step_eps);						// *GENERALIZED LEAPFROG* integration; input and output is Lfg-comm-sync;
	dist_rat = Lfg.dist_ratio;
}
//------------------------------------------------------------------------------------------
void RHMC1::process_new_point(const Mat &x)
{
	U0 = of0/2;
	of1_aux = pm_aux->ObjFunc_ACT(x1.ToVector());		// pm_aux-rank-0

	std::vector<double> grad;
	if (calc_grads())
		grad = pm->ObjFuncGrad(pm->tot_par(x1.ToVector())); 		// NOTE the fulldim gradient is taken

	updateVecs(x1.ToVector(), of1, ModelledData1, grad);		// only RNK-0 needs input params

	Lfg.ham0_msg("-----------RHMC1::process_new_point, calculate Hamiltonian-1\n");
	Kin1 = Lfg.of_ham1_ACT(x1.ToVector(), p1.ToVector()) - of1_aux/2;
	if (RNK == 0)
		dE = -(of1/2+Kin1) + of0/2+Kin0;				// total energy change (1-0)
}
//------------------------------------------------------------------------------------------
void RHMC1::save_output(const Mat &x)
{
	// x0 = x (RNK-0) now stores the current point (either old or updated)
	if (RNK == 0)
	{																																												// m_adj			// 0 - dummy
		HMC_point point(x.ToVector(), x1.ToVector(), ModelledData0, of0, step_eps, EUpd.acc_rate, lf_bounces, dist_rat, U0, of0_aux/2, of1/2, of1_aux/2, Kin0, Kin1, dE, eig_min, cond2, 0,	Lfg.iter_count, 0);
		point.SavePoint();
	}
	Lfg.iter_count = 0;
}
//------------------------------------------------------------------------------------------
void RHMC1::make_updates(Mat &x)
{
	// update proxy and epsilon
	if ((iter_counter+1) % upd_freq == 0)
	{
		if (std::find(proxy_dump_inds.begin(), proxy_dump_inds.end(), iter_counter) != proxy_dump_inds.end() && pm_aux->is_proxy())
		{
			if (RNK == 0)
				std::cout << "-------------------- Saving proxy-" << iter_counter << " to a dump file --------------------\n";		// this only goes to stdout, not report file
			proxy_int->SetDumpFlag(iter_counter);
		}

		if (iter_counter < burn_in)
			trainProxy(false);

		if (pm_aux->is_proxy())
			proxy_int->SetDumpFlag(-1);

		x.Bcast(0, MPI_COMM_WORLD);

		double e_mult = EUpd.EpsMult(upd_freq);
		MPI_Bcast(&e_mult, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		if (iter_counter < burn_in)
			step_eps *= e_mult;					// update epsilon
		if (step_eps > max_step_eps)
			step_eps = max_step_eps;
	}
}
//------------------------------------------------------------------------------------------
void RHMC1::trainProxy(bool is_initial)					// calls HMC1::trainProxy, and resets caches in Lfg
{
	Lfg.ham0_msg("-----------RHMC1::trainProxy------------------\n");
	HMC1::trainProxy(is_initial);
	Lfg.ResetCaches();
}
//------------------------------------------------------------------------------------------
RHMC1::RHMC1(PhysModel *p, PhysModel *aux, Rand g, NonlinearSystemSolver *sol, EpsUpdate1 eu, int bi, int lf_steps, double MM_shift, double eps, double maxeps, int upd, int Nadd, int LFG_maxref, std::vector<int> ind_gpts, std::vector<int> ind_gcomp, std::vector<int> dump_inds, int upd_type) :
		HMC1(p, aux, g, LeapFrog(), eu, bi, lf_steps, maxeps, upd, Nadd, ind_gpts, ind_gcomp, dump_inds, upd_type), Lfg(aux, sol, LFG_maxref, MM_shift), step_eps(eps), eig_min(0), cond2(0)
{
};
//------------------------------------------------------------------------------------------
// MMALA
//------------------------------------------------------------------------------------------
void MMALA::preprocess(const Mat &x)
{
	//if (proxy != nullptr) proxy->SetDumpFlag(1);		// DEBUG

	std::vector<double> grad;
	if (calc_grads())
		grad = pm->ObjFuncGrad(pm->tot_par(x.ToVector())); 		// NOTE the fulldim gradient is taken

	updateVecs(x.ToVector(), of0, ModelledData0, grad);			// only RNK-0 needs input params
	trainProxy(true);
	of0_aux = pm_aux->ObjFunc_ACT(x.ToVector());				// store for outer M-H test (rank-0)

	if (pm_aux->is_proxy()) proxy_int->SetDumpFlag(-1);	// DEBUG

	MMALA_steps = MMALA_accepts = 0;
}
//------------------------------------------------------------------------------------------
void MMALA::proposal(const Mat &x)
{
	Ham0.G.MsgToFile("/\\_________NEW_PROPOSAL_________/\\\n");		// separation mark for debug logging

	Mat x0 = x;
	double pr_of0 = pm_aux->ObjFunc_ACT(x0.ToVector());				// comm-RANKS-0 have 'pr_of0'

#ifdef TESTMASSMATR
	HMMPI::Mat MMinit = Ham0.G.Get(&Ham0, x0.ToVector());			// on rank-0
	if (RNK == 0)
	{
		std::vector<double> eig = MMinit.EigVal(0, 1);				// 'eig_min' and 'cond2' are only used for reporting; this calculation takes time, TURN OFF if necessary!
		eig_min = eig[0];
		std::vector<double> sg = MMinit.SgVal();
		cond2 = sg[0] / (*--sg.end());
	}
#endif

	for (int i = 0; i < Nsteps; i++)	// inner Nsteps done by MMALA -- these are done on the proxy model (pm_aux)
	{
		x0.Bcast(0, MPI_COMM_WORLD);

		Mat mu;
		if (type == 0)
			mu = Ham0.mu_MMALA.Get(&Ham0, std::pair<std::vector<double>, double>(x0.ToVector(), eps));		// sync on comm
		else if (type == 1)
			mu = Ham0.mu_simplMMALA.Get(&Ham0, std::pair<std::vector<double>, double>(x0.ToVector(), eps));	// sync on comm
		else if (type == 2)
			mu = Ham0.mu_MMALA_2.Get(&Ham0, std::pair<std::vector<double>, double>(x0.ToVector(), eps));	// sync on comm
		else
			throw Exception("Wrong 'type' in MMALA::proposal");

		Mat invU = Ham0.invU.Get(&Ham0, x0.ToVector());														// comm-rank-0

	#if 0			// debugging mu calculation, ONLY for full MMALA!
		Mat diff_mu = Ham0.calc_mu_alt(x0.ToVector(), eps) - mu;
//		std::cout << "Alternative calculations: mu     = " << mu.Tr().ToString("%-12.7g");
//		std::cout << "Alternative calculations: mu0-mu = " << diff_mu.Tr().ToString("%-12.7g");
		if (diff_mu.Norm2() > 1e-15)
			std::cout << "Alternative calculations: |mu0-mu|_2 = " << diff_mu.Norm2() << "\n";
	#endif

		if (RNK == 0)
			x1 = mu + eps * (invU * gen.RandN(x0.ICount(), 1));		// MMALA inner step; invU - on comm-RANKS-0
		x1.Bcast(0, MPI_COMM_WORLD);								// x1 - is the inner proposal

		int accept = 0;
		if (!pm->CheckLimits_ACT(x1.ToVector()))
			x1 = x0;												// if point violates bounds, reject it
		else
		{
			double pr_of1 = pm_aux->ObjFunc_ACT(x1.ToVector());		// comm-RANKS-0 have 'pr_of1'

			// "process new point"
			Ham0.G.MsgToFile("calculate logQ(x1->x0)\n");			// separation mark for debug logging
			if (type == 0)
				Kin0 = Ham1.MMALA_logQ_ACT(x1, x0, eps, Ham1.mu_MMALA);
			else if (type == 1)
				Kin0 = Ham1.MMALA_logQ_ACT(x1, x0, eps, Ham1.mu_simplMMALA);
			else if (type == 2)
				Kin0 = Ham1.MMALA_logQ_ACT(x1, x0, eps, Ham1.mu_MMALA_2);
			else
				throw Exception("Wrong 'type' in MMALA::proposal");

			Ham0.G.MsgToFile("calculate logQ(x0->x1)\n");			// separation mark for debug logging
			if (type == 0)
				Kin1 = Ham0.MMALA_logQ_ACT(x0, x1, eps, Ham0.mu_MMALA);
			else if (type == 1)
				Kin1 = Ham0.MMALA_logQ_ACT(x0, x1, eps, Ham0.mu_simplMMALA);
			else if (type == 2)
				Kin1 = Ham0.MMALA_logQ_ACT(x0, x1, eps, Ham0.mu_MMALA_2);
			else
				throw Exception("Wrong 'type' in MMALA::proposal");

			if (RNK == 0)
			{
				double dE_ = -pr_of1/2 + pr_of0/2 - Kin1 + Kin0;
				if (log(gen.RandU()) < dE_)							// Metropolis-Hastings test
				{
					accept = 1;
					x0 = x1;
					pr_of0 = pr_of1;
					MMALA_accepts++;
				}
			}

			x0.Bcast(0, MPI_COMM_WORLD);
			MPI_Bcast(&accept, 1, MPI_INT, 0, MPI_COMM_WORLD);

			if (accept)
				Ham0 = Ham1;										// NOTE when outer M-H test rejects, all previous caches will have been lost, and they are recalculated again
			else
				x1 = x0;				// x0 is sync
		}
	}	// inner Nsteps done by MMALA

	MMALA_steps += Nsteps;
	MPI_Bcast(&MMALA_accepts, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// output x1 is always withing the limits
}
//------------------------------------------------------------------------------------------
void MMALA::process_new_point(const Mat &x)		// take x = x0 !
{
	std::vector<double> grad;
	if (calc_grads())
		grad = pm->ObjFuncGrad(pm->tot_par(x1.ToVector())); 		// NOTE the fulldim gradient is taken

	updateVecs(x1.ToVector(), of1, ModelledData1, grad);			// only RNK-0 needs input params
	of1_aux = pm_aux->ObjFunc_ACT(x1.ToVector());					// for outer M-H test (rank-0)

	if (RNK == 0)
		dE = -of1/2 + of0/2 - (-of1_aux/2 + of0_aux/2);

	double inner_acc_rate;
	if (RNK == 0)
		inner_acc_rate = double(MMALA_accepts)/MMALA_steps;			// note that MMALA_accepts, MMALA_steps are reset only at make_updates()

	MPI_Bcast(&inner_acc_rate, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	EUpd.incr_inner(inner_acc_rate);
}
//------------------------------------------------------------------------------------------
void MMALA::save_output(const Mat &x)
{
	// x0 = x (RNK-0) now stores the current point (either old or updated)
	if (RNK == 0)
	{														// NOTE: EUpd.acc_rate - previous block, MMALA_accepts/MMALA_steps - current block
		HMC_point point(x.ToVector(), x1.ToVector(), ModelledData0, of0, eps, EUpd.acc_rate, -99, double(MMALA_accepts)/MMALA_steps, -99.99, of0_aux/2, of1/2, of1_aux/2, 0, 0, dE, eig_min, cond2, -99.99, EUpd.acc_rate_inner, 0);	// dummy value 0
		point.SavePoint();
	}
}
//------------------------------------------------------------------------------------------
void MMALA::accept_new_point()
{
	of0_aux = of1_aux;							// rank-0
}
//------------------------------------------------------------------------------------------
void MMALA::make_updates(Mat &x)
{
	// update proxy and epsilon
	if ((iter_counter+1) % upd_freq == 0)
	{
		if (std::find(proxy_dump_inds.begin(), proxy_dump_inds.end(), iter_counter) != proxy_dump_inds.end() && pm_aux->is_proxy())
		{
			if (RNK == 0)
				std::cout << "-------------------- Saving proxy-" << iter_counter << " to a dump file --------------------\n";		// this only goes to stdout, not report file
			proxy_int->SetDumpFlag(iter_counter);
		}

		if (iter_counter < burn_in)
			trainProxy(false);					// update the proxy

		if (pm_aux->is_proxy())
			proxy_int->SetDumpFlag(-1);

		double acc_mult = double(MMALA_accepts)/MMALA_steps;
		double e_mult = EUpd.EpsMult(upd_freq, acc_mult);
		MMALA_accepts = MMALA_steps = 0;
		MPI_Bcast(&e_mult, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		if (iter_counter < burn_in)
			eps *= e_mult;						// update epsilon
		if (eps > max_step_eps)
			eps = max_step_eps;
	}
}
//------------------------------------------------------------------------------------------
void MMALA::trainProxy(bool is_initial)			// calls HMC1::trainProxy, and resets caches in Ham0, Ham1
{
	Ham0.G.MsgToFile("-----------MMALA::trainProxy------------------\n");

	HMC1::trainProxy(is_initial);
	Ham0.ResetCaches();
	Ham1.ResetCaches();
}
//------------------------------------------------------------------------------------------
MMALA::MMALA(PhysModel *p, PhysModel *aux, Rand g, EpsUpdate1 eu, int bi, int ufreq, int Nadd, double MM_shift, double e, double maxeps, int steps, int Type, std::vector<int> ind_gpts, std::vector<int> ind_gcomp, std::vector<int> dump_inds, int upd_type) :
		HMC1(p, aux, g, LeapFrog(), eu, bi, 0, maxeps, ufreq, Nadd, ind_gpts, ind_gcomp, dump_inds, upd_type), MMALA_steps(0), MMALA_accepts(0), type(Type), eps(e), Nsteps(steps), Ham0(aux, MM_shift), Ham1(aux, MM_shift),
		lf_bounces(0), LF_steps(0), U0(0), dist_rat(0)
{
	eig_min = cond2 = 0;
}
//------------------------------------------------------------------------------------------
// I_MALA
//------------------------------------------------------------------------------------------
void I_MALA::preprocess(const Mat &x)
{
	//if (proxy != nullptr) proxy->SetDumpFlag(1);		// DEBUG

	std::vector<double> grad;
	if (calc_grads())
		grad = pm->ObjFuncGrad(pm->tot_par(x.ToVector())); 		// NOTE the fulldim gradient is taken

	updateVecs(x.ToVector(), of0, ModelledData0, grad);			// only RNK-0 needs input params
	trainProxy(true);
	of0_aux = pm_aux->ObjFunc_ACT(x.ToVector());				// rank-0

	if (pm_aux->is_proxy()) proxy_int->SetDumpFlag(-1);	// DEBUG

	MMALA_steps = MMALA_accepts = 0;

	zp = gen.RandN() > 0 ? 1 : -1;
	p0 = sample_p_I(x.ICount());
	MPI_Bcast(&zp, 1, MPI_INT, 0, MPI_COMM_WORLD);
	p0.Bcast(0, MPI_COMM_WORLD);
	fp0 = calc_fp_I(x, p0);
}
//------------------------------------------------------------------------------------------
Mat I_MALA::proposal_I(const Mat &xp)		// proposal for G=I, starting from 'xp'; forward/backward proposal - depending on 'zp'; the returned vector is in the doubled space
{
	Mat mu = Ham0.mu_Ifwd.Get(&Ham0, std::pair<std::vector<double>, std::pair<double, double>>(xp.ToVector(), std::pair<double, double>(eps, zp*alpha)));		// sync on comm

	Mat res;
	if (RNK == 0)
		res = mu + eps * (gen.RandN(xp.ICount(), 1));
	res.Bcast(0, MPI_COMM_WORLD);

	return res;
}
//------------------------------------------------------------------------------------------
double I_MALA::lnP_I(const PM_FullHamiltonian &ham, const Mat &xp, const Mat &xpnew, double eps, int sign)	// log of P for G=I; forward/backward P - depending on 'sign'; 'ham' should be Ham0 for xp={x0,p0}, Ham1 for xp={x1,p1}
{
	assert(xp.ICount() == xpnew.ICount());
	assert(xp.JCount() == xpnew.JCount() && xp.JCount() == 1);
	const size_t actdim_dbl = xp.ICount();

	Mat mu = ham.mu_Ifwd.Get(&ham, std::pair<std::vector<double>, std::pair<double, double>>(xp.ToVector(), std::pair<double, double>(eps, sign*alpha)));		// sync
	HMMPI::Mat diff = mu - xpnew;

	HMMPI::Mat MMdiff = diff;
	double lndet = 0;

	return 0.5*lndet - double(actdim_dbl)*log(eps) - 1/(2*eps*eps)*InnerProd(diff, MMdiff);
}
//------------------------------------------------------------------------------------------
Mat I_MALA::sample_p_I(size_t s)	// generate 'p', for G=I
{
	return gen.RandN(s, 1);
}
//------------------------------------------------------------------------------------------
double I_MALA::calc_fp_I(const Mat &x, const Mat &p)	// calculate fp0 or fp1, for G=I; this also includes 0.5*lndet(G(theta))
{
	return 0.5*InnerProd(p, p);
}
//------------------------------------------------------------------------------------------
void I_MALA::proposal(const Mat &x)
{
	Ham0.G.MsgToFile("/\\_________NEW_PROPOSAL_________/\\\n");		// separation mark for debug logging

#ifdef TESTMASSMATR
//	HMMPI::Mat MMinit = Ham0.G.Get(&Ham0, x.ToVector());			// on rank-0
//	if (RNK == 0)
//	{
//		std::vector<double> eig = MMinit.EigVal(0, 1);				// 'eig_min' and 'cond2' are only used for reporting; this calculation takes time, TURN OFF if necessary!
//		eig_min = eig[0];
//		std::vector<double> sg = MMinit.SgVal();
//		cond2 = sg[0] / (*--sg.end());
//	}
#endif

	std::vector<double> xp1 = proposal_I(x||p0).ToVector();			// sync
	x1 = Mat(std::vector<double>(xp1.begin(), xp1.begin() + xp1.size()/2));
	p1 = Mat(std::vector<double>(xp1.begin() + xp1.size()/2, xp1.end()));

	if (!pm->CheckLimits_ACT(x1.ToVector()))			// if point violates bounds, reject it
	{
		x1 = x;
		p1 = p0;
		ModelledData1 = ModelledData0;
		of1 = of0;

		fp1 = fp0;
		zp = -zp;
		EUpd.decr_count();								// decrease count now, because MH test will formally "accept" the rejected point (x_new - x_old = 0)
		MMALA_accepts--;

		std::cout << "DEBUG boundary rejection x = " << x.Tr().ToString();	// DEBUG	TODO check
	}

	MMALA_steps++;
}
//------------------------------------------------------------------------------------------
void I_MALA::process_new_point(const Mat &x)		// take x = x0 !
{
	std::vector<double> grad;
	if (calc_grads())
		grad = pm->ObjFuncGrad(pm->tot_par(x1.ToVector())); 		// NOTE the fulldim gradient is taken

	updateVecs(x1.ToVector(), of1, ModelledData1, grad);			// only RNK-0 needs input params
	of1_aux = pm_aux->ObjFunc_ACT(x1.ToVector());					// for outer M-H test (rank-0)
	fp1 = calc_fp_I(x1, p1);

	double P0, P1;							// log of P numerator and P denominator
	P0 = lnP_I(Ham1, x1||p1, x ||p0, eps, -zp);
	P1 = lnP_I(Ham0, x ||p0, x1||p1, eps, zp);

	if (RNK == 0)
		dE = -of1/2 + of0/2 - fp1 + fp0 + (P0 - P1);

	double inner_acc_rate;
	if (RNK == 0)
		inner_acc_rate = exp(-of1_aux/2 + of0_aux/2 - fp1 + fp0 + (P0 - P1));

	if (x1.ToVector() == x.ToVector())		// both are sync
		inner_acc_rate = 0;					// rejection (out of boundaries) took place

	MPI_Bcast(&inner_acc_rate, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	EUpd.incr_inner(inner_acc_rate);
}
//------------------------------------------------------------------------------------------
void I_MALA::accept_new_point()
{
	of0_aux = of1_aux;						// rank-0
	Ham0 = Ham1;
	p0 = p1;
	fp0 = fp1;
	MMALA_accepts++;
}
//------------------------------------------------------------------------------------------
void I_MALA::reject_new_point()
{
	zp = -zp;
}
//------------------------------------------------------------------------------------------
I_MALA::I_MALA(PhysModel *p, PhysModel *aux, Rand g, EpsUpdate1 eu, int bi, int ufreq, int Nadd, double MM_shift, double e, double maxeps, int steps, int Type, std::vector<int> ind_gpts, std::vector<int> ind_gcomp, std::vector<int> dump_inds, int upd_type, double Alpha) :
		MMALA(p, aux, g, eu, bi, ufreq, Nadd, MM_shift, e, maxeps, steps, Type, ind_gpts, ind_gcomp, dump_inds, upd_type),
		zp(1), fp0(0), fp1(0), alpha(Alpha), Kin0(0), Kin1(0)
{
	eig_min = cond2 = 0;
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------

} // namespace HMMPI

