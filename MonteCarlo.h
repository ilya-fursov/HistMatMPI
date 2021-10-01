/*
 * MonteCarlo.h
 *
 *  Created on: 26 Jul 2016
 *      Author: ilya fursov
 */

#ifndef MONTECARLO_H_
#define MONTECARLO_H_

//#define HMC_BFGS			// BFGS accumulation for mass matrix in HMC
//#define LF_GENERALIZED_OUT_X "x_LeapfrogGeneralized.txt"		// output of x for each LFG step

#include "mpi.h"
#include "MathUtils.h"
#include "PhysModels.h"
#include "GradientOpt.h"
#include <random>

namespace HMMPI
{
//------------------------------------------------------------------------------------------
// class for random number generation
// the seed is sync over MPI_COMM_WORLD
//
// THERE IS also a similar class HMMPI::RandNormal in MathUtils.h
//------------------------------------------------------------------------------------------
class Rand
{
protected:
	std::default_random_engine gen;					// generator
	std::uniform_real_distribution<double> uni;		// uniform distribution
	std::normal_distribution<double> norm;			// normal distribution
	unsigned int seed;								// sync over MPI_COMM_WORLD
public:
	Rand(unsigned int s = 0, double a = 0, double b = 1, double mu = 0, double sigma = 1);	// seed, parameters for uniform distribution, parameters for normal distribution
																							// if seed == 0, it will be initialized by time (on RANK-0); seed is sync over MPI_COMM_WORLD
						// the four functions below don't use MPI
	double RandU();		// uniform random number
	double RandN();		// normal random number
	Mat RandU(int I0, int J0);		// I0 x J0 matrix with uniform random numbers
	Mat RandN(int I0, int J0);		// I0 x J0 matrix with normal random numbers
	unsigned int Seed(){return seed;};		// sync over MPI_COMM_WORLD
};
//------------------------------------------------------------------------------------------
// class for making leapforg integration;
// also stores and updates the mass matrix;
// TO BE USED on all ranks in MPI_COMM_WORLD (MPI_COMM_WORLD is also a good choice for "comm")
// communicators should be compatible:
// LF-comm-rank == 0 -> pm-comm-rank == 0, so that LF-comm-rank-0 gets correct gradients etc													(*)
// LF-comm-rank == NULL -> pm-comm-rank == NULL, so that "pm" is used on all ranks where it's defined (pm-comm-rank != NULL)					(*)
//------------------------------------------------------------------------------------------
class LeapFrog
{
private:
	mutable Mat ssM;			// symmetric square root of M - cached value; filled on first call of SymSqrtM() after 'M' change; (comm-RANKS-0 only)
	mutable bool ssM_cached;	// if 'true', then 'ssM' stores up-to-date value (comm-RANKS-0 only)
								// it is set to 'false' in ctor and Recalc()
protected:
	const bool is_dummy = false;		// flag showing if the LeapFrog object is dummy
	MPI_Comm comm;
	double nu;			// in mass matrix update, its min eigenvalue is made to be >= 'nu' (RANK-0 only)
	std::string bounce_type;	// NEG, CHOL, EIG, HT
	std::string MM_type;		// HESS, FI, UNITY, MAT, BFGS
	Mat const_mat;				// mass matrix for "MAT" case

	const char dump_file[100] = "leapfrog_dump_%d.txt";
	int dump_flag;				// if this flag != -1, then LeapFrog will output (in Run2) different quantities to file "leapfrog_dump_$(dump_flag).txt" -- for debug purposes

									// the two vectors below are active-dim; referenced on comm-RANKS-0; mutable because they are changed in Run2()
	mutable std::vector<Mat> Xk;		// coordinate vectors for BFGS update
	mutable std::vector<Mat> Gk;		// gradient vectors for BFGS update
	Mat Sk, Ck, Hk, Bk;					// matrices for BFGS iterative updates, Hk ~ Hessian, Bk ~ inverse Hessian
	double gammaBFGS;					// for initial matrix (H0 = gammaBFGS*I)

													// call the velocity bouncing on comm-RANKS-0 only
	void BounceVel1(Mat &vel, int ci) const;		// velocity bounce #1 w.r.t. constraint 'ci': component 'ci' is simply negated
	void BounceVel2(Mat &vel, int ci, const Mat &factM) const;	// velocity bounce #2 w.r.t. constraint 'ci': uses mass matrix factor 'factM' (M = factM * factM') and Householder transformation;
																// this method preserves volume, energy, is time-reversible, and bounces to the proper semi-space
	void BounceVel3(Mat &vel, int ci) const;		// velocity bounce #3 w.r.t. constraint 'ci' -- equivalent to method #2, but works with 'M' directly (does not use decomposition)
	const Mat &SymSqrtM() const;		// symmetric square root of M (comm-RANKS-0 only)
public:
	Mat M;				// mass matrix (comm-RANKS-0 only)
	Mat L;				// its Cholesky factor (comm-RANKS-0 only)
	double eps;			// step size (comm-RANKS-0 only)
	double eig_min;		// min eigenvalue of 'M' -- for reporting, calculated in Recalc()
	double cond2;		// condition number (in 2-norm) of 'M' -- for reporting, calculated in Recalc()
	double m_adj;		// number "-d" from M = Hess - d*I -- for reporting, calculated in Recalc()

	LeapFrog() : ssM_cached(false), is_dummy(true), comm(MPI_COMM_WORLD), nu(0), dump_flag(-1), gammaBFGS(0), eps(0), eig_min(0), cond2(0), m_adj(0) {};		// default ctor creates a dummy LeapFrog object which should not be used
	LeapFrog(MPI_Comm c, double n, double e, double gamma, std::string bounce, std::string mm, Mat m);
	void Recalc(PhysModel *pm, const Mat &x);			// recalculate 'M' and 'L' for current point 'x', using pm->ObjFuncHess_ACT, or ObjFuncFisher_ACT, or unity matrix, or "MAT";
														// call on ALL RANKS; both parameters should be defined on ALL RANKS ("x" - sync)
	void updateBFGSVecs(const Mat &x, const Mat &g) const;		// updates vectors Xk, Gk with the new point
	void resetBFGSVecs() const ;								// resets the above vectors
	int Run1(PhysModel *pm, Mat &x, Mat &p, int N) const;	// makes N-step leapfrog update; x - coordinate vector, p - momentum vector;		*** OBSOLETE ***
															// "pm" is only used for ObjFuncGrad_ACT and CheckLimits_ACT; note that "pm" state may change;
															// if the point hits the boundary, the run stops (before reaching N steps) and the last point within the boundaries is returned;
															// the function RETURNS the number of steps actually completed (sync over "comm");
															// make sure this function is called on ALL RANKS; pm->comm should be compatible with 'comm';
															// input and output "x", "p", "N" are sync on 'comm'
															// __NOTE__ prior distribution is not used here!
	int Run2(PhysModel *pm, Mat &x, Mat &p, int N, double &dr) const;		// same as Run1, but if point hits the boundary, it bounces; the RETURN value is the number of bounces (sync over "comm");
																// makes N-step leapfrog update; x - coordinate vector, p - momentum vector;
																// "pm" is only used for ObjFuncGrad_ACT and CheckLimits_ACT; note that "pm" state may change;
																// make sure this function is called on ALL RANKS; pm->comm should be compatible with 'comm';
																// input and output "x", "p", "N", "dr" are sync on 'comm'
																// NEW 02.11.2016: distance ratio during LF run is saved to "dr" (sync over 'comm')

	int Run_SOL(PhysModel *pm, Mat &x, Mat &v, int N, double &dr) const;	// Hamiltonian integration for SOL-HMC (not a leap frog!).
																// "pm" should be the proxy model; x - coordinate vector, v - velocity vector, N - number of steps (of size "eps"), dr - output distance ratio
																// "M" is used as covariance (see SOL-HMC definition)
																// All other comments from Run2() apply. The algorithm is allowed to leave the boundary.
	int Run_SOL2(PhysModel *pm, Mat &x, Mat &v, int N, double &dr) const;	// Same as Run_SOL, but with bounces on the boundaries. Returns the number of bounces.
	int RunHor2(PhysModel *pm, Mat &x, Mat &p, int N, double &dr) const;	// Same as Run2, but is designed for Horowitz-type sampler

	static void comm_compatible(MPI_Comm c, MPI_Comm pm_comm, std::string where);	// throws exception if "c" and "pm_comm" are not compatible in the sense (*); call it on all ranks of MPI_COMM_WORLD
																					// 'where' is a string for message, e.g. = "LeapFrog"
	MPI_Comm GetComm() const {return comm;};
	void SetDumpFlag(int f){dump_flag = f;};
};
//------------------------------------------------------------------------------------------
// class for making generalized leapforg integration;
// handles everything related to the mass matrix;
// communicator is taken from "pm"
//------------------------------------------------------------------------------------------
class LeapFrogGeneralized
{
protected:
	typedef std::pair<int, double> bound;		// used in trajectory bouncing from the bounds, <constr_index, constr_value>

	MPI_Comm comm;				// 'comm' is taken from "pm", bounce type is always HT, mass matrix - always FI

								// Ham0, Ham1 (mathematically same models, but with different cache states) guide the whole generalized leap frog trajectory
	PM_FullHamiltonian Ham0;	// for Hamiltonians at x(tau), uses caching
	PM_FullHamiltonian Ham1;	// for Hamiltonians at x(tau + eps), uses caching

	NonlinearSystemSolver *solver;			// used for making implicit steps
	const int solver_init_pt;	// initial point for steps '...p0' (eq1) and '...x' (eq2); 1 - initial = current point, 2 - initial = "solution of explicit analogue of equation"
	int max_refine;				// max refinement of 'eps' step is 2^max_refine; equals MIN(max_refine3, max_eps_refine)
	int max_refine3;			// for make_step_x_bounce_3
	const double scale_factor_eps;			// used in make_step_p0, make_step_x
	const int max_eps_refine;				// used in definition of max_refine
																	// input and output for the following make step procedures is sync on comm; the vectors are ACT-DIM
	Mat make_step_p0(const Mat &x, const Mat &p, double eps, const Mat *init0 = NULL);	// p(tau) -> p(tau + eps/2), implicit equation; if init0 != NULL, it is always used as initial point
	Mat make_step_p0_ref(const Mat &x, const Mat &p, double eps);	// p(tau) -> p(tau + eps/2), implicit equation; in case of failure init points from refined stepping are tried
	Mat make_step_x(const Mat &x, const Mat &p, double eps, const Mat *init0 = NULL);	// x(tau) -> x(tau + eps), implicit equation, no boundary bouncing; if init0 != NULL, it is always used as initial point
	Mat make_step_x_ref(const Mat &x, const Mat &p, double eps, int minref = 0);		// x(tau) -> x(tau + eps), no boundary bouncing; in case of failure, init points from refined stepping are tried; 'minref' forces to make at least this refinement for init point
	void bounce_p(const Mat &x, Mat &p, int con_ind);				// perform velocity bounce at point 'x' w.r.t. coordinate 'con_ind', updating momentum 'p'
	double make_step_x_bounce_1(Mat &x, Mat &p, const double eps, int &bounce_count);	// x(tau) -> x(tau + eps'), boundary bouncing, 0 < eps' <= eps, exits after first bounce (updating "p"), or after reaching the step size "eps" (no bounce); returns the remaining step eps - eps'
	double make_step_x_bounce_2(Mat &x, Mat &p, const double eps, int &bounce_count);	// x(tau) -> x(tau + eps'), -"-, employs the refinement strategy in case of solver failure
	double make_step_x_bounce_3(Mat &x, Mat &p, const double eps, int &bounce_count);	// x(tau) -> x(tau + eps'), -"-, employs the bisection strategy in case of solver failure

	Mat make_Nstep_xint(Mat x, const Mat &p, double eps0, size_t N, int &con_ind, double &eps_part);		// makes at most N steps (size = eps0) for 'x', returns the first found "linear" intersection
		// (also filling 'con_ind' - intersected bound, and 'eps_part' - cumulative step length taken till intersection), or the final point; this function is for making the initial approximation candidates for make_step_x_bounce_2
	Mat make_Nstep_p0(const Mat &x, Mat p, double eps0, size_t N);	// makes N steps (size = eps0) for 'p'; this function is for making the initial approximation candidates for make_step_p0_ref
	Mat make_Nstep_x(Mat x, const Mat &p, double eps0, size_t N);	// makes N steps (size = eps0) for 'x'; this function is for making the initial approximation candidates for make_step_x_ref
	void make_bounds(const Mat &xcurr, std::vector<bound> &bounds) const;		// fills array of bounds (2*ACTDIM), ordered by increasing distance from 'xcurr' to the boundaries; bounds[0] is intact, and its value is not repeated in bounds[i>0]
	Mat make_step_p1(const Mat &x, const Mat &p, double eps);		// p(tau + eps/2) -> p(tau + eps), explicit equation
	std::string bounds_to_str(const std::vector<bound> &bounds) const;
	void BounceVel(const Mat &MM, Mat &vel, int ci) const;			// velocity bounce w.r.t. constraint 'ci', works with 'MM' directly (does not use decomposition)
	bool limits_and_proj(const std::vector<double> &x, std::vector<double> &x_proj, int &con_ind) const;	// returns FALSE if "x" violates the bounds; in this case sets "con_ind" to the index of the bound with strongest violation
																											// and sets "x_proj" to "x" with violated coordinates set to the corresponding exact bound values
public:
	int iter_count;		// counts solver iterations, filled by make_step_p0(), make_step_x(); set it to 0 when appropriate;
	double dist_ratio;	// distance ratio during LF run

	LeapFrogGeneralized(PhysModel *pm, NonlinearSystemSolver *sol, int maxref, double MM_shift);			// "pm" should be a model with gradients and FI (e.g. a proxy)
	int Run(Mat &x, Mat &p, int N, double eps);		// N-step generalized leapfrog update (each step of size 'eps'); x - starting coordinate vector, p - starting momentum vector;
													// returns the number of bounces; updates 'iter_count', 'dist_ratio'
													// all inputs and outputs are sync over 'comm'
	MPI_Comm GetComm() const {return comm;};

	HMMPI::Mat MM0(const std::vector<double> &x) const {return Ham0.G.Get(&Ham0, x);};		// mass matrix from Ham0; result is on comm-rank-0
	double of_ham0_ACT(const std::vector<double> &x, const std::vector<double> &p);			// objective function from Ham0, used to calculate "kinetic energy"; 'x', 'p' should be sync on comm; result is on comm-rank-0
	double of_ham1_ACT(const std::vector<double> &x, const std::vector<double> &p);			// -"- for Ham1
	void ham0_msg(const std::string &msg) const {Ham0.G.MsgToFile(msg);};
	void ResetCaches() {ham0_msg("*** LeapFrogGeneralized::ResetCaches ***\n"); Ham0.ResetCaches(); Ham1.ResetCaches();};							// resets caches for Ham0, Ham1
};
//------------------------------------------------------------------------------------------
// class with basic information on a single point of Markov Chain
//------------------------------------------------------------------------------------------
class MC_point
{
protected:
	const char par_file[100] = "HMC_Parameters.txt";
	const char par_all_file[100] = "HMC_Parameters_all.txt";
	const char mod_data_file[100] = "HMC_ModelledData.txt";
	const char etc_file[100] = "HMC_ObjFunc_etc.txt";

	std::vector<double> X;		// coords (active dim)
	std::vector<double> Xall;	// coords of all points, including rejected ones (active dim)
	std::vector<double> Mod;	// modelled data
	double y;					// function value

	double eps;					// epsilon for the block (cf. 'upd_freq')
	double acc_rate;			// acc_rate for the block (cf. 'upd_freq')

	virtual void save_point_2();	// saves additional information (o.f. value, etc) on the current point to the file
public:
	MC_point() : y(0), eps(0), acc_rate(0) {};			// default - empty ctor
	MC_point(std::vector<double> x, std::vector<double> xall, std::vector<double> md, double of, double e, double acc) : X(std::move(x)), Xall(std::move(xall)), Mod(std::move(md)), y(of), eps(e), acc_rate(acc) {};
	virtual ~MC_point(){};
	void ResetFiles();			// resets the output files
	void SavePoint();			// saves the current point to the file(s)
};
//------------------------------------------------------------------------------------------
// class with some additional point information from HMC1
//------------------------------------------------------------------------------------------
class HMC_point : public MC_point
{
protected:
	int lf_bounces;				// number of bounces
	double dist_ratio;			// distance travelled by LF divided by max distance during LF route

	double U0;			// potential energy before LF
	double U0_aux;		// potential energy of pm_aux before LF
	double U1;			// potential energy after LF
	double U1_aux;		// potential energy of pm_aux after LF
	double Kin0;		// kinetic energy before LF
	double Kin1;		// kinetic energy after LF
	double dE;			// change of full energy by LF -- this value is used in MH test
	double M_emin;		// min eigenvalue of M
	double M_cond2;		// condition number of M (in 2-norm)
	double m_adj;		// LF.m_adj - number from M eigenvalues adjustment
	double val0;		// some value-0
	double val1;		// some value-1

	virtual void save_point_2();
public:
	HMC_point(std::vector<double> x, std::vector<double> xall, std::vector<double> md, double of, double e, double a, int lf, double dr,
		double u0, double u0aux, double u1, double u1aux, double k0, double k1, double de, double memin, double mcon, double madj, double v0, double v1) :
		MC_point(std::move(x), std::move(xall), std::move(md), of, e, a), lf_bounces(lf), dist_ratio(dr),
		U0(u0), U0_aux(u0aux), U1(u1), U1_aux(u1aux), Kin0(k0), Kin1(k1), dE(de), M_emin(memin), M_cond2(mcon), m_adj(madj), val0(v0), val1(v1) {};
};
//------------------------------------------------------------------------------------------
// (SPECIFIC) class for updating epsilon in HMC and counting the accepted models
//------------------------------------------------------------------------------------------
class EpsUpdate1
{
protected:
	const double acc_targ = 0.70;		// target acceptance rate
	const double alpha = 1.0;
	const double beta = 10.0;
	const bool acc_from_inner = false;	// if true, 'eps' multiplier will be calculated from 'acc_rate_inner' (for HMC)
	int acc_block;     			// number of models accepted in current block
	double acc_block_inner;		// number of models accepted in current block in inner iteration

public:
	int acc_total;     			// number of models accepted in total
	double acc_rate;			// current acceptance rate (in block)
	double acc_rate_inner;		// current acceptance rate (in block) in inner iteration

	EpsUpdate1(double targ, double a, double b, bool acc_inner) : acc_targ(targ), alpha(a), beta(b), acc_from_inner(acc_inner), acc_block(0), acc_block_inner(0.0), acc_total(0), acc_rate(0.0), acc_rate_inner(0.0){};
	void AccModel(){acc_block++; acc_total++;};			// accept a model
	void incr_inner(double d){acc_block_inner += (IsNaN(d) ? 0 : d);};		// used in HMC (etc)
	void decr_count(){acc_block--; acc_total--;};		// used in RWM, PCN
	double EpsMult(int block_size, double acc_mult = 1.0);		// calculates acc_rate, acc_rate_inner, returns multiplier for epsilon (depending on 'acc_from_inner');
																// sets 'acc_block', 'acc_block_inner' to 0 (to work with the next block)
																// 'block_size' is the size of block which has been processed
																// 'acc_rate' is multiplied by 'acc_mult'; 'acc_rate_inner' is not multiplied
};
//------------------------------------------------------------------------------------------
// abstract class for MCMC sampling (uses Metropolis-Hastings test)
// the proposal step should be defined in derived classes
//------------------------------------------------------------------------------------------
class MCMC
{
protected:
	int RNK;			// from MPI_COMM_WORLD
	int rank_pm;		// from pm->comm
	Rand gen;			// random number generator

	int iter_counter;	// iteration counter
	double of0, of1;	// o.f. at x0, x1
	double dE;			// quantity for MH test: e.g. -of1/2 + of0/2, only referenced on RNK-0
	std::vector<double> ModelledData0, ModelledData1;		// m.d. at x0 and x1, only contains useful info on rank_pm-0
	Mat x1;				// active-dim vector

	PhysModel *pm;		// model used for Metropolis-Hastings testing; likelihood L = a*exp(-pm/2); 'pm' is used for ObjFunc, ModelledData, and bounds checking
	EpsUpdate1 EUpd;	// object for counting accepted models [and updating epsilon] (RNK-0)
	int burn_in;		// number of burn-in iterations: for HMC after burn-in eps and proxy are fixed; for RWM, pCN after burn-in eps is fixed

	int upd_freq;				// HMC: 'pm_aux', mass matrix and LF.eps are updated every 'upd_freq' iterations (if 'pm_aux' is not PM_Proxy*, it is not updated)
								// RWM: 'eps' is updated every 'upd_freq' iterations
	int Nadd_fval_pts;	// max number of points with func. vals added on each proxy update

	// use x = x0
	virtual void preprocess(const Mat &x) = 0;			// preparations (e.g. calc mass matrix) before the main loop; may be empty
	virtual void proposal(const Mat &x) = 0;			// propose a move from the current point (x0,...) to the new one (x1,...)
	virtual void process_new_point(const Mat &x) = 0;	// additional processing for the new point (e.g. take data for proxy update); should at least calculate 'dE' on RNK-0; "x" can be e.g. = x0;
	virtual void save_output(const Mat &x) = 0;			// save output to files
	virtual void accept_new_point(){};					// function called when a point is accepted
	virtual void reject_new_point(){};					// function called when a point is rejected
	virtual void make_updates(Mat &x) = 0;				// update: e.g. proxy, mass matrix, eps; may be empty

public:
	MCMC(PhysModel *p, Rand g, EpsUpdate1 eu, int bi, int ufreq, int Nadd);
	virtual ~MCMC(){};
	int Run(int count, Mat &x0, std::string *msg_time);	// sample 'count' points via MCMC, starting from 'x0' (active-dim); call this function on ALL RANKS in MPI_COMM_WORLD
									// 'x0' is updated on output; the function returns the total number of accepted models;
									// all input params should be sync on ALL RANKS (same is true for the returned value and output 'x0', 'msg_time')
									// msg_time is an optional string to store message on burn-in and main run time
};
//------------------------------------------------------------------------------------------
// class for simple Random Walk Metropolis sampling
// proposal = eps * N(0, I), or eps * N(0, C);
//------------------------------------------------------------------------------------------
class RWM1 : public MCMC
{
protected:
	double eps;		// for proposal step
	Mat C;			// proposal = current + eps*N(0, C); if C == empty, unity matrix is used
	Mat L;			// Chol(C), L*L' = C

	virtual void preprocess(const Mat &x){};
	virtual void proposal(const Mat &x);
	virtual void process_new_point(const Mat &x);	// take x = x0 !
	virtual void save_output(const Mat &x);
	virtual void make_updates(Mat &x);

public:
	RWM1(PhysModel *p, Rand g, EpsUpdate1 eu, int bi, int ufreq, int Nadd, double e, Mat cov);		// upper triangular part of 'cov' is used
};
//------------------------------------------------------------------------------------------
// class for preconditioned Crank-Nicholson MCMC
// inherits from RWM1; PhysModel should be PM_Posterior to provide Cpr; C is not used
// proposal is V = (1 - eps^2)^0.5 *x_curr + eps*W, where W ~ N(0, Cpr)
// eps should be in [0, 1]
//------------------------------------------------------------------------------------------
class pCN1 : public RWM1
{
protected:
	const Mat C;		// not to be used
	// L*L' = Cpr

	virtual void proposal(const Mat &x);
	virtual void process_new_point(const Mat &x);	// take x = x0 !
public:
	pCN1(PhysModel *p, Rand g, EpsUpdate1 eu, int bi, int ufreq, int Nadd, double e);
};
//------------------------------------------------------------------------------------------
// class for HMC sampling
// HMC1.Run() updates mass matrix, proxy model, and leapfrog step 'eps' every 'upd_freq' iterations
// so, the resulting chain may not be a single Markov Chain;
// TO BE USED on all ranks in MPI_COMM_WORLD
//------------------------------------------------------------------------------------------
class HMC1 : public MCMC
{
private:
	// the variables are referenced on ranks:
	// of0, of1, ModelledData0, Kin0, Kin1, EUpd -- RNK-0
	// LF.M, LF.L, LF.eps -- LF-comm-RANKS-0
	// x0, x1, p0, p1 -- either all ranks (WORLD), or RNK-0, with Bcasing where appropriate
	// other variables -- all ranks

	const int update_type;						// 0 - on every update "u" add Nadd_fval_pts points; 1 - on every update add u*Nadd_fval_pts points to the base-proxy
	Proxy_train_interface *proxy_int_base;		// this PROXY, or POSTERIOR<PROXY> is used for update_type == 1; it is trained by the Sobol points, and the single initial point (until fully trained, proxy_int_base = NULL)
																// vectors for proxy update; used on all ranks (MPI_COMM_WORLD)
	std::vector<std::vector<double>> Xnew;						// coords, full-dim
	std::vector<double> ynew;									// o.f. vals
	std::vector<std::vector<double>> Datanew;					// data vals
	std::vector<std::vector<double>> Gradnew;					// gradients of o.f. (to be used with simple proxy)

protected:
	double U0, of0_aux, of1_aux, Kin0, Kin1, dist_rat;		// pm(x0)/2, pm_aux(x0), pm_aux(x1); kinetic energy in p0, p1; distrance ratio; 'U0' is not updated in M-H accept
	Mat p0, p1;									// momentum in x0, x1
	int lf_bounces;								// number of bounces in the current leap frog trajectory

	PhysModel *pm_aux;					// model for gradients and Hessians (e.g. PROXY, or POSTERIOR<PROXY>), which may be different from "pm"; NOTE pm_aux should not be cast to PM_Proxy!
	Proxy_train_interface *proxy_int;	// = pm_aux, used for proxy training

	// Case:											pm_aux				proxy_int			proxy_int_base
	// pm_aux is not PROXY, or POSTERIOR<PROXY>			used for HMC		NULL				NULL
	// pm_aux is PROXY, or POSTERIOR<PROXY>				used for HMC		=pm_aux				(see below)
	// 	--"--, update_type 0								used for HMC		=pm_aux				NULL
	// 	--"--, update_type 1								=proxy_int			copy				copy (copies deleted in dtor)

	LeapFrog LF;		// object for leapfrog integration (contains mass matrix)
	int LF_steps;		// number of leapfrog steps to be done in each iteration
	double max_step_eps;			// maximum allowed value for 'step_eps'

	std::vector<int> ind_grad_add_pts;	// indices of points where gradients are taken for proxy training, the indices should range in [0, Nadd_fval_pts)
	std::vector<int> ind_grad_comps;	// indices of gradient components taken for proxy training, the indices should range in [0, fulldim)
	std::vector<int> proxy_dump_inds;			// for the iteration numbers provided here (0-based), proxy models will be saved to dump file

	bool calc_grads() const {return ind_grad_add_pts.size() > 0 && ind_grad_comps.size() > 0;};		// defines whether gradients for proxy should be calculated
	void resetVecs(bool is_initial);																// clear and reserve vectors Xnew, ynew, Datanew, Gradnew (call on ALL RANKS)
	void updateVecs(std::vector<double> x, double y, std::vector<double> d, std::vector<double> grad);		// add elements to the vectors (call on ALL RANKS); inputs are referenced @ RNK-0
																											// input 'x' is actdim params, but 'Xnew' gets the full-dimension version of 'x'
																											// 'grad' should be a fulldim vector (or empty vector, if not needed)
	virtual void preprocess(const Mat &x);
	virtual void proposal(const Mat &x);
	virtual void process_new_point(const Mat &x);
	virtual void save_output(const Mat &x);
	virtual void make_updates(Mat &x);
	virtual void trainProxy(bool is_initial);				// trains proxy with Xnew, ynew, Datanew; "is_initial" should be 'true' when used in preprocess()
public:
	HMC1(PhysModel *p, PhysModel *aux, Rand g, LeapFrog lf, EpsUpdate1 eu, int bi, int lf_steps, double maxstep, int upd, int Nadd, std::vector<int> ind_gpts, std::vector<int> ind_gcomp, std::vector<int> dump_inds, int upd_type);
	virtual ~HMC1();
	//PhysModel *pm_grads;	// DEBUG - model used for gradients calculation DEBUG
};
//------------------------------------------------------------------------------------------
// HMCrej sampler: same as HMC1, but when the point hits the boundary, it becomes rejected. There is no velocity bouncing
//------------------------------------------------------------------------------------------
class HMCrej : public HMC1
{
protected:
	// Formally, the trajectory may bounce, but the trajectory with bounces > 0 is then rejected.
	// However, 'dE_proxy' is updated based on the trajectory with bounces, as before.
	virtual void process_new_point(const Mat &x);

public:
	using HMC1::HMC1;
};
//------------------------------------------------------------------------------------------
// SOL-HMC with Gamma2=I, prior covariance matrix stored at LF.M is taken from MAT (and zero mean is used)
// Don't use the POSTERIOR model here, this sampler accounts for prior covariance automatically.
// 'eps' and proxy are updated as in HMC.
// p0, p1 now have the meaning of velocities.
// TO BE USED on all ranks in MPI_COMM_WORLD
//------------------------------------------------------------------------------------------
class SOL_HMC : public HMC1
{
private:
	// the variables are referenced on ranks:
	// of0, of1, ModelledData0, Kin0, Kin1, EUpd -- RNK-0
	// LF.M, LF.L, LF.eps -- LF-comm-RANKS-0
	// x0, x1, p0, p1 -- either all ranks (WORLD), or RNK-0, with Bcasing where appropriate
	// other variables -- all ranks

protected:
	double ii;				// parameter for OU integration step
	double KinQ0, KinQ1;	// 0.5*(x, C^(-1)*x)

	virtual void preprocess(const Mat &x);
	virtual void proposal(const Mat &x);
	virtual void process_new_point(const Mat &x);
	virtual void accept_new_point();
	virtual void reject_new_point();
public:
	SOL_HMC(PhysModel *p, PhysModel *aux, Rand g, LeapFrog lf, EpsUpdate1 eu, int bi, int lf_steps, double maxstep, int upd, int Nadd, std::vector<int> ind_gpts, std::vector<int> ind_gcomp, std::vector<int> dump_inds, int upd_type, double i0);
};
//------------------------------------------------------------------------------------------
// class for RHMC sampling
// during burn-in, RHMC1::Run() updates proxy model and leapfrog step 'eps' every 'upd_freq' iterations
// TO BE USED on all ranks in MPI_COMM_WORLD
//------------------------------------------------------------------------------------------
class RHMC1 : public HMC1
{
	// the variables are referenced on ranks:
	// of0, of1, ModelledData0, Kin0, Kin1, EUpd -- RNK-0
	// x0, x1, p0, p1 -- either all ranks (WORLD), or RNK-0, with Bcasing where appropriate
	// other variables -- all ranks

protected:
	// LeapFrog LF; - NOT USED
	// Kin0, Kin1 now store log(det) + p'*G^(-1)*p
	LeapFrogGeneralized Lfg;
	double step_eps;							// step size for generalized leap frog

	double eig_min;								// min eigenvalue of 'MMinit' (mass matrix at start of gen. LF) -- for reporting, calculated in proposal()
	double cond2;								// condition number (in 2-norm) of 'MMinit' (mass matrix at start of gen. LF) -- for reporting, calculated in proposal()

	virtual void preprocess(const Mat &x);
	virtual void proposal(const Mat &x);
	virtual void process_new_point(const Mat &x);
	virtual void save_output(const Mat &x);
	virtual void make_updates(Mat &x);
	virtual void trainProxy(bool is_initial);	// calls HMC1::trainProxy, and resets caches in Lfg
public:
	RHMC1(PhysModel *p, PhysModel *aux, Rand g, NonlinearSystemSolver *sol, EpsUpdate1 eu, int bi, int lf_steps, double MM_shift, double eps, double maxeps, int upd, int Nadd, int LFG_maxref, std::vector<int> ind_gpts, std::vector<int> ind_gcomp, std::vector<int> dump_inds, int upd_type);
												// 'sol' is the solver used in generalized leapfrog
};
//------------------------------------------------------------------------------------------
// class for Manifold MALA
// proposal = N(mu, eps^2 * FI^(-1));
//------------------------------------------------------------------------------------------
class MMALA : public HMC1
{
protected:						// these two are reset in preprocess() and after each 'eps' update
	int MMALA_steps;			// steps taken by inner MMALA proposals
	int MMALA_accepts;			// accepted --"--

	int type;					// 0 - MMALA1 (Calderhead-Girolami), 1 - simplified MMALA, 2 - MMALA2 (MFCW)
	double eps;					// for proposal step
	int Nsteps;					// number of inner steps
	PM_FullHamiltonian Ham0;	// handles geometric properties in the current point
	PM_FullHamiltonian Ham1;	// handles geometric properties in the proposed point

	const Mat p0, p1;			// shadowed (not for use)
	const int lf_bounces;		// shadowed (not for use)
	const int LF_steps;			// shadowed (not for use)
	const double U0, dist_rat;	// shadowed (not for use)
	// LeapFrog LF - will be a dummy object
	// of0_aux, of1_aux - same as before, and used in the outer M-H test
	// Kin0, Kin1 - q(x|xnew), q(xnew|x), used by internal MMALA steps
	// dE = -of1/2 + of0/2 - (-of1_aux/2 + of0_aux/2)

	double eig_min;				// min eigenvalue of 'MMinit' (mass matrix at start of each proposal) -- for reporting, calculated in proposal()
	double cond2;				// condition number (in 2-norm) of 'MMinit' -- for reporting, calculated in proposal()

	virtual void preprocess(const Mat &x);
	virtual void proposal(const Mat &x);			// NOTE: LF_steps from HMC settings defines the inner Nsteps
	virtual void process_new_point(const Mat &x);	// take x = x0 !
	virtual void save_output(const Mat &x);
	virtual void accept_new_point();
	virtual void make_updates(Mat &x);
	virtual void trainProxy(bool is_initial);		// calls HMC1::trainProxy, and resets caches in Ham0, Ham1

public:
	MMALA(PhysModel *p, PhysModel *aux, Rand g, EpsUpdate1 eu, int bi, int ufreq, int Nadd, double MM_shift, double e, double maxeps, int steps, int Type, std::vector<int> ind_gpts, std::vector<int> ind_gcomp, std::vector<int> dump_inds, int upd_type);
};
//------------------------------------------------------------------------------------------
// class for Irreversible MALA
// Can use unity mass matrix, or FI
// internally, z = {theta, p} - i.e. the space is doubled
//
// D = |G^(-1)  0|	 Q = alpha*| 0  I|
//	   |0	    G|			   |-I  0|
//
// where G = I or G = FI
// This is currently not a working version
//------------------------------------------------------------------------------------------
class I_MALA : public MMALA
{								// In the comments below, '*' shows that the data member has the same meaning as in MMALA
private:						// * these two are reset in preprocess() and after each 'eps' update
	//int MMALA_steps;			// * steps taken by [inner] proposals
	//int MMALA_accepts;		// * accepted --"--

	int zp;						// direction variable, {-1, +1}		  TODO check repeatability from seed
	Mat p0, p1;					// auxiliary momentum variable
	double fp0, fp1;			// used in "M-H"
	const double alpha;			// scaling factor for matrix Q

	Mat proposal_I(const Mat &xp);		// proposal for G=I, starting from 'xp'; forward/backward proposal - depending on 'zp'; the returned vector is in the doubled space
	double lnP_I(const PM_FullHamiltonian &ham, const Mat &xp, const Mat &xpnew, double eps, int sign);		// log of P for G=I; forward/backward P - depending on 'sign'; 'ham' should be Ham0 for xp={x0,p0}, Ham1 for xp={x1,p1}
	Mat sample_p_I(size_t s);	// generate 'p', for G=I
	double calc_fp_I(const Mat &x, const Mat &p);	// calculate fp0 or fp1, for G=I; this also includes 0.5*lndet(G(theta))

protected:
	//int type;					// NEW MEANING: 0 - MM=unity, 1 - MM=FI TODO
	//double eps;				// * for proposal step
	//const int Nsteps;			// * number of inner steps; TODO this should be 1 for now
	//PM_FullHamiltonian Ham0;	// * handles geometric properties in the current point
	//PM_FullHamiltonian Ham1;	// * handles geometric properties in the proposed point

	const double Kin0, Kin1;	// shadowed (not for use)
	// of0_aux, of1_aux - used for calculating the "inner acceptance rate"

	//double eig_min;			// * min eigenvalue of 'MMinit' (mass matrix at start of each proposal) -- for reporting, calculated in proposal() TODO
	//double cond2;				// * condition number (in 2-norm) of 'MMinit' -- for reporting, calculated in proposal()	TODO

	virtual void preprocess(const Mat &x);
	virtual void proposal(const Mat &x);
	virtual void process_new_point(const Mat &x);	// take x = x0 !
	//virtual void save_output(const Mat &x);	* as in MMALA
	virtual void accept_new_point();
	virtual void reject_new_point();
	//virtual void make_updates(Mat &x);		* as in MMALA
	//virtual void trainProxy(bool is_initial); * as in MMALA

public:
	I_MALA(PhysModel *p, PhysModel *aux, Rand g, EpsUpdate1 eu, int bi, int ufreq, int Nadd, double MM_shift, double e, double maxeps, int steps, int Type, std::vector<int> ind_gpts, std::vector<int> ind_gcomp, std::vector<int> dump_inds, int upd_type, double Alpha);
};
//------------------------------------------------------------------------------------------

}	// namespace HMMPI

#endif /* MONTECARLO_H_ */
