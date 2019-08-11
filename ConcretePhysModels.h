/*
 * ConcretePhysModels.h
 *
 *  Created on: Mar 26, 2013
 *      Author: ilya
 */

#ifndef CONCRETEPHYSMODELS_H_
#define CONCRETEPHYSMODELS_H_

#define ECLFORMATTED		// for PhysModelHM: read eclipse summary in formatted form (FUNSMRY) | otherwise - read in binary form (UNSMRY)
#define DIAGCOV_1X1			// for PMEclipse: if all covariance blocks are diagonal, they are further split into elementary 1x1 blocks - for better data points balancing on the processors (MPI)

//#define PUNQGRADS			// very adhoc code to read gradients from a specific Punq-S3 model
//#define PUNQADJ 			// adhoc code to read adjoint gradients for PunqS3 (formatted multiple output), 45 params; requires MULTOUT and "touch {1:%s}.FUNSMRY"

#include "Vectors.h"
#include "Parsing.h"
#include "PhysModels.h"
#include "EclSMRY.h"
//#include "MonteCarlo.h"
#include <iostream>
#include <fstream>
#include <string>

class Parser_1;
class Grid2D;
class VectCorrList;
class RegListSpat;

//---------------------------------------------------------------------------
// Reservoir Simulation model
// 'comm' is used in ObjFunc() to define size of the group each process belongs to
// for sizes > 1 parallel simulation is launched (if possible)
// results of ObjFunc() [objective function value and modelled_data] are Bcasted over 'comm'
// Verbose logs and messages are produced for RNK = 0 (in the legacy version it was for write_log = true)
// USER: define 'comm' with large groups to call PARALLEL SIMULATIONS
//---------------------------------------------------------------------------
class PhysModelHM : public PhysModel, public HMMPI::CorrelCreator, public HMMPI::StdCreator, public HMMPI::DataCreator
{
private:
	bool s_echo;				// variables for saving "state"
	int s_error, s_warning;

protected:
	Parser_1 *K;
	std::string CWD;

	std::vector<double> grid;            // çíà÷åíèÿ â ãðèäå, ïîñòðîåííûå ïî params

	static std::string log_file;
	HMMPI::Vector2<int> index_arr;
	int pet_seis_len;
	mutable size_t modelled_data_size = 0;

	VectCorrList *VCL;		// shared between copies of PhysModelHM
	RegListSpat *RLS;

#ifdef PUNQADJ
	std::vector<double> gradient;		// filled inside ObjFunc along with DataSens
	bool adjoint_run;					// if true, DATA file will be modified (ad hoc) for adjoint run; normally should be set to false
#endif

	virtual void RunSimulation(int i, int par_size);		// "i" = model ID, "par_size" = number of parallel processes for simulator run
	void GetParamsKrig() const;								// params_all + limits.norm -> params
	virtual void WriteData(int i);
	virtual void WriteModel(int i);
	virtual void WriteSWOF(int i);
	virtual void WriteSGOF(int i);
	virtual void WriteINC(int i);
	virtual bool check_limits_swgof() const;
	bool check_limits_krig() const;
	void add_limits_def(std::vector<std::vector<double>> &C, std::vector<double> &b);			// ~ Cx <= b
	void add_limits_swgof(std::vector<std::vector<double>> &C, std::vector<double> &b);			// ~ Cx <= b
	void add_limits_krig(std::vector<std::vector<double>> &C, std::vector<double> &b);			// ~ Cx <= b
	//void FillParamsAll(const std::vector<double> &params);	// active 'params' -> params_all	before 01.09.2016

	void write_params(std::ofstream &SW);		// Object = StreamWriter
	void write_smry(std::ofstream &SW, const HMMPI::Vector2<double> &smry_mod, const HMMPI::Vector2<double> &smry_hist, const std::vector<double> &of1_full, bool text_sigma);
	void write_smry(std::ofstream &SW, const HMMPI::Vector2<double> &smry_mod);							// only model data, no history
	void write_smry_hist(std::ofstream &SW, const HMMPI::Vector2<double> &smry_hist, bool text_sigma);	// only history (+ sigmas, corr. radii)
	static std::vector<Grid2D> SubtractBase(const std::vector<Grid2D> &grids);	// Object = Grid2D, [A, B, C,...] -> [B-A, C-A,...]
	//void write_limits_log(std::vector<double> vals, std::string fname);		// replaced by KW_limits::Write_params_log from 15.09.2016

	HMMPI::Vector2<std::vector<double>> calc_derivatives_dRdP(std::string mod_name) const;	// [Nsteps x Nvecs] x (vec of len 45) - derivatives of well data (R=WBHP,WOPR,WLPR,WGPR) w.r.t. porosity regions (P)
																							// Where vec is not one of R's above, zeros are set.
	void calc_derivatives_dRdP2(HMMPI::Vector2<std::vector<double>> &drdp, const HMMPI::Vector2<double> &smry) const;
																						// updates "drdp": [Nsteps x Nvecs] x (vec of len 45) - derivatives of well data (R=WWCT,WGOR) w.r.t. porosity regions (P)
																						// based on output from calc_derivatives_dRdP "drdp", and the usual modelling "smry"
public:
	mutable std::vector<double> params;		// krig
	mutable std::vector<double> params_all;	// = pi/ni, ò.å. íîðìèðîâàííûå çíà÷åíèÿ
	double f1, f2, f3, f4, f5;	// wells, SWAT, SGAS, k_apr, R2
	int sign;					// åñëè -1, òî o.f. = -o.f.
	static std::string uncert_dir;
	bool ignore_small_errors;		// if 'true', exceptions != EObjFunc are intercepted in ObjFunc, and the main loop (simulation etc) is restarted; EObjFunc always leads to immediate termination
	std::vector<Grid2D> pet_seis;	// perturbed seismic, size = Attr.size - 1

	//std::vector<double> smry_long; --> became "ModelledData" after 24.05.2016		// all well data, with "inactive data" skipped

	static HMMPI::Vector2<double> COEFFS;			// äëÿ êðèãèíãà
	static int K_type;
	static HMMPI::Vector2<double> pts;

	PhysModelHM(Parser_1 *k, KW_item *kw, std::string cwd, MPI_Comm c);		// all data are taken from keywords of "k"; "kw" is used only to handle prerequisites
	PhysModelHM(const PhysModelHM &PM);				// copy constructor, increments RLS_count
	virtual int ParamsDim() const noexcept;
	virtual void SavePMState();		// saves echo, errors & warnings counts before ObjFuncMPI
	virtual void RestorePMState();	// restores echo, errors & warnings counts after running ObjFuncMPI
	virtual ~PhysModelHM();
	std::string IndexMsg() const;	// ñîîáùåíèå ïðî èíäåêñû â index_arr
	virtual bool CheckLimits(const std::vector<double> &params) const;
	virtual double ObjFunc(const std::vector<double> &params);			// calculates objective function by running eclipse model with index = RNK; fills ModelledData if w1 != 0
																		// simulation is only done on comm-RANKS-0; o.f. and modelled_data are Bcasted to other ranks as well
	virtual std::vector<double> ObjFuncGrad(const std::vector<double> &params);		// works only for #define PUNQADJ
	virtual size_t ModelledDataSize() const;
	virtual std::string ObjFuncMsg() const;
	void Constraints(HMMPI::Vector2<double> &matrC, std::vector<double> &vectb);
	void PerturbWell(double w1);
	void PerturbSeis(double w5);
	virtual void PerturbData();		// data perturbation for RML (and sync between ranks in MPI_COMM_WORLD);
									// #ifdef WRITE_PET_DATA, perturbed seismic for each step X is written to "pet_seis_X.txt"
									// perturbed well data is always written to "TextSMRY_RML.txt"
									// MPI is used for sync, so call this function on all ranks of MPI_COMM_WORLD
	int PetSeisLen(){return pet_seis_len;};

	virtual std::vector<HMMPI::Mat> CorrBlocks() const;		// N x N blocks, or N x 1 diagonals
	virtual std::vector<double> Std() const;		// these sigmas (std's) are fully taken from TEXTSMRY, not from ECLVECTORS!
	virtual std::vector<double> Data() const;		// data are taken only where sigma != 0, i.e. same vector length as ModelledData
};
//---------------------------------------------------------------------------
// Reservoir Simulation model with a more clear code and interface
// 'comm' is used in ObjFunc() to define size of the group each process belongs to
// for sizes > 1 parallel simulation is launched (if possible)
// Messages are produced for comm-ranks-0, logging to file is done for RNK-0 && ignore_small_errors==false
// USER: define 'comm' with large groups to call PARALLEL SIMULATIONS
// at the same time the simulator command line should contain "-n $SIZE" to run simulation with "SIZE" parallel threads
//---------------------------------------------------------------------------
// TODO work with PATHs should be improved in filename and template substitution
// tNav allows running models with paths, e.g. $tNav path/model
// but it doesn't treat well include file names (inside data file) with absolute path (e.g. INCLUDE c:/path/model, although cygwin+HistMatMPI accept such paths).
// So, one may work with relative paths, but in this case they should be treated in a better way during file name substitution
// E.g. with p1/p2/mod.data and p1/p2/poro.inc, inside "mod.data" use either INCLUDE poro.inc, or INCLUDE ../../p1/p2/poro.inc
// Also, think over if some trick is necessary in orig_files.
//---------------------------------------------------------------------------
class PMEclipse : public PhysModel, public HMMPI::CorrelCreator, public HMMPI::StdCreator, public HMMPI::DataCreator
{
protected:
	Parser_1 *K;
	std::string CWD;

	const std::string log_file;		// ObjFuncLog.txt is only written by RNK-0
	mutable size_t modelled_data_size = 0;
	std::string obj_func_msg;		// filled in ObjFunc()
	mutable bool cov_is_diag;		// true, if covariance matrix is fully diagonal; filled by CorrBlocks() or ObjFunc()

	VectCorrList *VCL;				// shared between copies of PMEclipse

	void write_smry(std::ofstream &sw, const HMMPI::Vector2<double> &smry_mod, const HMMPI::Vector2<double> &smry_hist, const std::vector<double> &of_vec, bool text_sigma, bool only_hist = false);	// output summary data to ASCII file
									// if "only_hist" == true, then "smry_mod", "of_vec" are not written (and can be empty)
	void perturb_well();
public:
	bool ignore_small_errors;		// if 'true', exceptions != EObjFunc are intercepted in ObjFunc, and the main loop (simulation etc) is restarted; EObjFunc always leads to immediate termination
									// also, if 'true', no log file (ObjFuncLog.txt) is written
	HMMPI::SimSMRY *smry;			// model SMRY filled by the last call of ObjFunc()

	PMEclipse(Parser_1 *k, KW_item *kw, std::string cwd, MPI_Comm c);		// all data are taken from keywords of "k"; "kw" is used only to handle prerequisites
	PMEclipse(const PMEclipse &PM);											// copy constructor, increments VCL->ownerCount
	const PMEclipse &operator=(const PMEclipse &p) = delete;
	virtual int ParamsDim() const noexcept;
	virtual ~PMEclipse();
	virtual double ObjFunc(const std::vector<double> &params);				// calculates objective function by running the simulation model; fills modelled_data; simulation is only done on comm-RANKS-0
	virtual size_t ModelledDataSize() const;					// only works for TEXTSMRY case, otherwise produces error
	virtual std::string ObjFuncMsg() const {return obj_func_msg;};
	virtual void PerturbData();		// data perturbation for RML (and sync between ranks in MPI_COMM_WORLD);
									// perturbed well data is written to "TextSMRY_RML.txt"
									// call this function on all ranks of MPI_COMM_WORLD
	virtual std::vector<HMMPI::Mat> CorrBlocks() const;			// N x N blocks, or N x 1 diagonals; updates "cov_is_diag"
	virtual std::vector<double> Std() const;		// these sigmas (std's) are fully taken from TEXTSMRY, not from ECLVECTORS!
	virtual std::vector<double> Data() const;		// data are taken only where sigma != 0, i.e. same vector length as ModelledData
};
//---------------------------------------------------------------------------
// pConnect model:
// Parameterises (like PMEclipse) data file for pConnect, runs pConnect and reads its binary output: objective function, modelled data, historical data, sigmas [and gradient].
// If only objective function is required, pConnect data file should use RUNMATBAL, if gradient is also required, then use RUNMATBALGRAD
// ObjFunc(), ObjFuncGrad() may save computation by using caches (filled by both ObjFunc, ObjFuncGrad)
// _NOTE_ the FULL HistMatMPI parameters vector should correspond to the ACTIVE FREE pConnect parameters vector (to work with gradients in a consistent way)
//---------------------------------------------------------------------------
class PMpConnect : public PhysModel, public HMMPI::CorrelCreator, public HMMPI::StdCreator, public HMMPI::DataCreator
{
private:
	double of_cache;					// cached objective function
	std::vector<double> data_cache;		// cached modelled data
	std::vector<double> par_of_cache;	// for these parameter values 'of_cache' and 'data_cache' are correct

	std::vector<double> grad_cache;		// cached gradient of objective function
	std::vector<double> par_grad_cache;	// for these parameter values 'grad_cache' is  correct

	std::vector<double> hist_cache;		// cached historical data
	std::vector<double> sigma_cache;	// cached sigmas
	bool hist_sigmas_ok;				// if 'true', then 'hist_cache', 'sigma_cache' are valid

	void run_simulation(const std::vector<double> &params);			// runs pConnect to fill of_cache [grad_cache]
	static std::vector<double> fread_vector(FILE *file);
protected:
	Parser_1 *K;
	std::string CWD;
	std::string obj_func_msg;			// filled in ObjFunc()
	double scale;						// multiplier for objective function (and gradient), should equal the number of observed data, so that obj. func. is the plain sum of squares

public:
	PMpConnect(Parser_1 *k, KW_item *kw, std::string cwd, MPI_Comm c);			// all data are taken from keywords of "k"; "kw" is used only to handle prerequisites
	PMpConnect(const PMpConnect &PM);											// copy constructor
	const PMpConnect &operator=(const PMpConnect &p) = delete;
	virtual ~PMpConnect();
	virtual int ParamsDim() const noexcept;
	virtual size_t ModelledDataSize() const;
	virtual double ObjFunc(const std::vector<double> &params);					// calculates objective function [and gradient] by running the simulation model;
																				// modelled_data is also filled; simulation is only done on comm-RANKS-0
	virtual std::vector<double> ObjFuncGrad(const std::vector<double> &params);	// gradient of objective function; internally, run_simulation() is called
	virtual std::string ObjFuncMsg() const {return obj_func_msg;};

	virtual std::vector<HMMPI::Mat> CorrBlocks() const;		// 1 x 1 blocks
	virtual std::vector<double> Std() const;				// sigmas (only non-zero sigmas are used)
	virtual std::vector<double> Data() const;				// historical data
};
//---------------------------------------------------------------------------
// Conc model:
// Parameterises (like PMEclipse) data file for HMI_Fortran, runs HMI_Fortran and reads its ASCII output modelled data.
//---------------------------------------------------------------------------
class PMConc : public PhysModel, public HMMPI::CorrelCreator, public HMMPI::StdCreator, public HMMPI::DataCreator
{
private:
	void run_simulation(const std::vector<double> &params, std::vector<double> &out_t, std::vector<double> &out_conc);			// runs simulation, filling the output out_t, out_conc (which are sync)

protected:
	Parser_1 *K;
	std::string CWD;
	std::string obj_func_msg;				// filled in ObjFunc()

	std::vector<double> tt;					// historical data; 'tt' should exactly correspond to time in "out_t"
	std::vector<double> c_hist;
	std::vector<double> sigma;				// zero sigmas are ignored when calculating the objective function, and in CorrBlocks(), Std(), Data(), modelled_data
	std::vector<int> nonzero_sigma_ind;		// indices of non-zero sigmas, [0, ModelledDataSize) -> [0, sigma.size)

public:
	PMConc(Parser_1 *k, KW_item *kw, std::string cwd, MPI_Comm c);			// all data are taken from keywords of "k"; "kw" is used only to handle prerequisites
	const PMConc &operator=(const PMConc &p) = delete;
	virtual int ParamsDim() const noexcept;
	virtual size_t ModelledDataSize() const {return nonzero_sigma_ind.size();};
	virtual double ObjFunc(const std::vector<double> &params);				// calculates objective function by running the simulation model;
																			// modelled_data is also filled; simulation is only done on comm-RANKS-0
	virtual std::string ObjFuncMsg() const {return obj_func_msg;};

	virtual std::vector<HMMPI::Mat> CorrBlocks() const;		// 1 x 1 blocks
	virtual std::vector<double> Std() const;				// sigmas (only non-zero sigmas are used)
	virtual std::vector<double> Data() const;				// historical data
};
//---------------------------------------------------------------------------
// Multidimensional Rosenbrock function
// Currently works only for LIMITS (not PARAMETERS) to avoid parameters casting to external representation
// 'modelled_data' is 2*(dim-1) vector
class PM_Rosenbrock : public PhysModel, public HMMPI::CorrelCreator, public HMMPI::StdCreator, public HMMPI::DataCreator
{
protected:
	int dim;										// full dimension (N + A)

public:
	PM_Rosenbrock(int d) : dim(d) {name = "PM_Rosenbrock";};		// model with given dimension
	PM_Rosenbrock(Parser_1 *K, KW_item *kw, MPI_Comm c);			// LIMITS (and dim) are taken from "K"; "kw" is used only to handle prerequisites
	virtual ~PM_Rosenbrock();
	virtual double ObjFunc(const std::vector<double> &params);
	virtual std::vector<double> ObjFuncGrad(const std::vector<double> &params);
	virtual HMMPI::Mat ObjFuncHess(const std::vector<double> &params);
	virtual HMMPI::Mat ObjFuncFisher(const std::vector<double> &params);
	virtual HMMPI::Mat ObjFuncFisher_dxi(const std::vector<double> &params, const int i, int r = 0);
	virtual int ParamsDim() const noexcept {return dim;};
	virtual size_t ModelledDataSize() const {return 2*dim - 2;};
	virtual std::vector<HMMPI::Mat> CorrBlocks() const;				// a single block with unity diagonal matrix stored as {N x 1} array, where N = 2*(dim-1)
	virtual std::vector<double> Std() const;
	virtual std::vector<double> Data() const;
};
//---------------------------------------------------------------------------
// Model with linear forward operator (Gx-d0)^t * 1/C * (Gx-d0)
// Covariance matrix can be either diagonal or dense
// Currently works only for LIMITS (not PARAMETERS) to avoid parameters casting to external representation
class PM_Linear : public PhysModel
{
private:
	std::vector<double> cov_diag();	// get diagonal of covariance matrix - DiagCov or FullCov

protected:
	HMMPI::RandNormal *RndN;		// this random number generator may be left = 0 (unless RML is required)

	HMMPI::Mat G;					// linear operator
	HMMPI::Mat d0;					// observed data (possibly, perturbed for RML)
	HMMPI::Mat d0_orig;				// observed data (original, not perturbed)
	std::vector<double> DiagCov;	// diagonal covariance matrix
	HMMPI::Mat FullCov;				// dense covariance matrix - used when DiagCov is empty

	bool holding_chol = false;			// 'true' = Cholesky decomposition was done, is stored, and can be accessed
	HMMPI::Mat chol_FullCov;			// Cholesky decomposition

public:
	PM_Linear(HMMPI::Mat g, std::vector<double> d, std::vector<double> c, HMMPI::Mat fc, HMMPI::RandNormal *rn = 0) :	// in this constructor, provide empty vector 'c' to use dense covariance matrix 'fc'
		RndN(rn), G(std::move(g)), d0_orig(std::move(d)), DiagCov(std::move(c)), FullCov(std::move(fc)), holding_chol(false){d0 = d0_orig; name = "PM_Linear";};
	PM_Linear(Parser_1 *K, KW_item *kw, MPI_Comm c);		// easy constructor; all data are taken from keywords of "K"; "kw" is used only to handle prerequisites
	virtual ~PM_Linear();
	virtual double ObjFunc(const std::vector<double> &params);
	virtual std::vector<double> ObjFuncGrad(const std::vector<double> &params);
	virtual HMMPI::Mat ObjFuncHess(const std::vector<double> &params);
	virtual HMMPI::Mat ObjFuncFisher(const std::vector<double> &params);
	virtual HMMPI::Mat ObjFuncFisher_dxi(const std::vector<double> &params, const int i, int r = 0);
	virtual int ParamsDim() const noexcept {return G.JCount();};
	virtual size_t ModelledDataSize() const;
	virtual void PerturbData();		// perturb data for RML using 'DiagCov' or 'FullCov'; new data are stored in 'd0'; #ifdef WRITE_PET_DATA, perturbed data are written to file "pet_data_LIN.txt" on RNK-0
									// perturbed data is always written to "MatVecVec_RML.txt" on RNK-0
									// no MPI is used here, so make sure srand() seed is sync between ranks
};
//---------------------------------------------------------------------------
// Model with arbitrary forward operator F(x), obj. func. = (F(x)-d0)^t * 1/C * (F(x)-d0)
// C - diagonal covariance matrix (defined by its diagonal-vector)
// Currently works only for LIMITS (not PARAMETERS) to avoid parameters casting to external representation
class PM_Func : public PhysModel
{
protected:
	int Npar;
	std::vector<double> d0;				// data
	std::vector<double> C;				// inverse of covariance diagonal

	virtual std::vector<double> F(const std::vector<double> &par) const = 0;	// the forward operator, mapping vector to vector
	virtual HMMPI::Mat dF(const std::vector<double> &par) const = 0;			// Jacobian of F (d0_dim * Npar)
	virtual HMMPI::Mat dJk(const std::vector<double> &par, int k) const = 0;	// (d0_dim * Npar) matrix of derivatives of k-th column of Jacobian, dJk_ij = d2F_i / dx_k*dx_j

public:
	PM_Func(int param_dim, const std::vector<double> &data, const std::vector<double> &c);
	PM_Func(Parser_1 *K, KW_item *kw, MPI_Comm c) : PhysModel(K, kw, c), Npar(0){};		// auxiliary CTOR
	virtual ~PM_Func();
	virtual double ObjFunc(const std::vector<double> &params);
	virtual std::vector<double> ObjFuncGrad(const std::vector<double> &params);
	virtual HMMPI::Mat ObjFuncHess(const std::vector<double> &params);

	virtual int ParamsDim() const noexcept {return Npar;};
	virtual size_t ModelledDataSize() const {return d0.size();};
};
//---------------------------------------------------------------------------
// LINEAR model from MATVECVEC (for testing)
class PM_Func_lin : public PM_Func
{
protected:
	HMMPI::Mat G;			// the operator matrix

	virtual std::vector<double> F(const std::vector<double> &par) const;	// forward operator
	virtual HMMPI::Mat dF(const std::vector<double> &par) const;			// Jacobian
	virtual HMMPI::Mat dJk(const std::vector<double> &par, int k) const;	// derivatives of k-th column of Jacobian
public:
	PM_Func_lin(Parser_1 *K, KW_item *kw, MPI_Comm c);		// easy constructor; all data are taken from keywords of "K"; "kw" is used only to handle prerequisites
};
//---------------------------------------------------------------------------
// POWER model from MATVEC
// F(x)_i = ln(a*(Si - S0)^b) = ln(a) + b*ln(Si - S0), where {a, b, S0} = x, Si = MAT_ij, column j is fixed, d0 = ln(VEC), all Ci = 1
// Npar = 3, and parameter min/max are defined in CTOR (however init values are taken from LIMITS)
class PM_Func_pow : public PM_Func
{
private:
	std::vector<double> min;		// 3-dim vectors to be used as bounds during optimization; filled in CTOR
	std::vector<double> max;

protected:
	const double small;				// used in constraints definition
	const double big;

	std::vector<double> Si;			// j-th column of MAT

	virtual std::vector<double> F(const std::vector<double> &par) const;	// forward operator
	virtual HMMPI::Mat dF(const std::vector<double> &par) const;			// Jacobian
	virtual HMMPI::Mat dJk(const std::vector<double> &par, int k) const;	// derivatives of k-th column of Jacobian
public:

	std::vector<double> get_min() const {return min;};
	std::vector<double> get_max() const {return max;};
	PM_Func_pow(Parser_1 *K, KW_item *kw, MPI_Comm c, int j);	// easy constructor; all data are taken from keywords of "K"; "kw" is used only to handle prerequisites
};
//---------------------------------------------------------------------------

#endif /* CONCRETEPHYSMODELS_H_ */
