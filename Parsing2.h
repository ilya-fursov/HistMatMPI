/*
 * Parsing2.h
 *
 *  Created on: Mar 20, 2013
 *      Author: ilya
 */

#ifndef PARSING2_H_
#define PARSING2_H_

#include <tuple>
#include "Parsing.h"
#include "MathUtils.h"
#include "MonteCarlo.h"
#include "EclSMRY.h"
#include "CornerPointGrid.h"

// TODO uncomment for default behaviour:
//#define TEMPLATES_KEEP_NO_ASCII 	// TEMPLATES->keep is not read from ASCII file, but is set by function call set_keep();
									// _ALSO_ no params substitution happens in file names
//------------------------------------------------------------------------------------------
// NEXT GO THE USER DERIVED CLASSES FOR KW_run, KW_params, KW_fname
//------------------------------------------------------------------------------------------
// TODO descendants of KW_run
//------------------------------------------------------------------------------------------
class KW_echo : public KW_run
{
public:
	KW_echo();
	virtual void Run();
};
//------------------------------------------------------------------------------------------
class KW_noecho : public KW_run
{
public:
	KW_noecho();
	virtual void Run();
};
//------------------------------------------------------------------------------------------
class KW_runForward : public KW_run
{
public:
	KW_runForward();
    virtual void Run();
};
//------------------------------------------------------------------------------------------
class KW_runSingle : public KW_run		// single run of PMEclipse/PM_SimProxy; PMEclipse summary is added to ECLSMRY file
{
public:
	KW_runSingle();
    virtual void Run();
};
//------------------------------------------------------------------------------------------
class KW_runMultiple : public KW_run	// multiple run of PMEclipse, all resulting summaries are added to ECLSMRY file; sequence of parameters is taken according to MULTIPLE_SEQ
{
public:
	KW_runMultiple();
    virtual void Run();
};
//------------------------------------------------------------------------------------------
class KW_runOptProxy : public KW_run	// main optimization loop includes: {optimization of PM_SimProxy, run of PMEclipse, adding summary to ECLSMRY file}
{
public:
	KW_runOptProxy();
    virtual void Run();
};
//------------------------------------------------------------------------------------------
class KW_runPopModel : public KW_run	// remove the last model from ECLSMRY and save it to file
{
public:
	KW_runPopModel();
    virtual void Run();
};
//------------------------------------------------------------------------------------------
class KW_runViewSmry : public KW_run	// view (writing to the ASCII file) the selected data points (DATES x ECLVECTORS) from the ECLSMRY file
{
public:
	KW_runViewSmry();
    virtual void Run();
};
//------------------------------------------------------------------------------------------
class KW_runPlot : public KW_run
{
public:
	KW_runPlot();
    virtual void Run();
};
//------------------------------------------------------------------------------------------
class KW_runOpt : public KW_run
{
public:
	KW_runOpt();
    virtual void Run();
};
//------------------------------------------------------------------------------------------
class KW_runCritGrad : public KW_run	// find a point where gradient of PhysModel == 0
{
public:
	KW_runCritGrad();
    virtual void Run();
};
//------------------------------------------------------------------------------------------
class KW_runGrad : public KW_run
{
public:
	KW_runGrad();
    virtual void Run();
};
//------------------------------------------------------------------------------------------
class KW_runJac : public KW_run			// calculate Func and Jac for VM_gradient(PHYSMODEL), analytical and numerical gradients are found
{
public:
	KW_runJac();
    virtual void Run();
};
//------------------------------------------------------------------------------------------
class KW_runcalccovar : public KW_run
{
public:
	KW_runcalccovar();
    virtual void Run();
};
//------------------------------------------------------------------------------------------
class KW_runcalcwellcovar : public KW_run
{
public:
	KW_runcalcwellcovar();
    virtual void Run();
};
//------------------------------------------------------------------------------------------
class KW_runmpicheck : public KW_run
{
public:
	KW_runmpicheck();
    virtual void Run();
};
//------------------------------------------------------------------------------------------
class KW_runNNCfromgrid : public KW_run			// extract NNCs across the faults based on the mesh
{
public:
	KW_runNNCfromgrid();
    virtual void Run();
};
//------------------------------------------------------------------------------------------
class KW_runPinchMarkFromGrid : public KW_run	// marks the cells to control MULTZ for PINCHes
{
public:
	KW_runPinchMarkFromGrid();
    virtual void Run();
};
//------------------------------------------------------------------------------------------
class KW_runGridIJK_to_XYZ : public KW_run		// convert IJK to XYZ
{
public:
	KW_runGridIJK_to_XYZ();
    virtual void Run();
};
//------------------------------------------------------------------------------------------
class KW_runXYZ_to_GridIJK : public KW_run		// convert XYZ to IJK
{
public:
	KW_runXYZ_to_GridIJK();
    virtual void Run();
};
//------------------------------------------------------------------------------------------
class KW_runsoboltest : public KW_run
{
public:
	KW_runsoboltest();
	virtual void Run();
};
//------------------------------------------------------------------------------------------
class KW_runmatrixtest : public KW_run
{
public:
	KW_runmatrixtest();
	virtual void Run();
};
//------------------------------------------------------------------------------------------
class KW_runRosenbrock : public KW_run
{
public:
	KW_runRosenbrock();
	virtual void Run();
};
//------------------------------------------------------------------------------------------
class KW_runmatinv : public KW_run
{
public:
	KW_runmatinv();
	virtual void Run();
};
//------------------------------------------------------------------------------------------
class KW_rundebug : public KW_run
{
public:
	KW_rundebug();
	virtual void Run();
};
//------------------------------------------------------------------------------------------
class KW_runMCMC : public KW_run
{
public:
	KW_runMCMC();
	virtual void Run();
};
//------------------------------------------------------------------------------------------
// TODO descendants of KW_params
//------------------------------------------------------------------------------------------
class KW_verbosity : public KW_params
{
protected:
	virtual void UpdateParams() noexcept;	// updates global verbosity

public:
	int level;	// verbosity level: negative - low, positive - high

	KW_verbosity();
};
//------------------------------------------------------------------------------------------
class KW_gas : public KW_params
{
public:
	std::string on;    	// ON, OFF

	KW_gas();
};
//------------------------------------------------------------------------------------------
class KW_RML : public KW_params
{
public:
	std::string on;    	// ON, OFF
	int seed;		   	// 0 for seed = time(NULL)

	KW_RML();
};
//------------------------------------------------------------------------------------------
class KW_viewsmry_config : public KW_params
{
public:
	std::string out_file;   // file for writing
	std::string order; 		// DIRECT, SORT

	KW_viewsmry_config();
};
//------------------------------------------------------------------------------------------
class KW_multiple_seq : public KW_params	// defines the sequence for RUNMULTIPLE
{
protected:
	virtual void UpdateParams() noexcept;

public:
	const std::string logfile = "MultipleSequenceLog.txt";		// file where parameter values are saved

	int N;				// number of models in sequence
	double MaxHours;	// maximum time (hours)
	std::string type;   // SOBOL, RANDGAUSS
	int seed;		   	// 0 means take seed = time(NULL)
	double R;			// radius for RANDGAUSS

	KW_multiple_seq();
	virtual void FinalAction() noexcept;
	std::string msg() const;
};
//------------------------------------------------------------------------------------------
class KW_simcmd : public KW_multparams
{
public:
	std::vector<std::string> cmd;
	std::vector<std::string> cmd_work;		// after parameter substitution

	KW_simcmd();
	virtual void UpdateParams() noexcept;	// initial setup of "cmd_work"
	void RunCmd() const;					// runs all commands in "cmd_work" (on all ranks where it is called)
};
//------------------------------------------------------------------------------------------
class KW_shell : public KW_multparams
{
public:
	std::vector<std::string> cmd;

	KW_shell();
	virtual void FinalAction() noexcept;	// runs the shell commands on MPI_COMM_WORLD-RANK-0
};
//------------------------------------------------------------------------------------------
class KW_undef : public KW_params
{
public:
	double Uvect;
	double Uvectbhp;
	double Ugrid;

	KW_undef();
};
//------------------------------------------------------------------------------------------
class KW_variogram : public KW_params
{
public:
	double chi;
	double R;      // major radius
	double r;      // minor radius
	double sill;
	double nugget;
	std::string type;   	// EXP, SPHER, GAUSS
	std::string krig_type;	// SIM, ORD

	KW_variogram();
};
//------------------------------------------------------------------------------------------
class KW_variogram_Cs : public KW_variogram
{
public:
	KW_variogram_Cs();
};
//------------------------------------------------------------------------------------------
class KW_ofweights : public KW_params	// weights for objective function in KW_runForward (and others)
{
public:
	double w1;
	double w2;
	double w3;
	double w4;
	double w5;

	KW_ofweights();
};
//------------------------------------------------------------------------------------------
class KW_regressquadr : public KW_params
{
public:
	int P2;		// quadratic terms in regression: 0 - off, 1 - on
	int Sw2;
	int Sg2;
	int PSw;
	int PSg;
	int SwSg;

	KW_regressquadr();
	std::vector<int> IndActive(bool gas);	// v1, v2 = [0, 1, 2] <-> [P, Sw, Sg], returns array of v = v1 + 3*v2 - codes for selected variables
											// if gas == false -> ignore gas variables
};
//------------------------------------------------------------------------------------------
class KW_regressRs : public KW_params
{
public:
	int Rs;		// Rs terms: 0 - off, 1 - on
	int Rs2;
	int RsP;
	int RsSw;
	int RsSg;

	KW_regressRs();
	std::vector<int> IndActive(bool gas);	// v1 = [0, 1, 2, 3, 4] <-> [null, Rs, P, Sw, Sg], returns array of v1 - codes for selected variables
											// if gas == false -> ignore gas variables
											// e.g. [Rs, Rs2, RsSg] -> [0, 1, 4]
	virtual void UpdateParams() noexcept;
};
//------------------------------------------------------------------------------------------
class KW_Swco : public KW_params	// Swco is SWL
{
public:
	double Swco;

	KW_Swco();
};
//------------------------------------------------------------------------------------------
class KW_SWOFParams : public KW_params		// this class produces SWOF table
{
protected:
	std::vector<std::string> buffer;
	std::string prop_name;
	int div;

	void CheckMonoton();				// updates buffer
	void WriteToFile(std::string fn);	// writes buffer to file
public:
	std::string type;	// COREY, CHIERICI, LET
	double Swc;
	double Sor;
	double krw0;
	double p1;
	double p2;
	double p3;
	double p4;
	double p5;
	double p6;

	KW_SWOFParams();
	void WriteSWOF(std::string fn, std::vector<double> params);
	void SwcSor(std::vector<double> params, double &s_wc, double &s_or);		// values of S_wcr, S_or as written by WriteSWOF
	void CalcVal(double Sw, std::vector<double> params, double &krw, double &kro);
	int VarCount();		// counts "-1"
	std::vector<double> VarParams(std::vector<double> params_all, int i0, int i1);	// returns array of 9 parameters
	std::vector<int> SwcSorIndex(int i0);	// returns array of 2 indices, which = -1 if param is fixed; i0 is like in VarParams
											// where "-1" are replaced by values from params_all[i]*limits.norm[i], i0 <= i < i1
};
//------------------------------------------------------------------------------------------
class KW_SGOFParams : public KW_SWOFParams
{
public:
	KW_SGOFParams();
	void WriteSWOF(std::string fn, std::vector<double> params);
	void CalcVal(double Sg, std::vector<double> params, double &krg, double &kro);
};
//------------------------------------------------------------------------------------------
class KW_griddims : public KW_params
{
public:
	int Nx;					 // grid dimensions
	int Ny;
	int Nz;
	std::string krig_prop;
	std::string krig_file;
	std::string swof_file;
	std::string sgof_file;
	std::string wght;
	std::string satfmt;		// "%.8f", ...

	KW_griddims();
};
//------------------------------------------------------------------------------------------
class KW_griddimens : public KW_params
{
public:
	int Nx;					// grid dimensions
	int Ny;
	int Nz;
	double X0;				// coords origin
	double Y0;
	std::string grid_Y_axis;	// "POS" or "NEG"
	std::string actnum_name;
	double actnum_min;

	KW_griddimens();
};
//------------------------------------------------------------------------------------------
class KW_satsteps : public KW_parint
{
public:
	KW_satsteps();
};
//------------------------------------------------------------------------------------------
class KW_delta : public KW_pardouble
{
public:
	KW_delta();
};
//------------------------------------------------------------------------------------------
// multiple lines to describe the 'INCLUDES' for eclipse: [file_name  mod_name  par_count]
class KW_incfiles : public KW_multparams
{
public:
	std::vector<std::string> file;     // template_include_file_name
	std::vector<std::string> mod;      // include_file_suffix
	std::vector<int> pcount;		   // number of parameters in include file
	std::vector<std::string> Buffer;		// stores contents of each include_file

	KW_incfiles();
	virtual void UpdateParams() noexcept;	// some file reading
};
//------------------------------------------------------------------------------------------
// multiple lines specifying pairs of files for conversion, similar to KW_incfiles
class KW_templates : public KW_multparams
{
protected:
	mutable std::vector<std::string> work_file_subst;	// work file names after params substitution; filled by WriteFiles(), used by ClearFiles()
	int data_file_ind;						// index of DATA file within orig/work_file vectors, set in check_fnames(), used in WriteFiles(), ClearFilesEcl()

	void check_fnames() noexcept;			// checks orig and work file names: duplicates? *.DATA well defined? $RANK present?, and also sets 'data_file_ind'
	virtual void PrintParams() noexcept;
public:
	std::vector<std::string> orig_file;		// name of original file
	std::vector<std::string> mode;			// conversion mode: ">" to convert 'orig_file' into 'work_file' and replace names, "-" only to replace names
	std::vector<std::string> work_file;		// name of file which is used in simulation
	std::vector<std::string> keep;			// FIRST, NONE, ALL - defines whether 'work_file' is kept after simulation is complete
											// FIRST - keep only on WORLD-RANK-0, NONE - erase on all ranks, ALL - keep on all ranks

	std::vector<std::string> Buffer;		// stores contents of each 'orig_file' (if mode == ">")

	KW_templates();
	virtual void UpdateParams() noexcept;	// some file reading
	std::string WriteFiles(HMMPI::TagPrintfMap &par) const;	// replaces file names (orig->work), plugs in params, writes work files; returns parameters substitution message
															// sets MOD and PATH in "par", fills SIMCMD->cmd_work, fills "work_file_subst"
	void ClearFiles();						// clear work files according to 'keep'
	void ClearFilesEcl();					// clear files produced by Eclipse, according to 'keep'		-- TODO for tNav this still doesn't work
	std::string DataFileSubst() const;		// name of data file after $RANK substitution
	static std::vector<int> find_str_in_vec(const std::vector<std::string> &vec, const std::string &needle) noexcept;		// returns vector of all indices "i" such that "needle" is found in vec[i]
	static std::vector<int> find_end_str_in_vec(const std::vector<std::string> &vec, const std::string &needle) noexcept;	// returns vector of all indices "i" such that "needle" is found in the end of vec[i]
#ifdef TEMPLATES_KEEP_NO_ASCII
	void set_keep(std::string k);			// set all keep[i] = k
#endif
};
//------------------------------------------------------------------------------------------
// Multiple lines describing Eclipse vectors [WGname  vect  sigma  R  corr.func.]
class KW_eclvectors : public KW_multparams, public HMMPI::SigmaMessage
{
public:
	std::vector<HMMPI::SimSMRY::pair> vecs;	// vector of pairs, e.g. <FIELD, FWCT>

	std::vector<std::string> WGname;	// well/group name
	std::vector<std::string> vect;		// quantity, e.g. WBHP
	std::vector<double> sigma;			// std
	std::vector<double> R;		  		// correlation radius (in time)
	std::vector<std::string> func;		// correlation function

	std::vector<HMMPI::Func1D*> corr;	// correlation function

	KW_eclvectors();
	~KW_eclvectors();
	virtual void UpdateParams() noexcept;	// process "WGname", "vect", set default R's differently, fill "corr", "vecs"
	virtual std::string SigmaInfo(const std::string &wgname, const std::string &keyword) const;		// from HMMPI::SigmaMessage
};
//------------------------------------------------------------------------------------------
// defines constraints for derivatives: <= 0, >= 0, or no constraint
class KW_regressConstr : public KW_params
{
protected:
	std::vector<int> xi_ind;		// indices of active variables xi of the whole list of 14 variables
	std::vector<int> var_ind;		// array of 14 elements with indices in xi
	HMMPI::Vector2<int> deriv_ref;	// 4 x 14 aux. array for all partial derivatives of all variables.
							// e.g. deriv_ref[0, 3] = 100 means dP^2/dP index = 100%100 = 0 <-> P, plus multiplier 2 (since > 100)
							// deriv_ref[1, 6] = 0 means dPSw/dSw index = 0 <-> P, multiplier 1
							// deriv_ref[2, 2] = -1 means dSw/dSw = 1
							// deriv_ref[0, 1] = -99 means dSw/dP = 0
public:
	int dP;		// -1, 0, 1
	int dSw;
	int dSg;
	int dRs;

	KW_regressConstr();
	bool hasQuadr(int ind);		// ind = 0, 1, 2, 3 for P, Sw, Sg, Rs, determines if quadratic variables
								// are present (from REGRESSQUADR, REGRESSRS)
	void fill_ind();	// fills xi_ind, var_ind
	std::vector<double> getConstr(int ind, std::vector<double> xi, double a0);	// constraint for ind-th variable (0..3)
								// given values of active variables xi;
								// call after fill_ind()
	std::string getConstrStr(int ind);	// string version, for debug
	std::vector<double> getConstrFinal(int ind, std::vector<double> xi, double a0);	// same as getConstr, also includes sign from input (dP...dRs)
								// if input sign is zero, returns nullptr
	std::vector<double> getInitPoint(double A0sign, const std::vector<int> &A_V);
};
//------------------------------------------------------------------------------------------
class KW_plotparams : public KW_params
{
public:
	int Nx;
	int Ny;
	double delta;

	KW_plotparams();
};
//------------------------------------------------------------------------------------------
class KW_optimization : public KW_params
{
protected:
	std::vector<OptContext*> ctx;					// pointers to be freed are added here by MakeContext()
	std::vector<NonlinearSystemSolver*> nonlin;		// -"- MakeNonlinSolver()

public:
	std::string algorithm;		// CMAES, LM
	std::string fin_diff;		// OH1, OH2, OH4, OH8
	std::string nonlin_solver;	// FIXEDPOINT; (gsl) NEWTON, GNEWTON, HYBRIDPOWELL; (kinsol) KIN_NEWTON, KIN_NEWTON_LS, KIN_FP, KIN_PICARD; for nonlinear solver the other options are: 'maxit', 'epsG', ['epsX' - for KINSOL]
	int maxit;					// max iterations (both for LM optimization, and Newton non-linear solver)
	int maxJacIters;			// KINSOL only: max nonlinear iterations before Jacobian recalculation; "1" corresponds to exact Newton method, "0" would result in default value 10
	int maa;					// Anderson acceleration space dimension (0 - no acceleration)
	double epsG;				// stopping tolerances
	double epsF;				//
	double epsX;				//
	double R;					// sphere/cube radius; if it is > 0, constrained optimization is done
	std::string restr;			// CUBE, SPHERE

	KW_optimization();
	~KW_optimization();			// frees the pointers in "ctx"
	OptContext *MakeContext();						// *** FREEING the output pointer is done automatically ***
	NonlinearSystemSolver *MakeNonlinSolver();		// *** FREEING the output pointer is done automatically ***
};
//------------------------------------------------------------------------------------------
class KW_opt_config : public KW_params	// TODO in KW_params make new type available: long (in addition to int)
{
protected:
	virtual void UpdateParams() noexcept;

public:
	int MaxIter;			// stopping criteria for outer loop: maximum iterations
	double MaxHours;		// and maximum time (hours)
	std::string LMstart;	// CURR, SIMBEST - defines how the starting point for LM is taken; (INIT, ALL = {INIT, SIMBEST, CURR, LM_Nrand random points} -- obsolete)
	double r0;		// starting step for RESTRICTED STEP case (internal representation); RESTRICTED STEP is used whenever r0 > 0
	double rmin;	// minimum step for RESTRICTED STEP case
	double tau1;	// 0.25, for RESTRICTED STEP case
	double tau2;	// 0.75, for RESTRICTED STEP case
	double delta;	// small gap for spherical coordinates (RESTRICTED STEP case)
	std::string restr;		// restriction type: CUBE, SPHERE
	int LMmaxit;			// stopping criteria for each inner (LM) loop
	int LMmaxit_spher;		// max iterations for LM optimization on sphere (only for restr = SPHERE)
	double epsG;
	double epsF;
	double epsX;

	KW_opt_config();
};
//------------------------------------------------------------------------------------------
class KW_eclsmry : public KW_params
{
private:
	std::string copy_file_exists(const std::string &f0, int c); 	// if 'f0' exists (&& c > 0), then it is copied to 'f0~', and the function is called recursively on 'f0~'
																	// 'c' has the meaning of 'backup'; returns a message reporting the file names;
																	// to be called on all ranks
protected:
	HMMPI::SimProxyFile Data;

public:
	std::string fname;			// eclsmry file name; if 'fname' is empty -> empty SimProxyFile is created
	int backup;					// specifies how many previous versions are backed up
	double Xtol;				// if |xnew - xi| < Xtol, and dates & vecs lists have not changed, a new model will not be added

	KW_eclsmry();
	virtual void FinalAction() noexcept;	// reads the file
	std::string Save() __attribute__((warn_unused_result));			// save 'Data' to 'fname', making back-ups if necessary; to be called on all ranks; a message "ECLSMRY... saved to..." is returned
	const HMMPI::SimProxyFile &get_Data() const {return Data;};
	HMMPI::SimProxyFile &get_Data() {return Data;};
};
//------------------------------------------------------------------------------------------
class KW_wrcovar : public KW_params
{
public:
	int M;					 	// 1/2 of square side
	std::string cov_file;
	std::string count_file;

	KW_wrcovar();
};
//------------------------------------------------------------------------------------------
// 3 columns of 'double'
class KW_pilot : public KW_multparams
{
public:
	std::vector<double> x;
	std::vector<double> y;
	std::vector<double> z;

	KW_pilot();
};
//------------------------------------------------------------------------------------------
class KW_3points : public KW_pilot		// defines three full-dim points
{
protected:
	std::vector<double> _internal(const std::vector<double> &v) const;	// does the job for x_internal() etc

public:
	mutable std::string msg;			// message filled by _internal(): whether conversion external -> internal took place

	KW_3points();		// parameter names: P1, P2, P3

	std::vector<double> x_internal() const {return _internal(x);};		// if PARAMETERS[2] is not defined, these functions directly return x, y, z
	std::vector<double> y_internal() const {return _internal(y);};		// if PARAMETERS[2] is defined, then x, y, z are treated as external representations, and these functions return the corresponding internal representations
	std::vector<double> z_internal() const {return _internal(z);};
};
//------------------------------------------------------------------------------------------
// A common interface for KW_limits and KW_parameters.
// This interface only deals with internal representation of parameters.
class ParamsInterface : public HMMPI::BoundConstr
{
private:
	using HMMPI::BoundConstr::SobolSeq;		// hide these functions
	using HMMPI::BoundConstr::RandU;

protected:									//								values:				vector size:
	std::vector<int> act_ind;				// indices of active params		[0, full-dim)		act-dim
	std::vector<int> tot_ind;				// indices of full-dim params	[-1, act-dim)  		full-dim

	virtual void count_active() noexcept;	// fills 'act_ind', 'tot_ind' from 'act'
public:
											// min, max (internal) are inherited from HMMPI::BoundConstr
	std::vector<double> init;				// always stores internal representation
	std::vector<std::string> act;			// A, N

	virtual std::vector<double> actmin() const;						// min & max - INTERNAL ACTIVE parameters
	virtual std::vector<double> actmax() const;
	virtual std::vector<double> get_init_act() const;				// internal active values for "init"
	virtual const std::vector<int> &get_act_ind() const {return act_ind;};
	virtual const std::vector<int> &get_tot_ind() const {return tot_ind;};
	virtual std::string msg() const;								// message listing "init" values
	virtual std::vector<double> SobolDP(long long int &seed) const;	// generates Sobol design point (internal representation) in [min, max]; inactive params are set to 'init'
																	// note: 'seed' is incremented with each call
	virtual std::vector<double> RandUDP(HMMPI::Rand *rctx) const;		// generates uniform random design point (internal representation) in [min, max]; inactive params are set to 'init'
																		// state of 'rctx' changes with each call
	std::vector<std::vector<double>> SobolSequence(int n, long long int &seed) const;	// equivalent to "n" calls to SobolDP()
	std::vector<std::vector<double>> NormalSequence(int n, unsigned int seed, double R) const;		// generates n points; each point's active coords are ~ N(init, sigma^2), inactive coords = init; where sigma = R/sqrt(actdim)
																						// points which violate [min, max] are discarded
																						// the generated points will be approximately at distance R from 'init'
	virtual void Write_params_log(const std::vector<double> &p, std::string fname) const;
	virtual void Push_point(double Init, double Min, double Max, std::string AN, std::string Name);
	ParamsInterface *ActToSpherical(const HMMPI::SpherCoord &sc, double d) const;		// creates a new ParamsInterface (_DELETE_ it in the end!) by transforming the active parameters of 'this' to spherical coordinates
																						// 'd' is the gap at the poles, it will affect min & max (see PM_Spherical)
																						// if sphere center sc.c is on the box boundary, min/max for spherical coordinates will be adjusted
	ParamsInterface *CubeBounds(const std::vector<double> &c, double d) const;	// creates a new ParamsInterface (_DELETE_ it in the end!), which is an INF-NORM ball (cube) with center 'c', radius 'd' over ACTIVE parameters
																				// cut to fit within current min/max; active params are preserved; 'init' is adjusted to fit within the new bounds
};
//------------------------------------------------------------------------------------------
// min, max, initial value of parameters, and other stuff
class KW_limits : public KW_multparams, public ParamsInterface
{
protected:
	virtual void UpdateParams() noexcept;	// count active params and report
public:
								// min - 0, max - 1
	std::vector<double> norm;	// 2, init - 3
	std::vector<double> std;	// 4
	std::vector<std::string> func;			// 5 - I, EXP, LIN
	std::vector<double> dh;					// 6
	std::vector<std::string> dh_type;		// 7 - CONST, LIN
											// 8 - act (A, N)
	KW_limits();
	virtual std::string msg() const;		// message listing "init" values
	virtual void Write_params_log(const std::vector<double> &p, std::string fname) const;		// "p" - internal
	static std::string CheckPositive(const std::vector<double> &v, std::string vname);	// checks if all v[i] > 0, returns non-empty message if not (returns empty message if all ok)
																						// 'vname' is the name of vector 'v', e.g. "std", "dh"
	virtual void Push_point(double Init, double Min, double Max, std::string AN, std::string Name);
};
//------------------------------------------------------------------------------------------
// almost complete copy of LIMITS, but used for kriging meta-parameters (e.g. nugget, R, nu in KrigCorr)
class KW_limitsKrig : public KW_limits
{
public:
	KW_limitsKrig() : KW_limits() {name = "LIMITSKRIG";};
};
//------------------------------------------------------------------------------------------
// keyword for model's parameters list, similar to LIMITS, but immediately deals with external representations
class KW_parameters : public KW_multparams, public ParamsInterface, public HMMPI::ParamsTransform
{
private:
	const double ln10;						// ln(10)

	int apply_well_sc(int p, std::string s, std::vector<std::vector<double>> &work_vec, const std::vector<std::string> &wnames);		// applies "s" (e.g. "W2,W3/r2") to 2D array work_vec[N_wells x fulldim], with "wnames" - the uppercase well names, "p" - row/parameter number
																																		// returns the number of encountered errors "well not found"
protected:
	std::vector<double> norm;				// normalizing constant
	std::vector<double> logmin;				// min for func=LIN, and log10(min) for func=EXP

	std::vector<std::string> reserved_names;	// name[i] is not allowed to be one of these
	HMMPI::TagPrintfMap *par_map;			// used for writing to files

	virtual std::string par_name(int i) const;	// parameter name
	virtual double minrpt(int i) const {return min[i];};	// used in BoundConstr::Check for reporting; here, return external representation
	virtual double maxrpt(int i) const {return max[i];};
	void check_names() noexcept;			// check names for repeats, empty names, and MOD, PATH, RANK, SIZE
	void check_backvals() noexcept;			// check symbolic backvals
	virtual void fill_norm_logmin() noexcept;
	virtual void UpdateParams() noexcept;	// count active params, check names, check min <= val <= max, check min < max, check min > 0 for func=EXP, fill 'norm' and 'logmin', report
											// also, initialize 'par_map'; fill init, BoundConstr::min, BoundConstr::max (internal representation)
public:
	std::vector<std::string> name;			// 0 parameter name (tag)
	std::vector<double> val;				// 1 initial value (external) -> internal is 'init'
	std::vector<double> min;				// 2 min (external) -> internal is BoundConstr::min
	std::vector<double> max;				// 3 max (external) -> internal is BoundConstr::max
											// 4 - act [A, N]
	std::vector<std::string> backval;		// 5 value for backward propagation (number in external representation /OR/ parameter name)
	std::vector<std::string> func;			// 6 LIN, EXP
	std::vector<std::string> well_sc;		// 7 command to rescale the parameter's effect on wells, e.g. <All/r0>-<W1/r1>-<W2,W3/r2>

													// some stuff extracted from "well_sc"; 'color' is a unique tag for 'pscale'
	std::vector<std::vector<double>> uniq_sc;		// [N_colors x fulldim] table with unique 'pscale' vectors inferred from "well_sc", filled by fill_well_sc_table()
	std::vector<int> sc_colors;						// maps [0, N_wells) -> [0, N_colors), filled by fill_well_sc_table()

	KW_parameters();
	~KW_parameters();						// frees 'par_map'
	HMMPI::TagPrintfMap *get_tag_map() const {return par_map;};
	virtual std::string msg() const;		// message listing "val" values
	virtual void Write_params_log(const std::vector<double> &p, std::string fname) const;	// "p" - internal current values to be written instead of "val"
	const ParamsInterface *GetParamsInterface() const;		// if const ParamsInterface is enough for some work (no matter from KW_limits or KW_parameters), use this function;
															// it returns either KW_limits* or KW_parameters* or KW_parameters2*, depending on which one is defined (only one should be defined!);
	virtual void Push_point(double Init, double Min, double Max, std::string AN, std::string Name);

	// 'ParamsTransform' part
	virtual std::vector<double> InternalToExternal(const std::vector<double> &in) const;
	virtual std::vector<double> ExternalToInternal(const std::vector<double> &ex) const;
	virtual std::vector<double> dxe_To_dxi(const std::vector<double> &dxe, const std::vector<double> &in) const;		// transform gradient d/dxe -> d/dxi

	// work with "well_sc"
	void fill_well_sc_table(std::vector<std::string> wnames);	// fills "uniq_sc", "sc_colors"; 'wnames' are the well names (case insensitive, no duplicates)
	std::vector<int> sc_colors_textsmry();						// returns 'pscale colors' for the TEXTSMRY points with nonzero sigmas; uses DATES, ECLVECTORS, TEXTSMRY, calls fill_well_sc_table()
};
//------------------------------------------------------------------------------------------
// same as KW_parameters, but internally the values range in [-1, 1]
class KW_parameters2 : public KW_parameters
{
protected:
	virtual void fill_norm_logmin() noexcept;
public:
	KW_parameters2();
};
//------------------------------------------------------------------------------------------
// class defining the diagonal prior for certain parameters (the non-listed parameters are considered to have weak priors)
class KW_prior : public KW_multparams
{
protected:
	std::vector<int> inds_in_params;		// indices of "names" within the PARAMETERS, filled by UpdateParams()

	virtual void UpdateParams() noexcept;	// count non-weak prior parameters, check validity
public:
	std::vector<std::string> names;			// should be consistent with keyword PARAMETERS
	std::vector<double> mean;				// for the PRIOR parameter p_i corresponding to PARAMETER[S] x_j:
	std::vector<double> std;				// in case x_j is LIN: 'mean' and 'std' correspond to the range of external x_j
											// in case x_j is EXP: 'mean' and 'std' correspond to the range of log10 of external x_j
	KW_prior();
	void Mean_Cov(std::vector<double> &m, std::vector<double> &cov); 	// returns full-dim vectors "C_diag", "d" for PM_PosteriorDiag CTOR
																		// they will correspond to the internal representation
};
//------------------------------------------------------------------------------------------
// list of "day month year [hour min sec]" - dates/times used for working with well production data
class KW_dates : public KW_multparams
{
protected:
	virtual void UpdateParams() noexcept;	// fills "dates", checks their monotonicity

public:
	std::vector<int> D;	// day
	std::vector<int> M;	// month
	std::vector<int> Y;	// year

	std::vector<int> h;	// hours
	std::vector<int> m;	// minutes
	std::vector<int> s;	// seconds
	std::vector<HMMPI::Date> dates;

	KW_dates();
	std::vector<double> zeroBased();	  	// first date is subtracted from all dates, works for years from 1901 to 2099
};
//------------------------------------------------------------------------------------------
class KW_pConnect_config : public KW_params	// settings for pConnect PHYSMODEL
{
public:
	double scale;		// multiplier for objective function (should equal the number of observed data)

	KW_pConnect_config();
};
//------------------------------------------------------------------------------------------
class KW_soboltest : public KW_params	// settings for testing Sobol sequence
{
public:
	int dim;
	int seed;
	int num;
	std::string fname;

	KW_soboltest();
};
//------------------------------------------------------------------------------------------
class KW_matrixtest : public KW_params	// settings for testing HMMPI::Mat
{
public:
	int sizeA;			// number of rows for matrices A, B, C, D sequentially read from "filein"
	int sizeB;			// matrix E (vector) is read till the end of file
	int sizeC;
	int sizeD;
	std::string filein;
	std::string fileout;

	KW_matrixtest();
};
//------------------------------------------------------------------------------------------
class KW_proxyros : public KW_params	// settings for testing PM_Proxy with 2D Rosenbrock function -- a bit obsolete, kept for back-compatibility
{
public:
	double x0;				// the limits are: [x0, x1]x[y0, y1]
	double x1;
	double y0;
	double y1;
	int pts0;				// first group of points
	int pts1;				// second group of points
	std::string ptype;		// proxy type
	double R;				// correl. radius for kriging
	int trend;				// trend order: -1 (none), 0, 1, 2, 3
	int add_pts;			// maximum number of points to add (from the second group)
	int Nx;					// grid for testing: Nx x Ny
	int Ny;
	double dx;				// fin. diff. increments
	double dy;
	std::string fname;		// file name for report

	KW_proxyros();
};
//------------------------------------------------------------------------------------------
class KW_proxylin : public KW_params	// settings for testing PM_DataProxy with Linear forward operator -- a bit obsolete, kept for back-compatibility
{
public:
	double a;				// defines the limits: [-a, a]^dim
	int pts0;
	int pts1;
	std::string ptype;
	double R;
	int trend;
	int add_pts;
	int test_pts;			// number of additional points (Sobol) where testing is done
	double dx;				// fin. diff. increment
	std::string numgrad;	// YES/NO, whether numerical derivatives are calculated for MPI size = 1
	std::string psimple;	// YES/NO, whether simple proxy is calculated
	std::string fname;

	KW_proxylin();
};
//------------------------------------------------------------------------------------------
class _proxy_params : public KW_params	// common parent of KW_proxy, KW_model
{
public:								// Indented parameters (below) are meaningful only in KW_proxy
	int init_pts;					// initial number of design points; points will be taken from Sobol sequence scaled to min and max bounds from LIMITS
	int select_pts;					// on each proxy update, max 'select_pts' are taken
	std::string cfunc;		// GAUSS, SPHER, EXP, VARGAUSS, MATERN
	double nugget;					// [0, 1)
	double R;				// correlation radius
	int trend;				// -1 to 3
	double nu;				// only used for MATERN correlation function
	std::string opt;				// ON, OFF - optimize kriging parameters? (only for DataProxy2, stand-alone kriging proxy)

	std::vector<int> ind_grad_init_pts;		// these three arrays are filled in KW_proxy
	std::vector<int> ind_grad_add_pts;
	std::vector<int> ind_grad_comps;

	HMMPI::Func1D *corr;

	_proxy_params() : init_pts(0), select_pts(0), cfunc(""), nugget(0), R(0), trend(0), nu(0), opt(""), corr(0){};
	virtual ~_proxy_params(){delete corr;};
	virtual void UpdateParams() noexcept;
};
//------------------------------------------------------------------------------------------
class KW_proxy : public _proxy_params		// settings for proxy model (only kriging so far)
{
protected:
	std::string grad_init_pts;		// e.g. "0,2,3" - indices of points from 'init_pts' in which the gradients will be used for training
	std::string grad_add_pts;		// indices of points from 'select_pts'
	std::string grad_comps;			// e.g. "1,4" - indices of the gradient components participating in training

public:
	KW_proxy();
	virtual void UpdateParams() noexcept;
	static std::vector<int> str_to_vector_int(std::string s);
	static std::string vector_int_to_str(const std::vector<int> &v);
};
//------------------------------------------------------------------------------------------
class KW_proxy_dump : public KW_params		// settings for proxy's train_from_dump and dump_flag
{
protected:
	std::string dump_inds;					// comma-separated indices for dumping during MCMC, -1 is ignored

public:
	int train_ind;							// integer specifying dump file to train proxy from, in ModelFactory::Make(), -1 means train from Sobol sequence
	std::vector<int> vec_dump_inds;			// integers specifying MCMC iteration number, where proxy dump to files should be made

	KW_proxy_dump();
	virtual void UpdateParams() noexcept;
};
//------------------------------------------------------------------------------------------
class KW_model : public _proxy_params		// select between "SIM" (PMEclipse) and "PROXY" (PM_SimProxy), set simulator type (ECL, TNAV), and set proxy params
{
protected:
	PhysModel *mod;					// may store PMEclipse for automatic deletion
	PM_PosteriorDiag *mod_post_sim;	// stores the returned PM_PosteriorDiag (for automatic deletion)
	PM_PosteriorDiag *mod_post_proxy;

public:
	std::string type;		// SIM, PROXY
	std::string simulator;	// ECL, TNAV
	// proxy params: R, trend, cfunc, nu, nugget
	// 'cfunc' - here only GAUSS, MATERN

	KW_model();
	virtual ~KW_model();
	PhysModel *MakeModel(KW_item *kw, std::string cwd, bool is_posterior, std::string Type = "Default");
					// IF is_posterior == true:
							// generates PMEclipse or PM_SimProxy wrapped in PM_PosteriorDiag
							// thus, the model is POSTERIOR = PRIOR + LIKELIHOOD
					// IF is_posterior == false:
							// generates PMEclipse or PM_SimProxy
					// the model is created on MPI_COMM_WORLD
					// KW_item "kw" which called MakeModel() is only used to handle prerequisites.
					// Type can be "SIM" or "PROXY"; if Type == "Default", then this->type is used
					// the returned model is DELETED AUTOMATICALLY by DTOR
};
//------------------------------------------------------------------------------------------
class KW_matvecvec : public KW_item, public HMMPI::StdCreator, public HMMPI::DataCreator		// reads matrix and two vectors: Mat | Vec | Vec
{
public:
	HMMPI::Mat M;
	std::vector<double> v1;
	std::vector<double> v2;

	KW_matvecvec();
	virtual void ProcessParamTable() noexcept;
	virtual void Action() noexcept;			// some reporting

	virtual std::vector<double> Std() const;			// sqrt(v2)
	virtual std::vector<double> Data() const;			// v1
};
//------------------------------------------------------------------------------------------
class KW_matvec : public KW_item			// reads matrix and one vector: Mat | Vec
{
public:
	HMMPI::Mat M;
	std::vector<double> v1;

	KW_matvec();
	virtual void ProcessParamTable() noexcept;
	virtual void Action() noexcept;			// some reporting
};
//------------------------------------------------------------------------------------------
class KW_mat : public KW_item				// reads a matrix
{
public:
	HMMPI::Mat M;

	KW_mat();
	virtual void ProcessParamTable() noexcept;
	virtual void Action() noexcept;			// some reporting
};
//------------------------------------------------------------------------------------------
class KW_LinSolver : public KW_params		// selects the type(s) of solver(s) to be used for different linear systems
{
protected:
	std::vector<HMMPI::Solver*> vecsol;		// stores Solver pointers

public:
	std::string sol1;		// GAUSS, DGESV, DGELS, DGELSD, DGELSS, DGELSY

	const HMMPI::Solver *Sol(int i) const;		// pointer to Solver object #i
	int SolSize() const {return vecsol.size();};
	KW_LinSolver();
	virtual void UpdateParams() noexcept;	// creates Solver objects in 'vecsol'
	virtual ~KW_LinSolver();				// clears 'vecsol'
};
//------------------------------------------------------------------------------------------
class KW_MCMC_config : public KW_params	// settings for MCMC samplers
{
protected:
	ModelFactory Factory;	// MakeSampler() uses this factory to generate models, so the factory should live a long time

public:
	std::string sampler;	// RWM, PCN, HMC, HMCREJ, SOLHMC, RHMC, MMALA, SIMPLMMALA, MMALA2, I_MALA
	int iter;				// number of total iterations (= samples)
	int burn_in;			// number of burn-in iterations - during burn-in algorithm params are adjusted
	int seed;				// if seed == 0, it will be initialised from time
	std::string MM_type;	// mass matrix type: HESS - Hessian, FI - Fisher information matrix, UNITY - unity matrix, BFGS, MAT
	double nu;				// lower bound for eigenvalues of mass matrix
	double gamma;			// initial BFGS 'Hessian' = gamma*I
	int LFG_maxref;			// maximum refinement for LFG
	double eps;				// starting leapfrog step size
	double maxeps;			// maximum leapfrog step size (HMC, RHMC only)
	int LF_steps;			// number of leapfrog steps in one iteration
	std::string LF_bounce;	// type of bouncing at borders: NEG, CHOL, EIG, HT
	int upd_freq;			// update proxy, mass matrix and epsilon every 'upd_freq' iterations
	int upd_type;			// 0, 1
	double acc_targ;		// target acceptance rate; for epsilon update (when rate == target, mult = 1)
	double alpha;			// for epsilon update (when rate == 1, mult = beta^alpha)
	double beta;			// for epsilon update (when rate == 0, mult = beta^(-1))
	double I_alpha;			// alpha for I-MALA
	double ii;				// for SOL-HMC

	KW_MCMC_config();
	HMMPI::MCMC *MakeSampler(KW_item *kw);		// make a fully functional MCMC sampler based on the settings provided, esp. 'sampler' string
												// don't forget to DELETE in the end!
												// KW_item "kw" is only used to handle prerequisites
};
//------------------------------------------------------------------------------------------
class KW_corrstruct : public KW_multparams, public HMMPI::CorrelCreator		// correlation structure for covariance matrix
{
public:
	std::vector<int> Bsize;
	std::vector<double> R;
	std::vector<std::string> type;

	HMMPI::Mat Corr();								// calculate the whole correlation matrix
	std::vector<HMMPI::Mat> CorrBlocks() const;		// calculate the blocks of the correlation matrix (N x N blocks, or N x 1 diagonals)

	KW_corrstruct();
	virtual void UpdateParams() noexcept;		// set default Bsize's and R's differently
	int size();									// total size of all blocks
};
//------------------------------------------------------------------------------------------
class KW_physmodel : public KW_multparams
{
public:
	std::vector<std::string> type;	// ECLIPSE, SIMECL, PCONNECT, CONC, SIMPROXY, LIN, ROSEN, FUNC_LIN, FUNC_POW (plain types);
									// NUMGRAD, PROXY, DATAPROXY, DATAPROXY2, KRIGCORR, KRIGSIGMA, LAGRSPHER, SPHERICAL, CUBEBOUND, HAMILTONIAN, POSTERIOR, POSTERIOR_DIAG (referencing types)
	std::vector<int> ref;			// "ref" refers to another model defined in row № "ref"
									// ref == 0 does not refer to any row, and should be used for 'plain' models
	std::vector<bool> is_plain;		// indicates whether the given entry is of 'plain type'

	std::string CheckRefs();		// checks if "ref" values are within bounds, and that there are no loops; returns "" if all is ok, error message otherwise
									// fills 'is_plain', and checks "ref" validity for different types
	KW_physmodel();
	virtual void UpdateParams() noexcept;
};
//------------------------------------------------------------------------------------------
class KW_vectmodel : public KW_params
{
private:
	VectorModel *mod;		// freed in dtor

public:
	std::string type;		// GRADIENT, HAM_EQ1, HAM_EQ2, HAM_EQ2_EPS, HAM_EQ2_EPSFULL

	KW_vectmodel();
	virtual ~KW_vectmodel() {delete mod;};
	VectorModel *Make(PhysModel *pm);	// returned model is freed automatically in the end
};
//------------------------------------------------------------------------------------------
// TODO descendants of KW_fname
//------------------------------------------------------------------------------------------
class KW_include : public KW_fname
{
protected:
	std::vector<inputLN> IncludeLines(int shift, std::vector<std::string> lines);	// get IncludeLines from 'lines', with 'shift' and proper 'CWD'
	virtual void DataIO(int i);			// reads include file, passes it to K->AddInputLines()

public:
	KW_include();
};
//------------------------------------------------------------------------------------------
// multiple files, each with 2 columns (x, y) defining a piecewise linear function;
// can also accept "default value" 1* as a file name - which will be treated as "empty file"
class KW_functionXY : public KW_fname
{
protected:
	HMMPI::Vector2<double> ReadData(std::string fn);
	virtual void AllocateData() noexcept;
	virtual void DataIO(int i);
public:
	std::vector<HMMPI::Vector2<double>> data;   // [file] x [x, y]

	KW_functionXY();
	double FuncValBin(int func_ind, double x, int i1, int i2);
	double FuncVal(int func_ind, double x);
	virtual void Action() noexcept;
};
//------------------------------------------------------------------------------------------
class KW_Pcapill : public KW_functionXY
{
public:
	KW_Pcapill();
	double FuncValExtrapol(int func_ind, double x);		// extrapolates beyond the range using constants
};
//------------------------------------------------------------------------------------------
class KW_Dtable : public KW_fname			// one file with M x N table of <double>
{
protected:
	virtual void DataIO(int i);

public:
	std::vector<std::vector<double>> data;  // M x N, row-major storage

	KW_Dtable();
	static std::vector<std::vector<double>> ReadTableFromFile(std::string fn);
};
//------------------------------------------------------------------------------------------
class KW_conc_data : public KW_Dtable
{
protected:
	virtual void DataIO(int i);

public:
	KW_conc_data();
};
//------------------------------------------------------------------------------------------
class KW_fsmspec : public KW_fname
{
protected:
	std::string aux;
	std::string DELIM;

	std::string GetItem();	// reads leftmost '...' from aux, cuts from aux and returns it without ''; returns "" if std::string not found
	std::vector<std::string> ReadChars(std::string fn, std::string HDR);		// HDR = 'KEYWORDS', 'WGNAMES ', 'UNITS   '
	virtual void DataIO(int i);
public:
	std::vector<int> ind;	// for i-th item in ECLVECTORS, ind[i] is its index (0-based) in *.FSMSPEC
						// ECLVECTORS should be defined prior to reading
	std::vector<int> indH;	// index for historic vectors
	int Y, M, D;		// indices of year, month, day in *.FSMSPEC
	int not_found;		// counts not found vectors

	KW_fsmspec();
	std::vector<int> ReadData(std::string fname, int &y, int &m, int &d, std::string *K_msg);		// returns ind, updates y, m, d, updates K_msg (if not null)
	std::vector<int> ReadDataH(std::string fname, std::string *K_msg);								// returns indH, updates K_msg (if not null)
	void GetKeywordIndRange(std::string fname, std::string kwd, int &start, int &end);		// fills [start, end) - range of indices of some keyword, e.g. kwd = "QWBHP"
																							// only the first found contiguous range is taken; returns start = -1 if no range is found
};
//------------------------------------------------------------------------------------------
class KW_funrst : public KW_fname		// reads cube SWAT from FUNRST for specified SATSTEPS
{
protected:
	bool fixedFegrid;

	virtual void DataIO(int i);
public:
	typedef std::map<std::string, std::vector<double>> grad;		// e.g. <PORO, and derivatives w.r.t. PORO>

	std::vector<std::vector<double>> data;		// [time] x [cube]

	KW_funrst();
	std::vector<std::vector<double>> ReadData(std::string fname, std::string prop, bool fixed_fegrid);		// prop = 'SWAT    ', 'SGAS    ',...
																						// if fixed_fegrid, ACTNUM is loaded from KW_fegrid, otherwise - from fname.FEGRID
	HMMPI::Vector2<grad> ReadGrads(std::string mod_root);			// reads derivatives of well data w.r.t. grid properties (PORO, PERMX, etc) from multiple formatted files {mod_root.F000i}, using mod_root.FEGRID
																	// resulting array is [Nsteps, Nvecs] where Nsteps is from DATES, Nvecs is from ECLVECTORS
	static HMMPI::Vector2<double> GradsOfProperty(const HMMPI::Vector2<grad> &grad, std::string prop, int cell);	// extracts array [Nsteps, Nvecs] from "grad", where the derivatives are taken w.r.t. "prop" in cell "cell"
	static HMMPI::Vector2<double> GradsOfRegion(const HMMPI::Vector2<std::vector<double>> &grad, int reg);			// extracts array (*,*) = grad(*,*)[reg], "reg" is zero-based
	std::vector<Grid2D> ReadDataGrid2D(std::string fname, std::string prop, int cX, int cY, std::string fn_init);	// Grid2D (cX x cY)
	std::vector<Grid2D> GetGrid2D(int cX, int cY, std::string fn_init);			// data -> {Grid2D}
	std::vector<double> ReadDataInit(std::string fname, std::string prop);		// prop = 'PERMX   ',...
};
//------------------------------------------------------------------------------------------
class KW_funrstG : public KW_funrst		// reads SGAS
{
protected:
	virtual void DataIO(int i);

public:
	KW_funrstG();
};
//------------------------------------------------------------------------------------------
class KW_funrstA : public KW_funrst		// reads ATTR
{
protected:
	virtual void DataIO(int i);

public:
	KW_funrstA();
};
//------------------------------------------------------------------------------------------
class KW_fegrid : public KW_fname		// reads ACTNUM from FEGRID
{
protected:
	virtual void DataIO(int i);

public:
	std::vector<double> data;

	KW_fegrid();
	std::vector<double> ReadData(std::string fname, std::string prop);		// prop = 'ACTNUM  ',...
	double Sum();			  			// sum data[i]
};
//------------------------------------------------------------------------------------------
class KW_funsmry : public KW_fname
{
protected:
	int count_dates;	// count successfully loaded dates and vectors
	int count_vecs;

	virtual void DataIO(int i);
public:
	HMMPI::Vector2<double> data;            	// [time, vec]
	std::vector<int> taken_files;				// numbers of files (multiple output) taken by ReadData(#2) according to DATES list

	KW_funsmry();
	static int DateCmp(int Y1, int M1, int D1, int Y2, int M2, int D2);		// date comparison: -1, 0, 1
	HMMPI::Vector2<double> ReadData(std::string fname, bool read_hist = false);
	HMMPI::Vector2<double> ReadData(std::string mod_root, int i0, int i1);	// reads MULTIPLE formatted output from {mod_root.A000i}, i = [i0, i1), the file numbers "i" taken according to DATES are saved to "taken_files"
																			// taken_files.size = DATES.size, taken_files[j] = -1 for not found dates j.
																			// otherwise, works like ReadData(#1), but doesn't care about UNDEF
};
//------------------------------------------------------------------------------------------
class KW_textsmry : public KW_fname, public HMMPI::SigmaMessage
{
private:
	int warnings;						// inner count of warnings at DataIO
protected:
	HMMPI::Vector2<std::string> Hdr;	// 2-lines header, contains H and S columns
	std::vector<int> ind;				// indices of H-columns
	std::string msg;
	int not_found;						// counts not found vectors
	int found_ts;						// counts found time steps

	void ReadInd(std::string *K_msg);	// reads from "Hdr", fills "ind", "ind_sigma", updates "not_found"
	virtual void DataIO(int i);
public:
	HMMPI::RandNormal randn;
	std::vector<int> ind_sigma;			// indices of S-columns
	HMMPI::Vector2<double> data;       	// [time, 2 x vec], stores both historical values and sigmas
	HMMPI::Vector2<double> pet_dat;		// perturbed data - for RML

	KW_textsmry();
	HMMPI::Vector2<double> ReadData(std::string fname);
	virtual std::string SigmaInfo(const std::string &wgname, const std::string &keyword) const;	// from HMMPI::SigmaMessage
	std::vector<double> OnlySigmas() const;			// returns the 'sigmas' part of 'data', ordered as vec_0(all dates), vec_1(all dates), ...
};
//------------------------------------------------------------------------------------------
class KW_refmap : public KW_fname		// reads Eclipse-style 3D grid (ASCII)
{
protected:
	std::vector<double> ReadData(std::string fname);
	virtual void DataIO(int i);

public:
	std::vector<double> data;

	KW_refmap();
	Grid2D GetGrid2D(int cX, int cY, std::string fn_init);
	Grid2D GetGrid2D(int cX, int cY);					// weight = 1
};
//------------------------------------------------------------------------------------------
class KW_refmap_w : public KW_refmap
{
public:
	KW_refmap_w();
};
//------------------------------------------------------------------------------------------
class KW_mapreg : public KW_refmap
{
public:
	KW_mapreg();
};
//------------------------------------------------------------------------------------------
class KW_mapseisscale : public KW_refmap
{
public:
	KW_mapseisscale();
};
//------------------------------------------------------------------------------------------
class KW_mapseiswght : public KW_refmap
{
public:
	KW_mapseiswght();
};
//------------------------------------------------------------------------------------------
class KW_refmapM : public KW_refmap		// reads multiple 3D grids
{
protected:
	virtual void DataIO(int i);
	virtual void AllocateData() noexcept;
public:
	std::vector<std::vector<double>> data;		// "hides" KW_refmap::data

	KW_refmapM();
	std::vector<double> LoadFromFile(std::string fn);
};
//------------------------------------------------------------------------------------------
class KW_initcmaes : public KW_fname
{
protected:
	std::string fn_cmaes_init;

	virtual void DataIO(int i);
public:
	std::string data;

	KW_initcmaes();
	void WriteFromLimits(const std::vector<double> &x);		// x is full-dim inner variables (written instead of 'init'); to be called on all ranks
};
//------------------------------------------------------------------------------------------
class KW_datafile : public KW_fname
{
protected:					// PATH/MODEL0.DATA
	std::string contents;
	std::vector<std::string> cont_split;

	virtual void DataIO(int i);
public:
	std::string base_name;		// MODEL
	std::string path;			// PATH

	KW_datafile();
	void WriteDataFile(int i, bool adjrun = false);	// MODELi.DATA, "adjrun" enables ad hoc modifications to switch to adjoint run
	std::string GetDataFileName(int i);		// name of MODELi.DATA
};
//------------------------------------------------------------------------------------------
class KW_CoordZcorn : public KW_fname		// reads a file with COORD & ZCORN data
{
protected:
	virtual void DataIO(int i);

public:
	HMMPI::CornGrid CG;

	KW_CoordZcorn();
};
//------------------------------------------------------------------------------------------
class KW_Actnum : public KW_fname			// reads a file with ACTNUM data to KW_CoordZcorn::CG
{
protected:
	virtual void DataIO(int i);

public:
	KW_Actnum();
};
//------------------------------------------------------------------------------------------
// TODO descendants of KW_fwrite
//------------------------------------------------------------------------------------------
class KW_WRfunrst : public KW_fwrite
{
protected:
	virtual void DataIO(int i);

public:
	KW_WRfunrst();
};
//------------------------------------------------------------------------------------------
class KW_report : public KW_fwrite
{
protected:
	virtual void DataIO(int i);

public:
	KW_report();
};
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------

#endif /* PARSING2_H_ */
