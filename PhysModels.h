/*
 * PhysModels.h
 *
 *  Created on: 23 May 2016
 *      Author: ilya fursov
 */

#ifndef PHYSMODELS_H_
#define PHYSMODELS_H_

#include <vector>
#include "MathUtils.h"
#include "Abstract.h"
#include <mpi.h>
#include <map>

//---------------------------------------------------------------------------
class Parser_1;
class KW_item;
//---------------------------------------------------------------------------
// BDC_creator - base class for PhysModels, it handles the data covariance (BDC) stuff: creation and transfer.
// *** MPI philosophy: Use on all processes in MPI_COMM_WORLD. See more in 'PhysModel' comments below.
//---------------------------------------------------------------------------
class BDC_creator
{
protected:
	MPI_Comm comm;								// NOTE: "comm" handles distribution of data points (like in BDC), not the distribution of 'forward runs'
	int RNK;									// MPI_COMM_WORLD rank value filled at construction time - for reporting and file id

	const HMMPI::BlockDiagMat *BDC;				// block-diagonal covariance matrix (MPI-distributed with "comm")
												// if it is NULL, this should be sync

	HMMPI::Mat data_sens_loc;					// stores local fulldim sensitivities (MPI-distributed with "comm"), calculated in ObjFuncGrad, used in ObjFuncFisher, ObjFuncFisher_dxi
	HMMPI::Mat data_sens_loc_act;				// stores local actdim sensitivities, filled in ObjFuncGrad_ACT

public:
	BDC_creator(MPI_Comm c = MPI_COMM_SELF, const HMMPI::BlockDiagMat *bdc = nullptr);		// Communicator "comm" is given at construction, and cannot be changed afterwards.
	virtual ~BDC_creator(){};
	const HMMPI::BlockDiagMat *GetBDC(const HMMPI::BlockDiagMat **bdc) const;	// Philosophy:
			// The pointer filled in 'bdc' should be used in production, it provides the BDC mathematically assumed by the current model.
			// The pointer returned is for the memory management, it should be safely freed (from the outside) when the BDC is no longer needed.
			// Simple models are supposed to create (and return) a new BDC object.
			// Composite models are supposed to get a BDC at construction, and then transfer it when this function is called.
	const HMMPI::Mat &DataSens_loc() const {return data_sens_loc;};
	const HMMPI::Mat &DataSens_loc_act() const {return data_sens_loc_act;};
};
//---------------------------------------------------------------------------
// PhysModel - base class for all "physical models"
// physical models allow calculation of objective function values, gradients and Hessians for the given parameters vector
// parameters vector can be
// - full vector which matches the dimension of the model -- this works with ObjFuncXXX
// - (smaller) vector of active parameters; the inactive (remaining) parameters are pinned to some initial values -- this works with ObjFuncXXX_ACT
// *** MPI philosophy:
// The model should be created and used on all processes in MPI_COMM_WORLD.
// All input function parameters should be sync on MPI_COMM_WORLD (this is sufficient for safety), unless opposite is said.
// Communicator "comm" is given to the model at construction, and cannot be changed afterwards.
// "comm" is used to provide communication the model may need within its functions. It handles the distribution of data points (like in BDC).
// For simple cases comm = MPI_COMM_SELF (no communication is needed).
// The functions output (and many member variables) are guaranteed to be correct at comm-RANKS-0 (at other ranks -- no guarantee).
// "comm" may be MPI_COMM_NULL at some processes - in this case some functions may be abandoned earlier.
// *** USER: call on all processes in MPI_COMM_WORLD, collect results on comm-RANKS-0.
// *** DEVELOPER: make sure comm-RANKS-0 get correct results, make sure comm-RANKS-NULL are properly treated.
//---------------------------------------------------------------------------
class PhysModel : public HMMPI::ManagedObject, public BDC_creator
{
private:

protected:
										// philosophy: "init", "act_ind", "tot_ind", "con" should be sync between ranks of MPI_COMM_WORLD
	std::vector<double> init;			// (total) vector of initial parameters - used in ObjFuncXXX_ACT; its size = ParamsDim
	std::vector<size_t> act_ind;		// indices of active params within full-dim params; 0 <= act_ind[i] < ParamsDim; size = dimension of active parameters
	std::vector<size_t> tot_ind;		// indices of full-dim params within active params; 0 <= tot_ind[i] < dimension of active parameters, or tot_ind[i] = -1; size = ParamsDim
	const HMMPI::BoundConstr *con;		// object for checking constraints, see CheckLimits()

									// *** LOGIC for modelled data and sensitivities: after ObjFunc or ObjFuncGrad calls, the corresponding data structures
									// (modelled_data, data_sens, data_sens_loc, data_sens_act, data_sens_loc_act)
									// keep the up-to-date values until the next call to these functions; i.e. they work as caches.
	std::vector<double> modelled_data;	// vector of modelled data (ordered in some way); this is supposed to be filled, if possible, by ObjFunc()
	HMMPI::Mat data_sens;				// sensitivity matrix of modelled data, indexed as (ind_data, fulldim); this is supposed to be filled, if possible, by ObjFuncGrad()
	HMMPI::Mat data_sens_act;			// restriction of data_sens to active parameters (ind_data, actdim), filled by ObjFuncGrad_ACT()

	mutable std::string limits_msg;		// message optionally created by CheckLimits()

	bool of_cache_valid, grad_cache_valid, hess_cache_valid;		// some data for caching o.f., gradients and Hessians
	std::vector<double> last_x_of, last_x_grad, last_x_hess;
	double of_cache;
	std::vector<double> grad_cache;
	HMMPI::Mat hess_cache;
public:
	std::string name;					// model name -- mostly for debug purposes

	PhysModel(MPI_Comm c = MPI_COMM_SELF, const HMMPI::BlockDiagMat *bdc = nullptr);
	PhysModel(std::vector<double> in, std::vector<size_t> act, std::vector<size_t> tot, const HMMPI::BoundConstr *c);
	PhysModel(Parser_1 *K, KW_item *kw, MPI_Comm c);		// easy constructor; all data are taken from K->LIMITS/PARAMETERS; "kw" is used only to handle prerequisites
	virtual ~PhysModel();
										// philosophy: input (and hence output) for these two functions should be the same on all ranks of MPI_COMM_WORLD
	std::vector<double> tot_par(const std::vector<double> &act_par) const;	// convert active params vector to total params vector based on 'init', 'act_ind'
	std::vector<double> act_par(const std::vector<double> &tot_par) const;	// convert total params vector to active params vector based on 'act_ind'
	HMMPI::Mat act_mat(const HMMPI::Mat &tot_M) const;						// convert square matrix of "total" dimension to the "active" sub-matrix, based on 'act_ind'

	virtual double obj_func_work(const std::vector<double> &params);						// These lower case functions perform the concrete calculations (o.f., grad, Hess)
	virtual std::vector<double> obj_func_grad_work(const std::vector<double> &params);		// subsequently used in caching.
	virtual HMMPI::Mat obj_func_hess_work(const std::vector<double> &params);				// Override them in derived classes!

	virtual double ObjFunc(const std::vector<double> &params) final;					// Objective function value [and modelled_data].
	virtual std::vector<double> ObjFuncGrad(const std::vector<double> &params) final;	// Gradient of objective function [and data_sens]; if no gradient calculation is available, produces error.
	virtual HMMPI::Mat ObjFuncHess(const std::vector<double> &params) final;			// Hessian of objective function; if no Hessian calculation is available, produces error.

	virtual double ObjFuncGradDir(const std::vector<double> &params, const std::vector<double> &dir);		// grad' * dir; if necessary, override it by a more efficient implementation
	virtual HMMPI::Mat ObjFuncFisher(const std::vector<double> &params);			// Fisher Information matrix (only for models with data; for other models produces error); actually, FI corresponds to Likelihood = exp(-1/2*func)
	virtual HMMPI::Mat ObjFuncFisher_dxi(const std::vector<double> &params, const int i, int r = 0);		// derivative of FI matrix w.r.t. x_i (i - index in total dim); "r" is the rank where result should be accessed
	virtual HMMPI::Mat ObjFuncFisher_mix(const std::vector<double> &params);		// mix between FI and Hess for composite models, essentially PM_Spherical; replaces some expensive Hess parts by cheaper 2*FI parts; to be used instead of Hess in LM optimization
																					// by default falls back to 2*FI
	virtual double ObjFunc_ACT(const std::vector<double> &params);						// the following five versions of ObjFuncXXX only work with active parameters (act_ind)
	virtual std::vector<double> ObjFuncGrad_ACT(const std::vector<double> &params);		// inactive parameters always take the corresponding values from 'init'
	virtual double ObjFuncGradDir_ACT(const std::vector<double> &params, const std::vector<double> &dir);		// these functions are based on the five 'usual' functions above; normally ObjFuncXXX_ACT should work fine in all derived classes
	virtual HMMPI::Mat ObjFuncHess_ACT(const std::vector<double> &params);										// override these functions if more efficient implementation is needed
	virtual HMMPI::Mat ObjFuncFisher_ACT(const std::vector<double> &params);
	virtual HMMPI::Mat ObjFuncFisher_dxi_ACT(const std::vector<double> &params, const int i, int r = 0);		// i - index in act dim; "r" is the rank where result should be accessed
	virtual HMMPI::Mat ObjFuncFisher_mix_ACT(const std::vector<double> &params);

	// philosophy: CheckLimits[_ACT], ParamsDim[_ACT], FindIntersect[_ACT] should work identically on all ranks of MPI_COMM_WORLD
	virtual bool CheckLimits(const std::vector<double> &params) const;					// 'true', if "params" don't violate the constraints 'con'; if con == 0, 'true' is always returned
	virtual bool CheckLimits_ACT(const std::vector<double> &params) const;				// version which only works with active parameters
	virtual bool CheckLimitsEps(std::vector<double> &params, const double eps) const;	// works as CheckLimits, but allows small violations 'eps'; where these violations take place, 'params' is adjusted
	bool FindIntersect(const std::vector<double> &x0, const std::vector<double> &x1, std::vector<double> &xint, double &alpha, int &i) const;		// works like CheckLimits(x1), but also finds the intersection point if bounds are violated, see Constraints::FindIntersect
	bool FindIntersect_ACT(const std::vector<double> &x0, const std::vector<double> &x1, std::vector<double> &xint, double &alpha, int &i) const;	// version which only works with active parameters (this concerns the three vectors and 'i')
	virtual void WriteLimits(const std::vector<double> &p, std::string fname) const;			// write parameters 'p' to file 'fname'
	virtual void WriteLimits_ACT(const std::vector<double> &p, std::string fname) const final;	// write active parameters 'p' to file 'fname'
	virtual int ParamsDim() const noexcept = 0;										// dimension of parameters space
	int ParamsDim_ACT() const noexcept;												// dimension of active parameters

	virtual size_t ModelledDataSize() const {return 0;};	// returns the supposed size of "modelled_data", independently of "modelled_data";
															// IMPLEMENT IT such that it returns identical results for each rank of MPI_COMM_WORLD, and does not involve any communication
	virtual const std::vector<double> &ModelledData() const;		// returns "modelled_data" where it is available; produces error where it's not; NB "modelled_data" may not be sync!
	const HMMPI::Mat &DataSens() const {return data_sens;};
	const HMMPI::Mat &DataSens_act() const {return data_sens_act;};

	virtual void PerturbData();											// perturb data for RML (and sync/distribute between ranks); typically KW_textsmry->randn is used for random number generation;
																		// only models which have data (e.g. PhysModelHM, PM_Linear, PM_DataProxy) do something here (others produce exception);
																		// some other models (e.g. PhysModMPI, PhysModGradNum) perform PerturbData call for the referenced model
	virtual std::string ObjFuncMsg() const {return "";};			// a message which can be printed after ObjFunc calculation
	virtual void SavePMState(){};		// saves some "state" of PM before ObjFuncMPI
	virtual void RestorePMState(){};	// restores "state" of PM after running ObjFuncMPI
	virtual void SetIntTag(int tag){};	// set some tag, e.g. a sample number in MCMC
	void ExportIAC(PhysModel *p) const;				// exports 'init', 'act_ind', 'tot_ind', 'con' to "p"

	MPI_Comm GetComm() const {return comm;};							// get communicator
	const HMMPI::BoundConstr *GetConstr() const {return con;};			// get constraints
	virtual std::string proc_msg() const {return "";};					// meaningful message - for PM_Proxy and its descendants; SHOULD BE CALLED ON ALL RANKS!
	virtual std::string get_limits_msg() const {return limits_msg;};
	virtual bool is_proxy() const {return false;};
	virtual bool is_dataproxy() const {return false;};
	void print_comms() const;			// print comm and BDC->comm, for debug pusposes
};
//---------------------------------------------------------------------------
// ModelFactory - class for creating instances of different PhysModels
// Created PhysModels are stored on heap and are freed by ModelFactory destructor
// To free a PhysModel before destructor, call FreeModel()
//---------------------------------------------------------------------------
class ModelFactory
{
protected:
	std::map<PhysModel*, std::vector<HMMPI::ManagedObject*>> ptrs;	// pointers to objects are stored here to delete ManagedObjects when necessary

	void FillCreators(Parser_1 *K, KW_item *kw, HMMPI::CorrelCreator **cor, HMMPI::StdCreator **std, HMMPI::DataCreator **data);		// fill from CORRSTRUCT, MATVECVEC
	void MakeComms(MPI_Comm in, MPI_Comm *one, MPI_Comm *two, bool ref_is_dataproxy, bool ref_is_simproxy);		// create two communicators from "in": for NUMGRAD and its reference model
	static bool object_for_deletion(const HMMPI::ManagedObject *m);	// 'true', if "m" is to be deleted in dtor/FreeModel; 'false' otherwise (e.g. for KrigCorr)
public:
	PhysModel *Make(std::string &message, Parser_1 *K, KW_item *kw, std::string cwd, int num, MPI_Comm c = MPI_COMM_SELF, std::vector<HMMPI::ManagedObject*> *mngd = 0, bool train = true);
							// Create a PhysModel of given type -- according to line "num" (1-based) in keyword PHYSMODEL.
							// If line "num" in PHYSMODEL further refers to another line (say, 'num_2'), the model corresponding to 'num_2' will be created internally, and deleted automatically in the end.
							// All data are taken from keywords of "K"; KW_item "kw" which called Make() is only used to handle prerequisites.
							// Pointers returned by Make (or created internally) will be added to "ptrs" if mngd == 0 (for internal uses, with mngd != 0, these pointers are added to 'mngd').
							// Contents of "ptrs" are automatically freed in the end by ModelFactory destructor.
							// OR, pointers associated with a particular returned PhysModel* PM can be deleted by calling FreeModel(PM).
							// Communicator of the new PhysModel is set to 'c'.
							// For NUMGRAD, communicator is created from 'c' ('c' is split into two communicators).
							// For DATAPROXY, ECLIPSE and NUMGRAD, if c == MPI_COMM_SELF, then c = MPI_COMM_WORLD is used (with further splitting for NUMGRAD).
							// "message" is the output message indicating which model (chain of models) were employed.
							// If "train" == true, and the requested model is PROXY/DATAPROXY, it will be trained.
							// USER: call Make() with 'c' and 'mngd' as default parameters.
	~ModelFactory();
	void FreeModel(PhysModel* pm);			// frees "pm" and all objects (stored on heap) associated with it
};
//---------------------------------------------------------------------------
// PhysModMPI - wrapper class for massive calculations of objective function.
// It's not supposed to work with plain ObjFuncXXX; but otherwise it maintains the functionality of the PhysModel it wraps.
// All the member functions should be called on all ranks in MPI_COMM_WORLD.
// Communicators "comm" and "PM_comm" should be a 2-level partition of some communicator, e.g.
// 012345678 -> partitioned into comm: 0xx1xx2xx, PM_comm: [012][012][012].
//---------------------------------------------------------------------------
class PhysModMPI : public PhysModel
{
protected:
	PhysModel *PM;				// core PhysModel
	std::vector<int> countFIT, countPOP, countSMRY, displFIT, displPOP, displSMRY;		// vectors on comm-rank-0 for Scatterv/Gatherv
	void fill_counts_displs(int ind1);			// auxiliary function to fill the above vectors; call it on all ranks; "ind1" is the upper index for models on each rank

public:
	PhysModMPI(MPI_Comm c, PhysModel *pm, const HMMPI::BlockDiagMat *bdc = nullptr);		// 'c' and PM_comm should form a 2-level partition
	static void HMMPI_Comm_check(MPI_Comm first, MPI_Comm second, const std::string &where);		// Check if the two communicators form a 2-level partition, i.e.
										// first-ranks <-> second-ranks-0 && second-ranks-[1,2..] -> first-ranks-NULL
										// To be called on MPI_COMM_WORLD. If the test fails, exception is thrown (sync on MPI_COMM_WORLD).
										// "where" may be a function/class name, e.g. "PhysModMPI"
	static void HMMPI_Comm_split(int fst_size, MPI_Comm comm, MPI_Comm *first, MPI_Comm *second);	// Make 2-level partition <first, second> from "comm"
										// "fst_size" is the size [of each group] of "first" (= number of corresponding groups in "second").
										// Example: fst_size = 4, comm = MPI_COMM_WORLD(16) -> first = 0xxx1xxx2xxx3xxx, second = [0123][0123][0123][0123]
										// To be called on MPI_COMM_WORLD.
										// *** DONT' FORGET TO FREE the new communicators (!= MPI_COMM_NULL) in the end ***
	void ObjFuncMPI_ACT(int len, const double * const *POP, double *FIT, bool quiet_out_of_range = false, double **SMRY = 0);	// Calculate ObjFunc_ACT for ensemble of models using MPI.
								// This function should be called on ALL RANKS in MPI_COMM_WORLD, or at least on all PM_comm-ranks.
								// Parameters passed to (returned from) this function are only referenced on comm-ranks-0.
								// Exceptions from PM_comm-ranks are intercepted and re-thrown synchronously (on comm-ranks they will also carry a sync message).
								// Inputs:  POP = [len][actdim] - parameters for the population, "len" - number of models in population
								// 			quiet_out_of_range - if 'true', then models with violated params limits return o.f. = NaN; if 'false', such models try to calculate objective function.
								// Outputs: FIT = [len] - o.f. values for the population, (optional) SMRY = [len][smry_len] - modelled data for the population

	virtual double ObjFunc_ACT(const std::vector<double> &params);	// this function employs PM->ObjFunc_ACT, and Bcasts the results (and modelled_data) to "comm"

	// the following functions simply wrap the corresponding functions of "PM"; no Bcast over "comm" is done (to avoid MPI deadlocking)
	virtual int ParamsDim() const noexcept;									// no Bcast -- make sure PM->ParamsDim produces correct results on all ranks in MPI_COMM_WORLD
	virtual size_t ModelledDataSize() const;								// no Bcast -- make sure PM->ModelledDataSize produces correct results on all ranks in MPI_COMM_WORLD
	virtual void PerturbData() {PM->PerturbData();};				// all ranks do the same job
	virtual void SavePMState() {PM->SavePMState();};				// all ranks do the same job
	virtual void RestorePMState() {PM->RestorePMState();};			// all ranks do the same job
	virtual std::string get_limits_msg() const {return PM->get_limits_msg();};
};
//---------------------------------------------------------------------------
// PhysModGradNum - wrapper class which can calculate gradients, Hessians, and data sensitivities numerically
// the object of this class should be created and used on all processes in MPI_COMM_WORLD
// the 'active parameters style functions' ObjFuncXXX_ACT should be called, not ObjFuncXXX
//---------------------------------------------------------------------------
class PhysModGradNum : public PhysModMPI
{
protected:
	double of_val;				// objective function value from the last call of this->ObjFunc_ACT(); sync across "comm"
	std::vector<double> par;	// params values from the last call of this->ObjFunc_ACT(); sync across "comm"

	HMMPI::Vector2<double> make_coeff(size_t dim, std::string fd);	// two auxiliary functions to make [dim x 9] arrays of finite difference coeffs and indices;
	HMMPI::Vector2<int> make_ind (size_t dim, std::string fd);		// 'dim' = (active) parameter space dimension, 'fd' = OH1, OH2, OH4, OH8;
																	// applicable in calculation of full gradients and Hessians
public:
	std::string fin_diff;		// OH1, OH2, OH4, OH8 - finite difference type

										// the two arrays below specify the increment "h" for each parameter; depending on "fin_diff", multiples 2h, 3h, 4h may also be used
	std::vector<double> dh;				// for dh_type == CONST, increment for each parameter is dh; for dh_type == LIN, increment for each parameter is dh * |param_value|
	std::vector<std::string> dh_type;	// CONST, LIN

	PhysModGradNum(MPI_Comm c, PhysModel *pm, std::string fd, const std::vector<double> &h, const std::vector<std::string> &h_type) : PhysModMPI(c, pm), of_val(-1), fin_diff(fd), dh(h), dh_type(h_type) {name = "PhysModGradNum";};
	PhysModGradNum(MPI_Comm c, PhysModel *pm, Parser_1 *K, KW_item *kw, const HMMPI::BlockDiagMat *bdc);		// easy constructor
																				// "pm" should have appropriate communicator ("orthogonal" to the communicator 'c' of created model)
																				// all data are taken from keywords of "K"; "kw" is used only to handle prerequisites
	virtual ~PhysModGradNum();
	virtual double ObjFunc_ACT(const std::vector<double> &params);						// the four ObjFuncXXX_ACT functions Bcast the results to "comm"
	virtual std::vector<double> ObjFuncGrad_ACT(const std::vector<double> &params);		// calculates finite difference gradient; should be called on ALL RANKS with same "params"; exceptions are sync
																						// fills (and broadcasts to "comm") "data_sens_act"
	virtual HMMPI::Mat ObjFuncFisher_ACT(const std::vector<double> &params);	// calculates fin. diff. FI for the active params; TODO currently there is MPI mess with 'data_sens_loc'
	virtual double ObjFuncGradDir_ACT(const std::vector<double> &params, const std::vector<double> &dir);	// gradient along direction using finite differences; should be called on ALL RANKS with same "params"; exceptions are sync
																						// for OH1, this gradient calculation should be performed with the same "params" as the last call of this->ObjFunc_ACT()
	virtual HMMPI::Mat ObjFuncHess_ACT(const std::vector<double> &params);				// calculates finite difference Hessian; should be called on ALL RANKS with same "params"; exceptions are sync
};
//---------------------------------------------------------------------------
// ***** This model turned out not to solve the problem as intended)) But let it stay here. *****
// Lagrangian function, taking "PM" as the main function, and constraint |x - x0| = Hk with Lagrange multiplier 'lambda' (goes as the last parameter in the list of parameters)
// Member-functions should be called on all ranks; results should be accessed on comm-RANKS-0
// comm = PM->comm is taken;
// 'modelled_data', 'data_sens' are not filled at the moment;
//---------------------------------------------------------------------------
class PM_LagrangianSpher : public PhysModel
{
protected:
	PhysModel *PM;						// core PhysModel

public:
	std::vector<double> x0;				// center of the sphere
	double Hk;							// radius of the sphere

	PM_LagrangianSpher(PhysModel *pm, double lam);	// comm, init, act_ind, tot_ind, con are taken from PM (and one extra parameter is added); initial value for 'lambda' (last parameter) is taken equal 'lam'
	virtual ~PM_LagrangianSpher();		// deletes 'con' (which is a separate copy)

	virtual double obj_func_work(const std::vector<double> &params);						// objective function value
	virtual std::vector<double> obj_func_grad_work(const std::vector<double> &params);		// gradient of objective function
	virtual HMMPI::Mat obj_func_hess_work(const std::vector<double> &params);				// Hessian of objective function

	bool FindIntersect(const std::vector<double> &x0, const std::vector<double> &x1, std::vector<double> &xint, double &alpha, int &i) const;		// these two functions should not be called at the moment
	bool FindIntersect_ACT(const std::vector<double> &x0, const std::vector<double> &x1, std::vector<double> &xint, double &alpha, int &i) const;	// (need to check how "inf" is treated)
	virtual int ParamsDim() const noexcept;											// dimension of parameters space
	virtual size_t ModelledDataSize() const;				// returns the supposed size of "modelled_data" (same on all ranks, no communication), independently of "modelled_data";
	virtual std::string ObjFuncMsg() const;					// a message which can be printed after ObjFunc calculation
};
//---------------------------------------------------------------------------
// This model calls the underlying model 'PM' on sphere |x - x0| = Hk using spherical coordinates, sphere dimension is PM->actdim - 1.
// First spherical coordinates p0,.. pk-1 range on [0, pi], the last coordinate pk ranges on [0, 2*pi]; more narrow bounds may be imposed in case the sphere center is lying on the PM bounds
// Member-functions should be called on all ranks; results should be accessed on comm-RANKS-0
// comm = PM->comm is taken;
// 'modelled_data' is taken from "PM"; 'data_sens' is filled based on "PM"
//---------------------------------------------------------------------------
class PM_Spherical : public PhysModel
{
protected:
	PhysModel *PM;						// core PhysModel
	HMMPI::SpherCoord Sc;				// stores sphere center and radius
	const double delta;					// gap at poles: the first spherical coordinates will range in [delta, pi-delta]

public:
	PM_Spherical(PhysModel *pm, const HMMPI::BlockDiagMat *bdc, double R, const std::vector<double> &c, double d);	// comm = PM->comm, 'con' is built from PM->con (should be deleted in the end)
															// 'init' is a transform of active(PM->init) into spherical coordinates (with adjustment to min/max); 'act_ind' = 'tot_ind' (but only active params of PM are converted to spherical coordinates)
															// 'R' is sphere radius, 'c' is sphere center (both are in internal representation); c.size() should equal the actdim of PM; d is delta
	virtual ~PM_Spherical();			// deletes 'con'
	virtual double obj_func_work(const std::vector<double> &params);						// objective function value; NOTE 'params' is in spherical coordinates
	virtual std::vector<double> obj_func_grad_work(const std::vector<double> &params);		// gradient of objective function
	virtual HMMPI::Mat obj_func_hess_work_1(const std::vector<double> &params);				// Hessian of objective function, version before Jun 2022
	virtual HMMPI::Mat obj_func_hess_work(const std::vector<double> &params);				// Hessian of objective function, faster version accounting for symmetry
	virtual HMMPI::Mat ObjFuncFisher_mix_1(const std::vector<double> &params);		// mix between FI and Hess, version before Jul 2022
	virtual HMMPI::Mat ObjFuncFisher_mix(const std::vector<double> &params);		// mix between FI and Hess, faster version accounting for symmetry

	virtual int ParamsDim() const noexcept;											// dimension of parameters space (spherical coordinates)
	virtual size_t ModelledDataSize() const;				// returns the supposed size of "modelled_data" (same on all ranks, no communication), independently of "modelled_data";
	virtual std::string ObjFuncMsg() const;					// a message which can be printed after ObjFunc calculation
	const HMMPI::SpherCoord &SC(){return Sc;};
};
//---------------------------------------------------------------------------
// This model fully reproduces the underlying model 'PM'.
// The only difference is that 'con' is the cube |x - x0|_inf = Hk (over active coordinates).
// Member-functions should be called on all ranks; results should be accessed on comm-RANKS-0.
// comm = PM->comm is taken; 'modelled_data' and 'data_sens' are taken from "PM".
//---------------------------------------------------------------------------
class PM_CubeBounds : public PhysModel
{
protected:
	PhysModel *PM;						// core PhysModel
	const double R;						// cube radius
	const std::vector<double> c;		// cube center (full dim)

public:
	PM_CubeBounds(PhysModel *pm, const HMMPI::BlockDiagMat *bdc, double R0, const std::vector<double> &c0);		// comm = PM->comm, 'con' is built from PM->con (should be deleted in the end)
	virtual ~PM_CubeBounds();			// deletes 'con'
	virtual double obj_func_work(const std::vector<double> &params);						// objective function value
	virtual std::vector<double> obj_func_grad_work(const std::vector<double> &params);		// gradient of objective function
	virtual HMMPI::Mat obj_func_hess_work(const std::vector<double> &params);				// Hessian of objective function
	virtual int ParamsDim() const noexcept;											// dimension of parameters space
	virtual size_t ModelledDataSize() const;				// returns the supposed size of "modelled_data" (same on all ranks, no communication), independently of "modelled_data";
	virtual std::string ObjFuncMsg() const;					// a message which can be printed after ObjFunc calculation
};
//---------------------------------------------------------------------------
// This model gives expression for the full Hamiltonian (full energy)
// H(x, p) = 0.5*PM(x) + 0.5*log(det(G(x))) + 0.5*p'*inv(G(x))*p,
// where G(x) - mass matrix (Fisher Information), p - momentum vector (set separately), likelihood = exp(-0.5*PM(x))
// Constraints are inherited from PM. The usual parameters vector is 'x', vector 'p' is set separately.
// This model only works in terms of ACTIVE parameters!
// Member-functions should be called on all ranks; results should be accessed on comm-RANKS-0
// comm = PM->comm is taken; 'modelled_data', 'data_sens' are not filled
// _NOTE_ whenever PM is changed (e.g. proxy gets trained), Caches should be Reset!
//---------------------------------------------------------------------------
class PM_FullHamiltonian : public PhysModel
{
private:																		// calculation of Mass Matrix (Fisher Information + nu*I), its inverse and derivatives, only ACTIVE parameters are involved
	HMMPI::Mat calc_FI_ACT(const std::vector<double> &x) const;					// results - on rank-0
	HMMPI::Mat calc_invFI_ACT(const std::vector<double> &x) const;				// results - on rank-0; uses Cholesky decomposition to invert matrix
	HMMPI::Mat calc_invU_ACT(const std::vector<double> &x) const;				// inv(U), where U'*U = G; results - on rank-0
	HMMPI::Mat calc_dxi_FI_ACT(const std::vector<double> &x, int acti) const;	// results - on rank = Ranks[acti]; 'acti' - index of active parameters
	std::vector<double> calc_dx_H1_ACT(const std::vector<double> &x) const;		// gradient of the full Hamiltonian w.r.t. coordinates 'x' (only first two terms, which do not depend on momentum); results - on rank-0
	std::vector<double> calc_dx_H1_ACT_beta(const std::vector<double> &x) const;		// similar to calc_dx_H1_ACT, but has different scalar coeffs; to be used in MMALA; results - on rank-0
	HMMPI::Mat calc_grad_aux_ACT(const std::pair<std::vector<double>, std::vector<double>> &x_p) const;			// returns a comm-sync matrix with i-th row = (G^(-1) * dG/dx_i * G^(-1) * pact)^t, this is used in the subsequent Jacobian calculations
																				// NOTE x_p = {x, p}, but "p" is not used; before calling, MAKE SURE pact = p is set
	HMMPI::Mat calc_grad_aux_ACT_alpha(const std::vector<double> &x) const;		// similar to calc_grad_aux_ACT; calculates sum_j {G^(-1) * dG/dx_j * G^(-1)}_ij to be used in MMALA; result - on comm-rank-0
	std::vector<double> calc_grad_momentum_ACT(const std::pair<std::vector<double>, std::vector<double>> &x_p) const;	// dH/dp (comm-rank-0); x_p = {x, p}, but "p" is not used; before calling, MAKE SURE pact = p is set
	HMMPI::Mat calc_mu(const std::pair<std::vector<double>, double> &x_eps) const;			// mu for proposal in MMALA; x_eps = {x, eps}; result - sync on comm
	HMMPI::Mat calc_simpl_mu(const std::pair<std::vector<double>, double> &x_eps) const;	// mu for proposal in simplified MMALA; x_eps = {x, eps}; result - sync on comm
	HMMPI::Mat calc_mu_2(const std::pair<std::vector<double>, double> &x_eps) const;		// mu for proposal in MMALA-2; x_eps = {x, eps}; result - sync on comm
	HMMPI::Mat calc_mu_Ifwd(const std::pair<std::vector<double>, std::pair<double, double>> &x_eps_alpha) const;	// mu for forward proposal in I_MALA, G=I; x_eps_alpha = {{theta, p}, {eps, alpha}}; for backward proposal - change sign of 'alpha'; result - sync on comm

								// the following three arrays are for handling the distribution of dxi_G, i = [0, ACTDIM), across ranks
	std::vector<int> nums;		// nums[r] - number of active parameters related to rank 'r' (exists only on rank-0)
	std::vector<int> starts;	// starts[r], r = [0, size+1), is the starting index of active parameters associated with rank 'r', it is from [0, ACTDIM] (exists only on rank-0)
	std::vector<int> Ranks;		// Ranks[i], i = [0, ACTDIM), is the rank where active parameter 'i' belongs (exists on all ranks)
	int locstart, locend;		// for the current rank, the active parameters range is [locstart, locend)
	double MM_shift = 0;		// mass matrix = FI + MM_shift*I
protected:
	PhysModel *PM;				// PhysModel for likelihood

public:
	HMMPI::Mat pact;			// momentum (should be sync on comm), ACTDIM vector; used in ObjFunc_ACT, ObjFuncGrad_ACT
																								// NOTE: Cache::Get should be called on ALL RANKS; Cache::Get results should be collected from comm-ranks-0 (or rank = Ranks[i])
	HMMPI::Cache<PM_FullHamiltonian, std::vector<double>, HMMPI::Mat> G;						// mass matrix
	HMMPI::Cache<PM_FullHamiltonian, std::vector<double>, HMMPI::Mat> invG;						// inverse mass matrix
	HMMPI::Cache<PM_FullHamiltonian, std::vector<double>, HMMPI::Mat> invU;						// inverse of U, where U'*U = G (Cholesky decomposition); on comm-rank-0
	std::vector<HMMPI::Cache<PM_FullHamiltonian, std::vector<double>, HMMPI::Mat>> dxi_G;		// dG/dx_i, NOTE: these matrices are stored on different ranks, namely on rank = Ranks[i] for active parameter 'i'
	HMMPI::Cache<PM_FullHamiltonian, std::vector<double>, std::vector<double>> dx_H1;			// gradient of the first two terms in Hamiltonian w.r.t 'x'; results - on rank-0
	HMMPI::Cache<PM_FullHamiltonian, std::vector<double>, std::vector<double>> dx_H1_beta;		// -> calc_dx_H1_ACT_beta for MMALA; results - on rank-0
	HMMPI::Cache<PM_FullHamiltonian, std::pair<std::vector<double>, std::vector<double>>, HMMPI::Mat> Gaux_grad;		// see calc_grad_aux_ACT(); NOTE: before calling Get(), set pact = p; results are valid on all comm-ranks
	HMMPI::Cache<PM_FullHamiltonian, std::vector<double>, HMMPI::Mat> Gaux_grad_alpha;			// (for MMALA), see calc_grad_aux_ACT_alpha; result - on comm-rank-0
	HMMPI::Cache<PM_FullHamiltonian, std::pair<std::vector<double>, std::vector<double>>, std::vector<double>> dHdp;	// dH/dp (comm-rank-0); before calling Get(), MAKE SURE pact = p is set
	HMMPI::Cache<PM_FullHamiltonian, std::pair<std::vector<double>, double>, HMMPI::Mat> mu_MMALA;			// mu for proposal in MMALA; x_eps = {x, eps}; result - sync on comm
	HMMPI::Cache<PM_FullHamiltonian, std::pair<std::vector<double>, double>, HMMPI::Mat> mu_simplMMALA;		// mu for proposal in simplified MMALA; x_eps = {x, eps}; result - sync on comm
	HMMPI::Cache<PM_FullHamiltonian, std::pair<std::vector<double>, double>, HMMPI::Mat> mu_MMALA_2;		// mu for proposal in MMALA-2; x_eps = {x, eps}; result - sync on comm
	HMMPI::Cache<PM_FullHamiltonian, std::pair<std::vector<double>, std::pair<double, double>>, HMMPI::Mat> mu_Ifwd;		// mu for forward proposal in I_MALA, G=I; x_eps_alpha = {{theta, p}, {eps, alpha}}; for backward proposal - change sign of 'alpha'; result - sync on comm

	PM_FullHamiltonian(PhysModel *pm, double mm_nu);	// comm = PM->comm, 'con' = copy of PM->con (should be deleted in the end)
	PM_FullHamiltonian(const PM_FullHamiltonian &H) = delete;
	virtual ~PM_FullHamiltonian();		// deletes 'con'
	const PM_FullHamiltonian &operator=(const PM_FullHamiltonian &H);

	virtual double ObjFunc_ACT(const std::vector<double> &params);					// when calculating ObjFunc_ACT or ObjFuncGrad_ACT, make sure 'pact' is set correctly!
	virtual std::vector<double> ObjFuncGrad_ACT(const std::vector<double> &params);
	double MMALA_logQ_ACT(const HMMPI::Mat &x, const HMMPI::Mat &xnew, double eps, const HMMPI::Cache<PM_FullHamiltonian, std::pair<std::vector<double>, double>, HMMPI::Mat> &mu_mmala);
																					// ln {q(xnew | x, eps)}, where q(.) is the MMALA (simplMMALA, MMALA-2) proposal Gaussian pdf; result - sync on comm
																					// mu_mmala can be mu_MMALA, mu_simplMMALA, mu_MMALA_2 from the same object that called this function

	HMMPI::Mat calc_mu_alt(std::vector<double> x, double eps) const;				// alternative calculation of MMALA mu (slow, for debugging); only works with MPI size = 1

	virtual int ParamsDim() const noexcept;					// dimension of parameters space
	virtual size_t ModelledDataSize() const;
	virtual std::string ObjFuncMsg() const;					// a message which can be printed after ObjFunc calculation
	void ResetCaches();										// should be called when PM changes (e.g. proxy is trained)
	void nums_starts_to_file(FILE *f);						// outputs 'nums', 'starts' etc to a file (for debug); different ranks should use different files
};
//---------------------------------------------------------------------------
class ValCont;
//---------------------------------------------------------------------------
// Proxy_train_interface - common base class for PM_Proxy and PM_Posterior.
// The methods it declares are to be used in [R]HMC for proxy training
//---------------------------------------------------------------------------
class Proxy_train_interface
{
public:
	virtual ~Proxy_train_interface(){};
	virtual std::vector<size_t> PointsSubset(const std::vector<std::vector<double>> &X0, size_t count) const = 0;
	virtual std::string AddData(std::vector<std::vector<double>> X0, ValCont *VC, int Nfval_pts) = 0;		// returns message: number of design points, eff. rank of matrix
	virtual void SetDumpFlag(int f) = 0;
	virtual int GetDumpFlag() const = 0;
	virtual std::vector<int> Data_ind() const = 0;
	virtual Proxy_train_interface *Copy() const = 0;			// _DELETE_ in the end!
};
//---------------------------------------------------------------------------
// Sim_small_interface - common base class for the simulation models and PM_Posterior
//---------------------------------------------------------------------------
namespace HMMPI
{
	class SimSMRY;
}
class PM_PosteriorDiag;
class Sim_small_interface : public PhysModel
{
protected:
	const PM_PosteriorDiag *outer_post_diag;	// if (*this) is owned by some PM_PosteriorDiag, then 'outer_post_diag' points to the owner

public:
	Sim_small_interface(Parser_1 *K, KW_item *kw, MPI_Comm c) : PhysModel(K, kw, c), outer_post_diag(0) {};
	Sim_small_interface(const Sim_small_interface &PM) : PhysModel(PM), outer_post_diag(0) {};
	Sim_small_interface(MPI_Comm c) : PhysModel(c), outer_post_diag(0) {};

	virtual void set_ignore_small_errors(bool flag) = 0;
	virtual const HMMPI::SimSMRY *get_smry() const = 0;
	virtual bool is_sim() const = 0;			// TRUE if the simulation model is around
	void set_post_diag_owner(const PM_PosteriorDiag *pd){outer_post_diag = pd;};
	virtual void import_stats(const std::vector<double> &mod_data, const std::vector<double> &dx) = 0;		// supposed application - for importing the proxy modelled data and dX
};
//---------------------------------------------------------------------------
// PM_Posterior: this model adds the Gaussian prior term (x - x0)' * C_prior^(-1) * (x - x0) to the underlying likelihood 'PM'
// Member-functions should be called on all ranks; results should be accessed on comm-RANKS-0
// comm = PM->comm is taken;
// 'modelled_data' is formed by complementing: {x | PM->modelled_data}.
// 'data_sens', 'data_sens_loc' are complemented likewise.
// 'BDC' is complemented by block C_prior, handled by rank-0.
// If PM is PROXY, then PM_Posterior can delegate to it the training procedures.
//---------------------------------------------------------------------------
class PM_Posterior : public Sim_small_interface, public Proxy_train_interface
{
private:
	PhysModel *copy_PM;					// filled by COPY ctor, to free memory in the end

protected:
	PhysModel *PM;						// core PhysModel (likelihood)
	const HMMPI::Mat Cpr, dpr;			// these define the prior Gaussian distribution
	HMMPI::Mat invCpr;					// Cpr^(-1)

public:
	PM_Posterior(PhysModel *pm, HMMPI::Mat C, HMMPI::Mat d, const bool is_posterior_diag = false);	// comm = PM->comm, con = PM->con (copy as ParamsInterface), 'is_posterior_diag' = true if PM_PosteriorDiag is to be created further
	PM_Posterior(const PM_Posterior &p);						// copy CTOR will copy PM if it is PM_Proxy*, and will borrow the pointer otherwise
	virtual ~PM_Posterior();			// deletes 'con'
	virtual double obj_func_work(const std::vector<double> &params);
	virtual std::vector<double> obj_func_grad_work(const std::vector<double> &params);
	virtual HMMPI::Mat obj_func_hess_work(const std::vector<double> &params);
	virtual HMMPI::Mat ObjFuncFisher_dxi(const std::vector<double> &params, const int i, int r = 0);
	virtual void WriteLimits(const std::vector<double> &p, std::string fname) const;
	virtual int ParamsDim() const noexcept;
	virtual size_t ModelledDataSize() const;
	virtual std::string ObjFuncMsg() const;
	virtual std::string proc_msg() const;
	virtual std::string get_limits_msg() const {return PM->get_limits_msg();};
	virtual const HMMPI::Mat &cov_prior() const {return Cpr;};
	virtual void correct_of_grad(const std::vector<double> &params, double &y, std::vector<double> &grad) const;	// subtracts the prior component from the 'full posterior' y, grad; needed for training PROXY inside POSTERIOR based on POSTERIOR data

	virtual std::vector<size_t> PointsSubset(const std::vector<std::vector<double>> &X0, size_t count) const;		// these functions delegate to PM which should be PM_Proxy
	virtual std::string AddData(std::vector<std::vector<double>> X0, ValCont *VC, int Nfval_pts);
	virtual void SetDumpFlag(int f);
	virtual int GetDumpFlag() const;
	virtual std::vector<int> Data_ind() const;											// PM should be PM_DataProxy
	virtual Proxy_train_interface *Copy() const {return new PM_Posterior(*this);};		// _DELETE_ in the end!

	virtual bool is_proxy() const {return PM->is_proxy();};
	virtual bool is_dataproxy() const {return PM->is_dataproxy();};
	const PhysModel *get_PM() const {return PM;};

	virtual void set_ignore_small_errors(bool flag);		// delegates to PM, if PM can do this
	virtual const HMMPI::SimSMRY *get_smry() const;			// delegates to PM, if PM can do this
	virtual bool is_sim() const;							// delegates to PM, if PM can do this
	virtual void import_stats(const std::vector<double> &mod_data, const std::vector<double> &dx);		// delegates to PM, if PM can do this
};
//---------------------------------------------------------------------------
// Like PM_Posterior, this model adds the Gaussian prior term (x - x0)' * C^(-1) * (x - x0) to the underlying likelihood 'PM'
// *** Here, the prior covariance C is DIAGONAL; for components  "i" where C_ii = 0, weak prior is assumed
// Member-functions should be called on all ranks; results should be accessed on comm-RANKS-0
// comm = PM->comm is taken;
// 'modelled_data' is formed by complementing: {x | PM->modelled_data}.
// 'data_sens', 'data_sens_loc' are complemented likewise.
// 'BDC' is complemented by block C_prior, handled by rank-0.
// If PM is PROXY, then PM_PosteriorDiag can delegate to it the training procedures.
//---------------------------------------------------------------------------
class PM_PosteriorDiag : public PM_Posterior
{
//	PhysModel *copy_PM;					// filled by copy ctor, to free memory in the end
//	PhysModel *PM;						// core PhysModel (likelihood)
//	const HMMPI::Mat Cpr, dpr;			// these define the prior Gaussian distribution; Cpr is the diagonal (1D vector)
//	HMMPI::Mat invCpr;					// Cpr^(-1), a vector
private:
	const HMMPI::Mat c_pr;		// Full matrix corresponding to Cpr

public:
	std::vector<double> prior_contrib;	// fulldim vector filled by ObjFunc(), contains contribution to _prior_ of each parameter

	PM_PosteriorDiag(PhysModel *pm, const HMMPI::Mat &C_diag, HMMPI::Mat d);		// comm = PM->comm, con = PM->con (copy as ParamsInterface)
	PM_PosteriorDiag(const PM_PosteriorDiag &p);			// copy CTOR will copy PM if it is PM_Proxy*, and will borrow the pointer otherwise
	virtual ~PM_PosteriorDiag();
	virtual double obj_func_work(const std::vector<double> &params);
	virtual std::vector<double> obj_func_grad_work(const std::vector<double> &params);
	virtual HMMPI::Mat obj_func_hess_work(const std::vector<double> &params);
	virtual HMMPI::Mat ObjFuncFisher_dxi(const std::vector<double> &params, const int i, int r = 0);		//  not implemented
	virtual std::string proc_msg() const;
	virtual const HMMPI::Mat &cov_prior() const {return c_pr;};
	virtual void correct_of_grad(const std::vector<double> &params, double &y, std::vector<double> &grad) const;	// subtracts the prior component from the 'full posterior' y, grad; needed for training PROXY inside POSTERIOR based on POSTERIOR data
	virtual Proxy_train_interface *Copy() const {return new PM_PosteriorDiag(*this);};						// _DELETE_ in the end!
};
//---------------------------------------------------------------------------
// PM_Proxy - class for simple kriging proxy
// "comm" is not much used; it is used extensively in DataProxy
// PM->comm should be MPI_COMM_SELF (this is sufficient from the safety point of view) -- otherwise some functions involving "PM" should be revised
// USER: train proxy by calling Train(),
// OR (longer way), by filling ValCont container and calling AddData()
// NOTE: vectors of design points (X0 etc) should be full-dimension vectors (not active-only)
//---------------------------------------------------------------------------
class KrigStart;
class KrigEnd;
class KrigCorr;
class KrigSigma;
class _proxy_params;
//---------------------------------------------------------------------------
class PM_Proxy : public PhysModel, public Proxy_train_interface
{
private:												// these items are used by write_proxy_vals_begin/end()
	std::vector<double> of_before0;						// will have size of X0; selected 'inds' are not known yet
	std::vector<std::vector<double>> data_before0;
	std::vector<double> of_before;						// these store proxy values before and after updates (if WRITE_PROXY_VALS is #defined) -- used mostly for debug and analysis
	std::vector<double> of_after;						// the output of these arrays occurs to file "dump_vals"
	std::vector<std::vector<double>> data_before;
	std::vector<std::vector<double>> data_after;
	bool first_call = true;

	void write_proxy_vals_begin(const std::vector<std::vector<double>> &X0);		// output o.f. & data values of proxy before and after update; see "dump_flag"
	void write_proxy_vals_end(const std::vector<std::vector<double>> &X0, const std::vector<size_t> &inds);

protected:
	const char dump_X[100] = "proxy_dump_X_%d_pr%d.txt";
	const char dump_vals[100] = "proxy_dump_vals_%d.txt";			// file name for the o.f. values and modelled data before and after proxy update; doesn't apply for the first ensemble of points (when the proxy is just created)
																	// FORMAT: o.f. before | o.f. after | data before | data after
	PhysModel *PM;						// this PhysModel is used for checking limits, ExportIAC, and Train

	int dump_flag;						// if this flag != -1, then proxy will output (in AddData) different quantities to files "proxy_dump_XXX_$(dump_flag).txt" -- for debug purposes
	int train_from_dump;				// if this flag != -1, then Train will always take X and y from files "proxy_dump_X_$(train_from_dump).txt", "proxy_dump_Ity_$(train_from_dump)_rnk0_pr0.txt" -- for debug purposes
										// DataProxy[2]: if proxy was created with MPISIZE > 1, gluing is required to collect several dumps into single file, e.g.
										// for i in $(seq 1 19) ; do paste proxy_dump_Ity_22_rnk0_pr0.txt proxy_dump_Ity_22_rnk${i}_pr0.txt > proxy_dump_Ity1.txt; mv proxy_dump_Ity1.txt proxy_dump_Ity_22_rnk0_pr0.txt ; done
	// the main components for proxy calculation
	std::vector<KrigStart> starts;			// in PM_Proxy, both arrays are of size 1
	std::vector<KrigEnd> ends;

	virtual void AddPoints(const std::vector<std::vector<double>> &X0, const std::vector<std::vector<double>> &X1);		// adds 'X0' to starts[*].X_0, adds 'X1' to starts[*].X_1, updates distance matrices;
	virtual void RecalcVals();				// (after adding values) makes CinvZ calculation; works differently for different proxy types
	std::vector<std::vector<double>> XFromFile(std::string fname) const;		// returns 'pop[len][full_dim]' (nontrivial only on comm-RANKS-0) read from file "fname"; this function is mostly for debugging
	void copy_starts_ends_link(const PM_Proxy &p);		// takes the "starts" - "ends" link from "p" (internally "p" should be of the same PROXY type as "this"); call it in COPY CTORs before set_refs()
	void set_refs();						// sets the remaining links
	void reset_kc_ks_cache() const;			// resets cache for kc, ks objects in starts, ends

public:
	bool do_optimize_krig = false;		// defines whether OptimizeKrig should be run inside AddData
	std::string mat_eff_rank;			// info on effective rank of the full kriging matrix, if available (filled by RecalcVals)

	PM_Proxy(MPI_Comm c, PhysModel *pm, const HMMPI::BlockDiagMat *bdc, Parser_1 *K, KW_item *kw, _proxy_params *config);		// easy CTOR
	PM_Proxy(const PM_Proxy &p);								// copy CTOR; _NOTE_ PM is not copied, so make sure it is not deleted while the newly created proxy is still in use!
	const PM_Proxy &operator=(const PM_Proxy &b) = delete;
	virtual PM_Proxy *Copy() const {return new PM_Proxy(*this);};								// the pointer should be _DELETED_ in the end
	PhysModel *get_PM() const {return PM;};						// although this function is "const", it's INSECURE, because PM may be changed outside
	KrigCorr *get_KrigCorr();				// call this function only for simple proxy
	KrigSigma *get_KrigSigma();				// call this function only for simple proxy
	virtual void SetDumpFlag(int f);							// sets dump_flag for "this", starts and ends
	void SetTrainFromDump(int t){train_from_dump = t;};
	virtual int GetDumpFlag() const {return dump_flag;};
	std::string init_msg() const;								// message on how KrigCorr was initialised (LIMITSKRIG/PROXY_CONFIG)
	virtual std::string proc_msg() const;						// sync message showing: (1) effective rank of kriging matrix; (2) pscale, grad_inds for "starts[0]"
	virtual bool is_proxy() const {return true;};
	virtual bool CheckLimits(const std::vector<double> &params) const;		// if PM != 0, then PM->CheckLimits; no Bcast
	virtual int ParamsDim() const noexcept {return PM->ParamsDim();};
	virtual std::vector<size_t> PointsSubset(const std::vector<std::vector<double>> &X0, size_t count) const;	// selects 'count' points from X0 - these points are then to be added to starts[*].X_0;
																												// selection is based on starts[*].X_0 + X0
																												// the SYNC vector of selected indices (for 'X0') is returned
																												// selection works independently of possible differences in "starts"
	virtual std::string AddData(std::vector<std::vector<double>> X0, ValCont *VC, int Nfval_pts);	// adds new data (points X0 and values/gradients VC) and trains proxy; returns the message about the total number of points
																							// Nfval_pts shows how many points with func. vals should be selected (however, all points with grads are taken)
																							// if proxy (X_0) is empty, all points with func. vals are taken
																							// X0 can be defined on comm-RANKS-0 only
	virtual double obj_func_work(const std::vector<double> &params);
	virtual std::vector<double> obj_func_grad_work(const std::vector<double> &params);
	virtual HMMPI::Mat obj_func_hess_work(const std::vector<double> &params);
	virtual std::vector<double> ObjFuncHess_l(const std::vector<double> &params, int l);			// calculates l-th column of Hessian
	virtual std::vector<int> Data_ind() const {throw HMMPI::Exception("Illegal call to PM_Proxy::Data_ind");};
	std::string Train(std::vector<std::vector<double>> pop, std::vector<size_t> grad_ind, int Nfval_pts);	// trains the proxy (i.e. adds to the existing proxy state) based on design points pop[len][full_dim], and PM->ObjFuncMPI_ACT("pop") calculated in parallel via MPI
														// 'pop' is only referenced on comm-RANKS-0; underlying PM should have comm == "MPI_COMM_SELF" (i.e. all PM->comm-RANKS == 0)
														// If train_from_dump != -1, 'pop' is not used, X & y are taken from appropriate files (reading the same number of lines as in the original "pop"); currently gradients are not read from the files
														// "grad_ind" (comm-RANKS-0) are indices in [0, len) for points where gradients will be estimated and added to the proxy; these training points are always taken from "pop" ("pop" may be from the file)
														// Nfval_pts is the same as in AddData()
														// the returned message is [whether proxy was trained from dump] plus "Number of design points on starts..." from AddData()
};
//---------------------------------------------------------------------------
// Data proxy combines multiple "simple proxies" - for each data point of "ModelledData".
// This class is MPI-distributed ('ends' is distributed) according to "comm".
// By default, array 'starts' is of size 1, so all sub-proxies share the same covariance matrix.
// Derived classes may treat 'starts' differently
//---------------------------------------------------------------------------
class PM_DataProxy : public PM_Proxy
{
private:
	HMMPI::Mat Gr_loc;							// stores local gradients [re]used in ObjFuncHess_l
	std::vector<double> resid_loc;				// stores local residual = 2 * C^(-1) * (d_m - d_o) [re]used in ObjFuncHess_l
	size_t data_size;							// size of modelled data, sync on all MPI_COMM_WORLD ranks

	void process_data_size(const std::vector<double> &d);	// size of "d" is taken at comm-RANKS-0, and Bcasted over MPI_COMM_WORLD, filling "data_size"
															// call this function on all ranks of MPI_COMM_WORLD
protected:
	const char dump_CinvZ[100] = "proxy_dump_CinvZ_%d_rnk%d_pr%d.txt";
	const char dump_Ity[100] = "proxy_dump_Ity_%d_rnk%d_pr%d.txt";

	HMMPI::Mat d0;								// vector of observed data (MPI-distributed with "comm"); possibly, perturbed for RML
	HMMPI::Mat d0_orig;							// vector of observed data (MPI-distributed with "comm"); original, not perturbed
	HMMPI::RandNormal *RndN;					// this random number generator may be left = 0 (unless RML is required)

	virtual void RecalcVals();					// works with starts[0] and 'ends'
	virtual HMMPI::Mat ObjFuncSens_dxi(const std::vector<double> &params, const int i);	// dSens/dxi, each rank returns only part of the matrix corresponding to the LOCAL data points; size of the result is the same as for Sens matrix
	void data_ind_count_displ(std::vector<int> &counts, std::vector<int> &displs, int mult = 1) const;	// fills arrays recvcounts and displs needed for MPI_Gatherv; uses Data_ind; sync between ranks (comm)
																										// 'mult' multiplies the number of data sent (e.g. 'mult' may be = param_count)
public:
	PM_DataProxy(PhysModel *pm, Parser_1 *K, KW_item *kw, _proxy_params *config, const HMMPI::BlockDiagMat *bdc, const std::vector<double> &d);		// easy CTOR; to be called on all ranks of MPI_COMM_WORLD
														// 'bdc' (block diagonal covariance) should be created in advance, it will provide "comm" for PM_DataProxy
														// 'd' - observed data (only supply it on comm-RANKS-0)
	PM_DataProxy(const PM_DataProxy& p);		// copy CTOR
	const PM_DataProxy &operator=(const PM_DataProxy &b) = delete;
	virtual PM_Proxy *Copy() const {return new PM_DataProxy(*this);};						// the pointer should be _DELETED_ in the end
	virtual bool is_dataproxy() const {return true;};
	virtual double obj_func_work(const std::vector<double> &params);						// these functions gather the results (incl. modelled_data, data_sens) from ALL RANKS to comm-RANKS-0
	virtual std::vector<double> obj_func_grad_work(const std::vector<double> &params);
	virtual std::vector<double> ObjFuncHess_l(const std::vector<double> &params, int l);	// call this in increasing order of "l"
	virtual HMMPI::Mat ObjFuncFisher_dxi(const std::vector<double> &params, const int i, int r = 0);	// derivative of FI matrix w.r.t. x_i, call this in increasing order of "i", starting from 0; "r" is the rank where result should be accessed
	virtual size_t ModelledDataSize() const {return data_size;};
	virtual void PerturbData();												// RML data perturbation; #ifdef WRITE_PET_DATA && WORLD-RANK-0 == comm-RANK-0, perturbed data is written to "pet_DataProxy.txt"
	virtual std::vector<int> Data_ind() const {return BDC->Data_ind();};	// <sync(comm)> vector of beginnings and ends of all "data points" on each rank; obtained from the covariance matrix
	virtual std::string proc_msg() const;		// sync message showing (1) the number of 'ends' (data points) on each rank; (2) effective rank of kriging matrix; (3) pscale, grad_inds for "starts[0]"
};
//---------------------------------------------------------------------------
// For this data proxy 'starts' and 'ends' are of the same size, all sub-proxies have their own covariance matrices.
// 'starts' and 'ends are MPI-distributed according to "comm".
// This data proxy is different from PM_DataProxy only in the way the sub-proxies are calculated.
//---------------------------------------------------------------------------
class PM_DataProxy2 : public PM_DataProxy
{
protected:
	virtual void RecalcVals();							// works with 'starts' and 'ends'; NOTE currently "mat_eff_rank" is not transferred from other processes

public:
	PM_DataProxy2(PhysModel *pm, Parser_1 *K, KW_item *kw, _proxy_params *config, const HMMPI::BlockDiagMat *bdc, const std::vector<double> &d);
	const PM_DataProxy2 &operator=(const PM_DataProxy2 &b) = delete;
	virtual PM_Proxy *Copy() const {return new PM_DataProxy2(*this);};						// the pointer should be _DELETED_ in the end
	virtual std::string proc_msg() const;				// sync message showing (1) the number of 'ends' (data points) on each rank; (2) effective rank of kriging matrix[i]; (3) pscale for starts[i]
};
//---------------------------------------------------------------------------
// One more type of DataProxy,
// N_blocks <= |starts| <= |ends|, each starts[i] uses its own X, D, pscale; "starts" are not sync (but |starts| is sync, and non-empty starts[i] are sync)
// *block* refers to the data_points with the same number of design models, e.g.
//			data_points
// Np		|||
// des.pts. ||| |||
// 			||| ||| ||
// blocks:	b0  b1  b2
// Each block may contain more than one starts[i], if they have different 'pscale'
//---------------------------------------------------------------------------
class PM_SimProxy : public PM_DataProxy
{
private:
	std::vector<int> block_starts;						// <sync> this array maps [0, N_blocks+1) -> [0, Np] showing where in design points each block starts
	std::vector<int> start_to_block;					// <sync> this array maps [0, N_starts) -> [0, N_blocks), giving the block number for each start[i]. Filled in CTOR
	std::vector<std::pair<int, int>> dp_block_color;	// <sync> array of <block_i, color_i> which defines 'starts'; stored here for reporting. Filled in CTOR
	std::vector<int> start_Nends;						// <sync> array of length [0, N_starts) showing the number of ends[] in each starts[i] (summed over MPI processes). Filled in CTOR; used only for reporting

protected:
	virtual void AddPoints(const std::vector<std::vector<double>> &X0, const std::vector<std::vector<double>> &X1);		// adds different subsets of 'X0' to "starts"; currently 'X1' is not added;
																														// 'X0' should be sync on 'comm'
	virtual void RecalcVals();							// works with 'starts' and 'ends'
public:
	PM_SimProxy(PhysModel *pm, Parser_1 *K, KW_item *kw, _proxy_params *config, const HMMPI::BlockDiagMat *bdc, const std::vector<double> &d, const std::vector<int> &b_starts, const std::vector<int> &dp_block);
																						// <sync> "b_starts[0..N_blocks+1)" - array of indices in [0, Np] showing where each block starts
																						// <sync> "dp_block[0..smry_len)" - index of the first block where the given data point exists (see SimProxyFile)
	PM_SimProxy(const PM_SimProxy &p);					// copy CTOR
	const PM_SimProxy &operator=(const PM_SimProxy &b) = delete;
	virtual PM_Proxy *Copy() const {return new PM_SimProxy(*this);};					// the pointer should be _DELETED_ in the end
	virtual std::string proc_msg() const;				// sync message showing (1) the number of 'ends' (data points) on each process; (2) effective rank of kriging matrix for all 'starts'; (3) pscale for all 'starts'

	size_t start_to_block_size() const {return start_to_block.size();};
	int start_to_block_i(int i) const {return start_to_block[i];};
};
//---------------------------------------------------------------------------
// _NOTE_ currently KrigCorr and KrigSigma do not deal with gradients at design points
//
// KrigCorr - class for making and optimizing correlation/covariance matrix for kriging.
// Derivatives of correlation matrix are w.r.t. 3 parameters "xi" [Nugget, R (corr. radius), nu (for Matern)].
// For Gaussian correlation "nu" is dummy, and the corresponding derivatives are 0.
// Objects of this class are to be manipulated by KrigStart.
//---------------------------------------------------------------------------
//#define KRIG_CORR_DET_CHOL  			// if defined, det(R) and R^(-1) are calculated via Cholesky decomposition (DPOTRF), but for badly conditioned R this procedure may fail
										// if not defined, det(R) and R^(-1) are calculated via DSYTRI decomposition
class KrigCorr : public PhysModel
{
private:
	mutable std::vector<double> par_cache;		// cached params, they are set in CalculateR, CalculateDerivatives, CalculateDerivatives2
	mutable int is_valid;						// if is_valid & 2, 4, 8, then the cached R, d_R, d2_R can be reused
	mutable HMMPI::Mat R;						// correlation matrix
	mutable std::vector<HMMPI::Mat> d_R;		// dR/dxi - first derivatives [3 x 1]
	mutable HMMPI::Vector2<HMMPI::Mat> d2_R;	// d2R/dxi*dxj - second derivatives [3 x 3], upper-triangular Vector2

protected:
	Parser_1 *K0;				// used for messaging (warnings)
	const HMMPI::Mat *D;		// distance matrix
	HMMPI::Func1D_corr *func;		// 1D correlation function; its parameters can be changed
	const HMMPI::Func1D_corr *cfunc;	// same as "func", but used only when "func" happens to be 0 after copying

	const double dh = 1e-4;			// for numerical derivatives
	const HMMPI::OH oh = HMMPI::OH2;

	void CalculateDerivatives(const std::vector<double> &par) const;			// calculates first derivatives of R, only for GAUSS or MATERN correlations
	void CalculateDerivatives2(const std::vector<double> &par) const;			// calculates second derivatives of R, only for GAUSS or MATERN correlations
	double obj_func(const std::vector<double> &params) const;					// these three functions are just the const versions of ObjFunc, ObjFuncGrad, ObjFuncHess
	std::vector<double> obj_func_grad(const std::vector<double> &params) const;
	HMMPI::Mat obj_func_hess(const std::vector<double> &params) const;
public:
	std::string init_msg;		// message saying if PROXY_CONFIG or LIMITSKRIG were used for 'init' values

	friend class KrigSigma;

	KrigCorr();
	KrigCorr(const HMMPI::Func1D_corr *cf);										// cf - corr. func.
	KrigCorr(const HMMPI::Func1D_corr *cf, Parser_1 *K, _proxy_params *config);	// this ctor takes 'init' from: MODEL (if "config" == MODEL),
																			// 				 or from PROXY_CONFIG (if "config" == PROXY_CONFIG, LIMITSKRIG not defined), or from LIMITSKRIG (otherwise)
	KrigCorr(const KrigCorr &kc);											// NOTE: this does not copy "D", which should be set manually
	const KrigCorr &operator=(const KrigCorr &p);							// "D" is not copied
	~KrigCorr(){delete func;};
	const std::vector<double> &get_init() const {return init;};
	void set_init(const std::vector<double> &v) {init = v;};
	void take_refs(const HMMPI::Mat *d) {D = d; is_valid = 0;};
	const HMMPI::Func1D_corr *CalculateR(const std::vector<double> &par) const;		// calculates R, returns the 1D function used; par = {nugget, r, nu}
	const HMMPI::Func1D_corr *GetFuncFromCalculateR() const;							// returns the same Func1D_corr* as was returned by the last CalculateR()
	const HMMPI::Mat &Get_R() const {return R;};
	void reset_cache() const {is_valid = 0;};									// manual cache reset (e.g. in PM_Proxy::AddData)

	virtual double obj_func_work(const std::vector<double> &params);			// det(R)^(1/n)
	virtual std::vector<double> obj_func_grad_work(const std::vector<double> &params);
	virtual HMMPI::Mat obj_func_hess_work(const std::vector<double> &params);
	virtual int ParamsDim() const noexcept {return 3;};
};
//---------------------------------------------------------------------------
// KrigSigma - for calculating MLE sigma^2 (for kriging) and its derivatives
// Objects of this class are to be manipulated by KrigEnd.
//---------------------------------------------------------------------------
//#define OBJ_FUNC_SIGMA_2		// if defined, obj. func. = sigma^2 (test mode)
								// if not, obj. func. = sigma^2 * det(R)^(1/n) (work mode)
class KrigSigma	: public PhysModel
{
private:
	mutable HMMPI::Mat U_cache;			// cached value, U = P - Q
	mutable bool is_valid;
	mutable std::vector<double> par_cache;		// params cache

	const HMMPI::Mat &get_U(const std::vector<double> &params) const;
protected:
	const HMMPI::Mat *F;				// trend matrix
	const std::vector<double> *Ys;		// values at design points
	const KrigCorr *ref;				// this provides R (corr. matr.)

	double sigma2(const std::vector<double> &params) const;						// three functions which calculate MLE sigma^2 and its derivatives
	std::vector<double> sigma2_grad(const std::vector<double> &params) const;
	HMMPI::Mat sigma2_Hess(const std::vector<double> &params) const;
public:
	KrigSigma() : PhysModel(MPI_COMM_SELF), is_valid(false), F(0), Ys(0), ref(0) {name = "KRIGSIGMA";};
	KrigSigma(Parser_1 *K, _proxy_params *config);				// this ctor takes 'init' from: MODEL (if "config" == MODEL),
																// 				 or from PROXY_CONFIG (if "config" == PROXY_CONFIG, LIMITSKRIG not defined), or from LIMITSKRIG (otherwise)
	KrigSigma(const KrigSigma &ks) : PhysModel(ks), is_valid(false), F(0), Ys(0), ref(0) {};		// NOTE: this does not copy F, Ys, ref, which should be set manually
	const KrigSigma &operator=(const KrigSigma &p);													// --"--
	void take_refs(const HMMPI::Mat *f, const std::vector<double> *ys, const KrigCorr *kc) {F = f; Ys = ys; ref = kc; is_valid = false;};
	void reset_cache() const {is_valid = false;};				// manual cache reset (e.g. in PM_Proxy::AddData)

	virtual double obj_func_work(const std::vector<double> &params);
	virtual std::vector<double> obj_func_grad_work(const std::vector<double> &params);
	virtual HMMPI::Mat obj_func_hess_work(const std::vector<double> &params);
	virtual int ParamsDim() const noexcept {return 3;};
};
//---------------------------------------------------------------------------
// KrigStart
// KrigStart and KrigEnd - the main working parts of a kriging proxy; KrigEnd may alter KrigStart during kriging optimisation
//---------------------------------------------------------------------------
class KrigStart
{
private:
	const bool smooth_at_nugget = true;			// if 'true', there will be no discontinuity at design points for nugget > 0
	int trend_order;

	const char dump_D[100] = "proxy_dump_DistMat_%d_pr%d.txt";		// file names for debug output
	const char dump_X[100] = "proxy_dump_X_%d_pr%d.txt";
	const char dump_C[100] = "proxy_dump_fullC_%d_rnk%d_pr%d.txt";

	void rescale_vec(std::vector<double> &v) const;					// rescales v using 'pscale'
	void rescale_vecs(std::vector<std::vector<double>> &v) const;	// rescales v[i] using 'pscale'
	void rescaleBACK_vec(std::vector<double> &v) const;
	void rescaleBACK_vecs(std::vector<std::vector<double>> &v) const;
protected:
	HMMPI::Mat D;						// distance matrix for design points with func. values
	HMMPI::Mat DG, DGG;					// additional distance matrices including gradient points
	std::vector<std::vector<double>> X_0, X_1;	// full-dim points for func. values and for gradients; they are gradually accumulated as the proxy is trained
	std::vector<double> pscale;			// [fulldim] scaling of parameters influence; default = 1.0; pscale_i < 1.0 means x_i has less influence, which can be implemented as:
										// (a) increasing correlation radius for x_i by 1/pscale_i, (b) damping i-component of some gradients by pscale_i (TODO)
	std::vector<size_t> grad_inds;		// vector of size <= FULLDIM; gradient components with indices "grad_inds[i]" take part in proxy training;
										// 0 <= grad_inds[i] < FULLDIM; "grad_inds" should be monotonically increasing

	//HMMPI::Mat CM;	// "correlation" (func. vals) part of kriging matrix - is obtained from "kc"
	HMMPI::Mat N;		// "trend" part of kriging matrix (func. vals only)
	HMMPI::Mat Nfull;	// "trend" part, full version (func. vals. + gradients)
	HMMPI::Mat C;		// full kriging matrix
	KrigCorr kc;		// object for creating "correlation" part of kriging matrix (func. vals only), and its derivatives; kc.init stores the current kriging parameters

	double R;					// correlation radius in all directions
	const HMMPI::Func1D_corr *func;	// correlation function
	const HMMPI::Solver *sol;	// solver for inverting kriging matrix C
	std::vector<std::vector<int>> multi_ind;		// multiindices describing the polynomial trend components, e.g. for 2D case, [[], [0], [1], [0,1], [0,0], [1,1]] means trend components are: 1, x, y, xy, x^2, y^2
													// multi_ind[][] range is [0, dim)
	HMMPI::Mat C0, gM, lM;		// common auxiliary things for ObjFuncXXX

	void push_back_data(const std::vector<std::vector<double>> &X0, const std::vector<std::vector<double>> &X1);			// appends whole X0 to X_0, X1 to X_1
public:
	friend class KrigEnd;

	int dump_flag;				// same as PM_Proxy::dump_flag
	int index;					// index within "starts"
	bool is_empty;				// 'true' for empty KrigStart objects (not for calculations, only for padding)

	KrigStart() : trend_order(0), R(0), func(NULL), sol(NULL), dump_flag(-1), index(-1), is_empty(true){};	// creates empty object
	KrigStart(Parser_1 *K, KW_item *kw, _proxy_params *config);			// easy CTOR; all data are taken from keywords of "K"; 1st LINSOLVER is used; "kw" is used only to handle prerequisites; "config" can be PROXY_CONFIG or MODEL
	KrigStart(const KrigStart &p);						// 'index' is not copied
	const KrigStart &operator=(const KrigStart &p);		// 'index' is not copied

	std::vector<size_t> PointsSubsetKS(const std::vector<std::vector<double>> &x0, size_t count, bool &all_taken) const;	// selects 'count' points from 'x0', returning their indices; IndSignificant() is used;
																															// the distance matrix which guides the selection comes from X_0 + x0
																															// if X_0 is empty, full x0 indices are taken (in which case all_taken = true)
	void AddPoints(std::vector<std::vector<double>> X0, std::vector<std::vector<double>> X1);		// adds 'X0' to X_0, adds 'X1' to X_1, updates 'D', 'DG', 'DGG';
	void RecalcPoints();								// (after adding X0, X1) makes appropriate matrix calculations
	void ObjFuncCommon(std::vector<double> params);					// calculates C0
	void ObjFuncGradCommon(std::vector<double> params);				// calculates gM
	void ObjFuncHess_lCommon(std::vector<double> params, int l);	// calculates lM

	static HMMPI::Mat DistMatr(const std::vector<std::vector<double>> &X0, int i1, int i2, int j1, int j2);		// creates (i2-i1)x(j2-j1) distance matrix for distances X0[i1,i2) -- X0[j1,j2)
	static HMMPI::Mat RHS_dist_matr(std::vector<std::vector<double>> &Xarr, const std::vector<double> &params);	// distance "vector" for RHS of kriging system; its elements are distances from Xarr[i] to 'params'; Xarr can be X_0, X_1
	static std::vector<size_t> IndSignificant(const HMMPI::Mat &DM, size_t count, size_t start = 0);	// selects (at most) "count" points which are reasonably separated according to symmetric distance matrix "DM" (sort of SEQUENTIAL DESIGN)
													// selection starts from index "start";
													// each point is selected such that its sum of 1/distance^3 to the points [0, start) and the points already selected is minimum;
													// returns array of indices of these points -- these indices are 'local', so 'start' should be added to apply them to the full "DM"
	void set_refs() {kc.take_refs(&D);};
	KrigCorr *get_kc_ptr(){return &kc;};
	const HMMPI::Mat &get_C() const {return C;};
	const std::vector<std::vector<double>> &get_X_0() const {return X_0;};
	const std::vector<std::vector<double>> &get_X_1() const {return X_1;};
	const std::vector<double> &get_pscale() const {return pscale;};
	void set_pscale(const std::vector<double> &ps) {pscale = ps;};
	const std::vector<size_t> &get_grad_inds() const {return grad_inds;};
	const HMMPI::Solver *get_sol() const {return sol;};
	const std::vector<std::vector<int>> &get_multi_ind() const {return multi_ind;};
	void reset_kc_cache() const {kc.reset_cache();};
	static std::vector<std::vector<int>> make_multi_ind(int dim, int pow);		// make array of all multiindices of degree <= "pow" <= 3 over dimension "dim" -- an auxiliary function
	std::string init_msg() const;
};
//---------------------------------------------------------------------------
// KrigEnd
// _NOTE_ currently KrigCorr and KrigSigma do not deal with gradients at design points, so kriging optimisation does not account for these gradients
//---------------------------------------------------------------------------
class KrigEnd
{
private:
	const int maxit = 20;		// constants used in LM optimization of kriging parameters
	const double epsG = 1e-10;
	const double epsF = 0;
	const double epsX = 0;

	const char dump_CinvZ[100] = "proxy_dump_CinvZ_%d_rnk%d_pr%d.txt";
	const char dump_Ity[100] = "proxy_dump_Ity_%d_rnk%d_pr%d.txt";
	const char dump_C[100] = "proxy_dump_fullC_%d_rnk%d_pr%d.txt";
	const char dump_opt_krig[100] = "Proxy_opt_KRIG_rnk%d.txt";

protected:
	const KrigStart *start;		// this is used to get access to the common data; NOTE: KrigEnd may alter KrigStart during kriging optimisation

	std::vector<double> y;						// func. values at points X_0
	std::vector<std::vector<double>> grad_y;	// func. gradients [fulldim] at points X_1
	HMMPI::Mat CinvZ;	// vector C/(I'*y_full)
	KrigSigma ks;		// object for MLE sigma^2 and its derivatives

public:
	int dump_flag;		// same as PM_Proxy::dump_flag
	int index;			// index within "ends"
	int start_index;	// index of 'start' within "starts"; although start->index is the same, the latter may not always be valid

	KrigEnd() : start(NULL), dump_flag(-1), index(-1), start_index(-1) {};
	KrigEnd(Parser_1 *K, _proxy_params *config) : start(NULL), ks(K, config), dump_flag(-1), index(-1), start_index(-1){};
	KrigEnd(const KrigEnd &p) {*this = p;};			// start, index, start_index are not copied
	const KrigEnd &operator=(const KrigEnd &p);		// start, index, start_index are not copied
	double ObjFuncPrivate();
	std::vector<double> ObjFuncGradPrivate();
	std::vector<double> ObjFuncHess_lPrivate();
	void OptimizeKrig();							// optimizes kriging parameters (via "ks") for ends[i] in DataProxy2 (or stand alone simple proxy), and takes these parameters; should be called after adding vals, RecalcPoints()
	std::string RecalcVals();						// (after adding values) makes CinvZ calculation; returns message on mat_eff_rank

	void set_start(const KrigStart *s, int s_ind){start = s; start_index = s_ind;};
	void set_refs();								// sets some refs for "ks"
	KrigSigma *get_ks_ptr(){return &ks;};
	std::vector<double> get_y() const;									// returns the observed values at design points (values + gradients); the Gaussian Process will be conditioned on them
	void set_CinvZ(HMMPI::Mat z){CinvZ = std::move(z);};
	void reset_ks_cache() const {ks.reset_cache();};
	void push_back_vals(const std::vector<double> &y0);					// appends whole y0 to y
	void push_back_grads(const std::vector<std::vector<double>> &gr0);	// appends whole gr0 to grad_y
};
//---------------------------------------------------------------------------
// CONTAINERS
//---------------------------------------------------------------------------
// ValCont - container with values (and gradients) at design points, can store values/gradients for a single proxy, or multiple proxies (DataProxy)
// ValCont has a variety of constructors
//---------------------------------------------------------------------------
class ValCont : public HMMPI::ManagedObject
{
protected:
	const char dump_Ity[100] = "proxy_dump_Ity_%d_rnk%d_pr%d.txt";

	MPI_Comm comm;
	int len, smry_len;
	double *FIT = 0;			// [len] -- intermediate storage (filled by RunTrainPopulation, or from file); referenced only on comm-RANKS-0
	double **SMRY = 0;			// [len][smry_len]

	void write_FIT_SMRY() const;								// debug output to file; len, smry_len should be correctly set
	void RunTrainPopulation(PhysModel *pm, const std::vector<std::vector<double>> &pop);	// fills len, smry_len, FIT, SMRY based on design points pop[len][full_dim], using ObjFuncMPI_ACT; "pop" is only referenced on comm-RANKS-0
																							// 'pm' should have comm == "MPI_COMM_SELF"
	virtual void FitSmryFromFile(std::string fname, int l) = 0;	// to be used instead of RunTrainPopulation
																// fills len, FIT (for ValContDouble) or SMRY (for ValContVecDouble) from file "fname", reading "l" lines (= number of design points)
																// _NOTE_ "smry_len" should be properly defined before calling this function!
																// in case of SMRY, 'smry_len' numbers are expected in each line; this function is mostly for debugging
public:
	ValCont(MPI_Comm c) : comm(c), len(0), smry_len(0), FIT(NULL), SMRY(NULL) {};
	virtual ~ValCont();
	virtual int vals_count() const = 0;							// <sync> number of design points with func. values (to distinguish from gradients)
	virtual int total_count() const = 0;						// <sync> total number of design points (func. values + gradients)
	virtual void DistrValues(std::vector<KrigEnd> &dep, const std::vector<size_t> &inds) const = 0;	// distributes the stored values/gradients to the given proxies; 'inds' shows which design points (with func. values) should be taken
};
//---------------------------------------------------------------------------
// works for single proxy
class ValContDouble : public ValCont
{
protected:
	std::vector<double> V;						// [len]
	std::vector<std::vector<double>> Grad;		// [len_gr][fulldim]

	virtual void FitSmryFromFile(std::string fname, int l);
public:
	ValContDouble(MPI_Comm c, std::vector<double> vals, std::vector<std::vector<double>> grads) : ValCont(c), V(std::move(vals)), Grad(std::move(grads)){};
	ValContDouble(MPI_Comm c, PhysModel *pm, const std::vector<std::vector<double>> &pop, int train_from_dump);		// Gets func. values by running 'pm' at design points pop[len][full_dim]; or from a file.
														// 'pop' is only referenced on comm-RANKS-0; 'pm' should have comm == "MPI_COMM_SELF" (i.e. all pm->comm-RANKS == 0)
														// If train_from_dump != -1, 'pop' is not used, and pop.size() lines are taken from the appropriate file
														// This CTOR does not fill "Grad"
	virtual int vals_count() const;
	virtual int total_count() const;
	virtual void DistrValues(std::vector<KrigEnd> &dep, const std::vector<size_t> &inds) const;
	void Add(const ValContDouble &b);			// appends data from "b" to 'this'; MPI layout should be the same
};
//---------------------------------------------------------------------------
// works for data proxy
// this class is MPI-DISTRIBIUTED, i.e. each rank stores (or may store) only its own portion of the container's contents
class ValContVecDouble : public ValCont
{
private:
	//double obj_func_for_one_model(int m, const PM_DataProxy *dp, std::vector<double> &resid);		// calculates (d-d0)'*C^(-1)*(d-d0) for model "m", where "d" = Vecs[:][m]
																									// also outputs "resid" = d-d0 (it is distributed across the ranks)
																									// the returned o.f. value is same on all ranks
protected:
	std::vector<std::vector<double>> Vecs;					// MPI-distributed; Vecs[i] - values vector for proxy[i], i.e. indexing is "Vecs[smry_len][len]"
	std::vector<std::vector<std::vector<double>>> Grads;	// MPI-distributed; Grads[smry_len][len_gr][fulldim] - gradients at the corresponding design points		// TODO check when some params are inactive
															// NOTE: first dimension for Vecs and Grads is always consistent!
	virtual void FitSmryFromFile(std::string fname, int l);
	void FillVecs(const std::vector<int> &data_ind, const double* const *smry, int len);					// fills MPI-local 'Vecs' from comm-RANKS-0 'smry', see constructor for comments; 'smry' is referenced on comm-RANKS-0
public:
	ValContVecDouble(MPI_Comm c, std::vector<int> data_ind, const double* const *smry, int len, int smry_len);	// smry[len][smry_len] (from c-RANKS-0) is distributed among ranks in "c"
									// according to data indices "data_ind"; "data_ind" size is comm->size + 1, data_ind[last] = smry_len;
									// after calling this constructor 'smry' can be deleted (i.e. it is safe);
									// "len" - number of design points (with func. values);
									// "data_ind", "smry" are referenced on c-RANKS-0 only
									// This CTOR does not fill 'Grads'
	ValContVecDouble(MPI_Comm c, std::vector<int> data_ind, const std::vector<std::vector<double>> &v);		// v[len][smry_len] (from c-RANKS-0) is distributed among ranks in "c"
									// according to data indices "data_ind";
									// "data_ind", "v" are referenced on c-RANKS-0 only;
									// This CTOR does not fill 'Grads'
	ValContVecDouble(MPI_Comm c, PhysModel *pm, const std::vector<std::vector<double>> &pop, int train_from_dump, const std::vector<int> &data_ind);		// Gets func. values by running 'pm' at design points pop[len][full_dim]; or from a file.
														// 'pop' is only referenced on comm-RANKS-0; 'pm' should have comm == "MPI_COMM_SELF" (i.e. all pm->comm-RANKS == 0)
														// If train_from_dump != -1, 'pop' is not used, and pop.size() lines are taken from the appropriate file
														// "data_ind" (sync on all ranks) should be taken from BDC
														// This CTOR does not fill 'Grads'
	ValContVecDouble(MPI_Comm c, std::vector<int> data_ind, const std::vector<HMMPI::Mat> &data_sens);		// data_sens[len_gr][smry_len, fulldim]	(from c-RANKS-0) is distributed to fill 'Grads' according to 'data_ind' (c-RANKS-0)
																											// This CTOR does not fill 'Vecs'
	virtual int vals_count() const;
	virtual int total_count() const;
	virtual void DistrValues(std::vector<KrigEnd> &dep, const std::vector<size_t> &inds) const;		// 'dep' should be consistent with local (current rank's) 'Vecs' and 'Grads'
	void Add(const ValContVecDouble &b);				// appends data from "b" to 'this'; MPI layout should be the same
};
//---------------------------------------------------------------------------
// works with PM_SimProxy
// MPI-DISTRIBUTED class, similar to ValContVecDouble, but Vecs[i] corresponding to different sub-proxies "i" may have different sizes (i.e. different number of design points for sub-proxy "i")
//------------
// TODO adding the gradients not implemented yet!
//------------
class ValContSimProxy : public ValCont
{
protected:
	std::vector<std::vector<double>> Vecs;	// MPI-distributed; Vecs[i] - values vector for proxy[i], i.e. indexing is "Vecs[smry_len][...]"

	virtual void FitSmryFromFile(std::string fname, int l){};
public:
	ValContSimProxy(MPI_Comm c, std::vector<int> data_ind, const std::vector<std::vector<double>> &v);		// v[smry_len][...] (from c-RANKS-0) are distributed among ranks in "c"
											// according to data indices "data_ind"; "data_ind" size is comm->size + 1, data_ind[last] = smry_len;
											// "data_ind", "v" are referenced on c-RANKS-0 only;
	virtual int vals_count() const;					// number of design points with func. values (to distinguish from gradients)
	virtual int total_count() const;				// total number of design points (func. values + gradients)
	virtual void DistrValues(std::vector<KrigEnd> &dep, const std::vector<size_t> &inds) const;		// 'dep' should be consistent with local (current rank's) 'Vecs'
};																									// 'inds' is ignored: all points are taken
//---------------------------------------------------------------------------
// VectorModel - abstract class for mapping vector -> vector (involving only active parameters)
// can calculate Jacobians
// MPI: inputs and outputs should be sync on MPI_COMM_WORLD
//---------------------------------------------------------------------------
class VectorModel
{
public:
	std::string name;

	virtual std::vector<double> Func_ACT(const std::vector<double> &x) const = 0;
	virtual HMMPI::Mat Jac_ACT(const std::vector<double> &x) const = 0;
	virtual ~VectorModel() {};
};
//---------------------------------------------------------------------------
// VM_gradient - Func = grad_ACT(PM), Jac = Hessian_ACT(PM)
// a simple vector model for testing
//---------------------------------------------------------------------------
class VM_gradient : public VectorModel
{
protected:
	MPI_Comm comm;
	PhysModel *PM;

public:
	VM_gradient(PhysModel *pm) : comm(pm->GetComm()), PM(pm) {name = "VM_gradient";};
	virtual std::vector<double> Func_ACT(const std::vector<double> &x) const;
	virtual HMMPI::Mat Jac_ACT(const std::vector<double> &x) const;
};
//---------------------------------------------------------------------------
// VM_Ham_eq1 - class for calculating residual in Hamiltonian equation #1, input params is momentum vector 'p'; prior to use also set up vectors 'x', 'p0', and 'eps'
// It refers to PM_FullHamiltonian *Ham0 which may be storing caches
// comm = Ham0->comm
// All inputs and outputs - sync on 'comm'
//---------------------------------------------------------------------------
class VM_Ham_eq1 : public VectorModel
{
protected:
	MPI_Comm comm;
	PM_FullHamiltonian *Ham0;

public:
	std::vector<double> x;					// input coordinate vector, sync on 'comm'
	std::vector<double> p0;					// previous momentum vector
	double eps;								// leap frog step

	VM_Ham_eq1(PM_FullHamiltonian *ham0) : comm(ham0->GetComm()), Ham0(ham0), eps(0) {name = "VM_HAMILTONIAN_EQ1";};
	virtual std::vector<double> Func_ACT(const std::vector<double> &p) const;
	virtual HMMPI::Mat Jac_ACT(const std::vector<double> &p) const;
};
//---------------------------------------------------------------------------
// VM_Ham_eq2 - class for calculating residual in Hamiltonian equation #2, input params is coordinate vector 'x'; prior to use also set up vectors 'x0', 'p', and 'eps'
// It refers to PM_FullHamiltonian *Ham0, *Ham1 which may be storing caches
// comm = Ham0->comm
// All inputs and outputs - sync on 'comm'
//---------------------------------------------------------------------------
class VM_Ham_eq2 : public VectorModel
{
protected:
	MPI_Comm comm;
	PM_FullHamiltonian *Ham0, *Ham1;

public:
	std::vector<double> x0;					// previous coordinate vector, sync on 'comm'
	std::vector<double> p;					// momentum vector
	mutable double eps;						// leap frog step; "mutable" is for 'eps' adjustment in derived class VM_Ham_eq2_eps

	VM_Ham_eq2(PM_FullHamiltonian *ham0, PM_FullHamiltonian *ham1) : comm(ham0->GetComm()), Ham0(ham0), Ham1(ham1), eps(0) {name = "VM_HAMILTONIAN_EQ2";};
	virtual std::vector<double> Func_ACT(const std::vector<double> &x) const;
	virtual HMMPI::Mat Jac_ACT(const std::vector<double> &x) const;
};
//---------------------------------------------------------------------------
// Version of VM_Ham_eq2 where coordinate 'i0' of the solution vector is fixed to value 'M0', and 'eps' becomes a variable found from this constraint.
// Dimension of 'x' is decreased by 1 compared to the corresponding VM_Ham_eq2, dimensions of 'x0', 'p' are not affected.
// Mapping of 'x' from VM_Ham_eq2_eps to VM_Ham_eq2: x[0, i0){M0}[i0, actdim-1) <-> xfull[0, i0)[i0][i0+1, actdim)
// Prior to use, set 'i0', 'M0'
//---------------------------------------------------------------------------
class VM_Ham_eq2_eps : public VM_Ham_eq2
{
protected:
	void calc_eps(const std::vector<double> &xfull) const;					// calculates 'eps' from i0-constraint; 'xfull' dimension is ACTDIM
	std::vector<size_t> indices_no_i0() const;								// [0,.. i0-1, i0+1,.. ACTDIM), indices vector of dimension ACTDIM-1

public:
	int i0;										// index within ACTDIM
	double M0;

	VM_Ham_eq2_eps(const VM_Ham_eq2 &VM2, int i, double m) : VM_Ham_eq2(VM2), i0(i), M0(m) {name = "VM_HAMILTONIAN_EQ2_EPSILON";};
	virtual std::vector<double> Func_ACT(const std::vector<double> &x) const;		// prior to use, set 'x0', 'p', 'i0', 'M0'; after use, the found 'eps' can be retrieved
	virtual HMMPI::Mat Jac_ACT(const std::vector<double> &x) const;					// -"-
	virtual std::vector<double> map_x_xfull(const std::vector<double> &x) const;			// mapping of 'x' from VM_Ham_eq2_eps (ACTDIM-1) to VM_Ham_eq2 (ACTDIM)
	virtual std::vector<double> map_xfull_x(const std::vector<double> &xfull) const;		// auxiliary function, removes coordinate 'i0' from the vector
};
//---------------------------------------------------------------------------
// Similar to VM_Ham_eq2_eps, but 'eps' is a free variable
// Dimension of 'x' is retained, x[i0] carries eta = 10*eps
// Prior to use, set 'i0', 'M0'
//---------------------------------------------------------------------------
class VM_Ham_eq2_eps_full : public VM_Ham_eq2_eps
{
protected:
	const double Scale;
public:
	VM_Ham_eq2_eps_full(const VM_Ham_eq2 &VM2, int i, double m) : VM_Ham_eq2_eps(VM2, i, m), Scale(10.0) {name = "VM_HAMILTONIAN_EQ2_EPSILON_FULL";};
	virtual std::vector<double> Func_ACT(const std::vector<double> &x) const;		// prior to use, set 'x0', 'p', 'i0', 'M0'; "x" should have been mapped from "xfull"
	virtual HMMPI::Mat Jac_ACT(const std::vector<double> &x) const;					// -"-
	virtual std::vector<double> map_x_xfull(const std::vector<double> &x) const;			// mapping of "x" from VM_Ham_eq2_eps_full to VM_Ham_eq2 (filling 'eps')
	virtual std::vector<double> map_xfull_x(const std::vector<double> &xfull) const;		// "xfull" and current 'eps' are used to make "x" suitable for use by VM_Ham_eq2_eps_full
};
//---------------------------------------------------------------------------

#endif /* PHYSMODELS_H_ */
