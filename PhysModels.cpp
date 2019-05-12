/*
 * PhysModels.cpp
 *
 *  Created on: 23 May 2016
 *      Author: ilya fursov
 */

#include <algorithm>
#include <limits>
#include "Abstract.h"
#include "PhysModels.h"
#include "ConcretePhysModels.h"
#include "mpi.h"
#include "Parsing.h"
#include "Parsing2.h"

//#define TESTCOMM

//---------------------------------------------------------------------------
// PhysModel
//---------------------------------------------------------------------------
PhysModel::PhysModel(MPI_Comm c) : comm(c), con(nullptr), name("PhysModel")
{
	MPI_Comm_rank(MPI_COMM_WORLD, &RNK);

#ifdef TESTCOMM
	std::cout << "rank " << RNK << "\tPhysModel::PhysModel\n";
#endif
}
//---------------------------------------------------------------------------
PhysModel::PhysModel(std::vector<double> in, std::vector<int> act, std::vector<int> tot, const HMMPI::BoundConstr *c) : comm(MPI_COMM_SELF), init(std::move(in)), act_ind(std::move(act)), tot_ind(std::move(tot)), con(c), name("PhysModel")
{
	MPI_Comm_rank(MPI_COMM_WORLD, &RNK);

#ifdef TESTCOMM
	std::cout << "rank " << RNK << "\tPhysModel::PhysModel\n";
#endif
}
//---------------------------------------------------------------------------
PhysModel::PhysModel(Parser_1 *K, KW_item *kw, MPI_Comm c) : comm(c), name("PhysModel")
{
	MPI_Comm_rank(MPI_COMM_WORLD, &RNK);
	DECLKWD(params, KW_parameters, "PARAMETERS");
	const ParamsInterface *par_interface = params->GetParamsInterface();

	init = par_interface->init;
	act_ind = par_interface->get_act_ind();
	tot_ind = par_interface->get_tot_ind();
	con = par_interface;

#ifdef TESTCTOR
	std::ofstream testf(HMMPI::stringFormatArr("TESTCTOR_out_{0:%d}.txt", std::vector<int>{RNK}), std::ios::app);
	testf << "rank " << RNK << ", PhysModel easy CTOR, this = " << this << "\n";
	testf.close();
#endif
}
//---------------------------------------------------------------------------
PhysModel::~PhysModel()
{
#ifdef TESTCTOR
	std::ofstream testf(HMMPI::stringFormatArr("TESTCTOR_out_{0:%d}.txt", std::vector<int>{RNK}), std::ios::app);
	testf << "rank " << RNK << ", PhysModel -- DTOR --, this = " << this << "\n";
	testf.close();
#endif
}
//---------------------------------------------------------------------------
std::vector<double> PhysModel::tot_par(const std::vector<double> &act_par) const
{
	std::vector<double> res = init;
	size_t act_size = act_ind.size();
	int dim = ParamsDim();

	if (init.size() != (size_t)dim)
		throw HMMPI::Exception(HMMPI::stringFormatArr("init.size() != ParamsDim() [{0:%zu}, {1:%zu} respectively] in PhysModel::tot_par -- rank-{2:%zu}", std::vector<size_t>{init.size(), (size_t)dim, (size_t)RNK}));
	if (act_par.size() != act_size)
		throw HMMPI::Exception(HMMPI::stringFormatArr("act_par.size() != act_ind.size() [{0:%zu}, {1:%zu} respectively] in PhysModel::tot_par -- rank-{2:%zu}", std::vector<size_t>{act_par.size(), act_size, (size_t)RNK}));
	for (size_t i = 0; i < act_size; i++)
	{
		int ind = act_ind[i];
		if (ind < 0 || ind >= dim)
			throw HMMPI::Exception("Active parameter index is out of range in PhysModel::tot_par");
		res[ind] = act_par[i];
	}

	return res;
}
//---------------------------------------------------------------------------
std::vector<double> PhysModel::act_par(const std::vector<double> &tot_par) const
{
	return HMMPI::Reorder(tot_par, act_ind);
}
//---------------------------------------------------------------------------
HMMPI::Mat PhysModel::act_mat(const HMMPI::Mat &tot_M) const
{
	if (tot_M.ICount() != tot_M.JCount())
		HMMPI::Exception("Неквадратная матрица в PhysModel::act_mat", "Non-square matrix in PhysModel::act_mat");

	return tot_M.Reorder(act_ind, act_ind);
}
//---------------------------------------------------------------------------
std::vector<double> PhysModel::ObjFuncGrad(const std::vector<double> &params)
{
	throw HMMPI::Exception(HMMPI::stringFormatArr("Запрещенный вызов ObjFuncGrad ({0:%s})", "Illegal call to ObjFuncGrad ({0:%s})", name));
}
//---------------------------------------------------------------------------
double PhysModel::ObjFuncGradDir(const std::vector<double> &params, const std::vector<double> &dir)
{
	int rank;
	MPI_Comm_rank(comm, &rank);

	std::vector<double> grad = ObjFuncGrad(params);
	if (rank == 0)
		return HMMPI::InnerProd(grad, dir);
	else
		return 0;
}
//---------------------------------------------------------------------------
HMMPI::Mat PhysModel::ObjFuncHess(const std::vector<double> &params)
{
	throw HMMPI::Exception(HMMPI::stringFormatArr("Запрещенный вызов ObjFuncHess ({0:%s})", "Illegal call to ObjFuncHess ({0:%s})", name));
}
//---------------------------------------------------------------------------
HMMPI::Mat PhysModel::ObjFuncFisher(const std::vector<double> &params)
{
	throw HMMPI::Exception(HMMPI::stringFormatArr("Запрещенный вызов ObjFuncFisher ({0:%s})", "Illegal call to ObjFuncFisher ({0:%s})", name));
}
//---------------------------------------------------------------------------
HMMPI::Mat PhysModel::ObjFuncFisher_dxi(const std::vector<double> &params, const int i, int r)
{
	throw HMMPI::Exception(HMMPI::stringFormatArr("Запрещенный вызов ObjFuncFisher_dxi ({0:%s})", "Illegal call to ObjFuncFisher_dxi ({0:%s})", name));
}
//---------------------------------------------------------------------------
double PhysModel::ObjFunc_ACT(const std::vector<double> &params)
{
	return ObjFunc(tot_par(params));
}
//---------------------------------------------------------------------------
std::vector<double> PhysModel::ObjFuncGrad_ACT(const std::vector<double> &params)
{
	int rank;
	MPI_Comm_rank(comm, &rank);

	std::vector<double> grad = ObjFuncGrad(tot_par(params));		// long gradient
	if (rank == 0)
		return HMMPI::Reorder(grad, act_ind);
	else
		return std::vector<double>();
}
//---------------------------------------------------------------------------
double PhysModel::ObjFuncGradDir_ACT(const std::vector<double> &params, const std::vector<double> &dir)
{
	int rank;
	MPI_Comm_rank(comm, &rank);

	std::vector<double> grad = ObjFuncGrad_ACT(params);
	if (rank == 0)
		return HMMPI::InnerProd(grad, dir);
	else
		return 0;
}
//---------------------------------------------------------------------------
HMMPI::Mat PhysModel::ObjFuncHess_ACT(const std::vector<double> &params)
{
	int rank;
	MPI_Comm_rank(comm, &rank);

	HMMPI::Mat Hess = ObjFuncHess(tot_par(params));		// big matrix
	if (rank == 0)
		return Hess.Reorder(act_ind, act_ind);
	else
		return HMMPI::Mat();
}
//---------------------------------------------------------------------------
HMMPI::Mat PhysModel::ObjFuncFisher_ACT(const std::vector<double> &params)
{
	int rank;
	MPI_Comm_rank(comm, &rank);

	HMMPI::Mat FI = ObjFuncFisher(tot_par(params));		// big matrix
	if (rank == 0)
		return FI.Reorder(act_ind, act_ind);
	else
		return HMMPI::Mat();
}
//---------------------------------------------------------------------------
HMMPI::Mat PhysModel::ObjFuncFisher_dxi_ACT(const std::vector<double> &params, const int i, int r)					// i - index in act dim
{
	if (i < 0 || i >= (int)act_ind.size())
		throw HMMPI::Exception("Index 'i' out of range [0, actdim) in PhysModel::ObjFuncFisher_dxi_ACT");

	int rank;
	MPI_Comm_rank(comm, &rank);

	HMMPI::Mat dFI = ObjFuncFisher_dxi(tot_par(params), act_ind[i], r);		// big matrix
	if (rank == r)
		return dFI.Reorder(act_ind, act_ind);
	else
		return HMMPI::Mat();
}
//---------------------------------------------------------------------------
bool PhysModel::CheckLimits(const std::vector<double> &params) const
{
	if (params.size() != (size_t)ParamsDim())
		throw HMMPI::Exception("params.size() != ParamsDim() in PhysModel::CheckLimits");

	limits_msg = "";
	if (con != 0)
	{
		limits_msg = con->Check(params);
		if (limits_msg == "")
			return true;
		else
			return false;
	}
	else
		return true;
}
//---------------------------------------------------------------------------
bool PhysModel::CheckLimits_ACT(const std::vector<double> &params) const
{
	return CheckLimits(tot_par(params));
}
//---------------------------------------------------------------------------
bool PhysModel::CheckLimitsEps(std::vector<double> &params, const double eps) const
{
	if (params.size() != (size_t)ParamsDim())
		throw HMMPI::Exception("params.size() != ParamsDim() in PhysModel::CheckLimitsEps");

	limits_msg = "";
	if (con != 0)
	{
		limits_msg = con->CheckEps(params, eps);
		if (limits_msg == "")
			return true;
		else
			return false;
	}
	else
		return true;
}
//---------------------------------------------------------------------------
bool PhysModel::FindIntersect(const std::vector<double> &x0, const std::vector<double> &x1, std::vector<double> &xint, double &alpha, int &i) const
{
	if (x1.size() != (size_t)ParamsDim())
		throw HMMPI::Exception("x1.size() != ParamsDim() in PhysModel::FindIntersect");

	if (con != 0)
		return con->FindIntersect(x0, x1, xint, alpha, i);
	else
		return true;
}
//---------------------------------------------------------------------------
bool PhysModel::FindIntersect_ACT(const std::vector<double> &x0, const std::vector<double> &x1, std::vector<double> &xint, double &alpha, int &i) const
{
	std::vector<double> xint_tot;	// full-dim vector
	int i_tot;						// index in full-dim vector

	bool res = FindIntersect(tot_par(x0), tot_par(x1), xint_tot, alpha, i_tot);
	if (!res)
	{
		xint = act_par(xint_tot);	// convert full-dim params to active params
		i = tot_ind[i_tot];			// convert full-dim index to active index
		if (i == -1)
			throw HMMPI::Exception(HMMPI::stringFormatArr("Full-dim index {0:%d} corresponds to non-active parameter in PhysModel::FindIntersect_ACT", std::vector<int>{i_tot}));
	}

	return res;
}
//---------------------------------------------------------------------------
void PhysModel::WriteLimits(const std::vector<double> &p, std::string fname) const
{
	if (p.size() != (size_t)ParamsDim())
		throw HMMPI::Exception("p.size() != ParamsDim() in PhysModel::WriteLimits");

	if (con != 0)
		con->Write_params_log(p, fname);
}
//---------------------------------------------------------------------------
void PhysModel::WriteLimits_ACT(const std::vector<double> &p, std::string fname) const
{
	WriteLimits(tot_par(p), fname);
}
//---------------------------------------------------------------------------
int PhysModel::ParamsDim_ACT() const noexcept
{
	int res = act_ind.size();
	if (res == 0)			// in case 'act_ind' is not defined
		res = ParamsDim();

	return res;
}
//---------------------------------------------------------------------------
const std::vector<double> &PhysModel::ModelledData() const
{
	if (modelled_data.size() == 0 && ModelledDataSize() != 0)
		throw HMMPI::Exception(HMMPI::stringFormatArr("modelled_data.size() == 0 (expected size is {0:%ld}), it was not properly filled in the current physical model ", std::vector<size_t>{ModelledDataSize()}) + name);

	return modelled_data;
}
//---------------------------------------------------------------------------
void PhysModel::PerturbData()
{
	throw HMMPI::Exception("Запрещенный вызов PerturbData", "Illegal call to PerturbData");
}
//---------------------------------------------------------------------------
void PhysModel::ExportIAC(PhysModel *p) const
{
	p->init = init;
	p->act_ind = act_ind;
	p->tot_ind = tot_ind;
	p->con = con;
}
//---------------------------------------------------------------------------
// ModelFactory
//---------------------------------------------------------------------------
void ModelFactory::FillCreators(Parser_1 *K, KW_item *kw, HMMPI::CorrelCreator **cor, HMMPI::StdCreator **std, HMMPI::DataCreator **data)
{
	DECLKWD(corrstruct, KW_corrstruct, "CORRSTRUCT");
	DECLKWD(matvecvec, KW_matvecvec, "MATVECVEC");

	kw->Start_pre();
	kw->Add_pre("CORRSTRUCT");
	kw->Add_pre("MATVECVEC");
	kw->Finish_pre();

	*cor = corrstruct;
	*std = matvecvec;
	*data = matvecvec;
}
//---------------------------------------------------------------------------
void ModelFactory::MakeComms(MPI_Comm in, MPI_Comm *one, MPI_Comm *two, bool ref_is_dataproxy)
{
	int size = 0;
	if (in != MPI_COMM_NULL)
		MPI_Comm_size(in, &size);	// total number of processes in "in"

	int nwork;
	if (!ref_is_dataproxy)
		nwork = size;
	else
		nwork = sqrt(size);			// NUMGRAD + DATAPROXY

	PhysModMPI::HMMPI_Comm_split(nwork, in, one, two);
}
//---------------------------------------------------------------------------
bool ModelFactory::object_for_deletion(const HMMPI::ManagedObject *m)
{
	return dynamic_cast<const KrigCorr*>(m) == nullptr &&
		   dynamic_cast<const KrigSigma*>(m) == nullptr &&
		   dynamic_cast<const PM_SimProxy*>(m) == nullptr;
}
//---------------------------------------------------------------------------
PhysModel *ModelFactory::Make(std::string &message, Parser_1 *K, KW_item *kw, std::string cwd, int num, MPI_Comm c, std::vector<HMMPI::ManagedObject*> *mngd, bool train)
{
	PhysModel *Res = 0;
	DECLKWD(physmodel, KW_physmodel, "PHYSMODEL");
	DECLKWD(proxy, KW_proxy, "PROXY_CONFIG");
	DECLKWD(model, KW_model, "MODEL");

	kw->Start_pre();
	kw->Add_pre("PHYSMODEL");
	kw->Finish_pre();

	std::string msg = physmodel->CheckRefs();
	if (msg != "")
		throw HMMPI::Exception(msg);
	if (num < 1 || num > int(physmodel->ref.size()))		// "num" is 1-based
		throw HMMPI::Exception("num вне допустимого диапазона в ModelFactory::Make", "num out of range in ModelFactory::Make");

	const std::string Type = physmodel->type[num-1];
	int Ref = physmodel->ref[num-1];

	std::vector<HMMPI::ManagedObject*> managed;				// add managed objects here if mngd == 0
	std::vector<HMMPI::ManagedObject*> *mngd_ptr;
	if (mngd != 0)
		mngd_ptr = mngd;
	else
		mngd_ptr = &managed;

	// update the message
	if (mngd == 0)		// top-level call
		message = Type;
	else
		message += " <- " + Type;

	// define the communicators
	if ((Type == "DATAPROXY" || Type == "DATAPROXY2" || Type == "NUMGRAD" || Type == "ECLIPSE" || Type == "SIMECL" || Type == "PCONNECT" || Type == "CONC" || Type == "SIMPROXY") && c == MPI_COMM_SELF)
		c = MPI_COMM_WORLD;

	MPI_Comm this_comm = c, next_comm = MPI_COMM_SELF;
	if (!physmodel->is_plain[num-1] && (physmodel->type[Ref-1] == "ECLIPSE" || physmodel->type[Ref-1] == "SIMECL" || physmodel->type[Ref-1] == "PCONNECT" ||
										physmodel->type[Ref-1] == "CONC" || physmodel->type[Ref-1] == "SIMPROXY"))		// reference type which refers to ECLIPSE || SIMECL || PCONNECT || CONC || SIMPROXY
	{
		MPI_Comm_dup(MPI_COMM_SELF, &next_comm);		// duplicate of MPI_COMM_SELF which will not be changed to MPI_COMM_WORLD when ECLIPSE/SIMECL/PCONNECT/CONC/SIMPROXY model is made
		mngd_ptr->push_back(new HMMPI::ManagedComm(next_comm));
	}
	if (Type == "NUMGRAD")
	{
		MakeComms(c, &this_comm, &next_comm, bool(physmodel->type[Ref-1] == "DATAPROXY" || physmodel->type[Ref-1] == "DATAPROXY2"));
		mngd_ptr->push_back(new HMMPI::ManagedComm(this_comm));
		mngd_ptr->push_back(new HMMPI::ManagedComm(next_comm));
	}
	if (Type == "LAGRSPHER" || Type == "SPHERICAL" || Type == "CUBEBOUND" || Type == "HAMILTONIAN" || Type == "POSTERIOR")
		next_comm = this_comm;

	if (Type == "PROXY" || Type == "DATAPROXY" || Type == "DATAPROXY2")
	{
		kw->Start_pre();
		kw->Add_pre("PROXY_CONFIG");
		kw->Finish_pre();
	}

	// DEBUG -- output the ranks info
	std::string msg_ranks = HMMPI::MPI_Ranks(std::vector<MPI_Comm>{this_comm, next_comm});
	int RNK;
	MPI_Comm_rank(MPI_COMM_WORLD, &RNK);
	if (RNK == 0)
	{
		if (Ref != 0)
			std::cout << "----------------- ModelFactory::Make communicators: -----------------\nWORLD\t" << Type << "\t" << physmodel->type[Ref-1] << "\n" << msg_ranks << "\n";
		else
			std::cout << "----------------- ModelFactory::Make communicators: -----------------\nWORLD\t" << Type << "\n" << msg_ranks << "\n";
	}
	// DEBUG

	// create the reference model, if needed
	PhysModel *ref = 0;
	if (Ref != 0)
	{
		ref = Make(message, K, kw, cwd, Ref, next_comm, mngd_ptr, true);		// recursively make a reference model (for which 'train' = true)
		if (object_for_deletion(ref))
			mngd_ptr->push_back(ref);
		else
		{
//			std::cout << "DEBUG -------------------- not for deletion: " << dynamic_cast<PhysModel*>(ref)->name << "\n";	// DEBUG
		}
	}

	// create the main model based on type
	if (Type == "ECLIPSE")
		Res = new PhysModelHM(K, kw, cwd, this_comm);
	if (Type == "SIMECL")
		Res = new PMEclipse(K, kw, cwd, this_comm);
	if (Type == "PCONNECT")
		Res = new PMpConnect(K, kw, cwd, this_comm);
	if (Type == "CONC")
		Res = new PMConc(K, kw, cwd, this_comm);
	if (Type == "SIMPROXY")
	{
		kw->Start_pre();
		kw->Add_pre("MODEL");
		kw->Finish_pre();

		Res = model->MakeModel(kw, cwd, "PROXY");			// NOTE: "Res" will be deleted automatically; MPI_COMM_WORLD is always used here (internally)
	}
	if (Type == "LIN")
		Res = new PM_Linear(K, kw, this_comm);
	if (Type == "ROSEN")
		Res = new PM_Rosenbrock(K, kw, this_comm);

	if (Type == "NUMGRAD")
		Res = new PhysModGradNum(this_comm, ref, K, kw);
	if (Type == "POSTERIOR")
	{
		DECLKWD(matvec, KW_matvec, "MATVEC");
		kw->Start_pre();
		kw->Add_pre("MATVEC");
		kw->Finish_pre();

		Res = new PM_Posterior(ref, matvec->M, matvec->v1);
	}
	if (Type == "LAGRSPHER" || Type == "SPHERICAL" || Type == "CUBEBOUND" || Type == "HAMILTONIAN")
	{
		DECLKWD(points, KW_3points, "3POINTS");			// first point provides sphere center 'x0'; for HAMILTONIAN it provides momentum 'p'
		DECLKWD(config, KW_opt_config, "OPT_CONFIG");	// r0 provides sphere radius 'Hk', tau1 provides 'lambda', delta provides 'delta'

		kw->Start_pre();
		kw->Add_pre("3POINTS");
		kw->Add_pre("OPT_CONFIG");
		kw->Finish_pre();

		if (Type == "LAGRSPHER")
		{
			Res = new PM_LagrangianSpher(ref, config->tau1);
			if (int(points->x.size()) != Res->ParamsDim() - 1)
				throw HMMPI::Exception("Dimension in 3POINTS should be consistent with parameters dimension");
			dynamic_cast<PM_LagrangianSpher*>(Res)->x0 = points->x;
			dynamic_cast<PM_LagrangianSpher*>(Res)->Hk = config->r0;
		}
		else if (Type == "SPHERICAL")
			Res = new PM_Spherical(ref, config->r0, points->x, config->delta);
		else if (Type == "CUBEBOUND")
			Res = new PM_CubeBounds(ref, config->r0, points->x);
		else		   // HAMILTONIAN
		{
			const double MM_shift = 0;				// the shift is hardcoded so far
			if (points->x.size() != (size_t)ref->ParamsDim())
				throw HMMPI::Exception("Dimension in 3POINTS should equal the parameters dimension");
			PM_FullHamiltonian *Ham = new PM_FullHamiltonian(ref, MM_shift);
			Ham->pact = Ham->act_par(points->x);
			Res = Ham;
		}
	}
	if (Type == "PROXY")
		Res = new PM_Proxy(this_comm, ref, K, kw, proxy);
	if (Type == "KRIGCORR" || Type == "KRIGSIGMA")
	{
		PM_Proxy *pk = dynamic_cast<PM_Proxy*>(ref);
		if (pk == nullptr)
			throw HMMPI::Exception(Type + ": cannot convert reference model to PM_Proxy");

		if (Type == "KRIGCORR")
			Res = pk->get_KrigCorr();
		else
			Res = pk->get_KrigSigma();
	}
	if (Type == "DATAPROXY" || Type == "DATAPROXY2")
	{
		HMMPI::CovAndDataCreator *cov_creator = dynamic_cast<HMMPI::CovAndDataCreator*>(ref);		// if 'ref' is ECLIPSE/SIMECL/ROSENBROCK/PCONNECT/CONC, it will be used as CovAndDataCreator
		HMMPI::CorrelCreator *Corr = dynamic_cast<HMMPI::CorrelCreator*>(cov_creator);				// SIMPROXY doesn't have CovAndDataCreator, and is not supposed to be used inside DataProxy
		HMMPI::StdCreator *Std = dynamic_cast<HMMPI::StdCreator*>(cov_creator);
		HMMPI::DataCreator *Data = dynamic_cast<HMMPI::DataCreator*>(cov_creator);

		if (cov_creator == 0)
			FillCreators(K, kw, &Corr, &Std, &Data);	// ECLIPSE/SIMECL/ROSENBROCK/PCONNECT/CONC not detected; TODO Cov and Data creators should work in a more unified way for all models, preferably like it works now for ECLIPSE/SIMECL/...; i.e. all models with data should be convertible to these creators

		HMMPI::BlockDiagMat *BDC = new HMMPI::BlockDiagMat(this_comm, Corr, Std);
		mngd_ptr->push_back(BDC);

		if (Type == "DATAPROXY")
			Res = new PM_DataProxy(ref, K, kw, proxy, BDC, Data->Data());
		else
			Res = new PM_DataProxy2(ref, K, kw, proxy, BDC, Data->Data());
	}

	// train the proxy, if applicable
	PM_Proxy *Res_proxy = dynamic_cast<PM_Proxy*>(Res);
	if (Res_proxy != 0 && train && dynamic_cast<PM_SimProxy*>(Res) == nullptr)	// only PM_Proxy (except SIMPROXY) with train == true is considered
	{
		DECLKWD(params, KW_parameters, "PARAMETERS");
		DECLKWD(config, KW_proxy, "PROXY_CONFIG");
		DECLKWD(proxy_dump, KW_proxy_dump, "PROXY_DUMP");
		const ParamsInterface *par_interface = params->GetParamsInterface();

		kw->Start_pre();
		kw->Add_pre("PROXY_DUMP");
		kw->Finish_pre();

		K->AppText(Res_proxy->init_msg() + "\n");

		long long int seed = 1;										// seed = 1 for easier comparison with MULTIPLE_SEQ + SOBOL
		std::vector<std::vector<double>> Y0(proxy->init_pts);		// array of design points

		for (int i = 0; i < proxy->init_pts; i++)
			Y0[i] = par_interface->SobolDP(seed);

		//Res_proxy->SetDumpFlag(256);	// DEBUG
		Res_proxy->SetTrainFromDump(proxy_dump->train_ind);
		if (Y0.size() != 0)
		{
			std::string msg;
			msg = Res_proxy->Train(Y0, config->ind_grad_init_pts, 0);		// Nfval_pts = 0 since the proxy is empty so far
			K->AppText(msg);
		}
		Res_proxy->SetDumpFlag(-1);
		Res_proxy->SetTrainFromDump(-1);

		// DEBUG **********************	CHECKING proxy training in different ways, but with the same total set of design points
//		Y0 = std::vector<std::vector<double>>();
//		Res = Res_proxy = Res_proxy->Copy();	// make a copy before additional training
//		for (int i = 0; i < 4*(proxy->init_pts); i++)
//			Y0.push_back(par_interface->SobolDP(seed));
//
//		Res_proxy->Train(Y0);
		// DEBUG **********************

		if (Res_proxy->do_optimize_krig)
			K->AppText(HMMPI::MessageRE("Параметры кригинга оптимизируются при каждой тренировке прокси\n", "Kriging parameters are optimized on each proxy training\n"));
		else
			K->AppText(HMMPI::MessageRE("Параметры кригинга _НЕ_ оптимизируются при тренировке прокси\n", "Kriging parameters are _NOT_ optimized on proxy training\n"));
	}

	if (mngd == 0)		// top-level call
		if (dynamic_cast<PM_SimProxy*>(Res) == nullptr)		// for PM_SimProxy don't delete 'Res', and also there should be no dependent managed objects; NOTE: alternatively could make Res->Copy()
			ptrs[Res] = managed;

	MPI_Barrier(MPI_COMM_WORLD);		// to enforce synchronization of Bcasts and other communication (to avoid hidden bugs)
	return Res;
}
//---------------------------------------------------------------------------
ModelFactory::~ModelFactory()
{
	for (auto p : ptrs)
	{
		bool del_model = object_for_deletion(p.first);		// flag showing if the model itself should be deleted; KrigCorr should not be deleted, since it is deleted when the corresponding PM_ProxyKrig is deleted

//		if (!del_model)
//			std::cout << "DEBUG -------------------- not for deletion: " << dynamic_cast<PhysModel*>(p.first)->name << "\n";	// DEBUG

		for (auto m : p.second)		// m - managed object associated with model p.first
			delete m;

		if (del_model)
			delete p.first;
	}
}
//---------------------------------------------------------------------------
void ModelFactory::FreeModel(PhysModel* pm)
{
	if (ptrs.count(pm))		// 'pm' found
	{
		bool del_model = object_for_deletion(pm);			// flag showing if the model itself should be deleted

//		if (!del_model)
//			std::cout << "DEBUG -------------------- not for deletion: " << dynamic_cast<PhysModel*>(pm)->name << "\n";	// DEBUG

		for (auto m : ptrs[pm])		// m - managed object associated with model 'pm'
			delete m;

		ptrs.erase(pm);				// remove 'ptrs' entry

		if (del_model)
			delete pm;				// free 'pm'
	}
	else
		throw HMMPI::Exception("Attempt to delete a model which is not in the records in ModelFactory::FreeModel");
}
//---------------------------------------------------------------------------
// PhysModMPI
//---------------------------------------------------------------------------
void PhysModMPI::fill_counts_displs(int ind1)
{
	int smry_len = ModelledDataSize();
	int actdim = ParamsDim_ACT();

	int rank = -1, size = -1;
	if (comm != MPI_COMM_NULL)
	{
		MPI_Comm_size(comm, &size);
		MPI_Comm_rank(comm, &rank);
	}

	if (rank == 0)
	{
		displFIT = displPOP = displSMRY = std::vector<int>(size+1);
		displFIT[0] = 0;
	}
	if (comm != MPI_COMM_NULL)
		MPI_Gather(&ind1, 1, MPI_INT, displFIT.data()+1, 1, MPI_INT, 0, comm);

	if (rank == 0)
	{
		for (size_t j = 0; j < displFIT.size(); j++)
		{
			displPOP[j] = actdim * displFIT[j];
			displSMRY[j] = smry_len * displFIT[j];
		}
		countFIT = countPOP = countSMRY = std::vector<int>(size);
		for (size_t j = 1; j < displFIT.size(); j++)
		{
			countFIT[j-1] = displFIT[j] - displFIT[j-1];
			countPOP[j-1] = displPOP[j] - displPOP[j-1];
			countSMRY[j-1] = displSMRY[j] - displSMRY[j-1];
		}
	}
}
//---------------------------------------------------------------------------
PhysModMPI::PhysModMPI(MPI_Comm c, PhysModel *pm) : PhysModel(c), PM(pm)
{
	name = "PhysModMPI";

	MPI_Barrier(MPI_COMM_WORLD);							// enforce synchronization of Bcasts and other communication (to avoid hidden bugs)
	HMMPI_Comm_check(comm, pm->GetComm(), "PhysModMPI");	// check communicators compatibility
	pm->ExportIAC(this);

#ifdef TESTCOMM
	std::cout << "rank " << RNK << "\tPhysModMPI::PhysModMPI\n";
#endif
}
//---------------------------------------------------------------------------
void PhysModMPI::HMMPI_Comm_check(MPI_Comm first, MPI_Comm second, const std::string &where)
{
	int rank_1 = -1, rank_2 = -1;
	if (first != MPI_COMM_NULL)
		MPI_Comm_rank(first, &rank_1);

	if (second != MPI_COMM_NULL)
		MPI_Comm_rank(second, &rank_2);

	int err = 0;			// 0 - no error, >= 1 - error!
	if (rank_1 >= 0 && rank_2 != 0)
		err = 1;
	if (rank_2 == 0 && rank_1 == -1)
		err = 1;
	if (rank_2 > 0 && rank_1 != -1)
		err = 1;

	MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);		// err is sync over MPI_COMM_WORLD

	if (err > 0)			// the error message on MPI_COMM_WORLD-rank-0 will be slightly different (will include ranks list)
	{
		std::string msg = HMMPI::stringFormatArr("Несовместимые коммуникаторы в {0:%s}:\n", "Incompatible communicators in {0:%s}:\n", where);
		msg += HMMPI::MPI_Ranks(std::vector<MPI_Comm>{first, second});	// non-empty on MPI_COMM_WORLD-rank-0
		throw HMMPI::Exception(msg);
	}
}
//---------------------------------------------------------------------------
void PhysModMPI::HMMPI_Comm_split(int fst_size, MPI_Comm comm, MPI_Comm *first, MPI_Comm *second)
{
	if (comm == MPI_COMM_NULL || fst_size < 1)
	{
		*first = MPI_COMM_NULL;
		*second = MPI_COMM_NULL;
	}
	else
	{
		int size, rank;
		MPI_Comm_size(comm, &size);
		MPI_Comm_rank(comm, &rank);

		int color_1 = MPI_UNDEFINED, color_2 = MPI_UNDEFINED;					// colors for the two output communicators
		int ind0 = 0;
		for (int i = 0; i < fst_size; i++)			// determine which subgroup the current process belongs to
		{
			int ind1 = ind0 + size/fst_size + (i < size%fst_size ? 1 : 0);		// subgroup "i" spans comm-ranks [ind0, ind1)
			if (ind0 == rank && rank < ind1)
				color_1 = 0;
			if (ind0 <= rank && rank < ind1)
				color_2 = i;
			ind0 = ind1;
		}

		MPI_Comm_split(comm, color_1, 0, first);
		MPI_Comm_split(comm, color_2, 0, second);
	}

#ifdef TESTCTOR
	int rank00;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank00);
	std::ofstream testf(HMMPI::stringFormatArr("TESTCTOR_out_{0:%d}.txt", std::vector<int>{rank00}), std::ios::app);
	testf << "rank " << rank00 << ", HMMPI_Comm_split, first = " << *first << ", second = " << *second << "\n";
	testf.close();
#endif
}
//---------------------------------------------------------------------------
// this routine is run on the fine-grained level (involving PM_comm)
// but most fine-grained details are hidden, and essentially coarse-grained things happen (involving comm)
// example of comm: 0xxx1xxx2xxx3xxx, PM_comm: [0123][0123][0123][0123]
void PhysModMPI::ObjFuncMPI_ACT(int len, const double * const *POP, double *FIT, bool quiet_out_of_range, double **SMRY)
{
	// ranks which are:
	// comm-NOT_NULL are doing most of the job (MPI communication etc)
	// PM_comm-NOT_NULL are active throughout this function to properly call PM->ObjFunc_ACT()
	// comm-NULL && PM_comm-NULL exit in the beginning

	// ********************************
	// fill rank, size; sync some input
	MPI_Comm PM_comm = PM->GetComm();
	int rank = -1, size = -1;
	if (comm != MPI_COMM_NULL)
	{
		MPI_Comm_size(comm, &size);
		MPI_Comm_rank(comm, &rank);
	}
	else if (PM_comm == MPI_COMM_NULL)
		return;									// quit on "blank" ranks

	int smry_len = ModelledDataSize();
	int actdim = ParamsDim_ACT();
	bool smry_present = false;					// smry_present will be sync on comm, but SMRY is only referenced on comm-rank-0
	if (rank == 0 && SMRY != 0)
		smry_present = true;
	if (comm != MPI_COMM_NULL)
	{
		MPI_Bcast(&len, 1, MPI_INT, 0, comm);		// sync simple input arguments
		MPI_Bcast(&quiet_out_of_range, 1, MPI_BYTE, 0, comm);
		MPI_Bcast(&smry_present, 1, MPI_BYTE, 0, comm);
	}

	// ********************************
	// define how the models pool is distributed between ranks; this is sync over PM_comm for proper main loop behaviour
	int ind0, ind1 = 0;										// lower and upper indices for models handled by current rank - sync across PM_comm
	int num_mod = len/size + (rank < len%size ? 1 : 0);		// number of models to be handled by current rank - sync across PM_comm
	if (comm != MPI_COMM_NULL)
		MPI_Scan(&num_mod, &ind1, 1, MPI_INT, MPI_SUM, comm);

	MPI_Bcast(&num_mod, 1, MPI_INT, 0, PM_comm);			// this sync across PM_comm is needed for the main loop
	MPI_Bcast(&ind1, 1, MPI_INT, 0, PM_comm);
	ind0 = ind1 - num_mod;

	fill_counts_displs(ind1);					// fill counts and displs for Scatterv/Gatherv

//	printf("[%d] %d -- %d, num_mod = %d DEBUG\n[%d] displFIT = ", RNK, ind0, ind1, num_mod, RNK);	// DEBUG
//	printf("%s", HMMPI::ToString(displFIT, "%d").c_str());	// DEBUG

	// ********************************
	// distribute input arrays								all, stored on comm-rank-0	|	local, stored on each comm-rank
												// -------------------------------------------------------------------
	double *POPdist = 0, *POPloc = 0;			// params	|	POPdist 				|		POPloc
	double *SMRYdist = 0, *SMRYloc = 0;			// mod.data	|	SMRYdist				|		SMRYloc
	double *FITloc = 0;							// o.f.		|	FIT						|		FITloc

	char *ERRdist = 0;							// ERRdist (on each comm-rank) - stores (after all-gather) all ranks error messages
	char *ERRloc = 0;							// ERRloc (on each comm-rank) - stores local rank's error message
	if (rank == 0)
	{
		POPdist = new double[len*actdim];		// all params in one row
		if (smry_present)
			SMRYdist = new double[len*smry_len];
		for (int i = 0; i < len; i++)
			memcpy(POPdist + i*actdim, POP[i], actdim*sizeof(double));
	}
	if (comm != MPI_COMM_NULL)
	{
		POPloc = new double[num_mod*actdim];
		if (smry_present)
			SMRYloc = new double[num_mod*smry_len];
		FITloc = new double[num_mod];
		ERRdist = new char[size*HMMPI::BUFFSIZE];
		ERRloc = new char[HMMPI::BUFFSIZE];
		ERRloc[0] = 0;							// place empty messages (= no errors)

		MPI_Scatterv(POPdist, countPOP.data(), displPOP.data(), MPI_DOUBLE, POPloc, num_mod*actdim, MPI_DOUBLE, 0, comm);
		if (rank == 0)
		{
			delete [] POPdist;
			POPdist = 0;
		}
	}

	// ********************************
	// MAIN LOOP: sync over PM_comm
	PM->SavePMState();
	try
	{
		for (int i = ind0; i < ind1; i++)		// ATTENTION: the loop may have different length on different ranks!
		{										// i-ind0 = [0...num_mod)
			std::vector<double> actparams;
			if (comm != MPI_COMM_NULL)
				actparams = std::vector<double>(POPloc + (i-ind0)*actdim, POPloc + (i-ind0+1)*actdim);	// first define actparams on PM_comm-ranks-0
			HMMPI::Bcast_vector(actparams, 0, PM_comm);										// then Bcast to other PM_comm ranks

			double objfunc = 0;
			if (quiet_out_of_range && !PM->CheckLimits_ACT(actparams))
				objfunc = std::numeric_limits<double>::quiet_NaN();
			else
				objfunc = PM->ObjFunc_ACT(actparams);		// not "this->ObjFunc_ACT(actparams)" -- so PhysModGradNum::of_val and PhysModGradNum::par will not change

			if (comm != MPI_COMM_NULL)						// gather results (these ranks are the same as PM_comm-ranks-0)
			{
				FITloc[i-ind0] = objfunc;
				if (smry_present)
					memcpy(SMRYloc + (i-ind0)*smry_len, PM->ModelledData().data(), smry_len*sizeof(double));
			}
		}
	}
	catch (std::exception &e)
	{
		if (comm != MPI_COMM_NULL)														// only PM_comm-ranks-0 exceptions will pass their messages
			sprintf(ERRloc, "[%d] %1.*s", RNK, HMMPI::BUFFSIZE-50, e.what());			// print the message to ERRloc
	}
	PM->RestorePMState();

	// ********************************
	// pack results back to FIT, SMRY
	if (comm != MPI_COMM_NULL)
	{
		MPI_Gatherv(FITloc, num_mod, MPI_DOUBLE, FIT, countFIT.data(), displFIT.data(), MPI_DOUBLE, 0, comm);
		if (smry_present)
			MPI_Gatherv(SMRYloc, num_mod*smry_len, MPI_DOUBLE, SMRYdist, countSMRY.data(), displSMRY.data(), MPI_DOUBLE, 0, comm);
		delete [] FITloc;
		delete [] SMRYloc;
		delete [] POPloc;
	}
	if (rank == 0)
	{
		if (smry_present)
		{
			for (int i = 0; i < len; i++)
				memcpy(SMRY[i], SMRYdist + i*smry_len, smry_len*sizeof(double));
		}
		delete [] SMRYdist;
	}

	// ********************************
	// check if any rank has an error message: this is done in the end to free all arrays first
	bool error_found = false;
	std::string error_msg = "";
	if (comm != MPI_COMM_NULL)
	{
		MPI_Allgather(ERRloc, HMMPI::BUFFSIZE, MPI_CHAR, ERRdist, HMMPI::BUFFSIZE, MPI_CHAR, comm);		// gather messages from all comm-ranks; ERRdist is sync on 'comm'
		for (int j = 0; j < size; j++)
		{
			std::string msg = ERRdist + j*HMMPI::BUFFSIZE;
			if (msg != "")
			{
				error_found = true;
				if (error_msg != "")
					error_msg += "\n";
				error_msg += msg;
			}
		}
		delete [] ERRdist;
		delete [] ERRloc;
	}
	MPI_Bcast(&error_found, 1, MPI_BYTE, 0, PM_comm);	// error_found was sync on 'comm', now it's sync on 'PM_comm'
	if (error_found)
		throw HMMPI::Exception(error_msg);				// synchronous (PM_comm) exception
}
//---------------------------------------------------------------------------
double PhysModMPI::ObjFunc(const std::vector<double> &params)
{
	throw HMMPI::Exception("Запрещенный вызов ObjFunc", "Illegal call to ObjFunc");
}
//---------------------------------------------------------------------------
double PhysModMPI::ObjFunc_ACT(const std::vector<double> &params)
{
	double of_val = 0;
	if (PM->GetComm() != MPI_COMM_NULL)
	{
		of_val = PM->ObjFunc_ACT(params);				// valid on PM->comm-RANKS-0 -- i.e. all not-NULL comm-ranks

		int rank_pm;
		MPI_Comm_rank(PM->GetComm(), &rank_pm);
		if (rank_pm == 0)
			modelled_data = PM->ModelledData();
	}

	if (comm != MPI_COMM_NULL)							// PM->ObjFunc_ACT results on comm-ranks-1,2... are redundant as only result from comm-ranks-0 is used
	{
		MPI_Bcast(&of_val, 1, MPI_DOUBLE, 0, comm);		// Bcast from comm-ranks-0
		HMMPI::Bcast_vector(modelled_data, 0, comm);
	}

	return of_val;
}
//---------------------------------------------------------------------------
int PhysModMPI::ParamsDim() const noexcept
{
	return PM->ParamsDim();
}
//---------------------------------------------------------------------------
bool PhysModMPI::CheckLimits(const std::vector<double> &params) const
{
	bool res = PM->CheckLimits(params);
	limits_msg = PM->limits_msg;

	return res;
}
//---------------------------------------------------------------------------
size_t PhysModMPI::ModelledDataSize() const
{
	return PM->ModelledDataSize();
}
//---------------------------------------------------------------------------
// PhysModGradNum
//---------------------------------------------------------------------------
HMMPI::Vector2<double> PhysModGradNum::make_coeff(size_t dim, std::string fd)
{
	HMMPI::Vector2<double> coeff(dim, 9, 0.0);

	if (fd == "OH1")
	{
		for (size_t i = 0; i < dim; i++)
		{
			coeff(i, 4) = -1;
			coeff(i, 5) = 1;
		}
	}
	else if (fd == "OH2")
	{
		for (size_t i = 0; i < dim; i++)
		{
			coeff(i, 3) = -0.5;
			coeff(i, 5) = 0.5;
		}
	}
	else if (fd == "OH4")
	{
		for (size_t i = 0; i < dim; i++)
		{
			coeff(i, 2) = 1.0/12.0;
			coeff(i, 3) = -8.0/12.0;
			coeff(i, 5) = 8.0/12.0;
			coeff(i, 6) = -1.0/12.0;
		}
	}
	else if (fd == "OH8")
	{
		for (size_t i = 0; i < dim; i++)
		{
			coeff(i, 0) = 1.0/280.0;
			coeff(i, 1) = -4.0/105.0;
			coeff(i, 2) = 1.0/5.0;
			coeff(i, 3) = -4.0/5.0;
			coeff(i, 5) = 4.0/5.0;
			coeff(i, 6) = -1.0/5.0;
			coeff(i, 7) = 4.0/105.0;
			coeff(i, 8) = -1.0/280.0;
		}
	}
	else
		throw HMMPI::Exception("Некорректная формула конечных разностей в PhysModGradNum::make_coeff",
							   "Incorrect finite difference formula in PhysModGradNum::make_coeff");
	return coeff;
}
//---------------------------------------------------------------------------
HMMPI::Vector2<int> PhysModGradNum::make_ind(size_t dim, std::string fd)
{
	HMMPI::Vector2<int> ind(dim, 9, 0);

	if (fd == "OH1")
	{
		for (size_t i = 0; i < dim; i++)
		{
			ind(i, 4) = 0;
			ind(i, 5) = i+1;
		}
	}
	else if (fd == "OH2")
	{
		for (size_t i = 0; i < dim; i++)
		{
			ind(i, 3) = i;
			ind(i, 5) = i + dim;
		}
	}
	else if (fd == "OH4")
	{
		for (size_t i = 0; i < dim; i++)
		{
			ind(i, 2) = i;
			ind(i, 3) = i + dim;
			ind(i, 5) = i + 2*dim;
			ind(i, 6) = i + 3*dim;
		}
	}
	else if (fd == "OH8")
	{
		for (size_t i = 0; i < dim; i++)
		{
			ind(i, 0) = i;
			ind(i, 1) = i + dim;
			ind(i, 2) = i + 2*dim;
			ind(i, 3) = i + 3*dim;
			ind(i, 5) = i + 4*dim;
			ind(i, 6) = i + 5*dim;
			ind(i, 7) = i + 6*dim;
			ind(i, 8) = i + 7*dim;
		}
	}
	else
		throw HMMPI::Exception("Некорректная формула конечных разностей в PhysModGradNum::make_ind",
							   "Incorrect finite difference formula in PhysModGradNum::make_ind");

	return ind;
}
//---------------------------------------------------------------------------
PhysModGradNum::PhysModGradNum(MPI_Comm c, PhysModel *pm, Parser_1 *K, KW_item *kw) : PhysModMPI(c, pm), of_val(-1)
{
	DECLKWD(opt, KW_optimization, "OPTIMIZATION");
	//DECLKWD(limits, KW_limits, "LIMITS");
	//DECLKWD(params, KW_parameters, "PARAMETERS");
	//const ParamsInterface *par_interface = params->GetParamsInterface();

	const ParamsInterface *par_interface = dynamic_cast<const ParamsInterface*>(PM->GetConstr());

	name = "PhysModGradNum";

	kw->Start_pre();
	kw->Add_pre("OPTIMIZATION");
	kw->Finish_pre();

	init = par_interface->init;
	act_ind = par_interface->get_act_ind();
	tot_ind = par_interface->get_tot_ind();
	con = par_interface;

	fin_diff = opt->fin_diff;
	const KW_limits *limits = dynamic_cast<const KW_limits*>(par_interface);
	if (limits != nullptr)			// LIMITS[KRIG] case
	{
		dh = limits->dh;
		dh_type = limits->dh_type;
		std::string msg = limits->CheckPositive(limits->dh, "dh");
		if (msg != "")
			throw HMMPI::Exception(msg);

		K->AppText((std::string)HMMPI::MessageRE("NUMGRAD: dh, dh_type взяты из ", "NUMGRAD: dh, dh_type are taken from ") + limits->name + "\n");
	}
	else							// PARAMETERS case
	{
		//const double const_dh = 1e-5;		// TODO original
		//const double const_dh = 1e-4;
		const double const_dh = 1e-6;
		dh = std::vector<double>(init.size(), const_dh);
		dh_type = std::vector<std::string>(init.size(), "CONST");
		K->AppText(HMMPI::stringFormatArr("NUMGRAD: взяты dh = {0}, dh_type = CONST\n", "NUMGRAD: using dh = {0}, dh_type = CONST\n", const_dh));
	}

#ifdef TESTCTOR
	std::ofstream testf(HMMPI::stringFormatArr("TESTCTOR_out_{0:%d}.txt", std::vector<int>{RNK}), std::ios::app);
	testf << "rank " << RNK << ", PhysModGradNum easy CTOR, this = " << this << "\n";
	testf.close();
#endif
}
//---------------------------------------------------------------------------
PhysModGradNum::~PhysModGradNum()
{
#ifdef TESTCTOR
	std::ofstream testf(HMMPI::stringFormatArr("TESTCTOR_out_{0:%d}.txt", std::vector<int>{RNK}), std::ios::app);
	testf << "rank " << RNK << ", PhysModGradNum -- DTOR --, this = " << this << "\n";
	testf.close();
#endif
}
//---------------------------------------------------------------------------
double PhysModGradNum::ObjFunc_ACT(const std::vector<double> &params)
{
	if (PM->GetComm() != MPI_COMM_NULL)
	{
		of_val = PM->ObjFunc_ACT(params);	// valid on PM->comm-RANKS-0

		int rank_pm;
		MPI_Comm_rank(PM->GetComm(), &rank_pm);
		if (rank_pm == 0)
			modelled_data = PM->ModelledData();
	}
	par = params;

	if (comm != MPI_COMM_NULL)							// PM->ObjFunc_ACT results on comm-ranks-1,2... are redundant as only result from comm-ranks-0 is used
	{
		MPI_Bcast(&of_val, 1, MPI_DOUBLE, 0, comm);		// Bcast from comm-ranks-0
		HMMPI::Bcast_vector(modelled_data, 0, comm);
		HMMPI::Bcast_vector(par, 0, comm);
	}

	return of_val;
}
//---------------------------------------------------------------------------
std::vector<double> PhysModGradNum::ObjFuncGrad_ACT(const std::vector<double> &params)
{
	int rank = -1;
	if (comm != MPI_COMM_NULL)
		MPI_Comm_rank(comm, &rank);

	size_t dim = params.size(), Nmod = 0;		// dim - active params dimension
	double **POP = 0;
	double *FIT = 0;
	double **SENS = 0;
	int smry_len = 0;
	std::vector<int> IND9{8, 0, 7, 1, 6, 2, 5, 3, 4};
	HMMPI::Vector2<double> coeff;
	HMMPI::Vector2<int> ind;

	if (fin_diff == "OH1")
		Nmod = dim + 1;				// no check for par == params, since "of_val" is not used in this function
	else if (fin_diff == "OH2")
		Nmod = 2*dim;
	else if (fin_diff == "OH4")
		Nmod = 4*dim;
	else if (fin_diff == "OH8")
		Nmod = 8*dim;
	else
		throw HMMPI::Exception("Некорректная формула конечных разностей в PhysModGradNum::ObjFuncGrad_ACT",
							   "Incorrect finite difference formula in PhysModGradNum::ObjFuncGrad_ACT");
	if (rank == 0)
	{
		coeff = make_coeff(dim, fin_diff);
		ind = make_ind(dim, fin_diff);

		POP = new double*[Nmod];
		FIT = new double[Nmod];
		for (size_t i = 0; i < Nmod; i++)
			POP[i] = new double[dim];

		smry_len = PM->ModelledDataSize();
		SENS = new double*[Nmod];
		for (size_t i = 0; i < Nmod; i++)
			SENS[i] = new double[smry_len];				// this line is executed even for smry_len == 0

		for (size_t i = 0; i < dim; i++)
		{
			for (size_t j = 0; j < Nmod; j++)
				POP[j][i] = params[i];

			double hi = dh[i];
			if (dh_type[i] == "LIN")
				hi *= fabs(params[i]);

			for (int k = 0; k < 9; k++)
				if (coeff(i, k) != 0)
					POP[ind(i, k)][i] += hi*(k-4);
		}
	}

	if (comm != MPI_COMM_NULL)
		MPI_Bcast(&smry_len, 1, MPI_INT, 0, comm);
	ObjFuncMPI_ACT(Nmod, POP, FIT, false, SENS);		// should be called on MPI_COMM_WORLD

	std::vector<double> GRAD(dim);	// result

	if (rank == 0)
	{
		DataSens = HMMPI::Mat(smry_len, dim, 0.0);

		for (size_t i = 0; i < dim; i++)
		{
			double hi = dh[i];
			if (dh_type[i] == "LIN")
				hi *= fabs(params[i]);

			for (int k0 = 0; k0 < 9; k0++)
			{
				double aux_sum = 0;
				int k = IND9[k0];
				if (coeff(i, k) != 0)
					aux_sum += FIT[ind(i, k)] * coeff(i, k) / hi;

				if (k0 < 8)
				{
					k0++;
					k = IND9[k0];
					if (coeff(i, k) != 0)
						aux_sum += FIT[ind(i, k)] * coeff(i, k) / hi;
				}

				GRAD[i] += aux_sum;
			}

			// sensitivities
			for (int d = 0; d < smry_len; d++)
				for (int k0 = 0; k0 < 9; k0++)
				{
					double aux_sum = 0;
					int k = IND9[k0];
					if (coeff(i, k) != 0)
						aux_sum += SENS[ind(i, k)][d] * coeff(i, k) / hi;

					if (k0 < 8)
					{
						k0++;
						k = IND9[k0];
						if (coeff(i, k) != 0)
							aux_sum += SENS[ind(i, k)][d] * coeff(i, k) / hi;
					}

					DataSens(d, i) += aux_sum;
				}
		}

		// clean the memory
		for (size_t i = 0; i < Nmod; i++)
		{
			delete [] POP[i];
			if (SENS != 0)
				delete [] SENS[i];
		}
		delete [] POP;
		delete [] FIT;
		if (SENS != 0)
			delete [] SENS;
	}

	if (comm != MPI_COMM_NULL)
	{
		HMMPI::Bcast_vector(GRAD, 0, comm);
		DataSens.Bcast(0, comm);
	}

	return GRAD;
}
//---------------------------------------------------------------------------
double PhysModGradNum::ObjFuncGradDir_ACT(const std::vector<double> &params, const std::vector<double> &dir)
{
	int rank = -1;
	if (comm != MPI_COMM_NULL)
		MPI_Comm_rank(comm, &rank);

	size_t dim = params.size(), Nmod = 0;
	double **POP = 0;
	double *FIT = 0;
	std::vector<int> IND9{8, 0, 7, 1, 6, 2, 5, 3, 4};
	std::vector<double> coeff(9);
	std::vector<int> ind(9);
	double h = std::numeric_limits<double>::max();

	if (fin_diff == "OH1")
	{
		Nmod = 1;
		if (par.size() != params.size() || !std::equal(par.begin(), par.end(), params.begin()))
			throw HMMPI::Exception("fin_diff == OH1, но последний вызов ObjFunc_ACT был с другими параметрами",
								   "fin_diff == OH1, but the last call of ObjFunc_ACT used other parameters");
	}
	else if (fin_diff == "OH2")
		Nmod = 2;
	else if (fin_diff == "OH4")
		Nmod = 4;
	else if (fin_diff == "OH8")
		Nmod = 8;
	else
		throw HMMPI::Exception("Некорректная формула конечных разностей в PhysModGradNum::ObjFuncGradDir_ACT",
							   "Incorrect finite difference formula in PhysModGradNum::ObjFuncGradDir_ACT");

	if (rank == 0)	// master process -- prepare data
	{
		// create arrays
		for (size_t k = 0; k < 9; k++)
		{
			coeff[k] = 0;
			ind[k] = 0;
		}

		POP = new double*[Nmod];
		FIT = new double[Nmod];
		for (size_t i = 0; i < Nmod; i++)
			POP[i] = new double[dim];

		// fill arrays
		if (fin_diff == "OH1")
		{
			coeff[5] = 1;
			ind[5] = 0;
		}
		else if (fin_diff == "OH2")
		{
			coeff[3] = -0.5;
			coeff[5] = 0.5;
			ind[3] = 0;
			ind[5] = 1;
		}
		else if (fin_diff == "OH4")
		{
			coeff[2] = 1.0/12.0;
			coeff[3] = -8.0/12.0;
			coeff[5] = 8.0/12.0;
			coeff[6] = -1.0/12.0;
			ind[2] = 0;
			ind[3] = 1;
			ind[5] = 2;
			ind[6] = 3;
		}
		else // fin_diff == "OH8"
		{
			coeff[0] = 1.0/280.0;
			coeff[1] = -4.0/105.0;
			coeff[2] = 1.0/5.0;
			coeff[3] = -4.0/5.0;
			coeff[5] = 4.0/5.0;
			coeff[6] = -1.0/5.0;
			coeff[7] = 4.0/105.0;
			coeff[8] = -1.0/280.0;
			ind[0] = 0;
			ind[1] = 1;
			ind[2] = 2;
			ind[3] = 3;
			ind[5] = 4;
			ind[6] = 5;
			ind[7] = 6;
			ind[8] = 7;
		}

		// find h
		for (size_t i = 0; i < dim; i++)
		{
			double hi = dh[i];
			if (dh_type[i] == "LIN")
				hi *= fabs(params[i]);				// hi is the desirable increment for parameter i
			if (dir[i] != 0)						// h*dir[i] is the real increment for parameter i
			{
				double rat = hi/fabs(dir[i]);		// take h such that
				if (rat < h)						// |h*dir[i]| <= hi, for all i
					h = rat;
			}
		}

		// fill the models
		for (size_t i = 0; i < dim; i++)
		{
			for (size_t j = 0; j < Nmod; j++)
				POP[j][i] = params[i];

			for (int k = 0; k < 9; k++)
				if (coeff[k] != 0)
					POP[ind[k]][i] += h*(k-4)*dir[i];
		}
	}

	// calculate objective functions
	ObjFuncMPI_ACT(Nmod, POP, FIT);					// called on MPI_COMM_WORLD

	// gather results, find gradient
	double GRAD = 0;

	if (rank == 0)
	{
		if (fin_diff == "OH1")
			GRAD = -of_val/h;
		for (int k0 = 0; k0 < 9; k0++)
		{
			double aux_sum = 0;
			int k = IND9[k0];
			if (coeff[k] != 0)
				aux_sum += FIT[ind[k]] * coeff[k] / h;

			if (k0 < 8)
			{
				k0++;
				int k = IND9[k0];
				if (coeff[k] != 0)
					aux_sum += FIT[ind[k]] * coeff[k] / h;
			}

			GRAD += aux_sum;
		}

		// clean the memory
		for (size_t i = 0; i < Nmod; i++)
			delete [] POP[i];

		delete [] POP;
		delete [] FIT;
	}

	if (comm != MPI_COMM_NULL)
		MPI_Bcast(&GRAD, 1, MPI_DOUBLE, 0, comm);		// broadcast to "comm"

	return GRAD;
}
//---------------------------------------------------------------------------
HMMPI::Mat PhysModGradNum::ObjFuncHess_ACT(const std::vector<double> &params)
{
	int rank = -1;
	if (comm != MPI_COMM_NULL)
		MPI_Comm_rank(comm, &rank);

	size_t dim = params.size(), Nmod = 0;
	std::vector<int> IND9{8, 0, 7, 1, 6, 2, 5, 3, 4};
	HMMPI::Vector2<double> coeff;
	HMMPI::Vector2<int> ind;

	HMMPI::Mat DataSensSave = std::move(DataSens);		// save DataSens because it will be overwritten by gradient calculations

	if (fin_diff == "OH1")
		Nmod = dim + 1;				// no check for par == params, since "of_val" is not used in this function
	else if (fin_diff == "OH2")
		Nmod = 2*dim;
	else if (fin_diff == "OH4")
		Nmod = 4*dim;
	else if (fin_diff == "OH8")
		Nmod = 8*dim;
	else
		throw HMMPI::Exception("Некорректная формула конечных разностей в PhysModGradNum::ObjFuncHess_ACT",
							   "Incorrect finite difference formula in PhysModGradNum::ObjFuncHess_ACT");

	coeff = make_coeff(dim, fin_diff);
	ind = make_ind(dim, fin_diff);

	std::vector<std::vector<double>> POP(Nmod);
	std::vector<std::vector<double>> FIT(Nmod);		// will store gradients
	for (size_t i = 0; i < Nmod; i++)
		POP[i] = std::vector<double>(dim);

	for (size_t i = 0; i < dim; i++)
	{
		for (size_t j = 0; j < Nmod; j++)
			POP[j][i] = params[i];

		double hi = dh[i];
		if (dh_type[i] == "LIN")
			hi *= fabs(params[i]);

		for (int k = 0; k < 9; k++)
			if (coeff(i, k) != 0)
				POP[ind(i, k)][i] += hi*(k-4);
	}

	// calculate gradients for POP
	for (size_t j = 0; j < Nmod; j++)
		FIT[j] = ObjFuncGrad_ACT(POP[j]);	// each gradient is calculated via MPI on MPI_COMM_WORLD

	HMMPI::Mat Hess(dim, dim, 0.0);			// result
	if (rank == 0)
	{
		for (size_t i = 0; i < dim; i++)
		{
			double hi = dh[i];
			if (dh_type[i] == "LIN")
				hi *= fabs(params[i]);

			HMMPI::Mat aux_sum(dim, 1, 0.0);
			for (int k0 = 0; k0 < 9; k0++)
			{
				int k = IND9[k0];
				if (coeff(i, k) != 0)
					aux_sum += coeff(i, k) / hi * HMMPI::Mat(FIT[ind(i, k)]);

				if (k0 < 8)
				{
					k0++;
					k = IND9[k0];
					if (coeff(i, k) != 0)
						aux_sum += coeff(i, k) / hi * HMMPI::Mat(FIT[ind(i, k)]);
				}
			}

			for (size_t j = 0; j < dim; j++)
				Hess(i, j) = aux_sum(j, 0);
		}
	}

	if (comm != MPI_COMM_NULL)
		Hess.Bcast(0, comm);

	DataSens = std::move(DataSensSave);

	return Hess;
}
//---------------------------------------------------------------------------
// class PM_LagrangianSpher
//---------------------------------------------------------------------------
PM_LagrangianSpher::PM_LagrangianSpher(PhysModel *pm, double lam) : PhysModel(pm->GetComm()), PM(pm), Hk(0)
{
	const ParamsInterface *pm_con = dynamic_cast<const ParamsInterface*>(pm->GetConstr());
	assert(pm_con != nullptr);
	name = "LagrangianSpher";

	ParamsInterface *par = new ParamsInterface(*pm_con);		// this auxiliary copy will be used as new 'con'; it is deleted in DTOR
	par->Push_point(lam, -std::numeric_limits<double>::infinity(), 0, "A", "lambda");		// NOTE: the range for lambda is (-inf, 0]
	con = par;

	init = par->init;
	act_ind = par->get_act_ind();
	tot_ind = par->get_tot_ind();
}
//---------------------------------------------------------------------------
PM_LagrangianSpher::~PM_LagrangianSpher()
{
#ifdef TESTCTOR
	std::ofstream testf(HMMPI::stringFormatArr("TESTCTOR_out_{0:%d}.txt", std::vector<int>{RNK}), std::ios::app);
	testf << "rank " << RNK << ", PM_LagrangianSpher -- DTOR --, this = " << this << "\n";
	testf.close();
#endif
	delete con;
}
//---------------------------------------------------------------------------
double PM_LagrangianSpher::ObjFunc(const std::vector<double> &params)
{
	std::vector<double> params0(params.begin(), --params.end());
	double lambda = *--params.end();

	HMMPI::Mat x = params0;
	x -= x0;

	double of = PM->ObjFunc(params0);
	return of - lambda*(InnerProd(x, x) - Hk*Hk);
}
//---------------------------------------------------------------------------
std::vector<double> PM_LagrangianSpher::ObjFuncGrad(const std::vector<double> &params)
{
	int rank = -1;
	if (comm != MPI_COMM_NULL)
		MPI_Comm_rank(comm, &rank);

	std::vector<double> params0(params.begin(), --params.end());
	double lambda = *--params.end();

	HMMPI::Mat x = params0;
	x -= x0;

	HMMPI::Mat res0 = PM->ObjFuncGrad(params0);
	if (rank == 0)
		res0 -= (2*lambda)*x;		// on other ranks Mat sizes may be incompatible

	std::vector<double> res = res0.ToVector();
	res.push_back(-InnerProd(x, x) + Hk*Hk);

	return res;
}
//---------------------------------------------------------------------------
HMMPI::Mat PM_LagrangianSpher::ObjFuncHess(const std::vector<double> &params)
{
	int rank = -1;
	if (comm != MPI_COMM_NULL)
		MPI_Comm_rank(comm, &rank);

	std::vector<double> params0(params.begin(), --params.end());
	double lambda2 = *--params.end() * 2;

	HMMPI::Mat x = params0;
	x -= x0;
	x = -2*std::move(x);

	HMMPI::Mat res = PM->ObjFuncHess(params0);
	if (rank == 0)
		for (size_t i = 0; i < params0.size(); i++)
			res(i, i) -= lambda2;

	if (rank == 0)
		return (res && x)||(x.Tr() && HMMPI::Mat(1, 1, 0.0));
	else
		return HMMPI::Mat();
}
//---------------------------------------------------------------------------
bool PM_LagrangianSpher::FindIntersect(const std::vector<double> &x0, const std::vector<double> &x1, std::vector<double> &xint, double &alpha, int &i) const
{
	throw HMMPI::Exception("Illegal call to PM_LagrangianSpher::FindIntersect");
}
//---------------------------------------------------------------------------
bool PM_LagrangianSpher::FindIntersect_ACT(const std::vector<double> &x0, const std::vector<double> &x1, std::vector<double> &xint, double &alpha, int &i) const
{
	throw HMMPI::Exception("Illegal call to PM_LagrangianSpher::FindIntersect_ACT");
}
//---------------------------------------------------------------------------
int PM_LagrangianSpher::ParamsDim() const noexcept
{
	return PM->ParamsDim() + 1;
}
//---------------------------------------------------------------------------
size_t PM_LagrangianSpher::ModelledDataSize() const
{
	//return PM->ModelledDataSize();		-- uncomment when 'modelled_data' is taken from PM
	return 0;
}
//---------------------------------------------------------------------------
std::string PM_LagrangianSpher::ObjFuncMsg() const
{
	return PM->ObjFuncMsg();
}
//---------------------------------------------------------------------------
// PM_Spherical
//---------------------------------------------------------------------------
PM_Spherical::PM_Spherical(PhysModel *pm, double R, const std::vector<double> &c, double d) : PhysModel(pm->GetComm()), PM(pm), Sc(R, c), delta(d)
{
	const ParamsInterface *pm_con = dynamic_cast<const ParamsInterface*>(pm->GetConstr());
	assert(pm_con != nullptr);
	name = "Spherical";

	ParamsInterface *par = pm_con->ActToSpherical(Sc, delta);			// deleted in DTOR
	con = par;

	init = par->init;
	act_ind = par->get_act_ind();
	tot_ind = par->get_tot_ind();
}
//---------------------------------------------------------------------------
PM_Spherical::~PM_Spherical()
{
#ifdef TESTCTOR
	std::ofstream testf(HMMPI::stringFormatArr("TESTCTOR_out_{0:%d}.txt", std::vector<int>{RNK}), std::ios::app);
	testf << "rank " << RNK << ", PM_Spherical -- DTOR --, this = " << this << "\n";
	testf.close();
#endif
	delete con;
}
//---------------------------------------------------------------------------
double PM_Spherical::ObjFunc(const std::vector<double> &params)
{
	int rank;
	MPI_Comm_rank(comm, &rank);

	double of = PM->ObjFunc_ACT(Sc.spher_to_cart(params));
	if (rank == 0)
		modelled_data = PM->ModelledData();

	return of;
}
//---------------------------------------------------------------------------
std::vector<double> PM_Spherical::ObjFuncGrad(const std::vector<double> &params)
{
	int rank;
	MPI_Comm_rank(comm, &rank);

	HMMPI::Mat grad = PM->ObjFuncGrad_ACT(Sc.spher_to_cart(params));
	if (rank == 0)
	{
		HMMPI::Mat dxdp = Sc.dxdp(params);
		return (dxdp.Tr()*grad).ToVector();
	}
	else
		return std::vector<double>();
}
//---------------------------------------------------------------------------
HMMPI::Mat PM_Spherical::ObjFuncHess(const std::vector<double> &params)
{
	int rank;
	MPI_Comm_rank(comm, &rank);

	HMMPI::Mat grad = PM->ObjFuncGrad_ACT(Sc.spher_to_cart(params));
	HMMPI::Mat hess = PM->ObjFuncHess_ACT(Sc.spher_to_cart(params));

	if (rank == 0)
	{
		HMMPI::Mat dxdp = Sc.dxdp(params);
		HMMPI::Mat res = dxdp.Tr()*hess*dxdp;
		for (size_t j = 0; j < params.size(); j++)		// add a vector to each column 'j'
		{
			HMMPI::Mat add = Sc.dxdp_k(params, j).Tr() * grad;
			for (size_t i = 0; i < params.size(); i++)
				res(i, j) += add(i, 0);
		}
		return res;
	}
	else
		return HMMPI::Mat();
}
//---------------------------------------------------------------------------
int PM_Spherical::ParamsDim() const noexcept
{
	return Sc.dim-1;
}
//---------------------------------------------------------------------------
size_t PM_Spherical::ModelledDataSize() const
{
	return PM->ModelledDataSize();
}
//---------------------------------------------------------------------------
std::string PM_Spherical::ObjFuncMsg() const
{
	return PM->ObjFuncMsg();
}
//---------------------------------------------------------------------------
// PM_CubeBounds
//---------------------------------------------------------------------------
PM_CubeBounds::PM_CubeBounds(PhysModel *pm, double R0, const std::vector<double> &c0) : PhysModel(pm->GetComm()), PM(pm), R(R0), c(c0)
{
	const ParamsInterface *pm_con = dynamic_cast<const ParamsInterface*>(pm->GetConstr());
	assert(pm_con != nullptr);
	name = "CubeBounds";

	ParamsInterface *par = pm_con->CubeBounds(c, R);			// deleted in DTOR
	con = par;

	init = par->init;
	act_ind = par->get_act_ind();
	tot_ind = par->get_tot_ind();
}
//---------------------------------------------------------------------------
PM_CubeBounds::~PM_CubeBounds()
{
#ifdef TESTCTOR
	std::ofstream testf(HMMPI::stringFormatArr("TESTCTOR_out_{0:%d}.txt", std::vector<int>{RNK}), std::ios::app);
	testf << "rank " << RNK << ", PM_CubeBounds -- DTOR --, this = " << this << "\n";
	testf.close();
#endif
	delete con;
}
//---------------------------------------------------------------------------
double PM_CubeBounds::ObjFunc(const std::vector<double> &params)
{
	int rank;
	MPI_Comm_rank(comm, &rank);

	double of = PM->ObjFunc(params);
	if (rank == 0)
		modelled_data = PM->ModelledData();

	return of;
}
//---------------------------------------------------------------------------
std::vector<double> PM_CubeBounds::ObjFuncGrad(const std::vector<double> &params)
{
	return PM->ObjFuncGrad(params);
}
//---------------------------------------------------------------------------
HMMPI::Mat PM_CubeBounds::ObjFuncHess(const std::vector<double> &params)
{
	return PM->ObjFuncHess(params);
}
//---------------------------------------------------------------------------
int PM_CubeBounds::ParamsDim() const noexcept
{
	return PM->ParamsDim();
}
//---------------------------------------------------------------------------
size_t PM_CubeBounds::ModelledDataSize() const
{
	return PM->ModelledDataSize();
}
//---------------------------------------------------------------------------
std::string PM_CubeBounds::ObjFuncMsg() const
{
	return PM->ObjFuncMsg();
}
//---------------------------------------------------------------------------
// PM_FullHamiltonian
//---------------------------------------------------------------------------
HMMPI::Mat PM_FullHamiltonian::calc_FI_ACT(const std::vector<double> &x) const
{
	G.MsgToFile("calculate FI (MM)\n");				// separation mark for debug logging
	HMMPI::Mat res = PM->ObjFuncFisher_ACT(x);
	res += MM_shift * HMMPI::Mat(res.ICount());

	return res;
}
//---------------------------------------------------------------------------
HMMPI::Mat PM_FullHamiltonian::calc_invFI_ACT(const std::vector<double> &x) const
{
	int rank = -1;
	MPI_Comm_rank(comm, &rank);
	const HMMPI::Mat &MM = G.Get(this, x);
	if (rank == 0)
		return MM.InvSPO();
	else
		return HMMPI::Mat();
}
//---------------------------------------------------------------------------
HMMPI::Mat PM_FullHamiltonian::calc_invU_ACT(const std::vector<double> &x) const				// inv(U), where U'*U = G; results - on rank-0
{
	G.MsgToFile("calculate inv(U), U'*U = G\n");	// separation mark for debug logging

	int rank = -1;
	MPI_Comm_rank(comm, &rank);
	const HMMPI::Mat &MM = G.Get(this, x);
	if (rank == 0)
	{
		HMMPI::Mat U = MM.CholSPO();
		return U.InvU();
	}
	else
		return HMMPI::Mat();
}
//---------------------------------------------------------------------------
HMMPI::Mat PM_FullHamiltonian::calc_dxi_FI_ACT(const std::vector<double> &x, int acti) const
{
	G.MsgToFile("calculate dFI/dx_i\n");			// separation mark for debug logging
	return PM->ObjFuncFisher_dxi_ACT(x, acti, Ranks[acti]);
}
//---------------------------------------------------------------------------
std::vector<double> PM_FullHamiltonian::calc_dx_H1_ACT(const std::vector<double> &x) const		// gradient of the full Hamiltonian w.r.t. coordinates 'x' (only first two terms, which do not depend on momentum); results - on rank-0
{
	// _NOTE_ in this function inversion of the mass matrix takes place several times.
	// While this may be more time consuming than using a pre-calculated inverse matrix,
	// this might be more accurate for the case when mass matrix is badly conditioned
	// (although no tests regarding this were made).

	G.MsgToFile("calculate dH'/dx\n");				// separation mark for debug logging

	int rank;
	MPI_Comm_rank(comm, &rank);

	int actdim = ParamsDim_ACT();
	std::vector<double> res;						// valid on comm-rank-0
	if (rank == 0)
		res = std::vector<double>(actdim, 0);
	std::vector<double> grad = PM->ObjFuncGrad_ACT(x);		// valid results - on comm-rank-0

	HMMPI::Mat MM = G.Get(this, x);					// mass matrix
	MM.Bcast(0, comm);								// sync

	std::vector<HMMPI::Mat> dxi_MM(actdim);
	for (int i = 0; i < actdim; i++)
		dxi_MM[i] = dxi_G[i].Get(this, x);			// all ranks are involved in calling, but results are not on all ranks

	std::vector<double> locres(locend - locstart);	// local vector on each rank
	for (int i = locstart; i < locend; i++)			// now each rank deals with its own indices
		locres[i - locstart] = (MM / dxi_MM[i]).Trace();

	MPI_Gatherv(locres.data(), locres.size(), MPI_DOUBLE, res.data(), nums.data(), starts.data(), MPI_DOUBLE, 0, comm);

	if (rank == 0)
		for (int i = 0; i < actdim; i++)
			res[i] = 0.5*(res[i] + grad[i]);

	return res;
}
//---------------------------------------------------------------------------
std::vector<double> PM_FullHamiltonian::calc_dx_H1_ACT_beta(const std::vector<double> &x) const		// similar to calc_dx_H1_ACT, but has different scalar coeffs; to be used in MMALA
{
	// _NOTE_ in this function inversion of the mass matrix takes place several times.
	// While this may be more time consuming than using a pre-calculated inverse matrix,
	// this might be more accurate for the case when mass matrix is badly conditioned
	// (although no tests regarding this were made).

	G.MsgToFile("run calc_dx_H1_ACT_beta\n");	// separation mark for debug logging

	int rank;
	MPI_Comm_rank(comm, &rank);

	int actdim = ParamsDim_ACT();
	std::vector<double> res;						// valid on comm-rank-0
	if (rank == 0)
		res = std::vector<double>(actdim, 0);
	std::vector<double> grad = PM->ObjFuncGrad_ACT(x);		// valid results - on comm-rank-0

	HMMPI::Mat MM = G.Get(this, x);					// mass matrix
	MM.Bcast(0, comm);								// sync

	std::vector<HMMPI::Mat> dxi_MM(actdim);
	for (int i = 0; i < actdim; i++)
		dxi_MM[i] = dxi_G[i].Get(this, x);			// all ranks are involved in calling, but results are not on all ranks

	std::vector<double> locres(locend - locstart);	// local vector on each rank
	for (int i = locstart; i < locend; i++)			// now each rank deals with its own indices
		locres[i - locstart] = (MM / dxi_MM[i]).Trace();

	MPI_Gatherv(locres.data(), locres.size(), MPI_DOUBLE, res.data(), nums.data(), starts.data(), MPI_DOUBLE, 0, comm);

	if (rank == 0)
		for (int i = 0; i < actdim; i++)
			res[i] = res[i] - 0.5*grad[i];			// HERE is the main difference with PM_FullHamiltonian::calc_dx_H1_ACT

	return res;
}
//---------------------------------------------------------------------------
HMMPI::Mat PM_FullHamiltonian::calc_grad_aux_ACT(const std::pair<std::vector<double>, std::vector<double>> &x_p) const		// returns a comm-sync matrix with i-th row = (G^(-1) * dG/dx_i * G^(-1) * pact)^t, this is used in the subsequent Jacobian calculations
{																															// NOTE x_p = {x, p}, but "p" is not used; before calling, MAKE SURE pact = p is set
	G.MsgToFile("calculate aux grad HAM\n");				// separation mark for debug logging

	int rank;
	MPI_Comm_rank(comm, &rank);
	const size_t actdim = x_p.first.size();
	assert(pact.ToVector() == x_p.second);
	if (pact.ICount() != actdim)
		throw HMMPI::Exception("Size of momentum vector 'pact' does not match the active coordinates dimension in PM_FullHamiltonian::calc_grad_aux_ACT");

	HMMPI::Mat MM = G.Get(this, x_p.first);					// 'MM' is sync on comm
	MM.Bcast(0, comm);
	HMMPI::Mat res_final(actdim, actdim, 0);				// finally will be sync on comm

	std::vector<HMMPI::Mat> dxi_MM(actdim);
	for (size_t i = 0; i < actdim; i++)
		dxi_MM[i] = dxi_G[i].Get(this, x_p.first);			// results valid on Ranks[i]

	HMMPI::Mat invMMp;
	if (rank == 0)
		invMMp = MM/pact;
	invMMp.Bcast(0, comm);									// 'invMMp' is sync on comm

	HMMPI::Mat locres(actdim, locend - locstart, 0);		// local part of the resulting matrix; it will be transposed afterwards
	for (int i = locstart; i < locend; i++)					// now each rank deals with its own indices
	{
		HMMPI::Mat row_i = dxi_MM[i]*invMMp;
		for (size_t j = 0; j < actdim; j++)
			locres(j, i - locstart) = row_i(j, 0);
	}
	locres = (std::move(MM)/std::move(locres)).Tr();

	std::vector<int> numsmult = nums;
	for (auto &v : numsmult)
		v *= actdim;

	std::vector<int> startsmult = starts;
	for (auto &v : startsmult)
		v *= actdim;

	MPI_Gatherv(locres.ToVector().data(), locres.ToVector().size(), MPI_DOUBLE, res_final.ToVectorMutable().data(), numsmult.data(), startsmult.data(), MPI_DOUBLE, 0, comm);
	res_final.Bcast(0, comm);
	return res_final;
}
//---------------------------------------------------------------------------
HMMPI::Mat PM_FullHamiltonian::calc_grad_aux_ACT_alpha(const std::vector<double> &x) const		// similar to calc_grad_aux_ACT; calculates sum_j {G^(-1) * dG/dx_j * G^(-1)}_ij to be used in MMALA; result - on comm-rank-0
{
	G.MsgToFile("run calc_grad_aux_ACT_alpha\n");			// separation mark for debug logging

	int rank;
	MPI_Comm_rank(comm, &rank);
	const size_t actdim = x.size();

	HMMPI::Mat MM = G.Get(this, x);
	MM.Bcast(0, comm);										// 'MM' is sync on comm
	HMMPI::Mat invMM = invG.Get(this, x);
	invMM.Bcast(0, comm);									// 'invMM' is sync on comm

	std::vector<HMMPI::Mat> dxi_MM(actdim);
	for (size_t i = 0; i < actdim; i++)
		dxi_MM[i] = dxi_G[i].Get(this, x);					// results valid on Ranks[i]

	HMMPI::Mat res(actdim, 1, 0.0);
	for (int j = locstart; j < locend; j++)					// now each rank deals with its own indices
	{
		HMMPI::Mat rhs(actdim, 1, 0.0);						// j-th column of invMM
		for (size_t i = 0; i < actdim; i++)
			rhs(i, 0) = invMM(i, j);

		res += MM / (dxi_MM[j]*rhs);
	}

	if (rank == 0)
		MPI_Reduce(MPI_IN_PLACE, res.ToVectorMutable().data(), actdim, MPI_DOUBLE, MPI_SUM, 0, comm);
	else
		MPI_Reduce(res.ToVector().data(), NULL, actdim, MPI_DOUBLE, MPI_SUM, 0, comm);

	return res;
}
//---------------------------------------------------------------------------
std::vector<double> PM_FullHamiltonian::calc_grad_momentum_ACT(const std::pair<std::vector<double>, std::vector<double>> &x_p) const		// dH/dp; x_p = {x, p}, but "p" is not used; before calling, MAKE SURE pact = p is set
{
	G.MsgToFile("calculate dHAM/dp\n");				// separation mark for debug logging

	int rank;
	MPI_Comm_rank(comm, &rank);

	assert(pact.ToVector() == x_p.second);
	if (pact.ICount() != x_p.first.size())
		throw HMMPI::Exception(HMMPI::stringFormatArr("Size of momentum vector 'pact' ({0:%zu}) does not match the active coordinates dimension ({1:%zu}) in PM_FullHamiltonian::calc_grad_momentum_ACT",
													std::vector<size_t>{pact.ICount(), x_p.first.size()}));

	HMMPI::Mat MM = G.Get(this, x_p.first);			// valid on rank-0
	std::vector<double> grad;
	if (rank == 0)
		grad = (std::move(MM)/pact).ToVector();

	return grad;
}
//---------------------------------------------------------------------------
HMMPI::Mat PM_FullHamiltonian::calc_mu(const std::pair<std::vector<double>, double> &x_eps) const		// mu for proposal in MMALA; x_eps = {x, eps}; result - sync on comm
{
	G.MsgToFile("run calc_mu\n");					// separation mark for debug logging

	int rank;
	MPI_Comm_rank(comm, &rank);

	HMMPI::Mat MM = G.Get(this, x_eps.first);						// valid on rank-0
	HMMPI::Mat vbeta = dx_H1_beta.Get(this, x_eps.first);			// valid on rank-0
	if (rank == 0)
		vbeta = std::move(MM) / std::move(vbeta);
	vbeta.Bcast(0, comm);							// sync

	HMMPI::Mat valpha = Gaux_grad_alpha.Get(this, x_eps.first);		// valid on rank-0
	valpha.Bcast(0, comm);							// sync

	double eps2 = x_eps.second*x_eps.second;
	return (-eps2)*std::move(valpha) + (eps2/2)*std::move(vbeta) + HMMPI::Mat(x_eps.first);
}
//---------------------------------------------------------------------------
HMMPI::Mat PM_FullHamiltonian::calc_simpl_mu(const std::pair<std::vector<double>, double> &x_eps) const	// mu for proposal in simplified MMALA; x_eps = {x, eps}; result - sync on comm
{
	G.MsgToFile("run calc_simpl_mu\n");				// separation mark for debug logging

	int rank;
	MPI_Comm_rank(comm, &rank);

	HMMPI::Mat MM = G.Get(this, x_eps.first);				// valid on rank-0
	HMMPI::Mat grad = PM->ObjFuncGrad_ACT(x_eps.first);		// valid on rank-0
	HMMPI::Mat vbeta;

	if (rank == 0)
		vbeta = std::move(MM) / std::move(grad);
	vbeta.Bcast(0, comm);									// sync

	double eps2 = x_eps.second*x_eps.second;
	return (-eps2/4)*std::move(vbeta) + HMMPI::Mat(x_eps.first);
}
//---------------------------------------------------------------------------
HMMPI::Mat PM_FullHamiltonian::calc_mu_2(const std::pair<std::vector<double>, double> &x_eps) const		// mu for proposal in MMALA-2; x_eps = {x, eps}; result - sync on comm
{
	G.MsgToFile("run calc_mu_2\n");					// separation mark for debug logging

	int rank;
	MPI_Comm_rank(comm, &rank);

	HMMPI::Mat MM = G.Get(this, x_eps.first);						// valid on rank-0
	HMMPI::Mat grad = PM->ObjFuncGrad_ACT(x_eps.first);				// valid on rank-0
	HMMPI::Mat vbeta;

	if (rank == 0)
		vbeta = std::move(MM) / std::move(grad);
	vbeta.Bcast(0, comm);											// sync

	HMMPI::Mat valpha = Gaux_grad_alpha.Get(this, x_eps.first);		// valid on rank-0
	valpha.Bcast(0, comm);											// sync

	double eps2 = x_eps.second*x_eps.second;
	return (-eps2/2)*std::move(valpha) + (-eps2/4)*std::move(vbeta) + HMMPI::Mat(x_eps.first);
}
//---------------------------------------------------------------------------
HMMPI::Mat PM_FullHamiltonian::calc_mu_Ifwd(const std::pair<std::vector<double>, std::pair<double, double>> &x_eps_alpha) const		// mu for forward proposal in I_MALA, G=I; x_eps_alpha = {{theta, p}, {eps, alpha}}; for backward proposal - change sign of 'alpha'; result - sync on comm
{
	G.MsgToFile("run calc_mu_Ifwd\n");					// separation mark for debug logging

	const std::vector<double> &x = x_eps_alpha.first;
	const double eps = x_eps_alpha.second.first;
	const double alpha = x_eps_alpha.second.second;

	assert(x.size() % 2 == 0);
	std::vector<double> x0(x.begin(), x.begin() + x.size()/2);		// theta
	std::vector<double> x1(x.begin() + x.size()/2, x.end());		// p

	HMMPI::Mat grad = PM->ObjFuncGrad_ACT(x0);			// valid on rank-0
	grad.Bcast(0, comm);

	HMMPI::Mat v0 = grad + alpha*HMMPI::Mat(x1);
	HMMPI::Mat v1 = -alpha*grad + HMMPI::Mat(x1);

	return HMMPI::Mat(x) - (eps*eps/2)*(v0 || v1);
}
//---------------------------------------------------------------------------
PM_FullHamiltonian::PM_FullHamiltonian(PhysModel *pm, double mm_nu) : PhysModel(pm->GetComm()), MM_shift(mm_nu), PM(pm),
		G(&PM_FullHamiltonian::calc_FI_ACT), invG(&PM_FullHamiltonian::calc_invFI_ACT), invU(&PM_FullHamiltonian::calc_invU_ACT),
		dx_H1(&PM_FullHamiltonian::calc_dx_H1_ACT), dx_H1_beta(&PM_FullHamiltonian::calc_dx_H1_ACT_beta),
		Gaux_grad(&PM_FullHamiltonian::calc_grad_aux_ACT), Gaux_grad_alpha(&PM_FullHamiltonian::calc_grad_aux_ACT_alpha), dHdp(&PM_FullHamiltonian::calc_grad_momentum_ACT),
		mu_MMALA(&PM_FullHamiltonian::calc_mu), mu_simplMMALA(&PM_FullHamiltonian::calc_simpl_mu), mu_MMALA_2(&PM_FullHamiltonian::calc_mu_2),
		mu_Ifwd(&PM_FullHamiltonian::calc_mu_Ifwd)
{
	name = "FullHamiltonian";
	int actdim = PM->ParamsDim_ACT();

	const ParamsInterface *pm_con = dynamic_cast<const ParamsInterface*>(pm->GetConstr());
	assert(pm_con != nullptr);

	ParamsInterface *par = new ParamsInterface(*pm_con);			// deleted in DTOR
	con = par;
	init = par->init;
	act_ind = par->get_act_ind();
	tot_ind = par->get_tot_ind();

	dxi_G.resize(actdim);
	for (int i = 0; i < actdim; i++)
		dxi_G[i] = HMMPI::Cache<PM_FullHamiltonian, std::vector<double>, HMMPI::Mat>(std::bind(&PM_FullHamiltonian::calc_dxi_FI_ACT, std::placeholders::_1, std::placeholders::_2, i));

	// fill the MPI index arrays
	int rank, size, locnum;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);
	locnum = actdim/size + (rank < actdim % size);

	MPI_Scan(&locnum, &locstart, 1, MPI_INT, MPI_SUM, comm);
	if (rank == 0)
	{
		nums.resize(size);
		starts.resize(size+1);
		starts[0] = 0;
		Ranks = std::vector<int>(actdim, -1);
	}
	MPI_Gather(&locnum, 1, MPI_INT, nums.data(), 1, MPI_INT, 0, comm);
	MPI_Gather(&locstart, 1, MPI_INT, starts.data() + 1, 1, MPI_INT, 0, comm);

	if (rank == 0)
	{
		if (*--starts.end() != actdim)
			throw HMMPI::Exception("*--starts.end() != actdim in PM_FullHamiltonian::PM_FullHamiltonian");
		for (int r = 0; r < size; r++)
			for (int i = starts[r]; i < starts[r+1]; i++)
			{
				if (i < 0 || i >= actdim)
					throw HMMPI::Exception("Active parameter index 'i' out of range in PM_FullHamiltonian::PM_FullHamiltonian");
				Ranks[i] = r;
			}
	}
	HMMPI::Bcast_vector(Ranks, 0, comm);

	locend = locstart;
	locstart -= locnum;
}
//---------------------------------------------------------------------------
PM_FullHamiltonian::~PM_FullHamiltonian()
{
#ifdef TESTCTOR
	std::ofstream testf(HMMPI::stringFormatArr("TESTCTOR_out_{0:%d}.txt", std::vector<int>{RNK}), std::ios::app);
	testf << "rank " << RNK << ", PM_FullHamiltonian -- DTOR --, this = " << this << "\n";
	testf.close();
#endif
	delete con;
}
//---------------------------------------------------------------------------
const PM_FullHamiltonian &PM_FullHamiltonian::operator=(const PM_FullHamiltonian &H)
{
	if (this != &H)
	{
		delete con;					// NOTE this deletion goes first, because operator= will override 'con'
		PhysModel::operator=(H);

		nums = H.nums;
		starts = H.starts;
		Ranks = H.Ranks;
		locstart = H.locstart;
		locend = H.locend;
		MM_shift = H.MM_shift;
		PM = H.PM;

		const ParamsInterface *pm_con = dynamic_cast<const ParamsInterface*>(H.con);
		assert(pm_con != nullptr);
		con = new ParamsInterface(*pm_con);			// deleted in DTOR

		pact = H.pact;
		G = H.G;
		invG = H.invG;
		invU = H.invU;
		dxi_G = H.dxi_G;
		dx_H1 = H.dx_H1;
		dx_H1_beta = H.dx_H1_beta;
		Gaux_grad = H.Gaux_grad;
		Gaux_grad_alpha = H.Gaux_grad_alpha;
		dHdp = H.dHdp;
		mu_MMALA = H.mu_MMALA;
		mu_simplMMALA = H.mu_simplMMALA;
		mu_MMALA_2 = H.mu_MMALA_2;
	}

	return *this;
}
//---------------------------------------------------------------------------
double PM_FullHamiltonian::ObjFunc(const std::vector<double> &params)
{
	throw HMMPI::Exception("Запрещенный вызов PM_FullHamiltonian::ObjFunc", "Illegal call to PM_FullHamiltonian::ObjFunc");
}
//---------------------------------------------------------------------------
double PM_FullHamiltonian::ObjFunc_ACT(const std::vector<double> &params)
{
	G.MsgToFile("calculate HAM\n");				// separation mark for debug logging

	if (pact.ICount() != params.size())
		throw HMMPI::Exception("Size of momentum vector 'pact' does not match the active coordinates dimension in PM_FullHamiltonian::ObjFunc_ACT");

	int rank = -1, sign = 0;
	double lndet = 0, of;						// valid only on comm-ranks-0
	MPI_Comm_rank(comm, &rank);

	const HMMPI::Mat &MM = G.Get(this, params);	// MM is only valid on comm-ranks-0
	of = PM->ObjFunc_ACT(params);				// of is only valid on comm-ranks-0
	if (rank == 0)
		lndet = MM.LnDetSY(sign);
	MPI_Bcast(&sign, 1, MPI_INT, 0, comm);
	if (sign <= 0)
		throw HMMPI::Exception("Non-positive determinant for mass matrix in PM_FullHamiltonian::ObjFunc_ACT");

	double res = 0;
	if (rank == 0)
		res = 0.5*(of + lndet + InnerProd(pact, MM/pact));

	return res;
}
//---------------------------------------------------------------------------
std::vector<double> PM_FullHamiltonian::ObjFuncGrad_ACT(const std::vector<double> &params)
{
	G.MsgToFile("calculate dHAM/dx\n");						// separation mark for debug logging

	int rank;
	MPI_Comm_rank(comm, &rank);
	if (pact.ICount() != params.size())
		throw HMMPI::Exception("Size of momentum vector 'pact' does not match the active coordinates dimension in PM_FullHamiltonian::ObjFuncGrad_ACT");

	HMMPI::Mat MM = G.Get(this, params);					// results valid on comm-rank-0
	std::vector<double> res1 = dx_H1.Get(this, params);		// results valid on comm-rank-0
	std::vector<double> res_final;							// valid on comm-rank-0
	if (rank == 0)
		res_final = std::vector<double>(params.size(), 0);

	std::vector<HMMPI::Mat> dxi_MM(params.size());
	for (size_t i = 0; i < dxi_MM.size(); i++)
		dxi_MM[i] = dxi_G[i].Get(this, params);				// results valid on Ranks[i]

	HMMPI::Mat invMMp;
	if (rank == 0)
		invMMp = std::move(MM)/pact;
	invMMp.Bcast(0, comm);

	std::vector<double> locres(locend - locstart);			// local vector on each rank
	for (int i = locstart; i < locend; i++)					// now each rank deals with its own indices
		locres[i - locstart] = -0.5*InnerProd(invMMp, dxi_MM[i]*invMMp);

	MPI_Gatherv(locres.data(), locres.size(), MPI_DOUBLE, res_final.data(), nums.data(), starts.data(), MPI_DOUBLE, 0, comm);
	if (rank == 0)
		for (size_t i = 0; i < params.size(); i++)
			res_final[i] += res1[i];

	return res_final;
}
//---------------------------------------------------------------------------
double PM_FullHamiltonian::MMALA_logQ_ACT(const HMMPI::Mat &x, const HMMPI::Mat &xnew, double eps, const HMMPI::Cache<PM_FullHamiltonian, std::pair<std::vector<double>, double>, HMMPI::Mat> &mu_mmala)
{																										// ln {q(xnew | x, eps)}, where q(.) is the MMALA (simplMMALA, MMALA-2) proposal Gaussian pdf; result - sync on comm
	assert(x.ICount() == xnew.ICount());																// mu_mmala can be mu_MMALA, mu_simplMMALA, mu_MMALA_2 from the same object that called this function
	assert(x.JCount() == xnew.JCount() && x.JCount() == 1);
	const size_t actdim = x.ICount();

	int rank = -1;
	MPI_Comm_rank(comm, &rank);

	const HMMPI::Mat &MM = G.Get(this, x.ToVector());													// rank-0
	HMMPI::Mat mu = mu_mmala.Get(this, std::pair<std::vector<double>, double>(x.ToVector(), eps));		// sync
	HMMPI::Mat diff = mu - xnew;

	HMMPI::Mat MMdiff;
	double lndet;
	G.MsgToFile("calculate det(G)\n");			// separation mark for debug logging
	if (rank == 0)
	{
		MMdiff = MM*diff;
		lndet = MM.LnDetSPO();
	}
	MMdiff.Bcast(0, comm);
	MPI_Bcast(&lndet, 1, MPI_DOUBLE, 0, comm);

	return 0.5*lndet - double(actdim)*log(eps) - 1/(2*eps*eps)*InnerProd(diff, MMdiff);
}
//---------------------------------------------------------------------------
HMMPI::Mat PM_FullHamiltonian::calc_mu_alt(std::vector<double> x, double eps) const				// alternative calculation of MMALA mu (slow, for debugging); only works with MPI size = 1
{
	int size;
	MPI_Comm_size(comm, &size);
	if (size != 1)
		throw HMMPI::Exception("MPI size should be 1 in PM_FullHamiltonian::calc_mu_alt");

	const size_t actdim = x.size();
	double eps2 = eps*eps;

	// 1. get the data
	std::vector<double> grad = PM->ObjFuncGrad_ACT(x);
	HMMPI::Mat MM = G.Get(this, x);						// mass matrix
	HMMPI::Mat invMM = invG.Get(this, x);

	std::vector<HMMPI::Mat> dxi_MM(actdim);
	for (size_t i = 0; i < actdim; i++)
		dxi_MM[i] = dxi_G[i].Get(this, x);

	// 2. calculate
	std::vector<HMMPI::Mat> aux(actdim), aux2(actdim);
	for (size_t j = 0; j < actdim; j++)
	{
		aux[j] = MM / dxi_MM[j];
		aux2[j] = aux[j] * invMM;
	}

	HMMPI::Mat res = HMMPI::Mat(x) - (eps2/4)*(MM / grad);
	for (size_t i = 0; i < actdim; i++)
		for (size_t j = 0; j < actdim; j++)
			res(i, 0) += -eps2*aux2[j](i, j) + eps2/2*invMM(i, j)*(aux[j].Trace());

	return res;
}
//---------------------------------------------------------------------------
int PM_FullHamiltonian::ParamsDim() const noexcept
{
	return PM->ParamsDim();
}
//---------------------------------------------------------------------------
size_t PM_FullHamiltonian::ModelledDataSize() const
{
	return 0;
}
//---------------------------------------------------------------------------
std::string PM_FullHamiltonian::ObjFuncMsg() const
{
	return PM->ObjFuncMsg();
}
//---------------------------------------------------------------------------
void PM_FullHamiltonian::ResetCaches()
{
	G.Reset();
	invG.Reset();
	invU.Reset();
	for (auto &d : dxi_G)
		d.Reset();
	dx_H1.Reset();
	dx_H1_beta.Reset();
	Gaux_grad.Reset();
	Gaux_grad_alpha.Reset();
	dHdp.Reset();
	mu_MMALA.Reset();
	mu_simplMMALA.Reset();
	mu_MMALA_2.Reset();
}
//---------------------------------------------------------------------------
void PM_FullHamiltonian::nums_starts_to_file(FILE *f)
{
	fprintf(f, "locstart\t%d\nlocend  \t%d\nnums\t", locstart, locend);
	fputs(HMMPI::ToString(nums, "%d").c_str(), f);
	fprintf(f, "\nstarts\t");
	fputs(HMMPI::ToString(starts, "%d").c_str(), f);
	fprintf(f, "\nRanks\t");
	fputs(HMMPI::ToString(Ranks, "%d").c_str(), f);
	fprintf(f, "\n");
}
//---------------------------------------------------------------------------
// PM_Posterior
//---------------------------------------------------------------------------
PM_Posterior::PM_Posterior(PhysModel *pm, HMMPI::Mat C, HMMPI::Mat d) : PhysModel(pm->GetComm()), copy_PM(nullptr), PM(pm), Cpr(std::move(C)), dpr(std::move(d))			// comm = PM->comm, con = PM->con (copy as ParamsInterface)
{
	if ((int)Cpr.ICount() != PM->ParamsDim() || (int)Cpr.JCount() != PM->ParamsDim() || (int)dpr.ICount() != PM->ParamsDim() || dpr.JCount() != 1)
	{
		char msg[HMMPI::BUFFSIZE];
		sprintf(msg, "Inconsistent dimensions in PM_Posterior::PM_Posterior -- Cpr[%zu x %zu], dpr[%zu x %zu], ParamsDim = %d", Cpr.ICount(), Cpr.JCount(), dpr.ICount(), dpr.JCount(), PM->ParamsDim());
		throw HMMPI::Exception(msg);
	}

	const ParamsInterface *pm_con = dynamic_cast<const ParamsInterface*>(pm->GetConstr());
	assert(pm_con != nullptr);

	name = "Posterior";

	ParamsInterface *par = new ParamsInterface(*pm_con);						// deleted in DTOR
	con = par;
	init = par->init;
	act_ind = par->get_act_ind();
	tot_ind = par->get_tot_ind();

	invCpr = Cpr.InvSY();
}
//---------------------------------------------------------------------------
PM_Posterior::PM_Posterior(const PM_Posterior &p) : PhysModel(p), Cpr(p.Cpr), dpr(p.dpr), invCpr(p.invCpr)		// copy CTOR will copy PM if it is PM_Proxy*, and will borrow the pointer otherwise
{
	PM_Proxy *proxy = dynamic_cast<PM_Proxy*>(p.PM);
	if (proxy != nullptr)
	{
		PM = proxy->Copy();			// copy
		copy_PM = PM;				// will be deleted by DTOR
	}
	else
	{
		PM = p.PM;					// borrow
		copy_PM = nullptr;
	}

	const ParamsInterface *p_con = dynamic_cast<const ParamsInterface*>(p.con);
	assert(p_con != nullptr);
	ParamsInterface *par = new ParamsInterface(*p_con);							// new; deleted in DTOR
	con = par;
	init = par->init;
	act_ind = par->get_act_ind();
	tot_ind = par->get_tot_ind();
}
//---------------------------------------------------------------------------
PM_Posterior::~PM_Posterior()													// deletes 'con'
{
#ifdef TESTCTOR
	std::ofstream testf(HMMPI::stringFormatArr("TESTCTOR_out_{0:%d}.txt", std::vector<int>{RNK}), std::ios::app);
	testf << "rank " << RNK << ", PM_Posterior -- DTOR --, this = " << this << "\n";
	testf.close();
#endif
	delete con;
	delete copy_PM;
}
//---------------------------------------------------------------------------
double PM_Posterior::ObjFunc(const std::vector<double> &params)
{
	int rank;
	MPI_Comm_rank(comm, &rank);

	double res = PM->ObjFunc(params);
	if (rank == 0)
	{
		HMMPI::Mat r = HMMPI::Mat(params) - dpr;
		res += InnerProd(invCpr*r, r);
		modelled_data = PM->ModelledData();
	}

	return res;
}
//---------------------------------------------------------------------------
std::vector<double> PM_Posterior::ObjFuncGrad(const std::vector<double> &params)
{
	int rank;
	MPI_Comm_rank(comm, &rank);

	HMMPI::Mat res = PM->ObjFuncGrad(params);
	if (rank == 0)
		res += 2*(invCpr*(HMMPI::Mat(params) - dpr));

	return res.ToVector();
}
//---------------------------------------------------------------------------
HMMPI::Mat PM_Posterior::ObjFuncHess(const std::vector<double> &params)
{
	int rank;
	MPI_Comm_rank(comm, &rank);

	HMMPI::Mat res = PM->ObjFuncHess(params);
	if (rank == 0)
		res += 2*invCpr;

	return res;
}
//---------------------------------------------------------------------------
HMMPI::Mat PM_Posterior::ObjFuncFisher(const std::vector<double> &params)
{
	int rank;
	MPI_Comm_rank(comm, &rank);

	HMMPI::Mat res = PM->ObjFuncFisher(params);
	if (rank == 0)
		res += invCpr;

	return res;
}
//---------------------------------------------------------------------------
HMMPI::Mat PM_Posterior::ObjFuncFisher_dxi(const std::vector<double> &params, const int i, int r)
{
	return PM->ObjFuncFisher_dxi(params, i, r);		// prior component is zero matrix
}
//---------------------------------------------------------------------------
int PM_Posterior::ParamsDim() const noexcept
{
	return PM->ParamsDim();
}
//---------------------------------------------------------------------------
size_t PM_Posterior::ModelledDataSize() const
{
	return PM->ModelledDataSize();
}
//---------------------------------------------------------------------------
std::string PM_Posterior::ObjFuncMsg() const
{
	return PM->ObjFuncMsg();
}
//---------------------------------------------------------------------------
std::string PM_Posterior::proc_msg() const
{
	std::string res = PM->proc_msg();
	if (res != "")
		return PM->name + ": " + res;
	else
		return "";
}
//---------------------------------------------------------------------------
void PM_Posterior::correct_of_grad(const std::vector<double> &params, double &y, std::vector<double> &grad) const		// subtracts the prior component from the 'full posterior' y, grad; needed for training PROXY inside POSTERIOR based on POSTERIOR data
{
	HMMPI::Mat r = HMMPI::Mat(params) - dpr;
	HMMPI::Mat work = invCpr*r;
	y -= InnerProd(work, r);

	if (grad.size() != 0)
		grad = (HMMPI::Mat(grad) - 2*work).ToVector();
}
//---------------------------------------------------------------------------
std::vector<int> PM_Posterior::PointsSubset(const std::vector<std::vector<double>> &X0, int count) const
{
	const PM_Proxy *pr = dynamic_cast<const PM_Proxy*>(PM);
	if (pr == nullptr)
		throw HMMPI::Exception("PM should be PM_Proxy in PM_Posterior::PointsSubset");

	return pr->PointsSubset(X0, count);
}
//---------------------------------------------------------------------------
std::string PM_Posterior::AddData(std::vector<std::vector<double>> X0, ValCont *VC, int Nfval_pts)
{
	PM_Proxy *pr = dynamic_cast<PM_Proxy*>(PM);
	if (pr == nullptr)
		throw HMMPI::Exception("PM should be PM_Proxy in PM_Posterior::AddData");

	return pr->AddData(std::move(X0), VC, Nfval_pts);
}
//---------------------------------------------------------------------------
void PM_Posterior::SetDumpFlag(int f)
{
	PM_Proxy *pr = dynamic_cast<PM_Proxy*>(PM);
	if (pr == nullptr)
		throw HMMPI::Exception("PM should be PM_Proxy in PM_Posterior::SetDumpFlag");

	pr->SetDumpFlag(f);
}
//---------------------------------------------------------------------------
int PM_Posterior::GetDumpFlag() const
{
	PM_Proxy *pr = dynamic_cast<PM_Proxy*>(PM);
	if (pr == nullptr)
		throw HMMPI::Exception("PM should be PM_Proxy in PM_Posterior::GetDumpFlag");

	return pr->GetDumpFlag();
}
//---------------------------------------------------------------------------
std::vector<int> PM_Posterior::Data_ind() const
{
	const PM_DataProxy *pr = dynamic_cast<const PM_DataProxy*>(PM);
	if (pr == nullptr)
		throw HMMPI::Exception("PM should be PM_DataProxy in PM_Posterior::Data_ind");

	return pr->Data_ind();
}
//---------------------------------------------------------------------------
// VM_gradient
//---------------------------------------------------------------------------
std::vector<double> VM_gradient::Func_ACT(const std::vector<double> &x) const
{
	std::vector<double> grad = PM->ObjFuncGrad_ACT(x);
	HMMPI::Bcast_vector(grad, 0, comm);

	return grad;
}
//---------------------------------------------------------------------------
HMMPI::Mat VM_gradient::Jac_ACT(const std::vector<double> &x) const
{
	HMMPI::Mat Hess = PM->ObjFuncHess_ACT(x);
	Hess.Bcast(0, comm);

	return Hess;
}
//---------------------------------------------------------------------------
// VM_Ham_eq1
//---------------------------------------------------------------------------
std::vector<double> VM_Ham_eq1::Func_ACT(const std::vector<double> &p) const
{
	Ham0->G.MsgToFile("VM_Ham_eq1::Func_ACT\n");

	if (p.size() != x.size() || p.size() != p0.size())
		throw HMMPI::Exception("p.size() != x.size() || p.size() != p0.size() in VM_Ham_eq1::Func_ACT");

	Ham0->pact = p;
	HMMPI::Mat Hx = Ham0->ObjFuncGrad_ACT(x);
	Hx.Bcast(0, comm);

	return (HMMPI::Mat(p0) - HMMPI::Mat(p) - (eps/2)*Hx).ToVector();
}
//---------------------------------------------------------------------------
HMMPI::Mat VM_Ham_eq1::Jac_ACT(const std::vector<double> &p) const
{
	Ham0->G.MsgToFile("VM_Ham_eq1::Jac_ACT\n");

	// input: x, p0 (unused), eps; p
	if (p.size() != x.size() || p.size() != p0.size())
		throw HMMPI::Exception("p.size() != x.size() || p.size() != p0.size() in VM_Ham_eq1::Jac_ACT");

	Ham0->pact = p;
	HMMPI::Mat J0 = (0.5*eps) * Ham0->Gaux_grad.Get(Ham0, std::pair<std::vector<double>, std::vector<double>>(x, p));

	return std::move(J0) - HMMPI::Mat(p.size());
}
//---------------------------------------------------------------------------
// VM_Ham_eq2
//---------------------------------------------------------------------------
std::vector<double> VM_Ham_eq2::Func_ACT(const std::vector<double> &x) const
{
	Ham0->G.MsgToFile("VM_Ham_eq2::Func_ACT\n");

	Ham0->pact = p;
	Ham1->pact = p;
	HMMPI::Mat Hx0 = Ham0->dHdp.Get(Ham0, std::pair<std::vector<double>, std::vector<double>>(x0, p));
	HMMPI::Mat Hx1 = Ham1->dHdp.Get(Ham1, std::pair<std::vector<double>, std::vector<double>>(x, p));
	Hx0.Bcast(0, comm);
	Hx1.Bcast(0, comm);

	return (HMMPI::Mat(x0) - HMMPI::Mat(x) + (eps/2)*(Hx0 + Hx1)).ToVector();
}
//---------------------------------------------------------------------------
HMMPI::Mat VM_Ham_eq2::Jac_ACT(const std::vector<double> &x) const
{
	Ham0->G.MsgToFile("VM_Ham_eq2::Jac_ACT\n");

	// input: x0 (unused), p, eps; x
	if (x.size() != p.size() || x.size() != x0.size())
		throw HMMPI::Exception("x.size() != p.size() || x.size() != x0.size() in VM_Ham_eq2::Jac_ACT");

	Ham1->pact = p;
	HMMPI::Mat J0 = (-0.5*eps) * Ham1->Gaux_grad.Get(Ham1, std::pair<std::vector<double>, std::vector<double>>(x, p)).Tr();

	return std::move(J0) - HMMPI::Mat(x.size());
}
//---------------------------------------------------------------------------
// VM_Ham_eq2_eps
//---------------------------------------------------------------------------
void VM_Ham_eq2_eps::calc_eps(const std::vector<double> &xfull)	const					// calculates 'eps' from i0-constraint; 'xfull' dimension is ACTDIM
{
	Ham0->G.MsgToFile("VM_Ham_eq2_eps::calc_eps\n");

	int rank;
	MPI_Comm_rank(comm, &rank);

	Ham0->pact = p;
	Ham1->pact = p;
	std::vector<double> Hx0 = Ham0->dHdp.Get(Ham0, std::pair<std::vector<double>, std::vector<double>>(x0, p));
	std::vector<double> Hx1 = Ham1->dHdp.Get(Ham1, std::pair<std::vector<double>, std::vector<double>>(xfull, p));

	if (rank == 0)
		eps = 2*(M0 - x0[i0]) / (Hx0[i0] + Hx1[i0]);

	MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, comm);
}
//---------------------------------------------------------------------------
std::vector<int> VM_Ham_eq2_eps::indices_no_i0() const									// [0,.. i0-1, i0+1,.. ACTDIM), indices vector of dimension ACTDIM-1
{
	std::vector<int> res(x0.size() - 1);
	std::iota(res.begin(), res.begin() + i0, 0);
	std::iota(res.begin() + i0, res.end(), i0 + 1);

	return res;
}
//---------------------------------------------------------------------------
std::vector<double> VM_Ham_eq2_eps::Func_ACT(const std::vector<double> &x) const		// prior to use, set 'x0', 'p', 'i0', 'M0'; after use, the found 'eps' can be retrieved
{
	Ham0->G.MsgToFile("VM_Ham_eq2_eps::Func_ACT\n");

	std::vector<double> xfull = map_x_xfull(x);				// 'i0' bounds are checked here
	calc_eps(xfull);
	std::vector<double> func_full = VM_Ham_eq2::Func_ACT(xfull);

	return HMMPI::Reorder(func_full, indices_no_i0());		// remove component i0
}
//---------------------------------------------------------------------------
HMMPI::Mat VM_Ham_eq2_eps::Jac_ACT(const std::vector<double> &x) const
{
	Ham0->G.MsgToFile("VM_Ham_eq2_eps::Jac_ACT\n");

	int rank;
	MPI_Comm_rank(comm, &rank);

	std::vector<double> xfull = map_x_xfull(x);				// 'i0' bounds are checked here
	calc_eps(xfull);

	HMMPI::Mat Jfull_0 = VM_Ham_eq2::Jac_ACT(xfull);

	Ham0->G.MsgToFile("now calc dH0/dp, dH1/dp\n");
	Ham0->pact = p;
	Ham1->pact = p;
	HMMPI::Mat Hx0 = Ham0->dHdp.Get(Ham0, std::pair<std::vector<double>, std::vector<double>>(x0, p));			// comm-rank-0
	HMMPI::Mat Hx1 = Ham1->dHdp.Get(Ham1, std::pair<std::vector<double>, std::vector<double>>(xfull, p));		// comm-rank-0

	Ham0->G.MsgToFile("now calc Gaux_grad\n");
	HMMPI::Mat Grad_aux = Ham1->Gaux_grad.Get(Ham1, std::pair<std::vector<double>, std::vector<double>>(xfull, p)).Tr();
	HMMPI::Mat res;

	if (rank == 0)
	{
		std::vector<int> no_i0 = indices_no_i0();

		HMMPI::Mat vfull = (eps*eps/(4*(M0 - x0[i0]))) * (Hx0 + Hx1);
		HMMPI::Mat Grad_aux_row_i0(Grad_aux.JCount(), 1, 0.0);
		for (size_t j = 0; j < Grad_aux_row_i0.ICount(); j++)
			Grad_aux_row_i0(j, 0) = Grad_aux(i0, j);

		HMMPI::Mat Jfull = Jfull_0 + OuterProd(vfull, Grad_aux_row_i0);
		res = Jfull.Reorder(no_i0, no_i0);
	}
	res.Bcast(0, comm);

	return res;
}
//---------------------------------------------------------------------------
std::vector<double> VM_Ham_eq2_eps::map_x_xfull(const std::vector<double> &x) const				// mapping of 'x' from VM_Ham_eq2_eps (ACTDIM-1) to VM_Ham_eq2 (ACTDIM)
{
	int actdim = x.size() + 1;
	if (i0 < 0 || i0 >= actdim)
		throw HMMPI::Exception(HMMPI::stringFormatArr("i0 ({0:%d}) is out of range [0, {1:%d}) in VM_Ham_eq2_eps::map_x_xfull", std::vector<int>{i0, actdim}));

	std::vector<double> res(actdim);
	std::copy(x.begin(), x.begin() + i0, res.begin());
	res[i0] = M0;
	std::copy(x.begin() + i0, x.end(), res.begin() + i0 + 1);

	return res;
}
//---------------------------------------------------------------------------
std::vector<double> VM_Ham_eq2_eps::map_xfull_x(const std::vector<double> &xfull) const			// auxiliary function, removes coordinate 'i0' from the vector
{
	if (i0 < 0 || i0 >= (int)xfull.size())
		throw HMMPI::Exception(HMMPI::stringFormatArr("i0 ({0:%d}) is out of range [0, {1:%d}) in VM_Ham_eq2_eps::map_xfull_x", std::vector<int>{i0, (int)xfull.size()}));

	return HMMPI::Reorder(xfull, indices_no_i0());
}
//---------------------------------------------------------------------------
// VM_Ham_eq2_eps_full
//---------------------------------------------------------------------------
std::vector<double> VM_Ham_eq2_eps_full::Func_ACT(const std::vector<double> &x) const		// prior to use, set 'x0', 'p', 'i0', 'M0'; "x" should have been mapped from "xfull"
{
	Ham0->G.MsgToFile("VM_Ham_eq2_eps_full::Func_ACT\n");
	std::vector<double> xfull = map_x_xfull(x);		// 'eps' gets filled
	return VM_Ham_eq2::Func_ACT(xfull);
}
//---------------------------------------------------------------------------
HMMPI::Mat VM_Ham_eq2_eps_full::Jac_ACT(const std::vector<double> &x) const
{
	Ham0->G.MsgToFile("VM_Ham_eq2_eps_full::Jac_ACT\n");
	std::vector<double> xfull = map_x_xfull(x);		// 'eps' gets filled

	Ham0->G.MsgToFile("calc dH0/dp, dH1/dp\n");
	Ham0->pact = p;
	Ham1->pact = p;
	HMMPI::Mat Hx0 = Ham0->dHdp.Get(Ham0, std::pair<std::vector<double>, std::vector<double>>(x0, p));			// comm-rank-0
	HMMPI::Mat Hx1 = Ham1->dHdp.Get(Ham1, std::pair<std::vector<double>, std::vector<double>>(xfull, p));		// comm-rank-0
	Hx0.Bcast(0, comm);
	Hx1.Bcast(0, comm);

	HMMPI::Mat res = VM_Ham_eq2::Jac_ACT(xfull);
	assert(Hx0.ICount() == Hx1.ICount());
	assert(res.ICount() == Hx0.ICount());
	for (size_t i = 0; i < res.ICount(); i++)
		res(i, i0) = (Hx0(i, 0) + Hx1(i, 0))/(2*Scale);

	return res;
}
//---------------------------------------------------------------------------
std::vector<double> VM_Ham_eq2_eps_full::map_x_xfull(const std::vector<double> &x) const			// mapping of "x" from VM_Ham_eq2_eps_full to VM_Ham_eq2 (filling 'eps')
{
	if (i0 < 0 || i0 >= (int)x.size())
		throw HMMPI::Exception("i0 out of range in VM_Ham_eq2_eps_full::map_x_xfull");

	std::vector<double> res = x;
	res[i0] = M0;
	eps = x[i0]/Scale;

	return res;
}
//---------------------------------------------------------------------------
std::vector<double> VM_Ham_eq2_eps_full::map_xfull_x(const std::vector<double> &xfull) const		// "xfull" and current 'eps' are used to make "x" suitable for use by VM_Ham_eq2_eps_full
{
	if (i0 < 0 || i0 >= (int)xfull.size())
		throw HMMPI::Exception("i0 out of range in VM_Ham_eq2_eps_full::map_xfull_x");

	std::vector<double> res = xfull;
	res[i0] = eps*Scale;

	return res;
}
//---------------------------------------------------------------------------
