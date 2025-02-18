/*
 * PhysModProxy.cpp
 *
 *  Created on: 31 May 2016
 *      Author: ilya fursov
 */

#include "Abstract.h"
#include "PhysModels.h"
#include "MathUtils.h"
#include <cassert>
#include <algorithm>
#include <iostream>
#include <fstream>
#include "mpi.h"
#include "Parsing.h"
#include "Parsing2.h"

//#define TESTING
//#define TESTNEWPROXY
//#define PROXY_DEBUG			// messages like "adding YYY points out of ..."

#define KRIG_OPT_STDOUT		// verbose messaging to stdout for kriging parameters optimization

#ifdef TESTNEWPROXY
#include <iostream>
#include <fstream>
std::string test_fn = "x_TESTNEWPROXY_{0:%d}.txt";
#endif

//---------------------------------------------------------------------------
// PM_Proxy
//---------------------------------------------------------------------------
void PM_Proxy::write_proxy_vals_begin(const std::vector<std::vector<double>> &X0)
{
	MPI_Barrier(comm);
	of_before0 = std::vector<double>();
	data_before0 = std::vector<std::vector<double>>();
	first_call = true;

	int max_X0_size = 0;
	if (starts.size() > 0)
		max_X0_size = starts[0].get_X_0().size();
	MPI_Allreduce(MPI_IN_PLACE, &max_X0_size, 1, MPI_INT, MPI_MAX, comm);

	if (max_X0_size > 0)			// this condition is sync
	{
		first_call = false;
		of_before0.reserve(X0.size());				// calculate o.f. and data for all points in X0 (before proxy update)
		data_before0.reserve(X0.size());
		for (size_t i = 0; i < X0.size(); i++)
		{
			of_before0.push_back(ObjFunc(X0[i]));
			if (ModelledDataSize() != 0)
			{
				if (RNK == 0)
					data_before0.push_back(ModelledData());
				else
					data_before0.push_back(std::vector<double>());		// dummy
			}
		}
	}
}
//---------------------------------------------------------------------------
void PM_Proxy::write_proxy_vals_end(const std::vector<std::vector<double>> &X0, const std::vector<size_t> &inds)
{
	MPI_Barrier(comm);
	if (!first_call)
	{
		HMMPI::VecAppend(of_before, HMMPI::Reorder(of_before0, inds));			// append previously calculated values
		if (ModelledDataSize() != 0)
			HMMPI::VecAppend(data_before, HMMPI::Reorder(data_before0, inds));

		of_after.reserve(of_after.size() + inds.size());						// calculate o.f. and data for 'inds' points (after proxy update)
		data_after.reserve(data_after.size() + inds.size());
		for (size_t i = 0; i < inds.size(); i++)
		{
			of_after.push_back(ObjFunc(X0[inds[i]]));
			if (ModelledDataSize() != 0)
			{
				if (RNK == 0)
					data_after.push_back(ModelledData());
				else
					data_after.push_back(std::vector<double>());		// dummy
			}
		}

		assert(of_before.size() == of_after.size());
		if (ModelledDataSize() != 0)
			assert(of_before.size() == data_before.size() && of_before.size() == data_after.size());
		if (RNK == 0 && dump_flag != -1)			// output to files
		{
			char fname[100];
			sprintf(fname, dump_vals, dump_flag);
			FILE *f = fopen(fname, "w");
			for (size_t i = 0; i < of_before.size(); i++)
			{
				std::string dat0 = "\n";
				std::string dat1 = "\n";
				if (ModelledDataSize() != 0)
				{
					dat0 = HMMPI::ToString(data_before[i], "%20.16g");
					dat1 = HMMPI::ToString(data_after[i], "%20.16g");
				}
				dat0.pop_back();					// delete '\n'
				dat1.pop_back();
				fprintf(f, "%20.16g\t|\t%20.16g\t|\t%s\t|\t%s\n", of_before[i], of_after[i], dat0.c_str(), dat1.c_str());
			}
			fclose(f);
		}
	}
}
//---------------------------------------------------------------------------
void PM_Proxy::AddPoints(const std::vector<std::vector<double>> &X0, const std::vector<std::vector<double>> &X1)		// adds 'X0' to starts[*].X_0, adds 'X1' to starts[*].X_1, updates distance matrices;
{
	for (size_t i = 0; i < starts.size(); i++)
		if (!starts[i].is_empty)
			starts[i].AddPoints(X0, X1);
}
//---------------------------------------------------------------------------
void PM_Proxy::RecalcVals()				// (after adding values) makes CinvZ calculation; works differently for different proxy types
{
	assert(starts.size() == 1 && ends.size() == 1);		// it's simple proxy
	mat_eff_rank = ends[0].RecalcVals();
}
//---------------------------------------------------------------------------
std::vector<std::vector<double>> PM_Proxy::XFromFile(std::string fname) const		// returns 'pop[len][full_dim]' (nontrivial only on comm-RANKS-0) read from file "fname"; this function is mostly for debugging
{
	int dim = ParamsDim();				// full dimension; defined on all ranks
	assert(dim == PM->ParamsDim());

	int rnk = -1;
	if (comm != MPI_COMM_NULL)
		MPI_Comm_rank(comm, &rnk);

	std::vector<std::vector<double>> res;
	if (rnk == 0)
	{
		FILE *f0 = fopen(fname.c_str(), "r");
		if (f0 != 0)
		{
			while (!feof(f0))
			{
				std::vector<double> x0(dim);
				bool filled = false;

				for (int i = 0; i < dim; i++)
				{
					double d;
					int n = fscanf(f0, "%lg", &d);
					if (n == 1)
					{
						x0[i] = d;
						filled = true;
					}
					else if (i != 0)	// i == 0 && n == 0 may happen for empty line
					{
						fclose(f0);
						throw HMMPI::Exception("Number of items in a line is less than dimension in PM_Proxy::XFromFile");
					}
					else
						break;
				}

				if (filled)
					res.push_back(x0);
			}
			fclose(f0);
		}
		else
			throw HMMPI::Exception("Cannot open file in PM_Proxy::XFromFile");
	}

	return res;
}
//---------------------------------------------------------------------------
void PM_Proxy::copy_starts_ends_link(const PM_Proxy &p)		// takes the "starts" - "ends" link from "p" (internally "p" should be of the same PROXY type as "this"); call it in COPY CTORs before set_refs()
{
	if (starts.size() != p.starts.size() || ends.size() != p.ends.size())
		throw HMMPI::Exception("Inconsistent starts/ends sizes in PM_Proxy::copy_starts_ends_link");

	for (size_t i = 0; i < ends.size(); i++)
	{
		int ind = p.ends[i].start_index;
		ends[i].set_start(&starts[ind], ind);
	}
}
//---------------------------------------------------------------------------
void PM_Proxy::set_refs()					// sets the remaining links
{
	for (size_t i = 0; i < starts.size(); i++)
		if (!starts[i].is_empty)
		{
			starts[i].set_refs();
			starts[i].index = i;
		}
	for (size_t i = 0; i < ends.size(); i++)
	{
		ends[i].set_refs();
		ends[i].index = i;
	}
}
//---------------------------------------------------------------------------
void PM_Proxy::reset_kc_ks_cache() const
{
	for (auto &s : starts)
		if (!s.is_empty)
			s.reset_kc_cache();
	for (auto &e : ends)
		e.reset_ks_cache();
}
//---------------------------------------------------------------------------
PM_Proxy::PM_Proxy(MPI_Comm c, PhysModel *pm, const HMMPI::BlockDiagMat *bdc, Parser_1 *K, KW_item *kw, _proxy_params *config) : PhysModel(c, bdc), first_call(true), PM(pm), dump_flag(-1), train_from_dump(-1)
{
	name = "PM_Proxy";
	con = pm->GetConstr();

	starts = std::vector<KrigStart>(1, KrigStart(K, kw, config));
	ends = std::vector<KrigEnd>(1, KrigEnd(K, config));

	kw->Start_pre();
	kw->Add_pre(config->name);
	kw->Finish_pre();
	do_optimize_krig = (config->opt == "ON");

	pm->ExportIAC(this);		// take stuff from "pm" to "this"

	ends[0].set_start(&starts[0], 0);
	set_refs();
}
//---------------------------------------------------------------------------
PM_Proxy::PM_Proxy(const PM_Proxy &p) : PhysModel(p),
		of_before0(p.of_before0), data_before0(p.data_before0), of_before(p.of_before), of_after(p.of_after), data_before(p.data_before), data_after(p.data_after),
		dump_flag(p.dump_flag), train_from_dump(p.train_from_dump), starts(p.starts), ends(p.ends), do_optimize_krig(p.do_optimize_krig), mat_eff_rank(p.mat_eff_rank)
{
	PM = p.PM;
	first_call = p.first_call;

	copy_starts_ends_link(p);
	set_refs();
}
//---------------------------------------------------------------------------
KrigCorr *PM_Proxy::get_KrigCorr()				// call this function only for simple proxy
{
	assert(starts.size() == 1);
	if (!starts[0].is_empty)
		return starts[0].get_kc_ptr();
	else
		return nullptr;
};
//---------------------------------------------------------------------------
KrigSigma *PM_Proxy::get_KrigSigma()			// call this function only for simple proxy
{
	assert(ends.size() == 1);
	return ends[0].get_ks_ptr();
};
//---------------------------------------------------------------------------
void PM_Proxy::SetDumpFlag(int f)				// sets dump_flag for "this", starts and ends
{
	dump_flag = f;

	for (size_t i = 0; i < starts.size(); i++)
		if (!starts[i].is_empty)
			starts[i].dump_flag = f;

	for (size_t i = 0; i < ends.size(); i++)
		ends[i].dump_flag = f;
};
//---------------------------------------------------------------------------
std::string PM_Proxy::init_msg() const			// message on how KrigCorr was initialised (LIMITSKRIG/PROXY_CONFIG)
{
	if (starts.size() > 0 && !starts[0].is_empty)
		return starts[0].init_msg();
	else
		return "";		// may happen on some ranks of DataProxy2
}
//---------------------------------------------------------------------------
std::string PM_Proxy::proc_msg() const			// sync message showing: (1) effective rank of kriging matrix; (2) pscale, grad_inds for "starts[0]"
{												// for derived PROXY classes, proc_msg() is different
	assert(starts.size() == 1);
	std::vector<double> ps = starts[0].get_pscale();
	std::vector<size_t> gi = starts[0].get_grad_inds();
	HMMPI::Bcast_vector(ps, 0, comm);
	HMMPI::Bcast_vector(gi, 0, comm);

	std::string msg = (std::string)HMMPI::MessageRE("Эффективный ранг матрицы кригинга: ", "Effective rank of kriging matrix: ") + mat_eff_rank;
	msg += "pscale = " + HMMPI::ToString(ps, "%.4g", ", ");
	msg.pop_back();		// pop '\n'
	msg += "; grad_inds = " + (gi.size() == 0 ? "''\n" : HMMPI::ToString(gi, "%zu", ", "));
	return msg;
}
//---------------------------------------------------------------------------
bool PM_Proxy::CheckLimits(const std::vector<double> &params) const
{
	if (params.size() != (size_t)ParamsDim())
		throw HMMPI::Exception(HMMPI::stringFormatArr("params.size() != ParamsDim() [{0:%ld}, {1:%ld} respectively] in PM_Proxy::CheckLimits", std::vector<size_t>{params.size(), (size_t)ParamsDim()}));

	if (PM == 0)
	{
		limits_msg = "";
		return true;
	}
	else
	{
		bool res = PM->CheckLimits(params);
		limits_msg = PM->get_limits_msg();
		return res;
	}
}
//---------------------------------------------------------------------------
std::vector<size_t> PM_Proxy::PointsSubset(const std::vector<std::vector<double>> &X0, size_t count) const		// selects 'count' points from X0 - these points are then to be added to starts[*].X_0;
{																												// selection is based on starts[*].X_0 + X0
	std::vector<size_t> res;																						// the SYNC vector of selected indices (for 'X0') is returned
	int rank;																									// selection works independently of possible differences in "starts"
	MPI_Comm_rank(comm, &rank);

	int err = 0;							// 0 - inner check ok; 1 - ERROR
	if (rank == 0)
	{
		int ind = -1;						// index of a non-empty 'start'
		for (size_t i = 0; i < starts.size(); i++)
			if (!starts[i].is_empty)
			{
				ind = i;
				break;
			}

		assert(ind != -1);
		bool all_taken = true;
		res = starts[ind].PointsSubsetKS(X0, count, all_taken);

		const PM_SimProxy *simproxy = dynamic_cast<const PM_SimProxy*>(this);
		if (simproxy != nullptr && !all_taken)
		{
			assert(simproxy->start_to_block_size() == simproxy->starts.size());
			for (size_t i = 0; i < simproxy->start_to_block_size(); i++)
				if (simproxy->start_to_block_i(i) != 0)
				{
					err = 1;				// SIMPROXY with N_blocks > 1, and PointsSubsetKS() took a subset of 'X0' -- currently this situation is prohibited
					break;
				}
		}
	}

	MPI_Bcast(&err, 1, MPI_INT, 0, comm);
	HMMPI::Bcast_vector(res, 0, comm);

	if (err)
		throw HMMPI::Exception("SIMPROXY with N_blocks > 1, and PointsSubsetKS() took a subset of 'X0'");

	return res;
}
//---------------------------------------------------------------------------
std::string PM_Proxy::AddData(std::vector<std::vector<double>> X0, ValCont *VC, int Nfval_pts)		// adds new data (points X0 and values/gradients VC) and trains proxy; returns the message about the total number of points
{																									// Nfval_pts shows how many points with func. vals should be selected (however, all points with grads are taken)
	int vals_count = VC->vals_count();																// if proxy (X_0) is empty, all points with func. vals are taken
	int tot_count = VC->total_count();																// X0 can be defined on comm-RANKS-0 only
	HMMPI::Bcast_vector(X0, 0, comm);		// sync X0, since input X0 may be defined on comm-RANKS-0 only

	if (X0.size() != (size_t)tot_count)
		throw HMMPI::Exception(HMMPI::stringFormatArr("Inconsistent sizes of 'X0' ({0:%zu}) and 'VC' ({1:%zu}) in PM_Proxy::AddData", std::vector<size_t>{X0.size(), (size_t)tot_count}));

#ifdef WRITE_PROXY_VALS
	write_proxy_vals_begin(X0);				// proxy ObjFunc() are calculated here
#endif

	reset_kc_ks_cache();					// since new points will be added, cache in subordinate "kc", "ks" should be reset manually

	std::vector<std::vector<double>> x_0(X0.begin(), X0.begin() + vals_count);		// points for func. values
	std::vector<std::vector<double>> x_1(X0.begin() + vals_count, X0.end());		// points for func. gradients
	std::vector<size_t> inds = PointsSubset(x_0, Nfval_pts);

	x_0 = HMMPI::Reorder(x_0, inds);
	AddPoints(x_0, x_1);

	for (auto &s : starts)
		if (!s.is_empty)
			s.RecalcPoints();				// 'RecalcPoints'

	VC->DistrValues(ends, inds);			// 'AddVals'

	if (do_optimize_krig)
		for (auto &e : ends)
			e.OptimizeKrig();

	RecalcVals();

#ifdef WRITE_PROXY_VALS
	write_proxy_vals_end(X0, inds);			// proxy ObjFunc() are calculated here
#endif

	char msg[HMMPI::BUFFSIZE];
	if (starts.size() > 0 && !starts[0].is_empty)
		sprintf(msg, "Number of design points on starts[0]: %zu for func. values, %zu for func. gradients\n", starts[0].get_X_0().size(), starts[0].get_X_1().size());
	else
		msg[0] = 0;

	return msg;
}
//---------------------------------------------------------------------------
double PM_Proxy::obj_func_work(const std::vector<double> &params)
{
	assert(starts.size() == 1 && ends.size() == 1);
	starts[0].ObjFuncCommon(params);

	return ends[0].ObjFuncPrivate();
}
//---------------------------------------------------------------------------
std::vector<double> PM_Proxy::obj_func_grad_work(const std::vector<double> &params)
{
	assert(starts.size() == 1 && ends.size() == 1);
	starts[0].ObjFuncGradCommon(params);

	return ends[0].ObjFuncGradPrivate();
}
//---------------------------------------------------------------------------
HMMPI::Mat PM_Proxy::obj_func_hess_work(const std::vector<double> &params)
{
	size_t dim = params.size();
	HMMPI::Mat Hess(dim, dim, 0);			// result

	for (size_t l = 0; l < dim; l++)
	{
		std::vector<double> col = ObjFuncHess_l(params, l);		// all ranks receive and do the same
		for (size_t i = 0; i < dim; i++)
			Hess(i, l) = col[i];
	}

	return Hess;
}
//---------------------------------------------------------------------------
std::vector<double> PM_Proxy::ObjFuncHess_l(const std::vector<double> &params, int l)
{
	assert(starts.size() == 1 && ends.size() == 1);
	starts[0].ObjFuncHess_lCommon(params, l);

	return ends[0].ObjFuncHess_lPrivate();
}
//---------------------------------------------------------------------------
// trains the proxy (i.e. adds to the existing proxy state) based on design points pop[len][full_dim], and PM->ObjFuncMPI_ACT("pop") calculated in parallel via MPI
// 'pop' is only referenced on comm-RANKS-0; underlying PM should have comm == "MPI_COMM_SELF" (i.e. all PM->comm-RANKS == 0)
// If train_from_dump != -1, 'pop' is not used, X & y are taken from appropriate files (reading the same number of lines as in the original "pop"); currently gradients are not read from the files
// "grad_ind" (comm-RANKS-0) are indices in [0, len) for points where gradients will be estimated and added to the proxy; these training points are always taken from "pop" ("pop" may be from the file)
// Nfval_pts is the same as in AddData()
// the returned message is [whether proxy was trained from dump] plus "Number of design points on starts..." from AddData()
std::string PM_Proxy::Train(std::vector<std::vector<double>> pop, std::vector<size_t> grad_ind, int Nfval_pts)
{
	MPI_Barrier(comm);
	std::string msg = "";
	ValCont *VC;

	size_t dup;
	std::sort(grad_ind.begin(), grad_ind.end());
	if (HMMPI::FindDuplicate(grad_ind, dup))
		throw HMMPI::Exception(HMMPI::stringFormatArr("Duplicate index {0:%zu} for a gradient training point in PM_Proxy::Train", std::vector<size_t>{dup}));

	int rank = -1, size = 0;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	// I. Adding the points
	if (!is_dataproxy())
		VC = new ValContDouble(comm, PM, pop, train_from_dump);
	else
		VC = new ValContVecDouble(comm, PM, pop, train_from_dump, Data_ind());				// "data_ind" (sync on all ranks) is taken from BDC

	if (train_from_dump != -1 && rank == 0)
	{
		char fname[100];
		sprintf(fname, dump_X, train_from_dump, 0);		// _pr0
		const size_t old_pop_len = pop.size();
		pop = XFromFile(fname);
		if (pop.size() < old_pop_len)
			throw HMMPI::Exception(HMMPI::stringFormatArr("X dump file has less lines ({0:%zu}) than required ({1:%zu})", std::vector<size_t>{pop.size(), old_pop_len}));		// non-sync exception!
		pop = std::vector<std::vector<double>>(pop.begin(), pop.begin() + old_pop_len);		// only leave the number of lines as in the original "pop"
	}

	// output 'pop' to file
	if (RNK == 0)
	{
		FILE *sw = fopen("ProxyDesignPoints.txt", "w");
		if (sw != NULL)
		{
			for (const auto &p : pop)
				fprintf(sw, "%s", HMMPI::ToString(p, "%20.16g").c_str());
			fclose(sw);
		}
	}

	if (grad_ind.size() > 0 && (*--grad_ind.end() >= (size_t)VC->vals_count()))
		throw HMMPI::Exception("'grad_ind' values out of range in PM_Proxy::Train");

	// II. Adding the gradients
	// "grad_ind" (comm-RANKS-0) are indices in [0, len) for points where gradients will be estimated and added to the proxy; these training points are always taken from "pop" (irrespective of 'train_from_dump')
	std::vector<std::vector<double>> grad_pop;
	if (rank == 0)
		grad_pop = HMMPI::Reorder(pop, grad_ind);		// pop[len][full_dim]

	int gr_len = grad_pop.size();
	MPI_Bcast(&gr_len, 1, MPI_INT, 0, comm);			// sync number of points for gradients
	std::vector<std::vector<double>> grads_v(gr_len);	// for simple PROXY
	std::vector<HMMPI::Mat> grads_M(gr_len);			// for DATAPROXY

	for (int i = 0; i < gr_len; i++)	// calculate the gradients
	{
		std::vector<double> p;
		if (rank == 0)
			p = grad_pop[i];
		HMMPI::Bcast_vector(p, 0, comm);				// sync point for gradient
		std::vector<double> grad = PM->ObjFuncGrad(p);
		if (!is_dataproxy())
		{
			HMMPI::Bcast_vector(grad, 0, comm);
			grads_v[i] = grad;
		}
		else
		{
			HMMPI::Mat Sens = PM->DataSens();
			Sens.Bcast(0, comm);
			int sens_len = Sens.Length();
			if (sens_len == 0)
				throw HMMPI::Exception("No 'data_sens' available when trying to train DATAPROXY by gradients in PM_Proxy::Train");	// sync error
			grads_M[i] = Sens;
		}
	}

	if (!is_dataproxy())
		dynamic_cast<ValContDouble*>(VC)->Add(ValContDouble(comm, std::vector<double>(), grads_v));			// no func. values, only gradients
	else
	{
		std::vector<int> data_ind = Data_ind();			// sync
		if ((int)data_ind.size() != size + 1)
			throw HMMPI::Exception("Size of data_ind is not consistent with communicator in PM_Proxy::Train");

		int smry_len = data_ind[size];
		if ((size_t)smry_len != PM->ModelledDataSize())
			throw HMMPI::Exception(HMMPI::stringFormatArr("smry_len ({0:%zu}) != PM->ModelledDataSize ({1:%zu}) in PM_Proxy::Train", std::vector<size_t>{(size_t)smry_len, PM->ModelledDataSize()}));

		dynamic_cast<ValContVecDouble*>(VC)->Add(ValContVecDouble(comm, data_ind, grads_M));
	}

	HMMPI::VecAppend(pop, grad_pop);
	msg = AddData(pop, VC, Nfval_pts);
	delete VC;

	if (train_from_dump != -1)
	{
		std::string msg0 = HMMPI::stringFormatArr("\n**** ПРОКСИ была загружена из файла #{0:%d} ****\n", "\n*** PROXY was trained from dump file #{0:%d} ***\n", train_from_dump);
		msg = std::string(msg0.length()-2, '*') + msg0 + std::string(msg0.length()-2, '*') + "\n" + msg;
	}

	return msg;
}
//---------------------------------------------------------------------------
// PM_DataProxy
//---------------------------------------------------------------------------
void PM_DataProxy::process_data_size(const std::vector<double> &d)
{
	int size_w, rank_w;
	MPI_Comm_size(MPI_COMM_WORLD, &size_w);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank_w);

	int rank_c = -1;
	if (comm != MPI_COMM_NULL)
		MPI_Comm_rank(comm, &rank_c);		// comm-RANK - item #1

	int sz = d.size();						// item #2

	std::vector<int> Ranks(size_w);			// will store comm-ranks
	std::vector<int> Sizes(size_w);			// will store sizes of "d"

	MPI_Gather(&rank_c, 1, MPI_INT, Ranks.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);	// Ranks, Sizes are ok on MPI_COMM_WORLD-rank-0
	MPI_Gather(&sz, 1, MPI_INT, Sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

	if (rank_w == 0)
	{
		data_size = 0;
		for (int i = 0; i < size_w; i++)
		{
			if (data_size != 0 && Ranks[i] == 0 && Sizes[i] != (int)data_size)
				throw HMMPI::Exception("PM_DataProxy::process_data_size received inconsistent data sizes from comm-RANKS-0");
			if (data_size == 0 && Ranks[i] == 0)
				data_size = Sizes[i];
		}
	}

	MPI_Bcast(&data_size, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
}
//---------------------------------------------------------------------------
void PM_DataProxy::RecalcVals()
{
	assert(starts.size() == 1);													// other proxy types will have their own RecalcVals()
	size_t len_trend = starts[0].get_multi_ind().size();

	// create multiple right hand sides
	HMMPI::Mat ITY;
	for (size_t i = 0; i < ends.size(); i++)
	{
		std::vector<double> work = ends[i].get_y();
		int wsize = work.size();
		ITY = std::move(ITY) || HMMPI::Mat(std::move(work), 1, wsize);			// accumulate the transpose matrix for efficiency
	}

	ITY = ITY.Tr() || HMMPI::Mat(len_trend, ends.size(), 0.0);
	HMMPI::Mat BigCinvZ;
	mat_eff_rank = "-1\n";
	if (ends.size() > 0)
	{
		BigCinvZ = starts[0].get_sol()->Solve(starts[0].get_C(), ITY).Tr();		// transpose to convert columns to rows
		int rank = starts[0].get_sol()->rank;
		if (rank != -1)
			mat_eff_rank = HMMPI::ToString(std::vector<int>{rank}, "%d");
	}

	// fill all CinvZ's
	const double *pbcz = BigCinvZ.ToVector().data();
	size_t len = BigCinvZ.JCount();
	for (size_t i = 0; i < ends.size(); i++)
		ends[i].set_CinvZ(std::vector<double>(pbcz + len*i, pbcz + len*(i+1)));

	if (dump_flag != -1)	// debug output to files
	{
		char fname[100];
		sprintf(fname, dump_CinvZ, dump_flag, RNK, 0);		// _pr0
		FILE *f = fopen(fname, "w");
		BigCinvZ.Tr().SaveASCII(f, "%20.16g");
		fclose(f);

		sprintf(fname, dump_Ity, dump_flag, RNK, 0);		// _pr0
		f = fopen(fname, "w");
		ITY.SaveASCII(f, "%20.16g");
		fclose(f);
	}
}
//---------------------------------------------------------------------------
HMMPI::Mat PM_DataProxy::ObjFuncSens_dxi(const std::vector<double> &params, const int i)
{
	size_t dim = params.size();
	if (comm == MPI_COMM_NULL)
		return std::vector<double>(dim, 0);

	if (i < 0 || i >= (int)dim)
		throw HMMPI::Exception("'i' out of range in PM_DataProxy::ObjFuncSens_dxi");

	HMMPI::Mat res(d0.ICount(), dim, 0);

	for (auto &p : starts)							// run common parts
		if (!p.is_empty)
			p.ObjFuncHess_lCommon(params, i);

	for (size_t j = 0; j < d0.ICount(); j++)		// find i-columns of Hessians for all local data points "j"
	{
		std::vector<double> hess = ends[j].ObjFuncHess_lPrivate();
		memcpy(res.ToVectorMutable().data() + j*dim, hess.data(), sizeof(double)*dim);
	}

	return res;
}
//---------------------------------------------------------------------------
void PM_DataProxy::data_ind_count_displ(std::vector<int> &counts, std::vector<int> &displs, int mult) const
{
	std::vector<int> ind = Data_ind();
	assert(ind.size() >= 2);

	counts = std::vector<int>(ind.size()-1);
	displs = std::vector<int>(ind.size()-1);

	for (size_t i = 0; i < counts.size(); i++)
	{
		counts[i] = (ind[i+1] - ind[i]) * mult;
		displs[i] = ind[i] * mult;
	}
}
//---------------------------------------------------------------------------
PM_DataProxy::PM_DataProxy(PhysModel *pm, Parser_1 *K, KW_item *kw, _proxy_params *config, const HMMPI::BlockDiagMat *bdc, const std::vector<double> &d) : PM_Proxy(bdc->GetComm(), pm, bdc, K, kw, config)
{																		// easy CTOR; to be called on all ranks of MPI_COMM_WORLD
	name = "PM_DataProxy";												// 'bdc' (block diagonal covariance) should be created in advance, it will provide "comm" for PM_DataProxy
	int rank = 0, size;													// 'd' - observed data (only supply it on comm-RANKS-0)
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	do_optimize_krig = false;	// no kriging optimization for PM_DataProxy

	DECLKWD(textsmry, KW_textsmry, "TEXTSMRY");		// used only for "randn"
	RndN = &textsmry->randn;

	process_data_size(d);		// fills "data_size"; needed for correct work of ModelledDataSize()

	std::vector<int> data_ind = BDC->Data_ind();
	if ((int)data_ind.size() != size + 1)
		throw HMMPI::Exception("Некорректная длина data_ind в PM_DataProxy::PM_DataProxy", "Wrong data_ind size in PM_DataProxy::PM_DataProxy");

	starts = std::vector<KrigStart>(1, KrigStart(K, kw, config));								// create 'starts' (1-element vector)
	ends = std::vector<KrigEnd>(data_ind[rank+1] - data_ind[rank], KrigEnd(K, config));			// create MPI-distributed 'ends'

	for (size_t i = 0; i < ends.size(); i++)				// link 'starts[0]' with local 'ends'
		ends[i].set_start(&starts[0], 0);

	set_refs();

	// initialize local observed data 'd0'
	std::vector<double> locd0(data_ind[rank+1] - data_ind[rank]);	// local observed data
	std::vector<int> counts, displs;
	data_ind_count_displ(counts, displs);
	MPI_Scatterv(d.data(), counts.data(), displs.data(), MPI_DOUBLE, locd0.data(), locd0.size(), MPI_DOUBLE, 0, comm);
	d0 = std::move(locd0);
	d0_orig = d0;				// d0 is not perturbed yet

	print_comms();
}
//---------------------------------------------------------------------------
PM_DataProxy::PM_DataProxy(const PM_DataProxy &p) : PM_Proxy(p), RndN(p.RndN)
{
	Gr_loc = p.Gr_loc;
	resid_loc = p.resid_loc;
	data_size = p.data_size;
	d0 = p.d0;
	d0_orig = p.d0_orig;
}
//---------------------------------------------------------------------------
double PM_DataProxy::obj_func_work(const std::vector<double> &params)
{
	if (comm == MPI_COMM_NULL)
		return 0;

	for (auto &s : starts)
		if (!s.is_empty)
			s.ObjFuncCommon(params);

	std::vector<double> resid(d0.ICount());		// residual = model - observed = d_m - d_o
	std::vector<double> mod_data(d0.ICount());	// modelled data (local)
	for (size_t i = 0; i < resid.size(); i++)
	{
		mod_data[i] = ends[i].ObjFuncPrivate();
		resid[i] = mod_data[i] - d0(i, 0);
	}

	double res = 0;										// for reduced result
	double prod = BDC->InvTwoSideVecMult(resid);		// local result
	MPI_Reduce(&prod, &res, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

	// gather modelled data
	std::vector<int> counts, displs;
	data_ind_count_displ(counts, displs);		// sync between ranks

	int smry_len = ModelledDataSize();			// sync between ranks
	int rank;
	MPI_Comm_rank(comm, &rank);
	assert(counts[rank] == (int)mod_data.size());

	modelled_data = std::vector<double>();
	if (rank == 0)
		modelled_data = std::vector<double>(smry_len);
	MPI_Gatherv(mod_data.data(), mod_data.size(), MPI_DOUBLE, modelled_data.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, comm);

	return res;
}
//---------------------------------------------------------------------------
std::vector<double> PM_DataProxy::obj_func_grad_work(const std::vector<double> &params)
{
	if (comm == MPI_COMM_NULL)
		return std::vector<double>(params.size(), 0);

	for (auto &s : starts)
		if (!s.is_empty)
		{
			s.ObjFuncCommon(params);
			s.ObjFuncGradCommon(params);
		}

	std::vector<double> resid(d0.ICount());			// residual = 2*(model - observed) = 2*(d_m - d_o)
	HMMPI::Mat Gr(params.size(), d0.ICount(), 0);	// all local gradients

	for (size_t i = 0; i < resid.size(); i++)		// find objective functions and gradients for all local data points "i"
	{
		resid[i] = 2*(ends[i].ObjFuncPrivate() - d0(i, 0));

		std::vector<double> grad = ends[i].ObjFuncGradPrivate();
		for (size_t j = 0; j < grad.size(); j++)
			Gr(j, i) = grad[j];
	}

	std::vector<double> div = *BDC / resid;			// 2 * C^(-1) * (d_m - d_o)
	std::vector<double> Gr_div = Gr * div;

	std::vector<double> res(params.size(), 0);
	MPI_Reduce(Gr_div.data(), res.data(), res.size(), MPI_DOUBLE, MPI_SUM, 0, comm);

	// gather sensitivities
	std::vector<int> counts, displs;
	data_ind_count_displ(counts, displs, params.size());	// sync between ranks

	int smry_len = ModelledDataSize();						// sync between ranks
	int rank;
	MPI_Comm_rank(comm, &rank);
	assert(counts[rank] == int(params.size()*d0.ICount()));

	data_sens = HMMPI::Mat();
	if (rank == 0)
		data_sens = HMMPI::Mat(smry_len, params.size(), 0);
	data_sens_loc = Gr.Tr();								// both N and A parameters are kept!
	MPI_Gatherv(data_sens_loc.ToVector().data(), params.size()*d0.ICount(), MPI_DOUBLE, data_sens.ToVectorMutable().data(), counts.data(), displs.data(), MPI_DOUBLE, 0, comm);

	return res;
}
//---------------------------------------------------------------------------
std::vector<double> PM_DataProxy::ObjFuncHess_l(const std::vector<double> &params, int l)
{
	if (comm == MPI_COMM_NULL)
		return std::vector<double>(params.size(), 0);

	if (l < 0 || l >= (int)params.size())
		throw HMMPI::Exception("l вне допустимого диапазона в PM_DataProxy::ObjFuncHess_l", "l out of range in PM_DataProxy::ObjFuncHess_l");

	// run common parts
	for (auto &s : starts)
		if (!s.is_empty)
		{
			if (l == 0)
			{
				s.ObjFuncCommon(params);
				s.ObjFuncGradCommon(params);
			}
			s.ObjFuncHess_lCommon(params, l);
		}

	// initialize data structures
	if (l == 0)										// calculate only for l == 0, and then reuse
	{
		resid_loc = std::vector<double>(d0.ICount());			// local residual
		Gr_loc = HMMPI::Mat(params.size(), d0.ICount(), 0);		// all local gradients
	}
	std::vector<double> gr_l(d0.ICount());			// l-component of gradients * 2
	HMMPI::Mat Hl(params.size(), d0.ICount(), 0);	// all local l-columns of Hessians

	for (size_t i = 0; i < d0.ICount(); i++)		// find objective functions, gradients, and l-columns of Hessians for all local data points "i"
	{
		if (l == 0)
		{
			resid_loc[i] = 2*(ends[i].ObjFuncPrivate() - d0(i, 0));		// 2*(model - observed) = 2*(d_m - d_o)
			std::vector<double> grad = ends[i].ObjFuncGradPrivate();
			for (size_t j = 0; j < grad.size(); j++)
				Gr_loc(j, i) = grad[j];
		}

		gr_l[i] = 2*Gr_loc(l, i);

		std::vector<double> hessl = ends[i].ObjFuncHess_lPrivate();
		for (size_t j = 0; j < hessl.size(); j++)
			Hl(j, i) = hessl[j];
	}

	if (l == 0)
		resid_loc = *BDC / resid_loc;		// 2 * C^(-1) * (d_m - d_o)

	gr_l = *BDC / gr_l;						// C^(-1) * gr_l
	std::vector<double> sum = (HMMPI::Mat(Hl*resid_loc) + HMMPI::Mat(Gr_loc*gr_l)).ToVector();

	std::vector<double> res(params.size(), 0);
	MPI_Reduce(sum.data(), res.data(), res.size(), MPI_DOUBLE, MPI_SUM, 0, comm);

	if (l == (int)params.size()-1)			// clear auxiliary objects to prevent inadvertent reuse
	{
		resid_loc = std::vector<double>();
		Gr_loc = HMMPI::Mat();
	}

	return res;
}
//---------------------------------------------------------------------------
HMMPI::Mat PM_DataProxy::ObjFuncFisher_dxi(const std::vector<double> &params, const int i, int r)
{
	if (i == 0)
		ObjFuncGrad(params);						// calculate data_sens_loc
	HMMPI::Mat diS = ObjFuncSens_dxi(params, i);	// Sens derivative (local part)

	HMMPI::Mat M1 = diS.Tr() * (*BDC / data_sens_loc);
	M1 += M1.Tr();

	HMMPI::Mat res;
	int rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);
	assert(r >= 0 && r < size);
	if (rank == r)
		res = HMMPI::Mat(params.size(), params.size(), 0);
	MPI_Reduce(M1.ToVector().data(), res.ToVectorMutable().data(), params.size()*params.size(), MPI_DOUBLE, MPI_SUM, r, comm);		 	// element-wise summation of matrices

	return res;
}
//---------------------------------------------------------------------------
void PM_DataProxy::PerturbData()
{
	if (comm == MPI_COMM_NULL)
		return;

	int rank = 0, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	if (RndN == 0)
		throw HMMPI::Exception("RndN == 0 in PM_DataProxy::PerturbData");
	std::vector<double> stdNorm = RndN->get(data_size);

	std::vector<int> data_ind = BDC->Data_ind();
	if ((int)data_ind.size() != size + 1)
		throw HMMPI::Exception("Некорректная длина data_ind в PM_DataProxy::PerturbData", "Wrong data_ind size in PM_DataProxy::PerturbData");

	// create local Normal r.v.
	std::vector<double> local_rn(data_ind[rank+1] - data_ind[rank]);
	std::vector<int> counts, displs;
	data_ind_count_displ(counts, displs);
	MPI_Scatterv(stdNorm.data(), counts.data(), displs.data(), MPI_DOUBLE, local_rn.data(), local_rn.size(), MPI_DOUBLE, 0, comm);

	d0 = d0_orig + (*BDC) % local_rn;

#ifdef WRITE_PET_DATA
	std::vector<double> d0_rpt(data_size);
	MPI_Gatherv(d0.ToVector().data(), d0.ToVector().size(), MPI_DOUBLE, d0_rpt.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, comm);	// NOTE: gather is done to comm-RANKS-0, but below the file output is for MPI_COMM_WORLD-RANK-0
	if (RNK == 0 && rank == 0)																												// which may not coincide, resulting in no output
	{
		FILE *fpet = fopen("pet_DataProxy.txt", "w");
		if (fpet != NULL)
		{
			HMMPI::Mat(d0_rpt).SaveASCII(fpet);
			fclose(fpet);
		}
	}
#endif
}
//---------------------------------------------------------------------------
std::string PM_DataProxy::proc_msg() const				// sync message showing the number of 'ends' (data points) on each rank; and the effective rank of kriging matrix
{
	// although the required info can be obtained from Data_ind(), here it is directly extracted from ends()
	int size;
	MPI_Comm_size(comm, &size);

	std::vector<int> ends_size(size);
	int loc_size = ends.size();
	MPI_Allgather(&loc_size, 1, MPI_INT, ends_size.data(), 1, MPI_INT, comm);

	std::string msg = (std::string)HMMPI::MessageRE("Точки данных по процессам: ", "Data points on processors: ") + HMMPI::ToString(ends_size, "%d", ", ");
	return msg + PM_Proxy::proc_msg();
}
//---------------------------------------------------------------------------
// PM_DataProxy2
//---------------------------------------------------------------------------
void PM_DataProxy2::RecalcVals()
{
	assert(starts.size() == ends.size());
	std::string r = "";

	for (size_t i = 0; i < ends.size(); i++)
	{
		if (r.size() > 0)
			r += ", ";

		r += ends[i].RecalcVals();
		r.pop_back();					// popback '\n'
	}

	mat_eff_rank = r + "\n";
}
//---------------------------------------------------------------------------
//void PM_DataProxy2::FitSmryFromFile(std::string fname, int len)			// each ends[i] reads its own file; the data are then Gatherv'd
//{
//	assert(FIT == 0);
//	assert(SMRY == 0);
//	MPI_Bcast(&len, 1, MPI_INT, 0, comm);			// some sync
//
//	// read the individual Ity-files
//	for (size_t i = 0; i < ends.size(); i++)
//	{
//		char repl[HMMPI::BUFFSIZE];
//		sprintf(repl, "_pr%d.txt", ends[i]->pr_ind);
//		int count = 0;
//		std::string fn = HMMPI::Replace(fname, "_pr0.txt", repl, &count);		// modify the file name a bit
//		assert (count == 1);
//
//		ends[i]->FitSmryFromFile(fn, len);			// delegate the file reading; fills ends[i]->FIT
//	}
//
//	// allocate SMRY[len][smry_len] to store the result
//	int smry_len = ModelledDataSize();				// defined on all ranks
//	int rnk = -1;
//	if (comm != MPI_COMM_NULL)
//		MPI_Comm_rank(comm, &rnk);
//
//	if (rnk == 0)
//	{
//		SMRY = new double*[len];
//		for (int i = 0; i < len; i++)
//		{
//			if (smry_len != 0)
//				SMRY[i] = new double[smry_len];
//			else
//				SMRY[i] = 0;
//		}
//	}
//
//	std::vector<double> work(ends.size());			// array for local data-points
//	std::vector<int> counts, displs;
//	data_ind_count_displ(counts, displs);			// sync between ranks
//	for (int i = 0; i < len; i++)					// go through all design points
//	{
//		for (size_t j = 0; j < ends.size(); j++)		// take local data-points
//			work[j] = ends[j]->FIT[i];
//
//		double *recv_buff = 0;
//		if (rnk == 0)
//			recv_buff = SMRY[i];
//		MPI_Gatherv(work.data(), work.size(), MPI_DOUBLE, recv_buff, counts.data(), displs.data(), MPI_DOUBLE, 0, comm);
//	}
//
//	for (size_t j = 0; j < ends.size(); j++)		// free some memory
//	{
//		delete [] ends[j]->FIT;
//		ends[j]->FIT = 0;
//	}
//}
//---------------------------------------------------------------------------
PM_DataProxy2::PM_DataProxy2(PhysModel *pm, Parser_1 *K, KW_item *kw, _proxy_params *config, const HMMPI::BlockDiagMat *bdc, const std::vector<double> &d) : PM_DataProxy(pm, K, kw, config, bdc, d)
{
	name = "PM_DataProxy2";

	starts = std::vector<KrigStart>(ends.size(), KrigStart(K, kw, config));				// create 'starts' of the same size as 'ends' ('ends' were created in PM_DataProxy CTOR)
	for (size_t i = 0; i < ends.size(); i++)				// link starts[i] with ends[i]
		ends[i].set_start(&starts[i], i);

	set_refs();
	do_optimize_krig = (config->opt == "ON");

	if (do_optimize_krig)
	{
		char fname[HMMPI::BUFFSIZE];
		sprintf(fname, "Proxy_opt_KRIG_rnk%d.txt", RNK);
		FILE *f = fopen(fname, "w");
		fclose(f);
	}
}
//---------------------------------------------------------------------------
std::string PM_DataProxy2::proc_msg() const					// sync message showing (1) the number of 'ends' (data points) on each rank; (2) effective rank of kriging matrix[i]; (3) pscale for starts[i]
{
	int size, rank;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	// (1)
	std::vector<int> ends_size(size);
	int loc_size = ends.size();
	MPI_Allgather(&loc_size, 1, MPI_INT, ends_size.data(), 1, MPI_INT, comm);
	std::string msg = (std::string)HMMPI::MessageRE("Точки данных по процессам: ", "Data points on processors: ") + HMMPI::ToString(ends_size, "%d", ", ");

	// (2)
	msg += (std::string)HMMPI::MessageRE("Эффективный ранг матрицы кригинга: ", "Effective rank of kriging matrix: ") + mat_eff_rank;

	if (0)	// turn off pscale reporting
	{
		// (3)
		assert(ends.size() == starts.size());
		std::vector<int> Dind = Data_ind();
		assert(int(Dind.size()) == size+1);
		for (int r = 0; r < size; r++)							// compose the message rank by rank
		{
			std::string msg1 = "";
			if (r == rank)										// rank "r" gets correct message 'msg1'
				for (size_t i = 0; i < starts.size(); i++)
				{
					char msg0[HMMPI::BUFFSIZE];
					sprintf(msg0, "[%d] pr-%d pscale = ", rank, Dind[rank] + (int)i);
					msg1 += msg0;
					msg1 += HMMPI::ToString(starts[i].get_pscale(), "%.4g", ", ");
				}
			MPI_Barrier(comm);
			HMMPI::Bcast_string(msg1, r, comm);					// all ranks get message from rank-r; such message transfer is not much efficient (point-to-point communication would be more efficient, i.e. from "r" to "0")
			if (rank == 0)
				msg += msg1;									// rank-0 collects all the messages
		}
	}

	return msg;
}
//---------------------------------------------------------------------------
// PM_SimProxy
//---------------------------------------------------------------------------
void PM_SimProxy::AddPoints(const std::vector<std::vector<double>> &X0, const std::vector<std::vector<double>> &X1)		// adds different subsets of 'X0' to "starts"; currently 'X1' is not added;
{																														// 'X0' should be sync on 'comm'
	assert(starts.size() == start_to_block.size());					// start_to_block: [0, N_starts) -> [0, N_blocks)
	assert(starts.size() > 0);										// number of starts
	assert(block_starts.size() >= 2);
	if (block_starts.size() > 2)					// if N_blocks > 1, only a single proxy training is possible, and this consistency check for X0 is done
		assert(*--block_starts.end() == (int)X0.size());			// block_starts: [0, N_blocks+1) -> [0, Np]
	assert(X0.size() > 0);											// number of design points
	if (PM != 0)
		assert((int)X0[0].size() == PM->ParamsDim());

	for (size_t i = 0; i < starts.size(); i++)		// each starts[i] will get a different subset from X0
		if (!starts[i].is_empty)
		{
			if (block_starts.size() > 2)
				starts[i].AddPoints(std::vector<std::vector<double>>(X0.begin() + block_starts[start_to_block[i]], X0.begin() + *--block_starts.end()), std::vector<std::vector<double>>());		// empty vector for gradients
			else
				starts[i].AddPoints(X0, std::vector<std::vector<double>>());		// The case of 1 block, block_starts = {0, Np_init}, take whole X0. It may be the first-time training or additional training
		}
}
//---------------------------------------------------------------------------
void PM_SimProxy::RecalcVals()
{
	// form the right hand sides
	std::vector<HMMPI::Mat> ITY_for_start(starts.size());		// each Mat contains multiple right hand sides
	std::vector<HMMPI::Mat> CinvZ_for_start(starts.size());		// each Mat contains multiple solutions
	std::vector<int> offsets(ends.size(), -1);					// shows relative location of ends[i] in ITY_for_start[st_ind] or CinvZ_for_start[st_ind]
	for (size_t i = 0; i < ends.size(); i++)
	{
		const int st_ind = ends[i].start_index;					// start index (this 'start' is not empty)
		assert(st_ind >= 0 && st_ind < (int)starts.size());

		offsets[i] = ITY_for_start[st_ind].ICount();
		std::vector<double> work = ends[i].get_y();
		int wsize = work.size();
		ITY_for_start[st_ind] = std::move(ITY_for_start[st_ind]) || HMMPI::Mat(std::move(work), 1, wsize);
	}

	// solve
	std::vector<int> eff_ranks(starts.size(), 0);
	for (size_t j = 0; j < starts.size(); j++)
		if (!starts[j].is_empty && ITY_for_start[j].Length() > 0)
		{
			size_t len_trend = starts[j].get_multi_ind().size();

			ITY_for_start[j] = ITY_for_start[j].Tr() || HMMPI::Mat(len_trend, ITY_for_start[j].ICount(), 0.0);
			CinvZ_for_start[j] = starts[j].get_sol()->Solve(starts[j].get_C(), ITY_for_start[j]).Tr();	// transpose to convert columns to rows
			eff_ranks[j] = starts[j].get_sol()->rank;
			ITY_for_start[j] = ITY_for_start[j].Tr();													// transpose for subsequent reporting (cols -> rows)
		}

	mat_eff_rank = HMMPI::ToString(eff_ranks, "%4d", "\t");		// not-sync, each proc fills its own "mat_eff_rank"

	// fill all CinvZ's
	for (size_t i = 0; i < ends.size(); i++)
	{
		const int st_ind = ends[i].start_index;					// start index
		const int shift = offsets[i];							// will take row 'shift' from CinvZ_for_start[]
		const double *pbcz = CinvZ_for_start[st_ind].ToVector().data();
		size_t ld = CinvZ_for_start[st_ind].JCount();			// leading dimension (length of one row)

		ends[i].set_CinvZ(std::vector<double>(pbcz + ld*shift, pbcz + ld*(shift+1)));
	}

	if (dump_flag != -1)	// debug output to files
	{
		char fname[100];
		for (size_t i = 0; i < ends.size(); i++)		// for each ends[i] separate files are written
		{
			const int st_ind = ends[i].start_index;		// start index
			const int shift = offsets[i];
			const double *pbcz = CinvZ_for_start[st_ind].ToVector().data();
			size_t ld = CinvZ_for_start[st_ind].JCount();

			sprintf(fname, dump_CinvZ, dump_flag, RNK, ends[i].index);
			FILE *f = fopen(fname, "w");
			HMMPI::Mat(std::vector<double>(pbcz + ld*shift, pbcz + ld*(shift+1))).SaveASCII(f, "%20.16g");
			fclose(f);

			pbcz = ITY_for_start[st_ind].ToVector().data();
			ld = ITY_for_start[st_ind].JCount();

			sprintf(fname, dump_Ity, dump_flag, RNK, ends[i].index);
			f = fopen(fname, "w");
			HMMPI::Mat(std::vector<double>(pbcz + ld*shift, pbcz + ld*(shift+1))).SaveASCII(f, "%20.16g");
			fclose(f);
		}
	}

#ifdef TESTNEWPROXY
	int size00;
	MPI_Comm_size(MPI_COMM_WORLD, &size00);
	std::ofstream file0(HMMPI::stringFormatArr(test_fn, std::vector<int>{RNK}), std::ios_base::app);
	std::cout << "rank " << RNK << ", size " << size00 << "\tPM_SimProxy::RecalcVals(), offsets: " << HMMPI::ToString(offsets, "%d");
	file0 << "rank " << RNK << ", size " << size00 << "\tPM_SimProxy::RecalcVals(), offsets: " << HMMPI::ToString(offsets, "%d");
	file0.close();
#endif
}
//---------------------------------------------------------------------------
PM_SimProxy::PM_SimProxy(PhysModel *pm, Parser_1 *K, KW_item *kw, _proxy_params *config, const HMMPI::BlockDiagMat *bdc, const std::vector<double> &d, const std::vector<int> &b_starts, const std::vector<int> &dp_block) :
		PM_DataProxy(pm, K, kw, config, bdc, d), block_starts(b_starts)				// <sync> "b_starts[0..N_blocks+1)" - array of indices in [0, Np] showing where each block starts
{																					// <sync> "dp_block[0..smry_len)" - index of the first block where the given data point exists (see SimProxyFile)
	if (b_starts.size() < 2)
		throw HMMPI::Exception("b_starts.size() < 2 in PM_SimProxy::PM_SimProxy");

	if (comm == MPI_COMM_NULL)
		return;
	int rank;
	MPI_Comm_rank(comm, &rank);

	name = "PROXY";

	kw->Start_pre();
	DECLKWD(params, KW_parameters, "PARAMETERS");
	kw->Add_pre("PARAMETERS");
	kw->Finish_pre();

	std::vector<int> dp_colors = params->sc_colors_textsmry();				// sync
	assert(dp_block.size() == dp_colors.size());

	dp_block_color = std::vector<std::pair<int, int>>(dp_block.size());		// sync vector of <block, color> for each data point
	for (size_t i = 0; i < dp_block.size(); i++)
		dp_block_color[i] = std::pair<int, int>(dp_block[i], dp_colors[i]);

	dp_block_color = HMMPI::Unique(dp_block_color);							// <sync> unique pairs of <block, color> will define 'starts'; NOTE: 'starts' are indexed exactly as 'dp_block_color'

	starts = std::vector<KrigStart>(dp_block_color.size(), KrigStart());	// create vector of empty "starts"; "ends" remain from PM_DataProxy CTOR
	start_Nends = std::vector<int>(starts.size(), 0);
	const KrigStart ks_copy(K, kw, config);									// will be used to fill certain starts[i]

	start_to_block = std::vector<int>(dp_block_color.size());				// <sync> maps [0, N_starts) -> [0, N_blocks), giving the block number for each start[i]
	for (size_t i = 0; i < dp_block_color.size(); i++)
		start_to_block[i] = dp_block_color[i].first;

	std::vector<int> data_ind = BDC->Data_ind();
	for (size_t i = 0; i < ends.size(); i++)
	{
		int global_ind = i + data_ind[rank];								// global index of the given data point (ends[i])
		assert(global_ind >= 0 && global_ind < (int)dp_block.size());

		const std::pair<int, int> local_dp_ind(dp_block[global_ind], dp_colors[global_ind]);			// <block, color> index for the given local data point
		const auto it = std::find(dp_block_color.begin(), dp_block_color.end(), local_dp_ind);
		assert(it != dp_block_color.end());

		const int st_ind = it - dp_block_color.begin();						// start index corresponding to ends[i]
		assert(0 <= st_ind && st_ind < (int)starts.size());

		start_Nends[st_ind] += 1;											// each rank does its own increments for 'start_Nends'
		if (starts[st_ind].is_empty)
		{
			starts[st_ind] = ks_copy;
			starts[st_ind].set_pscale(params->uniq_sc[dp_block_color[st_ind].second]);					//  uniq_sc - [N_colors x fulldim]
		}

		ends[i].set_start(&starts[st_ind], st_ind);

#ifdef PROXY_DEBUG
		std::cout << "[" << RNK << "] DEBUG ends " << i << " ->global " << global_ind << " STARTS indx " << st_ind << "\n";
#endif
	}
	set_refs();

	MPI_Allreduce(MPI_IN_PLACE, start_Nends.data(), start_Nends.size(), MPI_INT, MPI_SUM, comm);

#ifdef PROXY_DEBUG
	std::cout << "[" << RNK << "] DEBUG PM_SimProxy starts count " << starts.size() << ", ENDS count " << ends.size() << "\n";
#endif

	print_comms();
}
//---------------------------------------------------------------------------
PM_SimProxy::PM_SimProxy(const PM_SimProxy &p) : PM_DataProxy(p), block_starts(p.block_starts), start_to_block(p.start_to_block), dp_block_color(p.dp_block_color), start_Nends(p.start_Nends)
{
};
//---------------------------------------------------------------------------
std::string PM_SimProxy::proc_msg() const					// sync message showing (1) the number of 'ends' (data points) on each process; (2) effective rank of kriging matrix for all 'starts'
{															// (3) pscale for all 'starts'
	int size, rank;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	// (1)
	std::vector<int> ends_size(size);
	int loc_size = ends.size();
	MPI_Allgather(&loc_size, 1, MPI_INT, ends_size.data(), 1, MPI_INT, comm);
	std::string res = (std::string)HMMPI::MessageRE("Точки данных по процессам: ", "Data points on processors: ") + HMMPI::ToString(ends_size, "%d", ", ");

	// (2), gather all mat_eff_rank's
	res += (std::string)HMMPI::MessageRE("Эффективный ранг матрицы кригинга (столбцы <-> MPI-процессы), Mat(pt_bl,color) - data_pts:\n",
										 "Effective rank of kriging matrix (columns <-> MPI-processes), Mat(pt_bl,color) - data_pts:\n");
	std::string msg = "";
	for (int r = 0; r < size; r++)							// compose the message rank by rank
	{
		std::string msg1 = "";
		if (r == rank)										// rank "r" gets correct message 'msg1'
			msg1 = mat_eff_rank;

		MPI_Barrier(comm);
		HMMPI::Bcast_string(msg1, r, comm);					// all ranks get message from rank-r; such message transfer is not much efficient (point-to-point communication would be more efficient, i.e. from "r" to "0")
		if (rank == 0)
			msg += msg1;									// rank-0 collects all the messages
	}

	HMMPI::Bcast_string(msg, 0, comm);
	std::vector<std::string> parts;

	HMMPI::tokenize(msg, parts, "\t\n", true);				// res = msg^{transposed}
	assert(parts.size() == starts.size()*size);
	assert(dp_block_color.size() == starts.size());

	char buff[HMMPI::BUFFSIZE + 5], buff1[HMMPI::BUFFSIZE];
	for (size_t i = 0; i < starts.size(); i++)
	{
		sprintf(buff1, "%zu(%d,%3d) -%4d", i, dp_block_color[i].first, dp_block_color[i].second, start_Nends[i]);	// start(point-block,color) - data_pts
		sprintf(buff, "%17s: ", buff1);
		res += buff;
		for (int r = 0; r < size; r++)
		{
			res += parts[r*starts.size() + i];													// effective rank of kriging matrix
			if (r < size-1)
				res += ", ";
		}
		res += "\n";
	}

	// (3) optional stuff
#if 0
	{
		std::string msg1 = "";
		for (int r = 0; r < size; r++)						// compose the message rank by rank
		{
			sprintf(buff, "Rank-%d pscale\n", rank);
			std::string loc_msg = buff;
			for (size_t i = 0; i < starts.size(); i++)
			{
				sprintf(buff, "%3zu: ", i);
				loc_msg += buff;
				if (!starts[i].is_empty)
					loc_msg += HMMPI::ToString(starts[i].get_pscale(), "%g");
				else
					loc_msg += "none\n";
			}

			MPI_Barrier(comm);
			HMMPI::Bcast_string(loc_msg, r, comm);			// all ranks get message from rank-r
			if (rank == 0)
				msg1 += loc_msg;							// rank-0 collects all the messages
		}
		HMMPI::Bcast_string(msg1, 0, comm);
		res += msg1;
	}
#endif

	return res + "\n";
}
//---------------------------------------------------------------------------
// KrigCorr
//---------------------------------------------------------------------------
void KrigCorr::CalculateDerivatives(const std::vector<double> &par) const				// NOTE cov. func. calls are made here! (cf. smooth_at_nugget)
{
	if (par != par_cache)
		is_valid = 0;

	if (!(is_valid & 4))
	{
		if (func == nullptr)
			throw HMMPI::Exception("func == 0 in KrigCorr::CalculateDerivatives");

		const double r = par[1];
		const double Nu = par[2];

		func->SetNugget(par[0]);					// set up "func"
		if (dynamic_cast<HMMPI::CorrMatern*>(func) != nullptr)
			dynamic_cast<HMMPI::CorrMatern*>(func)->SetNu(Nu);

		HMMPI::Func1D_corr *func_var = 0;
		if (tot_ind[2] != (size_t)-1)				// "nu" is active
			func_var = func->Copy();

		HMMPI::CorrMatern *func_var_mat = dynamic_cast<HMMPI::CorrMatern*>(func_var);				// will be used for perturbing "nu"

		d_R = std::vector<HMMPI::Mat>(3);
		std::vector<std::function<double (double)>> F1(3);											// differentiating functions

		F1[0] = [this, &par, r](double x) -> double {return x == 0 ? 0 : -func->f(x/r)/(1 - par[0]);};			// d/d_nugget
		F1[1] = [this, r](double x) -> double {return x == 0 ? 0 : -func->df(x/r)*x/(r*r);};					// d/d_r
		if (func_var_mat != nullptr)
		{
			auto f_y = [func_var_mat, r](double x, double nu) -> double {func_var_mat->SetNu(nu); return func_var_mat->f(x/r);};					// (x, nu) -> R(x, r, nu)
			F1[2] = [this, f_y, Nu](double x) -> double {return x == 0 ? 0 : HMMPI::NumD(std::bind(f_y, x, std::placeholders::_1), Nu, dh, oh);};	// d/d_nu
		}
		else
			F1[2] = [](double x) -> double {return 0;};

		for (int i = 0; i < 3; i++)
		{
			HMMPI::Mat work = *D;
			work.Func(F1[i]);
			d_R[i] = std::move(work);
		}

		delete func_var;

		par_cache = par;
		is_valid |= 4;
	}
}
//---------------------------------------------------------------------------
void KrigCorr::CalculateDerivatives2(const std::vector<double> &par) const				// NOTE cov. func. calls are made here! (cf. smooth_at_nugget)
{
	if (par != par_cache)
		is_valid = 0;

	if (!(is_valid & 8))
	{
		if (func == nullptr)
			throw HMMPI::Exception("func == 0 in KrigCorr::CalculateDerivatives2");

		const double r = par[1];
		const double Nu = par[2];

		func->SetNugget(par[0]);					// set up "func"
		if (dynamic_cast<HMMPI::CorrMatern*>(func) != nullptr)
			dynamic_cast<HMMPI::CorrMatern*>(func)->SetNu(Nu);

		HMMPI::Func1D_corr *func_var = 0;
		if (tot_ind[2] != (size_t)-1)				// "nu" is active
			func_var = func->Copy();

		HMMPI::CorrMatern *func_var_mat = dynamic_cast<HMMPI::CorrMatern*>(func_var);				// will be used for perturbing "nu"

		d2_R = HMMPI::Vector2<HMMPI::Mat>(3, 3);
		HMMPI::Vector2<std::function<double (double)>> F2(3, 3);

		F2(0, 0) = [](double x) -> double {return 0;};																					// d2/d_nugget^2
		F2(0, 1) = [this, &par, r](double x) -> double {return x == 0 ? 0 : func->df(x/r)*x/(r*r)/(1 - par[0]);};						// d2/d_nugget*d_r
		F2(1, 1) = [this, r](double x) -> double {double q = x/r; return x == 0 ? 0 : (func->d2f(q)*q + 2*func->df(q)) * q/(r*r);};		// d2/d_r^2
		if (func_var_mat != nullptr)																// numerical procedure
		{
			auto f0y = [func_var_mat, &par, r](double x, double nu) -> double {func_var_mat->SetNu(nu); return -func_var_mat->f(x/r)/(1 - par[0]);};	// (x, nu) -> dR/d_nugget(x, r, nu)
			F2(0, 2) = [this, f0y, Nu](double x) -> double {return x == 0 ? 0 : HMMPI::NumD(std::bind(f0y, x, std::placeholders::_1), Nu, dh, oh);};	// d2/d_nugget*d_nu

			auto f1y = [func_var_mat, r](double x, double nu) -> double {func_var_mat->SetNu(nu); return -func_var_mat->df(x/r)*x/(r*r);};				// (x, nu) -> dR/d_r(x, r, nu)
			F2(1, 2) = [this, f1y, Nu](double x) -> double {return x == 0 ? 0 : HMMPI::NumD(std::bind(f1y, x, std::placeholders::_1), Nu, dh, oh);};	// d2/d_r*d_nu

			auto f2y = [func_var_mat, r](double x, double nu) -> double {func_var_mat->SetNu(nu); return func_var_mat->f(x/r);};						// (x, nu) -> R(x, r, nu)
			F2(2, 2) = [this, f2y, Nu](double x) -> double {return x == 0 ? 0 : HMMPI::NumD2(std::bind(f2y, x, std::placeholders::_1), Nu, dh, oh);};	// d2/d_nu^2
		}
		else
			F2(2, 2) = F2(1, 2) = F2(0, 2) = F2(0, 0);

		for (int i = 0; i < 3; i++)
			for (int j = i; j < 3; j++)
			{
				HMMPI::Mat work = *D;
				work.Func(F2(i, j));
				d2_R(i, j) = std::move(work);
			}

		delete func_var;

		par_cache = par;
		is_valid |= 8;
	}
}
//---------------------------------------------------------------------------
double KrigCorr::obj_func(const std::vector<double> &params) const
{
	if (params.size() != 3)
		throw HMMPI::Exception("params.size() != 3 in KrigCorr::obj_func");

	CalculateR(params);
	double n = R.ICount();
#ifdef KRIG_CORR_DET_CHOL
	return exp(R.LnDetSPO()/n);
#else
	int sign = 0;
	double lndet = R.LnDetSY(sign);
	if (sign < 0 && K0 != nullptr)
	{
		K0->TotalWarnings++;
		K0->AppText(HMMPI::MessageRE("ПРЕДУПРЕЖДЕНИЕ: Отрицательный определитель для корреляционной матрицы\n",
									 "WARNING: Negative determinant for correlation matrix\n"));
	}
	return exp(lndet/n);
#endif
}
//---------------------------------------------------------------------------
std::vector<double> KrigCorr::obj_func_grad(const std::vector<double> &params) const
{
	if (params.size() != 3)
		throw HMMPI::Exception("params.size() != 3 in KrigCorr::obj_func_grad");

	CalculateR(params);
	CalculateDerivatives(params);

#ifdef KRIG_CORR_DET_CHOL
	HMMPI::Mat invR = R.InvSPO();
	double lndet = R.LnDetSPO();		// reuses Cholesky decomposition
#else
	int sign = 0;
	HMMPI::Mat invR = R.InvSY();
	double lndet = R.LnDetSY(sign);		// reuses DSYTRF decomposition
	if (sign < 0 && K0 != nullptr)
	{
		K0->TotalWarnings++;
		K0->AppText(HMMPI::MessageRE("ПРЕДУПРЕЖДЕНИЕ: Отрицательный определитель для корреляционной матрицы\n",
									 "WARNING: Negative determinant for correlation matrix\n"));
	}
#endif
	double n = R.ICount();

	std::vector<double> res(3);
	for (int i = 0; i < 3; i++)
		res[i] = exp(lndet/n)/n * (invR*d_R[i]).Trace();

	return res;
}
//---------------------------------------------------------------------------
HMMPI::Mat KrigCorr::obj_func_hess(const std::vector<double> &params) const
{
	if (params.size() != 3)
		throw HMMPI::Exception("params.size() != 3 in KrigCorr::obj_func_hess");

	CalculateR(params);
	CalculateDerivatives(params);
	CalculateDerivatives2(params);

#ifdef KRIG_CORR_DET_CHOL
	HMMPI::Mat invR = R.InvSPO();
	double lndet = R.LnDetSPO();		// reuses Cholesky decomposition
#else
	int sign = 0;
	HMMPI::Mat invR = R.InvSY();
	double lndet = R.LnDetSY(sign);		// reuses DSYTRF decomposition
	if (sign < 0 && K0 != nullptr)
	{
		K0->TotalWarnings++;
		K0->AppText(HMMPI::MessageRE("ПРЕДУПРЕЖДЕНИЕ: Отрицательный определитель для корреляционной матрицы\n",
									 "WARNING: Negative determinant for correlation matrix\n"));
	}
#endif
	double n = R.ICount();

	std::vector<HMMPI::Mat> work(3);	// invR * dR/dt_i
	for (int i = 0; i < 3; i++)
		work[i] = invR*d_R[i];

	HMMPI::Mat res(3, 3, 0.0);
	for (int i = 0; i < 3; i++)
		for (int j = i; j < 3; j++)
			res(j, i) = res(i, j) = exp(lndet/n)/n * (work[i].Trace()*work[j].Trace()/n - (work[i]*work[j]).Trace() + (invR*d2_R(i, j)).Trace());

	return res;
}
//---------------------------------------------------------------------------
KrigCorr::KrigCorr() : PhysModel(MPI_COMM_SELF), is_valid(0), K0(0), D(0), func(0), cfunc(0)
{
	name = "KRIGCORR";

	con = nullptr;
	init_msg = "";
}
//---------------------------------------------------------------------------
KrigCorr::KrigCorr(const HMMPI::Func1D_corr *cf) : PhysModel(MPI_COMM_SELF), is_valid(0), K0(0), D(0), func(cf->Copy()), cfunc(cf)
{
	// comm = MPI_COMM_SELF, since communication is not needed
	name = "KRIGCORR";

	con = nullptr;
	init_msg = "";
}
//---------------------------------------------------------------------------
KrigCorr::KrigCorr(const HMMPI::Func1D_corr *cf, Parser_1 *K, _proxy_params *config) : KrigCorr(cf)
{
	DECLKWD(limitsKrig, KW_limitsKrig, "LIMITSKRIG");

	name = "KRIGCORR";
	K0 = K;
	is_valid = 0;

	if (limitsKrig->GetState() == "" && dynamic_cast<KW_proxy*>(config) != nullptr)			// LIMITSKRIG is defined _AND_ config == PROXY_CONFIG
	{
		if (limitsKrig->init.size() != 3)
			throw HMMPI::Exception("В LIMITSKRIG ожидается 3 параметра (3 строки)", "Expected 3 parameters (3 lines) in LIMITSKRIG");
		init = limitsKrig->init;
		act_ind = limitsKrig->get_act_ind();
		tot_ind = limitsKrig->get_tot_ind();
		con = limitsKrig;

		init_msg = HMMPI::MessageRE("Параметры корреляции для кригинга взяты из \"LIMITSKRIG\"", "Correlation parameters for kriging are taken from \"LIMITSKRIG\"");
	}
	else																					// use PROXY_CONFIG or MODEL
	{
		if (config->GetState() != "")
		{
			if (dynamic_cast<KW_proxy*>(config) != nullptr)		// PROXY_CONFIG
				throw HMMPI::Exception("LIMITSKRIG и PROXY_CONFIG не заданы, либо заданы с ошибками", "LIMITSKRIG and PROXY_CONFIG are not defined, or defined with errors");
			else												// MODEL
				throw HMMPI::Exception("MODEL не задано, либо задано с ошибками", "MODEL is not defined, or defined with errors");
		}
		init = std::vector<double>{config->nugget, config->R, config->nu};
		act_ind = tot_ind = std::vector<size_t>();	// not supposed to be used
		con = nullptr;								// not supposed to be used

		init_msg = (std::string)HMMPI::MessageRE("Параметры корреляции для кригинга взяты из \"", "Correlation parameters for kriging are taken from \"") + config->name + "\"";
	}
}
//---------------------------------------------------------------------------
KrigCorr::KrigCorr(const KrigCorr &kc) : PhysModel(kc), par_cache(kc.par_cache), is_valid(0), R(kc.R), d_R(kc.d_R), d2_R(kc.d2_R), K0(kc.K0), D(0), cfunc(kc.cfunc), init_msg(kc.init_msg)
{
	if (kc.func != nullptr)
		func = kc.func->Copy();
	else
		func = nullptr;
}
//---------------------------------------------------------------------------
const KrigCorr &KrigCorr::operator=(const KrigCorr &p)
{
	PhysModel::operator=(p);

	par_cache = p.par_cache;
	is_valid = 0;
	R = p.R;
	d_R = p.d_R;
	d2_R = p.d2_R;
	K0 = p.K0;
	D = 0;
	cfunc = p.cfunc;
	init_msg = p.init_msg;

	if (p.func != nullptr)
		func = p.func->Copy();
	else
		func = nullptr;

	return *this;
}
//---------------------------------------------------------------------------
const HMMPI::Func1D_corr *KrigCorr::CalculateR(const std::vector<double> &par) const
{
	const HMMPI::Func1D_corr *corr_func;
	if (func == nullptr)				// case of correlation != GAUSS or MATERN
	{
		corr_func = cfunc;
	}
	else								// correlation == GAUSS or MATERN
	{
		func->SetNugget(par[0]);
		if (dynamic_cast<HMMPI::CorrMatern*>(func) != nullptr)
			dynamic_cast<HMMPI::CorrMatern*>(func)->SetNu(par[2]);
		corr_func = func;
	}

	if (par != par_cache)
		is_valid = 0;

	if (!(is_valid & 2))
	{
		// recalculate R
		auto f0 = [corr_func, &par](double x) -> double {return corr_func->f(x/par[1]);};		// lambda-function
		R = *D;
		R.Func(f0);

		par_cache = par;
		is_valid |= 2;
	}

	return corr_func;
}
//---------------------------------------------------------------------------
const HMMPI::Func1D_corr *KrigCorr::GetFuncFromCalculateR() const
{
	return func == nullptr ? cfunc : func;
}
//---------------------------------------------------------------------------
double KrigCorr::obj_func_work(const std::vector<double> &params)
{
	return obj_func(params);
}
//---------------------------------------------------------------------------
std::vector<double> KrigCorr::obj_func_grad_work(const std::vector<double> &params)
{
	return obj_func_grad(params);
}
//---------------------------------------------------------------------------
HMMPI::Mat KrigCorr::obj_func_hess_work(const std::vector<double> &params)
{
	return obj_func_hess(params);
}
//---------------------------------------------------------------------------
// KrigSigma
//---------------------------------------------------------------------------
const HMMPI::Mat &KrigSigma::get_U(const std::vector<double> &params) const
{
	if (params != par_cache)
		is_valid = false;

	if (!is_valid)
	{
		ref->CalculateR(params);
#ifdef KRIG_CORR_DET_CHOL
		HMMPI::Mat invR = ref->Get_R().InvSPO();
#else
		HMMPI::Mat invR = ref->Get_R().InvSY();
#endif
		HMMPI::Mat invR_F = invR * (*F);
		HMMPI::SolverDGESV Invert;
		U_cache = invR - invR_F * Invert.Solve(F->Tr()*invR_F, invR_F.Tr());

		par_cache = params;
		is_valid = true;
	}

	return U_cache;
}
//---------------------------------------------------------------------------
double KrigSigma::sigma2(const std::vector<double> &params) const
{
	HMMPI::Mat Y = *Ys;
	double n = Ys->size();

	return InnerProd(Y, get_U(params)*Y)/n;
}
//---------------------------------------------------------------------------
std::vector<double> KrigSigma::sigma2_grad(const std::vector<double> &params) const
{
	std::vector<double> res(3);

	HMMPI::Mat Y = *Ys;
	double n = Ys->size();

	ref->CalculateDerivatives(params);
	HMMPI::Mat U = get_U(params);

	for (int i = 0; i < 3; i++)
		res[i] = -InnerProd(Y, U.Tr()*ref->d_R[i]*U*Y)/n;

	return res;
}
//---------------------------------------------------------------------------
HMMPI::Mat KrigSigma::sigma2_Hess(const std::vector<double> &params) const
{
	HMMPI::Mat res(3, 3, 0.0);

	HMMPI::Mat Y = *Ys;
	double n = Ys->size();

	ref->CalculateDerivatives(params);
	ref->CalculateDerivatives2(params);
	HMMPI::Mat U = get_U(params);
	HMMPI::Mat Ut = U.Tr();

	for (int i = 0; i < 3; i++)
		for (int j = i; j < 3; j++)
			res(j, i) = res(i, j) = 2/n*InnerProd(Y, Ut*ref->d_R[i]*Ut*ref->d_R[j]*U*Y) - InnerProd(Y, Ut*ref->d2_R(i, j)*U*Y)/n;

	return res;
}
//---------------------------------------------------------------------------
KrigSigma::KrigSigma(Parser_1 *K, _proxy_params *config) : KrigSigma()
{
	DECLKWD(limitsKrig, KW_limitsKrig, "LIMITSKRIG");

	name = "KRIGSIGMA";
	is_valid = false;
	if (limitsKrig->GetState() == "" && dynamic_cast<KW_proxy*>(config) != nullptr)		// LIMITSKRIG is defined _AND_ config == PROXY_CONFIG
	{
		if (limitsKrig->init.size() != 3)
			throw HMMPI::Exception("В LIMITSKRIG ожидается 3 параметра (3 строки)", "Expected 3 parameters (3 lines) in LIMITSKRIG");
		init = limitsKrig->init;
		act_ind = limitsKrig->get_act_ind();
		tot_ind = limitsKrig->get_tot_ind();
		con = limitsKrig;
	}
	else																				// use PROXY_CONFIG or MODEL
	{
		if (config->GetState() != "")
		{
			if (dynamic_cast<KW_proxy*>(config) != nullptr)		// PROXY_CONFIG
				throw HMMPI::Exception("LIMITSKRIG и PROXY_CONFIG не заданы, либо заданы с ошибками", "LIMITSKRIG and PROXY_CONFIG are not defined, or defined with errors");
			else												// MODEL
				throw HMMPI::Exception("MODEL не задано, либо задано с ошибками", "MODEL is not defined, or defined with errors");
		}
		init = std::vector<double>{config->nugget, config->R, config->nu};
		act_ind = tot_ind = std::vector<size_t>();	// not supposed to be used
		con = nullptr;								// not supposed to be used
	}
}
//---------------------------------------------------------------------------
const KrigSigma &KrigSigma::operator=(const KrigSigma &p)
{
	PhysModel::operator=(p);

	is_valid = false;
	F = 0;
	Ys = 0;
	ref = 0;
	return *this;
};
//---------------------------------------------------------------------------
double KrigSigma::obj_func_work(const std::vector<double> &params)
{
#ifdef OBJ_FUNC_SIGMA_2			// test mode
	return sigma2(params);
#else							// work mode
	return sigma2(params) * ref->obj_func(params);
#endif
}
//---------------------------------------------------------------------------
std::vector<double> KrigSigma::obj_func_grad_work(const std::vector<double> &params)
{
#ifdef OBJ_FUNC_SIGMA_2			// test mode
	return sigma2_grad(params);
#else							// work mode
	double sigma = sigma2(params);
	double corr = ref->obj_func(params);
	HMMPI::Mat grad_sigma = sigma2_grad(params);
	HMMPI::Mat grad_corr = ref->obj_func_grad(params);

	return (sigma*std::move(grad_corr) + corr*std::move(grad_sigma)).ToVector();
#endif
}
//---------------------------------------------------------------------------
HMMPI::Mat KrigSigma::obj_func_hess_work(const std::vector<double> &params)
{
#ifdef OBJ_FUNC_SIGMA_2			// test mode
	return sigma2_Hess(params);
#else							// work mode
	double sigma = sigma2(params);
	double corr = ref->obj_func(params);
	HMMPI::Mat grad_sigma = sigma2_grad(params);
	HMMPI::Mat grad_corr = ref->obj_func_grad(params);
	HMMPI::Mat Hess_sigma = sigma2_Hess(params);
	HMMPI::Mat Hess_corr = ref->obj_func_hess(params);
	HMMPI::Mat prod = OuterProd(grad_sigma, grad_corr);

	return sigma*std::move(Hess_corr) + corr*std::move(Hess_sigma) + prod + prod.Tr();
#endif
}
//---------------------------------------------------------------------------
// KrigStart
//---------------------------------------------------------------------------
void KrigStart::rescale_vec(std::vector<double> &v) const						// rescales v using 'pscale'
{
	assert(v.size() == pscale.size());
	for (size_t j = 0; j < pscale.size(); j++)
		v[j] *= pscale[j];
}
//---------------------------------------------------------------------------
void KrigStart::rescale_vecs(std::vector<std::vector<double>> &v) const			// rescales v[i] using 'pscale'
{
	for (size_t i = 0; i < v.size(); i++)
		rescale_vec(v[i]);
}
//---------------------------------------------------------------------------
void KrigStart::rescaleBACK_vec(std::vector<double> &v) const
{
	assert(v.size() == pscale.size());
	for (size_t j = 0; j < pscale.size(); j++)
		v[j] /= pscale[j];
}
//---------------------------------------------------------------------------
void KrigStart::rescaleBACK_vecs(std::vector<std::vector<double>> &v) const
{
	for (size_t i = 0; i < v.size(); i++)
		rescaleBACK_vec(v[i]);
}
//---------------------------------------------------------------------------
void KrigStart::push_back_data(const std::vector<std::vector<double>> &X0, const std::vector<std::vector<double>> &X1)		// appends whole X0 to X_0, X1 to X_1
{
	if (X_0.size() > 0 && X0.size() > 0 && X_0[0].size() != X0[0].size())
		throw HMMPI::Exception("Не совпадают размерности векторов в KrigStart::push_back_data", "Vector dimensions do not match in KrigStart::push_back_data");
	if (X_1.size() > 0 && X1.size() > 0 && X_1[0].size() != X1[0].size())
		throw HMMPI::Exception("Не совпадают размерности векторов в KrigStart::push_back_data", "Vector dimensions do not match in KrigStart::push_back_data");

	HMMPI::VecAppend(X_0, X0);
	HMMPI::VecAppend(X_1, X1);
}
//---------------------------------------------------------------------------
KrigStart::KrigStart(Parser_1 *K, KW_item *kw, _proxy_params *config) :		// easy constructor; all data are taken from keywords of "K"; 1st LINSOLVER is used; "kw" is used only to handle prerequisites; "config" can be PROXY_CONFIG or MODEL
		trend_order(config->trend), kc(config->corr, K, config), dump_flag(-1), index(-1), is_empty(false)
{
	DECLKWD(solver, KW_LinSolver, "LINSOLVER");
	DECLKWD(params, KW_parameters, "PARAMETERS");
	const ParamsInterface *par_interface = params->GetParamsInterface();

	kw->Start_pre();
	kw->Add_pre(config->name);
	kw->Add_pre("LINSOLVER");
	kw->Finish_pre();

	int dim = par_interface->act.size();

	if (solver->SolSize() < 1)
		throw HMMPI::Exception("В LINSOLVER ничего не задано", "Empty LINSOLVER");

	grad_inds = std::vector<size_t>(config->ind_grad_comps.begin(), config->ind_grad_comps.end());
	pscale = std::vector<double>(dim, 1.0);			// default

	if (grad_inds.size() > 0 && *--grad_inds.end() >= (size_t)dim)
		throw HMMPI::Exception(HMMPI::stringFormatArr("Максимальный индекс компоненты градиента {0:%zu} >= DIM",
													  "Maximum gradient component index {0:%zu} >= DIM", *--grad_inds.end()));

	R = 0;
	func = 0;
	sol = solver->Sol(0);
	multi_ind = make_multi_ind(dim, config->trend);
	if (dynamic_cast<HMMPI::VarGauss*>(config->corr) != 0 && config->trend < 0)
		throw HMMPI::Exception("Для вариограммы VARGAUSS тренд должен быть порядка >= 0", "For VARGAUSS variogram the trend order should be >= 0");
}
//---------------------------------------------------------------------------
KrigStart::KrigStart(const KrigStart &p) : smooth_at_nugget(p.smooth_at_nugget)
{
	*this = p;
}
//---------------------------------------------------------------------------
const KrigStart &KrigStart::operator=(const KrigStart &p)
{
	trend_order = p.trend_order;
	D = p.D;
	DG = p.DG;
	DGG = p.DGG;
	X_0 = p.X_0;
	X_1 = p.X_1;
	pscale = p.pscale;
	grad_inds = p.grad_inds;
	N = p.N;
	Nfull = p.Nfull;
	C = p.C;
	kc = p.kc;
	R = p.R;
	sol = p.sol;
	multi_ind = p.multi_ind;
	C0 = p.C0;
	gM = p.gM;
	lM = p.lM;
	dump_flag = p.dump_flag;
	index = -1;
	is_empty = p.is_empty;
	func = kc.GetFuncFromCalculateR();

	return *this;
}
//---------------------------------------------------------------------------
std::vector<size_t> KrigStart::PointsSubsetKS(const std::vector<std::vector<double>> &x0, size_t count, bool &all_taken) const	// selects 'count' points from 'x0', returning their indices; IndSignificant() is used;
{																																// the distance matrix which guides the selection comes from X_0 + x0
	const size_t len_old = X_0.size();																							// if X_0 is empty, full x0 indices are taken (in which case all_taken = true)
	std::vector<size_t> inds;

	if (len_old != 0)							// already have some points
	{
		// add whole x0 to X_0, then select subset
		std::vector<std::vector<double>> Xfull = X_0;
		rescaleBACK_vecs(Xfull);
		HMMPI::VecAppend(Xfull, x0);			// Xfull is not scaled (but X_0 is scaled)

		const HMMPI::Mat Dfull = DistMatr(Xfull, 0, Xfull.size(), 0, Xfull.size());
		inds = IndSignificant(Dfull, count, len_old);
		std::sort(inds.begin(), inds.end());

		all_taken = false;
	}
	else										// no points yet
	{
		inds = std::vector<size_t>(x0.size());	// all indices are taken
		iota(inds.begin(), inds.end(), 0);

		all_taken = true;
	}

	return inds;
}
//---------------------------------------------------------------------------
void KrigStart::AddPoints(std::vector<std::vector<double>> x0, std::vector<std::vector<double>> x1)		// adds 'x0' to X_0, adds 'x1' to X_1, updates 'D', 'DG', 'DGG';
{
	const size_t len_old = X_0.size();
	rescale_vecs(x0);
	rescale_vecs(x1);

	// I. add 'x0' to X_0, 'x1' to X_1
	push_back_data(x0, x1);

	// II. update the distance matrix 'D'
	const HMMPI::Mat offD = DistMatr(X_0, 0, len_old, len_old, X_0.size());				// off-diagonal block
	const HMMPI::Mat Dadd = DistMatr(X_0, len_old, X_0.size(), len_old, X_0.size());	// added diagonal block
	D = (D && offD)||(offD.Tr() && Dadd);

	// III. make the distance matrices for gradients
	std::vector<std::vector<double>> X = X_0;
	HMMPI::VecAppend(X, X_1);										// X = X_0 && X_1

	DG = DistMatr(X, 0, X_0.size(), X_0.size(), X.size());
	DGG = DistMatr(X_1, 0, X_1.size(), 0, X_1.size());

	int RNK;
	MPI_Comm_rank(MPI_COMM_WORLD, &RNK);
	if (RNK == 0 && dump_flag != -1)			// debug output to files
	{
		HMMPI::Mat Dfull = (D && DG)||(DG.Tr() && DGG);		// for reporting

		char fname[100];
		sprintf(fname, dump_D, dump_flag, 0);	// _pr0
		FILE *f = fopen(fname, "w");
		Dfull.SaveASCII(f, "%20.16g");						// full distance matrix
		fclose(f);

		sprintf(fname, dump_X, dump_flag, 0);	// _pr0
		f = fopen(fname, "w");

		for (const auto &w : X)
			fputs(HMMPI::ToString(w, "%20.16g").c_str(), f);		// points (full X)
		fclose(f);
	}
}
//---------------------------------------------------------------------------
void KrigStart::RecalcPoints()								// (after adding X0, X1) makes appropriate matrix calculations
{
	// update the "correlation" part of kriging matrix (func. vals only), set 'func' and 'R'
	func = kc.CalculateR(kc.get_init());
	R = kc.get_init()[1];
	HMMPI::Mat CM = kc.Get_R();

	// Full covariance:
	// |CM  CG|
	// |CG' GG|

	// calculate the correlation part involving the gradients
	const int Ngrad_comps = grad_inds.size();				// number of gradient components which participate in training
	HMMPI::Mat CG(X_0.size(), X_1.size()*Ngrad_comps, 0.0);					// off-diagonal correlation block

	const double R2 = R*R;
	auto f1 = [this, R2](double x) -> double {return func->lim_df(x/R)/R2;};
	HMMPI::Mat Mults = DG;									// aux. matrix of multipliers to be used for "CG"
	Mults.Func(f1);
	for (size_t i = 0; i < X_0.size(); i++)
	{
		const std::vector<double> x0i = HMMPI::Reorder(X_0[i], grad_inds);
		for (size_t j = 0; j < X_1.size(); j++)
		{
			const std::vector<double> x1j = HMMPI::Reorder(X_1[j], grad_inds);
			for (int k = 0; k < Ngrad_comps; k++)
				CG(i, j*Ngrad_comps + k) = (x1j[k] - x0i[k])*Mults(i, j);
		}
	}

	HMMPI::Mat GG(X_1.size()*Ngrad_comps, X_1.size()*Ngrad_comps, 0.0);		// diagonal gradients block
	const double R4 = R2*R2;
	auto f2 = [this, R4](double x) -> double {return func->lim_d2f(x/R)/R4;};
	HMMPI::Mat Mults1 = DGG;
	HMMPI::Mat Mults2 = DGG;
	Mults1.Func(f1);
	Mults2.Func(f2);
	for (size_t i = 0; i < X_1.size(); i++)
	{
		const std::vector<double> x1i = HMMPI::Reorder(X_1[i], grad_inds);
		for (size_t j = 0; j < X_1.size(); j++)
		{
			const std::vector<double> x1j = HMMPI::Reorder(X_1[j], grad_inds);
			for (int k = 0; k < Ngrad_comps; k++)
				for (int l = 0; l < Ngrad_comps; l++)
					GG(i*Ngrad_comps + k, j*Ngrad_comps + l) = -Mults2(i, j)*(x1i[k] - x1j[k])*(x1i[l] - x1j[l]) - (k == l ? Mults1(i, j) : 0.0);
		}
	}

	CM = (CM && CG)||(CG.Tr() && GG);

	// update the "trend" part of kriging matrix (func. vals only)
	size_t len_trend = multi_ind.size();
	size_t len_old = N.ICount();			// N has not been updated yet
	int len_new = X_0.size() - len_old;
	if (len_new < 0)
		throw HMMPI::Exception("len_new < 0 в PM_ProxyKrig::RecalcPoints", "len_new < 0 in PM_ProxyKrig::RecalcPoints");

	HMMPI::Mat Nnew(len_new, len_trend, 0.0);
	for (int i = 0; i < len_new; i++)
		for (size_t j = 0; j < len_trend; j++)
			Nnew(i, j) = HMMPI::Vec_pow_multiind(X_0[len_old + i], multi_ind[j]);
	N = N || Nnew;

	// update the "trend" part for gradients				// pscale.size() = FULLDIM
	std::vector<int> re_grad_inds(pscale.size(), -1);		// index mapping inverse to "grad_inds"; grad_inds[re_grad_inds[i_full]] = i_full for 'i_full' in grad_inds; -1 otherwise
	int c = 0;												// 0 <= re_grad_inds[i_full] < Ngrad_comps (or may = -1); "re" = reduced/reversed
	for (size_t i = 0; i < re_grad_inds.size(); i++)
		if (c < Ngrad_comps && grad_inds[c] == i)
		{
			re_grad_inds[i] = c;
			c++;
			if (c >= Ngrad_comps)
				break;
		}

	HMMPI::Mat Ngrad(X_1.size()*Ngrad_comps, len_trend, 0.0);
	for (size_t i = 0; i < X_1.size(); i++)			// i - design point
		for (size_t j = 0; j < len_trend; j++)		// j - a trend component
			for (size_t k = 0; k < multi_ind[j].size(); k++)	// go through multiindex 'j'
				if (re_grad_inds[multi_ind[j][k]] != -1)		// take this gradient component
					Ngrad(i*Ngrad_comps + re_grad_inds[multi_ind[j][k]], j) += HMMPI::Vec_pow_multiind(X_1[i], multi_ind[j], k);		// add d/dk; multi_ind[j][k] -- index of the variable (parameter)

	Nfull = N || Ngrad;

	// make the full kriging matrix
	C = (CM && Nfull) || (Nfull.Tr() && HMMPI::Mat(Nfull.JCount(), Nfull.JCount(), 0));

	if (dump_flag != -1)	// debug output to files
	{
		char fname[100];
		int RNK;
		MPI_Comm_rank(MPI_COMM_WORLD, &RNK);
		sprintf(fname, dump_C, dump_flag, RNK, index);
		FILE *f = fopen(fname, "w");
		C.SaveASCII(f, "%20.16g");
		fclose(f);
	}
}
//---------------------------------------------------------------------------
void KrigStart::ObjFuncCommon(std::vector<double> params)
{
	rescale_vec(params);

	// make the current point vector [c0, x0]^t
	// 1. Correlation part for func. values
	HMMPI::Mat c0 = RHS_dist_matr(X_0, params);
	auto f0 = [this](double x) -> double {return func->f(x/R, smooth_at_nugget);};
	c0.Func(f0);

	// 1a. Correlation part for gradients
	const int Ngrad_comps = grad_inds.size();				// number of gradient components which participate in training
	HMMPI::Mat c1(X_1.size()*Ngrad_comps, 1, 0.0);
	if (X_1.size() > 0)
	{
		HMMPI::Mat mults = RHS_dist_matr(X_1, params);
		const double R2 = R*R;
		auto f1 = [this, R2](double x) -> double {return func->lim_df(x/R)/R2;};
		mults.Func(f1);

		const std::vector<double> par_reord = HMMPI::Reorder(params, grad_inds);
		for (size_t j = 0; j < X_1.size(); j++)
		{
			const std::vector<double> x1j = HMMPI::Reorder(X_1[j], grad_inds);
			for (int k = 0; k < Ngrad_comps; k++)
				c1(j*Ngrad_comps + k, 0) = (x1j[k] - par_reord[k])*mults(j, 0);
		}
	}

	// 2. Trend part
	size_t len_trend = multi_ind.size();
	HMMPI::Mat x0(len_trend, 1, 0.0);
	if (trend_order == 1)	// some code specialization
	{
		assert(len_trend-1 == params.size());
		x0(0, 0) = 1;
		for (size_t j = 1; j < len_trend; j++)
			x0(j, 0) = params[j-1];
	}
	else					// general case
	{
		for (size_t j = 0; j < len_trend; j++)
			x0(j, 0) = HMMPI::Vec_pow_multiind(params, multi_ind[j]);
	}

	C0 = std::move(c0)||c1||x0;

#ifdef TESTNEWPROXY
	int size00, RNK;
	MPI_Comm_size(MPI_COMM_WORLD, &size00);
	MPI_Comm_rank(MPI_COMM_WORLD, &RNK);
	std::ofstream file0(HMMPI::stringFormatArr(test_fn, std::vector<int>{RNK}), std::ios_base::app);
	std::cout << "rank " << RNK << ", size " << size00 << "\tKrigStart::ObjFuncCommon, C0 size " << C0.ICount() << " x " << C0.JCount() << "\n";
	file0 << "rank " << RNK << ", size " << size00 << "\tKrigStart::ObjFuncCommon, C0 size " << C0.ICount() << " x " << C0.JCount() << "\n";
	file0.close();
#endif
}
//---------------------------------------------------------------------------
void KrigStart::ObjFuncGradCommon(std::vector<double> params)
{
	rescale_vec(params);

	const size_t len_corr_0 = X_0.size();
	const size_t len_corr_1 = X_1.size();
	const int Ngrad_comps = grad_inds.size();							// number of gradient components which participate in training
	const size_t len_trend = multi_ind.size();
	const size_t dim = params.size();
	gM = HMMPI::Mat(dim, len_corr_0 + len_corr_1*Ngrad_comps + len_trend, 0.0);		// current "gradient matrix"

	// 1. Fill the correlation part, func. values only
	HMMPI::Mat aux = RHS_dist_matr(X_0, params);
	const double R2 = R*R;
	auto f1 = [this, R2](double x) -> double {return func->lim_df(x/R)/R2;};
	aux.Func(f1);

	for (size_t i = 0; i < dim; i++)
		for (size_t j = 0; j < len_corr_0; j++)
			gM(i, j) = (params[i] - X_0[j][i])*aux(j, 0);

	// 1a. Correlation part for gradients
	if (len_corr_1 > 0)
	{
		HMMPI::Mat aux1 = RHS_dist_matr(X_1, params);
		HMMPI::Mat aux1_0 = aux1;
		const double R4 = R2*R2;
		auto f2 = [this, R4](double x) -> double {return func->lim_d2f(x/R)/R4;};
		aux1.Func(f2);
		aux1_0.Func(f1);

		const std::vector<double> par_reord = HMMPI::Reorder(params, grad_inds);
		for (size_t l = 0; l < dim; l++)
			for (size_t j = 0; j < len_corr_1; j++)
			{
				const std::vector<double> x1j = HMMPI::Reorder(X_1[j], grad_inds);
				for (int i = 0; i < Ngrad_comps; i++)
					gM(l, len_corr_0 + j*Ngrad_comps + i) = -(par_reord[i] - x1j[i])*(params[l] - X_1[j][l])*aux1(j, 0) - aux1_0(j, 0)*(grad_inds[i] == l ? 1 : 0);
			}
	}

	// 2. Fill the trend part
	if (trend_order == 1)			// some code specialization
	{
		assert(len_trend-1 == params.size());
		for (size_t j = 1; j < len_trend; j++)		// j - a trend component
			gM(j-1, len_corr_0 + len_corr_1*Ngrad_comps + j) += 1;
	}
	else							// general case
	{
		for (size_t j = 0; j < len_trend; j++)		// j - a trend component
			for (size_t k = 0; k < multi_ind[j].size(); k++)	// go through multiindex 'j'
				gM(multi_ind[j][k], len_corr_0 + len_corr_1*Ngrad_comps + j) += HMMPI::Vec_pow_multiind(params, multi_ind[j], k);		// add d/dk; multi_ind[j][k] -- index of the variable (parameter)
	}

	// 3. Scaling
	gM = pscale % std::move(gM);

#ifdef TESTING
	std::cout << "gradient, gM\n" << gM.ToString("%8.6g");
#endif

#ifdef TESTNEWPROXY
	int size00, RNK;
	MPI_Comm_size(MPI_COMM_WORLD, &size00);
	MPI_Comm_rank(MPI_COMM_WORLD, &RNK);
	std::ofstream file0(HMMPI::stringFormatArr(test_fn, std::vector<int>{RNK}), std::ios_base::app);
	std::cout << "rank " << RNK << ", size " << size00 << "\tKrigStart::ObjFuncGradCommon, gM size " << gM.ICount() << " x " << gM.JCount() << "\n";
	file0 << "rank " << RNK << ", size " << size00 << "\tKrigStart::ObjFuncGradCommon, gM size " << gM.ICount() << " x " << gM.JCount() << "\n";
	file0.close();
#endif
}
//---------------------------------------------------------------------------
void KrigStart::ObjFuncHess_lCommon(std::vector<double> params, int l)
{
	rescale_vec(params);

	const size_t len_corr_0 = X_0.size();
	const size_t len_corr_1 = X_1.size();
	const size_t len_trend = multi_ind.size();
	const int Ngrad_comps = grad_inds.size();			// number of gradient components which participate in training
	const size_t dim = params.size();
	lM = HMMPI::Mat(dim, len_corr_0 + len_corr_1*Ngrad_comps + len_trend, 0.0);		// current matrix for column "l"

	// some stuff for the correlation part
	HMMPI::Mat aux1 = RHS_dist_matr(X_0, params), aux2;
	aux2 = aux1;
	const double R2 = R*R;
	const double R4 = R2*R2;
	auto f1 = [this, R2](double x) -> double {return func->lim_df(x/R)/R2;};
	auto f2 = [this, R4](double x) -> double {return func->lim_d2f(x/R)/R4;};
	aux1.Func(f1);
	aux2.Func(f2);

	// 1. Fill the correlation part (func. vals only)
	for (size_t j = 0; j < len_corr_0; j++)
	{
		for (size_t i = 0; i < dim; i++)
			lM(i, j) = (params[i] - X_0[j][i])*aux2(j, 0)*(params[l] - X_0[j][l]);	// (X*1_M^T - bold(X))*D_(l)

		lM(l, j) += aux1(j, 0);	// a_(l)*1_M^T*D
	}

	// 1a. Correlation part for gradient points
	if (len_corr_1 > 0)
	{
		HMMPI::Mat aux3 = RHS_dist_matr(X_1, params);		// [...]_*
		HMMPI::Mat aux4 = aux3;
		const double R6 = R2*R4;
		auto f3 = [this, R6](double x) -> double {return func->lim_d3f(x/R)/R6;};
		aux3.Func(f3);
		aux4.Func(f2);

		const std::vector<double> par_reord = HMMPI::Reorder(params, grad_inds);
		for (size_t k = 0; k < dim; k++)						// d/dk <-> row of lM
			for (size_t j = 0; j < len_corr_1; j++)				// design point
			{
				const std::vector<double> x1j = HMMPI::Reorder(X_1[j], grad_inds);
				for (int i = 0; i < Ngrad_comps; i++)			// i - gradient component participating in proxy training
					lM(k, len_corr_0 + j*Ngrad_comps + i) = (params[k] - X_1[j][k])*(par_reord[i] - x1j[i])*(params[l] - X_1[j][l])*aux3(j, 0) - aux4(j, 0)*(
						   (grad_inds[i] == k ? 1 : 0)*(params[l] - X_1[j][l]) +
								 ((int)k == l ? 1 : 0)*(par_reord[i] - x1j[i]) +
					  ((int)grad_inds[i] == l ? 1 : 0)*(params[k] - X_1[j][k]));
			}
	}

	// 2. Fill the trend part
	if (trend_order == 1)			// some code specialization
	{
		// pass
	}
	else							// general case
	{
		for (size_t j = 0; j < len_trend; j++)
		{
			for (size_t i = 0; i < multi_ind[j].size(); i++)		// go through multiindex 'j'
				for (size_t k = 0; k < multi_ind[j].size(); k++)	// go through multiindex 'j'
					if (i != k && multi_ind[j][i] == l)
						lM(multi_ind[j][k], len_corr_0 + len_corr_1*Ngrad_comps + j) += HMMPI::Vec_pow_multiind(params, multi_ind[j], i, k);		// add d2/(di*dk)
		}
	}

	// 3. Scaling
	assert(0 <= l && (size_t)l < pscale.size());
	lM = pscale[l] * (pscale % std::move(lM));

#ifdef TESTING
		std::cout << "Hessian, lM (l = " << l << ")\n" << lM.ToString("%8.6g");
#endif

#ifdef TESTNEWPROXY
	int size00, RNK;
	MPI_Comm_size(MPI_COMM_WORLD, &size00);
	MPI_Comm_rank(MPI_COMM_WORLD, &RNK);
	std::ofstream file0(HMMPI::stringFormatArr(test_fn, std::vector<int>{RNK}), std::ios_base::app);
	std::cout << "rank " << RNK << ", size " << size00 << "\tKrigStart::ObjFuncHess_lCommon, lM size " << lM.ICount() << " x " << lM.JCount() << "\n";
	file0 << "rank " << RNK << ", size " << size00 << "\tKrigStart::ObjFuncHess_lCommon, lM size " << lM.ICount() << " x " << lM.JCount() << "\n";
	file0.close();
#endif
}
//---------------------------------------------------------------------------
HMMPI::Mat KrigStart::DistMatr(const std::vector<std::vector<double>> &X0, int i1, int i2, int j1, int j2)			// creates (i2-i1)x(j2-j1) distance matrix for distances X0[i1,i2) -- X0[j1,j2)
{
	bool symmfill = (i1 == j1 && i2 == j2);				// if "i" and "j" ranges are the same, fill the matrix symmetrically
	HMMPI::Mat res(i2-i1, j2-j1, 0);

	for (int i = i1; i < i2; i++)
	{
		int J0 = (symmfill)?(i+1):(j1);
		for (int j = J0; j < j2; j++)
		{
			HMMPI::Mat diff = HMMPI::Mat(X0[i]) - HMMPI::Mat(X0[j]);
			res(i-i1, j-j1) = diff.Norm2();				// was: sqrt(InnerProd(diff, diff));
			if (symmfill)
				res(j-j1, i-i1) = res(i-i1, j-j1);
		}
	}

#ifdef TESTNEWPROXY
	int rank00;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank00);
	std::ofstream file0(HMMPI::stringFormatArr(test_fn, std::vector<int>{rank00}), std::ios_base::app);
	std::cout << "rank " << rank00 << "\tKrigStart::DistMatr, return matrix " << res.ICount() << " x " << res.JCount() << "\n";
	file0 << "rank " << rank00 << "\tKrigStart::DistMatr, return matrix " << res.ICount() << " x " << res.JCount() << "\n";
	file0.close();
#endif

	return res;
}
//---------------------------------------------------------------------------
HMMPI::Mat KrigStart::RHS_dist_matr(std::vector<std::vector<double>> &Xarr, const std::vector<double> &params)		// distance "vector" for RHS of kriging system; its elements are distances from Xarr[i] to 'params'; Xarr can be X_0, X_1
{
	int len = Xarr.size();
	Xarr.push_back(params);		// add extra point
	HMMPI::Mat res = DistMatr(Xarr, 0, len, len, len+1);
	Xarr.pop_back();			// remove extra point

#ifdef TESTNEWPROXY
	int size00, RNK;
	MPI_Comm_size(MPI_COMM_WORLD, &size00);
	MPI_Comm_rank(MPI_COMM_WORLD, &RNK);
	std::ofstream file0(HMMPI::stringFormatArr(test_fn, std::vector<int>{RNK}), std::ios_base::app);
	std::cout << "rank " << RNK << ", size " << size00 << "\tKrigStart::RHS_dist_matr, RHS size = " << res.ICount() << "\n";
	file0 << "rank " << RNK << ", size " << size00 << "\tKrigStart::RHS_dist_matr, RHS size = " << res.ICount() << "\n";
	file0.close();
#endif

	return res;
}
//---------------------------------------------------------------------------
std::vector<size_t> KrigStart::IndSignificant(const HMMPI::Mat &DM, size_t count, size_t start)
{
	if (DM.ICount() != DM.JCount())
		throw HMMPI::Exception("DM должна быть симметричной в KrigStart::IndSignificant", "DM should be symmetric in KrigStart::IndSignificant");
	if (start > DM.ICount())
		throw HMMPI::Exception("start вне диапазона в KrigStart::IndSignificant", "start out of range in KrigStart::IndSignificant");

	size_t N = DM.ICount() - start;		// number of points left when the first "start" points are removed
	std::vector<size_t> res;
	if (N == 0)
		return res;
	if (N == 1)
	{
		if (count > 0)
			res.push_back(0);
		return res;
	}

	if (count > N)
		count = N;
	res.reserve(count);

	std::vector<bool> taken(N);			// indicates which (local) indices have been taken
	for (bool &&t : taken)				// proxy iterator for vector<bool>, see http://stackoverflow.com/questions/15927033/what-is-the-correct-way-of-using-c11s-range-based-for
		t = false;

	size_t c0 = 0;
	if (start == 0)
	{
		size_t i1, i2;
		DM.Max(i1, i2);			// DM uses global indices; max DM(i1, i2) <--> min 1/DM(i1, i2)^3
		if (i1 == i2)
			i2 += 1;	// this is safe: N >= 2, DM is all zeros (although this is a strange input), i1 == i2 == 0

		res.push_back(i1);		// i1, i2 are 'local' (= 'global' here)
		res.push_back(i2);
		c0 = 2;

		taken[i1] = true;
		taken[i2] = true;
	}

	for (size_t c = c0; c < count; c++)
	{
		// find the min
		double min = std::numeric_limits<double>::max();
		size_t ind = -1;					// local index
		for (size_t j = 0; j < N; j++)		// j -- local
			if (!taken[j])
			{
				double sum = 0;
				for (size_t k = 0; k < start; k++)					// points [0...start)
					sum += pow(DM(j + start, k), -3);				// k -- global
				for (size_t k = 0; k < res.size(); k++)				// other added points
					sum += pow(DM(j + start, res[k] + start), -3);	// res[k] -- local

				if (sum < min)
				{
					min = sum;
					ind = j;
				}
			}

		if (ind == (size_t)-1)
			throw HMMPI::Exception("ind == -1 in KrigStart::IndSignificant");

		res.push_back(ind);		// save 'local' index
		taken[ind] = true;
	}

#ifdef TESTNEWPROXY
	int rank00;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank00);
	std::ofstream file0(HMMPI::stringFormatArr(test_fn, std::vector<int>{rank00}), std::ios_base::app);
	std::cout << "rank " << rank00 << "\tKrigStart::IndSignificant, return vector of size " << res.size() << "\n";
	file0 << "rank " << rank00 << "\tKrigStart::IndSignificant, return vector of size " << res.size() << "\n";
	file0.close();
#endif

	return res;
}
//---------------------------------------------------------------------------
std::vector<std::vector<int>> KrigStart::make_multi_ind(int dim, int pow)
{
	std::vector<std::vector<int>> res;

	if (pow >= 0)
		res.push_back(std::vector<int>());		// pow = 0 -- empty vector

	if (pow >= 1)
	{
		std::vector<int> v(1);
		for (int i = 0; i < dim; i++)
		{
			v[0] = i;
			res.push_back(v);
		}
	}

	if (pow >= 2)
	{
		std::vector<int> w(2);
		for (int i = 0; i < dim; i++)
		{
			w[0] = i;
			for (int j = i; j < dim; j++)
			{
				w[1] = j;
				res.push_back(w);
			}
		}
	}

	if (pow == 3)
	{
		std::vector<int> u(3);
		for (int i = 0; i < dim; i++)
		{
			u[0] = i;
			for (int j = i; j < dim; j++)
			{
				u[1] = j;
				for (int k = j; k < dim; k++)
				{
					u[2] = k;
					res.push_back(u);
				}
			}
		}
	}

	if (pow > 3)
		throw HMMPI::Exception("pow > 3 в KrigStart::make_multi_ind", "pow > 3 in KrigStart::make_multi_ind");

#ifdef TESTNEWPROXY
	int rank00, size00;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank00);
	MPI_Comm_size(MPI_COMM_WORLD, &size00);
	std::ofstream file0(HMMPI::stringFormatArr(test_fn, std::vector<int>{rank00}), std::ios_base::app);
	std::cout << "rank " << rank00 << ", size " << size00 << "\tKrigStart::make_multi_ind, res size " << res.size() << "\n";
	file0 << "rank " << rank00 << ", size " << size00 << "\tKrigStart::make_multi_ind, res size " << res.size() << "\n";
	file0.close();
#endif

	return res;
}
//---------------------------------------------------------------------------
std::string KrigStart::init_msg() const
{
	return kc.init_msg;
}
//---------------------------------------------------------------------------
// KrigEnd
//---------------------------------------------------------------------------
const KrigEnd &KrigEnd::operator=(const KrigEnd &p)		// start, index, start_index are not copied
{
	start = NULL;
	y = p.y;
	grad_y = p.grad_y;
	CinvZ = p.CinvZ;
	ks = p.ks;

	dump_flag = p.dump_flag;
	index = -1;
	start_index = -1;

	return *this;
}
//---------------------------------------------------------------------------
double KrigEnd::ObjFuncPrivate()
{
	return InnerProd(start->C0, CinvZ);
}
//---------------------------------------------------------------------------
std::vector<double> KrigEnd::ObjFuncGradPrivate()
{
	return start->gM * (CinvZ.ToVector());
}
//---------------------------------------------------------------------------
std::vector<double> KrigEnd::ObjFuncHess_lPrivate()
{
	return start->lM * (CinvZ.ToVector());
}
//---------------------------------------------------------------------------
void KrigEnd::OptimizeKrig()									// optimizes kriging parameters (via "ks") for ends[i] in DataProxy2 (or stand alone simple proxy), and takes these parameters; should be called after adding vals, RecalcPoints()
{																// TODO looks like currently LIMITSKRIG does not make parameters scaling, e.g. 'norm', 'func' have no effect
	// perform optimization
	Optimizer *Opt = Optimizer::Make("LM");
	OptCtxLM optctx(maxit, epsG, epsF, epsX);
	KrigStart *mut_start = const_cast<KrigStart*>(start);		// the associated KrigStart will be altered, so const-ness is temporarily removed

	int RNK;
	MPI_Comm_rank(MPI_COMM_WORLD, &RNK);

	if (ks.GetConstr() == nullptr)
		throw HMMPI::Exception("Cannot run KrigEnd::OptimizeKrig if LIMITSKRIG is not defined");

	if (start->Nfull.ICount() != start->N.ICount())
		throw HMMPI::Exception("Kriging optimisation is currently _not working_ with gradient design points (KrigEnd::OptimizeKrig)");

	std::vector<double> x0 = start->kc.get_init();
	std::vector<double> x1 = Opt->RunOpt(&ks, x0, &optctx);
	mut_start->kc.set_init(x1);		// save for further use

#ifdef KRIG_OPT_STDOUT
	{
		char fname[HMMPI::BUFFSIZE];
		sprintf(fname, dump_opt_krig, RNK);
		FILE *f = fopen(fname, "a");

		fprintf(f, "%d\t%d iter-s\t%g\t%g\t%g\t%g\t%g\t%g\n", index, dynamic_cast<OptLM*>(Opt)->iter_count, x0[0], x0[1], x0[2], x1[0], x1[1], x1[2]);
		printf("[%d] pr-%d kriging opt, %d iter-s (%g, %g, %g) -> (%g, %g, %g)\n", RNK, index, dynamic_cast<OptLM*>(Opt)->iter_count, x0[0], x0[1], x0[2], x1[0], x1[1], x1[2]);
		fclose(f);
	}
#endif

	delete Opt;

	// update the "correlation" part of kriging matrix, set 'func' and 'R'
	mut_start->func = start->kc.CalculateR(x1);
	mut_start->R = x1[1];
	HMMPI::Mat CM = start->kc.Get_R();

	// "trend" part of kriging matrix - not changed here

	// make the full matrix
	mut_start->C = (CM && start->N) || (start->N.Tr() && HMMPI::Mat(start->N.JCount(), start->N.JCount(), 0.0));

	if (dump_flag != -1)	// debug output to files
	{
		char fname[100];
		sprintf(fname, dump_C, dump_flag, RNK, start->index);	// for PROXY, DATAPROXY2: start->index = end->index
		FILE *f = fopen(fname, "w");
		start->C.SaveASCII(f, "%20.16g");
		fclose(f);
	}
}
//---------------------------------------------------------------------------
std::string KrigEnd::RecalcVals()								// (after adding values) makes CinvZ calculation; returns message on mat_eff_rank
{
	int RNK;
	std::string mat_eff_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &RNK);

	// find CinvZ
	std::vector<double> Ity = get_y();
	size_t len_trend = start->get_multi_ind().size();
	Ity.reserve(Ity.size() + len_trend);
	for (size_t i = 0; i < len_trend; i++)
		Ity.push_back(0);

	CinvZ = start->get_sol()->Solve(start->get_C(), Ity);
	mat_eff_rank = HMMPI::ToString(std::vector<int>{start->get_sol()->rank}, "%d");

	if (dump_flag != -1)	// debug output to files
	{
		char fname[100];
		sprintf(fname, dump_CinvZ, dump_flag, RNK, index);
		FILE *f = fopen(fname, "w");
		CinvZ.SaveASCII(f, "%20.16g");
		fclose(f);

		sprintf(fname, dump_Ity, dump_flag, RNK, index);
		f = fopen(fname, "w");
		HMMPI::Mat(Ity).SaveASCII(f, "%20.16g");
		fclose(f);
	}

	return mat_eff_rank;
}
//---------------------------------------------------------------------------
void KrigEnd::set_refs()								// sets some refs for "ks"
{
	ks.take_refs(&start->N, &y, &start->kc);
}
//---------------------------------------------------------------------------
std::vector<double> KrigEnd::get_y() const				// returns the observed values at design points (values + gradients); the Gaussian Process will be conditioned on them
{
	int len = y.size();
	int gr_len = grad_y.size();
	const int Ngrad_comps = start->grad_inds.size();
	std::vector<double> res(len + gr_len*Ngrad_comps);

	std::copy(y.begin(), y.end(), res.begin());			// res = y
	for (int i = 0; i < gr_len; i++)
	{
		const std::vector<double> grad = HMMPI::Reorder(grad_y[i], start->grad_inds);
		std::copy(grad.begin(), grad.end(), res.begin() + len + i*Ngrad_comps);		// res += grad_y[i]
	}

	return res;
}
//---------------------------------------------------------------------------
void KrigEnd::push_back_vals(const std::vector<double> &y0)					// appends whole y0 to y
{
	HMMPI::VecAppend(y, y0);
}
//---------------------------------------------------------------------------
void KrigEnd::push_back_grads(const std::vector<std::vector<double>> &gr0)	// appends whole gr0 to grad_y
{
	HMMPI::VecAppend(grad_y, gr0);
}
//---------------------------------------------------------------------------
// CONTAINERS
//---------------------------------------------------------------------------
// ValCont
//---------------------------------------------------------------------------
void ValCont::write_FIT_SMRY() const								// debug output to file; len, smry_len should be correctly set
{
	int RNK;
	MPI_Comm_rank(MPI_COMM_WORLD, &RNK);
	if (RNK == 0)
	{
		FILE *f_smry = fopen("ProxyDesignFIT_SMRY.txt", "w");		// output FIT, SMRY to file
		if (f_smry != NULL)
		{
			if (FIT != 0)
			{
				HMMPI::SaveASCII(f_smry, FIT, len);
				fprintf(f_smry, "\n");
			}
			if (SMRY != 0)
				HMMPI::SaveASCII(f_smry, SMRY, len, smry_len);
			fclose(f_smry);
		}
	}
}
//---------------------------------------------------------------------------
void ValCont::RunTrainPopulation(PhysModel *pm, const std::vector<std::vector<double>> &pop)	// fills len, smry_len, FIT, SMRY based on design points pop[len][full_dim], using ObjFuncMPI_ACT; "pop" is only referenced on comm-RANKS-0
{																								// 'pm' should have comm == "MPI_COMM_SELF"
	// "pop" is only referenced on comm-RANKS-0 => "pop" is defined on ofmpi_comm-RANKS-0

	MPI_Comm ofmpi_comm = comm;
	if (ofmpi_comm == MPI_COMM_SELF)
		ofmpi_comm = MPI_COMM_WORLD;	// may happen for simple proxy

#ifdef PROXY_DEBUG
	int RNK;
	MPI_Comm_rank(MPI_COMM_WORLD, &RNK);
	std::string msg = HMMPI::MPI_Ranks(std::vector<MPI_Comm>{comm, pm->GetComm(), ofmpi_comm});
	if (RNK == 0)
		std::cout << "PM_Proxy::RunTrainPopulation communicators:\nWORLD\tVALCONT\tPM\tOFMPI\n" << msg << "\n";		// VALCONT/WORLD and OFMPI should give the same rank numbers -- check!
#endif

	int rank_ofmpi = -1;
	if (ofmpi_comm != MPI_COMM_NULL)
		MPI_Comm_rank(ofmpi_comm, &rank_ofmpi);

	len = pop.size();						// ofmpi_comm-RANKS-0 (and comm-RANKS-0)
	int actdim = pm->ParamsDim_ACT();		// only active parameters; defined on all ranks
	smry_len = pm->ModelledDataSize();		// defined on all ranks

	double **POP = 0;
	assert(FIT == 0);
	assert(SMRY == 0);

	if (rank_ofmpi == 0)
	{
		POP = new double*[len];
		FIT = new double[len];
		SMRY = new double*[len];
		for (int i = 0; i < len; i++)
		{
			POP[i] = new double[actdim];
			memcpy(POP[i], pm->act_par(pop[i]).data(), sizeof(double)*actdim);
			SMRY[i] = new double[smry_len];			// even for smry_len == 0
		}
	}

	PhysModMPI PM_MPI(ofmpi_comm, pm);					// 'pm' is supposed to have comm == "MPI_COMM_SELF" (compatible one)
	PM_MPI.ObjFuncMPI_ACT(len, POP, FIT, false, SMRY);	// "len", "POP", "FIT", "SMRY" are only used by ofmpi_comm-RANKS-0

	// Bcast for the case comm == MPI_COMM_SELF (ofmpi_comm == MPI_COMM_WORLD), so that FIT/SMRY are defined on comm-RANKS-0
	if (comm == MPI_COMM_SELF)
	{
		if (rank_ofmpi != 0)		// MPI_COMM_WORLD-RANK != 0
		{
			// allocate memory (which will be cleared in ValContFactory)
			FIT = new double[len];
			SMRY = new double*[len];
			for (int i = 0; i < len; i++)
				SMRY[i] = new double[smry_len];		// even for smry_len == 0
		}
		MPI_Bcast(FIT, len, MPI_DOUBLE, 0, ofmpi_comm);			// Bcast to MPI_COMM_WORLD
		HMMPI::Bcast_vector(SMRY, len, smry_len, 0, ofmpi_comm);
	}

	if (POP != 0)
	{
		for (int i = 0; i < len; i++)
			delete [] POP[i];
		delete [] POP;
	}
}
//---------------------------------------------------------------------------
ValCont::~ValCont()
{
	if (FIT != 0 || SMRY != 0)
	{
		int RNK;
		MPI_Comm_rank(MPI_COMM_WORLD, &RNK);
		std::cout << "[" << RNK << "] ERROR in ValCont::~ValCont: len = " << len << ", smry_len = " << smry_len << ", FIT = " << FIT << ", SMRY = " << SMRY << "\n";
	}

	assert(FIT == 0 && SMRY == 0);		// to avoid memory leaks
}
//---------------------------------------------------------------------------
// ValContDouble
//---------------------------------------------------------------------------
void ValContDouble::FitSmryFromFile(std::string fname, int l)
{
	assert(FIT == 0);
	assert(SMRY == 0);
	len = l;

	int rnk = -1;
	if (comm != MPI_COMM_NULL)
		MPI_Comm_rank(comm, &rnk);

	if (rnk == 0)
	{
		FIT = new double[len];
		FILE *f0 = fopen(fname.c_str(), "r");
		if (f0 != 0)
		{
			for (int c = 0; c < len; c++)
			{
				double d;
				int n = fscanf(f0, "%lg", &d);
				if (n == 1)
					FIT[c] = d;
				else if (feof(f0))
				{
					fclose(f0);
					throw HMMPI::Exception("End of file reached before FIT was filled in ValContDouble::FitSmryFromFile");
				}
			}
			fclose(f0);
		}
		else
			throw HMMPI::Exception("Cannot open file in ValContDouble::FitSmryFromFile");
	}
}
//---------------------------------------------------------------------------
ValContDouble::ValContDouble(MPI_Comm c, PhysModel *pm, const std::vector<std::vector<double>> &pop, int train_from_dump) : ValCont(c)
{
	if (train_from_dump == -1)
		RunTrainPopulation(pm, pop);
	else
	{
		char fname[100];
		sprintf(fname, dump_Ity, train_from_dump, 0, 0);		// NOTE _rnk0_pr0 is used

		smry_len = pm->ModelledDataSize();
		int popsize = pop.size();
		MPI_Bcast(&popsize, 1, MPI_INT, 0, comm);

		FitSmryFromFile(fname, popsize);
	}

	write_FIT_SMRY();		// debug output to file

	assert(FIT != 0);
	V = std::vector<double>(FIT, FIT + len);
	// no gradients taken!

	if (SMRY != 0)
	{
		for (int i = 0; i < len; i++)
			delete [] SMRY[i];
		delete [] SMRY;
	}
	delete [] FIT;

	FIT = 0;
	SMRY = 0;
}
//---------------------------------------------------------------------------
int ValContDouble::vals_count() const
{
	int res = V.size();
	MPI_Allreduce(MPI_IN_PLACE, &res, 1, MPI_INT, MPI_MAX, comm);
	return res;
}
//---------------------------------------------------------------------------
int ValContDouble::total_count() const
{
	int res = V.size() + Grad.size();
	MPI_Allreduce(MPI_IN_PLACE, &res, 1, MPI_INT, MPI_MAX, comm);
	return res;
}
//---------------------------------------------------------------------------
void ValContDouble::DistrValues(std::vector<KrigEnd> &dep, const std::vector<size_t> &inds) const
{
	if (dep.size() != 1)
		throw HMMPI::Exception("std::vector<PM_Proxy*> должен быть длины 1 в ValContDouble::DistrValues",
							   "std::vector<PM_Proxy*> should have length 1 in ValContDouble::DistrValues");

	dep[0].push_back_vals(HMMPI::Reorder(V, inds));
	dep[0].push_back_grads(Grad);

#ifdef TESTNEWPROXY
	int rank00, size00;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank00);
	MPI_Comm_size(MPI_COMM_WORLD, &size00);
	std::ofstream file0(HMMPI::stringFormatArr(test_fn, std::vector<int>{rank00}), std::ios_base::app);
	std::cout << "rank " << rank00 << ", size " << size00 << "\tValContDouble::DistrValues, added " << inds.size() << " values to " << &dep[0] << "\n";
	file0 << "rank " << rank00 << ", size " << size00 << "\tValContDouble::DistrValues, added " << inds.size() << " values to " << &dep[0] << "\n";
	file0.close();
#endif
}
//---------------------------------------------------------------------------
void ValContDouble::Add(const ValContDouble &b)			// appends data from "b" to 'this'; MPI layout should be the same
{
	HMMPI::VecAppend(V, b.V);
	HMMPI::VecAppend(Grad, b.Grad);
}
//---------------------------------------------------------------------------
// ValContVecDouble
//---------------------------------------------------------------------------
//double ValContVecDouble::obj_func_for_one_model(int m, const PM_DataProxy *dp, std::vector<double> &resid)
//{
//	// indexing is "Vecs[smry_len_local][len]" -> calculate residual for model 'm'
//	resid = std::vector<double>(Vecs.size());				// on different processes size may be different
//	for (size_t i = 0; i < Vecs.size(); i++)
//		resid[i] = Vecs[i][m] - dp->d0(i, 0);
//
//	double res = 0;											// for reduced result (full o.f.)
//	double prod = dp->BDC->InvTwoSideVecMult(resid);		// local result
//	MPI_Allreduce(&prod, &res, 1, MPI_DOUBLE, MPI_SUM, comm);	// res = full o.f.
//
//	return res;
//}
//---------------------------------------------------------------------------
void ValContVecDouble::FitSmryFromFile(std::string fname, int l)
{
	len = l;
	MPI_Bcast(&smry_len, 1, MPI_INT, 0, comm);	// sync in case

	assert(FIT == 0);
	assert(SMRY == 0);

	int rnk = -1;
	if (comm != MPI_COMM_NULL)
		MPI_Comm_rank(comm, &rnk);

	if (rnk == 0)
	{
		SMRY = new double*[len];
		for (int i = 0; i < len; i++)
		{
			if (smry_len != 0)
				SMRY[i] = new double[smry_len];
			else
				SMRY[i] = 0;
		}

		FILE *f0 = fopen(fname.c_str(), "r");
		if (f0 != 0)
		{
			for (int c = 0; c < len; c++)
				for (int i = 0; i < smry_len; i++)
				{
					double d;
					int n = fscanf(f0, "%lg", &d);
					if (n == 1)
						SMRY[c][i] = d;
					else if (feof(f0))
					{
						fclose(f0);
						throw HMMPI::Exception("End of file reached before SMRY was filled in ValContVecDouble::FitSmryFromFile");
					}
				}

			fclose(f0);
		}
		else
			throw HMMPI::Exception("Cannot open file in ValContVecDouble::FitSmryFromFile");
	}
}
//---------------------------------------------------------------------------
void ValContVecDouble::FillVecs(const std::vector<int> &data_ind, const double* const *smry, int len)		// fills MPI-local 'Vecs' from comm-RANKS-0 'smry'; 'smry' is referenced on comm-RANKS-0
{
	if (comm == MPI_COMM_NULL)
		return;

	int rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	Vecs = std::vector<std::vector<double>>(data_ind[rank+1] - data_ind[rank]);		// create the container

	if (rank == 0)
	{
		for (int r = 0; r < size; r++)	// loop through ranks
		{
			for (int i = data_ind[r]; i < data_ind[r+1]; i++)		// take the data indices relating to rank "r"
			{
				// make auxiliary vector for data point "i"
				std::vector<double> vecaux(len);
				for (int j = 0; j < len; j++)
					vecaux[j] = smry[j][i];

				// send 'vecaux'
				if (r > 0)
					MPI_Ssend(vecaux.data(), len, MPI_DOUBLE, r, 0, comm);
				else
					Vecs[i - data_ind[r]] = std::move(vecaux);
			}
		}
	}
	else
	{
		for (int i = data_ind[rank]; i < data_ind[rank+1]; i++)		// take the data indices relating to "rank"
		{
			MPI_Status stat;
			Vecs[i - data_ind[rank]] = std::vector<double>(len);
			MPI_Recv(Vecs[i - data_ind[rank]].data(), len, MPI_DOUBLE, 0, 0, comm, &stat);
		}
	}

#ifdef TESTNEWPROXY
	std::ofstream file0(HMMPI::stringFormatArr(test_fn, std::vector<int>{rank}), std::ios_base::app);
	std::cout << "loc. rank " << rank << ", loc. size " << size << "\tValContVecDouble::FillVecs, data indices " << data_ind[rank] << " --- " << data_ind[rank+1] << "\n";
	file0 << "loc. rank " << rank << ", loc. size " << size << "\tValContVecDouble::FillVecs, data indices " << data_ind[rank] << " --- " << data_ind[rank+1] << "\n";;
	file0.close();
#endif
}
//---------------------------------------------------------------------------
ValContVecDouble::ValContVecDouble(MPI_Comm c, std::vector<int> data_ind, const double* const *smry, int len, int smry_len) : ValCont(c)
{
	if (comm == MPI_COMM_NULL)
		return;

	int size;
	MPI_Comm_size(comm, &size);

	HMMPI::Bcast_vector(data_ind, 0, comm);

	if ((int)data_ind.size() != size + 1)
		throw HMMPI::Exception("Размер data_ind не соответствует коммуникатору в ValContVecDouble::ValContVecDouble",
							   "Size of data_ind is not consistent with communicator in ValContVecDouble::ValContVecDouble");
	if (data_ind[0] != 0 || data_ind[size] != smry_len)
		throw HMMPI::Exception(HMMPI::stringFormatArr(HMMPI::MessageRE(
				"Неправильный первый ({0:%d}) / последний ({1:%d}) элемент data_ind в ValContVecDouble::ValContVecDouble (smry_len = {2:%d})",
				"Wrong first ({0:%d}) / last ({1:%d}) element of data_ind in ValContVecDouble::ValContVecDouble (smry_len = {2:%d})"), std::vector<int>{data_ind[0], data_ind[size], smry_len}));

	FillVecs(data_ind, smry, len);
	Grads = std::vector<std::vector<std::vector<double>>>(Vecs.size());
}
//---------------------------------------------------------------------------
ValContVecDouble::ValContVecDouble(MPI_Comm c, std::vector<int> data_ind, const std::vector<std::vector<double>> &v) : ValCont(c)
{
	if (comm == MPI_COMM_NULL)
		return;

	int rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	int len = v.size();
	int smry_len = 0;
	if (len > 0)
		smry_len = v[0].size();

	MPI_Bcast(&len, 1, MPI_INT, 0, comm);
	MPI_Bcast(&smry_len, 1, MPI_INT, 0, comm);
	HMMPI::Bcast_vector(data_ind, 0, comm);
	if (len == 0)
	{
		Vecs = std::vector<std::vector<double>>(data_ind[rank+1] - data_ind[rank]);		// create the container with empty elements
		Grads = std::vector<std::vector<std::vector<double>>>(Vecs.size());
		return;
	}

#ifdef TESTNEWPROXY
	std::cout << "loc. rank " << rank << "\tlen = " << len << ", smry_len = " << smry_len << "\n";
#endif

	if ((int)data_ind.size() != size + 1)
		throw HMMPI::Exception("Размер data_ind не соответствует коммуникатору в ValContVecDouble::ValContVecDouble",
							   "Size of data_ind is not consistent with communicator in ValContVecDouble::ValContVecDouble");
	if (data_ind[0] != 0 || data_ind[size] != smry_len)
		throw HMMPI::Exception(HMMPI::stringFormatArr(HMMPI::MessageRE(
				"Неправильный первый ({0:%d}) / последний ({1:%d}) элемент data_ind в ValContVecDouble::ValContVecDouble (smry_len = {2:%d})",
				"Wrong first ({0:%d}) / last ({1:%d}) element of data_ind in ValContVecDouble::ValContVecDouble (smry_len = {2:%d})"), std::vector<int>{data_ind[0], data_ind[size], smry_len}));

	const double **smry0 = 0;
	if (rank == 0)
	{
		smry0 = new const double*[len];
		for (int i = 0; i < len; i++)
			smry0[i] = v[i].data();
	}

	FillVecs(data_ind, smry0, len);
	Grads = std::vector<std::vector<std::vector<double>>>(Vecs.size());

	delete [] smry0;
}
//---------------------------------------------------------------------------
ValContVecDouble::ValContVecDouble(MPI_Comm c, PhysModel *pm, const std::vector<std::vector<double>> &pop, int train_from_dump, const std::vector<int> &data_ind) : ValCont(c)
{
	if (train_from_dump == -1)
		RunTrainPopulation(pm, pop);
	else
	{
		char fname[100];
		sprintf(fname, dump_Ity, train_from_dump, 0, 0);		// NOTE _rnk0_pr0 is used

		smry_len = pm->ModelledDataSize();
		int popsize = pop.size();
		MPI_Bcast(&popsize, 1, MPI_INT, 0, comm);

		FitSmryFromFile(fname, popsize);
	}

	write_FIT_SMRY();		// debug output to file

	int size;
	MPI_Comm_size(comm, &size);

	if ((int)data_ind.size() != size + 1)
		throw HMMPI::Exception("Размер data_ind не соответствует коммуникатору в ValContVecDouble::ValContVecDouble",
							   "Size of data_ind is not consistent with communicator in ValContVecDouble::ValContVecDouble");
	if (data_ind[0] != 0 || data_ind[size] != smry_len)
		throw HMMPI::Exception(HMMPI::stringFormatArr(HMMPI::MessageRE(
				"Неправильный первый ({0:%d}) / последний ({1:%d}) элемент data_ind в ValContVecDouble::ValContVecDouble (smry_len = {2:%d})",
				"Wrong first ({0:%d}) / last ({1:%d}) element of data_ind in ValContVecDouble::ValContVecDouble (smry_len = {2:%d})"), std::vector<int>{data_ind[0], data_ind[size], smry_len}));

	FillVecs(data_ind, SMRY, len);
	Grads = std::vector<std::vector<std::vector<double>>>(Vecs.size());
	// no gradients taken!

	if (SMRY != 0)
	{
		for (int i = 0; i < len; i++)
			delete [] SMRY[i];
		delete [] SMRY;
	}
	delete [] FIT;

	FIT = 0;
	SMRY = 0;
}
//---------------------------------------------------------------------------
ValContVecDouble::ValContVecDouble(MPI_Comm c, std::vector<int> data_ind, const std::vector<HMMPI::Mat> &data_sens) : ValCont(c)	// data_sens[len_gr][smry_len, fulldim]	(from c-RANKS-0) is distributed to fill 'Grads' according to 'data_ind' (c-RANKS-0)
{																																	// This CTOR does not fill 'Vecs'
	if (comm == MPI_COMM_NULL)
		return;

	int rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	int len_gr = data_sens.size();					// number of design points with gradients
	int smry_len = 0, fulldim = 0;					// total number of data points; full dimension
	if (len_gr > 0)
	{
		smry_len = data_sens[0].ICount();
		fulldim = data_sens[0].JCount();
	}

	MPI_Bcast(&len_gr, 1, MPI_INT, 0, comm);
	MPI_Bcast(&smry_len, 1, MPI_INT, 0, comm);
	MPI_Bcast(&fulldim, 1, MPI_INT, 0, comm);
	HMMPI::Bcast_vector(data_ind, 0, comm);
	if (len_gr == 0)
	{
		Vecs = std::vector<std::vector<double>>(data_ind[rank+1] - data_ind[rank]);		// create the container with empty elements
		Grads = std::vector<std::vector<std::vector<double>>>(Vecs.size());
		return;
	}

	assert(data_ind.size() == (size_t)size + 1);
	std::vector<std::vector<double>> work2(len_gr);				// entries of this vector are empty vectors
	Grads = std::vector<std::vector<std::vector<double>>>(data_ind[rank+1] - data_ind[rank], work2);		// create the container Grads[smry_len][len_gr][fulldim]; the third dimension is empty so far

	if (rank == 0)
	{
		for (int r = 0; r < size; r++)							// loop through ranks
			for (int i = data_ind[r]; i < data_ind[r+1]; i++)		// take the data indices relating to rank "r" -- smry_len
				for (int p = 0; p < len_gr; p++)						// indices of design points
				{
					std::vector<double> work(fulldim, 0.0);
					for (int j = 0; j < fulldim; j++)
						work[j] = data_sens[p](i, j);

					// send 'work'
					if (r > 0)
						MPI_Ssend(work.data(), fulldim, MPI_DOUBLE, r, 0, comm);
					else
						Grads[i - data_ind[r]][p] = std::move(work);
				}
	}
	else
	{
		for (int i = data_ind[rank]; i < data_ind[rank+1]; i++)		// take the data indices relating to "rank"
			for (int p = 0; p < len_gr; p++)						// indices of design points
			{
				MPI_Status stat;
				Grads[i - data_ind[rank]][p] = std::vector<double>(fulldim);
				MPI_Recv(Grads[i - data_ind[rank]][p].data(), fulldim, MPI_DOUBLE, 0, 0, comm, &stat);
			}
	}
	Vecs = std::vector<std::vector<double>>(Grads.size());
}
//---------------------------------------------------------------------------
int ValContVecDouble::vals_count() const
{
	// Vecs[smry_len][len]
	// Grads[smry_len][len_gr][fulldim]
	int vc = 0;
	if (Vecs.size() > 0)
		vc = Vecs[0].size();

	MPI_Allreduce(MPI_IN_PLACE, &vc, 1, MPI_INT, MPI_MAX, comm);
	return vc;
}
//---------------------------------------------------------------------------
int ValContVecDouble::total_count() const
{
	int vc = 0, gc = 0;
	if (Vecs.size() > 0)
		vc = Vecs[0].size();
	if (Grads.size() > 0)
		gc = Grads[0].size();

	int res = vc + gc;
	MPI_Allreduce(MPI_IN_PLACE, &res, 1, MPI_INT, MPI_MAX, comm);
	return res;
}
//---------------------------------------------------------------------------								// distributes the stored values/gradients to the given proxies; 'inds' shows which data points (with func. values) should be taken
void ValContVecDouble::DistrValues(std::vector<KrigEnd> &dep, const std::vector<size_t> &inds) const		// all points with func. gradients are taken (no subset is extracted)
{																											// 'dep' should be consistent with local (current rank's) 'Vecs' and 'Grads'
	// Each rank executes different stuff

	// Vecs[smry_len][len]
	// Grads[smry_len][len_gr][fulldim]

	if (Vecs.size() > 0 && dep.size() != Vecs.size())
		throw HMMPI::Exception("Size mismatch of std::vector<PM_Proxy*> and Vecs in ValContVecDouble::DistrValues");
	if (Grads.size() > 0 && dep.size() != Grads.size())
		throw HMMPI::Exception("Size mismatch of std::vector<PM_Proxy*> and Grads in ValContVecDouble::DistrValues");

	for (size_t i = 0; i < Vecs.size(); i++)	// take data point "i"
	{
		dep[i].push_back_vals(HMMPI::Reorder(Vecs[i], inds));

#ifdef TESTNEWPROXY
		int leny0 = inds.size();
		int rank00, size00;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank00);
		MPI_Comm_size(MPI_COMM_WORLD, &size00);
		std::ofstream file0(HMMPI::stringFormatArr(test_fn, std::vector<int>{rank00}), std::ios_base::app);
		std::cout << "rank " << rank00 << ", size " << size00 << "\tValContVecDouble::DistrValues, added " << leny0 << " values to " << &dep[i] << "\n";
		file0 << "rank " << rank00 << ", size " << size00 << "\tValContVecDouble::DistrValues, added " << leny0 << " values to " << &dep[i] << "\n";
		file0.close();
#endif
	}

	for (size_t i = 0; i < Grads.size(); i++)	// take data point with func. gradients "i"
		dep[i].push_back_grads(Grads[i]);
}
//---------------------------------------------------------------------------
void ValContVecDouble::Add(const ValContVecDouble &b)				// appends data from "b" to 'this'; MPI layout should be the same
{
	// MPI-distributed Vecs[smry_len][len]
	// MPI-distributed Grads[smry_len][len_gr][fulldim]

	if (Vecs.size() != b.Vecs.size())
		throw HMMPI::Exception(HMMPI::stringFormatArr("Vecs.size() ({0:%zu}) != b.Vecs.size() ({1:%zu}) in ValContVecDouble::Add", std::vector<size_t>{Vecs.size(), b.Vecs.size()}));
	assert(Grads.size() == b.Grads.size());
	assert(Vecs.size() == Grads.size());
	for (size_t i = 0; i < Vecs.size(); i++)
	{
		HMMPI::VecAppend(Vecs[i], b.Vecs[i]);
		HMMPI::VecAppend(Grads[i], b.Grads[i]);
	}
}
//---------------------------------------------------------------------------
// ValContSimProxy
//---------------------------------------------------------------------------
ValContSimProxy::ValContSimProxy(MPI_Comm c, std::vector<int> data_ind, const std::vector<std::vector<double>> &v) : ValCont(c)
{
	if (comm == MPI_COMM_NULL)
		return;

	int rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	int smry_len = v.size();
	MPI_Bcast(&smry_len, 1, MPI_INT, 0, comm);
	HMMPI::Bcast_vector(data_ind, 0, comm);

	if ((int)data_ind.size() != size + 1)
		throw HMMPI::Exception("Размер data_ind не соответствует коммуникатору в ValContSimProxy::ValContSimProxy",
							   "Size of data_ind is not consistent with communicator in ValContSimProxy::ValContSimProxy");
	if (data_ind[0] != 0 || data_ind[size] != smry_len)
		throw HMMPI::Exception(HMMPI::stringFormatArr(HMMPI::MessageRE(
				"Неправильный первый ({0:%d}) / последний ({1:%d}) элемент data_ind в ValContSimProxy::ValContSimProxy (smry_len = {2:%d})",
				"Wrong first ({0:%d}) / last ({1:%d}) element of data_ind in ValContSimProxy::ValContSimProxy (smry_len = {2:%d})"), std::vector<int>{data_ind[0], data_ind[size], smry_len}));

	Vecs = std::vector<std::vector<double>>(data_ind[rank+1] - data_ind[rank]);		// create the container
	if (rank == 0)
	{
		for (int r = 0; r < size; r++)								// loop through ranks
			for (int i = data_ind[r]; i < data_ind[r+1]; i++)		// take the data indices relating to rank "r"
			{
				if (r > 0)		// send v[i]
				{
					int len = v[i].size();
					MPI_Ssend(&len, 1, MPI_INT, r, 0, comm);
					MPI_Ssend(v[i].data(), len, MPI_DOUBLE, r, 0, comm);
				}
				else
					Vecs[i - data_ind[r]] = v[i];
			}
	}
	else
	{
		for (int i = data_ind[rank]; i < data_ind[rank+1]; i++)		// take the data indices relating to "rank"
		{
			MPI_Status stat;
			int len;
			MPI_Recv(&len, 1, MPI_INT, 0, 0, comm, &stat);
			Vecs[i - data_ind[rank]] = std::vector<double>(len);
			MPI_Recv(Vecs[i - data_ind[rank]].data(), len, MPI_DOUBLE, 0, 0, comm, &stat);
		}
	}

#ifdef TESTNEWPROXY
	std::ofstream file0(HMMPI::stringFormatArr(test_fn, std::vector<int>{rank}), std::ios_base::app);
	std::cout << "loc. rank " << rank << ", loc. size " << size << "\tValContSimProxy::ValContSimProxy, data indices " << data_ind[rank] << " --- " << data_ind[rank+1] << "\n";
	file0 << "loc. rank " << rank << ", loc. size " << size << "\tValContSimProxy::ValContSimProxy, data indices " << data_ind[rank] << " --- " << data_ind[rank+1] << "\n";;
	file0.close();
#endif
}
//---------------------------------------------------------------------------
int ValContSimProxy::vals_count() const					// number of design points with func. values (to distinguish from gradients)
{
	// Vecs[smry_len][...]
	int max = 0;
	for (size_t i = 0; i < Vecs.size(); i++)
		if ((int)Vecs[i].size() > max)
			max = Vecs[i].size();

	MPI_Allreduce(MPI_IN_PLACE, &max, 1, MPI_INT, MPI_MAX, comm);
	return max;
}
//---------------------------------------------------------------------------
int ValContSimProxy::total_count() const				// total number of design points (func. values + gradients)
{
	return vals_count();	// no gradients so far - TODO
}
//---------------------------------------------------------------------------
void ValContSimProxy::DistrValues(std::vector<KrigEnd> &dep, const std::vector<size_t> &inds) const
{
	// this function repeats the code from ValContVecDouble::DistrValues
	// each rank executes different stuff

	// Determine whether the Vecs have the same size - to decide whether 'inds' should be applied
	int Np = 0;									// length of Vecs[0] on rank-0
	if (Vecs.size() > 0)
		Np = Vecs[0].size();
	MPI_Bcast(&Np, 1, MPI_INT, 0, comm);

	int same_size = 1;							// 1, if all Vecs[i] on all ranks have size Np
	for (size_t i = 0; i < Vecs.size(); i++)	// Vecs[smry_len][...]
		if ((int)Vecs[i].size() != Np)
		{
			same_size = 0;
			break;
		}

	MPI_Allreduce(MPI_IN_PLACE, &same_size, 1, MPI_INT, MPI_MIN, comm);

	if (dep.size() != Vecs.size())
		throw HMMPI::Exception("Не совпадают длины vector<KrigEnd> и Vecs в ValContSimProxy::DistrValues",
							   "Size mismatch of vector<KrigEnd> and Vecs in ValContSimProxy::DistrValues");

	for (size_t i = 0; i < Vecs.size(); i++)	// take data point "i"
	{
		if (same_size)
			dep[i].push_back_vals(HMMPI::Reorder(Vecs[i], inds));
		else
			dep[i].push_back_vals(Vecs[i]);		// if Vecs of different size are present, don't apply 'inds'!
	}
}
//---------------------------------------------------------------------------
