/*
 * ParsingParams.cpp
 *
 *  Created on: Mar 25, 2013
 *      Author: ilya
 */

#define _BSD_SOURCE

#include "Abstract.h"
#include "MathUtils.h"
#include "lapacke_select.h"
#include "mpi.h"
#include "Vectors.h"
#include "MonteCarlo.h"
#include "Parsing.h"
#include "Parsing2.h"
#include "PhysModels.h"
#include "ConcretePhysModels.h"
#include "CornerPointGrid.h"
#include "CMAES_interface.h"
#include "Tracking.h"
#include "GradientOpt.h"
#include <algorithm>
#include <cmath>
#include <cassert>
#include <chrono>
#include <string>
#include <limits>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <sys/stat.h>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
	// TODO find some alternative - for RUNMPICHECK 
#else
	#include <sys/unistd.h>
	#include <sys/socket.h>
	#include <netdb.h>
	#include <netinet/in.h>
	#include <arpa/inet.h>
#endif

//------------------------------------------------------------------------------------------
// KW_run derivatives
//------------------------------------------------------------------------------------------
KW_echo::KW_echo()
{
	name = "ECHO";
}
//------------------------------------------------------------------------------------------
void KW_echo::Run()
{
	K->echo = true;
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_noecho::KW_noecho()
{
	name = "NOECHO";
}
//------------------------------------------------------------------------------------------
void KW_noecho::Run()
{
	K->echo = false;
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_runForward::KW_runForward()
{
	name = "RUNFORWARD";
}
//------------------------------------------------------------------------------------------
void KW_runForward::Run()
{
	ModelFactory Factory;
	std::string msg0;
	PhysModel *PM = Factory.Make(msg0, K, this, this->CWD, 1);

	const ParamsInterface *par_interface = dynamic_cast<const ParamsInterface*>(PM->GetConstr());
	if (par_interface == nullptr)
		throw HMMPI::Exception("PM->BoundConstr cannot be cast to ParamsInterface");

	K->AppText((std::string)HMMPI::MessageRE("Модель: ", "Model: ") + msg0 + "\n" + PM->proc_msg());

	if (PM->GetComm() == MPI_COMM_SELF && K->MPI_rank != 0)		// if PM is not communicating, and WORLD-rank is not master, leave
		return;

	Sim_small_interface *PMsim = dynamic_cast<Sim_small_interface*>(PM); 			// one specific setting for PhysModelHM, PMEclipse and their Posteriors
	if (PMsim != nullptr && PMsim->is_sim())
		PMsim->set_ignore_small_errors(false);

	std::string msg = par_interface->msg();
	std::vector<double> p = par_interface->get_init_act();
	if (!PM->CheckLimits_ACT(p))
	{
		K->AppText("WARNING: " + PM->get_limits_msg());
		K->TotalWarnings++;
	}
	K->AppText("\n" + msg + "\n");

	double of = PM->ObjFunc_ACT(p);

	K->AppText(PM->ObjFuncMsg());
	K->AppText(HMMPI::stringFormatArr("o.f. = {0:%-18.16g}\n", std::vector<double>{of}));

	// DEBUG
	if (K->MPI_rank == 0)
	{
		K->AppText("Modelled data:\n");			// output modelled data
		if (PM->ModelledDataSize() != 0)
			K->AppText(HMMPI::ToString(PM->ModelledData(), "%8.5g"));	// DEBUG
		else
			K->AppText("EMPTY\n");
	}
	// DEBUG



	// DEBUG ---------- some cache testing
//	Start_pre();
//	IMPORTKWD(points, KW_3points, "3POINTS");			// 3POINTS->x,y provide 'x0', 'p' for VM_Ham_eq2[_eps]
//	IMPORTKWD(opt, KW_optimization, "OPTIMIZATION");	// epsG provides 'eps' (or 'M0'), maa provides 'i0'
//	Finish_pre();
//
//	PM_FullHamiltonian Ham0(PM), Ham1(PM);
//	VM_Ham_eq2 VM2(&Ham0, &Ham1);
//	VM2.x0 = Ham0.act_par(points->x);
//	VM2.p = Ham0.act_par(points->y);
//	VM2.eps = 3.81456;
//	VM_Ham_eq2_eps VM2eps(VM2, opt->maa, opt->epsG);
//
//	std::vector<double> x = VM2eps.map_xfull_x(p);
//
//	Ham0.pact = Ham1.pact = VM2.p;
//	std::vector<double> dh0 = Ham0.dHdp.Get(&Ham0, std::pair<std::vector<double>, std::vector<double>>(VM2.x0, VM2.p));
//	std::vector<double> dh1 = Ham1.dHdp.Get(&Ham1, std::pair<std::vector<double>, std::vector<double>>(p, VM2.p));
//	HMMPI::Mat dh1aux = Ham1.Gaux_grad.Get(&Ham1, std::pair<std::vector<double>, std::vector<double>>(p, VM2.p));
//
//	std::vector<double> func0 = VM2.Func_ACT(p);
//	HMMPI::Mat jac0 = VM2.Jac_ACT(p);
//
//	std::vector<double> func = VM2eps.Func_ACT(x);
//	HMMPI::Mat jac = VM2eps.Jac_ACT(x);
//	std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++\n";
//	std::cout << "xfull\t" << HMMPI::ToString(p);
//	std::cout << "x    \t" << HMMPI::ToString(x);
//
//	std::cout << "dh0  \t" << HMMPI::ToString(dh0);
//	std::cout << "dh1  \t" << HMMPI::ToString(dh1);
//	std::cout << "dh1aux\n" << dh1aux.ToString();
//
//	std::cout << "func0\t" << HMMPI::ToString(func0);
//	std::cout << "jac0\n" << jac0.ToString();
//	std::cout << "func \t" << HMMPI::ToString(func);
//	std::cout << "jac\n" << jac.ToString();
//	std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++\n";
	// DEBUG

	// DEBUG
//	PM_FullHamiltonian *Ham1 = dynamic_cast<PM_FullHamiltonian*>(PM);
//	if (Ham1 != nullptr)
//	{
//		Ham1->tot_par(p);
//		HMMPI::Mat iMM = Ham1->invG.Get(Ham1, p);
//		std::vector<double> grad_p = Ham1->ObjFuncGrad_momentum_ACT(p);
//		if (K->MPI_rank == 0)
//		{
//			std::cout << "\nMM\n" << iMM.ToString() << "\n";
//			std::cout << "dH/dp\n" << HMMPI::ToString(grad_p) << "\n";
//		}
//	}
	// DEBUG


	// DEBUG----------------------------------------------------------***	FILE OUTPUT!!
//	PM_FullHamiltonian Ham(PM);
//	HMMPI::Mat G = Ham.G.Get(&Ham, p);
//	HMMPI::Mat invG = Ham.invG.Get(&Ham, p);
//
//	std::vector<HMMPI::Mat> dxiG(p.size());
//	for (size_t i = 0; i < p.size(); i++)
//		dxiG[i] = Ham.dxi_G[i].Get(&Ham, p);
//
//	char fname[HMMPI::BUFFSIZE];
//	sprintf(fname, "x_Hamiltonian_check_%d.txt", K->MPI_rank);
//	FILE *f = fopen(fname, "w");
//	Ham.nums_starts_to_file(f);
//	fprintf(f, "Mass matrix\n");
//	fputs(G.ToString().c_str(), f);
//
//	fprintf(f, "\nMass matrix inverse\n");
//	fputs(invG.ToString().c_str(), f);
//
//	//------------------
//	fprintf(f, "\nMass matrix-1\n");
//	fputs(Gaux1.ToString().c_str(), f);
//
//	fprintf(f, "\nMass matrix inverse-1\n");
//	fputs(invGaux1.ToString().c_str(), f);
//
//	//------------------
//	fprintf(f, "\nMass matrix-2\n");
//	fputs(Gaux2.ToString().c_str(), f);
//
//	fprintf(f, "\nMass matrix inverse-2\n");
//	fputs(invGaux2.ToString().c_str(), f);
//
//
//	for (size_t i = 0; i < p.size(); i++)
//	{
//		fprintf(f, "\ndMM/dx_%zu\n", i);
//		fputs(dxiG[i].ToString().c_str(), f);
//	}
//	for (size_t i = 0; i < p.size(); i++)
//		dxiG[i] = Ham.dxi_G[i].Get(&Ham, p);
//
//	fclose(f);
	// DEBUG----------------------------------------------------------***


}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_runSingle::KW_runSingle()
{
	name = "RUNSINGLE";
}
//------------------------------------------------------------------------------------------
void KW_runSingle::Run()
{
	Start_pre();
	IMPORTKWD(model, KW_model, "MODEL");
	IMPORTKWD(parameters, KW_parameters, "PARAMETERS");
	DECLKWD(smry, KW_eclsmry, "ECLSMRY");
	DECLKWD(templ, KW_templates, "TEMPLATES");
	Finish_pre();

	PhysModel *PM = model->MakeModel(this, this->CWD, true);		// make the posterior
	K->AppText((std::string)HMMPI::MessageRE("Модель: ", "Model: ") + "POSTERIOR(" + model->type + ")\n" + PM->proc_msg());

	Sim_small_interface *PMecl = dynamic_cast<Sim_small_interface*>(PM);
	if (PMecl != nullptr && PMecl->is_sim())						// one specific setting for PMEclipse/posterior
		PMecl->set_ignore_small_errors(false);

#ifdef TEMPLATES_KEEP_NO_ASCII
	templ->set_keep("FIRST");
#endif

	std::string msg = parameters->msg(K->StrListN());
	std::vector<double> p = parameters->get_init_act();
	if (!PM->CheckLimits_ACT(p))
	{
		K->AppText("WARNING: " + PM->get_limits_msg());		// checking bounds
		K->TotalWarnings++;
	}
	K->AppText("\n" + msg + "\n");

	double of = PM->ObjFunc_ACT(p);
	K->AppText(PM->ObjFuncMsg());
	K->AppText(HMMPI::stringFormatArr("o.f. = {0}\n\n", std::vector<double>{of}));

	if (PMecl != nullptr && PMecl->is_sim())
	{
		if (smry->GetState() == "")				// adding summary to ECLSMRY
		{
			K->AppText(HMMPI::MessageRE("Модель " + model->type + " добавляется в ECLSMRY\n", "Adding model " + model->type + " to ECLSMRY\n"));
			std::string msg1 = smry->get_Data().AddModel(parameters->name, parameters->val, parameters->backval, PMecl->get_smry());
			std::string msg2 = smry->Save();
			K->AppText(msg1);
			K->AppText(msg2);
		}
		else
			K->AppText(HMMPI::MessageRE("ECLSMRY не задано, и не будет обновлено\n", "ECLSMRY is not defined, and will not be updated\n"));
	}

	// DEBUG
//	K->AppText("ModelledData\n");						// DEBUG
//	if (K->MPI_rank == 0)
//		K->AppText(HMMPI::ToString(PM->ModelledData()));	// DEBUG
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_runMultiple::KW_runMultiple()
{
	name = "RUNMULTIPLE";
}
//------------------------------------------------------------------------------------------
void KW_runMultiple::Run()		// multiple run of PMEclipse, all resulting summaries are added to ECLSMRY file; sequence of parameters is taken according to MULTIPLE_SEQ
{
	Start_pre();
	IMPORTKWD(model, KW_model, "MODEL");
	IMPORTKWD(parameters, KW_parameters, "PARAMETERS");
	IMPORTKWD(smry, KW_eclsmry, "ECLSMRY");
	IMPORTKWD(seq, KW_multiple_seq, "MULTIPLE_SEQ");
	DECLKWD(templ, KW_templates, "TEMPLATES");
	Finish_pre();

	const std::string modtype_cache = model->type;
	model->type = "SIM";

	PhysModel *PM = model->MakeModel(this, this->CWD, true);		// make the posterior
	K->AppText((std::string)HMMPI::MessageRE("Модель: ", "Model: ") + "POSTERIOR(" + model->type + ")\n" + PM->proc_msg());
	K->AppText(seq->msg());

	Sim_small_interface *PMecl = dynamic_cast<Sim_small_interface*>(PM);
	assert(PMecl != nullptr && PMecl->is_sim());
	PMecl->set_ignore_small_errors(true);

#ifdef TEMPLATES_KEEP_NO_ASCII
	templ->set_keep("NONE");
#endif

	long long int seed = seq->seed;
	std::vector<std::vector<double>> params;						// will store internal representation, full-dim points
	if (seq->type == "SOBOL")
		params = parameters->SobolSequence(seq->N, seed);
	else if (seq->type == "RANDGAUSS")
		params = parameters->NormalSequence(seq->N, (unsigned int)seed, seq->R);
	else
		throw HMMPI::Exception("Wrong seq->type in KW_runMultiple::Run");

	// save the parameters sequence: header
	RANK0_SYNCERR_BEGIN(MPI_COMM_WORLD);
		FILE *fd = fopen(seq->logfile.c_str(), "w");
		if (fd != NULL)
		{
			fputs(HMMPI::ToString(parameters->name, "%-17.17s").c_str(), fd);
			fclose(fd);
		}
		else
			throw HMMPI::Exception("Cannot open file for writing " + seq->logfile);
	RANK0_SYNCERR_END(MPI_COMM_WORLD);

	time_t t0, t1;
	time(&t0);
	int i;
	for (i = 0; i < seq->N; i++)
	{
		double of = PMecl->ObjFunc(params[i]);
		K->AppText(HMMPI::stringFormatArr("\nМодель {0:%d}, ", "\nModel {0:%d}, ", i+1));
		K->AppText(HMMPI::stringFormatArr("o.f. = {0}\n", std::vector<double>{of}));

		K->AppText(HMMPI::MessageRE("Модель " + model->type + " добавляется в ECLSMRY\n", "Adding model " + model->type + " to ECLSMRY\n"));
		std::string msg1 = smry->get_Data().AddModel(parameters->name, parameters->InternalToExternal(params[i]), parameters->backval, PMecl->get_smry());
		std::string msg2 = smry->Save();
		K->AppText(msg1);
		K->AppText(msg2);

		// save the parameters sequence
		RANK0_SYNCERR_BEGIN(MPI_COMM_WORLD);
			FILE *fd = fopen(seq->logfile.c_str(), "a");
			if (fd != NULL)
			{
				fputs(HMMPI::ToString(parameters->InternalToExternal(params[i]), "%-17.12g").c_str(), fd);
				fclose(fd);
			}
			else
				throw HMMPI::Exception("Cannot open file for writing " + seq->logfile);
		RANK0_SYNCERR_END(MPI_COMM_WORLD);

		time(&t1);
		double dT = difftime(t1, t0)/double(3600);
		MPI_Bcast(&dT, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);		// dT sync
		if (dT > seq->MaxHours)
		{
			i++;
			break;
		}
	}

	if (i < seq->N)
		K->AppText(HMMPI::stringFormatArr("\n>> За отведенное время было просчитано {0:%d} модел(ей)\n",
										  "\n>> During the specified time {0:%d} model(s) were run\n", i));

	model->type = modtype_cache;		// restore model type
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_runOptProxy::KW_runOptProxy()
{
	name = "RUNOPTPROXY";
}
//------------------------------------------------------------------------------------------
void KW_runOptProxy::Run()
{
	// _NOTE_ Group property of iterations: [.][.] = [..] when intermediate PARAMETERS and Rk are properly selected, and LMstart = SIMBEST; for LMstart = CURR this does not generally hold

	// _NOTE_ When min/max change (-> params scaling changes), proxy behaviour changes because R is fixed (optimization behaviour may also depend on r0, rmin);
	// min/max for different parameters should be selected in a balanced manner

	// TODO[i] when parameters are permuted, results change (e.g. after 13 iterations, with  gradual kick-in)
	// looks like numerical instability

	Start_pre();
	IMPORTKWD(model, KW_model, "MODEL");
	IMPORTKWD(parameters, KW_parameters, "PARAMETERS");
	IMPORTKWD(config, KW_opt_config, "OPT_CONFIG");
	IMPORTKWD(smry, KW_eclsmry, "ECLSMRY");
	DECLKWD(templ, KW_templates, "TEMPLATES");
	Finish_pre();

	const double model_R_cache = model->R;
	const std::string modtype_cache = model->type;
	model->type = "PROXY";

	const std::string params_log_file = HMMPI::getFullPath(this->CWD, "ParamsLog.txt");			// resulting solution vector is saved here
	const std::string progress_file = HMMPI::getFullPath(this->CWD, "ObjFunc_progress.txt");

	RANK0_SYNCERR_BEGIN(MPI_COMM_WORLD);
		FILE *f = fopen(progress_file.c_str(), "w");
		if (f != 0)
		{
			fprintf(f, "%-5.5s\t%-8.8s\t%-10.10s\t%-11.11s\t%-17.17s\t%-10.10s\t%-2.2s\t%-17.17s\t%-12.12s\t%-12.12s\n", "#MOD", "HOURS", "DX", "PROXY", "SIM", "Tk", "*", "SIMBEST", "DIST_MIN", "DIST_AVG");
			fclose(f);
		}
	RANK0_SYNCERR_END(MPI_COMM_WORLD);

	std::vector<double> p = parameters->init;		// full-dim, internal params
	std::vector<double> simbest = p;				// p - current params, simbest - params for simulator-best point
	double of_simbest = std::numeric_limits<double>::max();				// obj. func. for simulator-best point; it's redefined below!

#ifdef TEMPLATES_KEEP_NO_ASCII
	templ->set_keep("NONE");
#endif

	double simk = 0;					// simulator o.f. at starting point of each iteration
	double Rk = 0;						// sphere/cube radius for restricted step; Rk == 0 means no restricted step
	bool request_Rk_decr = false;		// 'true' if the previous iteration requested Rk to decrease; two such consecutive requests trigger the actual decrease
	if (config->r0 > 0)
		Rk = config->r0;

	K->AppText((std::string)HMMPI::MessageRE("Оптимизация прокси: ", "Proxy optimization: ") + "LM(" + config->LM_mat + ")\n");

	int finished = 0;
	int iter = 0;
	time_t t0, t1;
	time(&t0);
	while (!finished)
	{
		K->AppText(HMMPI::stringFormatArr("\n-- Итерация {0:%d} --\n", "\n-- Iteration {0:%d} --\n", iter+1));
		bool write_params_log = false;			// flag to activate output of ParamsLog.txt

		if (Rk != 0)
			model->R = model_R_cache * HMMPI::Min(Rk*10, 1);			// scale the correlation radius directly in the keyword		***** TODO HERE hardcoded factor "10", it could be set as a parameter from the control file

		PhysModel *PMproxy;
		if (iter == 0)							// startup calculations
		{
			K->AppText(">> ");
			const double model_nug_cache = model->nugget;
			model->nugget = 0;											// zero nugget to get the exact proxy behavior
			PMproxy = model->MakeModel(this, this->CWD, true);			// make the posterior <- proxy, with zero nugget

			of_simbest = PMproxy->ObjFunc(simbest);						// define the starting SIMBEST value using proxy
			MPI_Bcast(&of_simbest, 1, MPI_DOUBLE, 0, PMproxy->GetComm());
			simk = of_simbest;											// define the starting "simk"; NOTE: this definition using proxy is approximate; it is only exact if the initial point is in ECLSMRY

			K->AppText(HMMPI::stringFormatArr("Начальная ц.ф. = {0}\n", "Initial o.f. = {0}\n", of_simbest));
			model->nugget = model_nug_cache;							// restore the nugget
		}

		PMproxy = model->MakeModel(this, this->CWD, true);				// make the posterior
		Sim_small_interface *PMecl = dynamic_cast<Sim_small_interface*>(model->MakeModel(this, this->CWD, true, "SIM"));	// make the posterior
		K->AppText(PMproxy->proc_msg());
		assert(PMecl != nullptr && PMecl->is_sim());
		PMecl->set_ignore_small_errors(true);

		OptCtxLM optctx(config->LMmaxit, config->epsG, config->epsF, config->epsX);
		OptCtxLM optctx_spher(config->LMmaxit_spher, config->epsG, config->epsF, config->epsX);
		Optimizer *Opt = nullptr;
		if (config->LM_mat == "HESS")
			Opt = Optimizer::Make("LM");
		else if (config->LM_mat == "FI")
			Opt = Optimizer::Make("LMFI");
		else if (config->LM_mat == "FIMIX")
			Opt = Optimizer::Make("LMFIMIX");
		else
			throw HMMPI::Exception("Wrong OPT_CONFIG.LM_mat in KW_runOptProxy::Run");

		// optimize proxy
		std::chrono::high_resolution_clock::time_point time1 = std::chrono::high_resolution_clock::now(), time2, time3, time4;
		std::vector<double> LM_start;
		if (config->LMstart == "SIMBEST")
			LM_start = simbest;
		else if (config->LMstart == "CURR")
			LM_start = p;
		else
			throw HMMPI::Exception("Incorrect OPT_CONFIG.LMstart type in KW_runOptProxy::Run");

		double qk = PMproxy->ObjFunc(LM_start);				// proxy value before optimization
		MPI_Bcast(&qk, 1, MPI_DOUBLE, 0, PMproxy->GetComm());
		std::string optmsg0, optmsg;
		if (Rk == 0)		// not restricted
		{
			p = Opt->RunOptMult(PMproxy, std::vector<std::vector<double>>{LM_start}, &optctx);		// NOTE "starts" array with one element is taken
			optmsg0 = HMMPI::MessageRE(", без ограничений ", ", no restrictions ");
			optmsg = Opt->ReportMsgMult();
		}
		else				// restricted
		{
			if (config->restr == "SPHERE")
			{
				std::string msg0;
				p = Opt->RunOptRestrict(PMproxy, LM_start, Rk, config->delta, &optctx, &optctx_spher, msg0);
				optmsg0 = HMMPI::stringFormatArr(", rk = {0} [сфера] ", ", rk = {0} [sphere] ", Rk);
				optmsg = Opt->ReportMsgMult() + msg0;
			}
			else if (config->restr == "CUBE")
			{
				p = Opt->RunOptRestrictCube(PMproxy, LM_start, Rk, &optctx);
				optmsg0 = HMMPI::stringFormatArr(", rk = {0} [куб] ", ", rk = {0} [cube] ", Rk);
				optmsg = Opt->ReportMsgMult();
			}
			else
				throw HMMPI::Exception("Incorrect restriction type " + config->restr);
		}
		double qk1 = Opt->get_best_of();						// proxy value after optimization
		MPI_Bcast(&qk1, 1, MPI_DOUBLE, 0, PMproxy->GetComm());

		time2 = std::chrono::high_resolution_clock::now();
		K->AppText(HMMPI::stringFormatArr("Оптимизация PROXY" + optmsg0 + "({0:%.3f} сек)\n",
										  "PROXY optimization" + optmsg0 + "({0:%.3f} sec)\n", std::chrono::duration_cast<std::chrono::duration<double>>(time2-time1).count()));
		K->AppText(optmsg);

		// recalculate simulation
		PMproxy->ObjFunc(p);									// first, calculate the proxy o.f. (with breakdown) for reporting
		std::vector<double> mod_data_likelihood = PMproxy->ModelledData();	// PMproxy is posterior, so strip the prior 'modelled data'
		if (mod_data_likelihood.size() > 0)
		{
			assert(mod_data_likelihood.size() >= p.size());
			mod_data_likelihood = std::vector<double>(mod_data_likelihood.begin() + p.size(), mod_data_likelihood.end());
		}
		PMecl->import_stats(mod_data_likelihood, (HMMPI::Mat(p) - HMMPI::Mat(LM_start)).ToVector());	// import the proxy modelled data and the step taken

		double of = PMecl->ObjFunc(p);
		MPI_Bcast(&of, 1, MPI_DOUBLE, 0, PMecl->GetComm());		// sync SIM o.f.
		if (of < of_simbest)
		{
			of_simbest = of;
			simbest = p;
			write_params_log = true;
		}

		double dX = (HMMPI::Mat(p) - HMMPI::Mat(LM_start)).Norm2();		// actual step taken
		double Tk = 0;		// declare the ratio Tk here, for reporting in the end
		if (Rk > 0)			// restricted step case
		{
			Tk = (simk - of)/(qk - qk1);						// Tk is used to control the radius Rk
			if (simk == of)
				Tk = 0;											// for dX == 0

			if (Tk < config->tau1)										// 0.25
			{
				if (request_Rk_decr)									// check if the previous iteration also requested a decrease
					Rk = HMMPI::Min(Rk, dX)/2;							// here take min(Rk, delta_k) to handle situations (x, x) when delta_k may become large

				request_Rk_decr = !request_Rk_decr;
			}
			else
				request_Rk_decr = false;								// forget any decrease request

			//else if (Tk > config->tau2 && Opt->restrict_choice == 2)	// version before 25.02.2018. Found an example where the search was stuck in a local minimum, Tk >> 1, optimization finishes in the starting point (f1), and Rk cannot increase
			if (Tk > config->tau2)										// 0.75
				Rk *= 2;

			if (Rk == 0)						// this happens when delta_k == 0, i.e. proxy optimization is stuck
				finished = 1;

			if (Rk < config->rmin)
				Rk = config->rmin;
		}

		if (config->LMstart == "SIMBEST")		// update simk
			simk = of_simbest;
		else if (config->LMstart == "CURR")
			simk = of;

		time3 = std::chrono::high_resolution_clock::now();
		K->AppText(HMMPI::stringFormatArr("\nРасчет SIM ({0:%.3f} сек)\n",
										  "\nSIM calculation ({0:%.3f} sec)\n", std::chrono::duration_cast<std::chrono::duration<double>>(time3-time2).count()));
		K->AppText(HMMPI::stringFormatArr("Целевая функция (posterior) = {0:%.8g}\n", "Objective function (posterior) = {0:%.8g}\n", of));

		std::string msg1 = smry->get_Data().AddModel(parameters->name, parameters->InternalToExternal(p), parameters->backval, PMecl->get_smry());
		std::string msg2 = smry->Save();
		time4 = std::chrono::high_resolution_clock::now();
		K->AppText(HMMPI::stringFormatArr("Модель SIM добавляется в ECLSMRY  ({0:%.3f} сек)\n",
										  "Adding model SIM to ECLSMRY ({0:%.3f} sec)\n", std::chrono::duration_cast<std::chrono::duration<double>>(time4-time3).count()));
		K->AppText(msg1);
		K->AppText(msg2);

		time(&t1);
		iter++;
		if (difftime(t1, t0)/double(3600) > config->MaxHours || iter >= config->MaxIter)
			finished = 1;

		HMMPI::MPI_BarrierSleepy(MPI_COMM_WORLD);
		RANK0_SYNCERR_BEGIN(MPI_COMM_WORLD);
			FILE *f = fopen(progress_file.c_str(), "a");
			if (f != 0)
			{
				fprintf(f, "%-5d\t%-8.3g\t%-10.5g\t%-11.6g\t%-17.12g\t%-10.5g\t%-2.2s\t%-17.12g\t%-12.6g\t%-12.6g\n",
						int(smry->get_Data().total_models()), difftime(t1, t0)/double(3600), dX, qk1, of, Tk, (request_Rk_decr ? "*" : " "),
							of_simbest, smry->get_Data().Xmin, smry->get_Data().Xavg);
				fclose(f);
			}

			if (write_params_log)
				PMecl->WriteLimits(simbest, params_log_file);
		RANK0_SYNCERR_END(MPI_COMM_WORLD);

		MPI_Bcast(&finished, 1, MPI_INT, 0, MPI_COMM_WORLD);
		delete Opt;
	}

	K->AppText(HMMPI::stringFormatArr("\n>> Наилучшая целевая функция для SIM (posterior) = {0:%.8g}\n",
									  "\n>> Best found SIM objective function (posterior) = {0:%.8g}\n", of_simbest));

	model->R = model_R_cache;		// restore the original radius
	model->type = modtype_cache;	// restore the model type
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_runPopModel::KW_runPopModel()
{
	name = "RUNPOPMODEL";
}
//------------------------------------------------------------------------------------------
void KW_runPopModel::Run()
{
	Start_pre();
	IMPORTKWD(smry, KW_eclsmry, "ECLSMRY");
	Finish_pre();

	smry->get_Data().PopModel();
	K->AppText(HMMPI::MessageRE("Последняя модель из ECLSMRY была удалена. Внимание! Список имен параметров в ECLSMRY остался прежним!\n",
								"The last model from ECLSMRY was removed. Attention! Parameters names list in ECLSMRY has not changed!\n"));
	std::string msg = smry->Save();
	K->AppText(msg);
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_runViewSmry::KW_runViewSmry()
{
	name = "RUNVIEWSMRY";
}
//------------------------------------------------------------------------------------------
void KW_runViewSmry::Run()
{
	Start_pre();
	IMPORTKWD(smry, KW_eclsmry, "ECLSMRY");
	IMPORTKWD(view, KW_viewsmry_config, "VIEWSMRY_CONFIG");
	IMPORTKWD(datesW, KW_dates, "DATES");
	IMPORTKWD(vect, KW_eclvectors, "ECLVECTORS");
	Add_pre("PARAMETERS");
	Finish_pre();

	smry->get_Data().ViewSmry(view->out_file, datesW->dates, vect->vecs, view->order == "DIRECT", K);
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_runSmryPlot::KW_runSmryPlot()
{
	name = "RUNSMRYPLOT";
}
//------------------------------------------------------------------------------------------
void KW_runSmryPlot::Run()
{
	Start_pre();
	IMPORTKWD(cfg, KW_smryplot_config, "SMRYPLOT_CONFIG");
	IMPORTKWD(params, KW_parameters, "PARAMETERS");
	IMPORTKWD(model, KW_model, "MODEL");
	IMPORTKWD(smry, KW_eclsmry, "ECLSMRY");
	Add_pre("DATES");
	Add_pre("ECLVECTORS");
	Add_pre("TEXTSMRY");
	Add_pre("PRIOR");
	Finish_pre();

	std::vector<double> min = params->ExternalToInternal(params->min);
	std::vector<double> max = params->ExternalToInternal(params->max);

	PhysModel *PMproxy = model->MakeModel(this, this->CWD, true, "PROXY");				// make the posterior (will be deleted automatically)
	K->AppText((std::string)HMMPI::MessageRE("Модель: ", "Model: ") + "POSTERIOR(PROXY)\n");
	K->AppText(PMproxy->proc_msg());


	// I. Plot over a 1D range
	FILE *F0 = NULL;
	RANK0_SYNCERR_BEGIN(MPI_COMM_WORLD);
		F0 = fopen(cfg->fname_range.c_str(), "w");
		if (F0 == NULL)
			throw HMMPI::Exception("Невозможно открыть файл для записи " + cfg->fname_range, "Cannot open file for writing " + cfg->fname_range);
	RANK0_SYNCERR_END(MPI_COMM_WORLD);

	try
	{
		if (K->MPI_rank == 0)
			fprintf(F0, "%-12s\t%d\n\n", "Nt", cfg->Nint + 1);
		for (size_t i = 0; i < params->name.size(); i++)
		{
			if (K->MPI_rank == 0)
				fprintf(F0, "%-12s\t%s\n", params->name[i].c_str(), "PROXY");

			const double dx = (max[i] - min[i])/cfg->Nint;		// internal
			for (int j = 0; j <= cfg->Nint; j++)
			{
				std::vector<double> X = params->init;			// internal
				X[i] = min[i] + dx*j;							// adjust the i-th coordinate
				double f = PMproxy->ObjFunc(X);
				double x = params->InternalToExternal(X)[i];	// external, i-th coordinate
				if (K->MPI_rank == 0)
					fprintf(F0, "%-12.7g\t%.7g\n", x, f);
			}
			if (K->MPI_rank == 0)
				fprintf(F0, "\n");
		}
	}
	catch (...)
	{
		if (F0 != NULL)
			fclose(F0);
		throw;
	}
	if (F0 != NULL)
		fclose(F0);


	// II. Plot over design points
	const double model_nug_cache = model->nugget;
	model->nugget = 0;														// zero nugget to get the exact proxy behavior
	PMproxy = model->MakeModel(this, this->CWD, true, "PROXY");				// make the posterior (will be deleted automatically)
	K->AppText((std::string)HMMPI::MessageRE("Модель: ", "Model: ") + "POSTERIOR(PROXY, nugget=0)\n");
	K->AppText(PMproxy->proc_msg());

	F0 = NULL;
	RANK0_SYNCERR_BEGIN(MPI_COMM_WORLD);
		F0 = fopen(cfg->fname_design.c_str(), "w");
		if (F0 == NULL)
			throw HMMPI::Exception("Невозможно открыть файл для записи " + cfg->fname_design, "Cannot open file for writing " + cfg->fname_design);
	RANK0_SYNCERR_END(MPI_COMM_WORLD);

	try
	{
		const HMMPI::SimProxyFile &data = smry->get_Data();
		std::vector<std::vector<double>> points = data.get_internal_parameters(params);		// internal
		HMMPI::Bcast_vector(points, 0, MPI_COMM_WORLD);

		const HMMPI::Mat init(params->init);			// internal
		std::vector<double> f(points.size());			// aux storage arrays
		std::vector<double> dist(points.size());
		for (size_t j = 0; j < points.size(); j++)		// loop over all design points
		{
			f[j] = PMproxy->ObjFunc(points[j]);
			dist[j] = (HMMPI::Mat(points[j]) - init).Norm2();		// distance in internal coords
		}

		if (K->MPI_rank == 0)
			fprintf(F0, "%-12s\t%-12s\t%zu\n\n", "Design", "points", points.size());
		for (size_t i = 0; i < params->name.size(); i++)
		{
			if (K->MPI_rank == 0)
				fprintf(F0, "%-12s\t%-12s\t%s\n", params->name[i].c_str(), "PROXY(n=0)", "|Xi-X0|");

			for (size_t j = 0; j < points.size(); j++)				// loop over all design points
				if (K->MPI_rank == 0)
					fprintf(F0, "%-12.7g\t%-12.7g\t%.7g\n", params->InternalToExternal(points[j])[i], f[j], dist[j]);		// X: external, i-th coordinate
			if (K->MPI_rank == 0)
				fprintf(F0, "\n");
		}
	}
	catch (...)
	{
		if (F0 != NULL)
			fclose(F0);
		throw;
	}
	if (F0 != NULL)
		fclose(F0);

	model->nugget = model_nug_cache;								// restore the nugget
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_runView_tNavSmry::KW_runView_tNavSmry()
{
	name = "RUNVIEW_TNAVSMRY";
}
//------------------------------------------------------------------------------------------
void KW_runView_tNavSmry::Run()
{
	Start_pre();
	IMPORTKWD(dates, KW_dates, "DATES");
	IMPORTKWD(groups, KW_groups, "GROUPS");
	IMPORTKWD(Sdate, KW_startdate, "STARTDATE");
	IMPORTKWD(vect, KW_eclvectors, "ECLVECTORS");
	IMPORTKWD(config, KW_view_tNavsmry_config, "VIEW_TNAVSMRY_CONFIG");
	IMPORTKWD(props, KW_view_tNavsmry_properties, "VIEW_TNAVSMRY_PROPERTIES");
	Finish_pre();

	std::string msg_dat_short, msg_vec_short;
	std::string msg_dat_full, msg_vec_full;			// unused variables
	std::string hdr1, hdr2;
	std::vector<double> fac;

	HMMPI::tNavSMRY smry(groups->sec_obj, Sdate->start);
	smry.ReadFiles(config->model);
	HMMPI::Mat M = smry.ExtractSummary(dates->dates, vect->vecs, msg_dat_short, msg_vec_short, msg_dat_full, msg_vec_full, K->StrListN());

	if (msg_dat_short != "")
	{
		K->AppText((std::string)HMMPI::MessageRE("ПРЕДУПРЕЖДЕНИЕ: ", "WARNING: ") + msg_dat_short + "\n");
		K->TotalWarnings++;
	}
	if (msg_vec_short != "")
	{
		K->AppText((std::string)HMMPI::MessageRE("ПРЕДУПРЕЖДЕНИЕ: ", "WARNING: ") + msg_vec_short + "\n");
		K->TotalWarnings++;
	}

	const int DateWidth = 21;							// 'date' column width
	const int wid = config->width;						// other columns width
	props->make_headers(hdr1, hdr2, fac, DateWidth, wid);

	assert(fac.size() == M.JCount());
	assert(dates->dates.size() == M.ICount());
	M = M % fac;										// scaling by factors

	// saving to file
	FILE *f0 = fopen(config->outfile.c_str(), "w");
	fprintf(f0, "%s\n", hdr1.c_str());
	fprintf(f0, "%s\n", hdr2.c_str());
	for (size_t i = 0; i < dates->dates.size(); i++)
	{
		fprintf(f0, "%-*.*s", DateWidth, DateWidth, dates->dates[i].ToString().c_str());
		for (size_t j = 0; j < M.JCount(); j++)
			fprintf(f0, "\t%-*.*g", wid, wid-5, M(i, j));
		fprintf(f0, "\n");
	}
	fclose(f0);
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_runPlot::KW_runPlot()
{
	name = "RUNPLOT";
}
//------------------------------------------------------------------------------------------
void KW_runPlot::Run()
{
	Start_pre();
	IMPORTKWD(physmodel, KW_physmodel, "PHYSMODEL");
	IMPORTKWD(plotparams, KW_plotparams, "PLOTPARAMS");
	DECLKWD(points, KW_3points, "3POINTS");
	Finish_pre();

	MPI_Comm ofmpi_comm, second_comm;
	if (physmodel->type[0] == "SIMPROXY" || physmodel->type[0] == "DATAPROXY")
		PhysModMPI::HMMPI_Comm_split(1, MPI_COMM_WORLD, &ofmpi_comm, &second_comm);					// ofmpi_comm = 0xxxxx.., second_comm = 0123.... (quasi-MPI_COMM_WORLD)
	else
		PhysModMPI::HMMPI_Comm_split(K->MPI_size, MPI_COMM_WORLD, &ofmpi_comm, &second_comm);		// second_comm = 00000000 (quasi-MPI_COMM_SELF)

	ModelFactory Factory;
	std::string msg0;
	PhysModel *PM = Factory.Make(msg0, K, this, this->CWD, 1, second_comm);

	const ParamsInterface *par_interface = dynamic_cast<const ParamsInterface*>(PM->GetConstr());	// min, max, A/N for parameters are taken from model
	if (par_interface == nullptr)
		throw HMMPI::Exception("PM->BoundConstr cannot be cast to ParamsInterface");

	K->AppText((std::string)HMMPI::MessageRE("Модель: ", "Model: ") + msg0 + "\n" + PM->proc_msg());

	std::vector<double> x0, y0, z0;					// ACTDIM
	bool points_defined = (points->GetState() == "") && (points->x.size() != 0);

	if (points_defined && points->x.size() != par_interface->act.size())
		throw HMMPI::Exception(HMMPI::stringFormatArr("Размерность 3POINTS должна соответствовать полной размерности параметров {0:%zu}",
							   	   	   	   	   	   	  "Dimension of 3POINTS should match the full dimension of parameters {0:%zu}", par_interface->act.size()));
	if (points_defined)
	{
		x0 = PM->act_par(points->x_internal());
		y0 = PM->act_par(points->y_internal());
		z0 = PM->act_par(points->z_internal());
	}

	if (plotparams->Nx < 2 || plotparams->Ny < 2)
		throw HMMPI::Exception("Nx, Ny в PLOTPARAMS д.б. >= 2", "Nx, Ny in PLOTPARAMS should be >= 2");
	if (!points_defined && par_interface->get_act_ind().size() != 2)
		throw HMMPI::Exception("Если 3POINTS не задано, RUNPLOT работает только для 2 активных параметров в модели",
							   "When 3POINTS are not defined, RUNPLOT only works for 2 active parameters in model");
	if (!points_defined)
		K->AppText(HMMPI::MessageRE("3POINTS не задано -> плоскость графика задается двумя активными параметрами\n",
				   	   	   	   	    "3POINTS are not specified -> plotting over the plane defined by the two active parameters\n"));
	else
		K->AppText(std::string(HMMPI::MessageRE("3POINTS задано -> плоскость графика задается тремя точками",
				   	   	   	   	    			"3POINTS are specified -> plotting over plane defined by them")) + points->msg + "\n");

	int len = plotparams->Nx * plotparams->Ny;
	int dim = 2;
	if (points_defined)
		dim = x0.size();

	double X0 = 0, Y0 = 0, DX = 0, DY = 0;

	std::chrono::high_resolution_clock::time_point time1 = std::chrono::high_resolution_clock::now(), time2;
	double **POP = 0;
	double *FIT = 0;

    // MPI
	if (K->MPI_rank == 0)	// master branch: prepare POP (parameters for each model in population)
	{
		POP = new double*[len];
		FIT = new double[len];
		const std::vector<double> min = par_interface->fullmin();
		const std::vector<double> max = par_interface->fullmax();
		if (!points_defined)
		{
			int ind_act1 = par_interface->get_act_ind()[0];
			int ind_act2 = par_interface->get_act_ind()[1];
			X0 = min[ind_act1];
			Y0 = min[ind_act2];
			DX = (max[ind_act1] - X0)/(plotparams->Nx - 1);
			DY = (max[ind_act2] - Y0)/(plotparams->Ny - 1);

			for (int i = 0; i < len; i++)
			{
				POP[i] = new double[dim];

				int x = i % plotparams->Nx;
				int y = i / plotparams->Nx;
				POP[i][0] = X0 + DX*x;
				POP[i][1] = Y0 + DY*y;
			}
		}
		else	// points_defined
		{
			std::vector<double> v1 = HMMPI::Vec_x_ay(y0, x0, -1);
			std::vector<double> v2 = HMMPI::Vec_x_ay(z0, x0, -1);
			double normv1_2 = HMMPI::InnerProd(v1, v1);
			double scalv1v2 = HMMPI::InnerProd(v1, v2);
			double normv1 = sqrt(normv1_2);

			std::vector<double> v2ort =  HMMPI::Vec_x_ay(v2, v1, -scalv1v2/normv1_2);
			double normv2ort = sqrt(HMMPI::InnerProd(v2ort, v2ort));

			std::vector<double> u1 = HMMPI::Vec_ax(v1, 1/normv1);
			std::vector<double> u2 = HMMPI::Vec_ax(v2ort, 1/normv2ort);

			double minX = 0, maxX = normv1;
			double minY = 0, maxY = normv2ort;
			double x3x = scalv1v2/normv1;
			if (x3x < minX)
				minX = x3x;
			if (x3x > maxX)
				maxX = x3x;

			X0 = minX - plotparams->delta;
			Y0 = minY - plotparams->delta;
			DX = (maxX - minX + 2*plotparams->delta)/(plotparams->Nx - 1);
			DY = (maxY - minY + 2*plotparams->delta)/(plotparams->Ny - 1);

			for (int i = 0; i < len; i++)
			{
				POP[i] = new double[dim];

				int x = i % plotparams->Nx;
				int y = i / plotparams->Nx;
				std::vector<double> z = HMMPI::Vec_x_ay(x0, u1, DX*x + X0);
				z = HMMPI::Vec_x_ay(z, u2, DY*y + Y0);

				for (int j = 0; j < dim; j++)
					POP[i][j] = z[j];
			}
		}	// points_defined
	}

	// all ranks: run simulations
	PhysModMPI PM_MPI(ofmpi_comm, PM);
	PM_MPI.ObjFuncMPI_ACT(len, POP, FIT, true);

	if (K->MPI_rank == 0)	// master branch: finalizing stuff
	{
#ifdef WRITE_PLOT_DATA
		FILE *f0 = fopen("ObjFuncPlot_dump.txt", "w");
		if (f0 != NULL)
		{
			for (int i = 0; i < len; i++)
			{
				fprintf(f0, "%20.16g\t", FIT[i]);
				HMMPI::SaveASCII(f0, POP[i], dim, "%20.16g");
			}
			fclose(f0);
		}
#endif

		Grid2D res(FIT, plotparams->Nx, plotparams->Ny, DX, DY, X0-DX/2, Y0-DY/2, "0.1E+31");
		for (int i = 0; i < len; i++)
			delete [] POP[i];

		delete [] POP;
		delete [] FIT;

		res.SaveToFile(HMMPI::getFullPath(this->CWD, "ObjFuncPlot.txt"));
		time2 = std::chrono::high_resolution_clock::now();
		K->AppText(HMMPI::stringFormatArr("Время {0:%.3f} сек.\n\n", "CPU {0:%.3f} sec.\n\n", std::chrono::duration_cast<std::chrono::duration<double>>(time2-time1).count()));
	}

	if (ofmpi_comm != MPI_COMM_NULL)
		MPI_Comm_free(&ofmpi_comm);
	if (second_comm != MPI_COMM_NULL)
		MPI_Comm_free(&second_comm);
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_runOpt::KW_runOpt()
{
	name = "RUNOPT";
}
//------------------------------------------------------------------------------------------
void KW_runOpt::Run()
{
	CWD_holder::N = this->CWD;				// only used by CMA-ES
	const std::string limits_log_file = HMMPI::getFullPath(this->CWD, "LimitsLog.txt");			// resulting solution vector is saved here

	Start_pre();
	IMPORTKWD(RML, KW_RML, "RML");
	IMPORTKWD(opt, KW_optimization, "OPTIMIZATION");
	Finish_pre();

	K->AppText((std::string)HMMPI::MessageRE("Алгоритм: ", "Optimizer: ") + opt->algorithm + "\n");
	if (RML->on == "ON")
	{
		K->AppText("Randomized maximum likelihood: ON\n");
		int seed = RML->seed;
		if (seed == 0)
			seed = time(NULL);
		MPI_Bcast(&seed, 1, MPI_INT, 0, MPI_COMM_WORLD);

		srand(seed);
		K->AppText(HMMPI::stringFormatArr("RML seed = {0:%d}\n", std::vector<int>{seed}));
	}
	else
		K->AppText("Randomized maximum likelihood: OFF\n");

	ModelFactory Factory;
	std::string msg0;
	PhysModel *PM = Factory.Make(msg0, K, this, this->CWD, 1);

	const ParamsInterface *par_interface = dynamic_cast<const ParamsInterface*>(PM->GetConstr());
	if (par_interface == nullptr)
		throw HMMPI::Exception("PM->BoundConstr cannot be cast to ParamsInterface");

	K->AppText((std::string)HMMPI::MessageRE("Модель: ", "Model: ") + msg0 + "\n" + PM->proc_msg());

	if (RML->on == "ON")
		PM->PerturbData();

	Optimizer *Opt = Optimizer::Make(opt->algorithm);
	const OptContext *optctx = opt->MakeContext();

	std::chrono::high_resolution_clock::time_point time1 = std::chrono::high_resolution_clock::now(), time2;
	std::vector<double> x_opt;
	std::string optmsg;
	if (opt->R > 0)			// optimization with restricted step
	{
		if (opt->restr == "SPHERE")
		{
			K->AppText(HMMPI::MessageRE("Оптимизация с ограничением шага (сфера)\n", "Optimization with restricted step (sphere)\n"));
			std::string msg0;
			x_opt = Opt->RunOptRestrict(PM, par_interface->init, opt->R, 1e-5, optctx, optctx, msg0);
			optmsg = Opt->ReportMsgMult() + msg0;
		}
		else if (opt->restr == "CUBE")
		{
			K->AppText(HMMPI::MessageRE("Оптимизация с ограничением шага (куб)\n", "Optimization with restricted step (cube)\n"));
			x_opt = Opt->RunOptRestrictCube(PM, par_interface->init, opt->R, optctx);
			optmsg = Opt->ReportMsgMult();
		}
		else
		{
			delete Opt;
			throw HMMPI::Exception("Incorrect restriction type " + opt->restr);
		}
	}
	else					// usual optimization
	{
		x_opt = Opt->RunOpt(PM, par_interface->init, optctx);
		optmsg = Opt->ReportMsg();
	}

	if (K->MPI_rank == 0)
	{
		PM->WriteLimits(x_opt, limits_log_file);

		time2 = std::chrono::high_resolution_clock::now();
		K->AppText(HMMPI::stringFormatArr("Время {0:%.3f} сек.\n", "CPU {0:%.3f} sec.\n",
				std::chrono::duration_cast<std::chrono::duration<double>>(time2-time1).count()));
		K->AppText(optmsg);
	}

	delete Opt;
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_runCritGrad::KW_runCritGrad()
{
	name = "RUNCRITGRAD";
}
//------------------------------------------------------------------------------------------
void KW_runCritGrad::Run()
{
	const std::string limits_log_file = HMMPI::getFullPath(this->CWD, "LimitsLog.txt");			// resulting solution vector is saved here

	Start_pre();
	IMPORTKWD(opt, KW_optimization, "OPTIMIZATION");
	Finish_pre();

	NonlinearSystemSolver *sol = opt->MakeNonlinSolver();
	K->AppText((std::string)HMMPI::MessageRE("Солвер: ", "Solver: ") + opt->nonlin_solver + "\n");

	ModelFactory Factory;
	std::string msg0;
	PhysModel *PM = Factory.Make(msg0, K, this, this->CWD, 1);
	K->AppText((std::string)HMMPI::MessageRE("Модель: ", "Model: ") + msg0 + "\n" + PM->proc_msg());

	const ParamsInterface *par_interface = dynamic_cast<const ParamsInterface*>(PM->GetConstr());
	if (par_interface == nullptr)
		throw HMMPI::Exception("PM->BoundConstr cannot be cast to ParamsInterface");

	VM_gradient VM(PM);
	sol->SetFuncFromVM(&VM);		// for sol->Solve

	std::chrono::high_resolution_clock::time_point time1 = std::chrono::high_resolution_clock::now(), time2;
	std::vector<double> x_opt = sol->Solve(par_interface->get_init_act());
	sol->SetFuncFromVM(&VM);		// for sol->msg()
	K->AppText(sol->msg(x_opt));

	if (K->MPI_rank == 0)
	{
		PM->WriteLimits_ACT(x_opt, limits_log_file);

		time2 = std::chrono::high_resolution_clock::now();
		K->AppText(HMMPI::stringFormatArr("Время {0:%.3f} сек.\n", "CPU {0:%.3f} sec.\n",
				std::chrono::duration_cast<std::chrono::duration<double>>(time2-time1).count()));
	}
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_runGrad::KW_runGrad()
{
	name = "RUNGRAD";
}
//------------------------------------------------------------------------------------------
void KW_runGrad::Run()
{
	std::chrono::high_resolution_clock::time_point time1 = std::chrono::high_resolution_clock::now(), time2;
	ModelFactory Factory;
	std::string msg0;
	PhysModel *PM = Factory.Make(msg0, K, this, this->CWD, 1);

	const ParamsInterface *par_interface = dynamic_cast<const ParamsInterface*>(PM->GetConstr());
	if (par_interface == nullptr)
		throw HMMPI::Exception("PM->BoundConstr cannot be cast to ParamsInterface");

	K->AppText((std::string)HMMPI::MessageRE("Модель: ", "Model: ") + msg0 + "\n" + PM->proc_msg());

	std::vector<double> p = par_interface->get_init_act();
	if (!PM->CheckLimits_ACT(p))
	{
		K->AppText("WARNING: " + PM->get_limits_msg());
		K->TotalWarnings++;
	}

	double f = PM->ObjFunc_ACT(p);										// objective function value
	K->AppText(PM->ObjFuncMsg());

	std::vector<double> grad = PM->ObjFuncGrad_ACT(p);					// objective function gradient
	const HMMPI::Mat Sens = PM->DataSens_act();

	std::vector<double> aux(PM->ParamsDim_ACT(), 1);					// "direction" vector (ones)
	double graddir1 = PM->ObjFuncGradDir_ACT(p, aux);					// objective function gradient along direction "aux"
	double graddir2 = 0;
	if (K->MPI_rank == 0)
		graddir2 = HMMPI::InnerProd(grad, aux);

	HMMPI::Mat Hess, FI, FImix;					// objective function Hessian, Fisher Information for L = exp(-1/2*f), FI-Hessian mix; only works for some models
	std::vector<HMMPI::Mat> di_FI(p.size());	// dFI/dx_i
	try
	{
		Hess = PM->ObjFuncHess_ACT(p);			// models which cannot calculate ObjFuncHess, ObjFuncFisher, produce an exception, for them Hess, FI = empty

		const bool YES_FI = true, YES_FI_MIX = true, YES_FI_DXI = false;		// switch as appropriate
		if (YES_FI)
		{
			FI = PM->ObjFuncFisher_ACT(p);
		}
		if (YES_FI_MIX)
		{
			FImix = PM->ObjFuncFisher_mix_ACT(p);
		}
		if (YES_FI_DXI)
		{
			for (size_t i = 0; i < p.size(); i++)
				di_FI[i] = PM->ObjFuncFisher_dxi_ACT(p, i);
		}
	}
	catch (...)
	{
	}

	if (K->MPI_rank == 0)	// writing the file with results
	{
		std::ofstream fileS;
		fileS.open(HMMPI::getFullPath(K->InitCWD, "ObjFuncSens.txt"));
		fileS << HMMPI::stringFormatArr("f (objective function)\n{0:%-18.16g}\n", std::vector<double>{f});
		fileS << HMMPI::stringFormatArr("\nones'*grad(f) via ObjFuncGradDir\n{0:%-18.16g}\n", std::vector<double>{graddir1});
		fileS << HMMPI::stringFormatArr("\nones'*grad(f) via inner product\n{0:%-18.16g}\n", std::vector<double>{graddir2});

		fileS << "\ngrad(f) (objective function gradient WRT inner variables)" << std::endl;
		fileS << HMMPI::ToString(grad, "%-20.16g");

		fileS << "\nSens (sensitivity matrix for modelled data WRT active inner variables)" << std::endl;
		fileS << Sens.ToString("%-20.16g");

		if (Hess.ICount() > 0)
		{
			fileS << "\nHessian of objective function (WRT inner variables)" << std::endl;
			fileS << Hess.ToString("%-20.16g");
		}

		if (FI.ICount() > 0)
		{
			fileS << "\nFisher Information matrix for L = exp(-1/2*f) (WRT inner variables)" << std::endl;
			fileS << FI.ToString("%-20.16g");
		}

		if (FImix.ICount() > 0)
		{
			fileS << "\nFI-Hessian mix [2*J'*FI*J + grad*T] (WRT inner variables)" << std::endl;
			fileS << FImix.ToString("%-20.16g");
		}

		for (size_t i = 0; i < p.size(); i++)
			if (di_FI[i].ICount() > 0)
			{
				fileS << "\ndFI/dx_" << i << std::endl;
				fileS << di_FI[i].ToString("%-20.16g");
			}

		fileS.close();
	}

	time2 = std::chrono::high_resolution_clock::now();
	K->AppText(HMMPI::stringFormatArr("Время {0:%.3f} сек.\n\n", "CPU {0:%.3f} sec.\n\n",
			std::chrono::duration_cast<std::chrono::duration<double>>(time2-time1).count()));
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_runJac::KW_runJac()
{
	name = "RUNJAC";
}
//------------------------------------------------------------------------------------------
void KW_runJac::Run()
{
	Start_pre();
	IMPORTKWD(vectmod, KW_vectmodel, "VECTMODEL");
	Finish_pre();

	std::chrono::high_resolution_clock::time_point time1 = std::chrono::high_resolution_clock::now(), time2;
	ModelFactory Factory;
	std::string msg0;
	PhysModel *PM = Factory.Make(msg0, K, this, this->CWD, 1);

	const ParamsInterface *par_interface = dynamic_cast<const ParamsInterface*>(PM->GetConstr());
	if (par_interface == nullptr)
		throw HMMPI::Exception("PM->BoundConstr cannot be cast to ParamsInterface");

	std::vector<double> p = par_interface->get_init_act();
	if (!PM->CheckLimits_ACT(p))
	{
		K->AppText("WARNING: " + PM->get_limits_msg());
		K->TotalWarnings++;
	}

	VectorModel *VM = vectmod->Make(PM);
	K->AppText("Model: " + VM->name + " from " + msg0 + "\n" + PM->proc_msg());

	if (dynamic_cast<VM_Ham_eq2_eps*>(VM) != nullptr)
		p = dynamic_cast<VM_Ham_eq2_eps*>(VM)->map_xfull_x(p);

	std::vector<double> func = VM->Func_ACT(p);			// analytical calculation
	HMMPI::Mat jac = VM->Jac_ACT(p);

	double eps = std::numeric_limits<double>::quiet_NaN();
	if (dynamic_cast<VM_Ham_eq2*>(VM) != nullptr)
		eps = dynamic_cast<VM_Ham_eq2*>(VM)->eps;

	HMMPI::Mat jacnum;									// numerical Jacobian, fixed dh is used, with central finite differences (OH2)
	const double dh = 1e-5;

	for (size_t i = 0; i < p.size(); i++)
	{
		p[i] += dh;
		HMMPI::Mat v = VM->Func_ACT(p);
		p[i] -= 2*dh;
		v = (0.5/dh)*(std::move(v) - VM->Func_ACT(p));
		p[i] += dh;

		v.Reshape(1, p.size());
		jacnum = std::move(jacnum) || v;
	}
	jacnum = jacnum.Tr();

	if (K->MPI_rank == 0)	// writing the file with results
	{
		std::ofstream fileS;
		fileS.open(HMMPI::getFullPath(K->InitCWD, "ObjFuncJacobian.txt"));
		fileS << "eps = " << eps << "\n";

		fileS << "Vector function\n";
		fileS << HMMPI::ToString(func, "%16.11g");

		fileS << "\nJacobian (analytical)\n";
		fileS << jac.ToString("%-20.16g");

		fileS << "\nJacobian (numerical), central differences with dh = " << dh << "\n";
		fileS << jacnum.ToString("%-20.16g");

		fileS.close();
	}

	time2 = std::chrono::high_resolution_clock::now();
	K->AppText(HMMPI::stringFormatArr("Время {0:%.3f} сек.\n\n", "CPU {0:%.3f} sec.\n\n",
			std::chrono::duration_cast<std::chrono::duration<double>>(time2-time1).count()));
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_runcalccovar::KW_runcalccovar()
{
	name = "RUNCALCCOVAR";
}
//------------------------------------------------------------------------------------------
void KW_runcalccovar::Run()
{
	Start_pre();
	IMPORTKWD(DIMS, KW_griddims, "GRIDDIMS");
	IMPORTKWD(wrcovar, KW_wrcovar, "WRCOVAR");
	IMPORTKWD(refmap, KW_refmap, "REFMAP");
	IMPORTKWD(undef, KW_undef, "UNDEF");
	Finish_pre();

	std::chrono::high_resolution_clock::time_point time1 = std::chrono::high_resolution_clock::now(), time2;
	if ((int)refmap->data.size() != DIMS->Nx * DIMS->Ny * DIMS->Nz)
		throw HMMPI::Exception("(eng) REFMAP Ð¸ GRIDDIMS",
							   "Inconsistent dimensions of REFMAP and GRIDDIMS");
	int M = wrcovar->M, Nx = DIMS->Nx, Ny = DIMS->Ny;
	if (2*M+1 > Nx || 2*M+1 > Ny)
		throw HMMPI::Exception("2M+1 (eng) GRIDDIMS",
							   "2M+1 (cf. WRCOVAR) should not exceed Nx, Ny from GRIDDIMS");

	Grid2D input = refmap->GetGrid2D(DIMS->Nx, DIMS->Ny);

	Grid2D res;
	res.InitData(Nx, Ny);
	res.SetGeom(-0.5, -0.5, 1, 1);
	res.SetVal(0);
	Grid2D count = res;

	for (int x = 0; x < Nx; x++)			// loop through input grid
		for (int y = 0; y < Ny; y++)
		{
			if (input.flag[x][y])
			{
				for (int i = 0; i < 2*M+1; i++)			// loop through 2D window
					for (int j = 0; j < 2*M+1; j++)
					{
						int oX = x + i-M;				// offset coordinates
						int oY = y + j-M;
						if (oX >= 0 && oX < Nx && oY >= 0 && oY < Ny && input.flag[oX][oY])
						{
							res.data[i][j] += input.data[x][y] * input.data[oX][oY];
							count.data[i][j] += 1;
						}
					}
			}
		}

	for (int i = 0; i < 2*M+1; i++)			// loop through 2D window -> find covariance
		for (int j = 0; j < 2*M+1; j++)
		{
			if (count.data[i][j] > 0)
				res.data[i][j] /= count.data[i][j];
			else
			{
				res.data[i][j] = std::numeric_limits<double>::quiet_NaN();
				res.flag[i][j] = 0;
			}
		}

	std::string fn_cov = HMMPI::getFullPath(CWD, wrcovar->cov_file);
	std::string fn_count = HMMPI::getFullPath(CWD, wrcovar->count_file);
	res.SaveProp3D(fn_cov, "COVARIANCE", undef->Ugrid, DIMS->Nz);
	count.SaveProp3D(fn_count, "COV_COUNT", undef->Ugrid, DIMS->Nz);
	K->AppText(std::string(HMMPI::MessageRE("Сохранены файлы:\n", "Saved files:\n")) + fn_cov + "\n" + fn_count + "\n");

	time2 = std::chrono::high_resolution_clock::now();
	K->AppText(HMMPI::stringFormatArr("Время {0:%.3f} сек.\n\n", "CPU {0:%.3f} sec.\n\n",
			std::chrono::duration_cast<std::chrono::duration<double>>(time2-time1).count()));
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_runcalcwellcovar::KW_runcalcwellcovar()
{
	name = "RUNCALCWELLCOVAR";
}
//------------------------------------------------------------------------------------------
void KW_runcalcwellcovar::Run()
{
	Start_pre();
	IMPORTKWD(wrcovar, KW_wrcovar, "WRCOVAR");
	IMPORTKWD(textsmry, KW_textsmry, "TEXTSMRY");
	IMPORTKWD(datesW, KW_dates, "DATES");
	Add_pre("ECLVECTORS");
	Finish_pre();

	std::chrono::high_resolution_clock::time_point time1 = std::chrono::high_resolution_clock::now(), time2;

	size_t Nvect = textsmry->data.JCount()/2;
	size_t Ndates = textsmry->data.ICount();

	// find average time step
	std::vector<double> dates = datesW->zeroBased();
	if (dates.size() != Ndates)
		throw HMMPI::Exception("(eng) KW_runcalcwellcovar::Run", "Inconsistent sizes of dates arrays from TEXTSMRY and DATES in KW_runcalcwellcovar::Run");

	double hav = 0;
	for (size_t i = 1; i < Ndates; i++)
		hav += dates[i] - dates[i-1];
	hav /= (Ndates - 1);			// step with which covariance C will be defined

	int LastInt = int((dates[Ndates-1])/hav + 0.5);
	size_t Nint = LastInt + 1;		// number of intervals/points where C will be defined

	HMMPI::Vector2<double> res(Nint, Nvect);
	HMMPI::Vector2<double> count(Nint, Nvect);
	for (size_t i = 0; i < Nint; i++)
		for (size_t j = 0; j < Nvect; j++)
		{
			res(i, j) = 0;
			count(i, j) = 0;
		}

	// go through the data -- accumulate statistics
	for (size_t v = 0; v < Nvect; v++)
		for (size_t i = 0; i < Ndates; i++)
			for (size_t j = 0; j <= i; j++)
			{
				double delta = dates[i] - dates[j];
				int ind = int(delta/hav + 0.5);
				if (!HMMPI::IsNaN(textsmry->data(i, v)) && !HMMPI::IsNaN(textsmry->data(j, v)) && textsmry->data(i, Nvect+v) > 0 && textsmry->data(j, Nvect+v) > 0)
				{
					res(ind, v) += textsmry->data(i, v) * textsmry->data(j, v);
					count(ind, v) += 1;
				}
			}

	for (size_t i = 0; i < Nint; i++)
		for (size_t j = 0; j < Nvect; j++)
		{
			if (count(i, j) > 0)
				res(i, j) /= count(i, j);
			else
				res(i, j) = std::numeric_limits<double>::quiet_NaN();
		}

	// writing the files

	std::string fn_cov = HMMPI::getFullPath(CWD, wrcovar->cov_file);
	std::string fn_count = HMMPI::getFullPath(CWD, wrcovar->count_file);

	std::ofstream sw_cov, sw_count;
	sw_cov.exceptions(std::ios_base::badbit | std::ios_base::failbit);
	sw_count.exceptions(std::ios_base::badbit | std::ios_base::failbit);

	try
	{
		sw_cov.open(fn_cov);
		sw_count.open(fn_count);

		for (size_t i = 0; i < Nint; i++)
		{
			sw_cov << HMMPI::stringFormatArr("{0}", std::vector<double>{hav * i});
			sw_count << HMMPI::stringFormatArr("{0}", std::vector<double>{hav * i});
			for (size_t v = 0; v < Nvect; v++)
			{
				sw_cov << HMMPI::stringFormatArr("\t{0}", std::vector<double>{res(i, v)});
				sw_count << HMMPI::stringFormatArr("\t{0}", std::vector<double>{count(i, v)});
			}
			sw_cov << "\n";
			sw_count << "\n";
		}

        sw_cov.close();
        sw_count.close();
        K->AppText(std::string(HMMPI::MessageRE("(eng):\n", "Saved files:\n")) + fn_cov + "\n" + fn_count + "\n");
	}
	catch (...)
	{
		if (sw_cov.is_open())
			sw_cov.close();
		if (sw_count.is_open())
			sw_count.close();
		throw;
	}

	time2 = std::chrono::high_resolution_clock::now();
	K->AppText(HMMPI::stringFormatArr("(eng) {0:%.3f} \n\n", "CPU {0:%.3f} sec.\n\n",
			std::chrono::duration_cast<std::chrono::duration<double>>(time2-time1).count()));
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_runmpicheck::KW_runmpicheck()
{
	name = "RUNMPICHECK";
}
//------------------------------------------------------------------------------------------
void KW_runmpicheck::Run()
{
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
	// TODO find some alternative - for RUNMPICHECK
#else
	int msize = K->MPI_size;
	int rank = K->MPI_rank;

	// make host message
	const int BUFFSIZE = 500;
	const int HSTSIZE = 100;
	char buff[BUFFSIZE], hst[HSTSIZE];

	std::string cat = HMMPI::stringFormatArr(HMMPI::MessageRE("Процесс {0:%2d}:{1:%d}\t", "Process {0:%2d}:{1:%d}\t"),
				 std::vector<int>{rank, msize});

	gethostname(hst, HSTSIZE);
	cat += hst;

	hostent *he = gethostbyname(hst);
	for (int i = 0; he->h_addr_list[i] != NULL; i++)
		cat += (std::string)"\t" + inet_ntoa(*((in_addr*)he->h_addr_list[i]));

	sprintf(buff, "%s", cat.c_str());

	// MPI exchange
	char *RCVBUFF = 0;
	if (rank == 0)
		RCVBUFF = new char[BUFFSIZE*msize];

	MPI_Gather(buff, BUFFSIZE, MPI_CHAR, RCVBUFF, BUFFSIZE, MPI_CHAR, 0, MPI_COMM_WORLD);

	// print the results
	if (rank == 0)
	{
		for (int i = 0; i < msize; i++)
			K->AppText((std::string)(RCVBUFF + i*BUFFSIZE) + "\n");
		delete [] RCVBUFF;
	}
#endif
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_runNNCfromgrid::KW_runNNCfromgrid()
{
	name = "RUNNNCFROMGRID";
}
//------------------------------------------------------------------------------------------
void KW_runNNCfromgrid::Run()
{
	Start_pre();
	IMPORTKWD(cz, KW_CoordZcorn, "COORDZCORN");
	Add_pre("GRIDDIMENS");
	Finish_pre();

	const std::string fout_name = "Output_NNC_from_grid.txt";
	std::string msg;
	std::vector<std::vector<HMMPI::NNC>> res = cz->CG.get_same_layer_NNC(msg);		// the result is significant on comm-rank-0
	K->AppText(msg + "\n");

	if (K->MPI_rank == 0)
	{
		FILE *f = fopen(fout_name.c_str(), "w");
		for (size_t i = 0; i < res.size(); i++)
		{
			for (size_t j = 0; j < res[i].size(); j++)
			{
				fprintf(f, "%6d\t%6d\t%6d\t%9d\t%6d\t%6d\tNNC%zu\n",
						res[i][j].N0.i + 1, res[i][j].N0.j + 1, res[i][j].N0.k + 1,
						res[i][j].N1.i + 1, res[i][j].N1.j + 1, res[i][j].N1.k + 1, i);
			}
			fprintf(f, "\n");
		}
		fclose(f);
	}

	K->AppText("NNCs are saved to '" + fout_name + "'\n");
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_runPinchMarkFromGrid::KW_runPinchMarkFromGrid()
{
	name = "RUNPINCHMARKFROMGRID";
}
//------------------------------------------------------------------------------------------
void KW_runPinchMarkFromGrid::Run()
{
	Start_pre();
	IMPORTKWD(cz, KW_CoordZcorn, "COORDZCORN");
	Add_pre("GRIDDIMENS");
	Add_pre("ACTNUM");
	Finish_pre();

	const std::string fout_name = "Output_PINCH_MARK_from_grid.txt";
	assert(!cz->CG.IsCellCoordFilled());
	K->AppText(cz->CG.fill_cell_coord() + "\n");

	std::vector<double> marks = cz->CG.MarkPinchBottoms();		// the result is significant on comm-rank-0
	if (K->MPI_rank == 0)
		cz->CG.SavePropertyToFile(fout_name, "ARRPINCHMARK", marks);
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_runGridIJK_to_XYZ::KW_runGridIJK_to_XYZ()
{
	name = "RUNGRIDIJK_TO_XYZ";
}
//------------------------------------------------------------------------------------------
void KW_runGridIJK_to_XYZ::Run()
{
	Start_pre();
	IMPORTKWD(cz, KW_CoordZcorn, "COORDZCORN");
	IMPORTKWD(pts, KW_3points, "3POINTS");
	Add_pre("GRIDDIMENS");
	Finish_pre();

	if (!cz->CG.IsCellCoordFilled())
		K->AppText(cz->CG.fill_cell_coord() + "\n");

	if (K->MPI_rank == 0)
	{
		for (size_t i = 0; i < pts->x.size(); i++)
		{
			double x, y, z;
			cz->CG.xyz_from_cell_ijk(pts->x[i]-1, pts->y[i]-1, pts->z[i]-1, x, y, z);
			char msg[HMMPI::BUFFSIZE];
			sprintf(msg, "%-3d\t%-3d\t%-3d\t->\t%-7.0f\t%-7.0f\t%-7.2f\n", (int)pts->x[i], (int)pts->y[i], (int)pts->z[i], x, y, z);
			K->AppText(msg);
		}
	}
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_runXYZ_to_GridIJK::KW_runXYZ_to_GridIJK()
{
	name = "RUNXYZ_TO_GRIDIJK";
}
//------------------------------------------------------------------------------------------
void KW_runXYZ_to_GridIJK::Run()
{
	Start_pre();
	IMPORTKWD(cz, KW_CoordZcorn, "COORDZCORN");
	IMPORTKWD(pts, KW_3points, "3POINTS");
	Add_pre("GRIDDIMENS");
	Finish_pre();

	if (!cz->CG.IsCellCoordFilled())
		K->AppText(cz->CG.fill_cell_coord() + "\n");

	for (size_t n = 0; n < pts->x.size(); n++)
	{
		int i = -1, j = -1, k = -1;
		try
		{
			cz->CG.find_cell(pts->x[n], pts->y[n], pts->z[n], i, j, k);
			char msg[HMMPI::BUFFSIZE];
			sprintf(msg, "%-7.0f\t%-7.0f\t%-7.2f\t->\t%-3d\t%-3d\t%-3d\n", pts->x[n], pts->y[n], pts->z[n], i+1, j+1, k+1);
			K->AppText(msg);
		}
		catch (const HMMPI::Exception &e)
		{
			SilentError(e.what());
		}
	}
	K->AppText(cz->CG.report_find_cell_stats() + "\n");
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
HMMPI::Vector2<int> KW_runKriging::get_points() const		// [call on all ranks] get (i,j,k) of the points defined in MAT, returns Np*3 matrix (sync)
{
	DECLKWD(cz, KW_CoordZcorn, "COORDZCORN");
	DECLKWD(mat, KW_mat, "MAT");

	if (mat->M.ICount() < 1)
		throw HMMPI::Exception("Ожидается 1 или более строк в MAT", "MAT should have 1 row or more");
	if (mat->M.JCount() < 4)
		throw HMMPI::Exception("Ожидается 4 или более столбцов в MAT", "MAT should have 4 columns or more");

	assert(cz->CG.IsCellCoordFilled());
	HMMPI::Vector2<int> res(mat->M.ICount(), 3);
	for (size_t r = 0; r < res.ICount(); r++)
	{
		int i, j, k;
		cz->CG.find_cell(mat->M(r,0), mat->M(r,1), mat->M(r,2), i, j, k);

		MPI_Bcast(&i, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&j, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);

		res(r,0) = i;
		res(r,1) = j;
		res(r,2) = k;
	}

	return res;
}
//------------------------------------------------------------------------------------------
HMMPI::Mat KW_runKriging::get_krig_mat(const HMMPI::Vector2<int> &pts, const HMMPI::Func1D_corr *corr) const	// [call on RANK-0] get the kriging matrix
{
	DECLKWD(cz, KW_CoordZcorn, "COORDZCORN");
	DECLKWD(var, KW_variogram_3D, "VARIOGRAM_3D");
	assert(cz->CG.IsCellCoordFilled());

	const int Np = pts.ICount();
	HMMPI::Mat C(Np+1, Np+1, 0.0);			// matrix for ordinary kriging

	const double chirad = var->chi/180*pi;
	const double cosx = cos(chirad);
	const double sinx = sin(chirad);
	for (int i = 0; i < Np; i++)
	{
		C(i, i) = 1.0;			// diagonal
		for (int j = i+1; j < Np; j++)
		{
			double dist = cz->CG.calc_scaled_dist(pts(i,0), pts(i,1), pts(i,2),
									   	   	      pts(j,0), pts(j,1), pts(j,2), var->R, var->r, var->rz, cosx, sinx);
			C(i, j) = corr->f(dist);
			C(j, i) = C(i, j);	// symmetric fill
		}

		C(Np, i) = 1.0;			// trend part
		C(i, Np) = 1.0;
	}

	return C;
}
//------------------------------------------------------------------------------------------
HMMPI::Mat KW_runKriging::get_krig_Ys() const				// [call on RANK-0] get the kriging RHS (values in the design points)
{
	DECLKWD(mat, KW_mat, "MAT");							// columns: x, y, z, v1, v2, .... - design points for kriging

	assert(mat->M.ICount() >= 1);
	assert(mat->M.JCount() >= 4);

	HMMPI::Mat res(mat->M.ICount() + 1, mat->M.JCount() - 3, 0.0);
	for (size_t i = 0; i < mat->M.ICount(); i++)
		for (size_t j = 0; j < res.JCount(); j++)
			res(i, j) = mat->M(i, j+3);

	return res;
}
//------------------------------------------------------------------------------------------
KW_runKriging::KW_runKriging() : pi(acos(-1.0))
{
	name = "RUNKRIGING";
}
//------------------------------------------------------------------------------------------
void KW_runKriging::Run()
{
	Start_pre();
	IMPORTKWD(cz, KW_CoordZcorn, "COORDZCORN");
	IMPORTKWD(mat, KW_mat, "MAT");
	IMPORTKWD(var, KW_variogram_3D, "VARIOGRAM_3D");
	IMPORTKWD(props, KW_krigprops, "KRIGPROPS");
	Add_pre("ACTNUM");					// CornGrid::LoadACTNUM() is obligatory
	Finish_pre();

	if (mat->M.JCount() != props->fname.size() + 3)
		throw HMMPI::Exception("MAT должна содержать столбцов на 3 больше, чем строк в KRIGPROPS",
							   "MAT should have 3 more columns than lines in KRIGPROPS");

	if (!cz->CG.IsCellCoordFilled())
		K->AppText(cz->CG.fill_cell_coord() + "\n");

	assert(cz->CG.IsActnumLoaded());

	HMMPI::Func1D_corr *corr = HMMPI::Func1D_corr_factory::New(var->type);		// correlation function
	corr->SetNugget(var->nugget);
	if (dynamic_cast<HMMPI::CorrMatern*>(corr) != nullptr)
		dynamic_cast<HMMPI::CorrMatern*>(corr)->SetNu(var->nu);

	HMMPI::Mat invK_Ys;
	HMMPI::Vector2<int> pts = get_points();

	RANK0_SYNCERR_BEGIN(MPI_COMM_WORLD);
	HMMPI::Mat K = get_krig_mat(pts, corr);				// ordinary kriging matrix
	HMMPI::Mat Ys = get_krig_Ys();

	HMMPI::SolverDGELSS dgelss;
	invK_Ys = dgelss.Solve(K, Ys);
	RANK0_SYNCERR_END(MPI_COMM_WORLD);

	invK_Ys.Bcast(0, MPI_COMM_WORLD);

	// now each rank does its distributed task
	const double chirad = var->chi/180*pi;
	std::vector<double> res = cz->CG.ord_krig_final_mult(pts, var->R, var->r, var->rz, chirad, corr, invK_Ys);

	// scatter the result for parallel writing to files
	const size_t NG = props->fname.size();
	const size_t actnum_count = cz->CG.GetActnumCount();
	std::vector<int> counts, displs;						// two arrays for distributing NG grids
	HMMPI::MPI_count_displ(MPI_COMM_WORLD, NG, counts, displs);
	if (actnum_count*NG >= (size_t)INT_MAX)
	{
		char buff[HMMPI::BUFFSIZE];
		sprintf(buff, "Array size (%zu) exceeds INT_MAX (%zu) in KW_runKriging::Run", actnum_count*NG, (size_t)INT_MAX);
		throw HMMPI::Exception(buff);
	}

	for (auto &v : counts)
		v *= actnum_count;
	for (auto &v : displs)
		v *= actnum_count;

	assert(K->MPI_rank < (int)counts.size());
	assert(K->MPI_rank < (int)displs.size());
	std::vector<double> res_loc(counts[K->MPI_rank]);		// only stores the local grids for writing to files
	MPI_Scatterv(res.data(), counts.data(), displs.data(), MPI_DOUBLE, res_loc.data(), res_loc.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// write results to files (parallel)
	for (auto &v : counts)		// return "counts" and "displs" back to 'NG'
		v /= actnum_count;
	for (auto &v : displs)
		v /= actnum_count;

	for (int n = 0; n < counts[K->MPI_rank]; n++)			// distributed "for"
	{
		int ind = displs[K->MPI_rank] + n;					// global index of the property
		assert(ind < (int)props->fname.size());

		std::vector<double> grid = cz->CG.krig_result_prop(res_loc, n);			// res_loc is NG_loc*actnum_count
		HMMPI::CornGrid::SavePropertyToFile(props->fname[ind], props->propname[ind], grid);
	}

	delete corr;
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_runIntegPoro::KW_runIntegPoro()
{
	name = "RUNINTEGPORO";
}
//------------------------------------------------------------------------------------------
void KW_runIntegPoro::Run()
{
	// * USER * define a list of lambda-functions (double->double) which are to be integrated
	std::vector<std::function<double(double)>> Funcs;
	Funcs.push_back([](double x) -> double {return 1;});		// NTG
	Funcs.push_back([](double x) -> double {return x;});		// test
	// * USER *

	Start_pre();
	IMPORTKWD(config, KW_integporo_config, "INTEGPORO_CONFIG");
	DECLKWD(func, KW_functionXY, "FUNCTIONXY");
	Finish_pre();

	// set up the transform, if any
	HMMPI::Func1D_CDF *transform = nullptr;
	if (func->data.size() > 0 && func->data[0].Length() > 0)
	{
		const size_t Ni = func->data[0].ICount();
		std::vector<double> x1(Ni, 0.0);
		std::vector<double> y1(Ni, 0.0);
		for (size_t i = 0; i < Ni; i++)
		{
			x1[i] = func->data[0](i, 0);
			y1[i] = func->data[0](i, 1);
		}
		transform = new HMMPI::Func1D_CDF(x1, y1);

		K->AppText(HMMPI::MessageRE("* Делается интегрирование с преобразованием, заданным через PDF из FUNCTIONXY *\n",
									"* Integrating with transform defined via PDF from FUNCTIONXY *\n"));
	}
	else
		K->AppText(HMMPI::MessageRE("В FUNCTIONXY не задана PDF -> делается интегрирование без преобразования...\n",
									"No PDF defined in FUNCTIONXY -> integrating without transform...\n"));

	const std::string mean_name = "PORO";
	const std::string var_name = "VAR";
	const size_t grid_size = size_t(config->Nx)*config->Ny*config->Nz;
	std::vector<std::vector<double>> data_mean(1), data_var(1);				// inputs, only at rank-0; the outermost size is sync though

	// MPI kicks in here
	int rank;
	std::vector<int> counts, displs;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	HMMPI::MPI_count_displ(MPI_COMM_WORLD, grid_size, counts, displs);		// ** NOTE ** too large 'grid_size' is not handled properly!

	std::vector<double> mean(counts[rank]), var(counts[rank]);		// these two are MPI-distributed

	// read the inputs
	char err[HMMPI::BUFFSIZE];		// error message
	err[0] = 0;						// this default state stands for 'no errors'
	if (rank == 0)
	{
		try
		{
			// read the mean
			HMMPI::CornGrid::ReadGrids(config->file_mean.c_str(), std::vector<size_t>{grid_size}, data_mean, std::vector<std::string>{mean_name}, "/");
			assert(data_mean.size() == 1);
			assert(data_mean[0].size() == grid_size);

			// read the variance
			HMMPI::CornGrid::ReadGrids(config->file_var.c_str(), std::vector<size_t>{grid_size}, data_var, std::vector<std::string>{var_name}, "/");
			assert(data_var.size() == 1);
			assert(data_var[0].size() == grid_size);
		}
		catch (HMMPI::Exception &e)
		{
			sprintf(err, "%.*s", HMMPI::BUFFSIZE-2, e.what());
		}
	}
	MPI_Bcast(err, HMMPI::BUFFSIZE, MPI_CHAR, 0, MPI_COMM_WORLD);
	if (err[0] != 0)				// err is sync
		throw HMMPI::Exception(err);

	// make MPI-distribution
	MPI_Scatterv(data_mean[0].data(), counts.data(), displs.data(), MPI_DOUBLE, mean.data(), counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatterv(data_var[0].data(), counts.data(), displs.data(), MPI_DOUBLE, var.data(), counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// calculate integral for each function and save
	std::vector<double> res, loc_res(counts[rank]);
	if (rank == 0)
		res = std::vector<double>(grid_size);		// res : a big guy on rank-0 only

	for (int f = 0; f < (int)Funcs.size(); f++)
	{
		for (int i = 0; i < counts[rank]; i++)
		{
			assert(var[i] >= 0);
			if (transform == nullptr)
				loc_res[i] = HMMPI::integr_Gauss(Funcs[f], config->n, config->phi0, mean[i], sqrt(var[i]));
			else
				loc_res[i] = HMMPI::integr_Gauss(Funcs[f], config->n, config->phi0, mean[i], sqrt(var[i]), *transform);
		}
		MPI_Gatherv(loc_res.data(), counts[rank], MPI_DOUBLE, res.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);	// gather the result

		char prop_name[HMMPI::BUFFSIZE], fname[HMMPI::BUFFSIZE], msg[HMMPI::BUFFSIZE];
		sprintf(prop_name, "PROP_%d", f);
		sprintf(fname, config->file_out_templ.c_str(), f);
		sprintf(msg, "Saved %.30s to %.450s\n", prop_name, fname);

		err[0] = 0;
		if (rank == 0)
		{
			try
			{
				HMMPI::CornGrid::SavePropertyToFile(fname, prop_name, res);
			}
			catch (HMMPI::Exception &e)
			{
				sprintf(err, "%.*s", HMMPI::BUFFSIZE-2, e.what());
			}
		}
		MPI_Bcast(err, HMMPI::BUFFSIZE, MPI_CHAR, 0, MPI_COMM_WORLD);
		if (err[0] != 0)				// err is sync
			throw HMMPI::Exception(err);

		K->AppText(msg);
	}

	delete transform;
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
std::vector<double> KW_runtimelinalg::form_vec(size_t dim)		// dim-vector, with elements in [-1, 1]
{
	assert(rand != nullptr);
	return rand->RandU(dim, 1).ToVector();
}
//------------------------------------------------------------------------------------------
HMMPI::Mat KW_runtimelinalg::form_mat(size_t dim)				// dim*dim matrix, with elements in [-1, 1]
{
	assert(rand != nullptr);
	return rand->RandU(dim, dim);
}
//------------------------------------------------------------------------------------------
HMMPI::TensorTTV KW_runtimelinalg::form_tens3(size_t dim)		// (TD1*dim, TD2*dim, TD3*dim) tensor, with elements in [-1, 1]
{
	assert(rand != nullptr);
	HMMPI::TensorTTV res(std::vector<size_t>({TD1*dim, TD2*dim, TD3*dim}));
	std::vector<double> aux = rand->RandU(TD1*dim * TD2*dim * TD3*dim, 1).ToVector();
	res.fill_from(aux);

	return res;
}
//------------------------------------------------------------------------------------------
// the "run" procedures below run on all ranks, with a barrier in the end;
// after all have finished, the time "t" (sync) is returned
std::vector<double> KW_runtimelinalg::run_dgemv(const HMMPI::Mat &A, const std::vector<double> &x, double &t)	// runs dgemv test on all ranks, returns A*x (different for each rank)
{
	MPI_Barrier(MPI_COMM_WORLD);
	time1 = std::chrono::high_resolution_clock::now();

	std::vector<double> res = A*x;

	MPI_Barrier(MPI_COMM_WORLD);
	time2 = std::chrono::high_resolution_clock::now();
	t = std::chrono::duration_cast<std::chrono::duration<double>>(time2-time1).count();
	MPI_Bcast(&t, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	return res;
}
//------------------------------------------------------------------------------------------
std::vector<double> KW_runtimelinalg::run_dgemm_vec_right(const HMMPI::Mat &A, const std::vector<double> &x, double &t)	// calculates and returns A*x, using dgemm
{
	MPI_Barrier(MPI_COMM_WORLD);
	time1 = std::chrono::high_resolution_clock::now();

	std::vector<double> res = A.MultvecR(x);

	MPI_Barrier(MPI_COMM_WORLD);
	time2 = std::chrono::high_resolution_clock::now();
	t = std::chrono::duration_cast<std::chrono::duration<double>>(time2-time1).count();
	MPI_Bcast(&t, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	return res;
}
//------------------------------------------------------------------------------------------
std::vector<double> KW_runtimelinalg::run_dgemm_vec_left(const HMMPI::Mat &A, const std::vector<double> &x, double &t)	// calculates and returns A*x, as (x'*A')', using dgemm
{
	MPI_Barrier(MPI_COMM_WORLD);
	time1 = std::chrono::high_resolution_clock::now();

    std::vector<double> res = A.MultvecL(x);

	MPI_Barrier(MPI_COMM_WORLD);
	time2 = std::chrono::high_resolution_clock::now();
	t = std::chrono::duration_cast<std::chrono::duration<double>>(time2-time1).count();
	MPI_Bcast(&t, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	return res;
}
//------------------------------------------------------------------------------------------
std::vector<double> KW_runtimelinalg::run_dgelss(const HMMPI::Mat &A, const std::vector<double> &b, double &t)	// runs dgelss test on all ranks, returns A^(-1)*x (different for each rank)
{
	MPI_Barrier(MPI_COMM_WORLD);
	time1 = std::chrono::high_resolution_clock::now();

	HMMPI::SolverDGELSS solv;
	std::vector<double> res = solv.Solve(A, b).ToVector();

	MPI_Barrier(MPI_COMM_WORLD);
	time2 = std::chrono::high_resolution_clock::now();
	t = std::chrono::duration_cast<std::chrono::duration<double>>(time2-time1).count();
	MPI_Bcast(&t, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	return res;
}
//------------------------------------------------------------------------------------------
void KW_runtimelinalg::run_dgemm(const HMMPI::Mat &A, const HMMPI::Mat &B, double &t)					// runs dgemm test on all ranks
{
	MPI_Barrier(MPI_COMM_WORLD);
	time1 = std::chrono::high_resolution_clock::now();

	HMMPI::Mat res = A*B;

	MPI_Barrier(MPI_COMM_WORLD);
	time2 = std::chrono::high_resolution_clock::now();
	t = std::chrono::duration_cast<std::chrono::duration<double>>(time2-time1).count();
	MPI_Bcast(&t, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}
//------------------------------------------------------------------------------------------
HMMPI::Mat KW_runtimelinalg::run_ttv_tlib(const HMMPI::TensorTTV &T, const std::vector<double> &b, size_t mode, double &t)			// runs and returns tlib T*b on all ranks
{
	Start_pre();
	IMPORTKWD(ttv, KW_TTV_config, "TTV_CONFIG");
	Finish_pre();

	MPI_Barrier(MPI_COMM_WORLD);
	time1 = std::chrono::high_resolution_clock::now();

	HMMPI::Mat res = T.MultVec(b, mode-1, ttv->slicing, ttv->loopfusion);

	MPI_Barrier(MPI_COMM_WORLD);
	time2 = std::chrono::high_resolution_clock::now();
	t = std::chrono::duration_cast<std::chrono::duration<double>>(time2-time1).count();
	MPI_Bcast(&t, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	return res;
}
//------------------------------------------------------------------------------------------
HMMPI::Mat KW_runtimelinalg::run_ttv_manual(const HMMPI::Tensor3 &T, const std::vector<double> &b, size_t mode, double &t)			// runs and returns manual T*b on all ranks
{
	MPI_Barrier(MPI_COMM_WORLD);
	time1 = std::chrono::high_resolution_clock::now();

	HMMPI::Mat res = T.MultVec(b, mode-1);		// mode is zero-based here

	MPI_Barrier(MPI_COMM_WORLD);
	time2 = std::chrono::high_resolution_clock::now();
	t = std::chrono::duration_cast<std::chrono::duration<double>>(time2-time1).count();
	MPI_Bcast(&t, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	return res;
}
//------------------------------------------------------------------------------------------
void KW_runtimelinalg::print_header(int dim)											// prints the header depending on TIMELINALG_CONFIG settings
{
	DECLKWD(cfg, KW_timelinalg_config, "TIMELINALG_CONFIG");
	char buff[HMMPI::BUFFSIZE];

	std::vector<std::string> v = {"DGEMV", "DGEMM_VL", "DGEMM_VR"};
	if (cfg->dgelss == "Y")
		v.push_back("DGELSS");
	if (cfg->dgemm == "Y")
		v.push_back("DGEMM");
	if (cfg->ttv1 == "Y")
	{
		v.push_back("TTV_1");
		v.push_back("TTV_1M");
	}
	if (cfg->ttv2 == "Y")
	{
		v.push_back("TTV_2");
		v.push_back("TTV_2M");
	}
	if (cfg->ttv3 == "Y")
	{
		v.push_back("TTV_3");
		v.push_back("TTV_3M");
	}

	std::string s = HMMPI::ToString(v, "%-10s", "\t");
	s.pop_back();		// final '\n'
	sprintf(buff, "D%5d\t%s\t%-23s\t%-s\n", dim, s.c_str(), "norm2 range", "seed");
	K->AppText(buff);
}
//------------------------------------------------------------------------------------------
void KW_runtimelinalg::print_iter(int k, int seed, double t_dgemv, double t_dgemm_vl, double t_dgemm_vr, double t_dgelss, double t_dgemm,
		 	 	 	 	 	 	 	 	 	 	   double t_ttv1, double t_ttv2, double t_ttv3,
		 	 	 	 	 	 	 	 	 	 	   double t_ttv1man, double t_ttv2man, double t_ttv3man, double diff_norm2)		// prints the k-th iteration results
{
	DECLKWD(cfg, KW_timelinalg_config, "TIMELINALG_CONFIG");
	char buff[HMMPI::BUFFSIZE];

	double diff0 = 0, diff1 = 0;		// min and max over ranks
	int seed0 = 0, seed1 = 0;			// min and max over ranks

	MPI_Reduce(&diff_norm2, &diff0, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&diff_norm2, &diff1, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	MPI_Reduce(&seed, &seed0, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&seed, &seed1, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

	std::vector<double> v = {t_dgemv, t_dgemm_vl, t_dgemm_vr};
	if (cfg->dgelss == "Y")
		v.push_back(t_dgelss);
	if (cfg->dgemm == "Y")
		v.push_back(t_dgemm);
	if (cfg->ttv1 == "Y")
	{
		v.push_back(t_ttv1);
		v.push_back(t_ttv1man);
	}
	if (cfg->ttv2 == "Y")
	{
		v.push_back(t_ttv2);
		v.push_back(t_ttv2man);
	}
	if (cfg->ttv3 == "Y")
	{
		v.push_back(t_ttv3);
		v.push_back(t_ttv3man);
	}

	std::string s = HMMPI::ToString(v, "%-10.5e", "\t");
	s.pop_back();		// final '\n'
	sprintf(buff, "k =%3d\t%s\t%-10.5g - %-10.5g\t%d - %d\n", k, s.c_str(), diff0, diff1, seed0, seed1);
	K->AppText(buff);
}
//------------------------------------------------------------------------------------------
void KW_runtimelinalg::print_mean(double ta_dgemv, double ta_dgemm_vl, double ta_dgemm_vr, double ta_dgelss, double ta_dgemm,
		 	 	 	 	 	 	  double ta_ttv1, double ta_ttv2, double ta_ttv3, double ta_ttv1man, double ta_ttv2man, double ta_ttv3man)		// prints mean times
{
	DECLKWD(cfg, KW_timelinalg_config, "TIMELINALG_CONFIG");
	char buff[HMMPI::BUFFSIZE];

	std::vector<double> v = {ta_dgemv, ta_dgemm_vl, ta_dgemm_vr};
	if (cfg->dgelss == "Y")
		v.push_back(ta_dgelss);
	if (cfg->dgemm == "Y")
		v.push_back(ta_dgemm);
	if (cfg->ttv1 == "Y")
	{
		v.push_back(ta_ttv1);
		v.push_back(ta_ttv1man);
	}
	if (cfg->ttv2 == "Y")
	{
		v.push_back(ta_ttv2);
		v.push_back(ta_ttv2man);
	}
	if (cfg->ttv3 == "Y")
	{
		v.push_back(ta_ttv3);
		v.push_back(ta_ttv3man);
	}

	std::string s = HMMPI::ToString(v, "%-10.5e", "\t");
	s.pop_back();		// final '\n'
	sprintf(buff, "%-6s\t%s\n", "mean", s.c_str());
	K->AppText(buff);
}
//------------------------------------------------------------------------------------------
std::vector<double> KW_runtimelinalg::run_iter(int k, int dim)		// runs test iteration 'k' (k = 0, 1, 2,...) for dimension 'dim'; prints time for all tests
{																	// also prints 2-norms: first taking their max over the monitored quantities, and then taking min/max over ranks, printing these min and max;
	DECLKWD(cfg, KW_timelinalg_config, "TIMELINALG_CONFIG");		// returns time for all tests (for subsequent averaging)

	HMMPI::Mat A, M;												// matrix and vector part
	std::vector<double> b, bL, bR, x, x1;
	double t, norm = 0;
	std::vector<double> res(11, 0.0);

	HMMPI::TensorTTV T;												// tensor part
	HMMPI::Tensor3 Ten3;
	std::vector<double> y1, y2, y3;

	assert(rand == nullptr);
	rand = new HMMPI::Rand(cfg->seed_0 + k, -1, 1, 0, 1, false);

	x = form_vec(dim);				// each rank has its own stuff
	A = form_mat(dim);
	if (cfg->dgemm == "Y")
		M = form_mat(dim);

	if (cfg->ttv1 == "Y" || cfg->ttv2 == "Y" || cfg->ttv3 == "Y")	// tensor part
	{
		T = form_tens3(dim);
		Ten3 = HMMPI::Tensor3(TD1*dim, TD2*dim, TD3*dim, T.data());
		y1 = form_vec(TD1*dim);
		y2 = form_vec(TD2*dim);
		y3 = form_vec(TD3*dim);
	}

	b = run_dgemv(A, x, t);
	res[0] = t;

	bL = run_dgemm_vec_left(A, x, t);
	norm = HMMPI::Max(norm, (HMMPI::Mat(b) - HMMPI::Mat(bL)).Norm2());			// each rank has different stuff
	res[1] = t;

	bR = run_dgemm_vec_right(A, x, t);
	norm = HMMPI::Max(norm, (HMMPI::Mat(b) - HMMPI::Mat(bR)).Norm2());			// each rank has different stuff
	res[2] = t;

	if (cfg->dgelss == "Y")
	{
		x1 = run_dgelss(A, b, t);
		res[3] = t;
		norm = HMMPI::Max(norm, (HMMPI::Mat(x) - HMMPI::Mat(x1)).Norm2());		// each rank has different stuff
	}

	if (cfg->dgemm == "Y")
	{
		run_dgemm(A, M, t);
		res[4] = t;
	}

	if (cfg->ttv1 == "Y")
	{
		HMMPI::Mat res0 = run_ttv_tlib(T, y1, 1, t);
		res[5] = t;
		HMMPI::Mat res1 = run_ttv_manual(Ten3, y1, 1, t);
		res[8] = t;
		norm = HMMPI::Max(norm, (res0 - res1).Norm2());							// each rank has different stuff
	}

	if (cfg->ttv2 == "Y")
	{
		HMMPI::Mat res0 = run_ttv_tlib(T, y2, 2, t);
		res[6] = t;
		HMMPI::Mat res1 = run_ttv_manual(Ten3, y2, 2, t);
		res[9] = t;
		norm = HMMPI::Max(norm, (res0 - res1).Norm2());							// each rank has different stuff
	}

	if (cfg->ttv3 == "Y")
	{
		HMMPI::Mat res0 = run_ttv_tlib(T, y3, 3, t);
		res[7] = t;
		HMMPI::Mat res1 = run_ttv_manual(Ten3, y3, 3, t);
		res[10] = t;
		norm = HMMPI::Max(norm, (res0 - res1).Norm2());							// each rank has different stuff
	}

	// 							   dgemv   dgemmL  dgemmR  dgelss  dgemm   ttv1    ttv2    ttv3    ttv1M   ttv2M   ttv3M
	print_iter(k, cfg->seed_0 + k, res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7], res[8], res[9], res[10], norm);

	delete rand;
	rand = nullptr;

	return res;
}
//------------------------------------------------------------------------------------------
KW_runtimelinalg::KW_runtimelinalg() : rand(nullptr)
{
	name = "RUNTIMELINALG";
}
//------------------------------------------------------------------------------------------
void KW_runtimelinalg::Run()
{
	Start_pre();
	IMPORTKWD(cfg, KW_timelinalg_config, "TIMELINALG_CONFIG");
	Add_pre("TTV_CONFIG");
	Finish_pre();

	int dim = cfg->D0;
	const int MaxDim = 100000;		// max allowable dimension

	if (cfg->ttv1 == "Y" || cfg->ttv2 == "Y" || cfg->ttv3 == "Y")	// tensor stuff
		K->AppText("Tensor shape ratios: " + HMMPI::ToString(std::vector<int>{TD1, TD2, TD3}, "%d", " - "));

	for (int i = 0; i < cfg->Ndims; i++)
	{
		if (dim > MaxDim)
			dim = MaxDim;

		print_header(dim);
		std::vector<double> meant(11, 0.0);

		for (int k = 0; k < cfg->Nk; k++)
		{
			std::vector<double> ta = run_iter(k, dim);

			assert(ta.size() == meant.size());
			for (size_t j = 0; j < meant.size(); j++)
				meant[j] += ta[j];
		}
		for (auto &d : meant)
			d /= cfg->Nk;		// find the average

		print_mean(meant[0], meant[1], meant[2], meant[3], meant[4], meant[5], meant[6], meant[7], meant[8], meant[9], meant[10]);
		K->AppText("\n");

		dim *= cfg->Dincfactor;
	}
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_runMCMC::KW_runMCMC()
{
	name = "RUNMCMC";
}
//------------------------------------------------------------------------------------------
void KW_runMCMC::Run()
{
	Start_pre();
	IMPORTKWD(hmc1, KW_MCMC_config, "MCMC_CONFIG");
	DECLKWD(params, KW_parameters, "PARAMETERS");
	Finish_pre();

	std::chrono::high_resolution_clock::time_point time1 = std::chrono::high_resolution_clock::now(), time2;
	ParamsInterface *par_interface = const_cast<ParamsInterface*>(params->GetParamsInterface());			// remove const-ness to update 'init' in the end
	std::string msg_time;

	HMMPI::Mat x0 = par_interface->get_init_act();				// initial point
	HMMPI::MCMC *sampler = hmc1->MakeSampler(this);				// make sampler
	int tot_acc = sampler->Run(hmc1->iter, x0, &msg_time);		// run

	// save the final point to LIMITS/PARAMETERS for subsequent reuse
	{
		std::vector<double> xnew = par_interface->init;
		HMMPI::VecAssign(xnew, par_interface->get_act_ind(), x0.ToVector());		// x0 is ACTDIM, xnew is FULLDIM
		par_interface->init = xnew;
	}

	K->AppText(HMMPI::stringFormatArr("Последняя точка сохранена в {0:%s} и может быть использована в следующем RUNMCMC\n",
									  "The final point was saved in {0:%s} and can be reused in the next RUNMCMC\n", dynamic_cast<KW_item*>(par_interface)->name));
	K->AppText(HMMPI::stringFormatArr("Принято моделей: {0:%f}\n", "Accepted models: {0:%f}\n", double(tot_acc)/hmc1->iter));
	K->AppText(msg_time);
	delete sampler;

	time2 = std::chrono::high_resolution_clock::now();
	K->AppText(HMMPI::stringFormatArr("Время {0:%.3f} сек.\n\n", "CPU {0:%.3f} sec.\n\n",
			std::chrono::duration_cast<std::chrono::duration<double>>(time2-time1).count()));
}
//------------------------------------------------------------------------------------------
