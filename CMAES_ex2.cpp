#include "cmaes.h"
#include "CMAES_interface.h"
#include "ConcretePhysModels.h"
#include "Parsing2.h"
#include "Abstract.h"
#include "Utils.h"
#include <cmath>
#include <ctime>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <vector>
#include <mpi.h>
//#include <sys/stat.h>
#include <filesystem>

//------------------------------------------------------------------------------------------
std::vector<double> OptCMAES::RunOpt(PhysModel *pm, std::vector<double> x0, const OptContext *ctx)
{
	Parser_1 *K = const_cast<Parser_1*>(dynamic_cast<const Parser_1*>(ctx));		// NOTE: a dirty trick removing const-ness
	if (K == nullptr)
		throw HMMPI::Exception("Cannot convert OptContext* to Parser_1* in OptCMAES::RunOpt");

	DECLKWD(initcmaes, KW_initcmaes, "INITCMAES");			// TODO check INITCMAES err
	if (initcmaes->GetState() != "")
		throw HMMPI::Exception((std::string)HMMPI::MessageRE("Проблемы с INITCMAES:\n", "Problems with INITCMAES:\n") + initcmaes->GetState());

	PhysModMPI *PM_MPI = dynamic_cast<PhysModMPI*>(pm);
	if (PM_MPI == nullptr)
		throw HMMPI::Exception("Cannot convert PhysModel* to PhysModMPI* in OptCMAES::RunOpt");

	const long maxResample = 5000000;
	initcmaes->WriteFromLimits(x0);		// only active params are written
	if (K->MPI_rank == 0)
	{
		#ifdef WRITE_RESAMPLES
			std::ofstream sw;
			sw.open(HMMPI::getFullPath(CWD_holder::N, resamples_file));
			sw << "resamples  \taccept. mdls\talpha\n";
			sw.close();
		#endif

		// mkdir(PhysModelHM::uncert_dir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);		deprecated 19.01.2022
		std::filesystem::create_directory(PhysModelHM::uncert_dir);
	}

	double *x = optimize(PM_MPI, 0, 2, maxResample, "cmaes_initials.par", K, of);
	std::vector<double> vec_x(x, x + PM_MPI->ParamsDim_ACT());
	free(x);

	return PM_MPI->tot_par(vec_x);
}
//------------------------------------------------------------------------------------------
std::string OptCMAES::ReportMsg()
{
	char res[HMMPI::BUFFSIZE];
	char reseng[HMMPI::BUFFSIZE];
	sprintf(res, "Целевая функция = %g\n", of);
	sprintf(reseng, "Objective function = %g\n", of);

	return HMMPI::MessageRE(res, reseng);
}
//------------------------------------------------------------------------------------------
// NOTE: some MPI synchronisation was introduced to cmaes.cpp, but not too perfect, so deadlocks are possible;
// But in most practical applications it is expected to work.
//------------------------------------------------------------------------------------------
double *optimize(PhysModMPI *PM, int nrestarts, double incpopsize, const long maxResample, const char *filename, Parser_1 *kw, double &fbestever)
{
  int RNK = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &RNK);

  Parser_1 *K = kw;
  cmaes_t evo;			// the optimizer
  double *const*pop;	// sampled population
  double *fitvals;		// objective function values of sampled population
  fbestever=0;
  double *xbestever=NULL; // store best solution
  double fmean;
  int i, irun,
	lambda = 0,			// offspring population size, 0 invokes default
	countevals = 0;		// used to set for restarts
  char const *stop;		// stop message

  for (irun = 0; irun < nrestarts+1; ++irun)	// restarts
  {
	  // Parameters can be set in three ways. Here as input parameter
	  // to cmaes_init, as value read from cmaes_initials.par in readpara_init
	  // during initialization, and as value read from cmaes_signals.par by
	  // calling cmaes_ReadSignals explicitely.

	  fitvals = cmaes_init(&evo, 0, NULL, NULL, 0, lambda, filename);	// allocs fitvals
	  int popSize = (int)cmaes_Get(&evo, "popsize");

	  K->AppText(std::string(cmaes_SayHello(&evo)) + "\n");
	  evo.countevals = countevals;					// a hack, effects the output and termination
	  cmaes_ReadSignals(&evo, "cmaes_signals.par");	// write initial values, headers in case

	  while(!(stop=cmaes_TestForTermination(&evo)))
	  {
		  // Generate population of new candidate solutions
		  pop = cmaes_SamplePopulation(&evo);	// do not change content of pop

		  // Here optionally handle constraints etc. on pop. You may
		  // call cmaes_ReSampleSingle(&evo, i) to resample the i-th
		  // vector pop[i], see below.  Do not change pop in any other
		  // way. You may also copy and modify (repair) pop[i] only
		  // for the evaluation of the fitness function and consider
		  // adding a penalty depending on the size of the
		  // modification.

		  // Resample
		  long tot_resamples = 0;
		  for (i = 0; i < popSize; ++i)
		  {
				// You may resample the solution i until it lies within the
				// feasible domain here, e.g. until it satisfies given
				// box constraints (variable boundaries). The function
				// is_feasible() needs to be user-defined.
				// Assumptions: the feasible domain is convex, the optimum
				// is not on (or very close to) the domain boundary,
				// initialX is feasible (or in case typicalX +- 2*initialStandardDeviations
				// is feasible) and initialStandardDeviations is (are)
				// sufficiently small to prevent quasi-infinite looping.

				long cnt = 0;
				while (!PM->CheckLimits_ACT(std::vector<double>(pop[i], pop[i] + evo.sp.N)))
				{
					cmaes_ReSampleSingle(&evo, i);
					cnt++;
					if (cnt > maxResample)
						throw HMMPI::Exception("Maximum number of resamples exceeded");
				}
				tot_resamples += cnt;
		  }

		  // find number of models in MC estimate
		  int est_count = 0;
		  for (i = 0; i < popSize; i++)
			  if (evo.dens_val[i] != 0)
				  est_count++;

		  //double alpha = 1 + double(tot_resamples)/popSize;
		  double alpha = 0;
		  if (est_count != 0)
			  alpha = double(tot_resamples + popSize)/est_count;

		  PM->ObjFuncMPI_ACT(popSize, pop, fitvals, false);
		  HMMPI::Bcast_vector(&fitvals, 1, popSize, 0, MPI_COMM_WORLD);		// sync o.f. values; before 29.03.2017 was PM->GetComm(), since it may have many NULL ranks, replaced by MPI_COMM_WORLD
		  #ifdef WRITE_RESAMPLES
		  {
			  if (RNK == 0)
			  {
				  std::string fn = HMMPI::getFullPath(kw->InitCWD, resamples_file);
				  std::ofstream sw;
				  sw.exceptions(std::ios_base::badbit | std::ios_base::failbit);
				  sw.open(fn, std::ios_base::app);
				  sw << HMMPI::stringFormatArr("{0:%-10ld}\t{1:%-10ld}\t", std::vector<long int>{tot_resamples, est_count});
				  sw << HMMPI::stringFormatArr("{0:%-10g}\n", std::vector<double>{alpha});
				  sw.close();
			  }
		  }
		  #endif

		  // update search distribution
		  cmaes_UpdateDistribution(&evo, fitvals);

		  // read control signals for output and termination
		  cmaes_ReadSignals(&evo, "cmaes_signals.par");	// from file cmaes_signals.par

		  fflush(stdout);
	  }	// while !cmaes_TestForTermination(&evo)

	  lambda = int(incpopsize * cmaes_Get(&evo, "lambda"));		// needed for the restart
	  countevals = int(cmaes_Get(&evo, "eval"));				// ditto

	  // print some "final" output
	  std::vector<double> pars {cmaes_Get(&evo, "gen"), cmaes_Get(&evo, "eval"), evo.eigenTimings.totaltime, cmaes_Get(&evo, "funval")};
	  K->AppText(HMMPI::stringFormatArr("{0} generations, {1} fevals ({2} sec): f(x) = {3}\n", pars));

	  pars = std::vector<double>{cmaes_Get(&evo, "maxaxislen")/cmaes_Get(&evo, "minaxislen"), cmaes_Get(&evo, "maxstddev"), cmaes_Get(&evo, "minstddev")};
	  K->AppText(HMMPI::stringFormatArr("axis-ratio = {0:%.3e}, max/min-stddev = {1:%.3e}/{2:%.3e}\n", pars));
	  K->AppText(HMMPI::stringFormatArr("models sampled {0:%d}, ignored {1:%d}, ", std::vector<int>{(int)evo.tot_models, (int)evo.tot_ignored}));
	  K->AppText(HMMPI::stringFormatArr("rate = {0:%f}\n", std::vector<double>{double(evo.tot_ignored)/evo.tot_models}));

	  K->AppText(HMMPI::stringFormatArr("Stop (run {0:%d}):\n", std::vector<int>{irun+1}));
	  K->AppText(std::string(cmaes_TestForTermination(&evo)));

	  // write some data
	  cmaes_WriteToFile(&evo, "all", "cmaes_all.dat");

	  // keep best ever solution
	  if (irun == 0 || cmaes_Get(&evo, "fbestever") < fbestever)
	  {
		  fbestever = cmaes_Get(&evo, "fbestever");
		  xbestever = cmaes_GetInto(&evo, "xbestever", xbestever);	// alloc mem if needed
	  }

	  // best estimator for the optimum is xmean, therefore check
	  const double *p_xmean = cmaes_GetPtr(&evo, "xmean");

	  // calculate a single ObjFunc value
	  fmean = PM->ObjFunc_ACT(std::vector<double>(p_xmean, p_xmean + evo.sp.N));
	  MPI_Bcast(&fmean, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);		// before 29.03.2017 was PM->GetComm()
	  if (fmean < fbestever)
	  {
		  fbestever = fmean;
		  xbestever = cmaes_GetInto(&evo, "xmean", xbestever);
	  }
	  cmaes_exit(&evo);		// does not effect the content of stop string and xbestever

	  // abandon restarts if target fitness value was achieved or MaxFunEvals reached
	  if (stop)				// as it can be NULL
	  {
		  if (strncmp(stop, "Fitness", 7) == 0 || strncmp(stop, "MaxFunEvals", 11) == 0)
			  break;
	  }
  } // for restarts

  return xbestever;		// was dynamically allocated, should be freed in the end
}
//------------------------------------------------------------------------------------------
