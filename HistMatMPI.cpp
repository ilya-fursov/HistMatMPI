
#include <math.h> 
#include <cstdlib>
#include <cstdio>
#include <string>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include "mpi.h"
#include "Abstract.h"
#include "Utils.h"
#include "Parsing.h"
#include "Parsing2.h"
#include "Vectors.h"
#include "Tracking.h"
#include "CMAES_interface.h"
#include "EclSMRY.h"

//using namespace std::chrono;

std::string CWD_holder::N;		// used in CMAES output

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &Parser_1::MPI_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &Parser_1::MPI_rank);

	const std::chrono::high_resolution_clock::time_point time1 = std::chrono::high_resolution_clock::now();
	Parser_1 kw1;

	HMMPI::TextAttr TA;
	try
	{
		std::string control_file;
		std::string cwd;
		if (argc == 0 || argc == 1)
			throw HMMPI::Exception("Не задан контрольный файл", "Control file not defined");

		cwd = HMMPI::getCWD(argv[0]); 						// cwd where the program started

		// check consistency of data sizes used for Eclipse output binary reading
		HMMPI::EclSMRYInitCheckSizes();

		// check consistency of 'size_t' and MPI_LONG_LONG, and other types
		const std::string mpi_size_check_msg = HMMPI::MPI_size_consistent();
		if (mpi_size_check_msg != "")
			throw HMMPI::Exception("Фатальная ошибка: несоответствие размеров данных\n" + mpi_size_check_msg, "Fatal error: data size mismatch\n" + mpi_size_check_msg);

		control_file = std::string(argv[1]);
		if (argc > 2)
		{
			std::string lang = HMMPI::ToUpper(argv[2]);
			if (lang == "RUS" || lang == "ENG")
				HMMPI::MessageRE::lang = lang;
			else if (lang == "NULL")
			{
				HMMPI::MessageRE::lang = "ENG";
				kw1.silent = true;
			}
			else
				throw HMMPI::Exception("Неправильный язык, ожидается: eng, rus, null", "Incorrect language, expected: eng, rus, null");
		}

		cwd = HMMPI::getCWD(HMMPI::getFullPath(cwd, control_file));		// cwd where control file is located
		control_file = HMMPI::getFile(control_file);

		kw1.AppText("CWD: " + cwd + "\n");
		kw1.AppText("Control file: " + control_file + "\n");
		kw1.AppText("Language: " + HMMPI::MessageRE::lang + "\n");
		kw1.AppText("Reading control file...\n\n");

		// Adding keyword items
		kw1.AddKW_item(new KW_include);
		kw1.AddKW_item(new KW_verbosity);
		kw1.AddKW_item(new KW_echo);
		kw1.AddKW_item(new KW_noecho);
		kw1.AddKW_item(new KW_variogram);
		kw1.AddKW_item(new KW_variogram_Cs);
		kw1.AddKW_item(new KW_variogram_3D);
		kw1.AddKW_item(new KW_krigprops);
		kw1.AddKW_item(new KW_report);
		kw1.AddKW_item(new KW_ofweights);
		kw1.AddKW_item(new KW_functionXY);
		kw1.AddKW_item(new KW_Dtable);
		kw1.AddKW_item(new KW_conc_data);
		kw1.AddKW_item(new KW_griddims);

		kw1.AddKW_item(new KW_griddimens);
		kw1.AddKW_item(new KW_CoordZcorn);
		kw1.AddKW_item(new KW_Actnum);
		kw1.AddKW_item(new KW_integporo_config);

		kw1.AddKW_item(new KW_datafile);
		kw1.AddKW_item(new KW_funrst);
		kw1.AddKW_item(new KW_funsmry);
		kw1.AddKW_item(new KW_textsmry);
		kw1.AddKW_item(new KW_refmap);
		kw1.AddKW_item(new KW_refmap_w);
		kw1.AddKW_item(new KW_dates);
		kw1.AddKW_item(new KW_startdate);
		kw1.AddKW_item(new KW_groups);
		kw1.AddKW_item(new KW_3points);
		kw1.AddKW_item(new KW_satsteps);
		kw1.AddKW_item(new KW_delta);
		kw1.AddKW_item(new KW_pilot);
		kw1.AddKW_item(new KW_simcmd);
		kw1.AddKW_item(new KW_limits);
		kw1.AddKW_item(new KW_limitsKrig);

		kw1.AddKW_item(new KW_Swco);
		kw1.AddKW_item(new KW_SWOFParams);
		kw1.AddKW_item(new KW_SGOFParams);
		kw1.AddKW_item(new KW_gas);
		kw1.AddKW_item(new KW_RML);
		kw1.AddKW_item(new KW_incfiles);
		kw1.AddKW_item(new KW_funrstG);
		kw1.AddKW_item(new KW_funrstA);
		kw1.AddKW_item(new KW_fegrid);
		kw1.AddKW_item(new KW_eclvectors);
		kw1.AddKW_item(new KW_fsmspec);
		kw1.AddKW_item(new KW_undef);
		kw1.AddKW_item(new KW_initcmaes);
		kw1.AddKW_item(new KW_mapreg);
		kw1.AddKW_item(new KW_refmapM);
		kw1.AddKW_item(new KW_WRfunrst);
		kw1.AddKW_item(new KW_Pcapill);
		kw1.AddKW_item(new KW_mapseisscale);
		kw1.AddKW_item(new KW_mapseiswght);
		kw1.AddKW_item(new KW_regressRs);
		kw1.AddKW_item(new KW_regressConstr);
		kw1.AddKW_item(new KW_regressquadr);
		kw1.AddKW_item(new KW_plotparams);
		kw1.AddKW_item(new KW_optimization);
		kw1.AddKW_item(new KW_wrcovar);

		kw1.AddKW_item(new KW_runForward);
		kw1.AddKW_item(new KW_runOpt);
		kw1.AddKW_item(new KW_runCritGrad);
		kw1.AddKW_item(new KW_runPlot);
		kw1.AddKW_item(new KW_runGrad);
		kw1.AddKW_item(new KW_runJac);
		kw1.AddKW_item(new KW_runMCMC);
		kw1.AddKW_item(new KW_runcalccovar);
		kw1.AddKW_item(new KW_runcalcwellcovar);
		kw1.AddKW_item(new KW_runmpicheck);
		kw1.AddKW_item(new KW_runNNCfromgrid);
		kw1.AddKW_item(new KW_runPinchMarkFromGrid);
		kw1.AddKW_item(new KW_runGridIJK_to_XYZ);
		kw1.AddKW_item(new KW_runXYZ_to_GridIJK);
		kw1.AddKW_item(new KW_runKriging);
		kw1.AddKW_item(new KW_runIntegPoro);
		kw1.AddKW_item(new KW_runtimelinalg);

		kw1.AddKW_item(new KW_LinSolver);
		kw1.AddKW_item(new KW_TTV_config);
		kw1.AddKW_item(new KW_soboltest);
		kw1.AddKW_item(new KW_runsoboltest);
		kw1.AddKW_item(new KW_matrixtest);
		kw1.AddKW_item(new KW_runmatrixtest);
		kw1.AddKW_item(new KW_runRosenbrock);
		kw1.AddKW_item(new KW_proxyros);
		kw1.AddKW_item(new KW_proxylin);
		kw1.AddKW_item(new KW_matvecvec);
		kw1.AddKW_item(new KW_matvec);
		kw1.AddKW_item(new KW_mat);
		kw1.AddKW_item(new KW_runmatinv);
		kw1.AddKW_item(new KW_rundebug);
		kw1.AddKW_item(new KW_MCMC_config);
		kw1.AddKW_item(new KW_pConnect_config);
		kw1.AddKW_item(new KW_viewsmry_config);
		kw1.AddKW_item(new KW_view_tNavsmry_config);
		kw1.AddKW_item(new KW_view_tNavsmry_properties);
		kw1.AddKW_item(new KW_corrstruct);

		kw1.AddKW_item(new KW_physmodel);
		kw1.AddKW_item(new KW_vectmodel);
		kw1.AddKW_item(new KW_proxy);
		kw1.AddKW_item(new KW_proxy_dump);
		kw1.AddKW_item(new KW_shell);

		kw1.AddKW_item(new KW_parameters);
		kw1.AddKW_item(new KW_parameters2);
		kw1.AddKW_item(new KW_prior);
		kw1.AddKW_item(new KW_templates);
		kw1.AddKW_item(new KW_eclsmry);
		kw1.AddKW_item(new KW_model);
		kw1.AddKW_item(new KW_multiple_seq);
		kw1.AddKW_item(new KW_timelinalg_config);
		kw1.AddKW_item(new KW_opt_config);
		kw1.AddKW_item(new KW_runSingle);
		kw1.AddKW_item(new KW_runMultiple);
		kw1.AddKW_item(new KW_runOptProxy);
		kw1.AddKW_item(new KW_runPopModel);
		kw1.AddKW_item(new KW_runViewSmry);
		kw1.AddKW_item(new KW_runView_tNavSmry);

		// Adding console text tweaks
		//kw1.AddCTT(new CTT_Keyword(&TA));
		kw1.AddCTT(new CTT_ColorString("ERROR", HMMPI::Color::VT_RED, &TA));
		kw1.AddCTT(new CTT_ColorString("WARNING", HMMPI::Color::VT_YELLOW, &TA));
		kw1.AddCTT(new CTT_ColorString("redund. ln.", HMMPI::Color::VT_MAGENTA, &TA));
		kw1.AddCTT(new CTT_ColorString("ОШИБКА", HMMPI::Color::VT_RED, &TA));
		kw1.AddCTT(new CTT_ColorString("ПРЕДУПРЕЖДЕНИЕ", HMMPI::Color::VT_YELLOW, &TA));
		kw1.AddCTT(new CTT_ColorString("Лишн. стр.", HMMPI::Color::VT_MAGENTA, &TA));

#ifdef TESTCTOR
		FILE *testf = fopen(HMMPI::stringFormatArr("TESTCTOR_out_{0:%d}.txt", std::vector<int>{Parser_1::MPI_rank}).c_str(), "w");		// create empty file
		fclose(testf);
#endif

#ifdef TEST_CACHE
	char fname[500];
	sprintf(fname, TEST_CACHE, Parser_1::MPI_rank);
	FILE *f = fopen(fname, "w");
	if (f != NULL)
		fclose(f);
#endif

		CWD_holder::N = cwd;	// CMAES output

		DataLines dl1;
		dl1.LoadFromFile(HMMPI::getFullPath(cwd, control_file));

		kw1.InitCWD = cwd;
		kw1.verbosity = 0;
		kw1.TotalErrors = 0;
		kw1.TotalWarnings = 0;

		kw1.ReadLines(dl1.EliminateEmpty(), 0, cwd);

		// final stuff
		kw1.AppText(HMMPI::MessageRE("Чтение управляющего файла завершено\n", "Finished reading control file\n"));
		kw1.AppText(HMMPI::stringFormatArr("Предупреждений: {0:%d}\n", "Warnings: {0:%d}\n", kw1.TotalWarnings));
		kw1.AppText(HMMPI::stringFormatArr("Ошибок: {0:%d}\n", "Errors: {0:%d}\n", kw1.TotalErrors));

		const std::chrono::high_resolution_clock::time_point time2 = std::chrono::high_resolution_clock::now();
		kw1.AppText(HMMPI::stringFormatArr("Время: {0:%.3f} сек.\n", "Time elapsed: {0:%.3f} sec.\n", std::chrono::duration_cast<std::chrono::duration<double>>(time2-time1).count()));

		KW_report *report = dynamic_cast<KW_report*>(kw1.GetKW_item("REPORT"));
		if (report->GetState() == "")
			report->data_io();

		kw1.DeleteItems();
		kw1.DeleteCTTs();
	}
	catch (const std::exception &e)
	{
		kw1.DeleteItems();
		kw1.DeleteCTTs();
		if (Parser_1::MPI_rank == 0)
			std::cerr << "Error: " << e.what() << std::endl;
	}

	MPI_Comm parent_comm;
	MPI_Comm_get_parent(&parent_comm);
	if (parent_comm != MPI_COMM_NULL)				// child process
	{												// should have a sync point here to signal the parent that it has finished
		HMMPI::MPI_BarrierSleepy(parent_comm);					// TODO it is good to Bcast the TotalErrors to the parent process!!!
		MPI_Comm_free(&parent_comm);
	}

	MPI_Bcast(&kw1.TotalErrors, 1, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Finalize();
	return kw1.TotalErrors;		// non-zero exit status on errors
}

