/*
 * ParsingTest.cpp
 *
 *  Created on: 29 Sep 2016
 *      Author: ilya fursov
 */

#include "Abstract.h"
#include "MathUtils.h"
#include "lapacke.h"
#include "mpi.h"
#include "Vectors.h"
#include "MonteCarlo.h"
#include "Parsing.h"
#include "Parsing2.h"
#include "PhysModels.h"
#include "ConcretePhysModels.h"
#include "Tracking.h"
#include "GradientOpt.h"
#include "CornerPointGrid.h"
#include "EclSMRY.h"
#include <algorithm>
#include <cmath>
#include <cassert>
#include <chrono>
#include <string>
#include <limits>
#include <cstdio>
#include <iostream>
#include <fstream>

#include "alglib-3.10.0_cpp/optimization.h"

// implementation of some testing RUN_SOMETHING procedures

using namespace alglib;
//------------------------------------------------------------------------------------------
KW_runsoboltest::KW_runsoboltest()
{
	name = "RUNSOBOLTEST";
}
//------------------------------------------------------------------------------------------
void KW_runsoboltest::Run()
{
	if (K->MPI_rank == 0)
	{
		Start_pre();
		IMPORTKWD(stest, KW_soboltest, "SOBOLTEST");
		Finish_pre();

		K->AppText((std::string)HMMPI::MessageRE("(eng)", "Saving Sobol sequence to file ") + stest->fname + " ...\n");

		std::string fn = this->CWD + "/" + stest->fname;
		FILE *sw = fopen(fn.c_str(), "w");
		if (!sw)
			throw HMMPI::Exception("(eng)", "Cannot open file for writing");

		fprintf(sw, "Seed\tvector\n");

		std::vector<double> vec(stest->dim);
		long long int seed = stest->seed;
		for (int i = 0; i < stest->num; i++)
		{
			fprintf(sw, "%lld", seed);
			HMMPI::Sobol(seed, vec);
			for (int j = 0; j < stest->dim; j++)
				fprintf(sw, "\t%10.8f", vec[j]);
			fprintf(sw, "\n");
		}
		fclose(sw);
	}
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_runmatrixtest::KW_runmatrixtest()
{
	name = "RUNMATRIXTEST";
}
//------------------------------------------------------------------------------------------
void KW_runmatrixtest::Run()
{
	if (K->MPI_rank == 0)
	{
		Start_pre();
		IMPORTKWD(mtest, KW_matrixtest, "MATRIXTEST");
		Finish_pre();

		HMMPI::Mat A, B, C, D, E;
		K->AppText((std::string)HMMPI::MessageRE("(eng)", "Reading matrices A, B, C, D, E from file ") + mtest->filein + " ...\n");

		// read
		FILE *f = fopen(mtest->filein.c_str(), "r");
		if (!f)
			throw HMMPI::Exception("Невозможно открыть файл для чтения", "Cannot open file for reading");
		A.LoadASCII(f, mtest->sizeA);
		B.LoadASCII(f, mtest->sizeB);
		C.LoadASCII(f, mtest->sizeC);
		D.LoadASCII(f, mtest->sizeD);
		E.LoadASCII(f);
		fclose(f);

		// start writing
		f = fopen(mtest->fileout.c_str(), "w");
		if (!f)
			throw HMMPI::Exception("Невозможно открыть файл для записи", "Cannot open file for writing");
		fprintf(f, "I\n");
		HMMPI::Mat(A.ICount()).SaveASCII(f);

		fprintf(f, "\nA\n");
		A.SaveASCII(f);

		fprintf(f, "\nA^t\n");
		A.Tr().SaveASCII(f);

		// calculate
		K->AppText(HMMPI::MessageRE("(eng)", "Calculating C*diag\n"));
		std::vector<double> diag1(C.JCount(), 0);
		std::iota(diag1.begin(), diag1.end(), 1);
		HMMPI::Mat Cdiag = C % diag1;

		K->AppText(HMMPI::MessageRE("(eng)", "Calculating diag*C\n"));
		std::vector<double> diag2(C.ICount(), 0);
		std::iota(diag2.begin(), diag2.end(), 1);
		HMMPI::Mat diagC = diag2 % C;

		K->AppText(HMMPI::MessageRE("(eng)", "Finding max(A)\n"));
		char buff[HMMPI::BUFFSIZE];
		int mai, maj;
		double maxA = A.Max(mai, maj);
		sprintf(buff, "max(A) = %g, i = %d, j = %d\n", maxA, mai, maj);
		K->AppText(buff);

		K->AppText(HMMPI::MessageRE("(eng)", "Calculating A + B\n"));
		HMMPI::Mat sum = A + B;

		K->AppText(HMMPI::MessageRE("(eng)", "Calculating A - B\n"));
		HMMPI::Mat diff = A - B;

		K->AppText(HMMPI::MessageRE("(eng)", "Calculating A && B\n"));
		HMMPI::Mat AB = A && B;

		K->AppText(HMMPI::MessageRE("(eng)", "Calculating A || B\n"));
		HMMPI::Mat A_B = A || B;

		K->AppText(HMMPI::MessageRE("(eng)", "Calculating A * C\n"));
		HMMPI::Mat mult = A * C;

		K->AppText(HMMPI::MessageRE("(eng)", "Calculating pi * A\n"));
		HMMPI::Mat multpi = acos(-1) * A;

		K->AppText(HMMPI::MessageRE("(eng)", "Calculating sin(A)\n"));
		auto func1 = [](double d){return sin(d);};
		HMMPI::Mat sinA = A;
		sinA.Func(func1);			// compiler has no problem here (no error)

		K->AppText(HMMPI::MessageRE("(eng)", "Calculating {100*i + j}\n"));
		auto func2 = [](int i, int j, double d){return 100*i + j;};
		HMMPI::Mat stoIJ = A;
		stoIJ.FuncInd(func2);

		K->AppText(HMMPI::MessageRE("(eng)", "Calculating A += B (print A)\n"));
		A += B;

		K->AppText(HMMPI::MessageRE("(eng)", "Calculating A -= B (print A)\n"));
		HMMPI::Mat A1 = A;
		A1 -= B;

		K->AppText(HMMPI::MessageRE("(eng)", "Calculating Chol(D)\n"));
		HMMPI::Mat ch = D.Chol();

		K->AppText(HMMPI::MessageRE("(eng)", "Calculating Eig(D)\n"));
		std::vector<double> eig1 = D.EigVal(0, 1);
		std::vector<double> eig2 = D.EigVal(0, 3);
		std::vector<double> eig3 = D.EigVal(0, D.ICount());

		K->AppText(HMMPI::MessageRE("(eng)", "Calculating D/ones\n"));
		HMMPI::Mat ones(D.ICount(), 1, 1.0);
		HMMPI::Mat Dones = D/std::move(ones);
		//HMMPI::Mat Dones = D/(ones + (ones + (ones + (ones + ones))));
		//HMMPI::Mat Dones = D/(ones + ones + ones + ones + ones);

		K->AppText(HMMPI::MessageRE("(eng)", "Calculating Sum(E)\n"));
		K->AppText(HMMPI::stringFormatArr("Sum = {0}\n", std::vector<double>{E.Sum()}));

		K->AppText(HMMPI::MessageRE("(eng)", "Calculating (E, E)\n"));
		sprintf(buff, "Inner product (E, E) = %g\n", InnerProd(E, E));
		K->AppText(buff);

		K->AppText(HMMPI::MessageRE("(eng)", "Calculating Autocorr(E)\n"));
		HMMPI::Mat ac = E.Autocorr();

		K->AppText(HMMPI::MessageRE("(eng)", "Calculating Ess(E)\n"));
		double ess;
		int lag = E.Ess(ess);
		sprintf(buff, "ess = %g, lag = %d\n", ess, lag);
		K->AppText(buff);

		// write the rest
		K->AppText((std::string)HMMPI::MessageRE("(eng)", "Saving results to file ") + mtest->fileout + " ...\n");

		fprintf(f, "\nB\n");
		B.SaveASCII(f);

		fprintf(f, "\nC\n");
		C.SaveASCII(f);

		fprintf(f, "\nC^t\n");
		C.Tr().SaveASCII(f);

		fprintf(f, "\nC*diag\n");
		Cdiag.SaveASCII(f);

		fprintf(f, "\ndiag*C\n");
		diagC.SaveASCII(f);

		fprintf(f, "\nD\n");
		D.SaveASCII(f);

		fprintf(f, "\nE\n");
		E.SaveASCII(f);

		fprintf(f, "\nA + B\n");
		sum.SaveASCII(f);

		fprintf(f, "\nA - B\n");
		diff.SaveASCII(f);

		fprintf(f, "\nA && B\n");
		AB.SaveASCII(f);

		fprintf(f, "\nA || B\n");
		A_B.SaveASCII(f);

		fprintf(f, "\nA * C\n");
		mult.SaveASCII(f);

		fprintf(f, "\npi * A\n");
		multpi.SaveASCII(f);

		fprintf(f, "\nsin(A)\n");
		sinA.SaveASCII(f);

		fprintf(f, "\n{100*i + j}\n");
		stoIJ.SaveASCII(f);

		fprintf(f, "\nA += B, print A\n");
		A.SaveASCII(f);

		fprintf(f, "\nA -= B, print A\n");
		A1.SaveASCII(f);

		fprintf(f, "\nChol(D)\n");
		ch.SaveASCII(f);

		fprintf(f, "\nEig(D): 1, 1-3, all\n");
		fputs(HMMPI::ToString(eig1).c_str(), f);
		fputs(HMMPI::ToString(eig2).c_str(), f);
		fputs(HMMPI::ToString(eig3).c_str(), f);

		fprintf(f, "\nD/ones\n");
		Dones.SaveASCII(f);

		fprintf(f, "\nAutocorr(E)\n");
		ac.SaveASCII(f);

		fclose(f);
	}
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_runRosenbrock::KW_runRosenbrock()
{
	name = "RUNROSENBROCK";
}
//------------------------------------------------------------------------------------------
void KW_runRosenbrock::Run()
{
	const char *filename = "TestRosenbrock.txt";

	Start_pre();
	IMPORTKWD(opt, KW_optimization, "OPTIMIZATION");
	IMPORTKWD(limits, KW_limits, "LIMITS");
	Finish_pre();

	if (limits->init.size() <= 1)
		throw HMMPI::Exception("Размерность должна быть >= 2", "Dimension >= 2 is required");

	std::vector<double> ones(limits->init.size());
	for (auto &i : ones)
		i = 1;

	PM_Rosenbrock PM(limits->init.size());
	PhysModGradNum PM_grad(MPI_COMM_WORLD, &PM, opt->fin_diff, limits->dh, limits->dh_type);

	double of1 = 0, of2 = 0;
	std::vector<double> grad1, grad2;
	double graddir1 = 0, graddir2 = 0;
	HMMPI::Mat Hess1, Hess2;

	if (K->MPI_rank == 0)
	{
		of1 = PM.ObjFunc(limits->init);
		grad1 = PM.ObjFuncGrad(limits->init);
		graddir1 = PM.ObjFuncGradDir(limits->init, ones);	// via full gradient
		Hess1 = PM.ObjFuncHess(limits->init);
	}

	of2 = PM_grad.ObjFunc(limits->init);
	grad2 = PM_grad.ObjFuncGrad(limits->init);
	graddir2 = PM_grad.ObjFuncGradDir(limits->init, ones);	// direct finite differences
	Hess2 = PM_grad.ObjFuncHess(limits->init);

	if (K->MPI_rank == 0)	// write the results
	{
		FILE *f = fopen(filename, "w");
		if (!f)
			throw HMMPI::Exception("Невозможно открыть файл для записи", "Cannot open file for writing");

		K->AppText((std::string)"Writing to file " + filename + "...\n");

		fprintf(f, "objective function 1\n%g\n", of1);
		fprintf(f, "objective function 2\n%g\n", of2);
		fprintf(f, "\ngradient along direction 1\n%g\n", graddir1);
		fprintf(f, "gradient along direction 2 (fin. diff.)\n%g\n", graddir2);

		fprintf(f, "\ngradient 1\n");
		HMMPI::Mat(grad1).Tr().SaveASCII(f);
		fprintf(f, "gradient 2 (fin. diff.)\n");
		HMMPI::Mat(grad2).Tr().SaveASCII(f);

		fprintf(f, "\nHessian 1\n");
		Hess1.SaveASCII(f);
		fprintf(f, "Hessian 2 (fin. diff.)\n");
		Hess2.SaveASCII(f);

		fclose(f);
	}
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_runmatinv::KW_runmatinv()
{
	name = "RUNMATINV";
}
//------------------------------------------------------------------------------------------
void KW_runmatinv::Run()
{
	Start_pre();
	IMPORTKWD(mat, KW_matvecvec, "MATVECVEC");
	IMPORTKWD(linsol, KW_LinSolver, "LINSOLVER");
	Finish_pre();

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	FILE *f = 0;
	try
	{
		HMMPI::Mat rhs = HMMPI::Mat(mat->v1) && HMMPI::Mat(mat->v2);
		HMMPI::Mat sol = linsol->Sol(0)->Solve(mat->M, rhs);

		if (rank == 0)
		{
			f = fopen("OutputRunMatInv.txt", "w");
			sol.SaveASCII(f);
		}

		fclose(f);
	}
	catch (...)
	{
		fclose(f);
		throw;
	}
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_rundebug::KW_rundebug()
{
	name = "RUNDEBUG";
}
//------------------------------------------------------------------------------------------
std::vector<double> ddk(const HMMPI::SpherCoord &sc, std::vector<double> v, int k)
{
	const double dh = 1e-4;
	v[k] += dh;
	std::vector<double> x1 = sc.spher_to_cart(v);
	v[k] -= 2*dh;
	std::vector<double> x2 = sc.spher_to_cart(v);
	for (size_t i = 0; i < x1.size(); i++)
		x1[i] = (x1[i] - x2[i])/(2*dh);

	return x1;
}
//------------------------------------------------------------------------------------------
void KW_rundebug::Run()
{
	// test ...
	Start_pre();
	//IMPORTKWD(mat, KW_mat, "MAT");
	//IMPORTKWD(opt, KW_opt_config, "OPT_CONFIG");
	//IMPORTKWD(limits, KW_limits, "LIMITS");
	IMPORTKWD(cz, KW_CoordZcorn, "COORDZCORN");
	IMPORTKWD(pts, KW_3points, "3POINTS");
	Add_pre("GRIDDIMENS");
	Finish_pre();

	std::string str1 = "Гнев, богиня, воспой Ахиллеса, Пелеева сына\nГрозный, который Ахеянаям тысчи бедствий соделал\n"
					   "Многие души он славных героев низринул\nВ мрачный Аид, а самих распростер их в корысть плотоядным\n"
					   "Птицам окрестным и псам (совершалася Зевсова воля)\nС оного дня как взаимной враждой воспылали\n"
					   "Пастырь народов Атрид и герой Ахиллес благородный\n";

	//FILE *f = fopen("output_vecs.bin", "wb");

	cz->CG.fill_cell_coord();

	for (size_t i = 0; i < pts->x.size(); i++)
	{
		double x, y;
		cz->CG.temp_coord_from_cell(pts->x[i]-1, pts->y[i]-1, pts->z[i]-1, x, y);
		printf("%d\t%d\t%d\t-\t%.0f\t%.0f\n", (int)pts->x[i], (int)pts->y[i], (int)pts->z[i], x, y);
	}
}
//------------------------------------------------------------------------------------------
