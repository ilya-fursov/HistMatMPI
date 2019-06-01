/*
 * ParsingParams2.cpp
 *
 *  Created on: Mar 18, 2013
 *      Author: ilya
 */

#include <iostream>
#include <fstream>
#include <cassert>
#include <algorithm>
#include "Abstract.h"
#include "Vectors.h"
#include "Parsing.h"
#include "Parsing2.h"
#include "ConcretePhysModels.h"
#include "lapacke.h"

//------------------------------------------------------------------------------------------
// in this file all sorts of "parameter" keywords are implemented
//------------------------------------------------------------------------------------------
void KW_verbosity::UpdateParams() noexcept
{
	K->verbosity = level;
}
//------------------------------------------------------------------------------------------
KW_verbosity::KW_verbosity()
{
	name = "VERBOSITY";

	DEFPAR(level, 0);
	FinalizeParams();
}
//------------------------------------------------------------------------------------------
KW_gas::KW_gas()
{
	name = "GAS";

	DEFPAR(on, "OFF");

	FinalizeParams();
	EXPECTED[0] = std::vector<std::string>{"OFF", "ON"};
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_RML::KW_RML()
{
	name = "RML";

	DEFPAR(on, "OFF");
	DEFPAR(seed, 0);

	FinalizeParams();
	EXPECTED[0] = std::vector<std::string>{"OFF", "ON"};
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_viewsmry_config::KW_viewsmry_config()
{
	name = "VIEWSMRY_CONFIG";

	DEFPAR(out_file, "Summary_view.txt");
	DEFPAR(order, "DIRECT");

	FinalizeParams();
	EXPECTED[1] = std::vector<std::string>{"DIRECT", "SORT"};
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
void KW_multiple_seq::UpdateParams() noexcept
{
	if (N < 0 || N > 1000000)
		SilentError(HMMPI::MessageRE("N должно быть в пределах от 0 до 10^6", "N should be within the range 0 to 10^6"));
}
//------------------------------------------------------------------------------------------
KW_multiple_seq::KW_multiple_seq()
{
	name = "MULTIPLE_SEQ";

	DEFPAR(N, 10);
	DEFPAR(MaxHours, 1.0);
	DEFPAR(type, "SOBOL");
	DEFPAR(seed, 0);
	DEFPAR(R, 0.2);

	FinalizeParams();
	EXPECTED[2] = std::vector<std::string>{"SOBOL", "RANDGAUSS"};		// "RANDU" to be added TODO
}
//------------------------------------------------------------------------------------------
void KW_multiple_seq::FinalAction() noexcept
{
	if (seed == 0)
	{
		seed = time(NULL);
		K->AppText(HMMPI::stringFormatArr("Используется сид {0:%d}\n", "Using seed {0:%d}\n", seed));
	}
	MPI_Bcast(&seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
}
//------------------------------------------------------------------------------------------
std::string KW_multiple_seq::msg() const
{
	char buff[HMMPI::BUFFSIZE];
	char buffeng[HMMPI::BUFFSIZE];

	std::string type0 = type;
	if (type == "RANDGAUSS")
		type0 += HMMPI::stringFormatArr(" (R = {0})", std::vector<double>{R});

	sprintf(buff, "Моделей в последовательности: %d, генератор случайных чисел: %s, сид: %d, последовательность параметров будет сохранена в файл %s\n", N, type0.c_str(), seed, logfile.c_str());
	sprintf(buffeng, "Models in sequence: %d, random numbers generator: %s, seed: %d, parameters sequence will be saved to file '%s'\n", N, type0.c_str(), seed, logfile.c_str());
	return HMMPI::MessageRE(buff, buffeng);
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_simcmd::KW_simcmd()
{
	name = "SIMCMD";
	delim = "";
	ecols = 1;
	dec_verb = -1;
	DEFPARMULT(cmd);

	FinalizeParams();
}
//------------------------------------------------------------------------------------------
void KW_simcmd::UpdateParams() noexcept
{
	cmd_work = cmd;
}
//------------------------------------------------------------------------------------------
void KW_simcmd::RunCmd() const
{
	for (const std::string &c : cmd_work)
		system(c.c_str());
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_shell::KW_shell()
{
	name = "SHELL";
	delim = "";
	ecols = 1;
	dec_verb = -1;
	DEFPARMULT(cmd);

	FinalizeParams();
}
//------------------------------------------------------------------------------------------
void KW_shell::FinalAction() noexcept
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	for (size_t i = 0; i < cmd.size(); i++)
	{
		if (rank == 0)
			system(cmd[i].c_str());
		MPI_Barrier(MPI_COMM_WORLD);
	}
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_undef::KW_undef()
{
	name = "UNDEF";

	DEFPAR(Uvect, -999.5);
	DEFPAR(Uvectbhp, -999.5);
	DEFPAR(Ugrid, -9999.0);

	FinalizeParams();
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_variogram::KW_variogram()
{
	name = "VARIOGRAM";

	DEFPAR(chi, 0);
	DEFPAR(R, 1000);
	DEFPAR(r, 1000);
	DEFPAR(sill, 1);
	DEFPAR(nugget, 0);
	DEFPAR(type, "SPHER");
	DEFPAR(krig_type, "ORD");

	FinalizeParams();
	EXPECTED[5] = std::vector<std::string>{"EXP", "SPHER", "GAUSS"};
	EXPECTED[6] = std::vector<std::string>{"SIM", "ORD"};
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_variogram_Cs::KW_variogram_Cs()
{
	name = "VARIOGRAM_CS";
	R = 0.01;
	r = 0.01;
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_ofweights::KW_ofweights()
{
	name = "OFWEIGHTS";

	DEFPAR(w1, 1);
	DEFPAR(w2, 0);
	DEFPAR(w3, 0);
	DEFPAR(w4, 0);
	DEFPAR(w5, 0);

	FinalizeParams();
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_regressquadr::KW_regressquadr()
{
	name = "REGRESSQUADR";

	DEFPAR(P2, 0);
	DEFPAR(Sw2, 0);
	DEFPAR(Sg2, 0);
	DEFPAR(PSw, 0);
	DEFPAR(PSg, 0);
	DEFPAR(SwSg, 0);

	FinalizeParams();
}
//------------------------------------------------------------------------------------------
std::vector<int> KW_regressquadr::IndActive(bool gas)
{
	std::vector<int> res;
	if (P2 != 0)
		res.push_back(0);
	if (Sw2 != 0)
		res.push_back(4);
	if (Sg2 != 0 && gas)
		res.push_back(8);
	if (PSw != 0)
		res.push_back(3);
	if (PSg != 0 && gas)
		res.push_back(6);
	if (SwSg != 0 && gas)
		res.push_back(7);

	return res;
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_regressRs::KW_regressRs()
{
	name = "REGRESSRS";

	DEFPAR(Rs, 0);
	DEFPAR(Rs2, 0);
	DEFPAR(RsP, 0);
	DEFPAR(RsSw, 0);
	DEFPAR(RsSg, 0);

	FinalizeParams();
}
//------------------------------------------------------------------------------------------
std::vector<int> KW_regressRs::IndActive(bool gas)
{
	std::vector<int> res;
	if (Rs != 0)
		res.push_back(0);
	if (Rs2 != 0)
		res.push_back(1);
	if (RsP != 0)
		res.push_back(2);
	if (RsSw != 0)
		res.push_back(3);
	if (RsSg != 0 && gas)
		res.push_back(4);

	return res;
}
//------------------------------------------------------------------------------------------
void KW_regressRs::UpdateParams() noexcept
{
	if ((Rs2 || RsP || RsSw || RsSg) && Rs == 0)
	{
		Rs = 1;

		K->TotalWarnings++;
		K->AppText(HMMPI::MessageRE("(eng)\n",
								    "WARNING: quadratic terms with Rs are defined, linear Rs term will be included\n"));
	}
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_Swco::KW_Swco()
{
	name = "SWCO";

	DEFPAR(Swco, 0.2);
	FinalizeParams();
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
void KW_SWOFParams::CheckMonoton()
{
	size_t len = buffer.size();
	std::vector<std::string> aux;
	std::string TRARR = " \t\r";

	double prev, cur;
	std::vector<std::string> items;
	HMMPI::tokenize(buffer[0], items, TRARR, true);
	prev = HMMPI::StoD(items[0]);
	aux.push_back(buffer[0]);
	for (size_t i = 1; i < len; i++)
	{
		HMMPI::tokenize(buffer[i], items, TRARR, true);
		cur = HMMPI::StoD(items[0]);
		if (cur > prev)
		{
			prev = cur;
			aux.push_back(buffer[i]);
		}
	}

	buffer = aux;
}
//------------------------------------------------------------------------------------------
void KW_SWOFParams::WriteToFile(std::string fn)
{
	std::ofstream sw;
	sw.exceptions(std::ios_base::badbit | std::ios_base::failbit);
	try
	{
		sw.open(fn);
		sw << (prop_name + "\n");

		for (size_t i = 0; i < buffer.size(); i++)
			sw << buffer[i];

		sw << "/\n";
		sw.close();
	}
	catch (...)
	{
		if (sw.is_open())
			sw.close();
		throw;
	}
}
//------------------------------------------------------------------------------------------
KW_SWOFParams::KW_SWOFParams()
{
	prop_name = "SWOF";
	div = 16;

	name = "SWOFPARAMS";

	DEFPAR(type, "COREY");
	DEFPAR(Swc, 0.2);
	DEFPAR(Sor, 0.2);
	DEFPAR(krw0, 1);
	DEFPAR(p1, 2);
	DEFPAR(p2, 2);
	DEFPAR(p3, 1);
	DEFPAR(p4, 1);
	DEFPAR(p5, 1);
	DEFPAR(p6, 1);

	FinalizeParams();
	EXPECTED[0] = std::vector<std::string>{"COREY", "CHIERICI", "LET"};
}
//------------------------------------------------------------------------------------------
void KW_SWOFParams::WriteSWOF(std::string fn, std::vector<double> params)
{
	Start_pre();
	IMPORTKWD(swconnate, KW_Swco, "SWCO");
	IMPORTKWD(Pcow, KW_Pcapill, "PCAPILL");
	IMPORTKWD(DIMS, KW_griddims , "GRIDDIMS");
	Finish_pre();

	std::string F = DIMS->satfmt;

	const size_t LEN = ecols-1;		// ecols here is fixed, and cannot be == -1
	assert(params.size() == LEN);

	double __swc = Swc;
	double __sor = Sor;
	if (__swc == -1)
		__swc = params[0];
	if (__sor == -1)
		__sor = params[1];

	double d = (1 - __swc - __sor)/div;
	double Sw = __swc;	// Swcr

	buffer = std::vector<std::string>();
	if (swconnate->Swco < __swc)
		buffer.push_back(HMMPI::stringFormatArr("{0:"+F+"}\t{1:"+F+"}\t{2:"+F+"}\t", std::vector<double>{swconnate->Swco, 0.0, 1.0}) +
						 HMMPI::stringFormatArr("{0:"+F+"}\n", std::vector<double>{Pcow->FuncValExtrapol(0, swconnate->Swco)}));

	for (int i = 0; i <= div; i++)
	{
		double kro, krw;
		CalcVal(Sw, params, krw, kro);
		if (i == 0)
			krw = 0;
		if (i == div)
			kro = 0;
		buffer.push_back(HMMPI::stringFormatArr("{0:"+F+"}\t{1:"+F+"}\t{2:"+F+"}\t", std::vector<double>{Sw, krw, kro}) +
						 HMMPI::stringFormatArr("{0:"+F+"}\n", std::vector<double>{Pcow->FuncValExtrapol(0, Sw)}));
		Sw += d;
	}
	buffer.push_back(HMMPI::stringFormatArr("{0:"+F+"}\t{1:"+F+"}\t{2:"+F+"}\t", std::vector<double>{1.0, 1.0, 0.0}) +
					 HMMPI::stringFormatArr("{0:"+F+"}\n", std::vector<double>{Pcow->FuncValExtrapol(0, 1)}));

	CheckMonoton();
	WriteToFile(fn);
}
//------------------------------------------------------------------------------------------
void KW_SWOFParams::SwcSor(std::vector<double> params, double &s_wc, double &s_or)
{
	s_wc = Swc;
	s_or = Sor;
	if (s_wc == -1)
		s_wc = params[0];
	if (s_or == -1)
		s_or = params[1];
}
//------------------------------------------------------------------------------------------
void KW_SWOFParams::CalcVal(double Sw, std::vector<double> params, double &krw, double &kro)
{
	double __swc = Swc;
	double __sor = Sor;
	double __krw0 = krw0;
	if (__swc == -1)
		__swc = params[0];
	if (__sor == -1)
		__sor = params[1];
	if (__krw0 == -1)
		__krw0 = params[2];

	double S = (Sw - __swc)/(1 - __swc - __sor);
	if (S <= 0)
	{
		krw = 0;
		kro = 1;
		return;
	}
	if (S >= 1)
	{
		krw = __krw0;
		kro = 0;
		return;
	}

	assert (type == "COREY" || type == "CHIERICI" || type == "LET");

	if (type == "COREY")
	{
		double No = p1;
		double Nw = p2;
		if (No == -1)
			No = params[3];
		if (Nw == -1)
			Nw = params[4];

		kro = pow(1-S, No);
		krw = __krw0 * pow(S, Nw);
	}
	else if (type == "CHIERICI")
	{
		double A = p1;
		double L = p2;
		double B = p3;
		double M = p4;
		if (A == -1)
			A = params[3];
		if (L == -1)
			L = params[4];
		if (B == -1)
			B = params[5];
		if (M == -1)
			M = params[6];

		double Rw = (Sw - __swc)/(1 - __sor - Sw);
		kro = exp(-A * pow(Rw, L));
		krw = __krw0 * exp(-B * pow(Rw, -M));
	}
	else if (type == "LET")
	{
		double Lo = p1;
		double Eo = p2;
		double To = p3;
		double Lw = p4;
		double Ew = p5;
		double Tw = p6;
		if (Lo == -1)
			Lo = params[3];
		if (Eo == -1)
			Eo = params[4];
		if (To == -1)
			To = params[5];
		if (Lw == -1)
			Lw = params[6];
		if (Ew == -1)
			Ew = params[7];
		if (Tw == -1)
			Tw = params[8];

		kro = pow(1-S, Lo)/(pow(1-S, Lo) + Eo*pow(S, To));
		krw = __krw0 * pow(S, Lw)/(pow(S, Lw) + Ew*pow(1-S, Tw));
	}
}
//------------------------------------------------------------------------------------------
int KW_SWOFParams::VarCount()
{
	int res = 0;
	if (Swc == -1)
		res++;
	if (Sor == -1)
		res++;
	if (krw0 == -1)
		res++;

	assert (type == "COREY" || type == "CHIERICI" || type == "LET");

	if (type == "COREY")
	{
		if (p1 == -1)
			res++;
		if (p2 == -1)
			res++;
	}
	else if (type == "CHIERICI")
	{
		if (p1 == -1)
			res++;
		if (p2 == -1)
			res++;
		if (p3 == -1)
			res++;
		if (p4 == -1)
			res++;
	}
	else if (type == "LET")
	{
		if (p1 == -1)
			res++;
		if (p2 == -1)
			res++;
		if (p3 == -1)
			res++;
		if (p4 == -1)
			res++;
		if (p5 == -1)
			res++;
		if (p6 == -1)
			res++;
	}

	return res;
}
//------------------------------------------------------------------------------------------
std::vector<double> KW_SWOFParams::VarParams(std::vector<double> params_all, int i0, int i1)
{
	Start_pre();
	IMPORTKWD(limits, KW_limits, "LIMITS");
	Finish_pre();

	std::vector<double> res(ecols-1);		// ecols here is fixed, and cannot be == -1
	int cur = i0;

	if (Swc == -1)
	{
		res[0] = params_all[cur] * limits->norm[cur];
		if (limits->func[cur] == "EXP")
			res[0] = pow(10, res[0]);
		else if (limits->func[cur] == "LIN")
			res[0] += limits->dh[cur];
		cur++;
	}
	if (Sor == -1)
	{
		res[1] = params_all[cur] * limits->norm[cur];
		if (limits->func[cur] == "EXP")
			res[1] = pow(10, res[1]);
		else if (limits->func[cur] == "LIN")
			res[1] += limits->dh[cur];
		cur++;
	}
	if (krw0 == -1)
	{
		res[2] = params_all[cur] * limits->norm[cur];
		if (limits->func[cur] == "EXP")
			res[2] = pow(10, res[2]);
		else if (limits->func[cur] == "LIN")
			res[2] += limits->dh[cur];
		cur++;
	}

	assert (type == "COREY" || type == "CHIERICI" || type == "LET");

	if (type == "COREY")
	{
		if (p1 == -1)
		{
			res[3] = params_all[cur] * limits->norm[cur];
			if (limits->func[cur] == "EXP")
				res[3] = pow(10, res[3]);
			else if (limits->func[cur] == "LIN")
				res[3] += limits->dh[cur];
			cur++;
		}
		if (p2 == -1)
		{
			res[4] = params_all[cur] * limits->norm[cur];
			if (limits->func[cur] == "EXP")
				res[4] = pow(10, res[4]);
			else if (limits->func[cur] == "LIN")
				res[4] += limits->dh[cur];
			cur++;
		}
	}
	else if (type == "CHIERICI")
	{
		if (p1 == -1)
		{
			res[3] = params_all[cur] * limits->norm[cur];
			if (limits->func[cur] == "EXP")
				res[3] = pow(10, res[3]);
			else if (limits->func[cur] == "LIN")
				res[3] += limits->dh[cur];
			cur++;
		}
		if (p2 == -1)
		{
			res[4] = params_all[cur] * limits->norm[cur];
			if (limits->func[cur] == "EXP")
				res[4] = pow(10, res[4]);
			else if (limits->func[cur] == "LIN")
				res[4] += limits->dh[cur];
			cur++;
		}
		if (p3 == -1)
		{
			res[5] = params_all[cur] * limits->norm[cur];
			if (limits->func[cur] == "EXP")
				res[5] = pow(10, res[5]);
			else if (limits->func[cur] == "LIN")
				res[5] += limits->dh[cur];
			cur++;
		}
		if (p4 == -1)
		{
			res[6] = params_all[cur] * limits->norm[cur];
			if (limits->func[cur] == "EXP")
				res[6] = pow(10, res[6]);
			else if (limits->func[cur] == "LIN")
				res[6] += limits->dh[cur];
			cur++;
		}
	}
	else if (type == "LET")
	{
		if (p1 == -1)
		{
			res[3] = params_all[cur] * limits->norm[cur];
			if (limits->func[cur] == "EXP")
				res[3] = pow(10, res[3]);
			else if (limits->func[cur] == "LIN")
				res[3] += limits->dh[cur];
			cur++;
		}
		if (p2 == -1)
		{
			res[4] = params_all[cur] * limits->norm[cur];
			if (limits->func[cur] == "EXP")
				res[4] = pow(10, res[4]);
			else if (limits->func[cur] == "LIN")
				res[4] += limits->dh[cur];
			cur++;
		}
		if (p3 == -1)
		{
			res[5] = params_all[cur] * limits->norm[cur];
			if (limits->func[cur] == "EXP")
				res[5] = pow(10, res[5]);
			else if (limits->func[cur] == "LIN")
				res[5] += limits->dh[cur];
			cur++;
		}
		if (p4 == -1)
		{
			res[6] = params_all[cur] * limits->norm[cur];
			if (limits->func[cur] == "EXP")
				res[6] = pow(10, res[6]);
			else if (limits->func[cur] == "LIN")
				res[6] += limits->dh[cur];
			cur++;
		}
		if (p5 == -1)
		{
			res[7] = params_all[cur] * limits->norm[cur];
			if (limits->func[cur] == "EXP")
				res[7] = pow(10, res[7]);
			else if (limits->func[cur] == "LIN")
				res[7] += limits->dh[cur];
			cur++;
		}
		if (p6 == -1)
		{
			res[8] = params_all[cur] * limits->norm[cur];
			if (limits->func[cur] == "EXP")
				res[8] = pow(10, res[8]);
			else if (limits->func[cur] == "LIN")
				res[8] += limits->dh[cur];
			cur++;
		}
	}

	assert(cur == i1);

	return res;
}
//------------------------------------------------------------------------------------------
std::vector<int> KW_SWOFParams::SwcSorIndex(int i0)
{
	std::vector<int> res(2);
	res[0] = res[1] = -1;
	int cur = i0;

	if (Swc == -1)
	{
		res[0] = cur;
		cur++;
	}
	if (Sor == -1)
		res[1] = cur;

	return res;
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_SGOFParams::KW_SGOFParams() : KW_SWOFParams()
{
	prop_name = "SGOF";
	name = "SGOFPARAMS";

	Sor = 0.8;	// = Sgmax

	NAMES = std::vector<std::string>{"type", "Sgcr", "Sgmax", "krg0", "p1", "p2", "p3", "p4", "p5", "p6"};
}
//------------------------------------------------------------------------------------------
void KW_SGOFParams::WriteSWOF(std::string fn, std::vector<double> params)
{
	Start_pre();
	IMPORTKWD(Pcow, KW_Pcapill, "PCAPILL");
	IMPORTKWD(DIMS, KW_griddims, "GRIDDIMS");
	Finish_pre();

	std::string F = DIMS->satfmt;

	const size_t LEN = ecols-1;		// ecols here is fixed, and cannot be == -1
	assert(params.size() == LEN);

	double __sgcr = Swc;
	double __sgmax = Sor;
	if (__sgcr == -1)
		__sgcr = params[0];
	if (__sgmax == -1)
		__sgmax = params[1];

	int div1 = int(div * __sgcr / __sgmax);
	if (div1 <= 0)
		div1 = 1;
	if (div1 >= div)
		div1 = div-1;
	int div2 = div - div1;
	double d1 = __sgcr/div1;
	double d2 = (__sgmax - __sgcr)/div2;

	buffer = std::vector<std::string>();
	double Sg = 0;
	for (int i = 0; i <= div1; i++)
	{
		double kro, krg;
		CalcVal(Sg, params, krg, kro);
		krg = 0;
		buffer.push_back(HMMPI::stringFormatArr("{0:"+F+"}\t{1:"+F+"}\t{2:"+F+"}\t", std::vector<double>{Sg, krg, kro}) +
						 HMMPI::stringFormatArr("{0:"+F+"}\n", std::vector<double>{Pcow->FuncValExtrapol(1, Sg)}));
		Sg += d1;
	}

	Sg = __sgcr + d2;
	for (int i = 1; i <= div2; i++)
	{
		double kro, krg;
		CalcVal(Sg, params, krg, kro);
		if (i == div2)
			kro = 0;
		buffer.push_back(HMMPI::stringFormatArr("{0:"+F+"}\t{1:"+F+"}\t{2:"+F+"}\t", std::vector<double>{Sg, krg, kro}) +
						 HMMPI::stringFormatArr("{0:"+F+"}\n", std::vector<double>{Pcow->FuncValExtrapol(1, Sg)}));
		Sg += d2;
	}

	CheckMonoton();
	WriteToFile(fn);
}
//------------------------------------------------------------------------------------------
void KW_SGOFParams::CalcVal(double Sg, std::vector<double> params, double &krg, double &kro)
{
	// Swc <-> Sgcr
	// Sor <-> Sgmax
	double __sgcr = Swc;
	double __sgmax = Sor;
	double __krg0 = krw0;
	if (__sgcr == -1)
		__sgcr = params[0];
	if (__sgmax == -1)
		__sgmax = params[1];
	if (__krg0 == -1)
		__krg0 = params[2];

	double S_gas = (Sg - __sgcr)/(__sgmax - __sgcr);	// ...
	double S_oil = Sg/__sgmax;

	assert (type == "COREY" || type == "CHIERICI" || type == "LET");

	if (type == "COREY")
	{
		double No = p1;
		double Nw = p2;
		if (No == -1)
			No = params[3];
		if (Nw == -1)
			Nw = params[4];

		kro = pow(1-S_oil, No);
		krg = __krg0 * pow(S_gas, Nw);
	}
	else if (type == "CHIERICI")
	{
		double A = p1;
		double L = p2;
		double B = p3;
		double M = p4;
		if (A == -1)
			A = params[3];
		if (L == -1)
			L = params[4];
		if (B == -1)
			B = params[5];
		if (M == -1)
			M = params[6];

		double Rw_gas = (Sg - __sgcr)/(__sgmax - Sg);
		double Rw_oil = Sg/(__sgmax - Sg);
		kro = exp(-A * pow(Rw_oil, L));
		krg = __krg0 * exp(-B * pow(Rw_gas, -M));
	}
	else if (type == "LET")
	{
		double Lo = p1;
		double Eo = p2;
		double To = p3;
		double Lw = p4;
		double Ew = p5;
		double Tw = p6;
		if (Lo == -1)
			Lo = params[3];
		if (Eo == -1)
			Eo = params[4];
		if (To == -1)
			To = params[5];
		if (Lw == -1)
			Lw = params[6];
		if (Ew == -1)
			Ew = params[7];
		if (Tw == -1)
			Tw = params[8];

		kro = pow(1-S_oil, Lo)/(pow(1-S_oil, Lo) + Eo*pow(S_oil, To));
		krg = __krg0 * pow(S_gas, Lw)/(pow(S_gas, Lw) + Ew*pow(1-S_gas, Tw));
	}

	if (S_gas <= 0)
		krg = 0;
	if (S_gas >= 1)
		krg = __krg0;

	if (S_oil <= 0)
		kro = 1;
	if (S_oil >= 1)
		kro = 0;
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_griddims::KW_griddims()
{
	name = "GRIDDIMS";

	DEFPAR(Nx, 100);
	DEFPAR(Ny, 100);
	DEFPAR(Nz, 1);
	DEFPAR(krig_prop, "PERMX");
	DEFPAR(krig_file, "_PERMX");
	DEFPAR(swof_file, "_SWOF");
	DEFPAR(sgof_file, "_SGOF");
	DEFPAR(wght, "PORV");
	DEFPAR(satfmt, "%.8f");

	FinalizeParams();
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_griddimens::KW_griddimens()
{
	name = "GRIDDIMENS";

	DEFPAR(Nx, 1);
	DEFPAR(Ny, 1);
	DEFPAR(Nz, 1);
	DEFPAR(X0, 0.0);
	DEFPAR(Y0, 0.0);
	DEFPAR(grid_Y_axis, "POS");

	FinalizeParams();
	EXPECTED[5] = std::vector<std::string>{"POS", "NEG"};
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_satsteps::KW_satsteps()
{
	name = "SATSTEPS";
	NAMES[0] = "t";
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_delta::KW_delta()
{
	name = "DELTA";
	NAMES[0] = "delta";
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_incfiles::KW_incfiles()
{
	name = "INCFILES";
	DEFPARMULT(file);
	DEFPARMULT(mod);
	DEFPARMULT(pcount);

	FinalizeParams();
}
//------------------------------------------------------------------------------------------
void KW_incfiles::UpdateParams() noexcept
{
	size_t count = file.size();
	Buffer = std::vector<std::string>(count);

	int successful = 0;		// counts successful loads for report
	for (size_t c = 0; c < count; c++)
	{
		std::string fread = this->CWD + "/" + file[c];
		std::string buffer = "";

		std::ifstream fR;
		fR.exceptions(std::ios_base::badbit);
		try
		{
			CheckFileOpen(fread);
			fR.open(fread);

			while (!fR.eof())
			{
				std::string line;
				getline(fR, line);
				buffer += line + "\n";
			}
			fR.close();

			Buffer[c] = buffer;
			successful++;
		}
		catch (...)
		{
			K->AppText(HMMPI::stringFormatArr("ОШИБКА чтения файла {0:%s}\n", "ERROR reading file {0:%s}\n", fread));
			AddState(HMMPI::stringFormatArr("Невозможно прочесть файл {0:%s}", "Failed to read file {0:%s}", fread));
			K->TotalErrors++;
			if (fR.is_open())
				fR.close();
		}
	}
	K->AppText(HMMPI::stringFormatArr("Загружено файлов: {0:%d}\n", "Successfully loaded: {0:%d}\n", successful));
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
void KW_templates::check_fnames() noexcept
{
	// upper case versions of filenames
	std::vector<std::string> OrigFile(orig_file.size());
	std::vector<std::string> WorkFile(work_file.size());
	for (size_t i = 0; i < OrigFile.size(); i++)
	{
		OrigFile[i] = HMMPI::ToUpper(orig_file[i]);
		WorkFile[i] = HMMPI::ToUpper(work_file[i]);
	}

	// check identical file names in orig and work
	for (size_t i = 0; i < OrigFile.size(); i++)
		if (OrigFile[i] == WorkFile[i])
		{
			SilentError(HMMPI::stringFormatArr("Одинаковые orig_file и work_file в строке {0:%d} (регистр не учитывается)",
								   	   	   	   "Identical orig_file and work_file in line {0:%d} (case insensitive)", i+1));
			return;
		}
	// check duplicates
	std::string dup;
	if (HMMPI::FindDuplicate(OrigFile, dup))
	{
		SilentError(HMMPI::stringFormatArr("Повторяется {0:%s} в столбце orig_file (регистр не учитывается)",
							   	   	   	   "Repeated {0:%s} in column orig_file (case insensitive)", dup));
		return;
	}
	if (HMMPI::FindDuplicate(WorkFile, dup))
	{
		SilentError(HMMPI::stringFormatArr("Повторяется {0:%s} в столбце work_file (регистр не учитывается)",
							   	   	   	   "Repeated {0:%s} in column work_file (case insensitive)", dup));
		return;
	}
	std::vector<std::string> OrigAndWork = OrigFile;
	HMMPI::VecAppend(OrigAndWork, WorkFile);
	if (HMMPI::FindDuplicate(OrigAndWork, dup))
	{
		SilentError(HMMPI::stringFormatArr("Повторяется {0:%s} в orig_file/work_file (регистр не учитывается)",
							   	   	   	   "Repeated {0:%s} in orig_file/work_file (case insensitive)", dup));
		return;
	}

	// check if *.DAT[A] is present
	std::vector<int> data_ind_orig = find_str_in_vec(OrigFile, ".DAT");		// was find_end_str_in_vec(..., ".DATA") before!
	std::vector<int> data_ind_work = find_str_in_vec(WorkFile, ".DAT");		// was find_end_str_in_vec(..., ".DATA") before!
	if (data_ind_orig.size() != 1 || data_ind_work.size() != 1)
		SilentError(HMMPI::MessageRE("DAT[A]-файл должен быть указан ровно один раз в orig_file, и соответственно в work_file",
							   	   	 "DAT[A]-file should be specified exactly once in orig_file, and correspondingly in work_file"));
	else if (data_ind_orig[0] != data_ind_work[0])
		SilentError(HMMPI::MessageRE("DAT[A]-файл в orig_file должен соответствовать DAT[A]-файлу в work_file",
							   	   	 "DAT[A]-file in orig_file should correspond to DAT[A]-file in work_file"));
	else
		data_file_ind = data_ind_work[0];

#ifndef TEMPLATES_KEEP_NO_ASCII
	// check that work_file's contain only $RANK parameter, if any
	std::vector<std::string> tags;
	for (size_t i = 0; i < work_file.size(); i++)
		HMMPI::VecAppend(tags, HMMPI::stringExtractTags(work_file[i]));
	for (size_t i = 0; i < tags.size(); i++)
		if (tags[i] != "RANK")
			SilentError(HMMPI::stringFormatArr("В именах файлов work_file допустим только параметр $RANK (найден ${0:%s})",
											   "In work_file file names only $RANK parameter is allowed (found ${0:%s})", tags[i]));

	// check that every work file has $RANK in parallel mode
	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	if (size > 1 && find_str_in_vec(work_file, "$RANK").size() < work_file.size())		// note: case-sensitive search
		SilentError(HMMPI::MessageRE("В параллельном режиме MPI каждый work_file должен содержать $RANK",
							   	   	 "In parallel MPI mode every work_file should contain $RANK"));
#endif
}
//------------------------------------------------------------------------------------------
void KW_templates::PrintParams() noexcept
{
	if (K->verbosity - dec_verb >= 1)		// report all parameter values
	{
		int maxname = 0;					// find the max length of orig_file[i]
		for (size_t i = 0; i < orig_file.size(); i++)
			if ((int)orig_file[i].length() > maxname)
				maxname = orig_file[i].length();

		char *buff = new char[HMMPI::BUFFSIZE + 2*maxname];
		std::string MSG = HMMPI::MessageRE("Текущие значения:\n", "Current values:\n");
		for (size_t i = 0; i < orig_file.size(); i++)
		{
#ifndef TEMPLATES_KEEP_NO_ASCII
			sprintf(buff, "%30.200s %s %-.200s (keep %s)\n", orig_file[i].c_str(), mode[i].c_str(), work_file[i].c_str(), keep[i].c_str());
#else
			sprintf(buff, "%*.*s %s %-.200s\n", maxname, maxname, orig_file[i].c_str(), mode[i].c_str(), work_file[i].c_str());
#endif
			MSG += buff;
		}
		K->AppText(MSG);
		delete [] buff;
	}
}
//------------------------------------------------------------------------------------------
KW_templates::KW_templates()
{
	name = "TEMPLATES";
	dec_verb = -1;
	data_file_ind = -1;
	DEFPARMULT(orig_file);
	DEFPARMULT(mode);
	DEFPARMULT(work_file);
#ifndef TEMPLATES_KEEP_NO_ASCII
	DEFPARMULT(keep);
#endif

	FinalizeParams();

	EXPECTED[1] = std::vector<std::string>{">", "-"};
#ifndef TEMPLATES_KEEP_NO_ASCII
	EXPECTED[3] = std::vector<std::string>{"FIRST", "NONE", "ALL"};
#endif
}
//------------------------------------------------------------------------------------------
void KW_templates::UpdateParams() noexcept
{
	check_fnames();
	if (GetState() != "")
		return;

	Buffer = std::vector<std::string>(orig_file.size());
	int count = 0;							// counts file loads for report
	for (size_t c = 0; c < orig_file.size(); c++)
	{
		std::string fread = this->CWD + "/" + orig_file[c];
		std::string buffer = "";

		std::ifstream fR;
		fR.exceptions(std::ios_base::badbit);
		try
		{
			if (mode[c] == ">")
			{
				CheckFileOpen(fread);
				fR.open(fread);

				while (!fR.eof())
				{
					std::string line;
					getline(fR, line);
					buffer += line + "\n";
				}
				fR.close();

				Buffer[c] = std::move(buffer);
				count++;
			}
		}
		catch (...)
		{
			K->AppText(HMMPI::stringFormatArr("ОШИБКА чтения файла {0:%s}\n", "ERROR reading file {0:%s}\n", fread));
			AddState(HMMPI::stringFormatArr("Невозможно прочесть файл {0:%s}", "Failed to read file {0:%s}", fread));
			K->TotalErrors++;
			if (fR.is_open())
				fR.close();
		}
	}

#ifdef TEMPLATES_KEEP_NO_ASCII
	keep = std::vector<std::string>(orig_file.size(), "FIRST");
#endif

	K->AppText(HMMPI::stringFormatArr("Загружено файлов: {0:%d}\n", "Loaded files: {0:%d}\n", count));
}
//------------------------------------------------------------------------------------------
std::string KW_templates::WriteFiles(HMMPI::TagPrintfMap &par) const
{
	DECLKWD(simcmd, KW_simcmd, "SIMCMD");

	int count = 0;		// for reporting
	char buff[HMMPI::BUFFSIZE], buff_rus[HMMPI::BUFFSIZE], buff_small[HMMPI::BUFFSIZE];
	sprintf(buff, "%-30.300s\t%-10s\t%-10s\n", "Replaced in", "params", "filenames");	// make message header
	sprintf(buff_rus, "%-36.300s\t%-15s\t%-17s\n", "Замен в", "парам.", "имён ф-ов");
	std::string res = HMMPI::MessageRE(buff_rus, buff);

	std::set<std::string> tags_left = par.get_tag_names();			// "par" will only be modified below in SetModPath(), which only affects MOD and PATH, which are of no interest in get_tag_names()

	std::string DataFile = HMMPI::stringTagPrintf(work_file[data_file_ind], par, count, tags_left);			// substitute params ($RANK) in template data-file
	std::string DataFile_nopath = HMMPI::getFile(DataFile);
	std::string path_DataFile = HMMPI::getCWD(DataFile);
	if (path_DataFile == "")
		path_DataFile = ".";
	par.SetModPath(DataFile_nopath.substr(0, DataFile_nopath.find_last_of(".")), path_DataFile);			// set MOD (without file extension) and PATH

	work_file_subst = work_file;
#ifndef TEMPLATES_KEEP_NO_ASCII
	for (auto &f : work_file_subst)
		f = HMMPI::stringTagPrintf(f, par, count, tags_left);		// substitute params ($RANK) in work file names
#endif

	// substitute filenames and params in SIMCMD
	for (size_t i = 0; i < simcmd->cmd.size(); i++)
	{
		sprintf(buff_small, "SIMCMD-%zu", i+1);
		try
		{
			int countfn = 0;
			simcmd->cmd_work[i] = simcmd->cmd[i];
			simcmd->cmd_work[i] = HMMPI::ReplaceArr(std::move(simcmd->cmd_work[i]), orig_file, work_file_subst, &countfn);
			simcmd->cmd_work[i] = HMMPI::stringTagPrintf(simcmd->cmd_work[i], par, count, tags_left);

			sprintf(buff, "%-30.300s\t  %-8d\t    %-6d\n", buff_small, count, countfn);
			res += buff;
		}
		catch (std::exception &e)
		{
			throw HMMPI::EObjFunc((std::string)e.what() + " (" + buff_small + ")");
		}
	}

	// substitute filenames and params in files (Buffer[i]), write files
	for (size_t i = 0; i < Buffer.size(); i++)
	{
		std::ofstream sw;
		sw.exceptions(std::ios_base::badbit | std::ios_base::failbit);
		try
		{
			int countfn = 0;
			count = 0;

			sw.open(work_file_subst[i]);
			if (mode[i] == ">")
			{
				std::string BufferSubst = HMMPI::ReplaceArr(Buffer[i], orig_file, work_file_subst, &countfn);
				sw << HMMPI::stringTagPrintf(BufferSubst, par, count, tags_left);
			}
			sw.close();

			sprintf(buff, "%-30.300s\t  %-8d\t    %-6d\n", orig_file[i].c_str(), count, countfn);
			res += buff;
		}
		catch (std::exception &e)
		{
			if (sw.is_open())
				sw.close();
			throw HMMPI::EObjFunc((std::string)e.what() + " (" + orig_file[i] + ")");
		}
	}

	if (tags_left.size() > 0)		// unused tags remain
	{
		std::string msg = "";
		for (const std::string &s : tags_left)
			if (msg.size() > 0)
				msg += ", " + s;
			else
				msg += s;

		K->TotalWarnings++;
		K->AppText(HMMPI::stringFormatArr("ПРЕДУПРЕЖДЕНИЕ: некоторые параметры не были использованы: {0:%s}\n",
										  "WARNING: some parameters were not used: {0:%s}\n", msg));
	}

	return res;
}
//------------------------------------------------------------------------------------------
void KW_templates::ClearFiles()						// _NOTE_ for the case "TEMPLATES_KEEP_NO_ASCII" the files "work_file_subst" are same on all ranks,
{													// so for keep[i] == "FIRST" && rank > 0 those ranks would erase the files;
	int rank;										// However this should not normally happen, if PMEclipse is created on MPI_COMM_WORLD,
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);			// and ClearFiles() is only called on PMEclipse-comm-RANKS-0

	for (size_t i = 0; i < work_file_subst.size(); i++)
		if ((keep[i] == "NONE")||(keep[i] == "FIRST" && rank > 0))
			remove(work_file_subst[i].c_str());
}
//------------------------------------------------------------------------------------------
void KW_templates::ClearFilesEcl()											// TODO for tNav this still doesn't work
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	assert(data_file_ind >= 0);
	if ((keep[data_file_ind] == "NONE")||(keep[data_file_ind] == "FIRST" && rank > 0))
	{
		try
		{
			std::string DataFile = DataFileSubst();
			DataFile = DataFile.substr(0, DataFile.find_last_of("."));		// strip file extension

			std::vector<std::string> extensions = {".EGRID", ".GRID", ".INIT", ".INSPEC", ".RSSPEC", ".SMSPEC", ".UNRST", ".UNSMRY", ".PRT", ".DBG", ".ECLEND", ".MSG", ".RSM"};
			for (const std::string &ext : extensions)
				remove((DataFile + ext).c_str());
		}
		catch (...)
		{
		}
	}
}
//------------------------------------------------------------------------------------------
std::string KW_templates::DataFileSubst() const
{
	return work_file_subst[data_file_ind];
}
//------------------------------------------------------------------------------------------
std::vector<int> KW_templates::find_str_in_vec(const std::vector<std::string> &vec, const std::string &needle) noexcept
{
	std::vector<int> res;
	for (size_t i = 0; i < vec.size(); i++)
		if (vec[i].find(needle) != std::string::npos)
			res.push_back(i);

	return res;
}
//------------------------------------------------------------------------------------------
std::vector<int> KW_templates::find_end_str_in_vec(const std::vector<std::string> &vec, const std::string &needle) noexcept
{
	std::vector<int> res;
	for (size_t i = 0; i < vec.size(); i++)
		if (vec[i].substr(vec[i].length() - needle.length(), needle.length()) == needle)
			res.push_back(i);

	return res;
}
//------------------------------------------------------------------------------------------
#ifdef TEMPLATES_KEEP_NO_ASCII
void KW_templates::set_keep(std::string k)
{
	for (auto &x : keep)
		x = k;
}
#endif
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_eclvectors::KW_eclvectors()
{
	name = "ECLVECTORS";
	DEFPARMULT(WGname);
	DEFPARMULT(vect);
	DEFPARMULT(sigma);
	DEFPARMULT(R);
	DEFPARMULT(func);

	FinalizeParams();

	EXPECTED[4] = std::vector<std::string>{"SPHER", "EXP", "GAUSS", "MATERN"};
}
//------------------------------------------------------------------------------------------
KW_eclvectors::~KW_eclvectors()
{
	for (auto i : corr)
		delete i;
}
//------------------------------------------------------------------------------------------
void KW_eclvectors::UpdateParams() noexcept
{
	DECLKWD(textsmry, KW_textsmry, "TEXTSMRY");

	int count = 0;			// counts 'R' replacements
	size_t L = R.size();
	corr = std::vector<HMMPI::Func1D*>(L);
	vecs = std::vector<HMMPI::SimSMRY::pair>(L);

	for (size_t i = 0; i < L; i++)
	{
		WGname[i] = HMMPI::ToUpper(WGname[i]);
		vect[i] = HMMPI::ToUpper(vect[i]);
		vecs[i].first = WGname[i];
		vecs[i].second = vect[i];

		if (R[i] <= 0)
		{
			R[i] = 0.01;
			count++;
		}

		corr[i] = HMMPI::Func1D_factory::New(func[i]);
	}

	HMMPI::SimSMRY::pair dup;
	if (HMMPI::FindDuplicate(vecs, dup))
		SilentError((std::string)HMMPI::MessageRE("Найден повторяющийся вектор ", "Found duplicate vector ") + dup.first + " " + dup.second);

	if (count > 0)
		K->AppText(HMMPI::stringFormatArr("Поскольку R <= 0 не допустим, он был заменен на 0.01 в {0:%d} строк(ах)\n",
										  "Since R <= 0 is not allowed, it was replaced by 0.01 in {0:%d} line(s)\n", count));
	if (textsmry->GetState() == "")
		textsmry->SetState(HMMPI::MessageRE("TEXTSMRY должно быть перезагружено после чтения ECLVECTORS\n", "TEXTSMRY should be reloaded after reading ECLVECTORS\n"));
}
//------------------------------------------------------------------------------------------
std::string KW_eclvectors::SigmaInfo(const std::string &wgname, const std::string &keyword) const
{
	std::string res = "N/A";
	for (size_t i = 0; i < WGname.size(); i++)
		if (WGname[i] == wgname && vect[i] == keyword)
		{
			res = HMMPI::stringFormatArr("{0:%g}", std::vector<double>{sigma[i]});
			break;
		}

	return res;
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_regressConstr::KW_regressConstr()
{
	name = "REGRESSCONSTR";

	deriv_ref = HMMPI::Vector2<int>(std::vector<int>{-1, -99, -99, 100, -99, -99,  1,   2,  -99, -99, -99,  9, -99, -99,
													-99,  -1, -99, -99, 101, -99,  0,  -99,  2,  -99, -99, -99,  9, -99,
													-99, -99,  -1, -99, -99, 102, -99,  0,   1,  -99, -99, -99, -99,  9,
													-99, -99, -99, -99, -99, -99, -99, -99, -99, -1,  109,  0,   1,   2}, 4, 14);
	DEFPAR(dP, 0);
	DEFPAR(dSw, 0);
	DEFPAR(dSg, 0);
	DEFPAR(dRs, 0);

	FinalizeParams();
}
//------------------------------------------------------------------------------------------
bool KW_regressConstr::hasQuadr(int ind)
{
	Start_pre();
	IMPORTKWD(quadr, KW_regressquadr, "REGRESSQUADR");
	IMPORTKWD(Rs, KW_regressRs, "REGRESSRS");
	Finish_pre();

	bool res;
	assert(ind >= 0 && ind <= 3);

	if (ind == 0)		// P
		res = (quadr->P2 || quadr->PSw || quadr->PSg || Rs->RsP);
	else if (ind == 1)	// Sw
		res = (quadr->Sw2 || quadr->PSw || quadr->SwSg || Rs->RsSw);
	else if (ind == 2)	// Sg
		res = (quadr->Sg2 || quadr->PSg || quadr->SwSg || Rs->RsSg);
	else if (ind == 3)	// Rs
		res = (Rs->Rs2 || Rs->RsP || Rs->RsSw || Rs->RsSg);

	return res;
}
//------------------------------------------------------------------------------------------
void KW_regressConstr::fill_ind()
{
	Start_pre();
	IMPORTKWD(quadr, KW_regressquadr, "REGRESSQUADR");
	IMPORTKWD(Rs, KW_regressRs, "REGRESSRS");
	IMPORTKWD(gas, KW_gas, "GAS");
	Finish_pre();

	if (gas->on == "OFF" &&
		(quadr->Sg2 || quadr->PSg || quadr->SwSg || Rs->RsSg))
		throw HMMPI::EObjFunc("(eng)", "Gas is OFF, but REGRESSQUADR, REGRESSRS contain gas terms");

	std::vector<int> aux = quadr->IndActive(gas->on == "ON");
	size_t len = aux.size() + 2;
	if (gas->on == "ON")
		len++;
	if (Rs->Rs)
		len++;
	if (Rs->Rs2)
		len++;
	if (Rs->RsP)
		len++;
	if (Rs->RsSw)
		len++;
	if (Rs->RsSg)
		len++;

	xi_ind = std::vector<int>(len);
	var_ind = std::vector<int>(14);
	for (int i = 0; i < 14; i++)
		var_ind[i] = -1;

	xi_ind[0] = 0;	var_ind[0] = 0;				// P
	xi_ind[1] = 1;	var_ind[1] = 1;				// Sw

	int c = 2;
	if (gas->on == "ON")						// Sg
	{
		xi_ind[c] = 2;	var_ind[2] = c;	c++;
	}

	if (quadr->P2)								// P2, Sw2, Sg2
	{
		xi_ind[c] = 3;	var_ind[3] = c;	c++;
	}
	if (quadr->Sw2)
	{
		xi_ind[c] = 4;	var_ind[4] = c;	c++;
	}
	if (quadr->Sg2)
	{
		xi_ind[c] = 5;	var_ind[5] = c;	c++;
	}

	if (quadr->PSw)								// PSw, PSg, SwSg
	{
		xi_ind[c] = 6;	var_ind[6] = c;	c++;
	}
	if (quadr->PSg)
	{
		xi_ind[c] = 7;	var_ind[7] = c;	c++;
	}
	if (quadr->SwSg)
	{
		xi_ind[c] = 8;	var_ind[8] = c;	c++;
	}

	if (Rs->Rs)									// Rs...
	{
		xi_ind[c] = 9;	var_ind[9] = c;	c++;
	}
	if (Rs->Rs2)
	{
		xi_ind[c] = 10;	var_ind[10] = c;	c++;
	}

	if (Rs->RsP)
	{
		xi_ind[c] = 11;	var_ind[11] = c;	c++;
	}
	if (Rs->RsSw)
	{
		xi_ind[c] = 12;	var_ind[12] = c;	c++;
	}
	if (Rs->RsSg)
	{
		xi_ind[c] = 13;	var_ind[13] = c;	c++;
	}
}
//------------------------------------------------------------------------------------------
std::vector<double> KW_regressConstr::getConstr(int ind, std::vector<double> xi, double a0)
{
	size_t len = xi.size();
	assert(len == xi_ind.size());

	std::vector<double> res(len);
	for (size_t i = 0; i < len; i++)
	{
		int CODE = deriv_ref(ind, xi_ind[i]);
		if (CODE == -99)
			res[i] = 0;
		else if (CODE == -1)
			res[i] = a0;
		else if (CODE >= 100)
		{
			int I0 = var_ind[CODE%100];
			res[i] = 2 * xi[I0];
		}
		else
		{
			int I0 = var_ind[CODE];
			res[i] = xi[I0];
		}

	}

	return res;
}
//------------------------------------------------------------------------------------------
std::string KW_regressConstr::getConstrStr(int ind)
{
	size_t len = xi_ind.size();

	std::vector<std::string> aux = std::vector<std::string>{"P", "Sw", "Sg",
															"P2", "Sw2", "Sg2", "PSw", "PSg", "SwSg",
															"Rs", "Rs2", "RsP", "RsSw", "RsSg"};
	std::vector<std::string> res(len);
	for (size_t i = 0; i < len; i++)
	{
		int CODE = deriv_ref(ind, xi_ind[i]);
		if (CODE == -99)
			res[i] = "0";
		else if (CODE == -1)
			res[i] = "A0";
		else if (CODE >= 100)
			res[i] = "2*" + aux[CODE%100] + "*A0";
		else
			res[i] = aux[CODE] + "*A0";
	}

	std::string res_str = "";
	for (size_t i = 0; i < len; i++)
		res_str += "\t" + res[i];

	return res_str;
}
//------------------------------------------------------------------------------------------
std::vector<double> KW_regressConstr::getConstrFinal(int ind, std::vector<double> xi, double a0)
{
	std::vector<double> res;
	if (*(int*)DATA[ind] != 0)
		res = getConstr(ind, xi, a0);

	if (*(int*)DATA[ind] < 0)
	{
		for (size_t i = 0; i < res.size(); i++)
			res[i] = -res[i];
	}

	bool non_zero = false;
	if (res.size() != 0)
		for (size_t i = 0; i < res.size(); i++)
			if (res[i] != 0)
			{
				non_zero = true;
				break;
			}

	if (non_zero)
		return res;
	else
		return std::vector<double>();
}
//------------------------------------------------------------------------------------------
std::vector<double> KW_regressConstr::getInitPoint(double A0sign, const std::vector<int> &A_V)
{
	size_t len = 0;
	for (size_t i = 0; i < A_V.size(); i++)
		if (A_V[i] != -1)
			len++;

	std::vector<double> res(len);
	for (size_t i = 0; i < len; i++)
		res[i] = 0;

	// +/- 1 should depend on A0sign
	for (int i = 0; i < 3; i++)
	{
		if (var_ind[i] != -1)	// variable is present in global list
		{
			if (A_V[var_ind[i]] != -1)	// variable remains active
			{
				res[A_V[var_ind[i]]] = A0sign;
				if (*(int*)DATA[i] < 0)
					res[A_V[var_ind[i]]] = -A0sign;
			}
		}
	}
	
	if (var_ind[9] != -1)
	{
		if (A_V[var_ind[9]] != -1)
		{
			res[A_V[var_ind[9]]] = A0sign;
			if (*(int*)DATA[3] < 0)
				res[A_V[var_ind[9]]] = -A0sign;
		}
	}

	return res;
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_plotparams::KW_plotparams()
{
	name = "PLOTPARAMS";

	DEFPAR(Nx, 10);
	DEFPAR(Ny, 10);
	DEFPAR(delta, 0);

	FinalizeParams();
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_optimization::KW_optimization()
{
	name = "OPTIMIZATION";

	DEFPAR(algorithm, "CMAES");
	DEFPAR(fin_diff, "OH1");
	DEFPAR(nonlin_solver, "FIXEDPOINT");
	DEFPAR(maxit, 10000);
	DEFPAR(maxJacIters, 1);
	DEFPAR(maa, 0);
	DEFPAR(epsG, 1e-10);
	DEFPAR(epsF, 0);
	DEFPAR(epsX, 0);
	DEFPAR(R, 0);
	DEFPAR(restr, "CUBE");

	FinalizeParams();
	EXPECTED[0] = std::vector<std::string>{"CMAES", "LM"};
	EXPECTED[1] = std::vector<std::string>{"OH1", "OH2", "OH4", "OH8"};
	EXPECTED[2] = std::vector<std::string>{"FIXEDPOINT", "NEWTON", "GNEWTON", "HYBRIDPOWELL", "KIN_NEWTON", "KIN_NEWTON_LS", "KIN_FP", "KIN_PICARD"};
	EXPECTED[10] = std::vector<std::string>{"CUBE", "SPHERE"};
}
//---------------------------------------------------------------------------
KW_optimization::~KW_optimization()
{
	for (auto p : ctx)
		delete p;

	for (auto q : nonlin)
		delete q;
}
//---------------------------------------------------------------------------
OptContext *KW_optimization::MakeContext()
{
	if (algorithm == "CMAES")
		return K;				// not added to "ctx" since it should not be deleted
	else if (algorithm == "LM")
	{
		OptCtxLM *res = new OptCtxLM(maxit, epsG, epsF, epsX);
		ctx.push_back(res);
		return res;
	}
	else
		throw HMMPI::Exception(HMMPI::stringFormatArr("Неправильный тип алгоритма {0:%s} в KW_optimization::MakeContext",
													  "Incorrect algorithm type {0:%s} in KW_optimization::MakeContext", algorithm));
}
//------------------------------------------------------------------------------------------
NonlinearSystemSolver *KW_optimization::MakeNonlinSolver()
{
	NonlinearSystemSolver *res;
	if (nonlin_solver == "FIXEDPOINT")
		res = new FixedPointIter(epsG, maxit);
	else if (nonlin_solver == "NEWTON" || nonlin_solver == "GNEWTON" || nonlin_solver == "HYBRIDPOWELL")
		res = new NewtonIter(nonlin_solver, epsG, maxit);
	else
		res = new SUNIter(nonlin_solver, epsG, epsX, maxit, maxJacIters, maa);

	nonlin.push_back(res);
	return res;
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
void KW_opt_config::UpdateParams() noexcept
{
	if (delta < 0 || delta > acos(-1)/4)
		SilentError(HMMPI::MessageRE("delta должно быть в пределах [0, п/4]", "delta should be within the range [0, п/4]"));
}
//------------------------------------------------------------------------------------------
KW_opt_config::KW_opt_config()
{
	name = "OPT_CONFIG";

	DEFPAR(MaxIter, 5);
	DEFPAR(MaxHours, 0.5);
	DEFPAR(LMstart, "SIMBEST");
	DEFPAR(r0, 0.2);
	DEFPAR(rmin, 0.01);
	DEFPAR(tau1, 0.25);
	DEFPAR(tau2, 0.75);
	DEFPAR(delta, 1e-5);
	DEFPAR(restr, "CUBE");
	DEFPAR(LMmaxit, 50);
	DEFPAR(LMmaxit_spher, 500);		// max iterations for LM optimization on sphere (only for restr = SPHERE)
	DEFPAR(epsG, 1e-10);
	DEFPAR(epsF, 0);
	DEFPAR(epsX, 0);

	FinalizeParams();
	EXPECTED[2] = std::vector<std::string>{"CURR", "SIMBEST"};
	EXPECTED[8] = std::vector<std::string>{"CUBE", "SPHERE"};
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
std::string KW_eclsmry::copy_file_exists(const std::string &f0, int c)
{
	int flag = 0;
	if (K->MPI_rank == 0)
		flag = HMMPI::FileExists(f0) && c > 0;
	MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

	std::string msg;
	if (flag)
	{
		msg = f0 + " -> " + f0 + "~\n";
		std::string cmd = "cp " + f0 + " " + f0 + "~";		// copy command
		msg += copy_file_exists(f0 + "~", c-1);
		if (K->MPI_rank == 0)
			system(cmd.c_str());
	}

	MPI_Barrier(MPI_COMM_WORLD);
	return msg;
}
//------------------------------------------------------------------------------------------
KW_eclsmry::KW_eclsmry() : Data(MPI_COMM_WORLD)
{
	name = "ECLSMRY";

	DEFPAR(fname, "");
	DEFPAR(backup, 3);
	DEFPAR(Xtol, 1e-8);

	FinalizeParams();
}
//------------------------------------------------------------------------------------------
void KW_eclsmry::FinalAction() noexcept
{
	if (fname == "")
		SilentError(HMMPI::MessageRE("Имя файла не должно быть пустым", "File name should not be empty"));
	if (fname.find_first_of("~") != std::string::npos)
		SilentError(HMMPI::MessageRE("Имя файла не должно содержать '~'", "File name should not contain '~'"));
	if (backup < 0)
		SilentError(HMMPI::MessageRE("Значение 'backup' должно быть >= 0", "Value of 'backup' should be >= 0"));

	if (GetState() != "")
		return;

	try
	{
		std::string msg = Data.LoadFromBinary(this->CWD + "/" + fname);
		Data.Xtol = Xtol;
		Data.set_par_tran(dynamic_cast<KW_parameters*>(K->GetKW_item("PARAMETERS")));
		K->AppText(msg);
	}
	catch (std::exception &e)
	{
		SilentError(e.what());
	}
}
//------------------------------------------------------------------------------------------
std::string KW_eclsmry::Save()
{
	std::string fn = this->CWD + "/" + fname, msg;
	msg = copy_file_exists(fn, backup);
	Data.SaveToBinary(fn);			// to be called on all ranks

	return "ECLSMRY (" + Data.models_params_msg() + (std::string)HMMPI::MessageRE(") сохранен в ", ") saved to ") + fn + "\n" + msg;
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_wrcovar::KW_wrcovar()
{
	name = "WRCOVAR";

	DEFPAR(M, 10);
	DEFPAR(cov_file, "Covariance.txt");
	DEFPAR(count_file, "Cells_count.txt");

	FinalizeParams();
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
// KW_pilot
//------------------------------------------------------------------------------------------
KW_pilot::KW_pilot()
{
	name = "PILOT";

	DEFPARMULT(x);
	DEFPARMULT(y);
	DEFPARMULT(z);

	FinalizeParams();
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
void ParamsInterface::count_active() noexcept
{
	act_ind = std::vector<int>();
	tot_ind = std::vector<int>(act.size(), -1);
	int n = 0; 		// counts active params
	for (size_t i = 0; i < act.size(); i++)
		if (act[i] == "A")
		{
			act_ind.push_back(i);
			tot_ind[i] = n;
			n++;
		}
}
//------------------------------------------------------------------------------------------
std::vector<double> ParamsInterface::actmin() const						// min & max - INTERNAL ACTIVE parameters
{
	return HMMPI::Reorder(min, act_ind);
}
//------------------------------------------------------------------------------------------
std::vector<double> ParamsInterface::actmax() const
{
	return HMMPI::Reorder(max, act_ind);
}
//------------------------------------------------------------------------------------------
std::vector<double> ParamsInterface::get_init_act() const
{
	return HMMPI::Reorder(init, act_ind);
}
//------------------------------------------------------------------------------------------
std::string ParamsInterface::msg() const
{
	std::string res = HMMPI::MessageRE("Загружены параметры:\n", "Loaded parameters:\n");
	res += HMMPI::MessageRE("парам.\tвнутренний    \tmin           \tmax           \tA/N\n", "param.\tinternal      \tmin           \tmax           \tA/N\n");
	for (size_t i = 0; i < init.size(); i++)
	{
		char buff[HMMPI::BUFFSIZE];
		sprintf(buff, "p%ld\t%-14.8g\t%-14.8g\t%-14.8g\t%s\n", i, init[i], min[i], max[i], act[i].c_str());
		res += buff;
	}

	res += HMMPI::stringFormatArr(HMMPI::MessageRE("Активных параметров: {0:%zu}/{1:%zu}\n",
												   "Active parameters: {0:%zu}/{1:%zu}\n"), std::vector<size_t>{act_ind.size(), tot_ind.size()});
	return res;
}
//------------------------------------------------------------------------------------------
std::vector<double> ParamsInterface::SobolDP(long long int &seed) const
{
	std::vector<double> res = init;												// take initial internal params
	HMMPI::VecAssign(res, act_ind, HMMPI::Reorder(SobolSeq(seed), act_ind));	// inject Sobol point for active params

	return res;
}
//------------------------------------------------------------------------------------------
std::vector<double> ParamsInterface::RandUDP(HMMPI::Rand *rctx) const
{
	std::vector<double> res = init;												// take initial internal params
	HMMPI::VecAssign(res, act_ind, HMMPI::Reorder(RandU(rctx), act_ind));		// inject RandU point for active params

	return res;
}
//------------------------------------------------------------------------------------------
std::vector<std::vector<double>> ParamsInterface::SobolSequence(int n, long long int &seed) const	// equivalent to "n" calls to SobolDP()
{
	std::vector<std::vector<double>> res(n);
	for (int i = 0; i < n; i++)
		res[i] = SobolDP(seed);

	return res;
}
//------------------------------------------------------------------------------------------
std::vector<std::vector<double>> ParamsInterface::NormalSequence(int n, unsigned int seed, double R) const		// generates n points; each point's active coords are ~ N(init, sigma^2), inactive coords = init; where sigma = R/sqrt(actdim)
{																					// points which violate [min, max] are discarded
	std::vector<std::vector<double>> res(n);										// the generated points will be approximately at distance R from 'init'
	double sigma = R/sqrt(double(act_ind.size()));	// R/sqrt(actdim)
	HMMPI::Rand rctx(seed, 0.0, 1.0, 0.0, sigma);	// N(0, sigma^2)

	for (int i = 0; i < n; i++)		// generate n points
	{
		std::vector<double> v = init;				// new point
		for (size_t j = 0; j < init.size(); j++)	// propose + accept/reject each coordinate
			if (tot_ind[j] != -1)	// only update the active coords
			{
				bool accepted = false;
				while (!accepted)
				{
					double z = v[j] + rctx.RandN();
					if (min[j] <= z && z <= max[j])
					{
						v[j] = z;
						accepted = true;
					}
				}
			}

		res[i] = v;
	}

	return res;
}
//------------------------------------------------------------------------------------------
void ParamsInterface::Write_params_log(const std::vector<double> &p, std::string fname) const
{
	assert(p.size() == init.size());
	FILE *sw = fopen(fname.c_str(), "w");
	if (sw == NULL)
		throw HMMPI::Exception("Cannot open file for writing in ParamsInterface::Write_params_log");

	fprintf(sw, "%-20s\t%-11s\t%-11s\t%-5s\n", "Param. (internal)", "min", "max", "A/N");
	for (size_t i = 0; i < p.size(); i++)
		fprintf(sw, "%-20.15g\t%-11.6g\t%-11.6g\t%-5.3s\n", p[i], min[i], max[i], act[i].c_str());

	fclose(sw);
}
//------------------------------------------------------------------------------------------
void ParamsInterface::Push_point(double Init, double Min, double Max, std::string AN, std::string Name)
{
	if (AN != "A" && AN != "N")
		throw HMMPI::Exception("Parameter flag not equal to A or N in ParamsInterface::Push_point");

	HMMPI::BoundConstr::Push_point(Init, Min, Max, AN, Name);

	init.push_back(Init);
	act.push_back(AN);

	if (AN == "A")
	{
		act_ind.push_back(int(act.size()) - 1);
		tot_ind.push_back(int(act_ind.size()) - 1);
	}
	else
		tot_ind.push_back(-1);
}
//------------------------------------------------------------------------------------------
ParamsInterface *ParamsInterface::ActToSpherical(const HMMPI::SpherCoord &sc, double d) const
{
	std::vector<double> x = get_init_act();
	if (x.size() < 2)
		throw HMMPI::Exception("Требуются по меньшей мере 2 активных параметра для перехода в сферические координаты",
							   "At least 2 active parameters are required to transform to spherical coordinates");

	ParamsInterface *res = new ParamsInterface;
	res->init = sc.cart_to_spher(x);
	const int sph_dim = res->init.size();
	res->act = std::vector<std::string>(sph_dim, "A");

	res->act_ind = std::vector<int>(sph_dim);
	std::iota(res->act_ind.begin(), res->act_ind.end(), 0);
	res->tot_ind = res->act_ind;

	// if sphere center "sc.c" is on the box boundary, min/max for spherical coordinates will be adjusted
	std::vector<double> actmin = HMMPI::Reorder(min, act_ind);
	std::vector<double> actmax = HMMPI::Reorder(max, act_ind);
	assert (actmin.size() == (size_t)sc.dim && actmax.size() == (size_t)sc.dim);
	res->min = std::vector<double>(res->init.size());
	res->max = std::vector<double>(res->init.size());
	for (int i = 0; i < sc.dim - 2; i++)
	{
		res->min[i] = d;
		res->max[i] = sc.pi - d;

		// 1) SIMPLE approach to impose bounds: only Cartesian cases >= 0, <= 0 are considered
//		const double eps = LAPACKE_dlamch('P');
//		if (fabs(sc.c[i] - actmin[i]) < eps)			// positive
//			res->max[i] = sc.pi / 2;
//		if (fabs(sc.c[i] - actmax[i]) < eps)			// negative
//			res->min[i] = sc.pi / 2;
		// END of 1)

		// 2) NEW approach to impose bounds: distance from the center to Cartesian min/max is converted to bounds for cosines
		double alpha_min = (actmin[i] - sc.c[i])/sc.R;
		double alpha_max = (actmax[i] - sc.c[i])/sc.R;
		if (fabs(alpha_min) < fabs(alpha_max) && fabs(alpha_min) <= 1)		// work with Cartesian min
		{
			double b = acos(alpha_min);
			b = HMMPI::Max(b, 2*d);
			res->max[i] = HMMPI::Min(b, sc.pi - d);
		}
		if (fabs(alpha_min) >= fabs(alpha_max) && fabs(alpha_max) <= 1)		// work with Cartesian max
		{
			double a = acos(alpha_max);
			a = HMMPI::Min(a, sc.pi - 2*d);
			res->min[i] = HMMPI::Max(a, d);
		}
		// END of 2)
	}

	// constraints for the last spherical coordinate
	double a = 0, b = sc.pi2;

	// 1) SIMPLE approach to impose bounds: only Cartesian cases >= 0, <= 0 are considered
//	if (fabs(sc.c[sc.dim - 2] - actmin[sc.dim - 2]) < eps)		// pre-last positive
//	{
//		a = -0.5*sc.pi;
//		b = 0.5*sc.pi;
//	}
//	if (fabs(sc.c[sc.dim - 2] - actmax[sc.dim - 2]) < eps)		// pre-last negative
//	{
//		a = 0.5*sc.pi;
//		b = 1.5*sc.pi;
//	}
//	if (fabs(sc.c[sc.dim - 1] - actmin[sc.dim - 1]) < eps)		// last positive
//	{
//		a = HMMPI::Max(a, 0);
//		b = HMMPI::Min(b, sc.pi);
//	}
//	if (fabs(sc.c[sc.dim - 1] - actmax[sc.dim - 1]) < eps)		// last negative
//	{
//		a = HMMPI::Max(a, sc.pi);
//		b = HMMPI::Min(b, sc.pi2);
//	}
//	if (fabs(sc.c[sc.dim - 2] - actmin[sc.dim - 2]) < eps && fabs(sc.c[sc.dim - 1] - actmax[sc.dim - 1]) < eps)		// special case: pre-last positive, last negative
//	{
//		a = 1.5*sc.pi;
//		b = sc.pi2;
//	}
	// END of 1)

	// 2) NEW approach to impose bounds: distance from the center to Cartesian min/max is converted to bounds for cos/sin
	double alpha_min = (actmin[sc.dim - 2] - sc.c[sc.dim - 2])/sc.R;
	double alpha_max = (actmax[sc.dim - 2] - sc.c[sc.dim - 2])/sc.R;
	double beta_min = (actmin[sc.dim - 1] - sc.c[sc.dim - 1])/sc.R;
	double beta_max = (actmax[sc.dim - 1] - sc.c[sc.dim - 1])/sc.R;
	bool cos_plus = false, sin_minus = false;
	if (fabs(alpha_min) < fabs(alpha_max) && fabs(alpha_min) <= 1)		// cos+
	{
		double v = acos(alpha_min);
		a = -v;
		b = v;
		cos_plus = true;
	}
	if (fabs(alpha_min) >= fabs(alpha_max) && fabs(alpha_max) <= 1)		// cos-
	{
		double v = acos(alpha_max);
		a = v;
		b = sc.pi2 - v;
	}
	if (fabs(beta_min) < fabs(beta_max) && fabs(beta_min) <= 1)			// sin+
	{
		double v = asin(beta_min);
		if (a == 0 && b == sc.pi2)
		{
			a = v;
			b = sc.pi - v;
		}
		else
		{
			a = HMMPI::Max(a, v);
			b = HMMPI::Min(b, sc.pi - v);
		}
	}
	if (fabs(beta_min) >= fabs(beta_max) && fabs(beta_max) <= 1)		// sin-
	{
		double v = asin(beta_max);
		if (a == 0 && b == sc.pi2)
		{
			a = sc.pi - v;
			b = sc.pi2 + v;
		}
		else
		{
			a = HMMPI::Max(a, sc.pi - v);
			b = HMMPI::Min(b, sc.pi2 + v);
		}
		sin_minus = true;
	}
	if (cos_plus && sin_minus)											// special case: cos+, sin-
	{
		double va = acos(alpha_min);
		double vb = asin(beta_max);
		a = HMMPI::Max(sc.pi2 - va, sc.pi - vb);
		b = HMMPI::Min(sc.pi2 + va, sc.pi2 + vb);
	}
	// END of 2)

	res->min[sc.dim - 2] = a;
	res->max[sc.dim - 2] = b;
	res->AdjustInitSpherical(res->init);

	return res;
}
//------------------------------------------------------------------------------------------
ParamsInterface *ParamsInterface::CubeBounds(const std::vector<double> &c, double d) const
{
	assert(d > 0);
	if (c.size() != min.size())
		throw HMMPI::Exception("Size mismatch for the cube center 'c' and full dimension in ParamsInterface::CubeBounds");

	ParamsInterface *res = new ParamsInterface;
	res->act = act;
	res->act_ind = act_ind;
	res->tot_ind = tot_ind;

	res->min = min;		// will be adjusted below for active parameters
	res->max = max;
	res->init = init;

	for (size_t i = 0; i < min.size(); i++)
	{
		if (c[i] < min[i] || c[i] > max[i])
		{
			delete res;
			throw HMMPI::Exception("Center 'c' violates min/max in ParamsInterface::CubeBounds");
		}
		if (act[i] == "A")
		{
			res->min[i] = HMMPI::Max(min[i], c[i] - d);
			res->max[i] = HMMPI::Min(max[i], c[i] + d);

			if (res->init[i] < res->min[i])
				res->init[i] = res->min[i];
			if (res->init[i] > res->max[i])
				res->init[i] = res->max[i];
		}
	}

	return res;
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_limits::KW_limits()
{
	name = "LIMITS";

	DEFPARMULT(min);
	DEFPARMULT(max);
	DEFPARMULT(norm);
	DEFPARMULT(init);
	DEFPARMULT(std);
	DEFPARMULT(func);
	DEFPARMULT(dh);
	DEFPARMULT(dh_type);
	DEFPARMULT(act);

	FinalizeParams();

	EXPECTED[5] = std::vector<std::string>{"I", "LIN", "EXP"};
	EXPECTED[7] = std::vector<std::string>{"CONST", "LIN"};
	EXPECTED[8] = std::vector<std::string>{"A", "N"};
}
//------------------------------------------------------------------------------------------
std::string KW_limits::msg() const
{
	std::string res = HMMPI::MessageRE("Загружены параметры:\n", "Loaded parameters:\n");
	res += HMMPI::MessageRE("парам.\tвнутренний\tвнешний \tA/N\n", "param.\tinternal\texternal\tA/N\n");
	for (size_t i = 0; i < init.size(); i++)
	{
		double Xi = init[i], Xe;	// internal and external (scaled) variables
		if (func[i] == "I")
			Xe = Xi * norm[i];
		else if (func[i] == "LIN")
			Xe = Xi * norm[i] + dh[i];
		else if (func[i] == "EXP")
			Xe = pow(10, Xi * norm[i]);
		else
			throw HMMPI::Exception("Некорректное функциональное преобразование в LIMITS", "Incorrect function transform in LIMITS");

		char buff[HMMPI::BUFFSIZE];
		sprintf(buff, "p%ld\t%-14.8g\t%-14.8g\t%s\n", i, Xi, Xe, act[i].c_str());
		res += buff;
	}

	res += HMMPI::stringFormatArr(HMMPI::MessageRE("Активных параметров: {0:%zu}/{1:%zu}\n",
												   "Active parameters: {0:%zu}/{1:%zu}\n"), std::vector<size_t>{act_ind.size(), tot_ind.size()});

	return res;
}
//------------------------------------------------------------------------------------------
void KW_limits::UpdateParams() noexcept
{
	count_active();
	K->AppText(HMMPI::stringFormatArr(HMMPI::MessageRE("{0:%ld}/{1:%ld} модельных параметров, заданных здесь, активные (A)\n",
													   "{0:%ld}/{1:%ld} model parameters specified here are active (A)\n"), std::vector<size_t>{act_ind.size(), init.size()}));
}
//------------------------------------------------------------------------------------------
void KW_limits::Write_params_log(const std::vector<double> &p, std::string fname) const
{
	assert(p.size() == init.size());
	FILE *sw = fopen(fname.c_str(), "w");
	if (sw == NULL)
		throw HMMPI::Exception("Не удается открыть файл для записи в KW_limits::Write_params_log", "Cannot open file for writing in KW_limits::Write_params_log");

	fprintf(sw, "LIMITS\n");
	for (size_t i = 0; i < p.size(); i++)
		fprintf(sw, "%11.7g\t%11.7g\t%11.7g\t%22.18g\t%11.7g\t%10.7s\t%11.7g\t%10.7s\t%5.3s\n", min[i], max[i], norm[i], p[i], std[i], func[i].c_str(), dh[i], dh_type[i].c_str(), act[i].c_str());

	fclose(sw);
}
//------------------------------------------------------------------------------------------
std::string KW_limits::CheckPositive(const std::vector<double> &v, std::string vname)
{
	for (size_t i = 0; i < v.size(); i++)
		if (v[i] <= 0)
			return HMMPI::stringFormatArr("LIMITS: Значение элемента {0:%ld} не положительно в векторе ", "LIMITS: Element {0:%ld} is not positive in vector ", i) + vname;

	return "";
}
//------------------------------------------------------------------------------------------
void KW_limits::Push_point(double Init, double Min, double Max, std::string AN, std::string Name)
{
	ParamsInterface::Push_point(Init, Min, Max, AN, Name);

	norm.push_back(1);				// add some default values
	std.push_back(1);
	func.push_back("I");
	dh.push_back(1e-5);
	dh_type.push_back("CONST");
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
int KW_parameters::apply_well_sc(int p, std::string s, std::vector<std::vector<double>> &work_vec, const std::vector<std::string> &wnames)	// applies "s" (e.g. "W2,W3/r2") to 2D array work_vec[N_wells x fulldim], with "wnames" - the uppercase well names, "p" - row/parameter number
{																																			// returns the number of encountered errors "well not found"
	const std::string delim = "/";

	s = HMMPI::Trim(s, " \t\r\n");
	s = HMMPI::ToUpper(s);
	size_t pos = s.find(delim);

	assert(wnames.size() == work_vec.size());
	assert(work_vec.size() > 0);
	assert(0 <= p && p < (int)work_vec[0].size());
	if (s.length() < 3 || pos == std::string::npos)
		throw HMMPI::Exception("Ожидается элемент строки формата '../..' в PARAMETERS.well_sc; найден: " + s,
							   "String element of format '../..' is expected in PARAMETERS.well_sc; found: " + s);

	std::string sw = s.substr(0, pos);
	std::string snum = s.substr(pos+1, s.length()-pos-1);
	const double num = HMMPI::StoD(snum);

	std::vector<std::string> sw_parts;				// sw_parts will contain the well names
	HMMPI::tokenize(sw, sw_parts, ",", true);

	int res_err_num = 0;
	for (size_t i = 0; i < sw_parts.size(); i++)
	{
		if (sw_parts[i] == "ALL")
			for (size_t j = 0; j < work_vec.size(); j++)
				work_vec[j][p] = num;
		else
		{
			const auto it = std::find(wnames.begin(), wnames.end(), sw_parts[i]);
			if (it == wnames.end())
			{
				res_err_num++;
				SilentError(HMMPI::stringFormatArr((std::string)"Скважина '" + sw_parts[i] + "' из строки {0:%d} PARAMETERS.well_sc не найдена в общем списке скважин (ECLVECTORS)",
									   	   	   	   (std::string)"Well '" + sw_parts[i] + "' from line {0:%d} of PARAMETERS.well_sc was not found in the wells list (ECLVECTORS)", p+1));
			}
			else
				work_vec[it - wnames.begin()][p] = num;
		}
	}

	return res_err_num;
}
//------------------------------------------------------------------------------------------
std::string KW_parameters::par_name(int i) const
{
	if (i < 0 || i >= (int)name.size())
		throw HMMPI::Exception("Index out of range in KW_parameters::par_name");

	return name[i];
}
//------------------------------------------------------------------------------------------
void KW_parameters::check_names() noexcept
{
	std::string rpt;
	if (HMMPI::FindDuplicate(name, rpt))
		SilentError(HMMPI::stringFormatArr("Повторение параметра {0:%s}", "Repeated parameter {0:%s}", rpt));

	for (size_t i = 0; i < name.size(); i++)								// check reserved names
		if (std::find(reserved_names.begin(), reserved_names.end(), name[i]) != reserved_names.end())
			SilentError(HMMPI::stringFormatArr("Параметр не может иметь имя '{0:%s}'", "Parameter cannot have name '{0:%s}'", name[i]));
}
//------------------------------------------------------------------------------------------
void KW_parameters::fill_norm_logmin() noexcept
{
	logmin = min;
	norm = std::vector<double>(logmin.size());
	for (size_t i = 0; i < logmin.size(); i++)
	{
		if (func[i] == "EXP")
		{
			logmin[i] = log10(logmin[i]);
			norm[i] = log10(max[i]) - logmin[i];
		}
		else
			norm[i] = max[i] - logmin[i];
	}
}
//------------------------------------------------------------------------------------------
void KW_parameters::UpdateParams() noexcept
{
	count_active();
	check_names();

	// check signs
	char buff[HMMPI::BUFFSIZE], buffrus[HMMPI::BUFFSIZE];
	for (size_t i = 0; i < min.size(); i++)
	{
		if (!(min[i] <= val[i] && val[i] <= max[i]))
		{
			sprintf(buffrus, "Нарушено min <= val <= max (не выполняется %g <= %g <= %g) для параметра %s в строке %zu", min[i], val[i], max[i], name[i].c_str(), i+1);
			sprintf(buff, "Condition min <= val <= max is violated (%g <= %g <= %g does not hold) for parameter %s in line %zu", min[i], val[i], max[i], name[i].c_str(), i+1);
			SilentError(HMMPI::MessageRE(buffrus, buff));
		}
		if (min[i] >= max[i])
		{
			sprintf(buffrus, "Нарушено min < max (не выполняется %g < %g) для параметра %s в строке %zu", min[i], max[i], name[i].c_str(), i+1);
			sprintf(buff, "Condition min < max is violated (%g < %g does not hold) for parameter %s in line %zu", min[i], max[i], name[i].c_str(), i+1);
			SilentError(HMMPI::MessageRE(buffrus, buff));
		}
		if (func[i] == "EXP" && min[i] <= 0)
		{
			sprintf(buffrus, "Для FUNC = EXP нарушено min > 0 (min = %g) для параметра %s в строке %zu", min[i], name[i].c_str(), i+1);
			sprintf(buff, "For FUNC = EXP, min > 0 is violated (min = %g) for parameter %s in line %zu", min[i], name[i].c_str(), i+1);
			SilentError(HMMPI::MessageRE(buffrus, buff));
		}
	}
	if (GetState() != "")
		return;			// silent exit with errors

	// fill 'norm', 'logmin'
	fill_norm_logmin();

	// initialize 'par_map'
	delete par_map;
	par_map = new HMMPI::TagPrintfMap(name, val);

	// fill init, BoundConstr::min, BoundConstr::max
	init = ExternalToInternal(val);
	BoundConstr::min = ExternalToInternal(min);
	BoundConstr::max = ExternalToInternal(max);

	K->AppText(HMMPI::stringFormatArr(HMMPI::MessageRE("Активных параметров: {0:%zu}/{1:%zu}\n",
													   "Active parameters: {0:%zu}/{1:%zu}\n"), std::vector<size_t>{act_ind.size(), tot_ind.size()}));
}
//------------------------------------------------------------------------------------------
KW_parameters::KW_parameters() : ln10(log(10)), reserved_names({"", "MOD", "PATH", "RANK", "SIZE"})
{
	KW_multparams::name = "PARAMETERS";
	par_map = nullptr;

	DEFPARMULT(name);		// 0
	DEFPARMULT(val);		// 1
	DEFPARMULT(min);		// 2
	DEFPARMULT(max);		// 3
	DEFPARMULT(act);		// 4
	DEFPARMULT(backval);	// 5
	DEFPARMULT(func);		// 6
	DEFPARMULT(well_sc);	// 7 command to rescale the parameter's effect on wells, e.g. <All/r0>-<W1/r1>-<W2,W3/r2>

	FinalizeParams();

	EXPECTED[4] = std::vector<std::string>{"A", "N"};
	EXPECTED[6] = std::vector<std::string>{"LIN", "EXP"};
}
//------------------------------------------------------------------------------------------
KW_parameters::~KW_parameters()
{
	delete par_map;
}
//------------------------------------------------------------------------------------------
std::string KW_parameters::msg() const
{
	char buff[HMMPI::BUFFSIZE], buffeng[HMMPI::BUFFSIZE];

	sprintf(buff, "Загружены параметры:\n%-10.300s\t%-11s\t%-5s\n", "параметр", "значение", "A/N");
	sprintf(buffeng, "Loaded parameters:\n%-10.300s\t%-11s\t%-5s\n", "parameter", "value", "A/N");
	std::string res = HMMPI::MessageRE(buff, buffeng);

	for (size_t i = 0; i < val.size(); i++)
	{
		sprintf(buff, "%-10.300s\t%-11.6g\t%-5.3s\n", name[i].c_str(), val[i], act[i].c_str());
		res += buff;
	}

	sprintf(buff, "Активных параметров: %zu/%zu\n", act_ind.size(), tot_ind.size());
	sprintf(buffeng, "Active parameters: %zu/%zu\n", act_ind.size(), tot_ind.size());
	res += HMMPI::MessageRE(buff, buffeng);

	return res;
}
//------------------------------------------------------------------------------------------
void KW_parameters::Write_params_log(const std::vector<double> &p, std::string fname) const
{
	assert(p.size() == val.size());
	std::vector<double> p_ext = InternalToExternal(p);
	FILE *sw = fopen(fname.c_str(), "w");
	if (sw == NULL)
		throw HMMPI::Exception("Не удается открыть файл для записи в KW_parameters::Write_params_log", "Cannot open file for writing in KW_parameters::Write_params_log");

	int maxname = 10;				// find the maximum length of parameter's name
	for (size_t i = 0; i < name.size(); i++)
		if ((int)name[i].length() > maxname)
			maxname = name[i].length();
	maxname += 2;					// additional padding

	fprintf(sw, "%-*.*s\t%-20s\t%-11s\t%-11s\t%-5s\t%-11s\t%-5s\t%-s\n", maxname, maxname, "PARAMETERS", "-- val", "min", "max", "A/N", "backval", "func", "well_sc");
	for (size_t i = 0; i < p_ext.size(); i++)
	{
		if (p_ext[i] < min[i])		// correct the output to be within the bounds
			p_ext[i] = min[i];
		if (p_ext[i] > max[i])
			p_ext[i] = max[i];
		fprintf(sw, "%-*.*s\t%-20.15g\t%-11.6g\t%-11.6g\t%-5.3s\t%-11s\t%-5.3s\t%-s\n", maxname, maxname, name[i].c_str(), p_ext[i], min[i], max[i], act[i].c_str(), backval[i].c_str(), func[i].c_str(), well_sc[i].c_str());
	}

	fclose(sw);
}
//------------------------------------------------------------------------------------------
const ParamsInterface *KW_parameters::GetParamsInterface() const
{
	DECLKWD(limits, KW_limits, "LIMITS");
	DECLKWD(params, KW_parameters, "PARAMETERS");
	DECLKWD(params2, KW_parameters2, "PARAMETERS2");

	const ParamsInterface *res = nullptr;
	int count = 0;
	if (limits->GetState() == "")
	{
		res = limits;
		count++;
	}
	if (params->GetState() == "")
	{
		res = params;
		count++;
	}
	if (params2->GetState() == "")
	{
		res = params2;
		count++;
	}

	if (count != 1)
		throw HMMPI::Exception("Должно быть задано либо PARAMETERS, либо PARAMETERS2, либо LIMITS (что-то одно)", "Either PARAMETERS or PARAMETERS2 or LIMITS (only one) should be defined");
	return res;
}
//------------------------------------------------------------------------------------------
void KW_parameters::Push_point(double Init, double Min, double Max, std::string AN, std::string Name)
{
	ParamsInterface::Push_point(Init, Min, Max, AN, Name);

	if (std::find(reserved_names.begin(), reserved_names.end(), Name) != reserved_names.end())
		throw HMMPI::Exception("'Name' cannot be one of the reserved names in KW_parameters::Push_point");

	norm.push_back(1);
	logmin.push_back(0);
	name.push_back(Name);

	val.push_back(Init);		// since norm=1 && logmin=0 && func=LIN -> external=internal
	min.push_back(Min);
	max.push_back(Max);

	backval.push_back("0");
	func.push_back("LIN");
	well_sc.push_back("");
}
//------------------------------------------------------------------------------------------
std::vector<double> KW_parameters::InternalToExternal(const std::vector<double> &in) const	// y -> x
{
	if (in.size() != logmin.size())
		throw HMMPI::EObjFunc("Вектор 'in' не соответствует полной размерности в KW_parameters::InternalToExternal",
							  "Vector 'in' is not consistent with full dimension in KW_parameters::InternalToExternal");

	std::vector<double> res(in.size());
	for (size_t i = 0; i < res.size(); i++)
		if (func[i] == "LIN")
			res[i] = in[i]*norm[i] + logmin[i];
		else if (func[i] == "EXP")
			res[i] = exp(ln10*(in[i]*norm[i] + logmin[i]));
		else
			throw HMMPI::EObjFunc("Not acceptable 'func' type in KW_parameters::InternalToExternal");

	return res;
}
//------------------------------------------------------------------------------------------
std::vector<double> KW_parameters::ExternalToInternal(const std::vector<double> &ex) const	// x -> y
{
	if (ex.size() != logmin.size())
		throw HMMPI::Exception("Вектор 'ex' не соответствует полной размерности в KW_parameters::ExternalToInternal",
							   "Vector 'ex' is not consistent with full dimension in KW_parameters::ExternalToInternal");

	std::vector<double> res(ex.size());
	for (size_t i = 0; i < res.size(); i++)
		if (func[i] == "LIN")
			res[i] = (ex[i] - logmin[i])/norm[i];
		else if (func[i] == "EXP")
			res[i] = (log10(ex[i]) - logmin[i])/norm[i];
		else
			throw HMMPI::Exception("Not acceptable 'func' type in KW_parameters::ExternalToInternal");

	return res;
}
//------------------------------------------------------------------------------------------
std::vector<double> KW_parameters::dxe_To_dxi(const std::vector<double> &dxe, const std::vector<double> &in) const		// transform gradient d/dxe -> d/dxi
{
	if (dxe.size() != logmin.size())
		throw HMMPI::Exception("Вектор 'dxe' не соответствует полной размерности в KW_parameters::dxe_To_dxi",
							   "Vector 'dxe' is not consistent with full dimension in KW_parameters::dxe_To_dxi");
	if (in.size() != logmin.size())
		throw HMMPI::Exception("Вектор 'in' не соответствует полной размерности в KW_parameters::dxe_To_dxi",
							   "Vector 'in' is not consistent with full dimension in KW_parameters::dxe_To_dxi");

	std::vector<double> res(dxe.size());
	for (size_t i = 0; i < res.size(); i++)
		if (func[i] == "LIN")
			res[i] = norm[i] * dxe[i];
		else if (func[i] == "EXP")
			res[i] = exp(ln10*(in[i]*norm[i] + logmin[i])) * ln10*norm[i] * dxe[i];
		else
			throw HMMPI::EObjFunc("Not acceptable 'func' type in KW_parameters::dxe_To_dxi");

	return res;
}
//------------------------------------------------------------------------------------------
void KW_parameters::fill_well_sc_table(std::vector<std::string> wnames)		// fills "uniq_sc", "sc_colors"; 'wnames' are the well names (case insensitive, no duplicates)
{
	std::vector<std::vector<double>> work_vec(wnames.size(), std::vector<double>(act.size(), 1.0));		// [N_wells, fulldim]

	for (auto &w : wnames)
		w = HMMPI::ToUpper(w);

	std::string dup;
	if (HMMPI::FindDuplicate(wnames, dup))
		throw HMMPI::Exception("Duplicate well name '" + dup + "' in KW_parameters::fill_well_sc_table");

	int err_num = 0;								// counts how many wells were not found
	for (size_t i = 0; i < act.size(); i++)			// fill the full table
	{
		std::vector<std::string> parts;				// parts will contain "../.."
		HMMPI::tokenizeExact(HMMPI::Trim(well_sc[i], "<>"), parts, ">-<", true);

		for (size_t j = 0; j < parts.size(); j++)
			err_num += apply_well_sc(i, parts[j], work_vec, wnames);
	}
	if (err_num > 0)
		throw HMMPI::Exception(HMMPI::stringFormatArr("Не найдено скважин в общем списке (ECLVECTORS): {0:%d}", "Not found wells in the list (ECLVECTORS): {0:%d}", err_num));

	uniq_sc = HMMPI::Unique(work_vec);

	sc_colors = std::vector<int>(wnames.size(), -1);
	for (size_t j = 0; j < wnames.size(); j++)
	{
		const auto it = std::find(uniq_sc.begin(), uniq_sc.end(), work_vec[j]);
		assert(it != uniq_sc.end());
		sc_colors[j] = it - uniq_sc.begin();
	}
}
//------------------------------------------------------------------------------------------
std::vector<int> KW_parameters::sc_colors_textsmry()			// returns 'pscale colors' for the TEXTSMRY points with nonzero sigmas; uses DATES, ECLVECTORS, TEXTSMRY, calls fill_well_sc_table()
{
	Start_pre();
	IMPORTKWD(dates, KW_dates, "DATES");
	IMPORTKWD(vectors, KW_eclvectors, "ECLVECTORS");
	IMPORTKWD(textsmry, KW_textsmry, "TEXTSMRY");
	Finish_pre();

	std::vector<std::string> uniq_wells = vectors->WGname;
	for (auto &w : uniq_wells)
		w = HMMPI::ToUpper(w);
	uniq_wells = HMMPI::Unique(uniq_wells);

	fill_well_sc_table(uniq_wells);								// fills "uniq_sc", "sc_colors"

	std::vector<double> sigmas = textsmry->OnlySigmas();		// all sigmas including zeros
	const size_t N_dates = dates->dates.size();
	assert(sigmas.size() == vectors->WGname.size() * N_dates);

	int smrylen = 0;											// length of the resulting vector
	for (double x : sigmas)
		if (x != 0)
			smrylen++;

	std::vector<int> res(smrylen);
	size_t c = 0;
	for (size_t v = 0; v < vectors->WGname.size(); v++)
	{
		const auto it = std::find(uniq_wells.begin(), uniq_wells.end(), vectors->WGname[v]);
		assert(it != uniq_wells.end());

		for (size_t d = 0; d < N_dates; d++)
			if (sigmas[v*N_dates + d] != 0)
			{
				res[c] = sc_colors[it - uniq_wells.begin()];
				c++;
			}
	}

	return res;
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
void KW_parameters2::fill_norm_logmin() noexcept
{
	logmin = norm = std::vector<double>(min.size());
	for (size_t i = 0; i < logmin.size(); i++)
	{
		if (func[i] == "EXP")
		{
			logmin[i] = (log10(max[i]) + log10(min[i]))/2;
			norm[i] = (log10(max[i]) - log10(min[i]))/2;
		}
		else
		{
			logmin[i] = (max[i] + min[i])/2;
			norm[i] = (max[i] - min[i])/2;
		}
	}
}
//------------------------------------------------------------------------------------------
KW_parameters2::KW_parameters2() : KW_parameters()
{
	KW_multparams::name = "PARAMETERS2";
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
void KW_dates::UpdateParams() noexcept
{
	DECLKWD(textsmry, KW_textsmry, "TEXTSMRY");

	dates.resize(D.size());
	for (size_t i = 0; i < dates.size(); i++)
	{
		dates[i].Day = D[i];
		dates[i].Month = M[i];
		dates[i].Year = Y[i];

		if (i > 0 && !(dates[i] > dates[i-1]))
		{
			char buff[HMMPI::BUFFSIZE], buffeng[HMMPI::BUFFSIZE];
			sprintf(buff, "Нарушено возрастание дат: %02d/%02d/%d и %02d/%02d/%d", dates[i-1].Day, dates[i-1].Month, dates[i-1].Year, dates[i].Day, dates[i].Month, dates[i].Year);
			sprintf(buffeng, "Dates are not increasing: %02d/%02d/%d and %02d/%02d/%d", dates[i-1].Day, dates[i-1].Month, dates[i-1].Year, dates[i].Day, dates[i].Month, dates[i].Year);
			SilentError(HMMPI::MessageRE(buff, buffeng));
		}
	}

	if (textsmry->GetState() == "")
		textsmry->SetState(HMMPI::MessageRE("TEXTSMRY должно быть перезагружено после чтения DATES\n", "TEXTSMRY should be reloaded after reading DATES\n"));
}
//------------------------------------------------------------------------------------------
KW_dates::KW_dates()
{
	name = "DATES";

	DEFPARMULT(D);
	DEFPARMULT(M);
	DEFPARMULT(Y);

	FinalizeParams();
}
//------------------------------------------------------------------------------------------
std::vector<int> KW_dates::zeroBased()
{
	size_t count = D.size();
	std::vector<int> res(count);
	if (count != 0)
	{
		std::vector<int> MLEN{0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334};	// length of months
		res[0] = 0;
		for (size_t i = 1; i < count; i++)
		{
			int deltaD = D[i] - D[0];
			int deltaM = MLEN[M[i]-1] - MLEN[M[0]-1];
			int deltaY = (Y[i] - Y[0])*365;

			// count leap years
			int leap1 = Y[0]/4;
			int leap2 = Y[i]/4;
			if (Y[0]%4 == 0 && M[0] <= 2)
				leap1--;
			if (Y[i]%4 == 0 && M[i] <= 2)
				leap2--;

			res[i] = deltaD + deltaM + deltaY + leap2-leap1;
		}
	}

	return res;
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
std::vector<double> KW_3points::_internal(const std::vector<double> &v) const	// does the job for x_internal() etc
{
	DECLKWD(params, KW_parameters, "PARAMETERS");
	DECLKWD(params2, KW_parameters2, "PARAMETERS2");
	if (params2->GetState() == "")
		params = params2;

	if (params->GetState() == "")
	{
		if (v.size() != params->act.size())
			throw HMMPI::Exception(HMMPI::stringFormatArr("Размерность 3POINTS должна соответствовать полной размерности PARAMETERS {0:%zu}",
								   	   	   	   	   	   	  "Dimension of 3POINTS should match the full dimension of PARAMETERS {0:%zu}", params->act.size()));
		msg = HMMPI::MessageRE(" (заданные в 3POINTS точки - во внешних переменных)",
							   " (points specified in 3POINTS are in external representation)");
		return params->ExternalToInternal(v);
	}
	else
	{
		msg = "";			// no conversion
		return v;
	}
}
//------------------------------------------------------------------------------------------
KW_3points::KW_3points()
{
	name = "3POINTS";

	NAMES = std::vector<std::string>{"P1", "P2", "P3"};
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_pConnect_config::KW_pConnect_config()
{
	name = "PCONNECT_CONFIG";

	DEFPAR(scale, 1.0);		// multiplier for objective function (should equal the number of observed data)
	FinalizeParams();
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_soboltest::KW_soboltest()
{
	name = "SOBOLTEST";

	DEFPAR(dim, 2);
	DEFPAR(seed, 0);
	DEFPAR(num, 10);
	DEFPAR(fname, "sobol_test_10.txt");

	FinalizeParams();
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_matrixtest::KW_matrixtest()
{
	name = "MATRIXTEST";

	DEFPAR(sizeA, 5);
	DEFPAR(sizeB, 5);
	DEFPAR(sizeC, 5);
	DEFPAR(sizeD, 5);
	DEFPAR(filein, "matrix_test_in.txt");
	DEFPAR(fileout, "matrix_test_out.txt");

	FinalizeParams();
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_proxyros::KW_proxyros()
{
	name = "PROXYROS";

	DEFPAR(x0, 0);
	DEFPAR(x1, 1);
	DEFPAR(y0, 0);
	DEFPAR(y1, 1);
	DEFPAR(pts0, 50);
	DEFPAR(pts1, 50);
	DEFPAR(ptype, "KGAUSS");
	DEFPAR(R, 1);
	DEFPAR(trend, 0);
	DEFPAR(add_pts, 10);
	DEFPAR(Nx, 5);
	DEFPAR(Ny, 5);
	DEFPAR(dx, 1e-3);
	DEFPAR(dy, 1e-3);
	DEFPAR(fname, "TestRosenbrockProxy.txt");

	FinalizeParams();
	EXPECTED[6] = std::vector<std::string>{"KGAUSS", "KSPHER", "TPS"};
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_proxylin::KW_proxylin()
{
	name = "PROXYLIN";

	DEFPAR(a, 1);
	DEFPAR(pts0, 50);
	DEFPAR(pts1, 50);
	DEFPAR(ptype, "KGAUSS");
	DEFPAR(R, 2);
	DEFPAR(trend, 0);
	DEFPAR(add_pts, 10);
	DEFPAR(test_pts, 50);
	DEFPAR(dx, 1e-3);
	DEFPAR(numgrad, "YES");
	DEFPAR(psimple, "YES");
	DEFPAR(fname, "TestLinDataProxy.txt");

	FinalizeParams();
	EXPECTED[3] = std::vector<std::string>{"KGAUSS", "KSPHER", "TPS"};
	EXPECTED[9] = std::vector<std::string>{"YES", "NO"};
	EXPECTED[10] = std::vector<std::string>{"YES", "NO"};
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
void _proxy_params::UpdateParams() noexcept
{
	corr = HMMPI::Func1D_factory::New(cfunc);
	corr->SetNugget(nugget);
	if (cfunc == "MATERN")
		dynamic_cast<HMMPI::CorrMatern*>(corr)->SetNu(nu);

	if (nugget < 0 || nugget >= 1)
		SilentError(HMMPI::MessageRE("Требуется 0 <= nugget < 1", "Need 0 <= nugget < 1"));

	if (trend < -1)
	{
		K->AppText(HMMPI::MessageRE("ПРЕДУПРЕЖДЕНИЕ: trend < -1 недопустим, взято значение -1\n", "WARNING: trend < -1 is not allowed, -1 is taken\n"));
		trend = -1;
		K->TotalWarnings++;
	}
	if (trend > 3)
	{
		K->AppText(HMMPI::MessageRE("ПРЕДУПРЕЖДЕНИЕ: trend > 3 недопустим, взято значение 3\n", "WARNING: trend > 3 is not allowed, 3 is taken\n"));
		trend = 3;
		K->TotalWarnings++;
	}
	if (init_pts < 1)
		SilentError(HMMPI::MessageRE("Требуется init_pts >= 1", "Need init_pts >= 1"));

	if (select_pts < 0)
		SilentError(HMMPI::MessageRE("Требуется select_pts >= 0", "Need select_pts >= 0"));

	if (nu <= 0)
		SilentError(HMMPI::MessageRE("Требуется nu > 0", "Need nu > 0"));
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_proxy::KW_proxy() : _proxy_params()
{
	name = "PROXY_CONFIG";

	DEFPAR(init_pts, 50);
	DEFPAR(select_pts, 10);
	DEFPAR(cfunc, "GAUSS");
	DEFPAR(nugget, 0);
	DEFPAR(R, 1);
	DEFPAR(trend, 1);
	DEFPAR(nu, 2);
	DEFPAR(opt, "OFF");
	DEFPAR(grad_init_pts, "");		// e.g. "0,2,3" - indices of points from 'init_pts' in which the gradients will be used for training
	DEFPAR(grad_add_pts, "");		// indices of points from 'select_pts'
	DEFPAR(grad_comps, "");			// e.g. "1,4" - indices of the gradient components participating in training

	FinalizeParams();
	EXPECTED[2] = std::vector<std::string>{"GAUSS", "SPHER", "EXP", "VARGAUSS", "MATERN"};
	EXPECTED[7] = std::vector<std::string>{"ON", "OFF"};
}
//------------------------------------------------------------------------------------------
void KW_proxy::UpdateParams() noexcept
{
	_proxy_params::UpdateParams();

	try
	{
		ind_grad_init_pts = str_to_vector_int(grad_init_pts);
		ind_grad_add_pts = str_to_vector_int(grad_add_pts);
		ind_grad_comps = str_to_vector_int(grad_comps);
	}
	catch (std::exception &e)
	{
		SilentError(e.what());
	}

	std::sort(ind_grad_init_pts.begin(), ind_grad_init_pts.end());		// sort the arrays
	std::sort(ind_grad_add_pts.begin(), ind_grad_add_pts.end());
	std::sort(ind_grad_comps.begin(), ind_grad_comps.end());

	int dup;
	if(HMMPI::FindDuplicate(ind_grad_init_pts, dup))
		SilentError(HMMPI::stringFormatArr("Повторяется значение {0:%d} в grad_init_pts", "Duplicate value {0:%d} in grad_init_pts", dup));		// check for duplicates
	if(HMMPI::FindDuplicate(ind_grad_add_pts, dup))
		SilentError(HMMPI::stringFormatArr("Повторяется значение {0:%d} в grad_add_pts", "Duplicate value {0:%d} in grad_add_pts", dup));
	if(HMMPI::FindDuplicate(ind_grad_comps, dup))
		SilentError(HMMPI::stringFormatArr("Повторяется значение {0:%d} в grad_comps", "Duplicate value {0:%d} in grad_comps", dup));

	grad_init_pts = vector_int_to_str(ind_grad_init_pts);				// update for printing
	grad_add_pts = vector_int_to_str(ind_grad_add_pts);
	grad_comps = vector_int_to_str(ind_grad_comps);
}
//------------------------------------------------------------------------------------------
std::vector<int> KW_proxy::str_to_vector_int(std::string s)
{
	std::vector<std::string> parts;
	HMMPI::tokenize(s, parts, ",'", true);
	std::vector<int> res(parts.size());

	for (size_t i = 0; i < res.size(); i++)
		res[i] = HMMPI::StoL(parts[i]);

	return res;
}
//------------------------------------------------------------------------------------------
std::string KW_proxy::vector_int_to_str(const std::vector<int> &v)
{
	std::string res = "'";
	for (size_t i = 0; i < v.size(); i++)
		if (i+1 < v.size())
			res += HMMPI::stringFormatArr("{0:%d},", std::vector<int>{v[i]});
		else
			res += HMMPI::stringFormatArr("{0:%d}", std::vector<int>{v[i]});

	return res + "'";
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_proxy_dump::KW_proxy_dump()
{
	name = "PROXY_DUMP";

	DEFPAR(train_ind, -1);					// integer specifying dump file to train proxy from, in ModelFactory::Make(), -1 means train from Sobol sequence
	DEFPAR(dump_inds, "");					// comma-separated indices for dumping during MCMC, -1 is ignored

	FinalizeParams();
}
//------------------------------------------------------------------------------------------
void KW_proxy_dump::UpdateParams() noexcept
{
	try
	{
		vec_dump_inds = KW_proxy::str_to_vector_int(dump_inds);
	}
	catch (std::exception &e)
	{
		SilentError(e.what());
	}

	std::sort(vec_dump_inds.begin(), vec_dump_inds.end());		// sort the array

	int dup;
	if(HMMPI::FindDuplicate(vec_dump_inds, dup))
		SilentError(HMMPI::stringFormatArr("Повторяется значение {0:%d} в dump_inds", "Duplicate value {0:%d} in dump_inds", dup));		// check for duplicates

	dump_inds = KW_proxy::vector_int_to_str(vec_dump_inds);		// update for printing
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_model::KW_model() : _proxy_params(), mod(0)
{
	name = "MODEL";

	DEFPAR(type, "SIM");
	DEFPAR(simulator, "ECL");
	DEFPAR(R, 1.0);
	DEFPAR(trend, 1);
	DEFPAR(cfunc, "GAUSS");
	DEFPAR(nu, 2.5);
	DEFPAR(nugget, 0.0);

	init_pts = 1;
	select_pts = 0;
	opt = "OFF";

	FinalizeParams();
	EXPECTED[0] = std::vector<std::string>{"SIM", "PROXY"};
	EXPECTED[1] = std::vector<std::string>{"ECL", "TNAV"};
	EXPECTED[4] = std::vector<std::string>{"GAUSS", "MATERN"};
}
//------------------------------------------------------------------------------------------
KW_model::~KW_model()
{
	delete mod;
}
//------------------------------------------------------------------------------------------
PhysModel *KW_model::MakeModel(KW_item *kw, std::string cwd, std::string Type)
{
	PhysModel *res = 0;
	if (Type == "Default")
		Type = type;
	if (Type == "SIM")
	{
		res = new PMEclipse(K, kw, cwd, MPI_COMM_WORLD);
		delete mod;
		mod = res;				// for automatic deletion
	}
	else if (Type == "PROXY")
	{
		DECLKWD(eclsmry, KW_eclsmry, "ECLSMRY");
		DECLKWD(dates, KW_dates, "DATES");
		DECLKWD(vectors, KW_eclvectors, "ECLVECTORS");
		DECLKWD(textsmry, KW_textsmry, "TEXTSMRY");

		kw->Start_pre();
		kw->Add_pre("ECLSMRY");
		kw->Add_pre("DATES");
		kw->Add_pre("ECLVECTORS");
		kw->Add_pre("TEXTSMRY");
		kw->Finish_pre();

		res = eclsmry->get_Data().MakeProxy(dates->dates, vectors->vecs, textsmry->OnlySigmas(), K, kw, cwd);
	}
	else
		throw HMMPI::Exception("Wrong model type in KW_model::MakeModel");

	return res;
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_matvecvec::KW_matvecvec()
{
	name = "MATVECVEC";
	erows = ecols = -1;
	dec_verb = 1;
}
//------------------------------------------------------------------------------------------
void KW_matvecvec::ProcessParamTable() noexcept
{
	try
	{
		if (GetState().size() > 0)
			return;

		size_t Ni = par_table.ICount();
		size_t Nj = par_table.JCount();

		if (Ni < 1 || Nj < 2)
			throw HMMPI::Exception("Требуется минимум 1 строка и 2 столбца в таблице", "Need at least 1 row and 2 columns in table");

		M = HMMPI::Mat(Ni, Nj - 2, 0);
		v1 = std::vector<double>(Ni);
		v2 = std::vector<double>(Ni);

		for (size_t i = 0; i < Ni; i++)
		{
			for (size_t j = 0; j < Nj-2; j++)
				M(i, j) = HMMPI::StoD(par_table(i, j));

			v1[i] = HMMPI::StoD(par_table(i, Nj-2));
			v2[i] = HMMPI::StoD(par_table(i, Nj-1));
		}
	}
	catch (std::exception &e)
	{
		SilentError(e.what());
	}
}
//------------------------------------------------------------------------------------------
void KW_matvecvec::Action() noexcept
{
	if (GetState().size() == 0)
		K->AppText(HMMPI::stringFormatArr(HMMPI::MessageRE("Загружена матрица {0:%ld} x {1:%ld}, и два вектора длины {2:%ld}\n",
														   "Loaded matrix {0:%ld} x {1:%ld}, and two vectors of size {2:%ld}\n"),
										  std::vector<size_t>{M.ICount(), M.JCount(), v1.size()}));
}
//------------------------------------------------------------------------------------------
std::vector<double> KW_matvecvec::Std() const
{
	std::vector<double> res = v2;
	std::transform(res.begin(), res.end(), res.begin(), HMMPI::_sqrt);
	return res;
}
//------------------------------------------------------------------------------------------
std::vector<double> KW_matvecvec::Data() const
{
	return v1;
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_matvec::KW_matvec()
{
	name = "MATVEC";
	erows = ecols = -1;
	dec_verb = 1;
}
//------------------------------------------------------------------------------------------
void KW_matvec::ProcessParamTable() noexcept
{
	try
	{
		if (GetState().size() > 0)
			return;

		size_t Ni = par_table.ICount();
		size_t Nj = par_table.JCount();

		if (Ni < 1 || Nj < 1)
			throw HMMPI::Exception("Требуется минимум 1 строка и 1 столбец в таблице", "Need at least 1 row and 1 column in table");

		M = HMMPI::Mat(Ni, Nj - 1, 0);
		v1 = std::vector<double>(Ni);

		for (size_t i = 0; i < Ni; i++)
		{
			for (size_t j = 0; j < Nj-1; j++)
				M(i, j) = HMMPI::StoD(par_table(i, j));

			v1[i] = HMMPI::StoD(par_table(i, Nj-1));
		}
	}
	catch (std::exception &e)
	{
		SilentError(e.what());
	}
}
//------------------------------------------------------------------------------------------
void KW_matvec::Action() noexcept
{
	if (GetState().size() == 0)
		K->AppText(HMMPI::stringFormatArr(HMMPI::MessageRE("Загружена матрица {0:%ld} x {1:%ld}, и вектор длины {2:%ld}\n",
														   "Loaded matrix {0:%ld} x {1:%ld}, and a vector of size {2:%ld}\n"),
										  	  	  	  	  std::vector<size_t>{M.ICount(), M.JCount(), v1.size()}));
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_mat::KW_mat()
{
	name = "MAT";
	erows = ecols = -1;
	dec_verb = 1;
}
//------------------------------------------------------------------------------------------
void KW_mat::ProcessParamTable() noexcept
{
	try
	{
		if (GetState().size() > 0)
			return;

		size_t Ni = par_table.ICount();
		size_t Nj = par_table.JCount();

		M = HMMPI::Mat(Ni, Nj, 0);
		for (size_t i = 0; i < Ni; i++)
			for (size_t j = 0; j < Nj; j++)
				M(i, j) = HMMPI::StoD(par_table(i, j));
	}
	catch (std::exception &e)
	{
		SilentError(e.what());
	}
}
//------------------------------------------------------------------------------------------
void KW_mat::Action() noexcept
{
	if (GetState().size() == 0)
		K->AppText(HMMPI::stringFormatArr(HMMPI::MessageRE(
			"Загружена матрица {0:%ld} x {1:%ld}\n", "Loaded matrix {0:%ld} x {1:%ld}\n"), std::vector<size_t>{M.ICount(), M.JCount()}));
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
const HMMPI::Solver *KW_LinSolver::Sol(int i) const
{
	if (i < 0 || i >= (int)vecsol.size())
		throw HMMPI::Exception("Индекс вне диапазона в KW_LinSolver::Sol", "Index out of range in KW_LinSolver::Sol");

	return vecsol[i];
}
//------------------------------------------------------------------------------------------
KW_LinSolver::KW_LinSolver()
{
	name = "LINSOLVER";
	DEFPAR(sol1, "GAUSS");

	FinalizeParams();
	EXPECTED[0] = std::vector<std::string>{"GAUSS", "DGESV", "DGELS", "DGELSD", "DGELSS", "DGELSY"};

	vecsol = std::vector<HMMPI::Solver*>(ecols, 0);
}
//------------------------------------------------------------------------------------------
void KW_LinSolver::UpdateParams() noexcept
{
	for (int i = 0; i < ecols; i++)
	{
		std::string S = *(std::string*)DATA[i];
		if (S == "GAUSS")
			vecsol[i] = new HMMPI::SolverGauss();
		else if (S == "DGESV")
			vecsol[i] = new HMMPI::SolverDGESV();
		else if (S == "DGELS")
			vecsol[i] = new HMMPI::SolverDGELS();
		else if (S == "DGELSD")
			vecsol[i] = new HMMPI::SolverDGELSD();
		else if (S == "DGELSS")
			vecsol[i] = new HMMPI::SolverDGELSS();
		else if (S == "DGELSY")
			vecsol[i] = new HMMPI::SolverDGELSY();
		else
			vecsol[i] = 0;
			// throw HMMPI::Exception(HMMPI::stringFormatArr("Нераспознанный солвер {0:%s}", "Solver {0:%s} not recognized", S));  don't throw exception because of 'noexcept'
	}
}
//------------------------------------------------------------------------------------------
KW_LinSolver::~KW_LinSolver()
{
	for (auto &s : vecsol)
		delete s;
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_MCMC_config::KW_MCMC_config()
{
	name = "MCMC_CONFIG";

	DEFPAR(sampler, "HMC");
	DEFPAR(iter, 1000);
	DEFPAR(burn_in, 100);
	DEFPAR(seed, 0);
	DEFPAR(MM_type, "FI");
	DEFPAR(nu, 0.0);
	DEFPAR(gamma, 1.0);
	DEFPAR(LFG_maxref, 10);
	DEFPAR(eps, 1e-3);
	DEFPAR(maxeps, 0.1);
	DEFPAR(LF_steps, 100);
	DEFPAR(LF_bounce, "HT");
	DEFPAR(upd_freq, 100);
	DEFPAR(upd_type, 0);
	DEFPAR(acc_targ, 0.7);
	DEFPAR(alpha, 1);
	DEFPAR(beta, 10);
	DEFPAR(I_alpha, 1.0);
	DEFPAR(ii, 0.7);				// for SOL-HMC

	FinalizeParams();
	EXPECTED[0] = std::vector<std::string>{"RWM", "PCN", "HMC", "HMCREJ", "SOLHMC", "RHMC", "MMALA", "SIMPLMMALA", "MMALA2", "I_MALA"};
	EXPECTED[4] = std::vector<std::string>{"HESS", "FI", "UNITY", "MAT", "BFGS"};
	EXPECTED[11] = std::vector<std::string>{"NEG", "CHOL", "EIG", "HT"};
}
//------------------------------------------------------------------------------------------
HMMPI::MCMC *KW_MCMC_config::MakeSampler(KW_item *kw)
{
	Start_pre();
	IMPORTKWD(physmodel, KW_physmodel, "PHYSMODEL");
	DECLKWD(mat, KW_mat, "MAT");
	DECLKWD(config, KW_proxy, "PROXY_CONFIG");
	DECLKWD(proxy_dump, KW_proxy_dump, "PROXY_DUMP");
	Finish_pre();

	DECLKWD(params, KW_parameters, "PARAMETERS");
	const ParamsInterface *par_interface = params->GetParamsInterface();	// used to check the bounds of initial point

	if (upd_type != 0 && upd_type != 1)
		throw HMMPI::Exception("upd_type should be 0 or 1 in " + name);

	HMMPI::MCMC *Sampler = 0;			// result
	K->AppText("Sampler: " + sampler + "\n");

	HMMPI::Rand rg(seed);
	K->AppText(HMMPI::stringFormatArr("Сид = {0:%u}\n", "Seed = {0:%u}\n", rg.Seed()));

	HMMPI::EpsUpdate1 EUpd(acc_targ, alpha, beta, false);

	std::string msg0;
	PhysModel *PM_main = Factory.Make(msg0, K, kw, this->CWD, 1);
	K->AppText((std::string)HMMPI::MessageRE("Модель для L: ", "Model for likelihood: ") + msg0 + "\n" + PM_main->proc_msg());

	HMMPI::Mat cov;						// COV matrix for RWM proposal, or CONST_MM for HMC
	if (mat->M.Length() != 0)
	{
		if (mat->M.ICount() != (size_t)PM_main->ParamsDim() || mat->M.JCount() != (size_t)PM_main->ParamsDim())
			throw HMMPI::Exception(HMMPI::stringFormatArr("В MAT ожидается квадратная матрица {0:%d} x {0:%d}, либо пустая матрица",
														  "Square {0:%d} x {0:%d} matrix, or empty matrix, is expected in MAT", PM_main->ParamsDim()));
		cov = PM_main->act_mat(mat->M);
	}

	if (sampler == "RWM")
	{
		K->AppText(HMMPI::stringFormatArr("Шаг eps = {0:%g}\n", "Step eps = {0:%g}\n", eps));
		if (mat->M.Length() != 0)
		{
			K->AppText(HMMPI::stringFormatArr(HMMPI::MessageRE(
					"В proposal используется ковариационная матрица MAT[{0:%ld}, {1:%ld}] (верхнетреугольная часть)\n",
					"Covariance matrix MAT[{0:%ld}, {1:%ld}] (upper-triangle part) is used in proposal\n"), std::vector<size_t>{cov.ICount(), cov.JCount()}));
		}
		else
			K->AppText(HMMPI::stringFormatArr("В proposal используется единичная ковариационная матрица {0:%d} x {0:%d}\n",
											  "Unity covariance matrix {0:%d} x {0:%d} is used in proposal\n", PM_main->ParamsDim_ACT()));

		Sampler = new HMMPI::RWM1(PM_main, rg, EUpd, burn_in, upd_freq, config->select_pts, eps, std::move(cov));			// *** DELETE *** in the end!
	}
	else if (sampler == "PCN")
	{
		K->AppText(HMMPI::stringFormatArr("eps = {0:%g}\n", std::vector<double>{eps}));
		Sampler = new HMMPI::pCN1(PM_main, rg, EUpd, burn_in, upd_freq, config->select_pts, eps);							// *** DELETE *** in the end!
	}
	else if (sampler == "HMC" || sampler == "HMCREJ" || sampler == "SOLHMC")
	{
		HMMPI::EpsUpdate1 EUpd_inner(acc_targ, alpha, beta, true);
		if (physmodel->type.size() < 2)
			throw HMMPI::Exception("По крайней мере 2 модели должны быть заданы в PHYSMODEL", "At least 2 models should be defined in PHYSMODEL");

		PhysModel *PM_grad = Factory.Make(msg0, K, kw, this->CWD, 2);
		K->AppText((std::string)HMMPI::MessageRE("Модель для приближения: ", "Model for approximation: ") + msg0 + "\n" + PM_grad->proc_msg());

		if (!PM_grad->is_proxy())
			K->AppText((std::string)HMMPI::MessageRE("Модель для приближения не является [DATA]PROXY, она не будет тренироваться в ",
													 "Model for approximation is not a [DATA]PROXY, it will not be trained in ") + sampler + "\n");

		HMMPI::LeapFrog LF(MPI_COMM_WORLD, nu, eps, gamma, LF_bounce, MM_type, std::move(cov));
		if (sampler == "HMC")
			Sampler = new HMMPI::HMC1(PM_main, PM_grad, rg, LF, EUpd_inner, burn_in, LF_steps, maxeps, upd_freq, config->select_pts, config->ind_grad_add_pts, config->ind_grad_comps, proxy_dump->vec_dump_inds, upd_type);		// *** DELETE *** in the end!
		else if (sampler == "HMCREJ")
			Sampler = new HMMPI::HMCrej(PM_main, PM_grad, rg, LF, EUpd_inner, burn_in, LF_steps, maxeps, upd_freq, config->select_pts, config->ind_grad_add_pts, config->ind_grad_comps, proxy_dump->vec_dump_inds, upd_type);		// *** DELETE *** in the end!
		else	// SOLHMC
		{
			if (mat->M.Length() == 0)
				throw HMMPI::Exception("For SOL-HMC, covariance matrix should be defined in MAT");
			if (MM_type != "MAT")
				throw HMMPI::Exception("For SOL-HMC, MCMC_CONFIG.MM_type should be MAT");
			Sampler = new HMMPI::SOL_HMC(PM_main, PM_grad, rg, LF, EUpd_inner, burn_in, LF_steps, maxeps, upd_freq, config->select_pts, config->ind_grad_add_pts, config->ind_grad_comps, proxy_dump->vec_dump_inds, upd_type, ii);		// *** DELETE *** in the end!
		}

		//Sampler = new HMMPI::HMC1(PM_grad, PM_grad, rg, LF, EUpd, LF_steps, upd_freq);	// DEBUG use PM_grad both for approximation and for MH
		// DEBUG
		//PhysModel *PM_grad_debug = Factory.Make(msg0, K, this, this->CWD, 3);	// DEBUG
		//K->AppText("Model for gradients (DEBUG): " + msg0 + "\n");		// DEBUG
		//dynamic_cast<HMMPI::HMC1*>(Sampler)->pm_grads = PM_grad_debug;	// DEBUG
		// DEBUG

	}
	else if (sampler == "RHMC")
	{
		Start_pre();
		IMPORTKWD(opt, KW_optimization, "OPTIMIZATION");
		Finish_pre();

		if (physmodel->type.size() < 2)
			throw HMMPI::Exception("По крайней мере 2 модели должны быть заданы в PHYSMODEL", "At least 2 models should be defined in PHYSMODEL");

		PhysModel *PM_grad = Factory.Make(msg0, K, kw, this->CWD, 2);
		K->AppText((std::string)HMMPI::MessageRE("Модель для приближения: ", "Model for approximation: ") + msg0 + "\n" + PM_grad->proc_msg());

		if (!PM_grad->is_proxy())
			K->AppText(HMMPI::MessageRE("Модель для приближения не является [DATA]PROXY, она не будет тренироваться в RHMC\n",
										"Model for approximation is not a [DATA]PROXY, it will not be trained in RHMC\n"));

		NonlinearSystemSolver *sol = opt->MakeNonlinSolver();
		Sampler = new HMMPI::RHMC1(PM_main, PM_grad, rg, sol, EUpd, burn_in, LF_steps, nu, eps, maxeps, upd_freq, config->select_pts, LFG_maxref, config->ind_grad_add_pts, config->ind_grad_comps, proxy_dump->vec_dump_inds, upd_type);		// *** DELETE *** in the end!
	}
	else if (sampler == "MMALA" || sampler == "SIMPLMMALA" || sampler == "MMALA2" || sampler == "I_MALA")
	{
		HMMPI::EpsUpdate1 EUpd_inner(acc_targ, alpha, beta, true);
		if (physmodel->type.size() < 2)
			throw HMMPI::Exception("По крайней мере 2 модели должны быть заданы в PHYSMODEL", "At least 2 models should be defined in PHYSMODEL");

		PhysModel *PM_grad = Factory.Make(msg0, K, kw, this->CWD, 2);
		K->AppText((std::string)HMMPI::MessageRE("Модель для приближения: ", "Model for approximation: ") + msg0 + "\n" + PM_grad->proc_msg());

		if (!PM_grad->is_proxy())
			K->AppText(HMMPI::MessageRE("Модель для приближения не является [DATA]PROXY, она не будет тренироваться в " + sampler + "\n",
										"Model for approximation is not a [DATA]PROXY, it will not be trained in " + sampler + "\n"));
		int Type = -1;
		if (sampler == "MMALA")
			Type = 0;
		else if (sampler == "SIMPLMMALA")
			Type = 1;
		else if (sampler == "MMALA2")
			Type = 2;

		if (sampler == "I_MALA")
			Sampler = new HMMPI::I_MALA(PM_main, PM_grad, rg, EUpd_inner, burn_in, upd_freq, config->select_pts, nu, eps, maxeps, LF_steps, Type, config->ind_grad_add_pts, config->ind_grad_comps, proxy_dump->vec_dump_inds, upd_type, I_alpha);	// *** DELETE *** in the end!
		else
			Sampler = new HMMPI::MMALA(PM_main, PM_grad, rg, EUpd_inner, burn_in, upd_freq, config->select_pts, nu, eps, maxeps, LF_steps, Type, config->ind_grad_add_pts, config->ind_grad_comps, proxy_dump->vec_dump_inds, upd_type);			// *** DELETE *** in the end!
	}
	else
		throw HMMPI::Exception(HMMPI::stringFormatArr("Нераспознанный sampler '{0:%s}' в KW_MCMC_config::MakeSampler",
													  "Unrecognized sampler '{0:%s}' in KW_MCMC_config::MakeSampler", sampler));

	std::vector<double> p = par_interface->get_init_act();		// check initial point
	if (!PM_main->CheckLimits_ACT(p))
	{
		delete Sampler;
		Sampler = 0;
		throw HMMPI::Exception(PM_main->limits_msg);
	}

	return Sampler;
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
HMMPI::Mat KW_corrstruct::Corr()
{
	HMMPI::Mat res;
	std::vector<HMMPI::Mat> Blocks = CorrBlocks();

	// make the full matrix from the blocks
	for (size_t i = 0; i < Blocks.size(); i++)
	{
		HMMPI::Mat oneBlock;
		if (R[i] > 0.01)
			oneBlock = Blocks[i];						// dense case
		else
			oneBlock = HMMPI::Mat(Blocks[i].ICount());	// diagonal case - unity matrix

		HMMPI::Mat N(res.ICount(), oneBlock.JCount(), 0);		// zero padding
		res = (res && N)||(N.Tr() && oneBlock);
	}

	return res;
}
//------------------------------------------------------------------------------------------
std::vector<HMMPI::Mat> KW_corrstruct::CorrBlocks() const
{
	std::vector<HMMPI::Mat> res(Bsize.size());
	for (size_t i = 0; i < Bsize.size(); i++)
	{
		if (R[i] > 0.01)
		{
			HMMPI::Mat M1(Bsize[i], Bsize[i], 0);		// distance matrix
			auto f1 = [](int x, int y, double d){return abs(x - y);};
			M1.FuncInd(f1);								// compiler has no problem here (no error)

			M1 = (1/R[i]) * std::move(M1);

			HMMPI::Func1D *F1D = HMMPI::Func1D_factory::New(type[i]);
			auto f2 = [F1D](double d){return F1D->f(d);};
			M1.Func(f2);

			res[i] = M1;
			delete F1D;
		}
		else
			res[i] = HMMPI::Mat(Bsize[i], 1, 1.0);
	}

	return res;
}
//------------------------------------------------------------------------------------------
KW_corrstruct::KW_corrstruct()
{
	dec_verb = -1;

	name = "CORRSTRUCT";
	DEFPARMULT(Bsize);
	DEFPARMULT(R);
	DEFPARMULT(type);

	FinalizeParams();
	EXPECTED[2] = std::vector<std::string>{"GAUSS", "SPHER", "EXP"};
}
//------------------------------------------------------------------------------------------
void KW_corrstruct::UpdateParams() noexcept
{
	int countB = 0, countR = 0;
	for (size_t i = 0; i < Bsize.size(); i++)
	{
		if (Bsize[i] < 0)
		{
			Bsize[i] = 0;
			countB++;
		}
		if (R[i] <= 0)
		{
			R[i] = 0.01;
			countR++;
		}
	}

	if (countB > 0)
		K->AppText(HMMPI::stringFormatArr("Поскольку Bsize < 0 не допустим, он был заменен на 0 в {0:%d} строк(ах)\n",
										  "Since Bsize < 0 is not allowed, it was replaced by 0 in {0:%d} line(s)\n", countB));
	if (countR > 0)
		K->AppText(HMMPI::stringFormatArr("Поскольку R <= 0 не допустим, он был заменен на 0.01 в {0:%d} строк(ах)\n",
										  "Since R <= 0 is not allowed, it was replaced by 0.01 in {0:%d} line(s)\n", countR));
}
//------------------------------------------------------------------------------------------
int KW_corrstruct::size()
{
	int res = 0;
	for (const auto &i : Bsize)
		res += i;

	return res;
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
std::string KW_physmodel::CheckRefs()
{
	// check ranges
	for (size_t i = 0; i < ref.size(); i++)
		if (ref[i] < 0 || ref[i] > int(ref.size()))
			return HMMPI::stringFormatArr(HMMPI::MessageRE(
					"В PHYSMODEL в строке {0:%d} ref = {1:%d} вне допустимого диапазона [{2:%d}, {3:%d}]",
					"In PHYSMODEL, line {0:%d}, ref = {1:%d} is out of range [{2:%d}, {3:%d}]"), std::vector<int>{(int)i+1, ref[i], 0, (int)ref.size()});

	// check loops starting from each row
	for (size_t i = 0; i < ref.size(); i++)
	{
		int r = ref[i];
		size_t j;
		for (j = 0; j < ref.size(); j++)
		{
			if (r == 0)
				break;
			else
				r = ref[r-1];
		}
		if (j >= ref.size())
			return HMMPI::stringFormatArr("В PHYSMODEL найдена петля, начинающаяся в строке {0:%d}", "In PHYSMODEL, there is a loop starting in line {0:%d}", i+1);
	}

	// fill 'is_plain'
	is_plain = std::vector<bool>(type.size());
	for (size_t i = 0; i < type.size(); i++)
		if (type[i] == "ECLIPSE" || type[i] == "SIMECL" || type[i] == "PCONNECT" || type[i] == "CONC" || type[i] == "SIMPROXY" || type[i] == "LIN" || type[i] == "ROSEN")
			is_plain[i] = true;
		else
			is_plain[i] = false;

	// check "ref" validity for different types
	for (size_t i = 0; i < type.size(); i++)
	{
		if (is_plain[i] && ref[i] != 0)
			return HMMPI::stringFormatArr("В PHYSMODEL для модели типа {0:%s} должен быть ref == 0", "In PHYSMODEL, type {0:%s} should correspond to ref == 0", type[i]);
		if (!is_plain[i] && ref[i] == 0)
			return HMMPI::stringFormatArr("В PHYSMODEL модель типа {0:%s} не согласуется с ref == 0", "In PHYSMODEL, type {0:%s} is not consistent with ref == 0", type[i]);

		if ((type[i] == "KRIGCORR" || type[i] == "KRIGSIGMA") && type[ref[i]-1] != "PROXY")
			return HMMPI::stringFormatArr("В PHYSMODEL модель типа {0:%s} должна ссылаться на PROXY", "In PHYSMODEL, model of type {0:%s} should refer to PROXY", type[i]);
		if ((type[i] == "DATAPROXY" || type[i] == "DATAPROXY2" || type[i] == "PROXY") && !is_plain[ref[i]-1])
			return HMMPI::stringFormatArr("В PHYSMODEL модель типа {0:%s} должна ссылаться на 'простой тип'", "In PHYSMODEL, model of type {0:%s} should refer to 'plain type'", type[i]);
		if (type[i] == "NUMGRAD" && type[ref[i]-1] == "NUMGRAD")
			return HMMPI::stringFormatArr("В PHYSMODEL модель типа {0:%s} не должна ссылаться на NUMGRAD", "In PHYSMODEL, model of type {0:%s} should not refer to NUMGRAD", type[i]);
	}

	return "";
}
//------------------------------------------------------------------------------------------
KW_physmodel::KW_physmodel()
{
	name = "PHYSMODEL";

	DEFPARMULT(type);
	DEFPARMULT(ref);

	FinalizeParams();

	EXPECTED[0] = std::vector<std::string>{"ECLIPSE", "SIMECL", "PCONNECT", "CONC", "SIMPROXY", "LIN", "ROSEN", "NUMGRAD", "PROXY", "DATAPROXY", "DATAPROXY2", "KRIGCORR", "KRIGSIGMA", "LAGRSPHER", "SPHERICAL", "CUBEBOUND", "HAMILTONIAN", "POSTERIOR"};
}
//------------------------------------------------------------------------------------------
void KW_physmodel::UpdateParams() noexcept
{
	std::string msg = CheckRefs();
	if (msg != "")
		SilentError(msg);
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_vectmodel::KW_vectmodel() : mod(nullptr)
{
	name = "VECTMODEL";

	DEFPAR(type, "GRADIENT");
	FinalizeParams();

	EXPECTED[0] = std::vector<std::string>{"GRADIENT", "HAM_EQ1", "HAM_EQ2", "HAM_EQ2_EPS", "HAM_EQ2_EPSFULL"};
}
//------------------------------------------------------------------------------------------
VectorModel *KW_vectmodel::Make(PhysModel *pm)	// returned model is freed automatically in the end
{
	delete mod;
	mod = nullptr;
	PM_FullHamiltonian *pm_ham = dynamic_cast<PM_FullHamiltonian*>(pm);

	if (type == "GRADIENT")									// model associated with gradient of objective function
		mod = new VM_gradient(pm);
	else if (type == "HAM_EQ1" || type == "HAM_EQ2" || type == "HAM_EQ2_EPS" || type == "HAM_EQ2_EPSFULL")		// models associated with Hamiltonian equations #1 and #2 in RHMC
	{
		Start_pre();
		IMPORTKWD(points, KW_3points, "3POINTS");			// 3POINTS->x,y provide 'x', 'p0' for VM_Ham_eq1; 'x0', 'p' for VM_Ham_eq2[_eps]
		IMPORTKWD(opt, KW_optimization, "OPTIMIZATION");	// epsG provides 'eps', epsF provides 'M0', maa provides 'i0'
		Finish_pre();

		if (pm_ham == nullptr)
			throw HMMPI::Exception("Для " + type + " требуется PHYSMODEL HAMILTONIAN", type + " requires PHYSMODEL HAMILTONIAN");

		if (type == "HAM_EQ1")
		{
			VM_Ham_eq1 *vm = new VM_Ham_eq1(pm_ham);			// x = 3POINTS->x, p0 = 3POINTS->y, p = PARAMS, eps = OPTIMIZATION->epsG
			vm->x = pm_ham->act_par(points->x);
			vm->p0 = pm_ham->act_par(points->y);
			vm->eps = opt->epsG;
			mod = vm;
		}
		else if (type == "HAM_EQ2")
		{
			VM_Ham_eq2 *vm = new VM_Ham_eq2(pm_ham, pm_ham);	// x0 = 3POINTS->x, p = 3POINTS->y, x = PARAMS, eps = OPTIMIZATION->epsG; NOTE: same model is used for Ham0, Ham1, so no efficient cache reuse will take place
			vm->x0 = pm_ham->act_par(points->x);
			vm->p = pm_ham->act_par(points->y);
			vm->eps = opt->epsG;
			mod = vm;
		}
		else if (type == "HAM_EQ2_EPS")
		{
			VM_Ham_eq2 vm2(pm_ham, pm_ham);						// x0 = 3POINTS->x, p = 3POINTS->y, x = PARAMS, i0 = OPTIMIZATION->maa, M0 = OPTIMIZATION->epsF; NOTE: same model is used for Ham0, Ham1, so no efficient cache reuse will take place
			vm2.x0 = pm_ham->act_par(points->x);
			vm2.p = pm_ham->act_par(points->y);
			vm2.eps = 0;

			VM_Ham_eq2_eps *vmeps = new VM_Ham_eq2_eps(vm2, opt->maa, opt->epsF);
			mod = vmeps;
		}
		else	// HAM_EQ2_EPSFULL
		{
			VM_Ham_eq2 vm2(pm_ham, pm_ham);						// x0 = 3POINTS->x, p = 3POINTS->y, x = PARAMS, eps = OPTIMIZATION->epsG, i0 = OPTIMIZATION->maa, M0 = OPTIMIZATION->epsF; NOTE: same model is used for Ham0, Ham1, so no efficient cache reuse will take place
			vm2.x0 = pm_ham->act_par(points->x);
			vm2.p = pm_ham->act_par(points->y);
			vm2.eps = opt->epsG;

			VM_Ham_eq2_eps_full *vmepsfull = new VM_Ham_eq2_eps_full(vm2, opt->maa, opt->epsF);
			mod = vmepsfull;
		}
	}
	else
		throw HMMPI::Exception("Unrecognized type " + type + " in KW_vectmodel::Make");

	return mod;
}
//------------------------------------------------------------------------------------------
