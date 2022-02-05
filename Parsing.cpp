/*
 * Parsing.cpp
 *
 *  Created on: Mar 18, 2013
 *      Author: ilya
 */

#include "Utils.h"
#include "Abstract.h"
#include "Parsing.h"
#include "Parsing2.h"
#include "Tracking.h"
#include "mpi.h"
#include "ConsoleCntr.h"
#include <string>
#include <exception>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cassert>
#include <sstream>


std::string Parser_1::InitCWD;
int Parser_1::verbosity;
int Parser_1::MPI_rank;
int Parser_1::MPI_size;

//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
// DataLines
//------------------------------------------------------------------------------------------
void DataLines::LoadFromFile(std::string fname)		// (OK)
{
	RANK0_SYNCERR_BEGIN(MPI_COMM_WORLD)
		std::ifstream sr;
		sr.exceptions(std::ios_base::badbit);

		try
		{
			sr.open(fname.c_str());
			if (sr.fail())
				throw HMMPI::Exception((std::string)"Невозможно открыть " + fname,
									   (std::string)"Failed opening " + fname);

			std::string line;
			while (!sr.eof())
			{
				getline(sr, line);
				lines.push_back(line);
			}
			sr.close();
		}
		catch (...)
		{
			if (sr.is_open())
				sr.close();
			throw;
		}
	RANK0_SYNCERR_END(MPI_COMM_WORLD)

	HMMPI::Bcast_vector(lines, 0, MPI_COMM_WORLD);
}
//------------------------------------------------------------------------------------------
std::vector<std::string> DataLines::EliminateEmpty()	// (OK)
{
	std::string trim = " \t\r";
	std::vector<std::string> res;
	for (size_t i = 0; i < lines.size(); i++)
	{
		size_t ind = lines[i].find("--");
		if (ind != std::string::npos)
			lines[i] = lines[i].substr(0, ind);			// remove comments

		std::string tr_ln = HMMPI::Trim(lines[i], trim);
		tr_ln = HMMPI::Replace(tr_ln, "-\\-", "--");	// treat special '--'

		if (tr_ln.size() > 0)  							// check the line is not empty
            res.push_back(tr_ln);
	}
	return res;
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
void CheckFileOpen(std::string fname)		// stand-alone function
{
	std::ifstream sr;
	sr.exceptions(std::ios_base::badbit);
	try
	{
		sr.open(fname.c_str());
		if (sr.fail())
			throw HMMPI::Exception("Невозможно открыть файл " + fname, "Cannot open file " + fname);
		sr.close();
	}
	catch (...)
	{
		if (sr.is_open())
			sr.close();
		throw;
	}
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
// KW_item
//------------------------------------------------------------------------------------------
std::string KW_item::ProblemKW()	// (OK)
{
	std::string res = "";

	assert(prerequisites.size() == kws_ok.size());

	for (size_t i = 0; i < prerequisites.size(); i++)
		if (!kws_ok[i])
		{
			if (res.size() > 0)
				res += ", ";
			res += prerequisites[i];
		}

	return res;
}
//------------------------------------------------------------------------------------------
std::string KW_item::ProblemKW_long()	// (OK)
{
	std::string res = "";

	assert(prerequisites.size() == kws_ok.size());

	for (size_t i = 0; i < prerequisites.size(); i++)
		if (!kws_ok[i])
			res += (std::string)"* " + prerequisites[i] + ": " + K->GetKW_item(prerequisites[i])->GetState();

	return res;
}
//------------------------------------------------------------------------------------------
bool KW_item::CheckKW()		// (OK)
{
	size_t L = prerequisites.size();
	kws_ok = std::vector<bool>(L);
	bool res = true;

	for (size_t i = 0; i < L; i++)
	{
		KW_item *k = K->GetKW_item(prerequisites[i]);
		assert (k != 0);

		if (k->GetState() == "")
			kws_ok[i] = true;
		else
		{
			kws_ok[i] = false;
			res = false;
		}
	}

	return res;
}
//------------------------------------------------------------------------------------------
bool KW_item::ReportKWProblem()		// (OK)
{
	if (CheckKW())
		return true;
	else
	{
		 K->AppText(std::string(80, '*'));
		 if (K->verbosity - dec_verb <= -1)
			 K->AppText(HMMPI::stringFormatArr(HMMPI::MessageRE("\nНевозможно выполнить {0:%s} из-за проблем с обязательными кл. словами: {1:%s}\n",
											   	   	   	   	    "\nCannot process {0:%s} due to problems with prerequisite keywords: {1:%s}\n"), std::vector<std::string>{name, ProblemKW()}));
		 else
			 K->AppText(HMMPI::stringFormatArr(HMMPI::MessageRE("\nНевозможно выполнить {0:%s} из-за проблем с обязательными кл. словами:\n{1:%s}",
											   	   	   	   	    "\nCannot process {0:%s} due to problems with prerequisite keywords:\n{1:%s}"), std::vector<std::string>{name, ProblemKW_long()}));
		 K->AppText(std::string(80, '*') + "\n");
		 return false;
	}
}
//------------------------------------------------------------------------------------------
void KW_item::AddState(std::string s)
{
	if (*--s.end() == '\n')
		state += s;
	else
		state += s + "\n";
}
//------------------------------------------------------------------------------------------
void KW_item::SilentError(std::string s)
{
	K->AppText(HMMPI::stringFormatArr("ОШИБКА: {0:%s}\n", "ERROR: {0:%s}\n", s));
	AddState(s);
	K->TotalErrors++;
}
//------------------------------------------------------------------------------------------
void KW_item::Start_pre() noexcept
{
	prerequisites = std::vector<std::string>();
}
//------------------------------------------------------------------------------------------
void KW_item::Add_pre(std::string p) noexcept
{
	prerequisites.push_back(p);
}
//------------------------------------------------------------------------------------------
void KW_item::Finish_pre()
{
	if (!ReportKWProblem())
		throw HMMPI::Exception("Проблемы с обязательными кл. словами", "Problems with prerequisite keywords");
}
//------------------------------------------------------------------------------------------
int KW_item::CheckDefault(std::string s)	// (OK)
{
	if (*--s.end() == '*')
		return HMMPI::StoL(s.substr(0, s.length()-1));
	else
		return 0;
}
//------------------------------------------------------------------------------------------
std::vector<std::string> KW_item::ReadDefaults(std::vector<std::string> p, int padto, int &def_count)	// (OK)
{
	std::vector<std::string> res;

	// main pass
	for (size_t i = 0; i < p.size(); i++)
	{
		int def = CheckDefault(p[i]);
		if (def == 0)
			res.push_back(p[i]);
		else
			for (int j = 0; j < def; j++)
			{
				res.push_back("");
				def_count++;
			}
	}

	// right padding
	int pad = padto - (int)res.size();
	for (int i = 0; i < pad; i++)
	{
		res.push_back("");
		def_count++;
	}

	return res;
}
//------------------------------------------------------------------------------------------
void KW_item::ReadParamTable(const std::vector<std::string> &SA) noexcept	// (OK)
{
	size_t Nrows = 0, Ncols = 0;
	int def_count = 0;

	// define the table size
	if (erows != -1)
		Nrows = erows;
	else
		Nrows = SA.size();

	if (ecols != -1)
		Ncols = ecols;
	else
	{
		if (SA.size() > 0)		// Ncols is taken according to the first line
		{
			std::vector<std::string> ARR;
			std::string tp = HMMPI::Trim(SA[0], trim);	// Trim first, because delim may be empty
			HMMPI::tokenize(tp, ARR, delim, true);
			ARR = ReadDefaults(ARR, 0, def_count);		// padto=0 to exclude padding, def_count is a dummy
			Ncols = ARR.size();
		}
		else
			Ncols = 0;
	}

	// initialize the table
	par_table = HMMPI::Vector2<std::string>(Nrows, Ncols, "");
	def_count = 0;				// initialize for proper counting the default params

	// tokenize all lines
	for (size_t r = 0; r < Nrows; r++)
	{
		std::string line = "";
		if (r < SA.size())
			line = SA[r];		// the other rows in par_table (r >= SA.size()) will remain empty

		// print the loaded lines and/or their count
		if (Nrows == 1 && K->verbosity - dec_verb >= 1)		// one row is loaded
			K->AppText(HMMPI::stringFormatArr("Параметры: {0:%s}\n", "Parameters: {0:%s}\n", line));
		if (Nrows > 1)										// many rows are loaded
		{
			if (r == 0)
			{
				if (K->verbosity - dec_verb >= 2)
				{
					char buff[HMMPI::BUFFSIZE], buffrus[HMMPI::BUFFSIZE];
					sprintf(buffrus, "Параметры (%d стр.):\n", (int)Nrows);
					sprintf(buff, "Parameters (%d lines):\n", (int)Nrows);
					K->AppText(HMMPI::MessageRE(buffrus, buff));
					K->AppText((std::string)"\t" + line + "\n");
				}
				else if (K->verbosity - dec_verb >= 0)
					K->AppText(HMMPI::stringFormatArr("Загружено {0:%d} стр.\n", "Loaded {0:%d} lines\n", Nrows));
			}
			else if (K->verbosity - dec_verb >= 2)
				K->AppText((std::string)"\t" + line + "\n");
		}

		std::vector<std::string> ARR;
		std::string tp = HMMPI::Trim(line, trim);	// Trim first, because delim may be empty
		HMMPI::tokenize(tp, ARR, delim, true);
		ARR = ReadDefaults(ARR, Ncols, def_count);	// padto=NCols, so padding is active; def_count is counting now

		if (ARR.size() > Ncols)
		{
			char buff[HMMPI::BUFFSIZE], buffrus[HMMPI::BUFFSIZE];
			sprintf(buffrus, "Слишком много значений (%d, при максимально допустимых %d) в строке %d", (int)ARR.size(), (int)Ncols, (int)r+1);
			sprintf(buff, "Too many values (%d, whereas maximum %d are acceptable) in row %d", (int)ARR.size(), (int)Ncols, (int)r+1);
			SilentError(HMMPI::MessageRE(buffrus, buff));

			continue;			// error is not thrown
		}

		for (size_t c = 0; c < ARR.size(); c++)
			par_table(r, c) = ARR[c];
	}

	if (def_count > 0)
		K->AppText(HMMPI::stringFormatArr("Использовано значений по умолчанию: {0:%d}\n", "Default values used: {0:%d}\n", def_count));
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
// KW_run
//------------------------------------------------------------------------------------------
void KW_run::Action() noexcept	// (OK)
{
	if ((name == "ECHO")||(name == "NOECHO"))
		return;

	try
	{
		K->AppText(HMMPI::stringFormatArr("Выполняется {0:%s}...\n", "Executing {0:%s}...\n", name));

		Run();
	}
	catch (const HMMPI::Exception &e)
	{
		K->AppText(HMMPI::stringFormatArr(HMMPI::MessageRE("ОШИБКА во время выполнения {0:%s}: {1:%s}\n\n",
											  	  	  	    "ERROR while running {0:%s}: {1:%s}\n\n"), std::vector<std::string>{name, e.what()}));
		AddState(e.what());
		K->TotalErrors++;
	}
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
// KW_fname
//------------------------------------------------------------------------------------------
void KW_fname::ProcessParamTable() noexcept	// (OK)
{
	assert(ecols == 1);

	size_t L = par_table.ICount();
	fnames = std::vector<std::string>(L);
	for (size_t i = 0; i < L; i++)
		fnames[i] = par_table(i, 0);

	AllocateData();
}
//------------------------------------------------------------------------------------------
void KW_fname::Action()	noexcept	// (OK)
{
	size_t L = fnames.size();
	for (size_t i = 0; i < L; i++)
	{
		try
		{
			if (fnames[i] == "")
			{
				char buff[HMMPI::BUFFSIZE];
				if (L > 1)
					sprintf(buff, "File name not specified in row %d", (int)i+1);
				else
					sprintf(buff, "File name not specified");
				throw HMMPI::Exception(buff);
			}
			DataIO(i);
		}
		catch (const HMMPI::Exception &e)
		{
			SilentError(e.what());
			K->AppText("\n");
		}
	}
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
// KW_fwrite
//------------------------------------------------------------------------------------------
void KW_fwrite::Action() noexcept	// (OK)
{
	if (K->MPI_rank == 0)	// only the master process performs writing
	{
		size_t L = fnames.size();
		for (size_t i = 0; i < L; i++)
		{
			try
			{
				if (fnames[i] == "")
				{
					char buff[HMMPI::BUFFSIZE];
					char buffrus[HMMPI::BUFFSIZE];
					if (L > 1)
					{
						sprintf(buff, "File name not specified in row %d", (int)i+1);
						sprintf(buffrus, "Не задано имя файла в строке %d", (int)i+1);
					}
					else
					{
						sprintf(buff, "File name not specified");
						sprintf(buffrus, "Не задано имя файла");
					}
					throw HMMPI::Exception(buffrus, buff);
				}
				DataIO(i);
			}
			catch (const HMMPI::Exception &e)
			{
				SilentError(e.what());
				K->AppText("\n");
			}
		}
	}
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
// KW_params
//------------------------------------------------------------------------------------------
void KW_params::AddParam(int *val, const char *pname)	// (OK)
{
	if (par_finalized)
		throw HMMPI::Exception("Attempt to AddParam() after finalization");

	DATA.push_back(val);
	TYPE.push_back(0);
	NAMES.push_back(pname);
}
//------------------------------------------------------------------------------------------
void KW_params::AddParam(double *val, const char *pname)	// (OK)
{
	if (par_finalized)
		throw HMMPI::Exception("Attempt to AddParam() after finalization");

	DATA.push_back(val);
	TYPE.push_back(1);
	NAMES.push_back(pname);
}
//------------------------------------------------------------------------------------------
void KW_params::AddParam(std::string *val, const char *pname)	// (OK)
{
	if (par_finalized)
		throw HMMPI::Exception("Attempt to AddParam() after finalization");

	DATA.push_back(val);
	TYPE.push_back(2);
	NAMES.push_back(pname);
}
//------------------------------------------------------------------------------------------
void KW_params::FinalizeParams()	// (OK)
{
	ecols = DATA.size();	// par_table not allocated yet
	EXPECTED = std::vector<std::vector<std::string>>(ecols);
	par_finalized = true;
}
//------------------------------------------------------------------------------------------
void KW_params::PrintParams()  noexcept
{
	std::string MSG = HMMPI::MessageRE("Текущие значения: ", "Current values: ");	// print all parameter values (these are the final values)
	for (size_t j = 0; j < par_table.JCount(); j++)
	{
		assert(TYPE[j] >=0 && TYPE[j] <= 2);

		char buff[HMMPI::BUFFSIZE];
		if (TYPE[j] == 0)
			sprintf(buff, "%d", *(int*)DATA[j]);
		else if (TYPE[j] == 1)
			sprintf(buff, "%g", *(double*)DATA[j]);
		else if (TYPE[j] == 2)
			sprintf(buff, "%s", ((std::string*)DATA[j])->c_str());

		MSG += NAMES[j] + " = " + buff;
		if (j < par_table.JCount()-1)
			MSG += ", ";
	}
	K->AppText(MSG + "\n");
}
//------------------------------------------------------------------------------------------
void KW_params::ProcessParamTable()  noexcept		// (OK)
{
	try
	{
		for (size_t j = 0; j < par_table.JCount(); j++)
		{
			assert(TYPE[j] >=0 && TYPE[j] <= 2);
			if (par_table(0, j) != "")
			{
				if (TYPE[j] == 0)
					*(int*)DATA[j] = HMMPI::StoL(HMMPI::Trim(par_table(0, j), " "));		// trim spaces in case the items are delimited by TAB
				else if (TYPE[j] == 1)
					*(double*)DATA[j] = HMMPI::StoD(HMMPI::Trim(par_table(0, j), " "));
				else if (TYPE[j] == 2)
				{
					if (EXPECTED[j].size() == 0)
						*(std::string*)DATA[j] = par_table(0, j);
					else
						*(std::string*)DATA[j] = HMMPI::ToUpper(par_table(0, j));

					std::string errmsg = CheckExpected(j);
					if (errmsg != "")
						throw HMMPI::Exception(errmsg);
				}
			}
		}
	}
	catch (...)
	{
		SilentError(make_err_msg());
	}
}
//------------------------------------------------------------------------------------------
void KW_params::Action()  noexcept	// (OK)
{
	assert(par_finalized);

	UpdateParams();
	if (GetState() != "")
		return;				// exit if error state is present

	PrintParams();
	FinalAction();
}
//------------------------------------------------------------------------------------------
std::string KW_params::make_err_msg()		// (OK)
{
	std::string msg = HMMPI::MessageRE("Найдены некорректные значения; ожидается:\n",
									   "Found incorrect values; expected:\n");
	for (size_t k = 0; k < par_table.JCount(); k++)
	{
		msg += NAMES[k] + " (";
		switch (TYPE[k])
		{
			case 0: msg += "INT"; break;
			case 1: msg += "DBL"; break;
			case 2: std::string aux = StrExpected(k);
				if (aux != "")
					msg += aux;
				else
					msg += "STR";
				break;
		}

		if (k < par_table.JCount()-1)
			msg += "), ";
		else
			msg += ")";
	}

	return msg;
}
//------------------------------------------------------------------------------------------
std::string KW_params::CheckExpected(int j)		// (OK)
{
	if (EXPECTED[j].begin() == EXPECTED[j].end())
		return "";	// no error

	if (std::find(EXPECTED[j].begin(), EXPECTED[j].end(), *(std::string*)DATA[j]) != EXPECTED[j].end())
		return "";	// no error
	else
		return (std::string)HMMPI::MessageRE("Ожидается ", "Expected ") + StrExpected(j);
}
//------------------------------------------------------------------------------------------
std::string KW_params::StrExpected(int j)		// (OK)
{
	std::string msg = "";
	for (size_t k = 0; k < EXPECTED[j].size(); k++)
	{
		msg += EXPECTED[j][k];
		if (k < EXPECTED[j].size() - 1)
			msg += ", ";
	}
	return msg;
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
// KW_multparams
//------------------------------------------------------------------------------------------
void KW_multparams::AddParam(std::vector<int> *val, const char *pname)
{
	if (par_finalized)
		throw HMMPI::Exception("Attempt to AddParam() after finalization");

	DATA.push_back(val);
	TYPE.push_back(0);
	NAMES.push_back(pname);
}
//------------------------------------------------------------------------------------------
void KW_multparams::AddParam(std::vector<double> *val, const char *pname)
{
	if (par_finalized)
		throw HMMPI::Exception("Attempt to AddParam() after finalization");

	DATA.push_back(val);
	TYPE.push_back(1);
	NAMES.push_back(pname);
}
//------------------------------------------------------------------------------------------
void KW_multparams::AddParam(std::vector<std::string> *val, const char *pname)
{
	if (par_finalized)
		throw HMMPI::Exception("Attempt to AddParam() after finalization");

	DATA.push_back(val);
	TYPE.push_back(2);
	NAMES.push_back(pname);
}
//------------------------------------------------------------------------------------------
std::string KW_multparams::CheckExpected(int i, int j)	// (OK)
{
	if (EXPECTED[j].begin() == EXPECTED[j].end())
		return "";	// no error

	if (std::find(EXPECTED[j].begin(), EXPECTED[j].end(), (*(std::vector<std::string>*)DATA[j])[i]) != EXPECTED[j].end())
		return "";	// no error
	else
		return (std::string)HMMPI::MessageRE("Ожидается ", "Expected ") + StrExpected(j);
}
//------------------------------------------------------------------------------------------
void KW_multparams::PrintParams() noexcept
{
	if (K->verbosity - dec_verb >= -1)		// report the parameter values
	{
		std::string MSG = HMMPI::MessageRE("Текущие значения:\n", "Current values:\n");

		int fmt_w1 = 1;						// formatting stuff
		if (par_table.ICount() > 10)
			fmt_w1 = 2;
		if (par_table.ICount() > 100)
			fmt_w1 = 3;
		if (par_table.ICount() > 1000)
			fmt_w1 = 4;
		if (par_table.ICount() > 10000)
			fmt_w1 = 5;

		HMMPI::StringListing SL("\t");

		for (size_t i = 0; i < par_table.ICount(); i++)
		{
			std::vector<std::string> line(par_table.JCount());
			for (size_t j = 0; j < par_table.JCount(); j++)
			{
				assert(TYPE[j] >= 0 && TYPE[j] <= 2);

				char buff[HMMPI::BUFFSIZE];
				if (TYPE[j] == 0)
					sprintf(buff, "%s%-*d = %-7d", NAMES[j].c_str(), fmt_w1, (int)i, (*(std::vector<int>*)DATA[j])[i]);
				else if (TYPE[j] == 1)
					sprintf(buff, "%s%-*d = %-7g", NAMES[j].c_str(), fmt_w1, (int)i, (*(std::vector<double>*)DATA[j])[i]);
				else if (TYPE[j] == 2)
					sprintf(buff, "%s%-*d = %-.*s", NAMES[j].c_str(), fmt_w1, (int)i, HMMPI::BUFFSIZE-5, (*(std::vector<std::string>*)DATA[j])[i].c_str());

				line[j] = buff;
			}
			SL.AddLine(line);
		}

		const int N = Parser_1::StrListN(dec_verb);
		K->AppText(MSG + SL.Print(N, N));
	}
}
//------------------------------------------------------------------------------------------
void KW_multparams::ProcessParamTable()  noexcept		// (OK)
{
	try
	{
		// allocate data vectors:
		// up to this point all DATA[i] point to empty vectors,
		// so make these vectors to have appropriate size
		for (size_t j = 0; j < par_table.JCount(); j++)
		{
			assert(TYPE[j] >= 0 && TYPE[j] <= 2);

			if (TYPE[j] == 0)
				*(std::vector<int>*)DATA[j] = std::vector<int>(par_table.ICount());
			else if (TYPE[j] == 1)
				*(std::vector<double>*)DATA[j] = std::vector<double>(par_table.ICount());
			else if (TYPE[j] == 2)
				*(std::vector<std::string>*)DATA[j] = std::vector<std::string>(par_table.ICount());
		}

		for (size_t i = 0; i < par_table.ICount(); i++)
		{
			for (size_t j = 0; j < par_table.JCount(); j++)
			{
				if (par_table(i, j) != "")		// non-default value
				{
					if (TYPE[j] == 0)
						(*(std::vector<int>*)DATA[j])[i] = HMMPI::StoL(HMMPI::Trim(par_table(i, j), " "));		// trim spaces in case the items are delimited by TAB
					else if (TYPE[j] == 1)
						(*(std::vector<double>*)DATA[j])[i] = HMMPI::StoD(HMMPI::Trim(par_table(i, j), " "));
					else if (TYPE[j] == 2)
					{
						if (EXPECTED[j].size() == 0)
							(*(std::vector<std::string>*)DATA[j])[i] = par_table(i, j);
						else
							(*(std::vector<std::string>*)DATA[j])[i] = HMMPI::ToUpper(par_table(i, j));

						std::string errmsg = CheckExpected(i, j);
						if (errmsg != "")
							throw HMMPI::Exception(errmsg);
					}
				}
				else				// default value: for all rows defaults are (int)0, (double)0, (string)"" / (string)expected[0]
				{
					if (TYPE[j] == 0)
						(*(std::vector<int>*)DATA[j])[i] = 0;
					else if (TYPE[j] == 1)
						(*(std::vector<double>*)DATA[j])[i] = 0.0;
					else if (TYPE[j] == 2)
					{
						if (EXPECTED[j].size() == 0)
							(*(std::vector<std::string>*)DATA[j])[i] = "";				// nothing expected - take ""
						else
							(*(std::vector<std::string>*)DATA[j])[i] = EXPECTED[j][0];	// something is expected - take the first one
					}
				}
			}
		}
	}
	catch (...)
	{
		SilentError(make_err_msg());
	}
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
// KW_pardouble
//------------------------------------------------------------------------------------------
KW_pardouble::KW_pardouble()	// (OK)
{
	erows = -1;
	ecols = 1;

	AddParam(&data, "data");
	FinalizeParams();
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
// KW_parint
//------------------------------------------------------------------------------------------
KW_parint::KW_parint()		// (OK)
{
	erows = -1;
	ecols = 1;

	AddParam(&data, "data");
	FinalizeParams();
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
// CTT_Keyword
//------------------------------------------------------------------------------------------
std::string CTT_Keyword::Tweak(std::string s) const
{
	std::string tok = "Keyword ";
	size_t tok_sz = tok.length();

	if (s.substr(0, tok_sz) == tok)
	{
		std::string s2 = s.substr(tok_sz);
		s = tok + TA->set_bold(true) + s2;
		s += TA->set_bold(false);
	}

	return s;
}
//------------------------------------------------------------------------------------------
std::string CTT_ColorString::Tweak(std::string s) const
{
	size_t pos = s.find(S);
	if (pos != std::string::npos)
	{
		std::string s1 = s.substr(0, pos);
		std::string s2 = s.substr(pos + S.length());
		s2 = Tweak(s2);						// recursion to process multiple occurrences of "S"
		s = s1 + TA->set_fg_color(C) + S;
		s += TA->set_fg_color(HMMPI::Color::VT_DEFAULT) + s2;
	}

	return s;
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
// Parser_1
//------------------------------------------------------------------------------------------
std::string Parser_1::ApplyCTT(std::string s)
{
	for (const auto &i: CTTList)
		s = i->Tweak(s);

	return s;
}
//------------------------------------------------------------------------------------------
Parser_1::Parser_1() : Shift(0), report(""), msg(""), echo(true), silent(false), TotalErrors(0), TotalWarnings(0)		// (OK)
{

}
//------------------------------------------------------------------------------------------
int Parser_1::StrListN(int dec_verb)		// number of lines for HMMPI::StringListing depending on 'verbosity - dec_verb'
{
	const int dv = verbosity - dec_verb;
	if (dv >= 2)
		return -1;				// all lines
	else if (dv >= 1)
		return 20;
	else if (dv >= 0)
		return 7;
	else if (dv >= -1)
		return 2;
	else
		return 0;
}
//------------------------------------------------------------------------------------------
void Parser_1::AddKW_item(KW_item *kwi)		// (OK)
{
	if (KWList.find(kwi->name) != KWList.end())
		throw HMMPI::Exception((std::string)"Duplicate addition of keyword " + kwi->name + " to Parser_1");

	KWList[kwi->name] = kwi;				// put <name, kwi> to the map
	KWList[kwi->name]->SetParser(this);		// give reference of 'this' to kwi
}
//------------------------------------------------------------------------------------------
void Parser_1::AddCTT(ConsTextTweak *ctt)
{
	CTTList.push_back(ctt);
}
//------------------------------------------------------------------------------------------
void Parser_1::DeleteItems()	// (OK)
{
	for (auto &i : KWList)		// & is for speed
	{
		delete i.second;
		i.second = 0;
	}
}
//------------------------------------------------------------------------------------------
void Parser_1::DeleteCTTs()
{
	for (auto &i : CTTList)
	{
		delete i;
		i = 0;
	}
}
//------------------------------------------------------------------------------------------
const KW_item *Parser_1::GetKW_item(std::string s) const	// (OK)
{
	auto it = KWList.find(HMMPI::ToUpper(s));	// 'ToUpper' since the input might be lowercase
	if (it != KWList.end())
		return it->second;
	else
		return 0;
}
//------------------------------------------------------------------------------------------
KW_item *Parser_1::GetKW_item(std::string s)
{
	return const_cast<KW_item*>(const_cast<const Parser_1*>(this)->GetKW_item(s));
}
//------------------------------------------------------------------------------------------
void Parser_1::AppText(std::string s)	// (OK)
{
	if (!silent && echo && MPI_rank == 0)
	{
		s = HMMPI::Replace(s, "/./", "/", NULL);	// print paths in a more clear way
		if (Shift == 0)
		{
			std::cout << ApplyCTT(s) << std::flush;
			report += s;
		}
		else
		{
			char buff[HMMPI::BUFFSIZE];
			sprintf(buff, "[%d] ->\t", Shift);		//	6.10.2013, C++98
			const std::string app = buff;

			std::string saux = app + s;
			bool endNewLine = false;
			if (*--saux.end() == '\n')
			{
				endNewLine = true;
				saux.pop_back();
			}

			saux = HMMPI::Replace(saux, "\n", "\n" + app);
			if (endNewLine)
				saux.push_back('\n');
			saux = HMMPI::Replace(saux, app + "\n", "\n");

			std::cout << ApplyCTT(saux) << std::flush;
			report += saux;
		}
	}
}
//------------------------------------------------------------------------------------------
// parses and executes the control file (main or include)
// use InputLines = DataLines::EliminateEmpty()
// set the increment 'shift_inc' to 0 for the main file, and 1 for the includes
// 'cwd' is the CWD of the file in question
void Parser_1::ReadLines(const std::vector<std::string> &InputLines, int shift_inc, std::string cwd)
{
	const std::chrono::high_resolution_clock::time_point time1 = std::chrono::high_resolution_clock::now();

	const int shift_save = Shift;
	Shift += shift_inc;
	for (size_t posit = 0; posit < InputLines.size(); posit++)
	{
		msg = "";
		KW_item *kw_it = GetKW_item(InputLines[posit]);
		if (kw_it != 0)
		{
			int erows, ecols;
			kw_it->ExpParams(erows, ecols);   			// expected parameters for the keyword

			std::vector<std::string> Spar;				// all lines till the next keyword
			int s_count = 0;
			while (posit+s_count+1 < InputLines.size() && GetKW_item(InputLines[posit+s_count+1]) == 0)	// add all lines which are not-keywords
			{
				Spar.push_back(InputLines[posit+s_count+1]);
				s_count++;
			}

			posit += s_count;   	// update the counter; now 'posit' = "just before the next keyword"
			AppText(HMMPI::stringFormatArr("Кл. слово {0:%s}\n", "Keyword {0:%s}\n", kw_it->name));		// report current keyword name

			kw_it->SetCWD(cwd);
			kw_it->ResetState();				// reset state = no errors
			kw_it->ReadParamTable(Spar);
			kw_it->ProcessParamTable();

			MPI_Barrier(MPI_COMM_WORLD);
			if (kw_it->GetState() == "")		// check if there are no errors so far
				kw_it->Action();

			if (erows != -1)
				for (int j = erows; j < s_count; j++)
					AppText(HMMPI::stringFormatArr("Лишн. стр. {0:%s}\n", "redund. ln. {0:%s}\n", Spar[j]));	// report redundant lines

			AppText(msg + "\n");
		}
		else	// keyword not recognised - report the error
		{
			AppText(HMMPI::stringFormatArr("Кл. слово {0:%s}\n", "Keyword {0:%s}\n", InputLines[posit]));
			AppText((std::string)(HMMPI::MessageRE("ОШИБКА: некорректное кл. слово\n\n", "ERROR: incorrect keyword\n\n")));
			TotalErrors++;
		}
	}

	if (Shift > 0)
	{
		const std::chrono::high_resolution_clock::time_point time2 = std::chrono::high_resolution_clock::now();
		AppText(HMMPI::stringFormatArr("Время: {0:%.3f} сек.\n", "Time: {0:%.3f} sec.\n", std::chrono::duration_cast<std::chrono::duration<double>>(time2-time1).count()));
	}

	Shift = shift_save;
}
//------------------------------------------------------------------------------------------


