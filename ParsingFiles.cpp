/*
 * ParsingFiles.cpp
 *
 *  Created on: Mar 20, 2013
 *      Author: ilya
 */

#include <iostream>
#include <fstream>
#include <cstdio>
#include <cassert>
#include "Abstract.h"
#include "MathUtils.h"
#include "Parsing.h"
#include "Parsing2.h"
#include "Tracking.h"


//------------------------------------------------------------------------------------------
// in this file descendants of KW_fname are implemented
////------------------------------------------------------------------------------------------
KW_include::KW_include()	// (OK)
{
	name = "INCLUDE";
	erows = 1;
}
//------------------------------------------------------------------------------------------
void KW_include::DataIO(int i)	// (OK)
{
	DataLines dl;
	std::string full_fname = HMMPI::getFullPath(CWD, fnames[0]);

	K->AppText("\n");
	dl.LoadFromFile(full_fname);				// read the file
	K->ReadLines(dl.EliminateEmpty(), 1, HMMPI::getCWD(full_fname));		// execute
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
HMMPI::Vector2<double> KW_functionXY::ReadData(std::string fn)
{
	if (fn == "")		// "default"
		return HMMPI::Vector2<double>();
	else
	{
		std::ifstream sr;
		sr.exceptions(std::ios_base::badbit);

		try
		{
			CheckFileOpen(fn);
			sr.open(fn);
			std::vector<double> xs;
			std::vector<double> ys;

			std::string line;
			while (!sr.eof())
			{
				getline(sr, line);
				std::vector<std::string> SA;
				HMMPI::tokenize(line, SA, file_delim, true);
				if (SA.size() > 0)
				{
					if (SA.size() != 2)
						throw HMMPI::Exception("Неверный формат файла: ожидается 2 числа в строке",
											   "Wrong file format: 2 items per line are expected");

					xs.push_back(HMMPI::StoD(SA[0]));
					ys.push_back(HMMPI::StoD(SA[1]));
				}
			}
			sr.close();

			size_t LEN = xs.size();
			HMMPI::Vector2<double> res(LEN, 2);
			for (size_t i = 0; i < LEN; i++)
			{
				res(i, 0) = xs[i];
				res(i, 1) = ys[i];
			}

			return res;
		}
		catch (const std::exception &e)
		{
			if (sr.is_open())
				sr.close();
			throw;
		}
	}
}
//------------------------------------------------------------------------------------------
KW_functionXY::KW_functionXY()
{
	name = "FUNCTIONXY";
	erows = -1;
}
//------------------------------------------------------------------------------------------
void KW_functionXY::Action() noexcept		// similar to KW_fname::Action(), but doesn't check if (fnames[i] == "")
{
	size_t L = fnames.size();
	for (size_t i = 0; i < L; i++)
	{
		try
		{
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
void KW_functionXY::AllocateData() noexcept
{
	data = std::vector<HMMPI::Vector2<double>>(fnames.size());
}
//------------------------------------------------------------------------------------------
void KW_functionXY::DataIO(int i)
{
	std::string fn;
	if (fnames[i] != "")
		fn = HMMPI::getFullPath(this->CWD, fnames[i]);
	else
		fn = "";

	data[i] = ReadData(fn);
	if (fn != "")
		K->AppText(HMMPI::stringFormatArr(HMMPI::MessageRE("Файл {0:%d} -> загружено пар [x, y]: {1:%d}\n", "File {0:%d} -> loaded [x, y] pairs: {1:%d}\n"),
										  std::vector<int>{i+1, (int)data[i].ICount()}));
	else
		K->AppText(HMMPI::stringFormatArr("Файл {0:%d} -> ничего не задано\n",
									  	  "File {0:%d} -> empty item\n", i+1));
}
//---------------------------------------------------------------------------
double KW_functionXY::FuncValBin(int func_ind, double x, int i1, int i2)
{
	HMMPI::Vector2<double> func = data[func_ind];

	if ((x < func(i1, 0))||(x > func(i2, 0)))
		return std::numeric_limits<double>::quiet_NaN();
	else if (i2 == i1)
		return func(i1, 1);
	else if (i2 == i1+1)
	{
		if (func(i1, 0) == func(i2, 0))
			return func(i1, 1);
		else
			return (func(i2, 1) - func(i1, 1))/(func(i2, 0) - func(i1, 0))*(x - func(i1, 0)) + func(i1, 1);
	}
	else
	{
		int a = (i1+i2)/2;
		if ((func(i1, 0) <= x)&&(x <= func(a, 0)))
			return FuncValBin(func_ind, x, i1, a);
		if ((func(a, 0) <= x)&&(x <= func(i2, 0)))
			return FuncValBin(func_ind, x, a, i2);
		if (func(i1, 0) == func(i2, 0))
			return func(i1, 1);
		return std::numeric_limits<double>::quiet_NaN();
	}
}
//------------------------------------------------------------------------------------------
double KW_functionXY::FuncVal(int func_ind, double x)
{
	HMMPI::Vector2<double> func = data[func_ind];
	int len = func.ICount();
	return FuncValBin(func_ind, x, 0, len-1);
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_Pcapill::KW_Pcapill()
{
	name = "PCAPILL";
	erows = 2;
}
//------------------------------------------------------------------------------------------
double KW_Pcapill::FuncValExtrapol(int func_ind, double x)
{
	if (data.size() == 0)
		return 0;

	const HMMPI::Vector2<double> &func = data[func_ind];
	if (func.Length() == 0)
		return 0;

	size_t len = func.ICount();
	if (x <= func(0, 0))
		return func(0, 1);
	else if (x >= func(len-1, 0))
		return func(len-1, 1);
	else
		return FuncValBin(func_ind, x, 0, len-1);
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
void KW_Dtable::DataIO(int i)
{
	std::string fn = HMMPI::getFullPath(this->CWD, fnames[i]);
	data = ReadTableFromFile(fn);

	int M = 0, N = 0;
	M = data.size();
	if (M > 0)
		N = data[0].size();
	K->AppText(HMMPI::stringFormatArr(HMMPI::MessageRE("Файл прочитан -> загружена таблица {0:%d} x {1:%d}\n",
								  	  	  	  	  	   "Reading the file -> loaded {0:%d} x {1:%d} table\n"), std::vector<int>{M, N}));
}
//------------------------------------------------------------------------------------------
KW_Dtable::KW_Dtable()
{
	name = "DTABLE";
}
//------------------------------------------------------------------------------------------
std::vector<std::vector<double>> KW_Dtable::ReadTableFromFile(std::string fn)
{
	std::ifstream sr;
	sr.exceptions(std::ios_base::badbit);
	std::vector<std::vector<double>> res;			// res size is unknown in advance, so the code may work slow for big data
	const std::string file_delim = " \t\r";

	try
	{
		CheckFileOpen(fn);
		sr.open(fn);

		while (!sr.eof())
		{
			std::string line;
			getline(sr, line);
			std::vector<std::string> SA;
			std::vector<double> row;
			HMMPI::tokenize(line, SA, file_delim, true);
			if (SA.size() > 0)
			{
				if (res.size() > 0 && SA.size() != res[0].size())
					throw HMMPI::Exception(fn + HMMPI::stringFormatArr(": найдено {0:%zu} элемент(ов) в строке, что не совпадает с первой строкой",
										   	   	   	   	   	   	  	   ": found {0:%zu} item(s) per line, which is not consistent with the first line", SA.size()));
				row.resize(SA.size());
				for (size_t i = 0; i < row.size(); i++)
					row[i] = HMMPI::StoD(SA[i]);

				res.push_back(row);
			}
		}
		sr.close();
	}
	catch (const std::exception &e)
	{
		if (sr.is_open())
			sr.close();
		throw;
	}

	return res;
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
void KW_conc_data::DataIO(int i)
{
	std::string fn = HMMPI::getFullPath(this->CWD, fnames[i]);
	data = ReadTableFromFile(fn);

	int M = 0, N = 0;
	M = data.size();
	if (M > 0)
		N = data[0].size();
	if (N != 3)						// this check is the only difference with KW_Dtable
		throw HMMPI::Exception("Ожидается таблица с тремя столбцами: T, conc, sigma", "Table with three columns is expected: T, conc, sigma");

	K->AppText(HMMPI::stringFormatArr(HMMPI::MessageRE("Файл прочитан -> загружена таблица {0:%d} x {1:%d}\n",
								  	  	  	  	  	   "Reading the file -> loaded {0:%d} x {1:%d} table\n"), std::vector<int>{M, N}));
}
//------------------------------------------------------------------------------------------
KW_conc_data::KW_conc_data()
{
	name = "CONC_DATA";
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
std::string KW_fsmspec::GetItem()
{
	size_t i0, i1;
	i0 = i1 = std::string::npos;
	std::string res = "";
	try
	{
		int len = aux.length();
		if (len > 0)
		{
			i0 = aux.find(DELIM, 0);
			i1 = aux.find(DELIM, i0+1);

			if ((i0 != std::string::npos)&&(i1 != std::string::npos)&&(i1 > i0))
			{
				res = aux.substr(i0+1, i1-i0-1);
				aux = aux.substr(i1+1, len-i1-1);
			}
		}
	}
	catch (...)
	{
		throw HMMPI::Exception("(eng) KW_fsmspec::GetItem",
							   "Parsing error in KW_fsmspec::GetItem");
	}

	return res;
}
//------------------------------------------------------------------------------------------
std::vector<std::string> KW_fsmspec::ReadChars(std::string fn, std::string HDR)
{
	std::vector<std::string> res;

	std::ifstream sr;
	sr.exceptions(std::ios_base::badbit);

	try
	{
		CheckFileOpen(fn);
		sr.open(fn);
		while (!sr.eof())
		{
			std::string line;
			getline(sr, line);
			size_t i_ = line.find(HDR);
			if (i_ != std::string::npos)
			{
				std::vector<std::string> line_aux;
				line = HMMPI::Trim(line.substr(i_ + HDR.length()), file_delim);
				HMMPI::tokenize(line, line_aux, file_delim, true);
				int count = HMMPI::StoL(line_aux[0]);
				res = std::vector<std::string>(count);

				int c = 0;
				while ((c < count)&&(!sr.eof()))
				{
					getline(sr, aux);
					std::string item = GetItem();
					while (item != "")
					{
						res[c] = HMMPI::Trim(item, file_delim);
						item = GetItem();
						c++;
					}
				}
				break;
			}
		}
		sr.close();
	}
	catch (...)
	{
		if (sr.is_open())
			sr.close();
		throw;
	}

	return res;
}
//------------------------------------------------------------------------------------------
KW_fsmspec::KW_fsmspec()
{
	name = "FSMSPEC";
	DELIM = "'";
	Y = M = D = -1;
	erows = 1;
	not_found = 0;
}
//------------------------------------------------------------------------------------------
void KW_fsmspec::DataIO(int i)
{
	std::string fn = HMMPI::getFullPath(this->CWD, fnames[0]);
	ind = ReadData(fn, Y, M, D, 0);
}
//------------------------------------------------------------------------------------------
std::vector<int> KW_fsmspec::ReadData(std::string fname, int &y, int &m, int &d, std::string *K_msg)
{
	Start_pre();
	IMPORTKWD(vect, KW_eclvectors, "ECLVECTORS");
	DECLKWD(textsmry, KW_textsmry, "TEXTSMRY");
	Finish_pre();

	y = m = d = -1;

	std::string msg;
	size_t Vlen = vect->WGname.size();
	std::vector<int> res = std::vector<int>(Vlen);

	std::vector<std::string> a_kw = ReadChars(fname, "'KEYWORDS'");
	std::vector<std::string> a_wgn = ReadChars(fname, "'WGNAMES '");
	std::vector<std::string> a_units = ReadChars(fname, "'UNITS   '");

	size_t len = a_kw.size();
	if ((len != a_wgn.size())||(len != a_units.size()))
		throw HMMPI::Exception("Неправильные размеры массивов в KW_fsmspec::ReadData",
							   "Inconsistent dimensions in KW_fsmspec::ReadData");

	for (size_t j = 0; j < len; j++)
	{
		if (a_wgn[j] == "" && a_kw[j][0] == 'F')
			a_wgn[j] = "FIELD";				// a hack to handle E300 output
	}

	msg = HMMPI::MessageRE("Найдены вектора:\n", "Found vectors:\n");
	not_found = 0;

	for (size_t i = 0; i < Vlen; i++)
	{
		res[i] = -1;
		for (size_t j = 0; j < len; j++)
		{
			if ((vect->WGname[i] == a_wgn[j])&&(vect->vect[i] == a_kw[j]))
			{
				res[i] = j;
				break;
			}
		}
		if (res[i] != -1)
		{
			std::string indi = HMMPI::stringFormatArr("{0:%d}", std::vector<int>{res[i]});

			std::string vectsigma;
			if (textsmry->data.Length() != 0 && textsmry->ind_sigma[i] != -1)
				vectsigma = a_kw[res[i]] + "S";
			else
				vectsigma = HMMPI::stringFormatArr("{0:%g}", std::vector<double>{vect->sigma[i]});
			msg += a_wgn[res[i]] + "\t" + a_kw[res[i]] +
					HMMPI::stringFormatArr(HMMPI::MessageRE("\tиндекс = {0:%s}, сигма = {1:%s}, единица = [{2:%s}]\n",
											  	  	  	    "\tindex = {0:%s}, sigma = {1:%s}, unit = [{2:%s}]\n"), std::vector<std::string>{indi, vectsigma, a_units[res[i]]});
		}
		else
			not_found++;
	}
	for (size_t j = 0; j < len; j++)
	{
		if (a_kw[j] == "YEAR")
			y = j;
		if (a_kw[j] == "MONTH")
			m = j;
		if (a_kw[j] == "DAY")
			d = j;
	}
	msg += HMMPI::stringFormatArr(HMMPI::MessageRE("(год, месяц, день) = ({0:%d}, {1:%d}, {2:%d})\n",
									 	 	 	   "(year, month, day) = ({0:%d}, {1:%d}, {2:%d})\n"), std::vector<int>{y, m, d});

	if (not_found > 0 && not_found < (int)Vlen)
	{
		char buff[HMMPI::BUFFSIZE];
		sprintf(buff, "WARNING: %s -> not found vectors: %d\n", HMMPI::getFile(fname).c_str(), not_found);
		msg += HMMPI::MessageRE("(eng)", buff);
		K->AppText(HMMPI::MessageRE("(eng)", buff));
		K->TotalWarnings++;
	}

	if (K_msg != 0)
		*K_msg += msg + "\n";

	if (y == -1 || m == -1 || d == -1)
		throw HMMPI::Exception("Индексы YEAR, MONTH, DAY не найдены в *.FSMSPEC", "Index of YEAR, MONTH, DAY not found in *.FSMSPEC");
	if (not_found == (int)Vlen)
		throw HMMPI::Exception("Вектора из ECLVECTORS не найдены в *.FSMSPEC", "No vectors from ECLVECTORS were found in *.FSMSPEC");

	return res;
}
//------------------------------------------------------------------------------------------
std::vector<int> KW_fsmspec::ReadDataH(std::string fname, std::string *K_msg)
{
	Start_pre();
	IMPORTKWD(vect, KW_eclvectors, "ECLVECTORS");
	Finish_pre();

	std::string msg;
	size_t Vlen = vect->WGname.size();
	std::vector<int> res = std::vector<int>(Vlen);

	std::vector<std::string> a_kw = ReadChars(fname, "'KEYWORDS'");
	std::vector<std::string> a_wgn = ReadChars(fname, "'WGNAMES '");
	std::vector<std::string> a_units = ReadChars(fname, "'UNITS   '");

	size_t len = a_kw.size();
	if ((len != a_wgn.size())||(len != a_units.size()))
		throw HMMPI::Exception("(eng) KW_fsmspec::ReadDataH",
								  "Inconsistent dimensions in KW_fsmspec::ReadDataH");

	msg = HMMPI::MessageRE("(eng) (H):\n", "Found vectors (H):\n");
	not_found = 0;

	for (size_t i = 0; i < Vlen; i++)
	{
		res[i] = -1;
		for (size_t j = 0; j < len; j++)
		{
			if ((vect->WGname[i] == a_wgn[j])&&(vect->vect[i] + "H" == a_kw[j]))
			{
				res[i] = j;
				break;
			}
		}
		if (res[i] != -1)
		{
			std::string indi = HMMPI::stringFormatArr("{0:%d}", std::vector<int>{res[i]});
			std::string vectsigma = HMMPI::stringFormatArr("{0:%g}", std::vector<double>{vect->sigma[i]});
			msg += a_wgn[res[i]] + "\t" + a_kw[res[i]] +
					HMMPI::stringFormatArr(HMMPI::MessageRE("\tÐ¸Ð½Ð´ÐµÐºÑ� = {0:%s}, Ñ�Ð¸Ð³Ð¼Ð° = {1:%s}, ÐµÐ´Ð¸Ð½Ð¸Ñ†Ð° = [{2:%s}]\n",
											  "\tindex = {0:%s}, sigma = {1:%s}, unit = [{2:%s}]\n"), std::vector<std::string>{indi, vectsigma, a_units[res[i]]});
		}
		else
			not_found++;
	}

	if (not_found > 0 && not_found < (int)Vlen)
	{
		char buff[HMMPI::BUFFSIZE];
		sprintf(buff, "WARNING: %s -> not found vectors: %d\n", HMMPI::getFile(fname).c_str(), not_found);
		msg += HMMPI::MessageRE("(eng)", buff);
		K->AppText(HMMPI::MessageRE("(eng)", buff));
		K->TotalWarnings++;
	}

	if (K_msg != 0)
		*K_msg += msg + "\n";

	if (not_found == (int)Vlen)
		throw HMMPI::Exception("(eng)", "No vectors from ECLVECTORS (+H) were found in *.FSMSPEC");

	return res;
}
//------------------------------------------------------------------------------------------
void KW_fsmspec::GetKeywordIndRange(std::string fname, std::string kwd, int &start, int &end)
{
	bool searching_first = true;
	std::vector<std::string> a_kw = ReadChars(fname, "'KEYWORDS'");

	start = -1;
	end = a_kw.size();		// points to end of array
	for (size_t i = 0; i < a_kw.size(); i++)
	{
		if (searching_first && a_kw[i] == kwd)
		{
			start = i;
			searching_first = false;
		}
		if (!searching_first && a_kw[i] != kwd)
		{
			end = i;
			break;			// exit after the first contiguous range
		}
	}
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_funrst::KW_funrst()
{
	name = "FUNRST";
	fixedFegrid = true;
	erows = 1;
}
//------------------------------------------------------------------------------------------
void KW_funrst::DataIO(int i)
{
	std::string fn = HMMPI::getFullPath(this->CWD, fnames[0]);
	data = ReadData(fn, "'SWAT    '", fixedFegrid);
	K->AppText(HMMPI::stringFormatArr("(eng): {0:%d}\n", "Reading the file...\nLoaded time steps: {0:%d}\n", (int)data.size()));
}
//------------------------------------------------------------------------------------------
std::vector<std::vector<double>> KW_funrst::ReadData(std::string fname, std::string prop, bool fixed_fegrid)
{
	Start_pre();
	IMPORTKWD(satsteps, KW_satsteps, "SATSTEPS");
	DECLKWD(fegrid, KW_fegrid, "FEGRID");
	Finish_pre();

	std::vector<double> act;
	if (fixed_fegrid)
	{
		act = fegrid->data;
		if (act.size() == 0)
		{
			K->AppText(HMMPI::MessageRE("(eng)\n",
									  	"WARNING: FEGRID not specified when reading FUNRST, active cells may be treated not properly\n"));
			K->TotalWarnings++;
		}
	}
	else
	{
		std::string aux_fname = fname.substr(0, fname.find_last_of(".")) + ".FEGRID";
		act = fegrid->ReadData(aux_fname, "'ACTNUM  '");
	}

	std::vector<int> steps = satsteps->data;
	size_t scount = steps.size();
	std::vector<std::vector<double>> res = std::vector<std::vector<double>>(scount);

	std::ifstream sr;
	sr.exceptions(std::ios_base::badbit);

	try
	{
		std::string STEPHD = "SEQNUM";
		std::string SWHD = prop;

		size_t count;
		size_t step_num = 0, j = 0;
		size_t cur_step = 0;

		CheckFileOpen(fname);
		sr.open(fname);
		while ((!sr.eof())&&(j < scount))
		{
			std::string line;
			getline(sr, line);
			if (line.find(STEPHD) != std::string::npos)
			{
				getline(sr, line);
				line = HMMPI::Trim(line, file_delim);
				cur_step = HMMPI::StoL(line);
				step_num = steps[j];
			}
			size_t i_ = line.find(SWHD);
			if (i_ != std::string::npos)
			{
				if (cur_step == step_num)
				{
					std::vector<std::string> line_aux;
					std::string line2 = HMMPI::Trim(line.substr(i_ + SWHD.length()), file_delim);
					HMMPI::tokenize(line2, line_aux, file_delim, true);
					count = HMMPI::StoL(line_aux[0]);
					if (act.size() == 0)
						res[j] = std::vector<double>(count);
					else
						res[j] = std::vector<double>(act.size());

					for (size_t k = 0; k < res[j].size(); k++)
						res[j][k] = std::numeric_limits<double>::quiet_NaN();

					if ((act.size() != 0)&&(count > act.size()))
						throw HMMPI::Exception(HMMPI::stringFormatArr(HMMPI::MessageRE("(eng) ({0:%d}), (eng) ({1:%d})",
															      "There are more property values ({0:%d}), than active cells in the grid ({1:%d})"), std::vector<int>{(int)count, (int)act.size()}));
					size_t i = 0, c = 0;
					while ((c < count)&&(!sr.eof()))
					{
						getline(sr, line);
						std::vector<std::string> line_aux;
						HMMPI::tokenize(line, line_aux, file_delim, true);

						for (size_t k = 0; k < line_aux.size(); k++)
						{
							if (act.size() != 0)
							{
								while ((i < act.size())&&(act[i] == 0))
								{
									res[j][i] = std::numeric_limits<double>::quiet_NaN();
									i++;
								}

								if (i >= act.size())
									throw HMMPI::Exception("(eng) KW_funrst::ReadData",
														   "Index error (i >= act.size()) in KW_funrst::ReadData");
							}
							else
								if (i >= count)
									throw HMMPI::Exception("(eng) KW_funrst::ReadData",
														   "Index error in KW_funrst::ReadData");

							res[j][i] = HMMPI::StoD(line_aux[k]);
							i++;
							c++;
						}
					}
					j++;
				}
			}
		}
		sr.close();

		if (res.size() == 0)
			throw HMMPI::Exception("(eng)", "No time steps were read from *.FUNRST file, SATSTEPS is likely to be empty");
		else if (res[0].size() == 0)
			throw HMMPI::Exception(HMMPI::stringFormatArr(HMMPI::MessageRE("(eng){0:%s}{1:%s}", "No data were read from *.FUNRST file; "
								   "the file {0:%s} doesn't have the required property {1:%s} or the steps specified in SATSTEPS"),
								   std::vector<std::string>{HMMPI::getFile(fname), HMMPI::Trim(prop, " '\t\r")}));
		else
			for (size_t j = 1; j < res.size(); j++)
			{
				if (res[0].size() != res[j].size())
					throw HMMPI::Exception(HMMPI::stringFormatArr("(eng){0:%d}",
									(std::string)"Variable number of values for different time steps while reading *.FUNRST (problematic step: {0:%d}); probably the file " +
									 HMMPI::getFile(fname) + " doesn't have some of the steps specified in SATSTEPS", steps[j]));
			}
	}
	catch (...)
	{
		if (sr.is_open())
			sr.close();
		throw;
	}

	return res;
}
//------------------------------------------------------------------------------------------
HMMPI::Vector2<KW_funrst::grad> KW_funrst::ReadGrads(std::string mod_root)
{
	Start_pre();
	IMPORTKWD(datesW, KW_dates, "DATES");
	IMPORTKWD(vecs, KW_eclvectors, "ECLVECTORS");
	DECLKWD(funsmry, KW_funsmry, "FUNSMRY");
	DECLKWD(fegrid, KW_fegrid, "FEGRID");
	Finish_pre();

	std::vector<double> act = fegrid->ReadData(mod_root + ".FEGRID", "'ACTNUM  '");
	size_t Nsteps = datesW->D.size();
	size_t Nvecs = vecs->vecs.size();
	assert(act.size() > 0);
	assert(Nsteps == funsmry->taken_files.size());

	HMMPI::Vector2<KW_funrst::grad> res(Nsteps, Nvecs);

	std::ifstream sr;
	sr.exceptions(std::ios_base::badbit);

	try
	{
		for (size_t step = 0; step < Nsteps; step++)
			if (funsmry->taken_files[step] != -1)
			{
				char fname[HMMPI::BUFFSIZE];
				sprintf(fname, (mod_root + ".F%.4d").c_str(), funsmry->taken_files[step]);

				CheckFileOpen(fname);
				sr.open(fname);

				std::string line, a, c;
				getline(sr, line);
				while (!sr.eof() && line.find("AJGNAMES") == std::string::npos)		// find the main header
					getline(sr, line);

				if (sr.eof())
					continue;											// for(step)

				int count, b = 0;
				HMMPI::ParseEclSmallHdr(line, a, count, c);

				std::vector<std::string> items_all;
				while (!sr.eof() && b < count)							// read entries below the main header
				{
					getline(sr, line);
					std::vector<std::string> items = HMMPI::ParseEclChar(line);
					HMMPI::VecAppend(items_all, items);
					b += items.size();
				}

				if (items_all.size() % 4 != 0)
					throw HMMPI::Exception("items_all.size() % 4 != 0 in KW_funrst::ReadGrads");

				std::map<std::string, std::pair<int, std::string>> grad_specs;		// e.g. <"AJGFN 15", <N, "PORO">>, where N is index in ECLVECTORS of the eclipse vector
				for (size_t i = 0; i < items_all.size(); i += 4)
				{
					std::string well = HMMPI::Trim(items_all[i+1], " \t\r\n");
					std::string rate = HMMPI::Trim(items_all[i+2], " \t\r\n");
					std::string prop = HMMPI::Trim(items_all[i+3], " \t\r\n");
					std::pair<std::string, std::string> v{well, rate};

					int N = std::find(vecs->vecs.begin(), vecs->vecs.end(), v) - vecs->vecs.begin();
					grad_specs[items_all[i]] = std::pair<int, std::string>(N, prop);			// take all vectors
				}

				// now proceed to reading the grids till the end of current file
				while (!sr.eof())
				{
					getline(sr, line);
					while (!sr.eof() && line.find("AJGFN") == std::string::npos)	// find the grid header
						getline(sr, line);

					if (sr.eof())
						break;											// -> for(step)

					HMMPI::ParseEclSmallHdr(line, a, count, c);
					if (count > (int)act.size())
						throw HMMPI::Exception(HMMPI::stringFormatArr("There are more property values ({0:%d}), than active cells in the grid ({1:%d})", std::vector<int>{count, (int)act.size()}));

					int ind_eclvec = grad_specs.at(a).first;
					std::string prop = grad_specs.at(a).second;
					std::vector<double> grid(act.size(), std::numeric_limits<double>::quiet_NaN());		// will be ultimately saved to "res"

					size_t i = 0, n = 0;								// i - index in "grid", n - index for active cells
					while ((int)n < count && !sr.eof())
					{
						getline(sr, line);
						std::vector<std::string> line_aux;
						HMMPI::tokenize(line, line_aux, " \t\r\n", true);

						for (size_t k = 0; k < line_aux.size(); k++)
						{
							while (i < act.size() && act[i] == 0)
								i++;

							if (i >= act.size())
								throw HMMPI::Exception("Index error (i >= act.size()) in KW_funrst::ReadGrads");
							if ((int)n >= count)
								throw HMMPI::Exception("Index error (n >= count) in KW_funrst::ReadGrads");

							grid[i] = HMMPI::StoD(line_aux[k]);
							i++;
							n++;
						}
					}

					if (ind_eclvec != (int)vecs->vecs.size())
						res(step, ind_eclvec)[prop] = std::move(grid);		// save the grid -- only for vectors found in ECLVECTORS
				}

				sr.close();
			}
	}
	catch (...)
	{
		if (sr.is_open())
			sr.close();
		throw;
	}

	return res;
}
//------------------------------------------------------------------------------------------
HMMPI::Vector2<double> KW_funrst::GradsOfProperty(const HMMPI::Vector2<grad> &grad, std::string prop, int cell)
{
	HMMPI::Vector2<double> res(grad.ICount(), grad.JCount());
	for (size_t i = 0; i < res.ICount(); i++)
		for (size_t j = 0; j < res.JCount(); j++)
		{
			if (grad(i, j).count(prop) > 0)
			{
				//std::cout << "DEBUG property found\n";	// DEBUG
				if (cell < 0 || cell >= (int)grad(i, j).at(prop).size())
					throw HMMPI::Exception("cell is out of range in KW_funrst::GradsOfProperty");
				res(i, j) = grad(i, j).at(prop)[cell];
			}
			else
				res(i, j) = 0;
		}

	return res;
}
//------------------------------------------------------------------------------------------
HMMPI::Vector2<double> KW_funrst::GradsOfRegion(const HMMPI::Vector2<std::vector<double>> &grad, int reg)
{
	HMMPI::Vector2<double> res(grad.ICount(), grad.JCount());
	for (size_t i = 0; i < res.ICount(); i++)
		for (size_t j = 0; j < res.JCount(); j++)
			res(i, j) = grad(i, j)[reg];

	return res;
}
//------------------------------------------------------------------------------------------
std::vector<Grid2D> KW_funrst::ReadDataGrid2D(std::string fname, std::string prop, int cX, int cY, std::string fn_init)
{
	Start_pre();
	IMPORTKWD(DIMS, KW_griddims, "GRIDDIMS");
	DECLKWD(seiswght, KW_mapseiswght, "MAPSEISWGHT"); 	// optional
	Finish_pre();

	int Nz = DIMS->Nz;
	std::string PR = DIMS->wght;
	if (PR.size() < 8)
		PR = "'" + PR + std::string(8 - PR.length(), ' ') + "'";
	else
		PR = "'" + PR + "'";

	std::vector<double> WGHT = ReadDataInit(fn_init, PR);
	if (WGHT.size() == 0)
		throw HMMPI::Exception("(eng) *.FINIT",
							   "Cannot read property for weighting (GRIDDIMS->wght) in *.FINIT");

	std::vector<std::vector<double>> grids = ReadData(fname, prop, false);	// ACTNUM from current model
	size_t steps = grids.size();
	size_t tot_len = grids[0].size();
	if (((int)tot_len != cX*cY*Nz)||(tot_len != WGHT.size()))
		throw HMMPI::Exception("(eng) KW_funrst::ReagDataGrid2D",
							   "Incorrect dimensions in KW_funrst::ReagDataGrid2D");
	if (seiswght->data.size() != 0 && tot_len != seiswght->data.size())
		throw HMMPI::Exception("(eng) KW_funrst::ReagDataGrid2D",
							   "Incorrect dimensions in KW_funrst::ReagDataGrid2D");

	std::vector<Grid2D> res(steps);
	for (size_t t = 0; t < steps; t++)
	{
		assert(grids[t].size() != 0);

		Grid2D aux;
		aux.InitData(cX, cY);
		aux.SetGeom(-0.5, -0.5, 1, 1);
		aux.SetVal(0);

		for (int i0 = 0; i0 < cX; i0++)
		{
			for (int j0 = 0; j0 < cY; j0++)
			{
				double d = 0;
				double w = 0;
				for (int k0 = 0; k0 < Nz; k0++)
				{
					double seisw = 1;
					if (seiswght->data.size() != 0)
						seisw = seiswght->data[i0 + (cY - j0 - 1)*cX + k0*cX*cY];

					double add = grids[t][i0 + (cY - j0 - 1)*cX + k0*cX*cY] * WGHT[i0 + (cY - j0 - 1)*cX + k0*cX*cY] * seisw;
					if (!HMMPI::IsNaN(add))
					{
						d += add;
						w += WGHT[i0 + (cY - j0 - 1)*cX + k0*cX*cY] * seisw;
					}
				}
				d /= w;
				if (w == 0)
					d = std::numeric_limits<double>::quiet_NaN();
				if (!HMMPI::IsNaN(d))
				{
					aux.data[i0][j0] = d;
					aux.flag[i0][j0] = 1;
				}
				else
				{
					aux.data[i0][j0] = 0;
					aux.flag[i0][j0] = 0;
				}
			}
		}
		res[t] = aux;
	}

	return res;
}
//------------------------------------------------------------------------------------------
std::vector<Grid2D> KW_funrst::GetGrid2D(int cX, int cY, std::string fn_init)
{
	Start_pre();
	IMPORTKWD(DIMS, KW_griddims, "GRIDDIMS");
	Add_pre("SATSTEPS");
	DECLKWD(seiswght, KW_mapseiswght, "MAPSEISWGHT");		// optional
	Finish_pre();

	int Nz = DIMS->Nz;
	std::string PR = DIMS->wght;
	if (PR.length() < 8)
		PR = "'" + PR + std::string(8 - PR.length(), ' ') + "'";
	else
		PR = "'" + PR + "'";

	std::vector<double> WGHT = ReadDataInit(fn_init, PR);
	if (WGHT.size() == 0)
		throw HMMPI::Exception("(eng) (GRIDDIMS->wght) Ð² *.FINIT",
							   "Cannot read property for weighting (GRIDDIMS->wght) in *.FINIT");
	std::vector<std::vector<double>> grids = data;

	size_t steps = grids.size();
	size_t tot_len = grids[0].size();
	if (((int)tot_len != cX*cY*Nz)||(tot_len != WGHT.size()))
		throw HMMPI::Exception("(eng) KW_funrst::GetGrid2D",
							   "Incorrect dimensions in KW_funrst::GetGrid2D");
	if (seiswght->data.size() != 0 && tot_len != seiswght->data.size())
		throw HMMPI::Exception("(eng) KW_funrst::GetGrid2D",
							   "Incorrect dimensions in KW_funrst::GetGrid2D");

	std::vector<Grid2D> res(steps);
	for (size_t t = 0; t < steps; t++)
	{
		assert(grids[t].size() != 0);

		Grid2D aux;
		aux.InitData(cX, cY);
		aux.SetGeom(-0.5, -0.5, 1, 1);
		aux.SetVal(0);

		for (int i0 = 0; i0 < cX; i0++)
		{
			for (int j0 = 0; j0 < cY; j0++)
			{
				double d = 0;
				double w = 0;
				for (int k0 = 0; k0 < Nz; k0++)
				{
					double seisw = 1;
					if (seiswght->data.size() != 0)
						seisw = seiswght->data[i0 + (cY - j0 - 1)*cX + k0*cX*cY];

					double add = grids[t][i0 + (cY - j0 - 1)*cX + k0*cX*cY] * WGHT[i0 + (cY - j0 - 1)*cX + k0*cX*cY] * seisw;
					if (!HMMPI::IsNaN(add))
					{
						d += add;
						w += WGHT[i0 + (cY - j0 - 1)*cX + k0*cX*cY] * seisw;
					}
				}
				d /= w;
				if (w == 0)
					d = std::numeric_limits<double>::quiet_NaN();
				if (!HMMPI::IsNaN(d))
				{
					aux.data[i0][j0] = d;
					aux.flag[i0][j0] = 1;
				}
				else
				{
					aux.data[i0][j0] = 0;
					aux.flag[i0][j0] = 0;
				}
			}
		}
		res[t] = aux;
	}

	return res;
}
//------------------------------------------------------------------------------------------
std::vector<double> KW_funrst::ReadDataInit(std::string fname, std::string prop)
{
	Start_pre();
	DECLKWD(fegrid, KW_fegrid, "FEGRID");
	Finish_pre();

	std::vector<double> act;
	// act = fegrid->data; 		old version - fixed ACTNUM
	std::string aux_fname = fname.substr(0, fname.find_last_of(".")) + ".FEGRID";

	if (prop == "'PORV    '")
		act = std::vector<double>();
	else
		act = fegrid->ReadData(aux_fname, "'ACTNUM  '");	// ACTNUM from current model

	std::ifstream sr;
	sr.exceptions(std::ios_base::badbit);

	std::string SWHD = prop;
	std::vector<double> res;

	try
	{
		CheckFileOpen(fname);
		sr.open(fname);
		while (!sr.eof())
		{
			std::string line;
			getline(sr, line);
			size_t i_ = line.find(SWHD);
			if (i_ != std::string::npos)
			{
				std::vector<std::string> line_aux;
				std::string Line2 = HMMPI::Trim(line.substr(i_ + SWHD.length()), file_delim);
				HMMPI::tokenize(Line2, line_aux, file_delim, true);

				size_t count = HMMPI::StoL(line_aux[0]);
				if (act.size() == 0)
					res = std::vector<double>(count);
				else
					res = std::vector<double>(act.size());

				for (size_t k = 0; k < res.size(); k++)
					res[k] = std::numeric_limits<double>::quiet_NaN();

				if ((act.size() != 0)&&(count > act.size()))
					throw HMMPI::Exception(HMMPI::stringFormatArr(HMMPI::MessageRE("(eng) ({0:%d})({1:%d})",
															  "There are more property values ({0:%d}), than active cells in the grid ({1:%d})"), std::vector<int>{(int)count, (int)act.size()}));

				size_t i = 0;
				size_t c = 0;
				while ((c < count)&&(!sr.eof()))
				{
					getline(sr, line);
					std::vector<std::string> line_aux;
					HMMPI::tokenize(line, line_aux, file_delim, true);

					for (size_t k = 0; k < line_aux.size(); k++)
					{
						if (act.size() != 0)
						{
							while ((i < act.size())&&(act[i] == 0))
							{
								res[i] = std::numeric_limits<double>::quiet_NaN();
								i++;
							}

							if (i >= act.size())
								throw HMMPI::Exception("(eng) (i >= act.size()) Ð² KW_funrst::ReadDataInit",
													   "Index error (i >= act.size()) in KW_funrst::ReadDataInit");
						}
						else
							if (i >= count)
								throw HMMPI::Exception("(eng) KW_funrst::ReadDataInit",
													   "Index error in KW_funrst::ReadDataInit");

						res[i] = HMMPI::StoD(line_aux[k]);
						i++;
						c++;
					}
				}
			}
		}
		sr.close();

		if (res.size() == 0)
			throw HMMPI::Exception(HMMPI::stringFormatArr(HMMPI::MessageRE("(eng){0:%s}{1:%s}",
					"Could not read property {0:%s} from file {1:%s}"), std::vector<std::string>{HMMPI::Trim(prop, " '\t\r"), HMMPI::getFile(fname)}));
	}
	catch(...)
	{
		if (sr.is_open())
			sr.close();
		throw;
	}

	return res;
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_funrstG::KW_funrstG()
{
	name = "FUNRSTG";
}
//------------------------------------------------------------------------------------------
void KW_funrstG::DataIO(int i)
{
	std::string fn = HMMPI::getFullPath(this->CWD, fnames[0]);
	data = ReadData(fn, "'SGAS    '", fixedFegrid);
	K->AppText(HMMPI::stringFormatArr("(eng): {0:%d}\n", "Reading the file...\nLoaded time steps: {0:%d}\n", (int)data.size()));

}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_funrstA::KW_funrstA()
{
	name = "FUNRSTA";
}
//------------------------------------------------------------------------------------------
void KW_funrstA::DataIO(int i)
{
	std::string fn = HMMPI::getFullPath(this->CWD, fnames[0]);
	data = ReadData(fn, "'ATTR    '", fixedFegrid);
	K->AppText(HMMPI::stringFormatArr("(eng): {0:%d}\n", "Reading the file...\nLoaded time steps: {0:%d}\n", (int)data.size()));
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_fegrid::KW_fegrid()
{
	name = "FEGRID";
	erows = 1;
}
//------------------------------------------------------------------------------------------
void KW_fegrid::DataIO(int i)
{
	std::string fn = HMMPI::getFullPath(this->CWD, fnames[0]);
	data = ReadData(fn, "'ACTNUM  '");
	K->AppText(HMMPI::stringFormatArr("(eng) {0:%d}\n",
									 "Reading the file...\nLoaded cells: {0:%d}\n", (int)data.size()));
}
//------------------------------------------------------------------------------------------
std::vector<double> KW_fegrid::ReadData(std::string fname, std::string prop)
{
	std::ifstream sr;
	sr.exceptions(std::ios_base::badbit);

	std::string HDR = prop;
	std::vector<double> res;

	try
	{
		CheckFileOpen(fname);
		sr.open(fname);
		while (!sr.eof())
		{
			std::string line;
			getline(sr, line);
			size_t i_ = line.find(HDR);
			if (i_ != std::string::npos)
			{
				std::vector<std::string> line_aux;
				std::string Line2 = HMMPI::Trim(line.substr(i_ + HDR.length()), file_delim);
				HMMPI::tokenize(Line2, line_aux, file_delim, true);

				int count = HMMPI::StoL(line_aux[0]);
				res = std::vector<double>(count);

				int c = 0;
				while ((c < count)&&(!sr.eof()))
				{
					getline(sr, line);
					std::vector<std::string> line_aux;
					HMMPI::tokenize(line, line_aux, file_delim, true);
					for (size_t k = 0; k < line_aux.size(); k++)
					{
						res[c] = HMMPI::StoD(line_aux[k]);
						c++;
					}
				}
				break;
			}
		}
		sr.close();

		if (res.size() == 0)
			throw HMMPI::Exception(HMMPI::stringFormatArr(HMMPI::MessageRE("(eng){0:%s}{1:%s}",
					"Could not read property {0:%s} from file {1:%s}"), std::vector<std::string>{HMMPI::Trim(prop, " '\t\r"), HMMPI::getFile(fname)}));
	}
	catch(...)
	{
		if (sr.is_open())
			sr.close();
		throw;
	}

	return res;
}
//------------------------------------------------------------------------------------------
double KW_fegrid::Sum()
{
	double res = 0;
	for (size_t i = 0; i < data.size(); i++)
		res += data[i];

	return res;
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
void KW_funsmry::DataIO(int i)
{
	Start_pre();
	Add_pre("FSMSPEC");		// check for FSMSPEC here, since this is not done in ReadData()
	Finish_pre();

	std::string fn = HMMPI::getFullPath(this->CWD, fnames[0]);
	data = ReadData(fn);
	if (count_dates == 0)
		count_vecs = 0;
	else
		count_vecs /= count_dates;
	K->AppText(HMMPI::stringFormatArr(HMMPI::MessageRE("(eng) {0:%d}, {1:%d}\n",
										"Reading the file...\nLoaded time steps: {0:%d}, loaded vectors: {1:%d}\n"), std::vector<int>{count_dates, count_vecs}));
}
//------------------------------------------------------------------------------------------
KW_funsmry::KW_funsmry()
{
	name = "FUNSMRY";
	erows = 1;
	count_dates = count_vecs = 0;
}
//------------------------------------------------------------------------------------------
int KW_funsmry::DateCmp(int Y1, int M1, int D1, int Y2, int M2, int D2)
{
	if (Y1 < Y2)
		return -1;
	else if (Y1 > Y2)
		return 1;
	else if (M1 < M2)
		return -1;
	else if (M1 > M2)
		return 1;
	else if (D1 < D2)
		return -1;
	else if (D1 > D2)
		return 1;
	else
		return 0;
}
//------------------------------------------------------------------------------------------
HMMPI::Vector2<double> KW_funsmry::ReadData(std::string fname, bool read_hist)
{
	Start_pre();
	IMPORTKWD(datesW, KW_dates, "DATES");
	DECLKWD(spec, KW_fsmspec, "FSMSPEC");
	IMPORTKWD(vect, KW_eclvectors, "ECLVECTORS");
	IMPORTKWD(undef, KW_undef, "UNDEF");
	Finish_pre();

	std::ifstream sr;
	sr.exceptions(std::ios_base::badbit);

	HMMPI::Vector2<double> res;

	try
	{
		CheckFileOpen(fname);
		std::string PARHD = "'PARAMS  '";

		if (read_hist && spec->indH.size() == 0)
			throw HMMPI::Exception("indH not defined in KW_funsmry::ReadData");

		std::vector<int> *IND;
		if (read_hist)
			IND = &(spec->indH);
		else
			IND = &(spec->ind);

		size_t dcount = datesW->D.size();
		size_t pcount = IND->size();

		count_dates = 0;
		count_vecs = 0;

		size_t j = 0;
		res = HMMPI::Vector2<double>(dcount, pcount);

		sr.open(fname);
		while ((!sr.eof())&&(j < dcount))
		{
			std::string line;
			getline(sr, line);
			size_t i_ = line.find(PARHD);
			if (i_ != std::string::npos)
			{
				std::vector<std::string> line_aux;
				std::string Line2 = HMMPI::Trim(line.substr(i_ + PARHD.length()), file_delim);
				HMMPI::tokenize(Line2, line_aux, file_delim, true);
				int count = HMMPI::StoL(line_aux[0]);

				std::vector<double> aux(count);
				int i = 0;
				while ((i < count)&&(!sr.eof()))
				{
					getline(sr, line);
					std::vector<std::string> line_aux;
					HMMPI::tokenize(line, line_aux, file_delim, true);

					for (size_t k = 0; k < line_aux.size(); k++)
					{
						aux[i] = HMMPI::StoD(line_aux[k]);
						i++;
					}
				}

				int D, M, Y;
				D = int(aux[spec->D]);
				M = int(aux[spec->M]);
				Y = int(aux[spec->Y]);

				while (DateCmp(Y, M, D, datesW->Y[j], datesW->M[j], datesW->D[j]) == 1)
					j++;

				if ((D == datesW->D[j])&&(M == datesW->M[j])&&(Y == datesW->Y[j]))
				{
					count_dates++;
					for (size_t k = 0; k < pcount; k++)
					{
						assert(k < IND->size());
						if (((*IND)[k] >= 0)&&((*IND)[k] < count))
						{
							count_vecs++;
							assert(j < res.ICount() && k < res.JCount());
							assert((*IND)[k] >= 0 && (*IND)[k] < (int)aux.size());
							res(j, k) = aux[(*IND)[k]];
							if (k < vect->vect.size() && vect->vect[k] == "WBHP")
							{
								if (res(j, k) == undef->Uvectbhp)
									res(j, k) = std::numeric_limits<double>::quiet_NaN();
							}
							else
							{
								if (res(j, k) == undef->Uvect)
									res(j, k) = std::numeric_limits<double>::quiet_NaN();
							}
						}
					}
					j++;
				}
			}
		}
		sr.close();
	}
	catch (...)
	{
		if (sr.is_open())
			sr.close();
		throw;
	}

	return res;
}
//------------------------------------------------------------------------------------------
HMMPI::Vector2<double> KW_funsmry::ReadData(std::string mod_root, int i0, int i1)
{
	Start_pre();
	IMPORTKWD(datesW, KW_dates, "DATES");
	DECLKWD(spec, KW_fsmspec, "FSMSPEC");
	Finish_pre();

	std::ifstream sr;
	sr.exceptions(std::ios_base::badbit);

	HMMPI::Vector2<double> res;

	try
	{
		std::string PARHD = "'PARAMS  '";
		std::vector<int> *IND = &(spec->ind);

		size_t dcount = datesW->D.size();
		size_t pcount = IND->size();

		count_dates = 0;
		count_vecs = 0;
		taken_files = std::vector<int>(datesW->D.size(), -1);

		size_t j = 0;
		res = HMMPI::Vector2<double>(dcount, pcount);

		for (int File = i0; File < i1; File++)
		{
			char fname[HMMPI::BUFFSIZE];
			sprintf(fname,(mod_root + ".A%.4d").c_str(), File);

			CheckFileOpen(fname);
			sr.open(fname);
			while ((!sr.eof())&&(j < dcount))
			{
				std::string line;
				getline(sr, line);
				size_t i_ = line.find(PARHD);
				if (i_ != std::string::npos)
				{
					std::vector<std::string> line_aux;
					std::string Line2 = HMMPI::Trim(line.substr(i_ + PARHD.length()), file_delim);
					HMMPI::tokenize(Line2, line_aux, file_delim, true);
					int count = HMMPI::StoL(line_aux[0]);

					std::vector<double> aux(count);
					int i = 0;
					while ((i < count)&&(!sr.eof()))
					{
						getline(sr, line);
						std::vector<std::string> line_aux;
						HMMPI::tokenize(line, line_aux, file_delim, true);

						for (size_t k = 0; k < line_aux.size(); k++)
						{
							aux[i] = HMMPI::StoD(line_aux[k]);
							i++;
						}
					}

					int D, M, Y;
					D = int(aux[spec->D]);
					M = int(aux[spec->M]);
					Y = int(aux[spec->Y]);

					while (DateCmp(Y, M, D, datesW->Y[j], datesW->M[j], datesW->D[j]) == 1)
						j++;

					if ((D == datesW->D[j])&&(M == datesW->M[j])&&(Y == datesW->Y[j]))
					{
						count_dates++;
						taken_files[j] = File;
						for (size_t k = 0; k < pcount; k++)
						{
							assert(k < IND->size());
							if (((*IND)[k] >= 0)&&((*IND)[k] < count))
							{
								count_vecs++;
								assert(j < res.ICount() && k < res.JCount());
								assert((*IND)[k] >= 0 && (*IND)[k] < (int)aux.size());
								res(j, k) = aux[(*IND)[k]];
							}
						}
						j++;
					}
				}
			}	// while
			sr.close();
		}	// for (File)
	}
	catch (...)
	{
		if (sr.is_open())
			sr.close();
		throw;
	}

	return res;
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
void KW_textsmry::CheckHdrRepeats() const		// checks for repeating columns in the header; throws an error if needed
{
	std::vector<std::pair<std::string, std::string>> work(Hdr.JCount());
	for (size_t i = 0; i < work.size(); i++)
		work[i] = std::make_pair(Hdr(0, i), Hdr(1, i));

	std::pair<std::string, std::string> dup;
	if (HMMPI::FindDuplicate(work, dup))
		throw HMMPI::Exception((std::string)HMMPI::MessageRE(
				"В TEXTSMRY найдены столбцы с одинаковым заголовком: ",
				"In TEXTSMRY there are columns with identical header: ") + dup.first + " " + dup.second);
}
//------------------------------------------------------------------------------------------
void KW_textsmry::ReadInd(std::string *K_msg)		// reads from "Hdr", fills "ind", "ind_sigma", updates "not_found"
{
	Start_pre();
	IMPORTKWD(vect, KW_eclvectors, "ECLVECTORS");
	Finish_pre();

	size_t Vlen = vect->WGname.size();	// total vectors
	ind = std::vector<int>(Vlen);
	ind_sigma = std::vector<int>(Vlen);
	size_t len = Hdr.JCount();			// total columns in text summary

	std::string auxmsg = HMMPI::MessageRE("Найдены вектора (TEXT):\n", "Found vectors (TEXT):\n");
	not_found = 0;
	HMMPI::StringListing stl_rus("\t"), stl_eng("\t"), stl_missing("\t");

	for (size_t i = 0; i < Vlen; i++)	// loop through all ECLVECTORS
	{
		ind[i] = -1;
		ind_sigma[i] = -1;
		int CNT = 2;
		for (size_t j = 0; j < len; j++)		// loop through all columns
		{
			if (vect->WGname[i] == Hdr(0, j) && vect->vect[i] + "H" == Hdr(1, j))
			{
				ind[i] = j;
				CNT--;
			}
			if (vect->WGname[i] == Hdr(0, j) && vect->vect[i] + "S" == Hdr(1, j))
			{
				ind_sigma[i] = j;
				CNT--;
			}
			if (CNT == 0)
				break;
		}
		if (ind[i] != -1)
		{
			std::string indi = HMMPI::stringFormatArr("{0:%d}", std::vector<int>{ind[i]+1});
			std::string vectsigma = vect->vect[i] + "S";
			if (ind_sigma[i] == -1)
				vectsigma = HMMPI::stringFormatArr("{0:%g}", std::vector<double>{vect->sigma[i]});

			stl_rus.AddLine(std::vector<std::string>{Hdr(0, ind[i]), Hdr(1, ind[i]), (std::string)"индекс = " + indi, (std::string)"сигма = " + vectsigma});
			stl_eng.AddLine(std::vector<std::string>{Hdr(0, ind[i]), Hdr(1, ind[i]), (std::string)"index = " + indi, (std::string)"sigma = " + vectsigma});
		}
		else
		{
			stl_missing.AddLine(std::vector<std::string>{(std::string)"* " + vect->WGname[i], vect->vect[i] + "H"});
			not_found++;
		}
	}
	const int N = K->StrListN();
	auxmsg += HMMPI::MessageRE(stl_rus.Print(N, N), stl_eng.Print(N, N));

	if (not_found > 0 && not_found < (int)Vlen)
	{
		auxmsg += HMMPI::stringFormatArr("ПРЕДУПРЕЖДЕНИЕ: В TEXTSMRY не найден(о) {0:%d} вектор(ов):\n",
										 "WARNING: In TEXTSMRY {0:%d} vector(s) were not found:\n", not_found) + stl_missing.Print(N, N);
		warnings++;
	}

	if (not_found == (int)Vlen)
		throw HMMPI::Exception("Вектора из ECLVECTORS не найдены", "No vectors from ECLVECTORS were found");

	if (K_msg != 0)
		*K_msg += auxmsg;
}
//------------------------------------------------------------------------------------------
void KW_textsmry::DataIO(int i)
{
	warnings = 0;
	std::string fn = HMMPI::getFullPath(this->CWD, fnames[0]);
	data = ReadData(fn);
	pet_dat = data;
	K->AppText(HMMPI::stringFormatArr(HMMPI::MessageRE("Чтение файла...\nЗагружено временных шагов: {2:%d}/{0:%d}, найдено векторов: {1:%d}\n",
													   "Reading the file...\nLoaded time steps: {2:%d}/{0:%d}, number of found vectors: {1:%d}\n"),
										std::vector<int>{(int)data.ICount(), (int)data.JCount()/2 - not_found, found_ts}));
	K->AppText(msg);
	K->TotalWarnings += warnings;
	warnings = 0;

	if (found_ts < (int)data.ICount())
	{
		K->AppText(HMMPI::MessageRE("ПРЕДУПРЕЖДЕНИЕ: Некоторые временные шаги из DATES не были найдены в TEXTSMRY:\n", "WARNING: Some time steps from DATES were not found in TEXTSMRY:\n"));
		K->AppText(missing_dates(K->StrListN()));
		K->TotalWarnings++;
	}
}
//------------------------------------------------------------------------------------------
std::string KW_textsmry::missing_dates(int N)	// list of missing dates, based on 'found_ts_flag'; N - number of lines for StringListing
{
	Start_pre();
	IMPORTKWD(datesW, KW_dates, "DATES");
	Finish_pre();

	HMMPI::StringListing stlist("\t");
	assert(found_ts_flag.size() == datesW->dates.size());

	for (size_t i = 0; i < found_ts_flag.size(); i++)
		if (!found_ts_flag[i])
			stlist.AddLine(std::vector<std::string>{datesW->dates[i].ToString()});

	return stlist.Print(N, N);
}
//------------------------------------------------------------------------------------------
KW_textsmry::KW_textsmry()
{
	name = "TEXTSMRY";
	not_found = 0;
	found_ts = 0;
	erows = 1;
	warnings = 0;
}
//------------------------------------------------------------------------------------------
HMMPI::Vector2<double> KW_textsmry::ReadData(std::string fname)
{
	Start_pre();
	IMPORTKWD(datesW, KW_dates, "DATES");
	IMPORTKWD(vect, KW_eclvectors, "ECLVECTORS");
	Finish_pre();

	std::ifstream sr;
	sr.exceptions(std::ios_base::badbit);

	HMMPI::Vector2<double> res;
	try
	{
		CheckFileOpen(fname);

		const size_t dcount = datesW->D.size();
		found_ts_flag = std::vector<bool>(dcount, false);
		size_t j = 0;

		sr.open(fname);

		// read header
		size_t vcount = 0;
		while (!sr.eof() && j < 2)
		{
			std::string line;
			std::vector<std::string> line_aux;

			getline(sr, line);
			size_t ind0 = line.find("--");
			if (ind0 != std::string::npos)
				line = line.substr(0, ind0);

			HMMPI::tokenize(line, line_aux, file_delim, true);
			if (line_aux.size() != 0)
			{
				if (j == 0)
				{
					vcount = line_aux.size();
					Hdr = HMMPI::Vector2<std::string>(2, vcount);
				}
				else if (vcount+1 != line_aux.size())
					throw HMMPI::Exception("Во второй строке заголовка должно быть элементов на один (дата/время) больше, чем в первой строке",
										   "Second line of header should have one element (date/time) more than the first line");

				for (size_t i = 0; i < vcount; i++)
					Hdr(j, i) = HMMPI::ToUpper(line_aux[i + j]);		// "+j" is to shift the second line, since it has one extra element in the beginning
				j++;
			}
		}
		if (j != 2)
			throw HMMPI::Exception("Неправильный заголовок", "Incorrect header");

		msg = "";
		CheckHdrRepeats();
		ReadInd(&msg);
		const size_t pcount = ind.size();
		assert(pcount == vect->WGname.size());
		res = HMMPI::Vector2<double>(dcount, pcount*2);
		found_ts = 0;

		j = 0;						// date index
		HMMPI::Date date0;			// previous date from TEXTSMRY
		while (!sr.eof() && j < dcount)
		{
			std::string line;
			std::vector<std::string> line_aux;

			getline(sr, line);
			size_t ind0 = line.find("--");
			if (ind0 != std::string::npos)
				line = line.substr(0, ind0);

			HMMPI::tokenize(line, line_aux, file_delim, true);
			if (line_aux.size() != 0)		// line is not empty
			{
				HMMPI::Date date1;
				int offset = 0;
				if (line_aux.size() == vcount+1)		// only date
				{
					date1 = HMMPI::Date(line_aux[0]);
					offset = 1;
				}
				else if (line_aux.size() == vcount+2)	// date + time
				{
					date1 = HMMPI::Date(line_aux[0] + " " + line_aux[1]);
					offset = 2;
				}
				else
					throw HMMPI::Exception("Число элементов в строке не соответствует заголовку", "Number of items per line is inconsistent with header");

				if (!(date0 < date1))
					throw HMMPI::Exception((std::string)HMMPI::MessageRE("Даты идут не в возрастающем порядке: ", "Dates are not in increasing order: ") +
							date0.ToString() + ", " + date1.ToString());

				while (j < dcount && date1 > datesW->dates[j])		// "j" is the time step currently being filled
					j++;											// scroll the DATES

				if (j >= dcount)
					continue;

				if (date1 == datesW->dates[j])			// date matches: fill the time step "j"
				{
					for (size_t k = 0; k < pcount; k++)
					{
						if (ind[k] >= 0 && ind[k] < (int)vcount)							// ind[k] is column index
							res(j, k) = HMMPI::StoD(line_aux[ind[k]+offset]);

						if (ind_sigma[k] >= 0 && ind_sigma[k] < (int)vcount)
							res(j, k+pcount) = HMMPI::StoD(line_aux[ind_sigma[k]+offset]);
						else
							res(j, k+pcount) = vect->sigma[k];
					}
					found_ts_flag[j] = true;
					found_ts++;
					j++;
				}
				date0 = date1;
			}
		}
		sr.close();
	}
	catch (...)
	{
		if (sr.is_open())
			sr.close();
		throw;
	}

	return res;
}
//------------------------------------------------------------------------------------------
std::string KW_textsmry::SigmaInfo(const std::string &wgname, const std::string &keyword) const
{
	std::string res = "N/A";
	for (size_t i = 0; i < Hdr.JCount(); i++)
		if (wgname == Hdr(0, i) && keyword + "S" == Hdr(1, i))
		{
			res = Hdr(1, i) + HMMPI::stringFormatArr(" (col. {0:%zu})", std::vector<size_t>{i+1});
			break;
		}

	return res;
}
//------------------------------------------------------------------------------------------
std::vector<double> KW_textsmry::OnlySigmas() const
{
	assert(data.JCount() % 2 == 0);

	size_t NI = data.ICount(), NJ = data.JCount()/2;
	std::vector<double> res(NI*NJ);
	for (size_t i = 0; i < NI; i++)
		for (size_t j = 0; j < NJ; j++)
			res[i + NI*j] = data(i, NJ + j);

	return res;
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_refmap::KW_refmap()
{
	name = "REFMAP";
	erows = 1;
}
//------------------------------------------------------------------------------------------
void KW_refmap::DataIO(int i)
{
	std::string fn = HMMPI::getFullPath(this->CWD, fnames[0]);
	data = ReadData(fn);
	K->AppText(HMMPI::stringFormatArr("(eng) {0:%d}\n", "Reading the file... Loaded values: {0:%d}\n", (int)data.size()));
}
//------------------------------------------------------------------------------------------
std::vector<double> KW_refmap::ReadData(std::string fname)
{
	Start_pre();
	IMPORTKWD(undef, KW_undef, "UNDEF");
	Finish_pre();

	std::ifstream sr;
	sr.exceptions(std::ios_base::badbit);
	std::vector<double> res;

	try
	{
		CheckFileOpen(fname);
		sr.open(fname);
		while (!sr.eof())
		{
			std::string line;
			getline(sr, line);
			std::vector<std::string> line_aux;
			HMMPI::tokenize(line, line_aux, file_delim, true);

			if (line_aux.size() > 0)
				for (size_t k = 0; k < line_aux.size(); k++)
				{
					double d = HMMPI::StoD(line_aux[k]);
					if (d == undef->Ugrid)
						d = std::numeric_limits<double>::quiet_NaN();
					res.push_back(d);
				}
		}
		sr.close();

		if (res.size() == 0)
			throw HMMPI::Exception("(eng)", "No data were loaded");
	}
	catch (...)
	{
		if (sr.is_open())
			sr.close();
		throw;
	}

	return res;
}
//------------------------------------------------------------------------------------------
Grid2D KW_refmap::GetGrid2D(int cX, int cY, std::string fn_init)
{
	Start_pre();
	DECLKWD(rst, KW_funrst, "FUNRST");
	IMPORTKWD(DIMS, KW_griddims, "GRIDDIMS");
	KW_mapseiswght *seiswght = dynamic_cast<KW_mapseiswght*>(K->GetKW_item("MAPSEISWGHT"));	// MAPSEISWGHT is optional
	Finish_pre();

	int Nz = DIMS->Nz;
	std::string PR = DIMS->wght;
	if (PR.length() < 8)
		PR = "'" + PR + std::string(8 - PR.length(), ' ') + "'";
	else
		PR = "'" + PR + "'";

	std::vector<double> WGHT = rst->ReadDataInit(fn_init, PR);
	if (WGHT.size() == 0)
		throw HMMPI::Exception("(eng) (GRIDDIMS->wght) Ã¢ *.FINIT",
							   "Cannot read property for weighting (GRIDDIMS->wght) in *.FINIT");
	if (data.size() == 0)
		return Grid2D();

	size_t tot_len = data.size();
	if (((int)tot_len != cX*cY*Nz)||(tot_len != WGHT.size()))
		throw HMMPI::Exception("(eng) KW_refmap::GetGrid2D",
							   "Incorrect dimensions in KW_refmap::GetGrid2D");
	if (seiswght->data.size() != 0 && tot_len != seiswght->data.size())
		throw HMMPI::Exception("(eng) KW_refmap::GetGrid2D",
							   "Incorrect dimensions in KW_refmap::GetGrid2D");

	Grid2D aux;
	aux.InitData(cX, cY);
	aux.SetGeom(-0.5, -0.5, 1, 1);
	aux.SetVal(0);

	for (int i0 = 0; i0 < cX; i0++)
	{
		for (int j0 = 0; j0 < cY; j0++)
		{
			double d = 0;
			double w = 0;
			for (int k0 = 0; k0 < Nz; k0++)
			{
				double seisw = 1;
				if (seiswght->data.size() != 0)
					seisw = seiswght->data[i0 + (cY - j0 - 1)*cX + k0*cX*cY];

				double add = data[i0 + (cY - j0 - 1)*cX + k0*cX*cY] * WGHT[i0 + (cY - j0 - 1)*cX + k0*cX*cY] * seisw;
				if (!HMMPI::IsNaN(add))
				{
					d += add;
					w += WGHT[i0 + (cY - j0 - 1)*cX + k0*cX*cY] * seisw;
				}
			}
			d /= w;
			if (w == 0)
				d = std::numeric_limits<double>::quiet_NaN();
			if (!HMMPI::IsNaN(d))
			{
				aux.data[i0][j0] = d;
				aux.flag[i0][j0] = 1;
			}
			else
			{
				aux.data[i0][j0] = 0;
				aux.flag[i0][j0] = 0;
			}
		}
	}

	return aux;
}
//------------------------------------------------------------------------------------------
Grid2D KW_refmap::GetGrid2D(int cX, int cY)
{
	Start_pre();
	IMPORTKWD(DIMS, KW_griddims, "GRIDDIMS");
	Finish_pre();

	int Nz = DIMS->Nz;
	if (data.size() == 0)
		return Grid2D();

	size_t tot_len = data.size();
	if ((int)tot_len != cX*cY*Nz)
		throw HMMPI::Exception("(eng) KW_refmap::GetGrid2D(int,int)",
							   "Incorrect dimensions in KW_refmap::GetGrid2D(int,int)");
	Grid2D aux;
	aux.InitData(cX, cY);
	aux.SetGeom(-0.5, -0.5, 1, 1);
	aux.SetVal(0);

	for (int i0 = 0; i0 < cX; i0++)
	{
		for (int j0 = 0; j0 < cY; j0++)
		{
			double d = 0;
			double w = 0;
			for (int k0 = 0; k0 < Nz; k0++)
			{
				double add = data[i0 + (cY - j0 - 1)*cX + k0*cX*cY];
				if (!HMMPI::IsNaN(add))
				{
					d += add;
					w += 1;
				}
			}
			d /= w;
			if (w == 0)
				d = std::numeric_limits<double>::quiet_NaN();
			if (!HMMPI::IsNaN(d))
			{
				aux.data[i0][j0] = d;
				aux.flag[i0][j0] = 1;
			}
			else
			{
				aux.data[i0][j0] = 0;
				aux.flag[i0][j0] = 0;
			}
		}
	}

	return aux;
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_refmap_w::KW_refmap_w()
{
	name = "REFMAP_W";
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_mapreg::KW_mapreg()
{
	name = "MAPREG";
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_mapseisscale::KW_mapseisscale()
{
	name = "MAPSEISSCALE";
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_mapseiswght::KW_mapseiswght()
{
	name = "MAPSEISWGHT";
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_refmapM::KW_refmapM()
{
	name = "REFMAPM";
	erows = -1;
}
//------------------------------------------------------------------------------------------
void KW_refmapM::DataIO(int i)
{
	std::string fn = HMMPI::getFullPath(this->CWD, fnames[i]);
	data[i] = ReadData(fn);
	K->AppText(HMMPI::stringFormatArr(HMMPI::MessageRE("(eng) {0:%d}{1:%d}\n", "Reading file #{0:%d}... Loaded values: {1:%d}\n"), std::vector<int>{i, (int)data[i].size()}));
}
//------------------------------------------------------------------------------------------
void KW_refmapM::AllocateData() noexcept
{
	data = std::vector<std::vector<double>>(fnames.size());
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_initcmaes::KW_initcmaes()
{
	name = "INITCMAES";
	fn_cmaes_init = "cmaes_initials.par";
	erows = 1;
}
//------------------------------------------------------------------------------------------
void KW_initcmaes::DataIO(int i)
{
	std::string fn = HMMPI::getFullPath(this->CWD, fnames[0]);

	std::ifstream sr;
	sr.exceptions(std::ios_base::badbit);

	data = "";
	try
	{
		CheckFileOpen(fn);
		sr.open(fn);
		while (!sr.eof())
		{
			std::string line;
			getline(sr, line);
			data += line + "\n";
		}

		K->AppText(HMMPI::MessageRE("Чтение файла...\n", "Reading the file...\n"));
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
void KW_initcmaes::WriteFromLimits(const std::vector<double> &x)	// to be called on all ranks
{
	DECLKWD(limits, KW_limits, "LIMITS");
	DECLKWD(params, KW_parameters, "PARAMETERS");
	const ParamsInterface *par_interface = params->GetParamsInterface();
	const std::vector<double> min = par_interface->actmin();
	const std::vector<double> max = par_interface->actmax();

	std::string fn = HMMPI::getFullPath(this->CWD, fn_cmaes_init);
	std::ofstream sw;		// only used on RNK-0
	if (K->MPI_rank == 0)
		sw.exceptions(std::ios_base::badbit | std::ios_base::failbit);

	const double std_frac = 0.2;	// for PARAMETERS case std = std_frac*(max-min)

	if (limits->GetState() == "")	// LIMITS case
	{
		std::string msg = limits->CheckPositive(limits->std, "std");
		if (msg != "")
			throw HMMPI::Exception(msg);
	}

	try
	{
		if (K->MPI_rank == 0)
			sw.open(fn);
		std::string init = "";
		std::string typ = "";
		std::string std = "";
		size_t len = par_interface->init.size();
		if (len != x.size())
			throw HMMPI::Exception("x.size() != limits->init.size() in KW_initcmaes::WriteFromLimits");

		int act_len = par_interface->get_act_ind().size();
		int cur = 0;
		for (size_t i = 0; i < len; i++)
		{
			if (par_interface->act[i] == "A")
			{
				std::string ret = "\n";
				if (cur == act_len-1)
					ret = "";

				double std_val;
				if (limits->GetState() == "")		// LIMITS case
					std_val = limits->std[i];
				else
					std_val = (max[i] - min[i])*std_frac;		// PARAMETERS case

				init += HMMPI::stringFormatArr("   {0}", std::vector<double>{x[i]}) + ret;
				typ += HMMPI::stringFormatArr("   {0}", std::vector<double>{x[i]}) + ret;	// was 'typ' before 29.08.2016
				std += HMMPI::stringFormatArr("   {0}", std::vector<double>{std_val}) + ret;
				cur++;
			}
		}

		std::string actlen = HMMPI::stringFormatArr("{0:%d}", std::vector<int>{act_len});
		std::vector<std::string> arr = std::vector<std::string>{actlen, init, typ, std};
		if (K->MPI_rank == 0)
		{
			sw << HMMPI::stringFormatArr(data, arr);
			sw.close();
		}
	}
	catch (...)
	{
		if (K->MPI_rank == 0 && sw.is_open())
			sw.close();
		throw;
	}
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_datafile::KW_datafile()
{
	name = "DATAFILE";
	erows = 1;
}
//------------------------------------------------------------------------------------------
void KW_datafile::DataIO(int i)
{
	std::string fn = HMMPI::getFullPath(this->CWD, fnames[0]);		// PATH/MODEL0.DATA

	std::ifstream sr;
	sr.exceptions(std::ios_base::badbit);

	try
	{
		CheckFileOpen(fn);

		std::string mod_name = HMMPI::getFile(fn);		
		path = HMMPI::getCWD(fn);	

		if (mod_name.length() <= 6 || mod_name.substr(mod_name.length()-6) != "0.DATA")
			throw HMMPI::Exception("Дата-файл должен оканчиваться на ...0.DATA", "Data file should have ...0.DATA ending");

		base_name = mod_name.substr(0, mod_name.length()-6);				// MODEL
		std::string cont_delim = mod_name.substr(0, mod_name.length()-5);	// MODEL0

		contents = "";

		sr.open(fn);
		while (!sr.eof())
		{
			std::string line;
			getline(sr, line);
			contents += line + "\n";
		}

		K->AppText(HMMPI::MessageRE("Чтение файла ...\n", "Reading the file...\n"));
		sr.close();

		HMMPI::tokenizeExact(contents, cont_split, cont_delim, true);
		K->AppText(HMMPI::stringFormatArr("Найдено {0:%d} вхождений шаблона\n", "Found {0:%d} template inclusion(s)\n", (int)cont_split.size()-1));
	}
	catch (...)
	{
		if (sr.is_open())
			sr.close();
		throw;
	}
}
//------------------------------------------------------------------------------------------
void KW_datafile::WriteDataFile(int i, bool adjrun)
{
	std::ofstream sw;
	sw.exceptions(std::ios_base::badbit | std::ios_base::failbit);

	try
	{
		sw.open(GetDataFileName(i));		// PATH/MODEL(i).DATA

		for (size_t k = 0; k < cont_split.size(); k++)
		{
			std::string segment = cont_split[k];

			if (adjrun)		// ad hoc switch of simulation type to adjoint gradient run for PUNQ-S3
			{
				segment = HMMPI::Replace(segment, "\nSkip", "\n--Skip");
				segment = HMMPI::Replace(segment, "\nEndskip", "\n--Endskip");
			}
			else			// ad hoc switch of simulation type to normal run for PUNQ-S3
			{
				segment = HMMPI::Replace(segment, "\n--Skip", "\nSkip");
				segment = HMMPI::Replace(segment, "\n--Endskip", "\nEndskip");

			}

			sw << segment;
			if (k < cont_split.size()-1)
				sw << base_name + HMMPI::stringFormatArr("{0:%d}", std::vector<int>{i});	// MODEL(i)
		}

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
std::string KW_datafile::GetDataFileName(int i)
{
	return HMMPI::getFullPath(path, base_name) + HMMPI::stringFormatArr("{0:%d}.DATA", std::vector<int>{i});
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_CoordZcorn::KW_CoordZcorn() : CG(MPI_COMM_WORLD)
{
	name = "COORDZCORN";
	erows = 1;
}
//------------------------------------------------------------------------------------------
void KW_CoordZcorn::DataIO(int i)
{
	Start_pre();
	IMPORTKWD(dims, KW_griddimens, "GRIDDIMENS");
	Finish_pre();

	std::string fn = HMMPI::getFullPath(this->CWD, fnames[0]);
	std::string msg = HMMPI::MessageRE("Чтение файла...\n", "Reading the file...\n");

	msg += CG.LoadCOORD_ZCORN(fn, dims->Nx, dims->Ny, dims->Nz, dims->X0, dims->Y0, (dims->grid_Y_axis == "POS"), dims->actnum_name, dims->actnum_min);
	K->AppText(msg);
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
KW_Actnum::KW_Actnum()
{
	name = "ACTNUM";
	erows = 1;
}
//------------------------------------------------------------------------------------------
void KW_Actnum::DataIO(int i)
{
	Start_pre();
	IMPORTKWD(cz, KW_CoordZcorn, "COORDZCORN");
	Finish_pre();

	std::string fn = HMMPI::getFullPath(this->CWD, fnames[0]);
	std::string msg = cz->CG.LoadACTNUM(fn);

	K->AppText(std::string(HMMPI::MessageRE("Чтение файла...\n", "Reading the file...\n")) + msg + "\n");
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
// descendants of KW_fwrite
//------------------------------------------------------------------------------------------
KW_WRfunrst::KW_WRfunrst()
{
	name = "WRFUNRST";
	erows = 1;
}
//------------------------------------------------------------------------------------------
void KW_WRfunrst::DataIO(int i)
{
	Start_pre();
	IMPORTKWD(DIMS, KW_griddims, "GRIDDIMS");
	IMPORTKWD(refmapM, KW_refmapM, "REFMAPM");
	IMPORTKWD(fegrid, KW_fegrid, "FEGRID");
	IMPORTKWD(satsteps, KW_satsteps, "SATSTEPS");
	Finish_pre();

	std::string fn = HMMPI::getFullPath(this->CWD, fnames[0]);
	std::string seq_hdr = " 'SEQNUM  '           1 'INTE'\n           {0:%d}\n";
	std::string prop_hdr = " '{0:%s}'          {1:%s} 'REAL'\n";
	std::string prop0 = DIMS->krig_prop;
	if (prop0.length() < 8)
		prop0 = prop0 + std::string(8 - prop0.length(), ' ');

	std::ofstream sw;
	sw.exceptions(std::ios_base::badbit | std::ios_base::failbit);

	size_t steps = refmapM->data.size();
	if (steps != satsteps->data.size())
		throw HMMPI::Exception(HMMPI::stringFormatArr(HMMPI::MessageRE("(eng) REFMAPM {0:%d}, SATSTEPS {1:%d}",
								"Different number of steps in REFMAPM ({0:%d}) and SATSTEPS ({1:%d})"), std::vector<int>{(int)steps, (int)satsteps->data.size()}));
	if ((int)fegrid->data.size() != DIMS->Nx * DIMS->Ny * DIMS->Nz)
		throw HMMPI::Exception("(eng) FEGRID GRIDDIMS", "Grid size in FEGRID is not consistent with GRIDDIMS");

	int act_count = int(fegrid->Sum());
	try
	{
		sw.open(fn);
		for (size_t s = 0; s < steps; s++)
		{
			size_t count = refmapM->data[s].size();
			if (count != fegrid->data.size())
				throw HMMPI::Exception(HMMPI::stringFormatArr(HMMPI::MessageRE("(eng) REFMAPM {0:%d}, FEGRID {1:%d}",
										"Different grid sizes in REFMAPM ({0:%d} cells) and FEGRID ({1:%d} cells)"), std::vector<int>{(int)count, (int)fegrid->data.size()}));

			std::string actcount = HMMPI::stringFormatArr("{0:%d}", std::vector<int>{act_count});
			sw << HMMPI::stringFormatArr(seq_hdr, std::vector<int>{satsteps->data[s]});
			sw << HMMPI::stringFormatArr(prop_hdr, std::vector<std::string>{prop0, actcount});
			int CNT = 0;
			for (size_t i = 0; i < count; i++)
			{
				if (fegrid->data[i] != 0)
				{
					sw << HMMPI::stringFormatArr("   {0:%16.8e}", std::vector<double>{refmapM->data[s][i]});
					CNT++;
				}
				if (CNT == 4)
				{
					sw << "\n";
					CNT = 0;
				}
			}
			if (CNT != 0)
				sw << "\n";
		}

		K->AppText(HMMPI::stringFormatArr(HMMPI::MessageRE("(eng) {0:%d} x {1:%d}\n",
											"Writing the file...\nSaved steps x values: {0:%d} x {1:%d}\n"), std::vector<int>{(int)steps, act_count}));
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
//------------------------------------------------------------------------------------------
KW_report::KW_report()
{
	name = "REPORT";
	erows = 1;
}
//------------------------------------------------------------------------------------------
void KW_report::data_io()					// actual output to the file, performed in the end
{
	if (K->MPI_rank == 0)
	{
		std::string fn = HMMPI::getFullPath(this->CWD, fnames[0]);

		std::ofstream sw;
		sw.exceptions(std::ios_base::badbit | std::ios_base::failbit);
		try
		{
			sw.open(fn);
			sw << K->report;
			sw.close();
		}
		catch (...)
		{
			if (sw.is_open())
				sw.close();
			throw;
		}
	}
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
