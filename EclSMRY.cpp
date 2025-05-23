/*
 * EclSMRY.cpp
 *
 *  Created on: 30 Jan 2017
 *      Author: ilya fursov
 */

#include <cassert>
#include <algorithm>
#include "EclSMRY.h"
#include "Utils.h"
#include "MathUtils.h"
#include "ConcretePhysModels.h"
#include "Parsing2.h"

#define NUM_TO_STR_00(x) #x
#define NUM_TO_STR(x) NUM_TO_STR_00(x)

namespace HMMPI
{
//--------------------------------------------------------------------------------------------------
void EclSMRYInitCheckSizes()
{
	assert(sizeof(float) == sizeof(int32_t));
	assert(sizeof(double) == sizeof(int64_t));
}
//--------------------------------------------------------------------------------------------------
inline void ToChar(int32_t x, char *s)			// 0-end should be set for 's' elsewhere
{
	s[0] = (x >> 24) & 0xFF;
	s[1] = (x >> 16) & 0xFF;
	s[2] = (x >> 8) & 0xFF;
	s[3] = x & 0xFF;
}
//--------------------------------------------------------------------------------------------------
inline int32_t SwapEndian(int32_t x)
{
	int32_t b0, b1, b2, b3;

	b0 = (x & 0x000000ff) << 24;
	b1 = (x & 0x0000ff00) << 8;
	b2 = (x & 0x00ff0000) >> 8;
	b3 = (x & 0xff000000) >> 24;

	return b0 | b1 | b2 | b3;
}
//--------------------------------------------------------------------------------------------------
template <>
inline int32_t ReadVal<int32_t>(FILE *f)
{
	int32_t res = 0;
	fread(&res, 4, 1, f);

	return SwapEndian(res);
}
//--------------------------------------------------------------------------------------------------
template <>
inline uint32_t ReadVal<uint32_t>(FILE *f)
{
	int32_t res = ReadVal<int32_t>(f);
	return *(uint32_t*)&res;
}
//--------------------------------------------------------------------------------------------------
template <>
inline std::string ReadVal<std::string>(FILE *f)
{
	char res[9];

	fread(res, 4, 1, f);
	fread(res+4, 4, 1, f);
	res[8] = 0;

	return (std::string)res;
}
//--------------------------------------------------------------------------------------------------
template <>
inline float ReadVal<float>(FILE *f)
{
	int32_t res = ReadVal<int32_t>(f);
	return *(float*)&res;
}
//--------------------------------------------------------------------------------------------------
template <>
inline double ReadVal<double>(FILE *f)
{
	int64_t a1 = ReadVal<int32_t>(f);
	int64_t a2 = ReadVal<int32_t>(f);
	int64_t res = (a1 << 32) | a2;

	return *(double*)&res;
}
//--------------------------------------------------------------------------------------------------
// Date
//--------------------------------------------------------------------------------------------------
Date::Date(const std::string &s)		// accepted 's' formats: DD.MM.YYYY, DD/MM/YYYY, optionally followed by " hh:mm::ss" or " hh:mm"
{
	// parsing date-time
	std::vector<std::string> date_time;
	tokenize(s, date_time, " \t", true);

	if (date_time.size() != 1 && date_time.size() != 2)
		throw Exception("Некорректный формат даты/времени: '" + s + "'", "Incorrect date/time format: '" + s + "'");

	// parsing date
	parse_date_time(date_time[0], "./", Day, Month, Year);

	fmt.delim_type = -1;
	if (date_time[0].find_first_of("/") != std::string::npos) fmt.delim_type = 0;
	if (date_time[0].find_first_of(".") != std::string::npos) {
		if (fmt.delim_type == 0)
			throw Exception("Несогласованные разделители (/ и .) в дате '" + s + "'", "Inconsistent delimiters (/ and .) in date '" + s + "'");
		fmt.delim_type = 1;
	}

	// parsing time
	if (date_time.size() == 2) {
		int hh, mm, ss, c;
		c = parse_date_time(date_time[1], ":", hh, mm, ss);
		sec = ss + mm*60 + hh*3600;
		fmt.hms = c - 1;
	} else {
		sec = 0;
		fmt.hms = 0;
	}
}
//--------------------------------------------------------------------------------------------------
bool Date::operator>(const Date &rhs) const
{
	if (Year > rhs.Year)
		return true;
	if (Year == rhs.Year && Month > rhs.Month)
		return true;
	if (Year == rhs.Year && Month == rhs.Month && Day > rhs.Day)
		return true;
	if (Year == rhs.Year && Month == rhs.Month && Day == rhs.Day && sec > rhs.sec)
		return true;

	return false;
}
//--------------------------------------------------------------------------------------------------
void Date::add(double days)				// shift the given date-time by the time interval provided (1.0 = 1 day)
{
	static const double T4 = 1461;		// days in 4-year cycle

	Date t0(1, 1, Year-Year%4, 0.0);	// beginning of a 4-year cycle
	double delta0 = diff(t0);

	double dfull = delta0 + days;
	const int n0 = floor(dfull/T4);		// number of cycles
	double a0 = dfull - n0*T4;			// residue in the cycle
	assert(a0 >= 0);

	const size_t y0 = std::upper_bound(T4LEN.begin(), T4LEN.end(), (int)a0) - T4LEN.begin();		// number of the year in the cycle
	assert(1 <= y0 && y0 < T4LEN.size());
	double b0 = a0 - T4LEN[y0-1];		// residue in the year
	assert(b0 >= 0);

	const std::vector<int> *monptr = &MLEN;
	if (y0 == 1) monptr = &MLENleap;
	const size_t m0 = std::upper_bound(monptr->begin(), monptr->end(), (int)b0) - monptr->begin();	// number of the month in the year
	assert(1 <= m0 && m0 < monptr->size());

	double d0 = b0 - (*monptr)[m0-1];
	assert(d0 >= 0);

	Year = t0.Year + n0*4 + y0 - 1;
	Month = m0;
	Day = (int)d0 + 1;
	sec = (d0 - (int)d0)*86400;
}
//--------------------------------------------------------------------------------------------------
double Date::diff(const Date &y) const	// diff = *this - y, difference in days
{
	const int deltaD = Day - y.Day;
	const int deltaM = MLEN[Month-1] - MLEN[y.Month-1];
	const int deltaY = (Year - y.Year)*365;
	const double deltaSec = (sec - y.sec)/86400;

	// count leap years: this approach is valid for years 1904 to 2096
	int leap = Year/4;
	if (Year%4 == 0 && Month <= 2)
		leap--;

	int leap_y = y.Year/4;
	if (y.Year%4 == 0 && y.Month <= 2)
		leap_y--;

	return deltaD + deltaM + deltaY + leap-leap_y + deltaSec;
}
//--------------------------------------------------------------------------------------------------
std::string Date::ToString(const DateFmt *f) const
{
	const static std::string date_dot   =  "{0:%02d}.{1:%02d}.{2:%04d}";
	const static std::string date_slash =  "{0:%02d}/{1:%02d}/{2:%04d}";
	const static std::string time2      = " {0:%02d}:{1:%02d}";
	const static std::string time3      = " {0:%02d}:{1:%02d}:{2:%02d}";
	const std::vector<int> DMY = {Day, Month, Year};

	const int s = (int)sec;
	const int ss = s % 60;
	const int mm = (s - ss) / 60 % 60;
	const int hh = (s - ss - mm * 60) / 3600;
	std::string res;

	if (f == nullptr) {			// default formatting
		res = stringFormatArr(MessageRE(date_dot, date_slash), DMY);
		if (s != 0) res += stringFormatArr(time3, std::vector<int>{hh, mm, ss});
	} else {					// user formatting
		res = stringFormatArr(f->delim_type ? date_dot : date_slash, DMY);
		if (f->hms == 1) res += stringFormatArr(time2, std::vector<int>{hh, mm});
		if (f->hms == 2) res += stringFormatArr(time3, std::vector<int>{hh, mm, ss});
	}
	return res;
}
//--------------------------------------------------------------------------------------------------
void Date::write_bin(FILE *fd) const
{
	fwrite(&Day, sizeof(Day), 1, fd);
	fwrite(&Month, sizeof(Month), 1, fd);
	fwrite(&Year, sizeof(Year), 1, fd);
	fwrite(&sec, sizeof(sec), 1, fd);
}
//--------------------------------------------------------------------------------------------------
void Date::read_bin(FILE *fd)
{
	fread_check(&Day, sizeof(Day), 1, fd);
	fread_check(&Month, sizeof(Month), 1, fd);
	fread_check(&Year, sizeof(Year), 1, fd);
	fread_check(&sec, sizeof(sec), 1, fd);
}
//--------------------------------------------------------------------------------------------------
int Date::parse_date_time(const std::string s, std::string delim, int &D, int &M, int &Y)	// can parse both DD.MM.YYYY, DD/MM/YYYY and hh:mm:ss
{																							// if the last item ("YYYY" or "ss") is empty, then Y = 0
	std::vector<std::string> parsed;														// returns the number of items filled (3 or 2)
	tokenize(s, parsed, delim, true);
	if (parsed.size() != 3 && parsed.size() != 2)
		throw Exception("Некорректный формат даты/времени: '" + s + "'",
						"Incorrect date/time format: '" + s + "'");

	bool ok = true;
	if (ok) D = StoL(parsed[0], ok);
	if (ok) M = StoL(parsed[1], ok);
	if (ok && parsed.size() == 3)
		Y = StoL(parsed[2], ok);
	else
		Y = 0;

	if (!ok)
		throw Exception("Некорректная дата/время (ошибка с числами) '" + s + "'",
						"Incorrect date/time (error with numbers) '" + s + "'");
	return parsed.size();
}
//--------------------------------------------------------------------------------------------------
std::vector<double> Date::SubtractFromAll(const std::vector<Date> &vec) const	// subtracts *this from all elements of 'vec'
{																				// the resulting elementwise difference in _days_ is returned
	const size_t count = vec.size();
	std::vector<double> res(count);

	for (size_t i = 0; i < count; i++) res[i] = vec[i].diff(*this);
	return res;
}
//--------------------------------------------------------------------------------------------------
// class SmryKwd
//--------------------------------------------------------------------------------------------------
KwdType SmryKwd::type_from_str(std::string s)
{
	if (s == "INTE")
		return INTE;
	else if (s == "REAL")
		return REAL;
	else if (s == "DOUB")
		return DOUB;
	else if (s == "CHAR")
		return CHAR;
	else if (s == "LOGI")
		return LOGI;
	else
		throw Exception("Unknown value type " + s);
}
//--------------------------------------------------------------------------------------------------
void SmryKwd::ReadHeader(FILE *f)
{
	int a = ReadVal<int32_t>(f);
	if (feof(f))
		return;
	assert(a == 16);

	name = ReadVal<std::string>(f);
	len = ReadVal<int32_t>(f);

	a = ReadVal<int32_t>(f);
	char valtype[5];
	ToChar(a, valtype);
	valtype[4] = 0;
	type = type_from_str(valtype);
}
//--------------------------------------------------------------------------------------------------
int SmryKwd::ReadNamedHeader(FILE *f)
{
	std::string seek_name = name;
	bool finished = false;

	while (!finished && !feof(f))
	{
		ReadHeader(f);
		if (name == seek_name)
			finished = true;
		else
			SkipReadData(f);
	}

	if (finished)
		return 0;
	else
		return 1;
}
//--------------------------------------------------------------------------------------------------
SmryKwd *SmryKwd::ReadData(FILE *f)
{
	SmryKwd *res = 0;
	if (type == INTE)
		res = new SmryKwdData<int32_t>(*this);
	else if (type == REAL)
		res = new SmryKwdData<float>(*this);
	else if (type == DOUB)
		res = new SmryKwdData<double>(*this);
	else if (type == CHAR)
		res = new SmryKwdData<std::string>(*this);
	else if (type == LOGI)
		res = new SmryKwdData<uint32_t>(*this);
	else
		throw Exception("Unknown value type " + type);

	int mult = 4;							// sizeof(type)
	if (type == CHAR || type == DOUB)
		mult = 8;

	int a = ReadVal<int32_t>(f);
	assert(a == 16);

	size_t total_read = 0;					// number of values currently read
	while (total_read < len)
	{
		int totlen = ReadVal<int32_t>(f);	// number of bytes in current block
		assert(totlen % mult == 0);
		totlen /= mult;						// number of values to read in current block

		res->read_array(f, total_read, totlen);

		a = ReadVal<int32_t>(f);
		assert(a == totlen*mult);
		total_read += totlen;
	}

	return res;
};
//--------------------------------------------------------------------------------------------------
void SmryKwd::SkipReadData(FILE *f)
{
	int mult = 4;							// sizeof(type)
	if (type == CHAR || type == DOUB)
		mult = 8;

	int a = ReadVal<int32_t>(f);
	if (feof(f))
		return;
	assert(a == 16);

	size_t total_read = 0;
	while (total_read < len)
	{
		int totlen = ReadVal<int32_t>(f);	// number of bytes in current block
		assert(totlen % mult == 0);

		fseek(f, totlen, SEEK_CUR);

		a = ReadVal<int32_t>(f);
		assert(a == totlen);
		total_read += totlen/mult;
	}
}
//--------------------------------------------------------------------------------------------------
// class SmryKwdData
//--------------------------------------------------------------------------------------------------
template <class T>
void SmryKwdData<T>::read_array(FILE *f, size_t start, size_t count)
{
	assert(start < len && start + count <= len);
	if (start == 0)
		data = std::vector<T>(len);

	for (size_t i = start; i < start + count; i++)
		data[i] = ReadVal<T>(f);
}
//--------------------------------------------------------------------------------------------------
template <class T>
void SmryKwdData<T>::cout()
{
	assert(len == data.size());

	std::cout << name << "\t" << len << "\t" << type << "\n";
	for (size_t i = 0; i < len; i++)
	{
		std::cout << data[i];
		if (i < len-1)
			std::cout << "\t";
		else
			std::cout << "\n";
	}
}
//--------------------------------------------------------------------------------------------------
// class SimSMRY
//--------------------------------------------------------------------------------------------------
std::string SimSMRY::DatesToStr() const
{
	std::string res = "";
	for (size_t i = 0; i < dates.size(); i++)
		res += dates[i].ToString() + "\n";

	return res;
}
//--------------------------------------------------------------------------------------------------
std::string SimSMRY::VecsToStr() const
{
	std::string res = "";
	for (size_t i = 0; i < vecs.size(); i++)
		res += stringFormatArr("{0:%s}\t{1:%s}\n", std::vector<std::string>{vecs[i].first, vecs[i].second});

	return res;
}
//--------------------------------------------------------------------------------------------------
Mat SimSMRY::ExtractSummary(const std::vector<Date> &dates1, std::vector<pair> vecs1, std::string &msg_dat_short, std::string &msg_vec_short,
						   std::string &msg_dat_full, std::string &msg_vec_full, int N, std::string suff) const
{																// extracts summary, as defined by [dates1 x vecs1],
	// attach "suff"											// fills summary with "0" where dates1[*] or vecs1[*] are not found in this->dates, this->vecs
	if (suff != "")												// before searching vectors, attaches "suff" (e.g. "H", "S") to vecs1[*].second, making e.g. WBHP+H, WWCT+S
		for (auto &v : vecs1)									// "msg_dat_short", "msg_vec_short" and their full versions return info about not-found dates and vectors
			v.second += suff;									// 'N' is the StringListing parameter used for the short message versions.

	std::vector<size_t> inddates = GetSubvecInd(dates, dates1);
	std::vector<size_t> indvecs = GetSubvecInd(vecs, vecs1);

	// generate "msgs"
	StringListing stldates("\t"), stlvecs("\t");
	int count_d = 0, count_v = 0;
	for (size_t i = 0; i < inddates.size(); i++)
		if (inddates[i] == (size_t)-1)
		{
			stldates.AddLine(std::vector<std::string>{dates1[i].ToString()});
			count_d++;
		}

	if (count_d > 0)
	{
		msg_dat_short = stringFormatArr("В " + dates_file() + " не найдены даты ({0:%d}):\n", "In " + dates_file() + " the following dates ({0:%d}) were not found:\n", count_d) + stldates.Print(N, N);
		msg_dat_full = stringFormatArr("В " + dates_file() + " не найдены даты ({0:%d}):\n", "In " + dates_file() + " the following dates ({0:%d}) were not found:\n", count_d) + stldates.Print(-1, -1);
	}

	for (size_t i = 0; i < indvecs.size(); i++)
		if (indvecs[i] == (size_t)-1)
		{
			stlvecs.AddLine(std::vector<std::string>{vecs1[i].first, vecs1[i].second});
			count_v++;
		}

	if (count_v > 0)
	{
		msg_vec_short = stringFormatArr("В " + vecs_file() + " не найдены вектора ({0:%d}):\n", "In " + vecs_file() + " the following vectors ({0:%d}) were not found:\n", count_v) + stlvecs.Print(N, N);
		msg_vec_full = stringFormatArr("В " + vecs_file() + " не найдены вектора ({0:%d}):\n", "In " + vecs_file() + " the following vectors ({0:%d}) were not found:\n", count_v) + stlvecs.Print(-1, -1);
	}

	return Mat(Reorder(Data, dates.size(), vecs.size(), inddates, indvecs, true), dates1.size(), vecs1.size());
}
//--------------------------------------------------------------------------------------------------
// class EclSMRY
//--------------------------------------------------------------------------------------------------
void EclSMRY::readSMSPEC(std::string modname)
{
	FILE *file = NULL;
	try
	{
		file = fopen((modname + ".SMSPEC").c_str(), "rb");
		if (file == NULL)
			throw Exception(stringFormatArr("Невозможно открыть файл {0:%s}", "Cannot open file {0:%s}", modname + ".SMSPEC"));

		std::vector<std::string> kwds = {"KEYWORDS", "WGNAMES ", "UNITS   "};
		std::vector<std::vector<std::string>> kwdata(kwds.size());		// kwdata[0]...kwdata[2] correspond to the three 'kwds' elements

		size_t count = 0;			// counts keywords that have been read
		SmryKwd K;
		while (!feof(file) && count < kwds.size())
		{
			K.ReadHeader(file);

			if (feof(file))
				continue;

			SmryKwd *pk = 0;
			int ind = std::find(kwds.begin(), kwds.end(), K.name) - kwds.begin();		// check if the current keyword is needed for this procedure
			if ((size_t)ind < kwds.size())		// found a keyword from 'kwds'
			{
				pk = K.ReadData(file);
				SmryKwdData<std::string> *pk_str = dynamic_cast<SmryKwdData<std::string>*>(pk);
				if (pk_str == 0)
					throw Exception("Summary keyword " + K.name + " is followed by unexpected type in EclSMRY::readSMSPEC");
				kwdata[ind] = std::move(pk_str->data);
				count++;
			}
			else
				K.SkipReadData(file);			// keyword not found -- keep reading

			delete pk;
		}
		if (file) fclose(file);
		file = NULL;

		// fill 'Units' and 'vecs'
		assert(kwdata[0].size() == kwdata[1].size() && kwdata[1].size() == kwdata[2].size());

		Units = std::move(kwdata[2]);
		vecs.resize(Units.size());
		for (size_t i = 0; i < vecs.size(); i++)
		{
			vecs[i].first = Trim(kwdata[1][i], " ");
			if (vecs[i].first == ":+:+:+:+")			// a hack; tNavigator and Eclipse may write ":+:+:+:+" and whitespace not in the same manner
				vecs[i].first = "";						// intended to properly load DAY, MONTH, YEAR
			vecs[i].second = Trim(kwdata[0][i], " ");
		}
	}
	catch (...)
	{
		if (file) fclose(file);
		throw;
	}
}
//--------------------------------------------------------------------------------------------------
void EclSMRY::readUNSMRY(std::string modname)
{
	FILE *file = NULL;
	try
	{
		file = fopen((modname + ".UNSMRY").c_str(), "rb");
		if (file == NULL)
			throw Exception(stringFormatArr("Невозможно открыть файл {0:%s}", "Cannot open file {0:%s}", modname + ".UNSMRY"));

		SmryKwd K;
		std::vector<float> data_fl;		// first get data to 'float' vector
		while (!feof(file))
		{
			K.name = "PARAMS  ";
			if (K.ReadNamedHeader(file) != 0)
				break;					// PARAMS not found

			if (feof(file))
				continue;

			SmryKwd *pk = K.ReadData(file);
			SmryKwdData<float> *pk_fl = dynamic_cast<SmryKwdData<float>*>(pk);
			if (pk_fl == 0)
				throw Exception("Summary keyword " + K.name + " is followed by unexpected type in EclSMRY::readUNSMRY");
			if (pk_fl->data.size() != vecs.size())
				throw Exception(stringFormatArr("Loaded {0:%zu} vectors, but {1:%zu} data values (at certain step)", std::vector<size_t>{vecs.size(), pk_fl->data.size()}));

			HMMPI::VecAppend(data_fl, pk_fl->data);
			delete pk;
		}
		if (file) fclose(file);
		file = NULL;

		// fill 'dates'
		std::vector<size_t> date_ind = HMMPI::GetSubvecInd(vecs, date_vec);		// find indices of DAY, MONTH, YEAR inside 'vecs'
		if (std::find(date_ind.begin(), date_ind.end(), -1) != date_ind.end())
			throw Exception("В файле " + modname + ".SMSPEC не найдены ключевые слова DAY, MONTH, YEAR; возможно, в секции SUMMARY дата-файла Эклипса не задано 'DATE'",
							"Keywords DAY, MONTH, YEAR were not found in " + modname + ".SMSPEC file; SUMMARY section of Eclipse data-file may be missing 'DATE'");

		Data = std::vector<double>(data_fl.begin(), data_fl.end());				// convert float -> double
		assert(Data.size() % vecs.size() == 0);
		dates.resize(Data.size() / vecs.size());				// size = number of time steps

		std::vector<size_t> date_seq(dates.size());				// [0, 1, 2,...]
		std::iota(date_seq.begin(), date_seq.end(), 0);
		std::vector<double> date_dbl = Reorder(Data, dates.size(), vecs.size(), date_seq, date_ind);		// all dates put into a single array
		for (size_t t = 0; t < dates.size(); t++)
			dates[t] = Date((int)date_dbl[3*t], (int)date_dbl[3*t+1], (int)date_dbl[3*t+2]);				// NB Date.sec = 0 is taken
	}
	catch (...)
	{
		if (file) fclose(file);
		throw;
	}
}
//--------------------------------------------------------------------------------------------------
EclSMRY::EclSMRY() : date_vec({pair("", "DAY"), pair("", "MONTH"), pair("", "YEAR")})
{
}
//--------------------------------------------------------------------------------------------------
SimSMRY *EclSMRY::Copy() const
{
	return new EclSMRY(*this);
}
//--------------------------------------------------------------------------------------------------
void EclSMRY::ReadFiles(std::string modname)
{
	readSMSPEC(modname);
	readUNSMRY(modname);
	mod = modname;
}
//--------------------------------------------------------------------------------------------------
std::string EclSMRY::VecsToStr() const
{
	std::string res = "";
	for (size_t i = 0; i < vecs.size(); i++)
		res += stringFormatArr("{0:%s}\t{1:%s}\t{2:%s}\n", std::vector<std::string>{vecs[i].first, vecs[i].second, Trim(Units[i], " ")});

	return res;
}
//--------------------------------------------------------------------------------------------------
std::string EclSMRY::dates_file() const
{
	return EraseSubstr(mod + ".UNSMRY", "./");
}
//--------------------------------------------------------------------------------------------------
std::string EclSMRY::vecs_file() const
{
	return EraseSubstr(mod + ".SMSPEC", "./");
}
//--------------------------------------------------------------------------------------------------
std::string EclSMRY::data_file() const				// name of file with data
{
	return EraseSubstr(mod + ".UNSMRY", "./");
}
//--------------------------------------------------------------------------------------------------
// tNavSMRY
//--------------------------------------------------------------------------------------------------
// tNavSMRY::T_sec_obj
//--------------------------------------------------------------------------------------------------
void tNavSMRY::T_sec_obj::func(double *x, int flag, int wght_offset) const		// HIGHLIGHT: calculation of *x based on the other data from array 'x'
{																				// 'flag' and 'wght_offset' should be taken from T_ecl_prop_transform
	if (flag == 0)
		*x = 0;
	else if (flag == 1)		// simple summation
	{
		double d = 0;
		for (size_t i = 0; i < offsets.size(); i++)
			d += *(x + offsets[i]);
		*x = d;
	}
	else					// weighted summation
	{
		double d = 0;
		for (size_t i = 0; i < offsets.size(); i++)
			d += *(x + offsets[i]) * *(x + offsets[i] + wght_offset);
		*x = d;
	}
}
//--------------------------------------------------------------------------------------------------
// NB! primary and secondary objects are assumed to be arranged in consecutive chunks: [primary | secondary]
// in the two CTORs below 'ind' is the index of the secondary object in the [secondary] chunk, i.e. 0, 1, 2,...
tNavSMRY::T_sec_obj::T_sec_obj(int ind, int obj_N) : name("FIELD")
{															// takes name = 'FIELD' and assumes that all (obj_N) primary objects will be summed
	offsets = std::vector<int>(obj_N);
	std::iota(offsets.begin(), offsets.end(), -obj_N - ind);
}
//--------------------------------------------------------------------------------------------------
tNavSMRY::T_sec_obj::T_sec_obj(int ind, std::string Name, const std::vector<std::string> &subord, const std::vector<std::string> &full) : name(Name)
{															// subordinate objects are found in the full list of primary objects to fill 'offsets'
	offsets = std::vector<int>(subord.size());
	for (size_t i = 0; i < subord.size(); i++)
	{
		size_t k = std::find(full.begin(), full.end(), subord[i]) - full.begin();
		if (k == full.size())		// object not found
		{
			char msgrus[BUFFSIZE], msgeng[BUFFSIZE];
			sprintf(msgrus, "При чтении *_well.meta НЕ НАЙДЕН объект/скважина %.100s из группы %.100s", subord[i].c_str(), name.c_str());
			sprintf(msgeng, "Object/well %.100s from group %.100s NOT FOUND while reading *_well.meta", subord[i].c_str(), name.c_str());
			throw Exception(msgrus, msgeng);
		}
		offsets[i] = (int)k - (int)full.size() - ind;
	}
}
//--------------------------------------------------------------------------------------------------
std::string tNavSMRY::T_sec_obj::ToString() const			// for debug output
{
	char msg[BUFFSIZE];
	sprintf(msg, "%.100s:", name.c_str());
	if (offsets.size() > 0)
		return (std::string)msg + " offsets = " + HMMPI::ToString(offsets, "%d");
	else
		return (std::string)msg + "\n";
}
//--------------------------------------------------------------------------------------------------
void tNavSMRY::ecl_prop_transform_check() const
{
	for (const auto &x : ecl_prop_transform)
	{
		if ((x.flag == 1 && x.args.size() == 0)||(x.flag != 1 && x.args.size() != 0))
			throw Exception("In 'ecl_prop_transform' found an element with inconsistent 'flag' and 'args'");
		if (x.wght_flag != 0 && x.wght_flag != 1 && x.wght_flag != 2)
			throw Exception("wght_flag = 0, 1, or 2 is expected in 'ecl_prop_transform' elements");
		if (x.wght_flag == 2 && x.wght_prop == "")
			throw Exception("In 'ecl_prop_transform' found an element with wght_flag == 2 and empty 'wght_prop'");
		if (x.args.size() != x.offsets.size())
			throw Exception("In 'ecl_prop_transform' found an element with inconsistent 'args' and 'offsets' sizes");
		if (x.name == "")
			throw Exception("In 'ecl_prop_transform' found an element with empty name");
	}
}
//--------------------------------------------------------------------------------------------------
int tNavSMRY::find_ind(const std::vector<T_ecl_prop_transform> &V, std::string name)			// returns index in V of element with "name"; -1 if not found
{
	for (size_t i = 0; i < V.size(); i++)
		if (V[i].name == name)
			return i;

	return -1;
}
//--------------------------------------------------------------------------------------------------
void tNavSMRY::read_meta_block(const std::string header, std::vector<std::string> &Str, std::vector<int> &Int)		// reads an open 'file_meta' filling arrays 'Str', 'Int'
{							// from lines "Str[i] = Int[i]", starting from header 'header', to the next found header "[...]", saving the latter as 'last_header' (in case of EOF, "" is saved)
	if (file_meta == NULL)
		throw Exception("file_meta == NULL in tNavSMRY::read_meta_block");

	bool fst_found = (header == last_header);
	bool last_found = false;

	Str.clear();
	Int.clear();

	while (!last_found)
	{
		std::string line;
		getline(*file_meta, line);
		line = Trim(line, " \t\r\n");

		if (!fst_found)					// look for the first header
		{
			if (line == header)
				fst_found = true;
		}
		else if (is_header(line))		// check if last header is found
		{
			last_found = true;
			last_header = line;
		}
		else if (line.size() > 0)		// fill the arrays, skip empty lines
		{
			std::vector<std::string> parsed;
			tokenize(line, parsed, "=\r\n", true);
			if (parsed.size() != 2)
				throw Exception("Expected format XXX = Y, but found '" + line + "'");
			Str.push_back(Trim(parsed[0], " \t"));
			Int.push_back(StoL(Trim(parsed[1], " \t")));
		}

		if (file_meta->eof())
		{
			last_found = true;
			last_header = "";
		}
	}
}
//--------------------------------------------------------------------------------------------------
bool tNavSMRY::is_header(std::string s)			// returns true if 's' is a header of format "[...]"
{
	return (s.size() >= 2 && s[0] == '[' && *--s.end() == ']');
}
//--------------------------------------------------------------------------------------------------
void tNavSMRY::open_meta(std::string fname)		// opens 'file_meta'
{
	close_meta();
	file_meta = new std::ifstream;

	file_meta->exceptions(std::ios_base::badbit);
	file_meta->open(fname.c_str());
	if (file_meta->fail())
		throw Exception("Невозможно открыть файл (для чтения) " + fname, "Cannot open file (for reading) " + fname);
}
//--------------------------------------------------------------------------------------------------
void tNavSMRY::close_meta()						// closes 'file_meta'
{
	if (file_meta != NULL && file_meta->is_open())
		file_meta->close();
	delete file_meta;
	last_header = "";
	file_meta = NULL;
}
//--------------------------------------------------------------------------------------------------
std::string tNavSMRY::get_fname_meta(std::string modname) const		// form filename "...well.meta"
{
	return "RESULTS/" + modname + "_well.meta";
}
//--------------------------------------------------------------------------------------------------
std::string tNavSMRY::get_fname_res(std::string modname) const		// form filename "...well.res"
{
	return "RESULTS/" + modname + "_well.res";
}
//--------------------------------------------------------------------------------------------------
void tNavSMRY::read_meta(std::string modname)	// reads "...well.meta"; fills 'vecs', 'dates', 'ind_dates', 'cumul_days', 'ind_obj', 'ecl_prop_ind', 'ecl_prop_transform', 'sec_objects', 'prop_N', 'obj_N'
{
	std::vector<std::string> name_prop, name_dates, name_obj;
	std::vector<int> ind_prop;

	open_meta(get_fname_meta(modname));
	read_meta_block("[properties]", name_prop, ind_prop);
	read_meta_block("[timesteps]", name_dates, ind_dates);
	read_meta_block("[objects]", name_obj, ind_obj);
	close_meta();

	prop_N = ind_prop.size();
	obj_N = ind_obj.size();						// primary objects
	const int fullobj_N = obj_N + sec_obj_N;	// all objects

	// 1) handle primary properties
	std::map<std::string, int> tnav_prop_map;			// <WELL_CALC_BHP, 1> - in fact, the list from [properties], without -1's
	for (int i = 0; i < prop_N; i++)
		if (ind_prop[i] != -1)
			tnav_prop_map[name_prop[i]] = ind_prop[i];	// only take entries with informative (!= -1) indices

	ecl_prop_transform.clear();							// <WBHP,...>
	ecl_prop_ind.clear();								// <1>
	ecl_prop_transform.reserve(ecl_prop_transform_full.size());
	ecl_prop_ind.reserve(ecl_prop_transform_full.size());
	for (const auto &p : ecl_tnav)						// p.first - Eclipse name, p.second - tNav name
		if (tnav_prop_map.count(p.second) > 0)			// take eclipse vector p.first and its index
		{
			int c = find_ind(ecl_prop_transform_full, p.first);
			if (c == -1)
				throw Exception(p.first + " not found in 'ecl_prop_transform_full'");
			ecl_prop_transform.push_back(ecl_prop_transform_full[c]);
			ecl_prop_ind.push_back(tnav_prop_map[p.second]);
		}

	std::vector<size_t> perm_ind = SortPermutation(ecl_prop_ind.begin(), ecl_prop_ind.end());
	ecl_prop_transform = Reorder(ecl_prop_transform, perm_ind);
	ecl_prop_ind = Reorder(ecl_prop_ind, perm_ind);

	// 1a) add secondary properties
	for (size_t i = 0; i < ecl_prop_transform_full.size(); i++)
	{
		T_ecl_prop_transform aux = ecl_prop_transform_full[i];
		if (aux.flag == 1 && find_ind(ecl_prop_transform, aux.name) == -1)		// consider only some new (not yet added) property with flag == 1
		{
			aux.offsets = std::vector<int>(aux.args.size(), 0);
			bool all_args_found = true;
			for (size_t j = 0; j < aux.args.size(); j++)						// check if the required args are available
			{
				int c = find_ind(ecl_prop_transform, aux.args[j]);
				if (c == -1)
				{
					all_args_found = false;
					break;			// go to #'add the secondary property', which will be skipped
				}
				else
					aux.offsets[j] = (c - (int)ecl_prop_transform.size())*fullobj_N;		// note the changing size of ecl_prop_transform[]
			}
			if (all_args_found)		// add the secondary property
			{
				ecl_prop_transform.push_back(aux);
				ecl_prop_ind.push_back(-2);								// secondary properties are marked by index -2
			}
		}
	}
	ecl_prop_transform_check();

	// 1b) fill 'wght_offset' for all properties in 'ecl_prop_transform'
	for (int i = 0; i < (int)ecl_prop_transform.size(); i++)
		if (ecl_prop_transform[i].wght_flag == 2)
		{
			const int k0 = find_ind(ecl_prop_transform, ecl_prop_transform[i].wght_prop);
			if (k0 == -1)
				throw Exception("Property '" + ecl_prop_transform[i].wght_prop + "' specified for weighting is not found in 'ecl_prop_transform' in tNavSMRY::read_meta");
			ecl_prop_transform[i].wght_offset = (k0 - i)*fullobj_N;
		}

	// 2) handle wells
	for (int i = 0; i < obj_N; i++)
	{
		name_obj[i] = Trim(name_obj[i], "'");
		if (ind_obj[i] == -1)
			throw Exception("Found index '-1' in [objects] in *.meta");
		if (i > 0 && ind_obj[i] <= ind_obj[i-1])
			throw Exception("Indices in [objects] should be in increasing order in *.meta");
	}

	// 2a) fill the secondary objects
	sec_objects.clear();
	sec_objects.reserve(sec_obj_N);
	for (size_t i = 0; i < sec_objects_raw.size(); i++)
		sec_objects.push_back(T_sec_obj(i, sec_objects_raw[i].first, sec_objects_raw[i].second, name_obj));

	sec_objects.push_back(T_sec_obj(sec_objects_raw.size(), obj_N));		// 'FIELD'

	// 2b) check the objects names are not duplicating
	std::vector<std::string> all_obj_names = name_obj;
	all_obj_names.reserve(name_obj.size() + sec_objects.size());
	for (const auto &x : sec_objects)
		all_obj_names.push_back(x.name);

	std::string dup;
	if (FindDuplicate(all_obj_names, dup))
	{
		std::string mfile = get_fname_meta(modname);
		char msgrus[BUFFSIZE], msgeng[BUFFSIZE];
		sprintf(msgrus, "При чтении файла тНавигатора %.200s найдены имена скважин, совпадающие с именами групп из кл. слова GROUPS: '%.100s'", mfile.c_str(), dup.c_str());
		sprintf(msgeng, "While reading tNavigator file %.200s, found well names which duplicate the group names from GROUPS keyword: '%.100s'", mfile.c_str(), dup.c_str());

		std::cout << "DEBUG name_obj\t" << HMMPI::ToString(name_obj, "%s", "\t");	//DEBUG
		std::cout << "DEBUG all_obj_names\t" << HMMPI::ToString(all_obj_names, "%s", "\t");	//DEBUG

		throw Exception(msgrus, msgeng);
	}

	// 3) fill the vectors pairs
	size_t props_found = ecl_prop_ind.size();
	vecs.clear();
	vecs.resize(props_found * fullobj_N);
	for (size_t i = 0; i < props_found; i++)
	{
		for (int j = 0; j < obj_N; j++)				// primary objects
			vecs[i*fullobj_N + j] = pair(name_obj[j], ecl_prop_transform[i].name);		// "property-major" order: <W1, P1>, <W2, P1>, ... <Wn, P1>; <W1, P2>,...
		for (int j = obj_N; j < fullobj_N; j++)		// secondary objects
			vecs[i*fullobj_N + j] = pair(sec_objects[j-obj_N].name, ecl_prop_transform[i].name);
	}

	// 4) handle timesteps
	dates.clear();
	dates.resize(name_dates.size());
	for (size_t i = 0; i < name_dates.size(); i++)
	{
		if (ind_dates[i] == -1)
			throw Exception("Found index '-1' in [timesteps] in *.meta");
		if (i > 0 && ind_dates[i] <= ind_dates[i-1])
			throw Exception("Indices in [timesteps] should be in increasing order in *.meta");

		try
		{
			dates[i] = Date(name_dates[i]);
		}
		catch (const Exception &e)
		{
			throw Exception((std::string)e.what() + " (tNavSMRY::read_meta)");
		}
	}

	// 4a) fill cumul_days[]
	if (start > dates[0])
	{
		char msgrus[BUFFSIZE], msgeng[BUFFSIZE];
		sprintf(msgrus, "Начальная дата STARTDATE (%s) идет после первой даты в *.meta (%s)", start.ToString().c_str(), dates[0].ToString().c_str());
		sprintf(msgeng, "Initial date STARTDATE (%s) is later than the first date in *.meta (%s)", start.ToString().c_str(), dates[0].ToString().c_str());
		throw Exception(msgrus, msgeng);
	}
	cumul_days = start.SubtractFromAll(dates);
}
//--------------------------------------------------------------------------------------------------
void tNavSMRY::read_res(std::string modname)		// reads data from "...well.res"; makes consistency checks and fills 'Data'
{
	std::string fname = get_fname_res(modname);
	FILE *f = fopen(fname.c_str(), "rb");
	if (f != NULL)
	{
		try
		{
			// 1) consistency checks
			int size_num, Nwb, Nprop, Nsteps;
			size_t props_found = ecl_prop_ind.size();

			if (fseek(f, 128, SEEK_SET))
				throw Exception("Error reading binary file *.res [src line " NUM_TO_STR(__LINE__) "]");
			fread(&size_num, sizeof(int), 1, f);
			fread(&Nwb, sizeof(int), 1, f);
			fread(&Nprop, sizeof(int), 1, f);
			fread(&Nsteps, sizeof(int), 1, f);
			if (fseek(f, 128*8, SEEK_SET))											// go to end of header
				throw Exception("Error reading binary file *.res [src line " NUM_TO_STR(__LINE__) "]");

			if (size_num != 8)
				throw Exception("Expected 8-byte numbers in *.res");
			if (Nwb < obj_N)
				throw Exception("Well block size in *.res is smaller than the number of objects (wells) in *.meta");
			if (Nprop > prop_N)														// Nprop does not count "-1" props
				throw Exception("Number of properties in *.res is not consistent with *.meta");
			if (Nsteps != (int)ind_dates.size())
				throw Exception("Number of time steps in *.res is not consistent with *.meta");
			if (ind_obj[obj_N-1] >= Nwb)
				throw Exception(stringFormatArr("ind_obj[last] ({0:%d}) >= Nwb ({1:%d}) in tNavSMRY::read_res", std::vector<int>{ind_obj[obj_N-1], Nwb}));
			if (ind_obj[0] < 0)
				throw Exception("ind_obj[0] < 0 in tNavSMRY::read_res");
			for (const auto &x : ecl_prop_ind)
				if (x >= Nprop)
					throw Exception("ecl_prop_ind[i] >= Nprop in tNavSMRY::read_res");
			if (ecl_prop_ind[0] < 0)
				throw Exception("ecl_prop_ind[0] < 0 in tNavSMRY::read_res");

			// 2) read and fill 'Data'
			const int fullobj_N = obj_N + sec_obj_N;
			const int propsfound_fullobj_N = props_found * fullobj_N;
			int Nprop_Nwb = Nprop * Nwb;
			size_t curr_offset = 0;					// offset in the file w.r.t. current position
			Data = std::vector<double>(ind_dates.size() * propsfound_fullobj_N);
			assert(ind_dates.size() == cumul_days.size());
			assert(sec_objects.size() == (size_t)sec_obj_N);

			for (size_t t = 0; t < ind_dates.size(); t++)
			{
				for (size_t i = 0; i < props_found; i++)		// fill all props
					for (int j = 0; j < obj_N; j++)				// for the primary objects
					{
						if (ecl_prop_transform[i].flag == 0 || ecl_prop_transform[i].flag == -1)	// read number from file
						{
							if (feof(f))
								throw Exception("Unexpected end of file encountered while reading *.res");

							size_t new_offset = (ind_dates[t]*Nprop_Nwb + ecl_prop_ind[i]*Nwb + ind_obj[j])*size_num;	// NOTE 'new_offset' are in increasing order, since ind_dates, ecl_prop_ind, ind_obj - increasing,
							if (new_offset < curr_offset)																// 				   ecl_prop_ind[last] - ecl_prop_ind[0] < Nprop, ind_obj[last] - ind_obj[0] < Nwb
								throw Exception(stringFormatArr("Index monotonicity error in tNavSMRY::read_res, err = {8:%d}, t = {0:%d}, i = {1:%d}, j = {2:%d}, ind_dates.size() = {3:%d}, props_found = {4:%d}, obj_N = {5:%d}, Nprop = {6:%d}, Nwb = {7:%d}",
																											std::vector<int>{(int)t, (int)i, j, (int)ind_dates.size(), (int)props_found, obj_N, Nprop, Nwb, int(curr_offset - new_offset)}));
							else if (new_offset > curr_offset)
								if (fseek(f, new_offset - curr_offset, SEEK_CUR))
									throw Exception("Error reading binary file *.res [src line " NUM_TO_STR(__LINE__) "]");

							double d;
							fread(&d, sizeof(d), 1, f);
							if (ecl_prop_transform[i].flag == -1 && d != 0)
								d = -d;

							// a very special case: extract WEFAC from 'WELL_CALC_WORKING_DAYS'
							if (ecl_prop_transform[i].name == "WEFAC")
							{
								double days_interval = 0;
								if (t == 0)
									days_interval = cumul_days[t];
								else
									days_interval = cumul_days[t] - cumul_days[t-1];

								if (fabs(days_interval) < 1e-10)
									d = 1.0;		// too small time step -> take WEFAC = 1.0
								else
									d /= days_interval;
							}

							Data[t*propsfound_fullobj_N + i*fullobj_N + j] = d;
							curr_offset = new_offset + sizeof(d);
						}
						else
							ecl_prop_transform[i].func(Data.data() + t*propsfound_fullobj_N + i*fullobj_N + j);			// recalculate
					}

				for (size_t i = 0; i < props_found; i++)		// then fill all props
					for (int j = obj_N; j < fullobj_N; j++)		// for the secondary objects
					{
						if (ecl_prop_transform[i].flag == 0 || ecl_prop_transform[i].flag == -1)	// primary property
							sec_objects[j-obj_N].func(Data.data() + t*propsfound_fullobj_N + i*fullobj_N + j, ecl_prop_transform[i].wght_flag, ecl_prop_transform[i].wght_offset);
						else																		// secondary property
							ecl_prop_transform[i].func(Data.data() + t*propsfound_fullobj_N + i*fullobj_N + j);			// recalculate
					}
			}
			if (f) fclose(f);
		}
		catch (...)
		{
			if (f) fclose(f);
			throw;
		}
	}
	else
		throw Exception("Невозможно открыть файл (для чтения) " + fname, "Cannot open file (for reading) " + fname);
}
//--------------------------------------------------------------------------------------------------
tNavSMRY::tNavSMRY(std::vector<SecObj> secobj, Date s) : SimSMRY(), last_header(""), file_meta(NULL), sec_objects_raw(std::move(secobj)), prop_N(0), obj_N(0), sec_obj_N(0), start(s)
{
	sec_obj_N = sec_objects_raw.size() + 1;			// '+1' for 'FIELD' group

	// NOTE this "table" establishes the link between Eclipse vector names and tNavigator vector names, expand it as appropriate
	ecl_tnav["WBHP"] = "WELL_CALC_BHP";
	ecl_tnav["WBP9"] = "WELL_WPAVE9_PRESSURE";
	ecl_tnav["WOPR"] = "WELL_CALC_OIL_RATE";
	ecl_tnav["WWPR"] = "WELL_CALC_WATER_RATE";
	ecl_tnav["WGPR"] = "WELL_CALC_GAS_RATE";
	ecl_tnav["WWIR"] = "WELL_CALC_WATER_INJ";
	ecl_tnav["WOPT"] = "WELL_CALC_ACCUM_OIL_RATE";
	ecl_tnav["WWPT"] = "WELL_CALC_ACCUM_WATER_RATE";
	ecl_tnav["WGPT"] = "WELL_CALC_ACCUM_GAS_RATE";
	ecl_tnav["WWIT"] = "WELL_CALC_ACCUM_WATER_INJ";
	ecl_tnav["WEFAC"] = "WELL_CALC_WORKING_DAYS";

	// NOTE this "table" says how the vectors should be processed: e.g. some (WOPR etc) should be negated, the others (WWCT etc) recalculated; expand as appropriate
	// all expected Eclipse vectors should be listed here
	ecl_prop_transform_full.resize(16);
	ecl_prop_transform_full[0] = {"WBHP", 0, [](const int*, double*){}, {}, {}, 0, 0, ""};						// primary props
	ecl_prop_transform_full[1] = {"WBP9", 0, [](const int*, double*){}, {}, {}, 0, 0, ""};
	ecl_prop_transform_full[2] = {"WOPR", -1, [](const int*, double*){}, {}, {}, 2, 0, "WEFAC"};
	ecl_prop_transform_full[3] = {"WWPR", -1, [](const int*, double*){}, {}, {}, 2, 0, "WEFAC"};
	ecl_prop_transform_full[4] = {"WGPR", -1, [](const int*, double*){}, {}, {}, 2, 0, "WEFAC"};
	ecl_prop_transform_full[5] = {"WWIR", 0, [](const int*, double*){}, {}, {}, 2, 0, "WEFAC"};
	ecl_prop_transform_full[6] = {"WOPT", -1, [](const int*, double*){}, {}, {}, 1, 0, ""};
	ecl_prop_transform_full[7] = {"WWPT", -1, [](const int*, double*){}, {}, {}, 1, 0, ""};
	ecl_prop_transform_full[8] = {"WGPT", -1, [](const int*, double*){}, {}, {}, 1, 0, ""};
	ecl_prop_transform_full[9] = {"WWIT", 0, [](const int*, double*){}, {}, {}, 1, 0, ""};
	ecl_prop_transform_full[10] = {"WEFAC", 0, [](const int*, double*){}, {}, {}, 0, 0, ""};

	ecl_prop_transform_full[11] = {"WWCT", 1, T_ecl_prop_transform::f_wct, {}, {"WOPR", "WWPR"}, 0, 0, ""};		// secondary props: group summation will not be applied anyway
	ecl_prop_transform_full[12] = {"WGOR", 1, T_ecl_prop_transform::f_gor, {}, {"WOPR", "WGPR"}, 0, 0, ""};
	ecl_prop_transform_full[13] = {"WLPR", 1, T_ecl_prop_transform::f_sum, {}, {"WOPR", "WWPR"}, 0, 0, ""};
	ecl_prop_transform_full[14] = {"WLPT", 1, T_ecl_prop_transform::f_sum, {}, {"WOPT", "WWPT"}, 0, 0, ""};
	ecl_prop_transform_full[15] = {"WPI", 1, T_ecl_prop_transform::f_wpi, {}, {"WOPR", "WWPR", "WBP9", "WBHP"}, 0, 0, ""};
};
//--------------------------------------------------------------------------------------------------
tNavSMRY::~tNavSMRY()
{
	close_meta();
}
//--------------------------------------------------------------------------------------------------
const tNavSMRY &tNavSMRY::operator=(const tNavSMRY &p)
{
	close_meta();
	SimSMRY::operator=(p);
	ecl_tnav = p.ecl_tnav;
	ind_dates = p.ind_dates;
	cumul_days = p.cumul_days;
	ind_obj = p.ind_obj;
	ecl_prop_ind = p.ecl_prop_ind;
	ecl_prop_transform = p.ecl_prop_transform;
	ecl_prop_transform_full = p.ecl_prop_transform_full;
	sec_objects_raw = p.sec_objects_raw;
	sec_objects = p.sec_objects;
	prop_N = p.prop_N;
	obj_N = p.obj_N;
	sec_obj_N = p.sec_obj_N;
	start = p.start;

	return *this;
}
//--------------------------------------------------------------------------------------------------
SimSMRY *tNavSMRY::Copy() const
{
	return new tNavSMRY(*this);
}
//--------------------------------------------------------------------------------------------------
void tNavSMRY::ReadFiles(std::string modname)
{
	read_meta(modname);
	read_res(modname);
	mod = modname;
}
//--------------------------------------------------------------------------------------------------
void tNavSMRY::dump_all(std::string fname) const
{
	FILE *f = fopen(fname.c_str(), "w");

	fprintf(f, "tNavSMRY dump\n");
	fprintf(f, "mod: %s\n", mod.c_str());
	fprintf(f, "last_header: '%s'\n", last_header.c_str());

	fprintf(f, "ecl_tnav:\n");
	for (auto x : ecl_tnav)
		fprintf(f, "%s\t%s\n", x.first.c_str(), x.second.c_str());

	fprintf(f, "\nprop_N: %d\nobj_N: %d\nsec_obj_N: %d\n\n", prop_N, obj_N, sec_obj_N);
	fprintf(f, "start date: %s\n\n", start.ToString().c_str());
	fprintf(f, "ind_dates:\t%s\n", HMMPI::ToString(ind_dates, "%d").c_str());
	fprintf(f, "cumul_days:\t%s\n", HMMPI::ToString(cumul_days, "%g").c_str());
	fprintf(f, "ind_obj:\t%s\n", HMMPI::ToString(ind_obj, "%d").c_str());
	fprintf(f, "ecl_prop_ind:\t%s\n", HMMPI::ToString(ecl_prop_ind, "%d").c_str());

	fprintf(f, "ecl_prop_transform (%zu):\n", ecl_prop_transform.size());
	for (const auto &x : ecl_prop_transform)
	{
		fprintf(f, "%s -> flag = %d, wght_flag = %d, wght_offset = %d ('%s')\n", x.name.c_str(), x.flag, x.wght_flag, x.wght_offset, x.wght_prop.c_str());
		for (size_t i = 0; i < x.offsets.size(); i++)
			fprintf(f, "* %s\t%d\n", x.args[i].c_str(), x.offsets[i]);
	}

	fprintf(f, "\nsec_objects_raw (%zu):\n", sec_objects_raw.size());
	for (const auto &x : sec_objects_raw)
		if (x.second.size() > 0)
			fprintf(f, "%s:\t%s", x.first.c_str(), HMMPI::ToString(x.second, "%s").c_str());
		else
			fprintf(f, "%s:\n", x.first.c_str());

	fprintf(f, "\nsec_objects (%zu):\n", sec_objects.size());
	for (const auto &x : sec_objects)
		fprintf(f, "%s", x.ToString().c_str());

	const size_t Nvecs = vecs.size();
	fprintf(f, "\nData:\n");
	if (Data.size() == Nvecs*dates.size())
	{
		for (size_t i = 0; i < dates.size(); i++)
			for (size_t j = 0; j < Nvecs; j++)
			{
				fprintf(f, "%-11.5g", Data[i*Nvecs + j]);
				if (j == Nvecs-1)
					fprintf(f, "\n");
				else
					fprintf(f, "\t");
			}
	}
	else
		fprintf(f, "N/A\n");

	fprintf(f, "\ndates:");
	for (auto x : dates)
		fprintf(f, "\n%s", x.ToString().c_str());

	fprintf(f, "\n\nvecs:");
	for (auto x : vecs)
		fprintf(f, "\t%s_%s", x.first.c_str(), x.second.c_str());

	if (f) fclose(f);
}
//--------------------------------------------------------------------------------------------------
std::string tNavSMRY::dates_file() const		// name of file with dates
{
	return EraseSubstr(get_fname_meta(mod), "./");
}
//--------------------------------------------------------------------------------------------------
std::string tNavSMRY::vecs_file() const			// name of file with vecs
{
	return EraseSubstr(get_fname_meta(mod), "./");
}
//--------------------------------------------------------------------------------------------------
std::string tNavSMRY::data_file() const			// name of file with data
{
	return EraseSubstr(get_fname_res(mod), "./");
}
//--------------------------------------------------------------------------------------------------
// tNavSMRY22
//--------------------------------------------------------------------------------------------------
std::string tNavSMRY22::get_fname_meta(std::string modname) const		// form filename "...well.meta"
{
	return "RESULTS/" + modname + "/well.meta";
}
//--------------------------------------------------------------------------------------------------
std::string tNavSMRY22::get_fname_res(std::string modname) const		// form filename "...well.res"
{
	return "RESULTS/" + modname + "/well.res";
}
//--------------------------------------------------------------------------------------------------
const tNavSMRY22 &tNavSMRY22::operator=(const tNavSMRY22 &p)
{
	tNavSMRY::operator=(p);
	return *this;
}
//--------------------------------------------------------------------------------------------------
SimSMRY *tNavSMRY22::Copy() const
{
	return new tNavSMRY22(*this);
}
//--------------------------------------------------------------------------------------------------
// SimProxyFile
//--------------------------------------------------------------------------------------------------
void SimProxyFile::data_stamp::stamp_file(FILE *fd, int ver)	// write data stamp to a binary file (marking the file validity); besides, stamp[2] will be set to 'ver'
{
	assert(ver == 1 || ver == 2);
	stamp[2] = ver;
	fwrite(stamp, sizeof(double), 4, fd);
}
//--------------------------------------------------------------------------------------------------
int SimProxyFile::data_stamp::check_stamp(FILE *fd)		// read data stamp and check its validity; returns: 0 - invalid stamp, 1 - empty file,
{														// 2 - valid non-empty file (for which stamp[2] will hold the version: 1 or 2)
	double buff[4];
	size_t c = fread(buff, sizeof(double), 4, fd);		// c = number of objects read successfully
	if (c == 0)						// 0 elements read -> empty file
		return 1;
	else if (c < 4)					// couldn't read the whole stamp -> invalid file
		return 0;
	else if (buff[0] == stamp[0] &&
			 buff[1] == stamp[1] &&
			(buff[2] == 1 || buff[2] == 2) &&
			 buff[3] == stamp[3])	// stamp is valid, version is valid
	{
		stamp[2] = buff[2];
		return 2;
	}
	else							// stamp is not valid
		return 0;
}
//--------------------------------------------------------------------------------------------------
std::string SimProxyFile::msg_contents() const			// message reporting what is stored; should be called on all ranks
{
	std::string msg;
	std::vector<int> bs = block_starts();
	const int ver = (int)D_stamp.get_stamp2();
	assert(ver == 1 || ver == 2);
	if (Rank == 0) {
		char buffrus[BUFFSIZE], buffeng[BUFFSIZE];
		const char *format = (ver == 1 ? "STANDARD" : "COMPRESSED");
		sprintf(buffrus, "Формат файла на входе: %.20s\nЗагружено %zu модел(ей), количество параметров %zu\n", format, block_ind.size(), par_names.size());
		sprintf(buffeng, "Input file format: %.20s\nLoaded %zu model(s), number of parameters %zu\n", format, block_ind.size(), par_names.size());
		msg = MessageRE(buffrus, buffeng);

		assert(data_dates.size() == data_vecs.size());
		assert(data_dates.size()+1 == bs.size());

		for (size_t i = 0; i < data_dates.size(); i++) {
			sprintf(buffrus, "* model-block %zu: количество моделей %d, количество дат %zu, количество векторов %zu\n", i, bs[i+1] - bs[i], data_dates[i].size(), data_vecs[i].size());
			sprintf(buffeng, "* model-block %zu: number of models %d, number of dates %zu, number of vectors %zu\n", i, bs[i+1] - bs[i], data_dates[i].size(), data_vecs[i].size());
			msg += MessageRE(buffrus, buffeng);
		}
	}
	return msg;
}
//--------------------------------------------------------------------------------------------------
std::vector<std::vector<double>> SimProxyFile::extract_proxy_vals(const std::vector<Date> &dates, const std::vector<SimSMRY::pair> &vecs, const std::vector<double> &all_sigmas, const std::vector<int> &bs) const
{
	int err = 0;		// for exceptions synchronising
	std::vector<std::vector<double>> res;
	try
	{
		if (Rank == 0)
		{
			if (all_sigmas.size() != dates.size()*vecs.size())
				throw Exception("all_sigmas.size() != dates.size()*vecs.size() in SimProxyFile::extract_proxy_vals");

			int Nblock = data_dates.size();			// number of blocks
			if (Nblock+1 != (int)bs.size())
				throw Exception("Nblock+1 != bs.size() in SimProxyFile::extract_proxy_vals");

			int Np = data.size();
			if (Np == 0)
				throw Exception("Попытка создать прокси-модель из пустого объекта ECLSMRY", "Attempt to create proxy model from empty ECLSMRY object");

			if (Np != bs[Nblock])
				throw Exception("Np != bs[Nblock] in SimProxyFile::extract_proxy_vals");

			int len = 0;							// length of the resulting vector
			for (double x : all_sigmas)
				if (x != 0)
					len++;

			res = std::vector<std::vector<double>>(len);
			datapoint_block = std::vector<int>(len);
			datapoint_modcount = std::vector<int>(len);

			int c = 0;
			for (size_t v = 0; v < vecs.size(); v++)
				for (size_t d = 0; d < dates.size(); d++)
					if (all_sigmas[v*dates.size() + d] != 0)		// go through all requested data points
					{
						if (c >= len)
							throw Exception("c >= len in SimProxyFile::extract_proxy_vals");

						bool first_encounter = true;				// this flag is set for each new data point
						for (int b = 0; b < Nblock; b++)
						{
							int dateind = FindBinary(data_dates[b].begin(), data_dates[b].end(), dates[d]) - data_dates[b].begin();		// dateind, vecind may depend on block number "b"
							int vecind = FindBinary(data_vecs[b].begin(), data_vecs[b].end(), vecs[v]) - data_vecs[b].begin();
							if (dateind != (int)data_dates[b].size() && vecind != (int)data_vecs[b].size())		// data point is found in block "b"
							{
								if (first_encounter)
								{
									datapoint_block[c] = b;
									datapoint_modcount[c] = bs[Nblock] - bs[b];
									res[c] = std::vector<double>(datapoint_modcount[c]);
									first_encounter = false;
								}

								for (int i = bs[b]; i < bs[b+1]; i++)
									res[c][i - bs[datapoint_block[c]]] = data[i][dateind*data_vecs[b].size() + vecind];
							}
						}
						if (first_encounter)
							throw Exception((std::string)"Точка данных " + dates[d].ToString() + " " + vecs[v].first + " " + vecs[v].second + " не найдена ни в одной записи в ECLSMRY",
											(std::string)"Data point " + dates[d].ToString() + " " + vecs[v].first + " " + vecs[v].second + " was not found within ECLSMRY entries");
						c++;
					}
		}
	}
	catch (...)
	{
		err = 1;
		MPI_Bcast(&err, 1, MPI_INT, 0, comm);	// comm-RANKS-0
		throw;
	}

	MPI_Bcast(&err, 1, MPI_INT, 0, comm);		// comm-RANKS-!0 and (comm-RANKS-0 with no error)
	if (err)
		throw Exception();						// only comm-RANKS-!0 can get here

	MPI_Barrier(comm);
	Bcast_vector(datapoint_block, 0, comm);
	Bcast_vector(datapoint_modcount, 0, comm);
	return res;
}
//--------------------------------------------------------------------------------------------------
std::vector<int> SimProxyFile::block_starts() const
{
	std::vector<int> res;
	if (Rank == 0)						// fill on Rank-0
	{
		assert(data_dates.size() == data_vecs.size());

		int Np = block_ind.size(), Nblock = data_dates.size(), bs = 0;
		res.resize(Nblock+1);			// indices in [0, Np) where each block starts; the last element is Np
		for (int i = 0; i < Np; i++)
			if (block_ind[i] == bs)
			{
				if (bs >= Nblock)
					throw Exception("bs >= Nblock in SimProxyFile::block_starts");
				res[bs++] = i;
			}
		res[Nblock] = Np;
	}

	MPI_Barrier(comm);
	Bcast_vector(res, 0, comm);		// sync to comm
	return res;
}
//--------------------------------------------------------------------------------------------------
void SimProxyFile::recalc_blocks(int last_block)	// for 'block_ind' with some elements marked as '-1', removes them, recalculates 'block_ind',
{													// and clears the elements of 'data_dates', 'data_vecs' where appropriate.
	int b_new = -1;									// 'last_block' is the last block index in the old 'block_ind' as of before setting any '-1' marks.
	int b_last = -1;

	// 1. Find the blocks to be erased
	std::vector<int> unique_olds = Unique(block_ind);
	std::vector<size_t> erase_list;
	for (int b = 0; b <= last_block; b++)	// b : all the old blocks
		if (std::find(unique_olds.begin(), unique_olds.end(), b) == unique_olds.end()) erase_list.push_back(b);		// block 'b' not found -> erase it

	// 2. Recalculate block_ind
	std::vector<int> block_ind_new;
	block_ind_new.reserve(block_ind.size());
	for (size_t i = 0; i < block_ind.size(); i++) {
		if (block_ind[i] != -1) {
			assert(block_ind[i] >= b_last);
			if (block_ind[i] != b_last) b_new++;
			block_ind_new.push_back(b_new);
			b_last = block_ind[i];
		}
	}
	block_ind = std::move(block_ind_new);

	// 3. Clear 'data_dates', 'data_vecs'
	for (size_t i = 0; i < erase_list.size(); i++) {
		assert(erase_list[i] < data_dates.size());
		data_dates[erase_list[i]].clear();
		data_vecs[erase_list[i]].clear();
	}
}
//--------------------------------------------------------------------------------------------------
void SimProxyFile::print_debug_1(const std::string &msg) const	// print some stuff for debugging adding the models
{
	if (Rank == 0) {
		printf("%s\nblock_ind (len = %zu): %s", msg.c_str(), block_ind.size(), ToString(block_ind, "%d", " ").c_str());
		printf("Arrays lengths\tdata_dates %zu\tdata_vecs %zu\tdata %zu\tparams %zu\n", data_dates.size(), data_vecs.size(), data.size(), params.size());
		printf("Xmin = %g\tXavg = %g\n", Xmin, Xavg);
	}
}
//--------------------------------------------------------------------------------------------------
void SimProxyFile::transpose_data(bool rows_are_dates) const	// transpose each data[i] in place, assuming row-major storage, with Nrows from 'data_dates' (if rows_are_dates),
{																// or from 'data_vecs'
	assert(block_ind.size() == data.size());
	for (size_t i = 0; i < data.size(); i++) {
		const size_t bl = block_ind[i];				// bl = block for model 'i'
		assert (bl < data_dates.size() && bl < data_vecs.size());
		if (rows_are_dates) data[i] = VecTranspose(data[i], data_dates[bl].size());
		else                data[i] = VecTranspose(data[i], data_vecs[bl].size());
	}
}
//--------------------------------------------------------------------------------------------------
SimProxyFile::~SimProxyFile()
{
	delete Ecl;
	delete SimProxy;
	delete BDC;
}
//--------------------------------------------------------------------------------------------------
std::string SimProxyFile::AddModel(const std::vector<std::string> &pname, const std::vector<double> &pval, const std::vector<std::string> &backval, const std::vector<std::string> &func, const SimSMRY *smry)
{
	datapoint_block.clear();
	datapoint_modcount.clear();

	int err = 0;		// for exceptions synchronising
	try {
		if (Rank == 0) {
			assert(block_ind.size() == params.size());
			assert(data_dates.size() == data_vecs.size());
			assert(block_ind.size() == data.size());

			// 1. handle parameters
			if (pname.size() != pval.size() || pname.size() != backval.size() || pname.size() != func.size())
				throw Exception("pname.size() != pval.size() || pname.size() != backval.size() || pname.size() != func.size()");
			if (pname.size() == 0)
				throw Exception("Попытка добавить модель с пустым списком параметров", "Attempt to add a model with empty parameters list");

			const std::vector<size_t> perm_param = SortPermutation(pname.begin(), pname.end());		// permutation indices
			const std::vector<std::string> pname_sorted = Reorder(pname, perm_param);				// sorted array of new params names
			const std::vector<double> pval_sorted = Reorder(pval, perm_param);						// corresp. params vals
			const std::vector<std::string> bv_sorted = Reorder(backval, perm_param);				// corresp. backvals
			const std::vector<std::string> func_sorted = Reorder(func, perm_param);					// corresp. funcs

			if (!std::includes(pname_sorted.begin(), pname_sorted.end(), par_names.begin(), par_names.end()))
				throw Exception("Попытка добавить модель с параметрами, которые не являются надмножеством параметров сохраненных ранее моделей",
								"Attempt to add a model with parameters which are not a superset of parameters of previously saved models");
			std::string dup;
			if (FindDuplicate(pname_sorted, dup))
				throw Exception(stringFormatArr("Добавление модели с повторяющимся параметром '{0:%s}' в SimProxyFile::AddModel()",
												"Adding a model with duplicate parameter '{0:%s}' in SimProxyFile::AddModel()", dup));

			std::vector<size_t> par_subind = GetSubvecIndSorted(par_names, pname_sorted);	// resulting indices may contain "-1", as par_names <= pname_sorted

			const size_t dim = pval_sorted.size();											// new dimension
			assert(par_subind.size() == dim);
			std::vector<std::vector<double>> params_new(params.size() + 1, std::vector<double>(dim));	// new array of parameter values: added one model

			std::vector<int> count_err(dim, 0);							// count the "non-positive" errors for "EXP" new params
			int count_err_total = 0;
			for (size_t i = 0; i < params.size(); i++) {				// transfer the previous params values, i = [0, Np) - model number
				for (size_t j = 0; j < dim; j++) {						// go through new params
					if (par_subind[j] != (size_t)-1)
						params_new[i][j] = params[i][par_subind[j]];	// take from old param value
					else {												// new param: engage backval
						bool is_num = false;
						double val = StoD(bv_sorted[j], is_num);		// is_num = true if the whole string is a number
						if (!is_num) {									// backval is not a number ... keep processing it
							const auto it = FindBinary(par_names.begin(), par_names.end(), bv_sorted[j]);	// binary search in a SORTED range [first, last); returns "last" if not found
							const size_t ind = it - par_names.begin();
							if (ind < par_names.size()) val = params[i][ind];		// backval is a single parameter
							else {													// backval is an expression (if not, then it has no sense)
								TagValMap tmap(par_names, params[i]);
								int count_dummy = 0;
								std::set<std::string> set_dummy;
								std::string comment = stringFormatArr(" [см. backval для параметра '{0:%s}' в PARAMETERS]",
																	  " [see backval for parameter '{0:%s}' in PARAMETERS]", pname_sorted[j]);
								std::string msg_params = MessageRE(", записанных в ECLSMRY", " stored in ECLSMRY");

								// bv_sorted[j] is the expression
								std::vector<std::string> infix = StringToInfix(bv_sorted[j]); // bv_sorted[j] is the expression
								std::vector<const ValBase*> postfix = InfixToPostfix(infix, tmap, count_dummy, set_dummy, bv_sorted[j], comment, msg_params);
								const ValBase *res_postfix = CalcPostfix(postfix, bv_sorted[j], comment);
								const Val<double> *res_double = dynamic_cast<const Val<double>*>(res_postfix);
								if (res_double == nullptr)
									throw Exception(stringFormatArr("Выражение '{0:%s}' при вычислении не дает число (double)",
																	"Evaluating the expression '{0:%s}' does not result in a number (double)", bv_sorted[j]) + comment);
								val = res_double->get_val();
								delete res_postfix;
							}
						}
						assert (func_sorted[j] == "EXP" || func_sorted[j] == "LIN");
						if (func_sorted[j] == "EXP" && val <= 0) {		// check "EXP"
							count_err[j] += 1;
							count_err_total += 1;
						}
						params_new[i][j] = val;
					}
				}
			}
			if (count_err_total) {
				std::string errmsg = "";
				char msgrus[BUFFSIZE*2];
				char msgeng[BUFFSIZE*2];
				for (size_t j = 0; j < dim; j++)
					if (count_err[j] > 0) {
						sprintf(msgrus, "При применении backval = '%.500s' для параметра '%.100s' (тип EXP) получились значения <= 0 для %d модели(ей) из ECLSMRY", bv_sorted[j].c_str(), pname_sorted[j].c_str(), count_err[j]);
						sprintf(msgeng, "When engaging backval = '%.500s' for parameter '%.100s' (EXP type), values <= 0 were obtained for %d model(s) in ECLSMRY", bv_sorted[j].c_str(), pname_sorted[j].c_str(), count_err[j]);
						if (errmsg.size() > 0) errmsg += "\n";
						errmsg += MessageRE(msgrus, msgeng);
					}
				throw Exception(errmsg);
			}

			params_new[params.size()] = std::move(pval_sorted);			// transfer the params values from the model which we are adding

			par_names = std::move(pname_sorted);
			params = std::move(params_new);

			// 2. handle data values
			if (smry->dates.size() == 0)
				throw Exception("Попытка добавить модель с пустым списком дат", "Attempt to add a model with empty dates list");
			if (smry->vecs.size() == 0)
				throw Exception("Попытка добавить модель с пустым списком векторов", "Attempt to add a model with empty vectors list");

			std::vector<size_t> perm_dates = SortPermutation(smry->dates.begin(), smry->dates.end());
			std::vector<size_t> perm_vecs = SortPermutation(smry->vecs.begin(), smry->vecs.end());
			std::vector<Date> dates_sorted = Reorder(smry->dates, perm_dates);
			std::vector<SimSMRY::pair> vecs_sorted = Reorder(smry->vecs, perm_vecs);
			std::vector<double> data_sorted = Reorder(smry->Data, smry->dates.size(), smry->vecs.size(), perm_dates, perm_vecs);

			data.push_back(std::move(data_sorted));						// add data

			if (!(data_dates.size() > 0 && dates_sorted == *--data_dates.end() && vecs_sorted == *--data_vecs.end()))		// create a new data block
			{
				if (data_dates.size() > 0 && !std::includes(dates_sorted.begin(), dates_sorted.end(), (*--data_dates.end()).begin(), (*--data_dates.end()).end()))
					throw Exception("Попытка добавить модель с датами, которые не являются надмножеством дат сохраненных ранее моделей",
									"Attempt to add a model with dates which are not a superset of dates of previously saved models");

				if (data_vecs.size() > 0 && !std::includes(vecs_sorted.begin(), vecs_sorted.end(), (*--data_vecs.end()).begin(), (*--data_vecs.end()).end()))
					throw Exception("Попытка добавить модель с векторами, которые не являются надмножеством векторов сохраненных ранее моделей",
									"Attempt to add a model with vectors which are not a superset of vectors of previously saved models");

				Date dup0;
				if (FindDuplicate(dates_sorted, dup0))
				{
					char msg[BUFFSIZE];
					sprintf(msg, "Adding a model with duplicate date %s in SimProxyFile::AddModel", dup0.ToString().c_str());
					throw Exception(msg);
				}

				SimSMRY::pair dup1;
				if (FindDuplicate(vecs_sorted, dup1))
					throw Exception((std::string)"Adding a model with duplicate vector " + dup1.first + " " + dup1.second + " in SimProxyFile::AddModel");

				data_dates.push_back(dates_sorted);
				data_vecs.push_back(vecs_sorted);
			}

			block_ind.push_back(int(data_dates.size()) - 1);			// add block index for the model
		}
	}
	catch (...)
	{
		err = 1;
		MPI_Bcast(&err, 1, MPI_INT, 0, comm);	// comm-RANKS-0
		throw;
	}

	MPI_Bcast(&err, 1, MPI_INT, 0, comm);		// comm-RANKS-!0 and (comm-RANKS-0 with no error)
	if (err)
		throw Exception();						// only comm-RANKS-!0 can get here

	MPI_Barrier(comm);

	// Check if the added model is approximately repeating early models, and delete these early models
	std::string msg = "";
	Xmin = Xavg = 0;
	if (Rank == 0 && block_ind.size() > 1) {
		const int last_block = *--block_ind.end();
		const size_t Np = params.size();		// Number of models (including the just added one)
		std::vector<std::vector<double>> params_int(Np);
		std::vector<size_t> ind = GetSubvecIndSorted(par_names, pname);
		assert(pname.size() == ind.size() && pname.size() == par_names.size());
		for (size_t i = 0; i < Np; i++) {
			params_int[i] = Reorder(params[i], ind);
			params_int[i] = par_tran->ExternalToInternal(params_int[i]);	// convert to internal params
		}

		Mat dist = KrigStart::DistMatr(params_int, 0, Np-1, Np-1, Np);		// (Np-1)*1 distance matrix

		Xmin = std::numeric_limits<double>::max();
		Xavg = 0;
		int count = 0;
		for (size_t i = 0; i < dist.ICount(); i++) {	// i = [0, Np-1), i.e. all previous models
			if (dist(i, 0) < Xtol) {	// the just-added model is close to model 'i': mark model 'i' for deletion
				char msgrus[BUFFSIZE];
				char msgeng[BUFFSIZE];
				sprintf(msgrus, "Существующая модель-%zu из model-block-%d удаляется из ECLSMRY из-за близости к новой модели (dist = %g)\n", i, block_ind[i], dist(i, 0));
				sprintf(msgeng, "Existing model-%zu from model-block-%d is removed from ECLSMRY due to proximity to the new model (dist = %g)\n", i, block_ind[i], dist(i, 0));
				msg += MessageRE(msgrus, msgeng);

				params[i].clear();
				data[i].clear();
				block_ind[i] = -1;
			} else {	// model 'i' is preserved: calculate stats
				if (dist(i, 0) < Xmin) Xmin = dist(i, 0);
				Xavg += dist(i, 0);
				count++;
			}
		}
		if (count > 0) Xavg /= count;
		else Xmin = Xavg = 0;

		if (msg != "") {
			recalc_blocks(last_block);

			params = SqueezeVec(std::move(params));		// remove the possible gaps in the outer vectors
			data_dates = SqueezeVec(std::move(data_dates));
			data_vecs = SqueezeVec(std::move(data_vecs));
			data = SqueezeVec(std::move(data));
		}
	}
	return msg;
}
//--------------------------------------------------------------------------------------------------
void SimProxyFile::PopModel()
{
	datapoint_block.clear();
	datapoint_modcount.clear();

	int ierr = 0;
	if (Rank == 0 && block_ind.size() == 0)
		ierr = 1;
	MPI_Bcast(&ierr, 1, MPI_INT, 0, comm);		// ierr is sync
	if (ierr)
		throw Exception("Попытка удалить модель из пустого ECLSMRY", "Attempt to delete a model from the empty ECLSMRY");

	if (Rank == 0)
	{
		if (*--block_ind.end() != *----block_ind.end())		// the last model makes a new block
		{
			data_dates.pop_back();
			data_vecs.pop_back();
		}
		block_ind.pop_back();
		params.pop_back();
		data.pop_back();
	}
}
//--------------------------------------------------------------------------------------------------
void SimProxyFile::SaveToBinary(const std::string &fname, int ver) const	// saves to binary file (only comm-RANKS-0 is working, but call it on all ranks)
{
	assert(data_dates.size() == data_vecs.size());

	int par_names_size = par_names.size();
	int params_size = params.size();
	int dates0_size = 0, vecs0_size = 0;
	if (data_dates.size() > 0)
	{
		dates0_size = data_dates[0].size();		// number of dates in block-0
		vecs0_size = data_vecs[0].size();		// number of vecs in block-0
	}

	MPI_Barrier(comm);
	MPI_Bcast(&par_names_size, 1, MPI_INT, 0, comm);	// sync
	MPI_Bcast(&params_size, 1, MPI_INT, 0, comm);
	MPI_Bcast(&dates0_size, 1, MPI_INT, 0, comm);
	MPI_Bcast(&vecs0_size, 1, MPI_INT, 0, comm);

	if (par_names_size == 0)
		throw Exception("Попытка сохранить ECLSMRY с пустым списком параметров", "Attempt to save ECLSMRY with empty parameters list");
	if (params_size == 0)
		throw Exception("Попытка сохранить ECLSMRY с пустым списком моделей", "Attempt to save ECLSMRY with empty models list");
	if (dates0_size == 0)
		throw Exception("Попытка сохранить ECLSMRY с моделью, имеющей пустой список дат", "Attempt to save ECLSMRY with a model having empty dates list");
	if (vecs0_size == 0)
		throw Exception("Попытка сохранить ECLSMRY с моделью, имеющей пустой список векторов", "Attempt to save ECLSMRY with a model having empty vectors list");

	if (Rank == 0) {
		FILE *fd = fopen(fname.c_str(), "wb");
		if (fd == NULL)
			throw Exception((std::string)MessageRE("Невозможно открыть файл для записи '", "Cannot open file for writing '") + fname + "'");	// this exception and other exceptions on Rank-0 are not sync, but they are not likely to happen

		try
		{
			assert(ver == 1 || ver == 2);
			if (ver == 2) transpose_data(true);		// transpose forward
			D_stamp.stamp_file(fd, ver);
			write_bin(fd, block_ind, ver);
			write_bin(fd, par_names);
			write_bin(fd, params);
			write_bin(fd, data_dates);
			write_bin(fd, data_vecs);
			write_bin(fd, data, ver);
			if (ver == 2) transpose_data(false);	// transpose backward
		}
		catch (...)
		{
			if (fd) fclose(fd);
			throw;
		}
		if (fd) fclose(fd);
	}
}
//--------------------------------------------------------------------------------------------------
std::string SimProxyFile::LoadFromBinary(const std::string &fname)	// loads from binary file, returns a short message;
{																	// only comm-RANKS-0 is working, but "fname" should be sync on all ranks
	datapoint_block.clear();
	datapoint_modcount.clear();

	FILE *fd = NULL;		// only comm-RANKS-0 will open the file

	RANK0_SYNCERR_BEGIN(comm);
	fd = fopen(fname.c_str(), "rb");
	if (fd == NULL)
		throw Exception((std::string)MessageRE("Невозможно открыть файл для чтения '", "Cannot open file for reading '") + fname + "'");
	RANK0_SYNCERR_END(comm);

	std::string msg = "";
	int check;
	if (Rank == 0) check = D_stamp.check_stamp(fd);		// fills stamp[2] = version

	MPI_Barrier(comm);
	MPI_Bcast(&check, 1, MPI_INT, 0, comm);										// sync the stamp check

	if (check == 1)
		msg = MessageRE("Пустой файл ECLSMRY\n", "Empty ECLSMRY file\n");		// nothing to be read; msg is sync
	else if (check == 0) {
		if (fd) fclose(fd);
		throw Exception(MessageRE((std::string)"Файл " + fname + " не является корректным файлом ECLSMRY",		// a sync exception
								  (std::string)"File " + fname + " is not a correct ECLSMRY file"));
	} else if (Rank == 0) {
		try
		{
			const int ver = (int)D_stamp.get_stamp2();
			assert(ver == 1 || ver == 2);
			read_bin(fd, block_ind, ver);
			read_bin(fd, par_names);
			read_bin(fd, params);
			read_bin(fd, data_dates);
			read_bin(fd, data_vecs);
			read_bin(fd, data, ver);
			if (ver == 2) transpose_data(false);
		}
		catch (...)
		{
			if (fd) fclose(fd);
			throw;					// this exception on Rank-0 is not sync, but it's not likely to happen
		}
	}

	if (fd) fclose(fd);
	if (msg == "")
		msg = msg_contents();		// called on all ranks

	return msg;
}
//--------------------------------------------------------------------------------------------------
void SimProxyFile::SaveToAscii(FILE *fd) const
{
	if (Rank == 0)
	{
		write_ascii(fd, block_ind);
		write_ascii(fd, par_names);
		write_ascii(fd, params);
		write_ascii(fd, data_dates);
		write_ascii(fd, data_vecs);
		write_ascii(fd, data);
	}
}
//--------------------------------------------------------------------------------------------------
PM_SimProxy *SimProxyFile::MakeProxy(const std::vector<Date> &dates, const std::vector<SimSMRY::pair> &vecs, const std::vector<double> &all_sigmas, Parser_1 *K, KW_item *kw, std::string cwd) const
{
	DECLKWD(model, KW_model, "MODEL");
	DECLKWD(parameters, KW_parameters, "PARAMETERS");

	kw->Start_pre();
	kw->Add_pre("MODEL");
	kw->Add_pre("PARAMETERS");
	kw->Finish_pre();

	delete Ecl;
	delete SimProxy;
	delete BDC;

	Ecl = new PMEclipse(K, kw, cwd, comm);
	BDC = new HMMPI::BlockDiagMat(comm, Ecl, Ecl);
	BDC->PrintToFile("SIMPROXY");							// just a debug report
	std::vector<int> b_starts = block_starts();				// the result is sync on all ranks
	std::vector<std::vector<double>> vals = extract_proxy_vals(dates, vecs, all_sigmas, b_starts);	// fills 'datapoint_block' (sync on "comm"); all input and output parameters are only used on comm-RANKS-0

	RANK0_SYNCERR_BEGIN(comm);
	if (vals.size() == 0)
		throw HMMPI::Exception("При заданных ECLSMRY и прочих настройках, не удалось извлечь данные для инициализации прокси-модели. Также стоит проверить, что DATES, ECLVECTORS и сигмы заданы корректно",
							   "Given the ECLSMRY and other settings, failed to extract any data for proxy model initialisation. Also check that the DATES, ECLVECTORS and sigmas are ok");
	RANK0_SYNCERR_END(comm);

	SimProxy = new PM_SimProxy(Ecl, K, kw, model, BDC, Ecl->Data(), b_starts, datapoint_block);
	std::vector<std::vector<double>> int_params = get_internal_parameters(parameters);				// valid on comm-RANKS-0

	// proxy training
	ValContSimProxy vc(comm, BDC->Data_ind(), vals);		// vals[smry_len][...] (from RANKS-0) are distributed among ranks
	SimProxy->AddData(int_params, &vc, 0);					// Nfval_pts = 0, but all points will be taken since it's a new proxy

	return SimProxy;
}
//--------------------------------------------------------------------------------------------------
// Writes data for the selected data points (dates x vecs) to ASCII file 'fname'
// If 'plain_order' = true, models are listed in their direct order, otherwise they are reordered starting from the densely spaced models to more stand-alone models
// K->KW_parameters is used to transfer external "params" to the internal representation used in the models ordering mentioned above
// All input parameters are only used on comm-RANKS-0
void SimProxyFile::ViewSmry(const std::string &fname, const std::vector<Date> &dates, const std::vector<SimSMRY::pair> &vecs, bool plain_order, Parser_1 *K) const	// This function is to be used for monitoring purposes
{
	DECLKWD(parameters, KW_parameters, "PARAMETERS");
	assert(parameters != 0);

	// 1. Get the data values
	const std::vector<int> b_starts = block_starts();					// the result is sync on all ranks
	const size_t N = dates.size()*vecs.size();
	std::vector<double> dummy_sigmas(N, 1.0);							// take all dates x vecs
	const std::vector<std::vector<double>> vals = extract_proxy_vals(dates, vecs, dummy_sigmas, b_starts);	// fills 'datapoint_block' (sync on "comm"); all input and output parameters are only used on comm-RANKS-0
																											// exceptions are sync
	// 2. Convert the parameters
	std::vector<std::vector<double>> int_params = get_internal_parameters(parameters);		// valid on comm-RANKS-0
	std::vector<int> mod_indices(int_params.size());					// accompanying indices of the models (design points)
	std::iota(mod_indices.begin(), mod_indices.end(), 0);
	std::vector<int> block_indices = block_ind;

	// 3. Make reordering of the models if necessary
	if (!plain_order)
	{
		Mat DMinternal = KrigStart::DistMatr(int_params, 0, int_params.size(), 0, int_params.size());	// distance matrix based on internal parameters
		std::vector<size_t> ord_models = KrigStart::IndSignificant(DMinternal, int_params.size());		// the order is: sparse to dense
		assert(ord_models.size() == int_params.size());
		assert(ord_models.size() == block_indices.size());
		std::reverse(ord_models.begin(), ord_models.end());												// now the order will be from dense to sparse
		int_params = Reorder(int_params, ord_models);
		mod_indices = Reorder(mod_indices, ord_models);
		block_indices = Reorder(block_indices, ord_models);
	}

	// 4. Output to the file
	if (Rank == 0)
	{
		FILE *file = fopen(fname.c_str(), "w");
		assert(file != NULL);

		fprintf(file, "%-5.5s\t%-5.5s\t%-11.11s", "", "", "");						// Header line 1: well names
		for (size_t j = 0; j < parameters->name.size(); j++)
			fprintf(file, "\t%-17.17s", "");
		for (size_t v = 0; v < vecs.size(); v++)
			for (size_t d = 0; d < dates.size(); d++)
				fprintf(file, "\t%-21.21s", vecs[v].first.c_str());
		fprintf(file, "\n");

		fprintf(file, "%-5.5s\t%-5.5s\t%-11.11s", "model", "model", "distance");	// Header line 2: property names
		for (size_t j = 0; j < parameters->name.size(); j++)
			fprintf(file, "\t%-17.17s", "");
		for (size_t v = 0; v < vecs.size(); v++)
			for (size_t d = 0; d < dates.size(); d++)
				fprintf(file, "\t%-21.21s", vecs[v].second.c_str());
		fprintf(file, "\n");

		fprintf(file, "%-5.5s\t%-5.5s\t%-11.11s", "index", "block", "(i-1, i)");	// Header line 3: indices, param names, dates
		for (size_t j = 0; j < parameters->name.size(); j++)
			fprintf(file, "\t%-17.17s", parameters->name[j].c_str());
		for (size_t v = 0; v < vecs.size(); v++)
			for (size_t d = 0; d < dates.size(); d++)
				fprintf(file, "\t%-21.21s", dates[d].ToString().c_str());
		fprintf(file, "\n");

		assert(datapoint_block.size() == N);										// datapoint_block is sync
		Mat int_params_prev;							// will store internal representation for distance calculation
		for (size_t i = 0; i < int_params.size(); i++)
		{
			fprintf(file, "%-5d", mod_indices[i]);			// Model index
			fprintf(file, "\t%-5d", block_indices[i]);		// Block index
			if (i == 0)
				fprintf(file, "\t%-11.11s", "--");			// Distance (i-1, i)
			else
			{
				double d = (int_params_prev - Mat(int_params[i])).Norm2();
				fprintf(file, "\t%-11.6g", d);
			}

			int_params_prev = Mat(int_params[i]);
			int_params[i] = parameters->InternalToExternal(int_params[i]);		// convert back to external representation for reporting
			assert(int_params[i].size() == parameters->name.size());

			for (size_t j = 0; j < parameters->name.size(); j++)
				fprintf(file, "\t%-17.12g", int_params[i][j]);					// Print the parameters

			for (size_t v = 0; v < vecs.size(); v++)		// Print the main array of values
				for (size_t d = 0; d < dates.size(); d++)
				{
					size_t dp_ind = v*dates.size() + d;		// data point index
					assert(dp_ind < N);
					int bl = datapoint_block[dp_ind];		// block index
					assert(bl < (int)b_starts.size()-1);
					int first_model = b_starts[bl];			// index of the first model where dp_ind exists

					if (mod_indices[i] < first_model)
						fprintf(file, "\t%-21.21s", "--");
					else
					{
						assert(mod_indices[i] - first_model < (int)vals[dp_ind].size());
						fprintf(file, "\t%-21.16g", vals[dp_ind][mod_indices[i] - first_model]);
					}
				}
			fprintf(file, "\n");
		}
		if (file) fclose(file);
	}
}
//--------------------------------------------------------------------------------------------------
std::vector<std::vector<double>> SimProxyFile::get_internal_parameters(const KW_parameters *par) const	// returns reordered 'params' such that their 'par_names' follow KW_parameters order,
{																										// and making conversion to the internal representation (output - comm-RANKS-0)
	// 1. Check that ECLSMRY parameters list is consistent with keyword PARAMETERS
	size_t par_names_size = par_names.size();
	MPI_Bcast(&par_names_size, 1, MPI_LONG_LONG, 0, comm);		// par_names_size is sync
	if (par_names_size != par->name.size())
		throw Exception(stringFormatArr(MessageRE("Количество параметров в файле ECLSMRY ({0:%zu}) не совпадает с PARAMETERS ({1:%zu})",
												  "Number of parameters in ECLSMRY file ({0:%zu}) does not match PARAMETERS ({1:%zu})"), std::vector<size_t>{par_names_size, par->name.size()}));

	std::vector<size_t> par_ind = GetSubvecIndSorted(par_names, par->name);
	HMMPI::Bcast_vector(par_ind, 0, comm);						// par_ind is sync
	std::vector<std::string> not_found = SubvecNotFound(par->name, par_ind);		// not_found is sync
	if (not_found.size() > 0)
		throw Exception((std::string)MessageRE("Следующие параметры из PARAMETERS не найдены в ECLSMRY: ", "The following parameters from PARAMETERS were not found in ECLSMRY: ") + ToString(not_found, "%s"));

	// 2. Convert params to internal representation
	std::vector<std::vector<double>> int_params(params.size());
	for (size_t i = 0; i < params.size(); i++)
	{
		int_params[i] = Reorder(params[i], par_ind);				// reorder according to PARAMETERS order
		int_params[i] = par->ExternalToInternal(int_params[i]);		// make internal parameters
	}

	return int_params;
}
//--------------------------------------------------------------------------------------------------
std::string SimProxyFile::models_params_msg() const
{
	return stringFormatArr(MessageRE("моделей: {0:%zu}, параметров: {1:%zu}", "models: {0:%zu}, parameters: {1:%zu}"), std::vector<size_t>{block_ind.size(), par_names.size()});
}
//--------------------------------------------------------------------------------------------------
}	// namespace HMMPI


