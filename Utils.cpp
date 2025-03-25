/*
 * Utils.cpp
 *
 *  Created on: 29 Apr 2016
 *      Author: ilya fursov
 */

#include "Utils.h"
#include "MathUtils.h"
#include "EclSMRY.h"
#include "Parsing.h"
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <chrono>
#include <thread>
#include <cstdio>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
	// TODO currently not available - for WEXITSTATUS, WIFEXITED
#else
	#include <sys/wait.h>
#endif

namespace HMMPI
{

std::string MessageRE::lang = "ENG";
const int barrier_sleep_ms = 500;				// for MPI_BarrierSleepy

//------------------------------------------------------------------------------------------
void MsgToFileApp(const std::string &msg)		// output to TEST_CACHE file
{
#ifdef TEST_CACHE
	char fname[500];
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	sprintf(fname, TEST_CACHE, rank);
	FILE *f = fopen(fname, "a");
	if (f != NULL)
	{
		fputs(msg.c_str(), f);
		fclose(f);
	}
#endif
}
//------------------------------------------------------------------------------------------
bool FileExists(const std::string &fname)
{
	if (FILE *file = fopen(fname.c_str(), "r"))
	{
		fclose(file);
		return true;
	}
	else
		return false;
}
//------------------------------------------------------------------------------------------
int ExitStatus(int stat_val)			// get the sub-process exit code based on 'stat_val' returned from system(), using WEXITSTATUS
{										// if WIFEXITED == 0 (sub-process not exited normally), this function returns 1 to indicate something went wrong
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)						// also note, the exit code is 8-bit
	// TODO currently not available
	return stat_val;
#else
	if (WIFEXITED(stat_val))
		return WEXITSTATUS(stat_val);
	else
		return 1;
#endif
}
//------------------------------------------------------------------------------------------
// MessageRE
//------------------------------------------------------------------------------------------
MessageRE::operator std::string()
{
	if (lang == "RUS")
		return msg_rus;
	else if (lang == "ENG")
		return msg_eng;
	else
		throw Exception("Error in MessageRE::operator std::string");
}
//------------------------------------------------------------------------------------------
// Exception
//------------------------------------------------------------------------------------------
Exception::Exception(std::string s) : ExceptionBase(s)
{
#ifdef ERROR_TO_FILE
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	std::ofstream testf(HMMPI::stringFormatArr("ERROR_rank_{0:%d}.txt", std::vector<int>{rank}), std::ios::out);
	testf << msg;
	testf.close();
#endif
}
//------------------------------------------------------------------------------------------
Exception::Exception(std::string rus, std::string eng) : ExceptionBase(MessageRE(rus, eng))
{
#ifdef ERROR_TO_FILE
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	std::ofstream testf(HMMPI::stringFormatArr("ERROR_rank_{0:%d}.txt", std::vector<int>{rank}), std::ios::out);
	testf << msg;
	testf.close();
#endif
}
//------------------------------------------------------------------------------------------
// ManagedComm
//------------------------------------------------------------------------------------------
ManagedComm::~ManagedComm()
{
#ifdef TESTCTOR
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	std::ofstream testf(HMMPI::stringFormatArr("TESTCTOR_out_{0:%d}.txt", std::vector<int>{rank}), std::ios::app);
	testf << "rank " << rank << ", MPI_Comm_free, comm = " << comm << (comm == MPI_COMM_NULL ? " (NULL)" : "") << "\n";
	testf.close();
#endif

	if (comm != MPI_COMM_NULL)
		MPI_Comm_free(&comm);
}
//------------------------------------------------------------------------------------------
// StrUtils
//------------------------------------------------------------------------------------------
long StoL(std::string s, bool &complete)
{
//	size_t sz;
//	long res = stol(s, &sz);
//	complete = (sz == s.length());

	const char *input = s.c_str();	// 6.10.2013, using C++98 conversion
	char *end;
	double res = strtol(input, &end, 10);
	if (end == input || *end != '\0')
		complete = false;
	else
		complete = true;

	return res;
}
//------------------------------------------------------------------------------------------
double StoD(std::string s, bool &complete)
{
//	size_t sz;
//	double res = stod(s, &sz);
//	complete = (sz == s.length());

	const char *input = s.c_str();	// 6.10.2013, using C++98 conversion
	char *end;
	double res = strtod(input, &end);
	if (end == input || *end != '\0')
		complete = false;
	else
		complete = true;

	return res;
}
//------------------------------------------------------------------------------------------
long StoL(std::string s)
{
	bool complete;
	double res = StoL(s, complete);
	if (!complete)
		throw Exception("Невозможно преобразовать строку '" + s + "' в long int", "Cannot convert string '" + s + "' to long int");

	return res;
}
//------------------------------------------------------------------------------------------
double StoD(std::string s)
{
	bool complete;
	double res = StoD(s, complete);
	if (!complete)
		throw Exception("Невозможно преобразовать строку '" + s + "' в double", "Cannot convert string '" + s + "' to double");

	return res;
}
//------------------------------------------------------------------------------------------
int StrLen(const std::string &str)				// string length, counting russian characters properly
{
	int res_eng = 0;
	int res_rus = 0;
	for (const char &a : str)
		if (int(a) < 0)
			res_rus++;
		else
			res_eng++;

	if (res_rus % 2 != 0)
		throw Exception("Odd number of chars in russian characters in StrLen()");

	return res_eng + res_rus/2;
}
//------------------------------------------------------------------------------------------
std::string ToUpper(const std::string &s)
{
	std::string res = s;

	for (auto &i : res)
		i = toupper(i);

	return res;
}
//------------------------------------------------------------------------------------------
std::string Replace(const std::string &source, const std::string &find, const std::string &repl, int *count)
{
	std::string res;
	bool finished = false;

	std::string::size_type pos, lastPos = 0;
	while(!finished)
	{
		pos = source.find(find, lastPos);
		if (pos != std::string::npos)
		{
			res += std::string(source.data()+lastPos, pos-lastPos);
			res += repl;
			if (count != NULL)
				(*count)++;
		}
		else
		{
			res += std::string(source.data()+lastPos, source.length()-lastPos);
			finished = true;
		}
		lastPos = pos + find.length();
		if (lastPos >= source.length())
			finished = true;
	}

	return res;
}
//------------------------------------------------------------------------------------------
std::string ReplaceArr(std::string source, const std::vector<std::string> &find, const std::vector<std::string> &repl, int *count)
{
	if (find.size() != repl.size())
		throw EObjFunc("find.size() != repl.size() in ReplaceArr");

	std::string res = std::move(source);
	for (size_t i = 0; i < find.size(); i++)
		res = Replace(res, find[i], repl[i], count);

	return res;
}
//------------------------------------------------------------------------------------------
std::vector<const char *> vec_c_str_dodgy(const std::vector<std::string> &v)	// {string, string,...} -> {char*, char*,...}, DON'T use the resulting pointers once "v" is out of scope!
{
	std::vector<const char *> res(v.size(), NULL);
	for (size_t i = 0; i < v.size(); i++)
		res[i] = v[i].c_str();

	return res;
}
//------------------------------------------------------------------------------------------
std::string Trim(const std::string &s, const std::string &trim_chars)			// removes "trim_chars" from left and right
{
	size_t first, last;
	first = s.find_first_not_of(trim_chars);
	last = s.find_last_not_of(trim_chars);

	if (first == std::string::npos || last == std::string::npos)
		return "";
	else
		return s.substr(first, last-first+1);
}
//------------------------------------------------------------------------------------------
std::string EraseSubstr(std::string s, const std::string &substr)				// erases all substrings 'substr' from 's'
{
	size_t N = substr.length();
	for (size_t i = s.find(substr); i != std::string::npos; i = s.find(substr))
		s.erase(i, N);

	return s;
}
//------------------------------------------------------------------------------------------
std::vector<std::string> ParseEclChar(const std::string &s)
{
	std::vector<std::string> res;
	bool finished = false;
	bool searhing_first = true;

	size_t i, i0, from = 0;
	while (!finished)
	{
		i = s.find_first_of("'", from);

		if (i == std::string::npos)
			finished = true;
		else if (searhing_first)
		{
			from = i0 = i+1;
			searhing_first = false;
		}
		else
		{
			from = i+1;
			res.push_back(s.substr(i0, i-i0));
			searhing_first = true;
		}
	}

	return res;
}
//------------------------------------------------------------------------------------------
void ParseEclSmallHdr(const std::string &s, std::string &a, int &b, std::string &c)
{
	size_t i0, i1, i2, i3;
	i0 = s.find_first_of("'");
	i1 = s.find_first_of("'", i0+1);
	i2 = s.find_first_of("'", i1+1);
	i3 = s.find_first_of("'", i2+1);
	if (i0 == std::string::npos || i1 == std::string::npos || i2 == std::string::npos || i3 == std::string::npos)
		throw HMMPI::Exception("Ошибка парсинга в ParseEclSmallHdr", "Parsing error in ParseEclSmallHdr");

	a = s.substr(i0+1, i1-i0-1);
	c = s.substr(i2+1, i3-i2-1);
	b = HMMPI::StoL(HMMPI::Trim(s.substr(i1+1, i2-i1-1), " \t\r\n"));
}
//------------------------------------------------------------------------------------------
std::string getCWD(std::string fullpath)
{
	size_t ind = fullpath.find_last_of("/");
	if (ind != std::string::npos)
		return fullpath.substr(0, ind);
	else
		return "";
}
//------------------------------------------------------------------------------------------
std::string getFile(std::string fullpath)
{
	size_t ind = fullpath.find_last_of("/");
	return fullpath.substr(ind+1, std::string::npos);
}
//------------------------------------------------------------------------------------------
std::string getFullPath(std::string path, std::string file)		// combines 'path' and 'file'
{
	if (path == "")
		return file;
	else
		return path + "/" + file;
}
//------------------------------------------------------------------------------------------
// StringListing
//------------------------------------------------------------------------------------------
inline void StringListing::fill_max_length(size_t i, std::vector<size_t> &maxlen) const	// helper function
{
	assert(i < data.size());
	for (size_t j = 0; j < n; j++)
	{
		const size_t len = StrLen(data[i][j]);
		if (len > maxlen[j])
			maxlen[j] = len;
	}
}
//------------------------------------------------------------------------------------------
std::string StringListing::print(size_t i, const std::vector<size_t> &maxlen) const	// helper function
{
	assert(i < data.size());

	std::string res = "";
	char msg[BUFFSIZE];
	for (size_t j = 0; j < n; j++)
	{
		int rus_count = (int)data[i][j].length() - StrLen(data[i][j]);
		if (rus_count < 0)
			throw Exception("rus_count < 0 in StringListing::print");

		sprintf(msg, "%-*.*s", (int)maxlen[j] + rus_count, BUFFSIZE-5, data[i][j].c_str());
		if (j > 0)
			res += delim;
		res += msg;
	}

	return res;
}
//------------------------------------------------------------------------------------------
std::string StringListing::print_dots(const std::vector<size_t> &maxlen) const			// helper function, prints "..."
{
	std::string res;
	char msg[BUFFSIZE];
	for (size_t j = 0; j < n; j++)
	{
		sprintf(msg, "%-*.*s", (int)maxlen[j], BUFFSIZE-5, dots.c_str());
		if (j > 0)
			res += delim;
		res += msg;
	}

	return res;
}
//------------------------------------------------------------------------------------------
StringListing::StringListing(std::string d) : dots("..."), delim(d), n(0)
{
}
//------------------------------------------------------------------------------------------
void StringListing::AddLine(const std::vector<std::string> &line)		// append the 'line'
{
	if (data.size() == 0)
		n = line.size();
	else
	{
		if (line.size() != n)
		{
			char msg[BUFFSIZE];
			sprintf(msg, "Attempt to add a line of size %zu to StringListing with lines of size %zu", line.size(), n);
			throw Exception(msg);
		}
	}

	data.push_back(line);
}
//------------------------------------------------------------------------------------------
std::string StringListing::Print(int begin_max, int end_max) const
{							// formatted output, 'begin_max', 'end_max' - max. number of lines to print in the beginning/end
							// "-1" means output all lines
	// 'one_piece' = 'true' if one chunk is printed (no lines are skipped)
	const bool one_piece = (begin_max + end_max >= (int)data.size()) || (begin_max < 0) || (end_max < 0);

	// find the max. length
	size_t maxlen_0 = 0;
	if (!one_piece)
		maxlen_0 = dots.length();
	std::vector<size_t> maxlen(n, maxlen_0);				// max. string length for each column
	if (one_piece)
		for (size_t i = 0; i < data.size(); i++)
			fill_max_length(i, maxlen);
	else
	{
		for (size_t i = 0; i < (size_t)begin_max; i++)
			fill_max_length(i, maxlen);
		for (size_t i = data.size() - end_max; i < data.size(); i++)
			fill_max_length(i, maxlen);
	}

	// print
	std::string res;
	if (one_piece)
		for (size_t i = 0; i < data.size(); i++)
			res += print(i, maxlen) + "\n";
	else
	{
		for (size_t i = 0; i < (size_t)begin_max; i++)
			res += print(i, maxlen) + "\n";

		res += print_dots(maxlen) + "\n";

		for (size_t i = data.size() - end_max; i < data.size(); i++)
			res += print(i, maxlen) + "\n";
	}

	return res;
}
//------------------------------------------------------------------------------------------
// CmdLauncher
//------------------------------------------------------------------------------------------
void CmdLauncher::clear_mem() const			// clears 'mem'
{
	for (auto &v : mem)
		delete [] v;
}
//------------------------------------------------------------------------------------------
bool CmdLauncher::T_option_num::Read(std::string s)			// reads "NUMx", val = x
{
	int res;
	int num = sscanf(s.c_str(), tok.c_str(), &res);
	if (num == 1)
	{
		val = res;
		return true;
	}
	else
		return false;
}
//------------------------------------------------------------------------------------------
bool CmdLauncher::T_option_0::Read(std::string s)			// reads "TOK", val = 1
{
	if (s == tok)
	{
		val = 1;
		return true;
	}
	else
		return false;
}
//------------------------------------------------------------------------------------------
std::vector<CmdLauncher::T_option*> CmdLauncher::Options::MakeVec()			// combines the options in one vector
{
	return std::vector<CmdLauncher::T_option*>{&Err, &Runfile, &Mpi, &Tnav};
}
//------------------------------------------------------------------------------------------
std::vector<std::string> CmdLauncher::ReadOptions(const std::vector<std::string> &toks, CmdLauncher::Options &opts)	// updates 'opts' from 'toks', returns the remaining part of 'toks'
{
	std::vector<CmdLauncher::T_option*> ovec = opts.MakeVec();
	size_t i = 0;
	for (i = 0; i < toks.size(); i++)
	{
		bool option_ok = false;
		for (auto &x : ovec)
			if (x->Read(toks[i]))	// found a valid option; the input 'opts' is updated automatically
			{
				option_ok = true;
				break;
			}
		if (!option_ok)				// toks[i] is not a valid option
			break;
	}

	// i <-> the first toks[i] which is not a valid option
	return std::vector<std::string>(toks.begin() + i, toks.end());
}
//------------------------------------------------------------------------------------------
// Parses 'cmd', to extract: options (returned), main command (main_cmd), and
// its arguments (argv, NULL-terminated, their deallocation is handled internally)
// The available options are:
// 		ERRx - expected exist status / error count (currently this check is not done everywhere)
// 		RUNFILE (followed by exactly one filename) - run as include-file
// 		MPIx - launch via MPI_Comm_spawn, n=x
// 		TNAV - in case of MPIx run, synchronize the job finish using the *.end file
// Options compatibility (+ for allowed, ++ for required, - for ignored):
// run	ERRx	RUNFILE	MPIx	TNAV	command
// sys	+		-		-		-		+
// mpi	+		-		++		+		++
// file	+		++		-		-		++
CmdLauncher::Options CmdLauncher::ParseCmd(std::string cmd, std::string &main_cmd, std::vector<char*> &argv) const
{
	CmdLauncher::Options res;

	std::vector<std::string> toks;
	tokenize(cmd, toks, " \t\r\n", true);

	toks = ReadOptions(toks, res);					// parse the leading options, strip them from 'toks'

	if (toks.size() == 1 && res.Runfile.val == 1)	// include-file case
	{
		res.Mpi.val = 0;							// disable some options
		res.Tnav.val = 0;

		main_cmd = toks[0];
		argv = std::vector<char*>();				// empty vector
	}
	else if (toks.size() >= 1 && res.Mpi.val > 0)	// MPI case
	{
		res.Runfile.val = 0;						// MPI disables RUNFILE
		main_cmd = toks[0];

		std::vector<std::string> new_toks(toks.begin() + 1, toks.end());	// fill the arguments
		argv = std::vector<char*>(new_toks.size());
		for (size_t i = 0; i < new_toks.size(); i++)
		{
			argv[i] = new char[new_toks[i].length()+1];
			memcpy(argv[i], new_toks[i].c_str(), new_toks[i].length()+1);
		}
		argv.push_back(NULL);						// NULL-termination
	}
	else											// system() case
	{
		res.Runfile.val = 0;						// disable RUNFILE
		res.Mpi.val = 0;							// disable MPI
		res.Tnav.val = 0;							// disable TNAV

		main_cmd = HMMPI::ToString(toks, "%s", " ");
		if (main_cmd.length() > 0)
			main_cmd.pop_back();					// remove '\n'
		argv = std::vector<char*>();				// empty vector
	}

	if (mem.size() > 0)
		clear_mem();
	mem = argv;										// save for further deletion

	return res;
}
//------------------------------------------------------------------------------------------
std::vector<std::string> CmdLauncher::ReportCmd(int num, CmdLauncher::Options opts, std::string main_cmd, std::vector<char*> argv)		// formatted report for SIMCMD
{
	std::vector<CmdLauncher::T_option*> ovec = opts.MakeVec();
	std::vector<std::string> res(ovec.size() + 2);

	char buff[HMMPI::BUFFSIZE];
	sprintf(buff, "%2d)", num);
	res[0] = buff;

	for (size_t i = 0; i < ovec.size(); i++)
	{
		sprintf(buff, "=%d", ovec[i]->val);
		if (ovec[i]->val > 0)
			res[i+1] = ovec[i]->name + buff;
		else
			res[i+1] = "";
	}

	sprintf(buff, "%%.%ds", HMMPI::BUFFSIZE - 5);		// buff <- format for strings
	if (argv.size() > 0)
		argv.pop_back();								// remove the final NULL
	std::string args = HMMPI::ToString(argv, buff, " ");
	if (args.length() > 0)
		args.pop_back();
	if (args.length() > 0)
		args = " (" + args + ")";

	res[ovec.size()+1] = main_cmd + args;

	return res;
}
//------------------------------------------------------------------------------------------
std::vector<std::string> CmdLauncher::HostList(int np) const	// creates a list of hosts; to be called on MPI_COMM_WORLD; result is only valid on rank-0
{																// 'np' (ref. on rank-0) is the number of MPI processes to be launched
	int reslen, size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	char buff[MPI_MAX_PROCESSOR_NAME];
	char *Buff0 = nullptr;
	if (rank == 0)
		Buff0 = new char[MPI_MAX_PROCESSOR_NAME*size];		// collected buffer

	MPI_Get_processor_name(buff, &reslen);					// local result
	MPI_Gather(buff, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
			   Buff0, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);

	std::vector<std::string> res(size);
	if (rank == 0)
	{
		for (int i = 0; i < size; i++)
			res[i] = Buff0 + MPI_MAX_PROCESSOR_NAME*i;		// raw collection to 'res'
		res = Unique(res);

		if (res.size() > (size_t)np)
			res = std::vector<std::string>(res.begin(), res.begin() + np);	// if 'np' is small, restrict the list of hosts

		delete [] Buff0;
	}

	return res;
}
//------------------------------------------------------------------------------------------
std::string CmdLauncher::MakeHostFile(int np) const		// creates a hostfile (returning its name on rank-0), avoiding file name conflicts in the CWD; to be called on MPI_COMM_WORLD
{														// 'np' (ref. on rank-0) is the number of MPI processes to be launched
	std::vector<std::string> hosts = HostList(np);			// result valid on rank-0
	std::string res;										// significant on rank-0

	if (rank == 0)
	{
		bool name_ok = false;
		int c = 0;
		char fname[BUFFSIZE];
		while (!name_ok)									// first, generate appropriate file name
		{
			sprintf(fname, host_templ, c);
			if (!FileExists(fname))
				name_ok = true;
			c++;
		}

		FILE *f = fopen(fname, "w");
		for (size_t i = 0; i < hosts.size(); i++)			// write to the file
			fprintf(f, "%s\n", hosts[i].c_str());
		fclose(f);

		res = fname;
	}

	return res;
}
//------------------------------------------------------------------------------------------
// NOTE this function was not tested after adding tn22 stuff!
int CmdLauncher::sync_tNav(std::string data_file) const noexcept
{										// waits until an up-to-date tNavigator *.end file is available, returns the (mpi-sync) number of tNav errors
	int res = 1;
	if (rank == 0)
	{
		bool waiting = true;			// 'waiting for the file'
		std::string end_file18 = get_end_file(data_file, false);
		std::string end_file22 = get_end_file(data_file, true);
		while (waiting)
		{
			std::string end_file;
			bool tn22;
			if (FileExists(data_file) && FileExists(end_file18) && FileModCompare(data_file, end_file18) < 0)	// identify the end file format on the run
			{
				end_file = end_file18;
				tn22 = false;
			}
			else
			{
				end_file = end_file22;
				tn22 = true;
			}

			if (FileExists(data_file) && FileExists(end_file) && FileModCompare(data_file, end_file) < 0)
			{
				std::this_thread::sleep_for(std::chrono::seconds(2));	// wait 2 seconds to avoid conflicts
				FILE *file = fopen(end_file.c_str(), "r");

				std::string token = "Errors %d";
				if (tn22)
					token = "Ошибок %d";								// TODO the token is currently set based on 'tn' version (misleading), not its language!

				while (file != NULL && !feof(file))
				{
					if (fscanf(file, token.c_str(), &res) == 1)
					{
						waiting = false;
						break;
					}
					else
					{
						char c;
						fscanf(file, "%c", &c);
					}
				}
				fclose(file);
			}
			else
				std::this_thread::sleep_for(std::chrono::seconds(5));	// no "end file" found: wait 5 seconds
		}
	}

	MPI_BarrierSleepy(MPI_COMM_WORLD);
	MPI_Bcast(&res, 1, MPI_INT, 0, MPI_COMM_WORLD);

	return res;
}
//------------------------------------------------------------------------------------------
std::string CmdLauncher::get_end_file(const std::string &data_file, bool tn22)	// get the end file name; flag 'tn22' indicates the tNav22 format
{
	size_t pos1 = data_file.find_last_of("/");
	size_t pos2 = data_file.find_last_of(".");
	int pos1p = pos1 + 1;

	if (pos1 == std::string::npos)
		pos1p = 0;
	if (pos2 == std::string::npos)
		pos2 = data_file.length();

	std::string res;
	if (tn22)
		res = data_file.substr(0, pos1p) + "RESULTS/" +
			  data_file.substr(pos1p, pos2-pos1p) + "/result.end";		// newer format
	else
		res = data_file.substr(0, pos1p) + "RESULTS/" +
			  data_file.substr(pos1p, pos2-pos1p) + ".end";				// older format

	return res;
}
//------------------------------------------------------------------------------------------
CmdLauncher::CmdLauncher() : host_templ("HMMPI_hostfile_%d.txt")
{															// CTOR to be called on MPI_COMM_WORLD
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
};
//------------------------------------------------------------------------------------------
CmdLauncher::~CmdLauncher()
{
	clear_mem();
}
//------------------------------------------------------------------------------------------
// Runs command "cmd" (significant at Comm-ranks-0), followed by a Barrier; should be called on all ranks of "Comm".
// For ordinary command: uses system() on Comm-ranks-0, and throws a sync exception if the exit status != ERRx.
// For MPI command (MPIx option): "Comm" must be MPI_COMM_WORLD;
// 					uses MPI_Comm_spawn(), the program invoked should have a synchronizing MPI_BarrierSleepy() in the end,
//					if tNavigator is invoked (TNAV option), the synchronization is based on *.end file
// For include-file command (RUNFILE option): parser "K" executes the specified file

// TODO this function was not thoroughly tested after refactoring on 02.11.2022 /added Options stuff/
// TODO currently any CWDs are not used; may need to add them
void CmdLauncher::Run(std::string cmd, Parser_1 *K, MPI_Comm Comm) const
{
	std::string main_cmd;
	std::vector<char*> argv;
	int Crank;

	MPI_Comm_rank(Comm, &Crank);
	CmdLauncher::Options Opts;
	if (Crank == 0)
		Opts = ParseCmd(cmd, main_cmd, argv);						// fill the options on Comm-ranks-0

	std::vector<CmdLauncher::T_option*> Ovec = Opts.MakeVec();
	for (auto op : Ovec)
		MPI_Bcast(&op->val, 1, MPI_INT, 0, Comm);					// sync the options

	if (Opts.Mpi.val == 0 && Opts.Runfile.val == 0)					// ordinary command; sync here
	{
		int status;
		if (Crank == 0)
			status = ExitStatus(system(main_cmd.c_str()));

		MPI_BarrierSleepy(Comm);
		MPI_Bcast(&status, 1, MPI_INT, 0, Comm);
		if (status != Opts.Err.val)									// error status; sync
		{
			char msg[BUFFSIZE], msgrus[BUFFSIZE];
			if (Crank == 0)
			{
				sprintf(msg, "Exit status %d in command: %.400s", status, main_cmd.c_str());
				sprintf(msgrus, "Exit status %d в команде: %.400s", status, main_cmd.c_str());
			}
			MPI_Bcast(msg, BUFFSIZE, MPI_CHAR, 0, Comm);
			MPI_Bcast(msgrus, BUFFSIZE, MPI_CHAR, 0, Comm);
			throw EObjFunc(msgrus, msg);
		}
	}
	else if (Opts.Mpi.val > 0)										// MPI; sync here
	{
		MPI_Comm newcomm;
		MPI_Info info;
		int np = Opts.Mpi.val;										// sync

		if (Comm != MPI_COMM_WORLD)
			throw Exception("Communicator should be 'MPI_COMM_WORLD' for the MPI_Comm_spawn() branch in CmdLauncher::Run()");

		std::string hfile = MakeHostFile(np);		// hfile - on rank-0
		MPI_Info_create(&info);
		if (rank == 0)
			MPI_Info_set(info, "hostfile", hfile.c_str());

		//			       				     		 root <---|
		MPI_Comm_spawn(main_cmd.c_str(), argv.data(), np, info, 0, MPI_COMM_WORLD, &newcomm, MPI_ERRCODES_IGNORE);

		// SYNCHRONIZATION WITH CHILDREN
		if (Opts.Tnav.val == 0)						// sync
		{
			MPI_BarrierSleepy(newcomm);				// default synchronization
		}
		else
		{											// tNav synchronization
			std::string data_file = "dummy";
			int err_count;
			if (argv.size() >= 2)
				data_file = argv[argv.size()-2];	// data file is supposed to be the last non-NULL argument

			err_count = sync_tNav(data_file);		// the BARRIER

			if (err_count != Opts.Err.val)			// sync
			{
				char msg[BUFFSIZE], msgrus[BUFFSIZE];
				if (rank == 0)
				{
					sprintf(msg, "tNavigator finished with %d error(s)", err_count);
					sprintf(msgrus, "tNavigator завершился с %d ошибка(ми)", err_count);
				}
				MPI_Bcast(msg, BUFFSIZE, MPI_CHAR, 0, MPI_COMM_WORLD);
				MPI_Bcast(msgrus, BUFFSIZE, MPI_CHAR, 0, MPI_COMM_WORLD);
				MPI_Info_free(&info);
				MPI_Comm_free(&newcomm);
				if (rank == 0)
					remove(hfile.c_str());

				throw EObjFunc(msgrus, msg);
			}
		}

		MPI_Info_free(&info);
		MPI_Comm_free(&newcomm);
		if (rank == 0)
			remove(hfile.c_str());
	}
	else if (Opts.Runfile.val == 1)				// include-file case; sync here
	{
		int error_cache = K->TotalErrors, delta;		// save the current error count

		Bcast_string(main_cmd, 0, Comm);		// main_cmd contains the file name
		DataLines dl;
		K->AppText("\n");
		dl.LoadFromFile(main_cmd);				// read the file
		K->ReadLines(dl.EliminateEmpty(), 1, HMMPI::getCWD(main_cmd));		// execute

		delta = K->TotalErrors - error_cache;	// check if the error count has increased
		MPI_Allreduce(MPI_IN_PLACE, &delta, 1, MPI_INT, MPI_MAX, Comm);
		if (delta != Opts.Err.val) {
			char msg[BUFFSIZE], msgrus[BUFFSIZE];
			sprintf(msg, "Running the file '%.300s' finished with %d error(s)", main_cmd.c_str(), delta);
			sprintf(msgrus, "Исполнение файла '%.300s' завершилось с %d ошибкой(ами)", main_cmd.c_str(), delta);
			throw EObjFunc(msgrus, msg);
		}
	}
	else
		throw Exception("Incorrect Opts in CmdLauncher::Run()");			// sync
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
std::string MPI_size_consistent()		// checks consistency of type sizes: 'size_t' -- MPI_LONG_LONG, 'char' -- MPI_CHAR, 'bool' -- MPI_BYTE
{										// on errors returns the message, on success returns ""
	int mpival_ulong = 0, mpival_char = 0, mpival_byte = 0, mpival_long = 0, mpival_longlong = 0;
	MPI_Type_size(MPI_UNSIGNED_LONG, &mpival_ulong);
	MPI_Type_size(MPI_CHAR, &mpival_char);
	MPI_Type_size(MPI_BYTE, &mpival_byte);
	MPI_Type_size(MPI_LONG, &mpival_long);
	MPI_Type_size(MPI_LONG_LONG, &mpival_longlong);

	char msg[BUFFSIZE];
	sprintf(msg, "MPI_LONG_LONG    : %d\tsize_t : %zu\n"
				 "MPI_CHAR         : %d\tchar   : %zu\n"
				 "MPI_BYTE         : %d\tbool   : %zu\n"
				 "MPI_UNSIGNED_LONG: %d\tclock_t: %zu\n"
				 "MPI_LONG_LONG    : %d\ttime_t : %zu\n"
				 "MPI_LONG         : %d\n",
		mpival_longlong, sizeof(size_t), mpival_char, sizeof(char), mpival_byte, sizeof(bool), mpival_ulong, sizeof(clock_t), mpival_longlong, sizeof(time_t), mpival_long);

	if (sizeof(size_t) == mpival_longlong &&	// MPI_LONG_LONG
		sizeof(char) == mpival_char &&			// MPI_CHAR
		sizeof(bool) == mpival_byte &&			// MPI_BYTE
		sizeof(clock_t) == mpival_ulong &&		// MPI_UNSIGNED_LONG
		sizeof(time_t) == mpival_longlong)		// MPI_LONG_LONG
		return "";
	else
		return msg;
}
//------------------------------------------------------------------------------------------
void MPI_BarrierSleepy(MPI_Comm comm)		// A (less responsive) barrier which does not consume much CPU
{
	MPI_Request req;

	int finished = 0;			// 'finished' will be set to TRUE 'simultaneously' on all procs once all procs hit "MPI_Ibarrier"
	MPI_Ibarrier(comm, &req);
	while (!finished)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(barrier_sleep_ms));
		MPI_Test(&req, &finished, MPI_STATUS_IGNORE);
	}
}
//------------------------------------------------------------------------------------------
void MPI_count_displ(MPI_Comm comm, int M, std::vector<int> &counts, std::vector<int> &displs)
{								// fills 'counts' and 'displs' needed for MPI_Gatherv/MPI_Scatterv for distributing the vector of size M on "comm"
	if (comm == MPI_COMM_NULL)	// all inputs and outputs are sync on "comm"
		return;

	int n, r, szr;
	MPI_Comm_size(comm, &n);
	MPI_Comm_rank(comm, &r);
	szr = M/n + (r < M%n);		// number of points on the given process

	displs = counts = std::vector<int>(n);
	MPI_Allgather(&szr, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);	// counts is sync

	displs[0] = 0;
	for (int i = 1; i < n; i++)
		displs[i] = displs[i-1] + counts[i-1];							// displs is sync
}
//------------------------------------------------------------------------------------------
std::string MPI_Ranks(std::vector<MPI_Comm> vc)
{
	int rank, size;							// global rank and size
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	std::vector<int> Ranks(vc.size());		// ranks of current process on each communicator
	for (size_t j = 0; j < vc.size(); j++)
		if (vc[j] != MPI_COMM_NULL)
			MPI_Comm_rank(vc[j], &Ranks[j]);
		else
			Ranks[j] = -1;

	std::vector<int*> CollRanks(vc.size(), 0);		// [vc.size] x [size] - array for collecting info at global RANK-0
	if (rank == 0)
		for (size_t j = 0; j < vc.size(); j++)
			CollRanks[j] = new int[size];

	for (size_t j = 0; j < vc.size(); j++)
		MPI_Gather(&Ranks[j], 1, MPI_INT, CollRanks[j], 1, MPI_INT, 0, MPI_COMM_WORLD);

	std::string res = "";

	if (rank == 0)
	{
		for (int i = 0; i < size; i++)
		{
			res += stringFormatArr("{0:%d}", std::vector<int>{i});		// global rank in the first column
			for (size_t j = 0; j < vc.size(); j++)
				res += stringFormatArr("\t{0:%d}", std::vector<int>{CollRanks[j][i]});		// local ranks
			res += "\n";
		}
		for (size_t j = 0; j < vc.size(); j++)
			delete [] CollRanks[j];
	}

	return res;
}
//------------------------------------------------------------------------------------------
int FileModCompare(std::string f1, std::string f2)
{
	time_t mt1, mt2;
	if (FileModTime(f1.c_str(), &mt1) != 0)
		throw Exception(stringFormatArr("Невозможно получить информацию о файле {0:%s}", "Cannot get info on file {0:%s}", f1));
	if (FileModTime(f2.c_str(), &mt2) != 0)
		throw Exception(stringFormatArr("Невозможно получить информацию о файле {0:%s}", "Cannot get info on file {0:%s}", f2));

	if (mt1 < mt2)
		return -1;
	else if (mt1 == mt2)
		return 0;
	else
		return 1;
}
//------------------------------------------------------------------------------------------
void fread_check(void *data, size_t size, size_t count, FILE *fd)
{
	size_t c;
	c = fread(data, size, count, fd);
	if (c != count)
		throw Exception(stringFormatArr("Неудачное чтение {0:%zu} байт из файла", "Failed to read {0:%zu} bytes from file", size*count));
}
//------------------------------------------------------------------------------------------
template <>
std::string stringFormatArr(std::string str, const std::vector<std::string> &data)
{
	std::string::size_type pos, lastPos = 0;
	std::string res = "";
	std::string fmt = "";
	std::string bracket = "{";
	bool finished = false;

	while(!finished)
	{
		pos = str.find_first_of(bracket, lastPos);
		if (pos != std::string::npos)
		{
			if (bracket == "{")
			{
				res += std::string(str.data()+lastPos, pos-lastPos);
				fmt = "";
				bracket = "}";
			}
			else
			{
				fmt = std::string(str.data()+lastPos, pos-lastPos);
				std::vector<std::string> fmt_parts;
				tokenize(fmt, fmt_parts, " :", true);
				if (fmt_parts.size() != 1 && fmt_parts.size() != 2)
					throw Exception("Некорректный формат, ожидается {N:fmt}",
									"Incorrect format, expected {N:fmt} in stringFormatArr");

				bool cmpl = false;
				int num = StoL(fmt_parts[0], cmpl);
				if (!cmpl)
					throw Exception("Некорректный формат, ожидается {N:fmt}",
									"Incorrect format, expected {N:fmt} in stringFormatArr");

				if (num < 0 || num >= (int)data.size())
					throw Exception("Некорректный индекс N в {N:fmt}",
									"Incorrect index N in {N:fmt} in stringFormatArr");

				size_t DYNBUFF = BUFFSIZE*2;
				while (data[num].length() + BUFFSIZE > DYNBUFF-1)
					DYNBUFF *= 2;

				char *buff = new char[DYNBUFF];
				int n = -1;

				if (fmt_parts.size() != 1)	// 6.10.2013, C++98
					n = sprintf(buff, fmt_parts[1].c_str(), data[num].c_str());
				else
					n = sprintf(buff, "%g", *(double*)data[num].c_str());	// type selected to suppress warning

				if (n < 0 || n >= (int)DYNBUFF)
				{
					delete [] buff;
					throw Exception("Formatted output not successful in stringFormatArr");
				}

				res += std::string(buff);
				bracket = "{";

				delete [] buff;
			}
		}
		else
		{
			if (bracket == "{")
			{
				res += std::string(str.data()+lastPos, str.length()-lastPos);
				finished = true;
			}
			else
				throw Exception("Скобки {} несбалансированы в определении формата {N:fmt}",
								"Opening/closing format brackets {} not balanced in stringFormatArr");
		}

		lastPos = pos + 1;
	}

	return res;
}
//------------------------------------------------------------------------------------------
// I/O of vectors to files
//------------------------------------------------------------------------------------------	AUXILIARY OVERLOADS
void write_bin(FILE *fd, const std::string &s, int mode)
{
	size_t len = s.size();
	fwrite(&len, sizeof(len), 1, fd);
	fwrite(s.data(), sizeof(char), len, fd);
}
//------------------------------------------------------------------------------------------
void read_bin(FILE *fd, std::string &s, int mode)
{
	size_t len;
	fread_check(&len, sizeof(len), 1, fd);

	char *buff = new char[len+1];				// create 0-terminated c-string
	fread_check(buff, sizeof(char), len, fd);
	buff[len] = 0;

	s = std::string(buff);
	delete [] buff;
}
//------------------------------------------------------------------------------------------
void write_ascii(FILE *fd, const std::string &s)
{
	fprintf(fd, "%9s\t", s.c_str());
}
//------------------------------------------------------------------------------------------
void write_bin(FILE *fd, const std::pair<std::string, std::string> &p, int mode)
{
	write_bin(fd, p.first, mode);
	write_bin(fd, p.second, mode);
}
//------------------------------------------------------------------------------------------
void read_bin(FILE *fd, std::pair<std::string, std::string> &p, int mode)
{
	read_bin(fd, p.first, mode);
	read_bin(fd, p.second, mode);
}
//------------------------------------------------------------------------------------------
void write_ascii(FILE *fd, const std::pair<std::string, std::string> &p)
{
	fprintf(fd, "%9s %s\t", p.first.c_str(), p.second.c_str());
}
//------------------------------------------------------------------------------------------
void write_bin(FILE *fd, const Date &d, int mode)
{
	d.write_bin(fd);
}
//------------------------------------------------------------------------------------------
void read_bin(FILE *fd, Date &d, int mode)
{
	d.read_bin(fd);
}
//------------------------------------------------------------------------------------------
void write_ascii(FILE *fd, const Date &d)
{
	fprintf(fd, "%s\t", d.ToString().c_str());
}
//------------------------------------------------------------------------------------------	SPECIALIZATIONS
template<>
void write_bin(FILE *fd, const std::vector<double, std::allocator<double>> &v, int mode)
{
	write_bin_work<double>(fd, v, mode);
}
//------------------------------------------------------------------------------------------
template<>
void read_bin(FILE *fd, std::vector<double, std::allocator<double>> &v, int mode)
{
	read_bin_work<double>(fd, v, mode);
}
//------------------------------------------------------------------------------------------
template<>
void write_ascii(FILE *fd, const std::vector<double, std::allocator<double>> &v)
{
	size_t len = v.size();
	fprintf(fd, "%10zu\t", len);
	SaveASCII(fd, v.data(), len);
}
//------------------------------------------------------------------------------------------
template<>
void write_bin(FILE *fd, const std::vector<int, std::allocator<int>> &v, int mode)
{
	write_bin_work<int>(fd, v, mode);
}
//------------------------------------------------------------------------------------------
template<>
void read_bin(FILE *fd, std::vector<int, std::allocator<int>> &v, int mode)
{
	read_bin_work<int>(fd, v, mode);
}
//------------------------------------------------------------------------------------------
template<>
void write_ascii(FILE *fd, const std::vector<int, std::allocator<int>> &v)
{
	size_t len = v.size();
	fprintf(fd, "%10zu\t", len);
	SaveASCII(fd, v.data(), len, "%d");
}
//------------------------------------------------------------------------------------------
template<>
bool not_equal(const double &x, const double &y)
{
	return std::signbit(x) != std::signbit(y) || x != y;
}
//------------------------------------------------------------------------------------------

}	// namespace HMMPI
