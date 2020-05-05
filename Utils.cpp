/*
 * Utils.cpp
 *
 *  Created on: 29 Apr 2016
 *      Author: ilya fursov
 */

#include "Utils.h"
#include "MathUtils.h"
#include "EclSMRY.h"
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <chrono>
#include <thread>

namespace HMMPI
{

std::string MessageRE::lang = "ENG";
char TagPrintfValBase::buff[BUFFSIZE];
const int barrier_sleep_ms = 500;			// for MPI_BarrierSleepy

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
Exception::Exception(std::string s) : msg(s)
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
Exception::Exception(std::string rus, std::string eng) : msg(MessageRE(rus, eng))
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
std::string getCWD(std::string fullname)
{
	size_t ind = fullname.find_last_of("/");
	if (ind != std::string::npos)
		return fullname.substr(0, ind);
	else
		return "";
}
//------------------------------------------------------------------------------------------
std::string getFile(std::string fullname)
{
	size_t ind = fullname.find_last_of("/");
	return fullname.substr(ind+1, std::string::npos);
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
void CmdLauncher::clear_mem() const			// clears 'mem'
{
	for (auto &v : mem)
		delete [] v;
}
//------------------------------------------------------------------------------------------
void CmdLauncher::ParseCmd(std::string cmd, bool &IsMPI, int &N, std::string &main_cmd, std::vector<char*> &argv, int &sync_flag) const
{											// Parses 'cmd' to decide whether it is an MPI command (and setting IsMPI flag)
	std::vector<std::string> toks;			// In the MPI case also filling: N (from -n N, -np N), the main command 'main_cmd' (mpirun/mpiexec removed),
	tokenize(cmd, toks, " \t\r\n", true);	// 		its arguments 'argv' (NULL-terminated; their deallocation is handled internally),
											//		and 'sync_flag' indicating the synchronization type required: 1 (default) - MPI_BarrierSleepy(), 2 - tNav *.end file

	if (toks.size() > 1 && (toks[0] == "mpirun" || toks[0] == "mpiexec"))	// MPI case
	{
		IsMPI = true;
		N = 1;
		int main_start = 1;				// index where the main cmd tokens start
		if (toks.size() > 3 && (toks[1] == "-n" || toks[1] == "-np"))
		{
			N = StoL(toks[2]);
			main_start = 3;
		}

		main_cmd = toks[main_start];

		std::vector<std::string> new_toks(toks.begin() + main_start + 1, toks.end());	// fill the arguments
		argv = std::vector<char*>(new_toks.size());
		for (size_t i = 0; i < new_toks.size(); i++)
		{
			argv[i] = new char[new_toks[i].length()+1];
			memcpy(argv[i], new_toks[i].c_str(), new_toks[i].length()+1);
		}
		argv.push_back(NULL);			// NULL-termination

		sync_flag = get_sync_flag(main_cmd);
	}
	else								// non-MPI case
	{
		IsMPI = false;
		N = 1;
		main_cmd = cmd;
		argv = std::vector<char*>();	// empty vector
		sync_flag = 1;
	}

	if (mem.size() > 0)					// save for further deletion
		clear_mem();
	mem = argv;
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
int CmdLauncher::sync_tNav(std::string data_file) const noexcept
{										// waits until an up-to-date tNavigator *.end file is available, returns the (mpi-sync) number of tNav errors
	int res = 1;
	if (rank == 0)
	{
		bool waiting = true;			// 'waiting for the file'
		std::string end_file = get_end_file(data_file);
		while (waiting)
		{
			if (FileExists(data_file) && FileExists(end_file) && FileModCompare(data_file, end_file) < 0)
			{
				std::this_thread::sleep_for(std::chrono::seconds(2));	// wait 2 seconds to avoid conflicts
				FILE *file = fopen(end_file.c_str(), "r");
				while (file != NULL && !feof(file))
				{
					if (fscanf(file, "Errors %d", &res) == 1)
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
std::string CmdLauncher::get_end_file(const std::string &data_file)		// get the end file name
{
	size_t pos1 = data_file.find_last_of("/");
	size_t pos2 = data_file.find_last_of(".");
	int pos1p = pos1 + 1;

	if (pos1 == std::string::npos)
		pos1p = 0;
	if (pos2 == std::string::npos)
		pos2 = data_file.length();

	std::string res = data_file.substr(0, pos1p) + "RESULTS/" +
					  data_file.substr(pos1p, pos2-pos1p) + ".end";

	return res;
}
//------------------------------------------------------------------------------------------
int CmdLauncher::get_sync_flag(std::string main_cmd) const		// returns the sync flag for 'main_cmd'
{
	int res = 0;
	std::string work = ToUpper(main_cmd);
	if (work.find("TNAV") == std::string::npos)		// whenever 'main_cmd' has 'tNav' (case-insensitive), FLAG = 2
		res = 1;
	else
		res = 2;

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
void CmdLauncher::Run(std::string cmd) const	// Runs command "cmd" (significant at rank-0), followed by a Barrier; should be called on all ranks of MPI_COMM_WORLD.
{												// For non-MPI command: uses system() on rank-0, and throws a sync exception if the exit status is non-zero
	std::string main_cmd;						// For MPI command: uses MPI_Comm_spawn(), the program invoked should have a synchronizing MPI_BarrierSleepy() in the end,
	std::vector<char*> argv;					//					if tNavigator is invoked, the synchronization is based on *.end file
	int sync_flag;
	int np;
	bool ismpi;

	if (rank == 0)
		ParseCmd(cmd, ismpi, np, main_cmd, argv, sync_flag);
	MPI_Bcast(&ismpi, 1, MPI_BYTE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&sync_flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if (!ismpi)			// non-MPI; sync here
	{
		int status;
		if (rank == 0)
			status = system(cmd.c_str());
		MPI_BarrierSleepy(MPI_COMM_WORLD);
		MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
		if (status)		// error status; sync
		{
			char msg[BUFFSIZE], msgrus[BUFFSIZE];
			if (rank == 0)
			{
				sprintf(msg, "Exit status %d in command: %.400s", status, cmd.c_str());
				sprintf(msgrus, "Exit status %d в команде: %.400s", status, cmd.c_str());
			}
			MPI_Bcast(msg, BUFFSIZE, MPI_CHAR, 0, MPI_COMM_WORLD);
			MPI_Bcast(msgrus, BUFFSIZE, MPI_CHAR, 0, MPI_COMM_WORLD);
			throw EObjFunc(msgrus, msg);
		}
	}
	else				// MPI
	{
		MPI_Comm newcomm;
		MPI_Info info;

		std::string hfile = MakeHostFile(np);		// hfile - on rank-0, np - on rank-0
		MPI_Info_create(&info);
		if (rank == 0)
			MPI_Info_set(info, "hostfile", hfile.c_str());

		//			       				     		 root <---|
		MPI_Comm_spawn(main_cmd.c_str(), argv.data(), np, info, 0, MPI_COMM_WORLD, &newcomm, MPI_ERRCODES_IGNORE);

		// SYNCHRONIZATION WITH CHILDREN
		if (sync_flag == 1)				// sync
			MPI_BarrierSleepy(newcomm);				// default synchronization
		else
		{											// tNav synchronization
			std::string data_file = "dummy";
			int err_count;
			if (argv.size() >= 2)
				data_file = argv[argv.size()-2];	// data file is supposed to be the last non-NULL argument

			err_count = sync_tNav(data_file);		// the BARRIER

			if (err_count > 0)			// sync
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
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
bool MPI_size_consistent()
{
	int mpival_ulong = 0, mpival_char = 0, mpival_byte = 0, mpival_long = 0;
	MPI_Type_size(MPI_UNSIGNED_LONG, &mpival_ulong);
	MPI_Type_size(MPI_CHAR, &mpival_char);
	MPI_Type_size(MPI_BYTE, &mpival_byte);
	MPI_Type_size(MPI_LONG, &mpival_long);

	return (sizeof(size_t) == mpival_ulong &&		// MPI_UNSIGNED_LONG
			sizeof(char) == mpival_char &&			// MPI_CHAR
			sizeof(bool) == mpival_byte &&			// MPI_BYTE
			sizeof(clock_t) == mpival_ulong &&		// MPI_UNSIGNED_LONG
			sizeof(time_t) == mpival_long);			// MPI_LONG
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
void MPI_count_displ(MPI_Comm comm, int M, std::vector<int> &counts, std::vector<int> &displs)		// fills 'counts' and 'displs' needed for MPI_Gatherv/MPI_Scatterv for distributing the vector of size M on "comm"
{																									// all inputs and outputs are sync on "comm"
	if (comm == MPI_COMM_NULL)
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
void write_bin(FILE *fd, const std::string &s)
{
	size_t len = s.size();
	fwrite(&len, sizeof(len), 1, fd);
	fwrite(s.data(), sizeof(char), len, fd);
}
//------------------------------------------------------------------------------------------
void read_bin(FILE *fd, std::string &s)
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
void write_bin(FILE *fd, const std::pair<std::string, std::string> &p)
{
	write_bin(fd, p.first);
	write_bin(fd, p.second);
}
//------------------------------------------------------------------------------------------
void read_bin(FILE *fd, std::pair<std::string, std::string> &p)
{
	read_bin(fd, p.first);
	read_bin(fd, p.second);
}
//------------------------------------------------------------------------------------------
void write_ascii(FILE *fd, const std::pair<std::string, std::string> &p)
{
	fprintf(fd, "%9s %s\t", p.first.c_str(), p.second.c_str());
}
//------------------------------------------------------------------------------------------
void write_bin(FILE *fd, const Date &d)
{
	d.write_bin(fd);
}
//------------------------------------------------------------------------------------------
void read_bin(FILE *fd, Date &d)
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
void write_bin(FILE *fd, const std::vector<double, std::allocator<double>> &v)
{
	size_t len = v.size();
	fwrite(&len, sizeof(len), 1, fd);
	fwrite(v.data(), sizeof(double), len, fd);
}
//------------------------------------------------------------------------------------------
template<>
void read_bin(FILE *fd, std::vector<double, std::allocator<double>> &v)
{
	size_t len;
	fread_check(&len, sizeof(len), 1, fd);
	v = std::vector<double>(len);
	fread_check(v.data(), sizeof(double), len, fd);
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
void write_bin(FILE *fd, const std::vector<int, std::allocator<int>> &v)
{
	size_t len = v.size();
	fwrite(&len, sizeof(len), 1, fd);
	fwrite(v.data(), sizeof(int), len, fd);
}
//------------------------------------------------------------------------------------------
template<>
void read_bin(FILE *fd, std::vector<int, std::allocator<int>> &v)
{
	size_t len;
	fread_check(&len, sizeof(len), 1, fd);
	v = std::vector<int>(len);
	fread_check(v.data(), sizeof(int), len, fd);
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
// TagPrintfVal<T> specializations
//------------------------------------------------------------------------------------------
template <>
std::string TagPrintfVal<int>::get_fmt() const
{
	return "%d";
}
//------------------------------------------------------------------------------------------
template <>
std::string TagPrintfVal<double>::get_fmt() const
{
	return "%g";
}
//------------------------------------------------------------------------------------------
template <>
std::string TagPrintfVal<std::string>::get_fmt() const
{
	return "%s";
}
//------------------------------------------------------------------------------------------
template <>
std::string TagPrintfVal<std::string>::ToString(std::string fmt) const
{
	if (fmt == "")
		fmt = get_fmt();

	char *dynbuff = new char[val.size() + BUFFSIZE];	// take some extra length; val = std::string
	int n = sprintf(dynbuff, fmt.c_str(), val.c_str());
	if (n < 0 || n >= (int)val.size() + BUFFSIZE)
	{
		delete [] dynbuff;
		throw Exception("Ошибка форматированной записи в TagPrintfVal<string>::ToString",
						"Formatted output not successful in TagPrintfVal<string>::ToString");
	}

	std::string res(dynbuff);
	delete [] dynbuff;

	return res;
}
//------------------------------------------------------------------------------------------
// class TagPrintfMap
//------------------------------------------------------------------------------------------
TagPrintfMap::TagPrintfMap()
{
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	(*this)["MOD"] = new TagPrintfVal<std::string>("");
	(*this)["PATH"] = new TagPrintfVal<std::string>("");
	(*this)["RANK"] = new TagPrintfVal<int>(rank);
	(*this)["SIZE"] = new TagPrintfVal<int>(size);
}
//------------------------------------------------------------------------------------------
TagPrintfMap::TagPrintfMap(const std::vector<std::string> &tags, const std::vector<double> &vals) : TagPrintfMap()	// calls default ctor
{
	if (tags.size() != vals.size())
		throw Exception("tags.size() != vals.size() in TagPrintfMap ctor");

	for (size_t i = 0; i < tags.size(); i++)
	{
		iterator it = find(tags[i]);
		if (it != end())			// tag already exists
			throw Exception("Повторное добавление тэга " + tags[i], "Duplicate tag " + tags[i]);

		(*this)[tags[i]] = new TagPrintfVal<double>(vals[i]);
	}
}
//------------------------------------------------------------------------------------------
TagPrintfMap::~TagPrintfMap()
{
	for (auto &v : *this)
	{
		delete v.second;
		v.second = 0;
	}
}
//------------------------------------------------------------------------------------------
void TagPrintfMap::SetModPath(std::string mod, std::string path)
{
	delete (*this)["MOD"];
	(*this)["MOD"] = new TagPrintfVal<std::string>(mod);

	delete (*this)["PATH"];
	(*this)["PATH"] = new TagPrintfVal<std::string>(path);
}
//------------------------------------------------------------------------------------------
void TagPrintfMap::SetSize(int size)
{
	delete (*this)["SIZE"];
	(*this)["SIZE"] = new TagPrintfVal<int>(size);
}
//------------------------------------------------------------------------------------------
void TagPrintfMap::SetDoubles(const std::vector<std::string> &tags, const std::vector<double> &vals)	// sets "vals" for "tags", where "tags" is a subset of {this->first}
{
	if (tags.size() != vals.size())
		throw EObjFunc("tags.size() != vals.size() in TagPrintfMap::SetDoubles");

	for (size_t i = 0; i < tags.size(); i++)
	{
		iterator it = find(tags[i]);
		if (it == end())			// tag does not exist
			throw EObjFunc("Не найден тэг " + tags[i] + " в TagPrintfMap::SetDoubles", "Tag " + tags[i] + " was not found in TagPrintfMap::SetDoubles");

		delete it->second;
		it->second = new TagPrintfVal<double>(vals[i]);
	}
}
//------------------------------------------------------------------------------------------
std::set<std::string> TagPrintfMap::get_tag_names() const		// returns the set of all tag names (except MOD, PATH, RANK, SIZE)
{
	std::vector<std::string> names;								// first, bring the names to the vector
	names.reserve(this->size());
	for (const auto &p : *this)
		names.push_back(p.first);

	std::set<std::string> res(names.begin(), names.end());		// second, create the set
	res.erase("MOD");											// third, erase the default tags
	res.erase("PATH");
	res.erase("RANK");
	res.erase("SIZE");

	return res;
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
std::string stringTagPrintf(const std::string &format, const std::map<std::string, TagPrintfValBase*> &tag_val, int &count, std::set<std::string> &tags_left)
{
	size_t lastPos = 0;				// first position after the last bracket
	std::string res = "";
	std::string bracket = "$";		// bracket (tag delimiter) which is currently to be found
	bool finished = false;
	count = 0;

	while(!finished)
	{
		size_t pos = format.find_first_of(bracket, lastPos);
		if (pos == std::string::npos)		// if bracket is not found, the "end()" element is considered as the bracket
		{
			pos = format.size();
			finished = true;
		}

		if (bracket == "$")
		{
			res += std::string(format.data()+lastPos, pos-lastPos);
			bracket = "; \r\n\t";
		}
		else
		{
			std::string tag = std::string(format.data()+lastPos, pos-lastPos);	// currently processed tag from "format"
			std::vector<std::string> tag_parts;
			tokenize(tag, tag_parts, "%", true);
			if (tag_parts.size() != 1 && tag_parts.size() != 2)
				throw EObjFunc("Некорректный формат тэга '" + tag + "', ожидается: $TAG $TAG; $TAG%FMT $TAG%FMT;",
							   "Incorrect tag format '" + tag + "', expected: $TAG $TAG; $TAG%FMT $TAG%FMT;");

			auto it = tag_val.find(tag_parts[0]);		// search for tag from "format" within the "tag_val" list
			if (it == tag_val.end())
				throw EObjFunc("Тэг '" + tag_parts[0] + "' не найден в списке", "Tag '" + tag_parts[0] + "' was not found in the list");

			std::string fmt = "";
			if (tag_parts.size() == 2)
				fmt = "%" + tag_parts[1];

			res += it->second->ToString(fmt);			// it->second is TagPrintfValBase* value corresponding to current tag
			count++;
			tags_left.erase(tag_parts[0]);
			bracket = "$";
		}

		lastPos = pos;
		if (lastPos < format.size() && (format[lastPos] == ';' || format[lastPos] == '$'))		// for white-space brackets lastPos is not incremented
			lastPos++;
	}

	return res;
}
//------------------------------------------------------------------------------------------
std::vector<std::string> stringExtractTags(const std::string &format)
{
	size_t lastPos = 0;				// first position after the last bracket
	std::vector<std::string> res;
	std::string bracket = "$";		// bracket (tag delimiter) which is currently to be found
	bool finished = false;

	while(!finished)
	{
		size_t pos = format.find_first_of(bracket, lastPos);
		if (pos == std::string::npos)		// if bracket is not found, the "end()" element is considered as the bracket
		{
			pos = format.size();
			finished = true;
		}

		if (bracket == "$")
			bracket = "; \r\n\t";
		else
		{
			std::string tag = std::string(format.data()+lastPos, pos-lastPos);	// currently processed tag from "format"
			std::vector<std::string> tag_parts;
			tokenize(tag, tag_parts, "%", true);
			if (tag_parts.size() != 1 && tag_parts.size() != 2)
				throw Exception("Некорректный формат тэга '" + tag + "', ожидается: $TAG $TAG; $TAG%FMT $TAG%FMT;",
								"Incorrect tag format '" + tag + "', expected: $TAG $TAG; $TAG%FMT $TAG%FMT;");

			res.push_back(tag_parts[0]);
			bracket = "$";
		}

		lastPos = pos;
		if (lastPos < format.size() && (format[lastPos] == ';' || format[lastPos] == '$'))		// for white-space brackets lastPos is not incremented
			lastPos++;
	}

	return res;
}
//------------------------------------------------------------------------------------------

}	// namespace HMMPI
