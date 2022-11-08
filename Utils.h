/*
 * Utils.h
 *
 *  Created on: 29 Apr 2016
 *      Author: ilya fursov
 */

#ifndef UTILS_H_
#define UTILS_H_

// this header defines some utilities

#include "Abstract.h"
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>
#include <type_traits>
#include <mpi.h>
#include <map>
#include <utility>
#include <set>

// macro for a shorter definition of parameters in constructors, use in descendants of KW_params
// this line will do four things: add parameter address to DATA, add parameter type to TYPE, add parameter name to NAMES, set parameter value to VAL
#define DEFPAR(PAR, VAL) AddParam(&(PAR), #PAR); PAR = VAL

// same as above, but no value assignment - use in descendants of KW_multparams
#define DEFPARMULT(PAR) AddParam(&(PAR), #PAR)

// macro for shorter definition of imported keywords (within some routines performing calculations in descendants of KW_item)
// this line declares a variable KWvar of type KWtype, initialises it with keyword KWname, and also adds KWname to the prerequisites list
#define IMPORTKWD(KWvar, KWtype, KWname) KWtype *KWvar = dynamic_cast<KWtype*>(K->GetKW_item(KWname)); Add_pre(KWname)

// same as above, but no adding to the prerequisites list
#define DECLKWD(KWvar, KWtype, KWname) KWtype *KWvar = dynamic_cast<KWtype*>(K->GetKW_item(KWname))

// const declaration, needed if "K" is const
//#define DECLKWDCONST(KWvar, KWtype, KWname) const KWtype *KWvar = dynamic_cast<const KWtype*>(K->GetKW_item(KWname))	TODO const version of GetKW_item() was not checked

// A pair of macro brackets: they execute the code located between them on comm-rank-0
// The std::exceptions on comm-rank-0 are caught, and synchronously thrown on all ranks
// Example use:
// RANK0_SYNCERR_BEGIN(MPI_COMM_WORLD);
// ... code ...
// RANK0_SYNCERR_END(MPI_COMM_WORLD);
#define RANK0_SYNCERR_BEGIN(comm) 	{ 															\
										char errmsg_syncerr[500];								\
										errmsg_syncerr[0] = 0;									\
										int rank_syncerr;										\
										MPI_Comm_rank(comm, &rank_syncerr);						\
										if (rank_syncerr == 0)									\
										{														\
											try													\
											{													\
												// the body
#define RANK0_SYNCERR_END(comm)				} 													\
											catch (const std::exception &e)						\
											{													\
												sprintf(errmsg_syncerr, "%.495s", e.what());	\
											}													\
										}														\
										MPI_Bcast(errmsg_syncerr, 500, MPI_CHAR, 0, comm);		\
										if (errmsg_syncerr[0] != 0)								\
											throw HMMPI::Exception(errmsg_syncerr);				\
									}

// A pair of macro brackets for a critical section
// Example use:
// CRITICAL_BEGIN(MPI_COMM_WORLD);
// ... code ...
// CRITICAL_END(MPI_COMM_WORLD);
#define CRITICAL_BEGIN(comm)	{																\
									int rank_critical, size_critical;							\
									MPI_Comm_rank(comm, &rank_critical);						\
									MPI_Comm_size(comm, &size_critical);						\
									for (int i_critical = 0; i_critical < size_critical; i_critical++)	\
									{															\
										if (rank_critical == i_critical)						\
										{														\
											// the body
#define CRITICAL_END(comm)				}														\
										MPI_Barrier(comm);										\
									}															\
								}

extern "C" int FileModTime(const char *file, time_t *time);		// C function, sets "time" to "file" modification time (seconds); returns "0" if all ok

class Parser_1;													// forward declaration for CmdLauncher::Run()

namespace HMMPI
{

const int BUFFSIZE = 500;

bool FileExists(const std::string &fname);						// returns 'true' if file exists

int ExitStatus(int stat_val);									// get the sub-process exit code based on 'stat_val' returned from system(), using WEXITSTATUS
																// if WIFEXITED == 0 (sub-process not exited normally), this function returns 1 to indicate something went wrong
																// also note, the exit code is 8-bit
//------------------------------------------------------------------------------------------
// MessageRE - a class for displaying a message in the current language - Russian or English
//------------------------------------------------------------------------------------------
class MessageRE
{
protected:
	std::string msg_rus;
	std::string msg_eng;
public:
	static std::string lang;	// RUS, ENG
	MessageRE(std::string r, std::string e) : msg_rus(r), msg_eng(e){};
	operator std::string();
};
//------------------------------------------------------------------------------------------
// some concrete classes for exception handling
//------------------------------------------------------------------------------------------
class Exception : public ExceptionBase
{
public:
	Exception() : ExceptionBase("") {};
	Exception(std::string s);						// #ifdef ERROR_TO_FILE, each rank also writes message to ERROR file
	Exception(std::string rus, std::string eng);	// #ifdef ERROR_TO_FILE, each rank also writes message to ERROR file
};
//------------------------------------------------------------------------------------------
class EObjFunc : public Exception		// a more severe error within objective function calculation
{
public:
	EObjFunc(std::string e) : Exception(e) {};
	EObjFunc(std::string rus, std::string eng) : Exception(rus, eng) {};
};
//------------------------------------------------------------------------------------------
// class for managing communicators within ModelFactory
//------------------------------------------------------------------------------------------
class ManagedComm : public ManagedObject
{
protected:
	MPI_Comm comm;

public:
	ManagedComm(MPI_Comm c) : comm(c){};
	virtual ~ManagedComm();
};
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
// utilities for strings
//------------------------------------------------------------------------------------------
long StoL(std::string s, bool &complete);		// complete = true if whole string is a number
double StoD(std::string s, bool &complete);		// complete = true if whole string is a number
long StoL(std::string s);		// throws exception if whole string is not a number
double StoD(std::string s);		// throws exception if whole string is not a number

int StrLen(const std::string &str);				// string length, counting russian characters properly
std::string ToUpper(const std::string &s);				// converts string to uppercase
std::string Replace(const std::string &source, const std::string &find, const std::string &repl, int *count = NULL);	// in 'source', replaces all occurrences of 'find' with 'repl'; replacements count is ADDED to "count" if the latter is not NULL
std::string ReplaceArr(std::string source, const std::vector<std::string> &find, const std::vector<std::string> &repl, int *count = NULL);	// in 'source', replaces all occurrences of 'find[i]' with 'repl[i]'
																																			// replacements count is ADDED to "count" if the latter is not NULL
std::vector<const char *> vec_c_str_dodgy(const std::vector<std::string> &v);	// {string, string,...} -> {char*, char*,...}, DON'T use the resulting pointers once "v" is out of scope!
std::string Trim(const std::string &s, const std::string &trim_chars);		// removes "trim_chars" from left and right
std::string EraseSubstr(std::string s, const std::string &substr);			// erases all substrings 'substr' from 's'
std::vector<std::string> ParseEclChar(const std::string &s);							// gets array of tokens {a1, a2, a3,..} from a string "'a1'  'a2'  'a3' ..." (CHAR entries in eclipse formatted output)
void ParseEclSmallHdr(const std::string &s, std::string &a, int &b, std::string &c);	// gets {a, b, c} from a string "'a' b 'c'"

std::string getCWD(std::string fullpath);			// get 'path' from 'path+file'
std::string getFile(std::string fullpath);			// get 'file' from 'path+file'
std::string getFullPath(std::string path, std::string file);		// combines 'path' and 'file'
//------------------------------------------------------------------------------------------
// Class for accumulating the strings (lines) and then generating a formatted output for print.
// Handles the sub-string (item) width for different rows, and omits the rows from the middle if necessary
class StringListing
{
private:
	const std::string dots;

	inline void fill_max_length(size_t i, std::vector<size_t> &maxlen) const;	// helper function
	std::string print(size_t i, const std::vector<size_t> &maxlen) const;		// helper function
	std::string print_dots(const std::vector<size_t> &maxlen) const;			// helper function, prints "..."

protected:
	std::string delim;						// delimiter
	std::vector<std::vector<std::string>> data;
	size_t n;								// number of items per line

public:
	StringListing(std::string d);			// 'd' - delimiter between items in a line
	void AddLine(const std::vector<std::string> &line);		// append the 'line'
	std::string Print(int begin_max, int end_max) const;	// formatted output; 'begin_max', 'end_max' - max. number of lines to print in the beginning/end
															// "-1" means output all lines
};
//------------------------------------------------------------------------------------------
// CmdLauncher - class for executing commands via system(), or MPI_Comm_spawn(), or the inner parser.
// system() : works with a user-provided communicator
// MPI_Comm_spawn() : currently only deals with MPI_COMM_WORLD, since only one hostfile is currently supported
// inner parser : executes some include (control) file
//
// The compatibility of mpi calls (spawn) and non-mpi (system) calls of HistMatMPI
// and another program (standard one: std, or MPI one: MPI) is shown in the scheme below:
// 				 call type	:	mpi	 non-mpi	mpi	 non-mpi
// 				 prog type	:	std	   std		MPI	   MPI
// --------------------------------------------------------
// 	   mpi HMMPI			|	hang   ok		ok*	   --
// non-mpi HMMPI			|	hang   ok		ok*	   ok		 ok* = no exit code handling
//------------------------------------------------------------------------------------------
class CmdLauncher
{
private:
	mutable std::vector<char*> mem;			// holds the memory to be freed

	void clear_mem() const;					// clears 'mem'

protected:
	const char *host_templ;			// host file template name (with %d)
	int rank;						// rank in MPI_COMM_WORLD

public:
	class T_option									// class for handling the additional options in the beginning of the command line
	{
	protected:
		std::string tok;							// token for parsing
	public:
		std::string name;							// name for reporting
		int val;
		T_option() : tok(""), name(""), val(0){};
		virtual bool Read(std::string s) = 0; 		// attempts reading the option from "s", on success returns 'true' and overwrites 'val'
	};
	class T_option_num : public T_option			// reads "NUMx", val = x
	{
	public:
		T_option_num(std::string t, std::string n) {tok = t; name = n;};
		virtual bool Read(std::string s);
	};
	class T_option_0 : public T_option				// reads "TOK", val = 1
	{
	public:
		T_option_0(std::string t) {tok = name = t;};
		virtual bool Read(std::string s);
	};

	class Options									// structure to hold the command line options
	{												// Developer: expand it if necessary!
	public:
		T_option_num Err;
		T_option_0 Runfile;
		T_option_num Mpi;
		T_option_0 Tnav;

		Options() : Err("ERR%d", "ERR"), Runfile("RUNFILE"), Mpi("MPI%d", "MPI"), Tnav("TNAV"){};
		std::vector<T_option*> MakeVec();			// combines the options in one vector
	};

	static std::vector<std::string> ReadOptions(const std::vector<std::string> &toks, Options &opts);
									// updates 'opts' from 'toks', returns the remaining part of 'toks'

	Options ParseCmd(std::string cmd, std::string &main_cmd, std::vector<char*> &argv) const;
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
									// file	-		++		-		-		++

	static std::vector<std::string> ReportCmd(int num, Options opts, std::string main_cmd, std::vector<char*> argv);		// formatted report for SIMCMD

	std::vector<std::string> HostList(int np) const;	// creates a list of hosts; to be called on MPI_COMM_WORLD; result is only valid on rank-0
																// 'np' (ref. on rank-0) is the number of MPI processes to be launched
	std::string MakeHostFile(int np) const;				// creates a hostfile (returning its name on rank-0), avoiding file name conflicts in the CWD; to be called on MPI_COMM_WORLD
																// 'np' (ref. on rank-0) is the number of MPI processes to be launched
	int sync_tNav(std::string data_file) const noexcept;	// waits until an up-to-date tNavigator *.err file is available, returns the (mpi-sync) number of tNav errors
	static std::string get_end_file(const std::string &data_file, bool tn22);	// get the end file name; flag 'tn22' indicates the tNav22 format

	CmdLauncher();					// CTOR to be called on MPI_COMM_WORLD
	~CmdLauncher();

	void Run(std::string cmd, Parser_1 *K, MPI_Comm Comm = MPI_COMM_WORLD) const;
						// Runs command "cmd" (significant at Comm-ranks-0), followed by a Barrier; should be called on all ranks of "Comm".
						// For ordinary command: uses system() on Comm-ranks-0, and throws a sync exception if the exit status != ERRx.
						// For MPI command (MPIx option): "Comm" must be MPI_COMM_WORLD;
						// 					uses MPI_Comm_spawn(), the program invoked should have a synchronizing MPI_BarrierSleepy() in the end,
						//					if tNavigator is invoked (TNAV option), the synchronization is based on *.end file
						// For include-file command (RUNFILE option): parser "K" executes the specified file
};
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
// some other utilities
//------------------------------------------------------------------------------------------
std::string MPI_size_consistent();		// checks consistency of type sizes: 'size_t' -- MPI_LONG_LONG, 'char' -- MPI_CHAR, 'bool' -- MPI_BYTE
										// on errors returns the message, on success returns ""
void MPI_BarrierSleepy(MPI_Comm comm);		// A (less responsive) barrier which does not consume much CPU
void MPI_count_displ(MPI_Comm comm, int M, std::vector<int> &counts, std::vector<int> &displs);		// fills 'counts' and 'displs' needed for MPI_Gatherv/MPI_Scatterv for distributing the vector of size M on "comm"
																									// all inputs and outputs are sync on "comm"
std::string MPI_Ranks(std::vector<MPI_Comm> vc);	// get a string containing the ranks of each process (row) for each communicator in 'vc' (column)
													// non-empty result is returned to RANK-0 of MPI_COMM_WORLD; to be called on MPI_COMM_WORLD
int FileModCompare(std::string f1, std::string f2);	// returns -1 if modification time mt(f1) < mt(f2),
													//			0 if mt(f1) == mt(f2)
													//			1 if mt(f1) > mt(f2)
void fread_check(void *data, size_t size, size_t count, FILE *fd);		// same as "fread", but produces exception if the number of elements read != count
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
// some template functions for strings:

template <class ContainerT>																//  #####  ###   #  #  #####  #   #  ###  #####  #####
void tokenize(const std::string& str, ContainerT& tokens,								//    #   #   #  # #   #      ##  #   #      #   #
              const std::string& delimiters = " ", const bool trimEmpty = false);		//    #   #   #  ##    ####   # # #   #     #    ####
																						//    #   #   #  # #   #      #  ##   #    #     #
template <class ContainerT>																//    #    ###   #  #  #####  #   #  ###  #####  #####
void tokenizeExact(const std::string& str, ContainerT& tokens,
              const std::string& delimiters = " ", const bool trimEmpty = false);

template <class T>
std::string stringFormatArr(std::string str, const std::vector<T> &data);				// writes data[] in formatted manner to str = "{0}...{1:%f}...{2}..."

template <class T>
std::string stringFormatArr(std::string str_rus, std::string str_eng, const T &item);	// simplified version of "stringFormatArr" with two format string variants and one item (not an array)

// functions for I/O of vectors to files - used e.g. in SimProxyFile
template <typename T, typename A>
void write_bin(FILE *fd, const std::vector<T, A> &v);									//  ####   ###  #   #        ###     #  ###
																						//  #   #   #   ##  #         #     #  #   #
template <typename T, typename A>														//  ####    #   # # #         #    #   #   #
void read_bin(FILE *fd, std::vector<T, A> &v);											//  #   #   #   #  ##         #   #    #   #
																						//  ####   ###  #   #        ### #      ###
template <typename T, typename A>
void write_ascii(FILE *fd, const std::vector<T, A> &v);

void write_bin(FILE *fd, const std::string &s);									// auxiliary overloads for string
void read_bin(FILE *fd, std::string &s);
void write_ascii(FILE *fd, const std::string &s);

void write_bin(FILE *fd, const std::pair<std::string, std::string> &p);			// auxiliary overloads for pair<string, string>
void read_bin(FILE *fd, std::pair<std::string, std::string> &p);
void write_ascii(FILE *fd, const std::pair<std::string, std::string> &p);

struct Date;
void write_bin(FILE *fd, const Date &d);										// auxiliary overloads for Date
void read_bin(FILE *fd, Date &d);
void write_ascii(FILE *fd, const Date &d);

//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
// classes and functions for formatted string writing with tags
//------------------------------------------------------------------------------------------
class TagPrintfValBase		// base class for wrappers of int, double, string
{
protected:
	static char buff[BUFFSIZE];

public:
	virtual ~TagPrintfValBase(){};
	virtual std::string ToString(std::string fmt = "") const = 0;	// essentially, = sprintf(fmt, *this); for default fmt = "", format is taken which is different for different derived classes
};
//------------------------------------------------------------------------------------------
template <class T>
class TagPrintfVal : public TagPrintfValBase		// T is expected to be: int, double, std::string
{
protected:
	T val;

	std::string get_fmt() const;					// format string to use by default; this function is 100% specialized for int, double, std::string
public:
	TagPrintfVal(T x) : val(x){};
	virtual std::string ToString(std::string fmt = "") const;
};
//------------------------------------------------------------------------------------------
// class that stores pairs <tag, value>, to be used in formatted output;
// essentially it's an std::map with some additional handy routines
class TagPrintfMap : public std::map<std::string, TagPrintfValBase*>
{
public:
	TagPrintfMap();											// adds tags: MOD, PATH, RANK, SIZE, SMPL; for RANK, SIZE also sets values (global MPI rank and size)
	TagPrintfMap(const std::vector<std::string> &tags, const std::vector<double> &vals);	// apart from 5 default tags, adds "tags" with "vals"
	TagPrintfMap(const TagPrintfMap &M) = delete;			// no copies so far
	const TagPrintfMap &operator=(const TagPrintfMap &M) = delete;
	~TagPrintfMap();										// frees "vals"
	void SetModPath(std::string mod, std::string path);		// sets values for MOD and PATH tags
	void SetSize(int size);									// sets value for SIZE tag
	void SetSmpl(int smpl);									// sets value for SMPL tag
	void SetDoubles(const std::vector<std::string> &tags, const std::vector<double> &vals);	// sets "vals" for "tags", where "tags" is a subset of {this->first}
	std::set<std::string> get_tag_names() const;			// returns the set of all tag names (except MOD, PATH, RANK, SIZE, SMPL)
};
//------------------------------------------------------------------------------------------
std::string stringTagPrintf(const std::string &format, const std::map<std::string, TagPrintfValBase*> &tag_val, int &count, std::set<std::string> &tags_left);	// Writes vals corresponding to tags (tag_val) to appropriate tag locations in "format".
															// In "format", tag locations may be of form $tag, $tag%fmt (e.g. %fmt = %20.16g).
															// For simple form $tag, default format is taken depending on the corresp. vals type.
															// The end of tag locations is marked by whitespace (excluded from the tag substring), or semicolon ";" (included into tag, and then rejected).
															// TagPrintfMap object can be conveniently used as "tag_val".
															// Output "count" shows how many tags were replaced by values.
															// Tag names encountered in "format" are removed from "tags_left" set.
std::vector<std::string> stringExtractTags(const std::string &format);		// returns array of tag names (without "$" and "%fmt" ) found in 'format'; useful to check what parameters are present in string 'format'
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// BELOW ARE THE TEMPLATE DEFINITIONS ******************************************************************************************************************************************************************************************************************************************
//------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <class ContainerT>
void tokenize(const std::string& str, ContainerT& tokens,
              const std::string& delimiters, const bool trimEmpty)
{
   std::string::size_type pos, lastPos = 0;
   tokens = ContainerT();
   while(true)
   {
      pos = str.find_first_of(delimiters, lastPos);
      if(pos == std::string::npos)
      {
         pos = str.length();

         if(pos != lastPos || !trimEmpty)
            tokens.push_back(typename ContainerT::value_type(str.data()+lastPos,
                  (typename ContainerT::value_type::size_type)(pos - lastPos)));

         break;
      }
      else
      {
         if(pos != lastPos || !trimEmpty)
            tokens.push_back(typename ContainerT::value_type(str.data()+lastPos,
                  (typename ContainerT::value_type::size_type)(pos - lastPos)));
      }

      lastPos = pos + 1;
   }
};
//------------------------------------------------------------------------------------------
template <class ContainerT>
void tokenizeExact(const std::string& str, ContainerT& tokens,
              const std::string& delimiters, const bool trimEmpty)
{
   std::string::size_type pos, lastPos = 0;
   tokens = ContainerT();
   while(true)
   {
      pos = str.find(delimiters, lastPos);
      if(pos == std::string::npos)
      {
         pos = str.length();

         if(pos != lastPos || !trimEmpty)
            tokens.push_back(typename ContainerT::value_type(str.data()+lastPos,
                  (typename ContainerT::value_type::size_type)(pos - lastPos)));

         break;
      }
      else
      {
         if(pos != lastPos || !trimEmpty)
            tokens.push_back(typename ContainerT::value_type(str.data()+lastPos,
                  (typename ContainerT::value_type::size_type)(pos - lastPos)));
      }

      lastPos = pos + delimiters.length();
      if (lastPos >= str.length())
    	  break;
   }
};
//------------------------------------------------------------------------------------------
// writes data[] in formatted manner to str = "{0}...{1:%f}...{2}..."
template <class T>
std::string stringFormatArr(std::string str, const std::vector<T> &data)
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

				size_t DYNBUFF = BUFFSIZE;
//				if (std::is_same<T, std::string>::value)		// T == std::string is processed separately
//					while (data[num].length() > DYNBUFF-1)
//						DYNBUFF *= 2;

				char *buff = new char[DYNBUFF];
				int n = -1;

				if (fmt_parts.size() != 1)	// 6.10.2013
					n = sprintf(buff, fmt_parts[1].c_str(), data[num]);
				else
					n = sprintf(buff, "%g", (double)data[num]);

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
template <>
std::string stringFormatArr(std::string str, const std::vector<std::string> &data);
//------------------------------------------------------------------------------------------
// simplified version of stringFormatArr with two format string variants and one item (not an array)
template <class T>
std::string stringFormatArr(std::string str_rus, std::string str_eng, const T &item)
{
	return stringFormatArr<T>(MessageRE(str_rus, str_eng), std::vector<T>{item});
}
//------------------------------------------------------------------------------------------
// file I/O for vectors
//------------------------------------------------------------------------------------------	SPECIALIZATIONS
template <>																			// double
void write_bin(FILE *fd, const std::vector<double, std::allocator<double>> &v);
template <>
void read_bin(FILE *fd, std::vector<double, std::allocator<double>> &v);
template <>
void write_ascii(FILE *fd, const std::vector<double, std::allocator<double>> &v);
template <>																			// int
void write_bin(FILE *fd, const std::vector<int, std::allocator<int>> &v);
template <>
void read_bin(FILE *fd, std::vector<int, std::allocator<int>> &v);
template <>
void write_ascii(FILE *fd, const std::vector<int, std::allocator<int>> &v);
//------------------------------------------------------------------------------------------	DEFINITIONS
template <typename T, typename A>
void write_bin(FILE *fd, const std::vector<T, A> &v)
{
	size_t len = v.size();
	fwrite(&len, sizeof(len), 1, fd);
	for (size_t i = 0; i < len; i++)
		write_bin(fd, v[i]);
}
//------------------------------------------------------------------------------------------
template <typename T, typename A>
void read_bin(FILE *fd, std::vector<T, A> &v)
{
	size_t len;
	fread_check(&len, sizeof(len), 1, fd);
	v = std::vector<T, A>(len);

	for (size_t i = 0; i < len; i++)
		read_bin(fd, v[i]);
}
//------------------------------------------------------------------------------------------
template <typename T, typename A>
void write_ascii(FILE *fd, const std::vector<T, A> &v)
{
	size_t len = v.size();
	fprintf(fd, "%zu\n", len);
	for (size_t i = 0; i < len; i++)
		write_ascii(fd, v[i]);
	fprintf(fd, "\n");
}
//------------------------------------------------------------------------------------------
// class TagPrintfVal<T>
//------------------------------------------------------------------------------------------
template <class T>
std::string TagPrintfVal<T>::ToString(std::string fmt) const
{
	if (fmt == "")
		fmt = get_fmt();

	int n = sprintf(buff, fmt.c_str(), val);
	if (n < 0 || n >= BUFFSIZE)
	{
		throw Exception("Ошибка форматированной записи в TagPrintfVal<T>::ToString",
						"Formatted output not successful in TagPrintfVal<T>::ToString");
	}

	return buff;
}
//------------------------------------------------------------------------------------------
template <>
std::string TagPrintfVal<std::string>::ToString(std::string fmt) const;		// for T = std::string a specialization is used
//------------------------------------------------------------------------------------------
}	// namespace HistMatMPI
//------------------------------------------------------------------------------------------
#endif /* UTILS_H_ */
