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

extern "C" int FileModTime(const char *file, time_t *time);		// C function, sets "time" to "file" modification time (seconds); returns "0" if all ok

namespace HMMPI
{

const int BUFFSIZE = 500;

bool FileExists(const std::string &fname);						// returns 'true' if file exists
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
// some classes for exception handling
//------------------------------------------------------------------------------------------
class Exception : public std::exception
{
protected:
	std::string msg;
public:
	Exception() : msg(""){};
	Exception(std::string s);						// #ifdef ERROR_TO_FILE, each rank also writes message to ERROR file
	Exception(std::string rus, std::string eng);	// #ifdef ERROR_TO_FILE, each rank also writes message to ERROR file
	~Exception() noexcept {};
	const char *what() const noexcept {return msg.c_str();};
};
//------------------------------------------------------------------------------------------
class EObjFunc : public Exception		// a more severe error within objective function calculation
{
public:
	EObjFunc(std::string e) : Exception(e){};
	EObjFunc(std::string rus, std::string eng) : Exception(rus, eng){};
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

std::string ToUpper(const std::string &s);				// converts string to uppercase
std::string Replace(const std::string &source, const std::string &find, const std::string &repl, int *count = NULL);	// in 'source', replaces all occurrences of 'find' with 'repl'; replacements count is ADDED to "count" if the latter is not NULL
std::string ReplaceArr(std::string source, const std::vector<std::string> &find, const std::vector<std::string> &repl, int *count = NULL);	// in 'source', replaces all occurrences of 'find[i]' with 'repl[i]'
																																			// replacements count is ADDED to "count" if the latter is not NULL
std::vector<const char *> vec_c_str_dodgy(const std::vector<std::string> &v);	// {string, string,...} -> {char*, char*,...}, DON'T use the resulting pointers once "v" is out of scope!
std::string Trim(const std::string &s, const std::string &trim_chars);		// removes "trim_chars" from left and right
std::string EraseSubstr(std::string s, const std::string &substr);			// erases all substrings 'substr' from 's'
std::vector<std::string> ParseEclChar(const std::string &s);							// gets array of tokens {a1, a2, a3,..} from a string "'a1'  'a2'  'a3' ..." (CHAR entries in eclipse formatted output)
void ParseEclSmallHdr(const std::string &s, std::string &a, int &b, std::string &c);	// gets {a, b, c} from a string "'a' b 'c'"

std::string getCWD(std::string fullname);			// get 'path' from 'path+file'
std::string getFile(std::string fullname);			// get 'file' from 'path+file'
//------------------------------------------------------------------------------------------
// CmdLauncher - class for executing commands via system() or MPI_Comm_spawn()
// Currently only deals with MPI_COMM_WORLD, since only one hostfile is currently supported
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

	void clear_mem() const;			// clears 'mem'

protected:
	const char *host_templ;			// host file template name (with %d)
	int rank;						// rank in MPI_COMM_WORLD

public:
	void ParseCmd(std::string cmd, bool &IsMPI, int &N, std::string &main_cmd, std::vector<char*> &argv, int &sync_flag) const;
									// Parses 'cmd' to decide whether it is an MPI command (and setting IsMPI flag)
									// In the MPI case also filling: N (from -n N, -np N), the main command 'main_cmd' (mpirun/mpiexec removed),
									// 		its arguments 'argv' (NULL-terminated; their deallocation is handled internally),
									//		and 'sync_flag' indicating the synchronization type required: 1 (default) - MPI_BarrierSleepy(), 2 - tNav *.end file
	std::vector<std::string> HostList() const;		// creates a list of hosts; to be called on MPI_COMM_WORLD; result is only valid on rank-0
	std::string MakeHostFile() const;				// creates a hostfile (returning its name on rank-0), avoiding file name conflicts in the CWD; to be called on MPI_COMM_WORLD

	int sync_tNav(std::string data_file) const noexcept;	// waits until an up-to-date tNavigator *.err file is available, returns the (mpi-sync) number of tNav errors
	static std::string get_end_file(const std::string &data_file);	// get the end file name
	int get_sync_flag(std::string main_cmd) const;			// returns the sync flag for 'main_cmd'

public:
	CmdLauncher();					// CTOR to be called on MPI_COMM_WORLD
	~CmdLauncher();

	void Run(std::string cmd) const;	// Runs command "cmd" (significant at rank-0), followed by a Barrier; should be called on all ranks of MPI_COMM_WORLD.
										// For non-MPI command: uses system() on rank-0, and throws a sync exception if the exit status is non-zero
										// For MPI command: uses MPI_Comm_spawn(), the program invoked should have a synchronizing MPI_BarrierSleepy() in the end,
										//					if tNavigator is invoked, the synchronization is based on *.end file
};
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
// some other utilities
//------------------------------------------------------------------------------------------
bool MPI_size_consistent();		// checks consistency of type sizes: 'size_t' -- MPI_UNSIGNED_LONG
								// 'char' -- MPI_CHAR, 'bool' -- MPI_BYTE
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
	TagPrintfMap();											// adds tags: MOD, PATH, RANK, SIZE; for the latter two also sets values (global MPI rank and size)
	TagPrintfMap(const std::vector<std::string> &tags, const std::vector<double> &vals);	// apart from 4 default tags, adds "tags" with "vals"
	TagPrintfMap(const TagPrintfMap &M) = delete;			// no copies so far
	const TagPrintfMap &operator=(const TagPrintfMap &M) = delete;
	~TagPrintfMap();										// frees "vals"
	void SetModPath(std::string mod, std::string path);		// sets values for MOD and PATH tags
	void SetSize(int size);									// sets value for SIZE tag
	void SetDoubles(const std::vector<std::string> &tags, const std::vector<double> &vals);	// sets "vals" for "tags", where "tags" is a subset of {this->first}
	std::set<std::string> get_tag_names() const;			// returns the set of all tag names (except MOD, PATH, RANK, SIZE)
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
