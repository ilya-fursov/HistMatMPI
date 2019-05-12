/*
 * Parsing.h
 *
 *  Created on: Mar 18, 2013
 *      Author: ilya
 */

#ifndef PARSING_H_
#define PARSING_H_


#include "Vectors.h"
#include "Utils.h"
#include "ConsoleCntr.h"
#include "GradientOpt.h"
#include <string>
#include <vector>
#include <map>
#include <list>
#include <typeinfo>

class Grid2D;
class Parser_1;				// defined later

//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
// lines read from the Control file
class DataLines
{
private:
	std::vector<std::string> lines;
public:
	DataLines(){};
	void LoadFromFile(std::string fname);			// read the lines in MPI manner
	int Length(){return lines.size();};
	std::string Get(int n){return lines[n];};    	// get n-th element
	std::vector<std::string> EliminateEmpty(); 		// eliminates empty entries and returns the VECTOR of lines
};
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
// a struct for handling "INCLUDES"
struct inputLN
{
	std::string line;		// line of text
	int shift;				// 'include' shift level
	std::string cwd;		// CWD where the include file resides; this CWD is tracked so that the include file can use paths relative to its own location
};
//------------------------------------------------------------------------------------------
// this stand-alone function throws exception if file cannot be opened
void CheckFileOpen(std::string fname);
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
// base class for all kinds of keywords and their associated parameters/actions
// USER: redefine at construction: name, delim, trim, erows, ecols, dec_verb
//		 redefine functions: constructor, ProcessParamTable(), Action()
//		 redefine/use at runtime: prerequisites, ReportKWProblem(), this->AddState("some_error"),
class KW_item
{
private:
	std::string state;		// empty string if this keyword's input/processing is ok, or error message (with '\n') if not
	std::vector<bool> kws_ok;					// same length as 'prerequisites', "true" if the corresponding prerequisite keyword is ok
	std::vector<std::string> prerequisites;		// list of prerequisite keywords (e.g. files, params) which should have been successfully processed for this keyword to be processed; 'prerequisites' can be set and used multiple times during runtime

	std::string ProblemKW();			// returns comma-separated list of problematic keywords; use after CheckKW(); used for verbosity<=-1
	std::string ProblemKW_long();		// returns '\n'-separated list of problematic keywords + their problem description; use after CheckKW(); used for verbosity>=0
	bool CheckKW();						// checks keywords from 'prerequisites' and sets "kws_ok", returns 'true' if all checked keywords are ok
	int CheckDefault(std::string s);		// converts string "N*" -> N, "other strings" -> 0
	std::vector<std::string> ReadDefaults(std::vector<std::string> p, int padto, int &def_count);	// puts "" (default values) where necessary in the array of strings 'p'
											// entries "N*" of 'p' are expanded to "", "",..., "" (N times); subsequently these can be treated in some "default" manner
	 	 	 	 	 	 	 	 	 	 	// if resulting array length is < 'padto', the array is padded with ""-entries from right
											// 'def_count' is incremented by the number of default values used
protected:
	Parser_1 *K;    		// Parser_1 *K contains the list of all other keywords, so that any keyword can access any other keyword. Thus, there's a lack of isolation, and it could be improved in the future.
	std::string delim;		// delimiters used by the keyword to parse parameters
	std::string trim;		// symbols removed from left and right of each parameters line
	std::string CWD;		// current working directory
	int erows, ecols;		// expected number of rows and columns as parameters; "-1" means any number of parameters
	int dec_verb;			// verbosity decrement for this keyword; when printing messages, this keyword will use [verbosity] = [global 'verbosity'] - [dec_verb]
	HMMPI::Vector2<std::string> par_table;	// erows x ecols table of parameters; this is the raw table, before any processing;
											// if erows==-1 or ecols==-1, the table size is defined in ReadParamTable()

	bool ReportKWProblem();				// prints keywords with problems (from 'prerequisites' list), accounting for verbosity, returns 'true' if there are no problems (does not throw exceptions)
	void AddState(std::string s);		// adds 's' to 'state', also handling the final "\n"
	void SilentError(std::string s);	// produces ERROR (K->AppText(), AddState(), K->TotalErrors++) without throwing exceptions

public:
	std::string name;		// name of the keyword; use UPPERCASE here

	KW_item(){delim = " \t\r"; trim = " \t\r"; K = 0; state = "Nothing is specified\n"; erows = 1; ecols = 0; dec_verb = 0;};		// in derived classes/functions explicitly ResetState() to show there are no errors
	virtual ~KW_item(){};
	void SetParser(Parser_1 *a){K = a;};       // set Parser pointer K
	void SetCWD(std::string s){CWD = s;};	   // set CWD
	std::string GetState() const {return state;};
	void SetState(std::string s) {state = s;};			// "s" should have '\n'
	void ExpParams(int &rows, int &cols){rows = erows; cols = ecols;};   // how many parameters are expected by the keyword (rows - how many lines, cols - how many columns);
	void ResetState(){state = "";};			   // for resetting 'state' in the beginning of this keyword processing

	void Start_pre() noexcept;					// start (reset) 'prerequisites'
	void Add_pre(std::string p) noexcept;		// add an item to 'prerequisites'
	void Finish_pre();							// check 'prerequisites' and throw exception

	void ReadParamTable(const std::vector<std::string> &SA) noexcept;		    // reads the table of parameters from "SA" into par_table, prints some report
							// if erows==-1, all lines are taken;
							// if ecols==-1, number of columns in the table is defined by the first row
							// if a row contains less parameters than needed, the rest are filled with ""
							// if a row contains more parameters than needed, an error is thrown
							// excessive rows in "SA" are ignored; if "SA" has less lines than needed, empty lines are used ("added") in the end
							// prints one line with the loaded parameters (or the lines count) -- for verbosity==1
							// prints multiple lines with the loaded parameters -- for verbosity>=2
	virtual void ProcessParamTable() noexcept {};	// processes "par_table" to get parameters from it; OVERRIDE it in descendant classes
	virtual void Action() noexcept {};		   // some action: read files or run something; OVERRIDE it in descendant classes
											   // Action() should not throw exceptions outside (except programming error exceptions), use try-catch inside!
											   // also ensure all proper MPI synchronisation is done inside of Action()
};
//------------------------------------------------------------------------------------------
// base class for running something
// USER:	redefine Run(), constructor(name)
class KW_run : public KW_item
{
protected:
	virtual void ProcessParamTable() noexcept {};		// 0 parameters, nothing to process

public:
	KW_run(){erows = 0; ecols = 0;};
	virtual void Action() noexcept;
	virtual void Run() = 0;					// the actual running procedure; OVERRIDE it for the different "run" descendants; CAN throw exceptions outside
};
//------------------------------------------------------------------------------------------
// base class for reading/writing a file or multiple (erows) similar files
// USER:	redefine DataIO(), constructor(name, erows)
//			in case of multiple files: redefine AllocateData(), add "data_vector"
class KW_fname : public KW_item
{
protected:
	std::string file_delim;						// delimiters for parsing files
	std::vector<std::string> fnames;			// this array is defined in ProcessParamTable()

	virtual void ProcessParamTable() noexcept;
	virtual void AllocateData() noexcept {};	// should allocate "data_vector" if multiple files are expected; this function is called from ProcessParamTable()
												// if only one file is expected, there's no need for "data_vector"
	virtual void DataIO(int i) = 0;	// read from (write to) file "fnames[i]"; this is the primary function to OVERRIDE in derived classes; CAN throw exceptions outside;
									// the derived class is supposed to provide some "data_vector", so DataIO(i) will work with data_vector[i]
public:
	KW_fname(){erows = 1; ecols = 1; delim = ""; file_delim = " \t\r"; dec_verb = -1;};		// redefine 'erows' in descendants; delim="", so parameter lines are not split into columns
	virtual void Action() noexcept;
};
//------------------------------------------------------------------------------------------
// base class for writing to a file or multiple (erows) similar files
// USER:	redefine DataIO(), constructor(name, erows)
//			in case of multiple files: redefine AllocateData(), add "data_vector"
class KW_fwrite : public KW_fname
{
public:
	virtual void Action() noexcept;		// the difference with KW_fname is that only process with MPI_rank == 0 performs I/O (writing)
};
//------------------------------------------------------------------------------------------
// base class for a row of parameters of different nature;
// USER:	redefine constructor(name, new_data_members, EXPECTED; use AddParam(), FinalizeParams())
//			optionally redefine UpdateParams()
//
// the parameters are supposed to correspond to the different simple data members (int, double, string) in the derived classes;
// specify their default values directly in the derived class constructors;
// for the derived classes normally only CONSTRUCTOR definition is necessary;
// ecols = fixed, erows = 1
class KW_params : public KW_item
{
protected:
	bool par_finalized;				// if 'true', AddParam() will not work
	std::vector<void*> DATA;		// DATA, TYPE, NAMES, EXPECTED - have length 'ecols'; DATA[i] may contain int*, double*, string*
	std::vector<int> TYPE;			// 0 - int, 1 - double, 2 - std::string
	std::vector<std::string> NAMES;	// parameter names for reporting
	std::vector<std::vector<std::string>> EXPECTED;		// in case of string parameters, only string from the specified set will be accepted
		// the whole mess with DATA, TYPE, NAMES, EXPECTED is to make processing of the parameters (which may be of different types!) in a unified way

	virtual void AddParam(int *val, const char *pname);	// calling this function makes a push_back of the necessary stuff to DATA, TYPE, NAME -- use it in constructors of derived classes
	virtual void AddParam(double *val, const char *pname);
	virtual void AddParam(std::string *val, const char *pname);
	void FinalizeParams();								// call this function after adding all the params with 'AddParam' (in constructors of derived classes) -- to set 'ecols' and allocate 'EXPECTED'
	std::string make_err_msg();				// message to print if some parameters are wrong
	std::string CheckExpected(int j);		// returns "" if string parameter #j has a valid value, error message otherwise
	std::string StrExpected(int j);			// returns a message - what are the valid values of string parameter #j
	virtual void UpdateParams() noexcept {};		// this update (and maybe some I/O) happens after parameters were read and processed, but before they are printed
	virtual void PrintParams() noexcept;			// print params values

public:
	KW_params(){erows = 1; ecols = 0; par_finalized = false;};
	virtual void ProcessParamTable() noexcept;	// process 'par_table'; normally does not need redefinition in derived classes
	virtual void Action() noexcept;				// UpdateParams, PrintParams, do FinalAction
	virtual void FinalAction() noexcept {};		// final thing to do in Action()
};
//------------------------------------------------------------------------------------------
// base class for multiple similar rows of parameters of different nature (e.g. column of 'int', column of 'double', etc);
// USER:	redefine constructor(name, new_data_members, EXPECTED; use AddParam(), FinalizeParams()); vector data members don't need to be allocated in constructor, this is done in ProcessParamTable()
//			optionally redefine UpdateParams()
//
// each column of parameters corresponds to a vector (of int, double, string) in the derived classes;
// for each row, simple default values are taken: 0/0.0/"";
// for the derived classes normally only CONSTRUCTOR definition is necessary;
// ecols = fixed, erows = any
class KW_multparams : public KW_params
{
protected:
	// DATA, TYPE, NAMES, EXPECTED - same as for "KW_params", but DATA[i] contains pointer to vector of int, double, string;
	// TYPE, NAMES, EXPECTED correspond to the whole column

	virtual void AddParam(std::vector<int> *val, const char *pname);	// calling this function makes a push_back of the necessary stuff to DATA, TYPE, NAME -- use it in constructors of derived classes
	virtual void AddParam(std::vector<double> *val, const char *pname);
	virtual void AddParam(std::vector<std::string> *val, const char *pname);
	std::string CheckExpected(int i, int j);		// returns "" if string parameter at row=i, col=j has a valid value, error message otherwise
	virtual void PrintParams() noexcept;			// print only if verbosity >= 1

public:
	KW_multparams(){erows = -1; ecols = 0;};
	virtual void ProcessParamTable() noexcept;		// process 'par_table'; will ALLOCATE new data vectors; normally does not need redefinition in derived classes
};
//------------------------------------------------------------------------------------------
// base class for a column of 'double'
// USER:	redefine constructor(name)
//			optionally redefine UpdateParams()
//
class KW_pardouble : public KW_multparams
{
public:
	std::vector<double> data;

	KW_pardouble();
};
//------------------------------------------------------------------------------------------
// base class for a column of 'int'
// USER:	redefine constructor(name)
//			optionally redefine UpdateParams()
//
class KW_parint : public KW_multparams
{
public:
	std::vector<int> data;

	KW_parint();
};
//------------------------------------------------------------------------------------------
// ConsTextTweak - base class for some tweaks of console text, e.g. colour;
// followed by its concrete descendants
//------------------------------------------------------------------------------------------
class ConsTextTweak
{
protected:
	HMMPI::TextAttr *TA;		// this class produces strings for tweaking the text

public:
	ConsTextTweak(HMMPI::TextAttr *ta) : TA(ta){};
	virtual ~ConsTextTweak(){};
	virtual std::string Tweak(std::string s) const = 0;
};
//------------------------------------------------------------------------------------------
// highlight "Keyword "
class CTT_Keyword : public ConsTextTweak
{
public:
	CTT_Keyword(HMMPI::TextAttr *ta) : ConsTextTweak(ta){};
	virtual std::string Tweak(std::string s) const;
};
//------------------------------------------------------------------------------------------
// specified string will be displayed with the specified colour
class CTT_ColorString : public ConsTextTweak
{
protected:
	size_t Slen;
	std::string S;
	HMMPI::Color C;

public:
	CTT_ColorString(std::string str, HMMPI::Color col, HMMPI::TextAttr *ta) : ConsTextTweak(ta), S(str), C(col) {Slen = S.length();};
	virtual std::string Tweak(std::string s) const;
};
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
// Parser_1 - the parser class
//------------------------------------------------------------------------------------------
class Parser_1 : public OptContext
{
private:
	std::vector<inputLN> InputLines;   	// it is assumed InputLines = DataLines::EliminateEmpty()

protected:
	std::map<std::string, KW_item*> KWList;		// list (map) of all recognisable keywords, stored here as pointers
	std::vector<ConsTextTweak*> CTTList;		// list of all text tweaks applied during AppText()

	std::string ApplyCTT(std::string s);		// applies all rules from CTTList to "s" and returns the result -- to be used in AppText()
public:
	static int Shift;				// is used to indicate the level of 'include'
	static size_t posit;			// position (index) to read in 'InputLines'
	std::string report;					// accumulates message for KW_report
	std::string msg;                	// buffer for accumulating messages for one keyword which is being parsed and processed
	bool echo;                    		// if 'true', text will be output by MPI process 0
	static std::string InitCWD;		// initial CWD (where program starts)
	static int verbosity;			// global verbosity for printing messages, default = 0, larger positive numbers - more verbose


	static int MPI_rank;			// in MPI_COMM_WORLD
	static int MPI_size;			// in MPI_COMM_WORLD
	int TotalErrors;					// counts all errors for the final report
	int TotalWarnings;

	Parser_1();
	void AddKW_item(KW_item *kwi);      // adds 'kwi' to 'KWList'
	void AddCTT(ConsTextTweak *ctt);	// adds 'ctt' to 'CTTList'
	void DeleteItems();					// deletes all KW_items from heap
	void DeleteCTTs();					// deletes all CTTList entries from heap
	void SetInputLines(const std::vector<std::string> &IL);			// sets 'InputLines' from 'IL' (from the main control file)
	void AddInputLines(const std::vector<inputLN> &newIL, int i);	// inserts 'newIL' in the middle, starting from index i ('newIL' - from the include file)


	template <class T> T *GetKW();	 	// pointer to concrete descendant of KW_item
	const KW_item *GetKW_item(std::string s) const;	 // pointer to KW_item with name "s", NULL if not found; search is done inside KWList;
	KW_item *GetKW_item(std::string s);	// non-const version of the above
	void AppText(std::string s);      	// sends/appends 's' to "cout" and to 'report', also adding the 'include' level numbers
	void ReadAll2();              		// parses the whole control file
};
//------------------------------------------------------------------------------------------
template <class T>
T *Parser_1::GetKW()
{
	T item;		// dummy variable, only to get the name;
				// it is hoped all KW_items have lightweight constructors
	return dynamic_cast<T*>(GetKW_item(item.name));
}
//------------------------------------------------------------------------------------------

#endif /* PARSING_H_ */
