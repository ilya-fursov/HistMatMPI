/*
 * EclSMRY.h
 *
 *  Created on: 30 Jan 2017
 *      Author: ilya fursov
 */

#ifndef ECLSMRY_H_
#define ECLSMRY_H_

#include <vector>
#include <string>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <utility>
#include <cmath>
#include "MathUtils.h"
#include "PhysModels.h"
#include "Utils.h"

class PMEclipse;		// forward declarations
class KW_parameters;

namespace HMMPI
{
//--------------------------------------------------------------------------------------------------
// classes and functions for elementary reading of Eclipse binary output
//--------------------------------------------------------------------------------------------------
enum KwdType{INTE, REAL, DOUB, CHAR, LOGI};			// keyword type
//--------------------------------------------------------------------------------------------------
struct Date											// 'time coordinate' of the modelled data
{
protected:
	int Day;
	int Month;
	int Year;
	double sec;				// seconds

public:
	Date() {Day = Month = Year = 0; sec = 0.0;};
	Date(const std::string &s);						// accepted 's' formats: DD.MM.YYYY, DD/MM/YYYY, optionally followed by " hh:mm::ss" or " hh:mm"
	Date(int d, int m, int y, double s = 0) : Day(d), Month(m), Year(y), sec(s) {};
	bool operator==(const Date &rhs) const {return Day == rhs.Day && Month == rhs.Month && Year == rhs.Year && sec == rhs.sec;};
	bool operator>(const Date &rhs) const;
	bool operator<(const Date &rhs) const {return !(*this > rhs) && !(*this == rhs);};
	double get_sec() const {return sec;};

	std::string ToString() const;
	void write_bin(FILE *fd) const;
	void read_bin(FILE *fd);

	static void parse_date_time(const std::string s, std::string delim, int &D, int &M, int &Y);	// can parse both DD.MM.YYYY, DD/MM/YYYY and hh:mm::ss
																									// if the last item ("YYYY" or "ss") is empty, then Y = 0
	std::vector<double> SubtractFromAll(const std::vector<Date> &vec) const;		// subtracts *this from all elements of 'vec'
																					// the resulting elementwise difference in _days_ is returned
};
//--------------------------------------------------------------------------------------------------
class SmryKwd			// base class for storing 'header(name, type) + data'
{
protected:
	KwdType type;		// type of values, e.g. REAL
	size_t len;			// number of values, e.g. 13

	KwdType type_from_str(std::string s);	// convert string CHAR, INTE, REAL, DOUB, LOGI to KwdType
	virtual void read_array(FILE *f, size_t start, size_t count){};		// this function will be non-trivial for SmryKwdData
public:
	std::string name;	// keyword name, e.g. PARAMS

	virtual ~SmryKwd(){};
	void ReadHeader(FILE *f);		// attempts to read the header from the current position; fills 'name', 'type', 'len'
	int ReadNamedHeader(FILE *f);	// searches and reads the nearest header with name = 'name', then fills 'type', 'len'
										// returns 0 on success, 1 on failure (header not found)
	SmryKwd *ReadData(FILE *f);		// reads (from the current position) data of size 'len', type 'type'
										// returns SmryKwdData<type> with 'name', 'type', 'len' as in "this"; 'data' - as read from the file
										// the returned pointer should be *** DELETED *** in the end
	void SkipReadData(FILE *f);		// skips the block of data (from the current position) - to reach the new position in the file
	virtual void cout(){};			// cout all sorts of data -- for debug
};
//--------------------------------------------------------------------------------------------------
template <class T>
class SmryKwdData : public SmryKwd					// class for reading and storing concrete data of different types (int32, float etc)
{
protected:
	virtual void read_array(FILE *f, size_t start, size_t count);		// allocates and reads 'data' [start, start + count) based on "T"

public:
	std::vector<T> data;

	SmryKwdData(const SmryKwd &k) : SmryKwd(k){};
	virtual void cout();
};
//--------------------------------------------------------------------------------------------------
void EclSMRYInitCheckSizes();
inline void ToChar(int32_t x, char *s);				// sets s[0]...s[3] from 'x'; 0-end should be set for 's' elsewhere
inline int32_t SwapEndian(int32_t x);

template <class T>
inline T ReadVal(FILE *f);							// template function for reading values of different types from binary file
template <> inline int32_t ReadVal<int32_t>(FILE *f);
template <> inline uint32_t ReadVal<uint32_t>(FILE *f);
template <> inline std::string ReadVal<std::string>(FILE *f);
template <> inline float ReadVal<float>(FILE *f);
template <> inline double ReadVal<double>(FILE *f);
//--------------------------------------------------------------------------------------------------
// abstract class for reading summary from binary files; derived classes read from Eclipse and tNavigator files
//--------------------------------------------------------------------------------------------------
class SimSMRY
{
public:
	typedef std::pair<std::string, std::string> pair;	// first = WGNAMES, second = KEYWORD; relational operators of 'pair' can be used e.g. for sorting vector<pair>

protected:
	std::string mod;					// "modname" in the last call to ReadFiles(); used for reporting

public:
	std::vector<double> Data;			// all modelled data in one row (for all dates and all vecs), ordered in row-major format, i.e.
										// 1st date + all vecs, 2nd date + all vecs,...
	std::vector<Date> dates;			// <D, M, Y>
	std::vector<pair> vecs;				// first = WGNAMES, second = KEYWORD

	SimSMRY() : mod("*"){};
	virtual ~SimSMRY(){};
	virtual SimSMRY *Copy() const = 0;					// creates a full copy of current object; should be _DELETED_ in the end
	virtual void ReadFiles(std::string modname) = 0;	// should set 'mod'!
	virtual std::string DatesToStr() const;				// forms strings for output
	virtual std::string VecsToStr() const;
	virtual std::string dates_file() const = 0;			// name of file with dates
	virtual std::string vecs_file() const = 0;			// name of file with vecs
	virtual std::string data_file() const = 0;			// name of file with data

	Mat ExtractSummary(const std::vector<Date> &dates1, std::vector<pair> vecs1, std::string &msg_dat, std::string &msg_vec, std::string suff = "") const;	// extracts summary, as defined by [dates1 x vecs1],
										// fills summary with "0" where dates1[*] or vecs1[*] are not found in this->dates, this->vecs
										// before searching vectors, attaches "suff" (e.g. "H", "S") to vecs1[*].second, making e.g. WBHP+H, WWCT+S
										// "msg_dat", "msg_vec" return info about not-found dates and vectors
};
//--------------------------------------------------------------------------------------------------
// class for reading binary *.SMSPEC, *.UNSMRY
// TODO currently doesn't work with hours, minutes, seconds
//--------------------------------------------------------------------------------------------------
class EclSMRY : public SimSMRY
{
protected:
	const std::vector<pair> date_vec;		// e.g. pairs <"", "DAY">, i.e. names of vectors to define dates

	void readSMSPEC(std::string modname);	// reads KEYWORDS, WGNAMES, UNITS from "modname.SMSPEC"; fills 'vecs', 'Units'
	void readUNSMRY(std::string modname);	// reads PARAMS from "modname.UNSMRY"; fills 'dates', 'Data'

public:
	std::vector<std::string> Units;			// units for each vector; only for information

	EclSMRY();
	virtual SimSMRY *Copy() const;			// _DELETE_ in the end!
	virtual void ReadFiles(std::string modname);		// reads summary from "modname.SMSPEC", "modname.UNSMRY"
	virtual std::string VecsToStr() const;
	virtual std::string dates_file() const;				// name of file with dates
	virtual std::string vecs_file() const;				// name of file with vecs
	virtual std::string data_file() const;				// name of file with data
};
//--------------------------------------------------------------------------------------------------
// class for reading *.meta, *.res (tNavigator)
//--------------------------------------------------------------------------------------------------
class tNavSMRY : public SimSMRY
{
public:
	typedef std::pair<std::string, std::vector<std::string>> SecObj;	// for definition of secondary objects

private:
	class T_ecl_prop_transform					// this class is essentially for handling the secondary (calculated) properties;
	{											// 'offsets', 'wght_offset' are filled in "read_meta", other fields are filled in CTOR
	public:
		std::string name;						// e.g. WWCT
		int flag;								// -1: negate, 0: nothing, 1: apply func()
		std::function<void(const int *offs, double*)> func_h;		// calculates *x = ..., involving values *(x + offs[i]) for different 'i'

		std::vector<int> offsets;				// offsets w.r.t. currently considered value 'x' in "Data"
		std::vector<std::string> args;			// (same size as 'offsets') - symbolic names of func's arguments; e.g. WOPR

		// these two int's control how summation takes place for the secondary objects
		int wght_flag;			// 0: all weights = 0 (e.g. for WBHP), 1: all weights = 1 (e.g. for WOPT), 2: all weights are taken from a property (e.g. for WOPR with WEFAC)
		int wght_offset;		// full offset for the weighting property w.r.t. current property; filled in read_meta()
		std::string wght_prop;	// name of the weighting property, e.g. WEFAC

		void func(double *x) const {func_h(offsets.data(), x);};	// calculates *x using 'offsets'

		// functions used as 'func'; USER: expand this list as appropriate
		static void f_wct(const int *offs, double *x) {double d = *(x + offs[0]) + *(x + offs[1]); if (d != 0) *x = *(x + offs[1])/d; else *x = 0;};	// args <-> WOPR, WWPR
		static void f_gor(const int *offs, double *x) {double d = *(x + offs[0]); if (d != 0) *x = *(x + offs[1])/d; else *x = 0;};						// args <-> WOPR, WGPR
		static void f_sum(const int *offs, double *x) {*x = *(x + offs[0]) + *(x + offs[1]);};
		static void f_wpi(const int *offs, double *x) {double d = *(x + offs[2]) - *(x + offs[3]);
													   if (d != 0) *x = (*(x + offs[0]) + *(x + offs[1]))/d; else *x = 0;};		// args <-> WOPR, WWPR, WBP9, WBHP
	};

	class T_sec_obj								// class for handling the secondary (calculated) objects
	{											// these secondary objects normally do summation (possibly with weighting) of the primary objects
	public:										// summation should only be applied to the primary properties!
		std::string name;
		std::vector<int> offsets;	// offsets for the subordinate objects

		void func(double *x, int flag, int wght_offset) const;	// HIGHLIGHT: calculation of *x based on the other data from array 'x'
																// 'flag' and 'wght_offset' should be taken from T_ecl_prop_transform
		// NB! primary and secondary objects are assumed to be arranged in consecutive chunks: [primary | secondary]
		// in the two CTORs below 'ind' is the index of the secondary object in the [secondary] chunk, i.e. 0, 1, 2,...
		T_sec_obj(int ind, int obj_N);			// takes name = 'FIELD' and assumes that all (obj_N) primary objects will be summed
		T_sec_obj(int ind, std::string Name, const std::vector<std::string> &subord, const std::vector<std::string> &full);	// subordinate objects are found in the full list of primary objects to fill 'offsets'
		std::string ToString() const;			// for debug output
	};

	std::string last_header;					// the header found by read_meta_block() before it exited
	std::ifstream *file_meta;
	std::map<std::string, std::string> ecl_tnav;				// contains pairs like <WBHP, WELL_CALC_BHP>; this map is filled in CTOR and should not be modified
	std::vector<T_ecl_prop_transform> ecl_prop_transform_full;	// full table of available transforms; filled in CTOR and should not be modified
	std::vector<int> ind_dates;					// indices of dates (timesteps) as defined in *.meta; increasing order
	std::vector<double> cumul_days;				// cumulative days since 'dates[0]', filled in read_meta()
	std::vector<int> ind_obj;					// indices of primary objects (wells) as defined in *.meta; increasing order
	std::vector<int> ecl_prop_ind;				// indices from *.meta [properties], corresponding to ecl_prop_transform[.] items; increasing order
	std::vector<T_ecl_prop_transform> ecl_prop_transform;		// the transforms from 'ecl_prop_transform_full' which are present in [properties] (i.e. not -1)
																// eclipse names (WBHP etc) are used here

	std::vector<SecObj> sec_objects_raw;		// secondary objects will be formed from this; 'FIELD' is added on top
	std::vector<T_sec_obj> sec_objects;			// size = sec_obj_N, filled in read_meta()
	int prop_N;					// number of entries in [properties] (incl. -1); filled by read_meta(); used for consistency checks
	int obj_N;					// number of primary objects, filled in read_meta()
	int sec_obj_N;				// number of secondary objects (sums of primary objects), filled in CTOR
	Date start;					// start date for WEFAC calculation

	void ecl_prop_transform_check() const;		// consistency check
	static int find_ind(const std::vector<T_ecl_prop_transform> &V, std::string name);			// returns index in V of element with "name"; -1 if not found
	void read_meta_block(const std::string header, std::vector<std::string> &Str, std::vector<int> &Int);		// reads an open 'file_meta' filling arrays 'Str', 'Int'
								// from lines "Str[i] = Int[i]", starting from header 'header', to the next found header "[...]", saving the latter as 'last_header' (in case of EOF, "" is saved)
	static bool is_header(std::string s);		// returns true if 's' is a header of format "[...]"
	void open_meta(std::string fname);			// opens 'file_meta'
	void close_meta();							// closes 'file_meta'

protected:
	void read_meta(std::string modname);	// reads "modname_well.meta"; fills 'vecs', 'dates', 'ind_dates', 'cumul_days', 'ind_obj', 'ecl_prop_ind', 'ecl_prop_transform', 'sec_objects', 'prop_N', 'obj_N'
	void read_res(std::string modname);		// reads data from "modname_well.res"; makes consistency checks and fills 'Data'

public:
	tNavSMRY(std::vector<SecObj> secobj, Date s);
	tNavSMRY(const tNavSMRY &p) {*this = p;};
	virtual ~tNavSMRY();
	const tNavSMRY &operator=(const tNavSMRY &p);
	virtual SimSMRY *Copy() const;						// _DELETE_ in the end!
	virtual void ReadFiles(std::string modname);		// reads summary from "modname_well.meta", "modname_well.res"
	void dump_all(std::string fname) const;				// dump contents to ASCII file (for debug)
	virtual std::string dates_file() const;				// name of file with dates
	virtual std::string vecs_file() const;				// name of file with vecs
	virtual std::string data_file() const;				// name of file with data
};
//--------------------------------------------------------------------------------------------------
// Class for storing model parameters and corresponding data values (for multiple proxies on data points)
// To import data: LoadFromBinary(), and/or AddModel()
// To export data: SaveToBinary(), SaveToAscii(), MakeProxy()
// NOTE that the parameter values stored here are external representation
// All functions should be called on all ranks (although some of them internally will work only on comm-RANKS-0)
//--------------------------------------------------------------------------------------------------
class SimProxyFile
{
private:
	const double stamp[4] = {exp(1), acos(-1), 1, 0};		// e, п, 1, 0 - for data stamp
	void stamp_file(FILE *fd) const;						// write data stamp to a binary file to distinguish a valid file for SimProxyFile
	int check_stamp(FILE *fd) const;						// check if data stamp is valid; returns: 0 - invalid stamp, 1 - empty file, 2 - valid non-empty file
	std::string msg_contents() const;						// message reporting what is stored; should be called on all ranks

	mutable PMEclipse *Ecl;									// these objects are not copied by copy ctor
	mutable PM_SimProxy *SimProxy;
	mutable HMMPI::BlockDiagMat *BDC;
	const ParamsTransform *par_tran;						// used for transforming parameters in AddModel() for model repeat checking
													// two vectors of size 'len' filled by extract_proxy_vals(); sync on "comm"
	mutable std::vector<int> datapoint_block;		// index of the first block where the given data point exists
	mutable std::vector<int> datapoint_modcount;	// number of models in which the given data point exists

	std::vector<std::vector<double>> extract_proxy_vals(const std::vector<Date> &dates, const std::vector<SimSMRY::pair> &vecs, const std::vector<double> &all_sigmas, const std::vector<int> &bs) const;
													// extracts values for proxies (proxies inside data proxy)
													// returns vector of size 'len' (= number of non-zeros in 'all_sigmas'), which corresponds to taking 'len' particular columns (data points) from "this->data"
													// also fills 'datapoint_block', 'datapoint_modcount'
													// all_sigmas = dates.size() x vecs.size() - all sigmas, including zeros; ordered as: vec_0(all dates), vec_1(all dates), ...
													// "bs" should come from block_starts()
													// all input and output parameters are only used on comm-RANKS-0
	std::vector<int> block_starts() const;	// returns Nblock+1 array of indices in [0, Np) showing where each block starts
											// the result is sync on all ranks
	std::vector<std::vector<double>> get_internal_parameters(const KW_parameters *par) const;	// reorders 'params' such that their 'par_names' follow KW_parameters order, and converts to the internal representation (output - comm-RANKS-0)
protected:
	MPI_Comm comm;									// should be the same comm as in PM_SimProxy
	int Rank;
													// All the data below are stored on comm-RANKS-0 only
	std::vector<int> block_ind;						// Np-length array, block_ind[i] stores index of block to which model "i" corresponds, Np - number of design points (models)

	std::vector<std::string> par_names;				// sorted parameter names (names as in KW_parameters)
	std::vector<std::vector<double>> params;		// Np x full_dim array of [external] parameter values, Np - number of design points (models)

	std::vector<std::vector<Date>> data_dates;				// data_dates[block][d] - sorted list of dates for "block", 0 <= d < Nd(block), Nd(block+1) >= Nd(block)
	std::vector<std::vector<SimSMRY::pair>> data_vecs;		// data_vecs[block][v] - sorted list of eclipse vectors for "block", 0 <= v < Nv(block), Nv(block+1) >= Nv(block)
															// 0 <= block < Nblock
	std::vector<std::vector<double>> data;			// Np x ... data values, inner vectors can have different length (they store 1st date + all vecs, 2nd date + all vecs, ...)



public:
	double Xtol;									// if |xnew - xi| < Xtol, and dates & vecs lists for 'i' and 'new' are the same, the new model will not be added
	double Xmin, Xavg;								// after AddModel, distances |xnew - xi| are calculated, and their min and mean are saved here (only RANKS-0)

	SimProxyFile(MPI_Comm c) : Ecl(0), SimProxy(0), BDC(0), par_tran(0), comm(c), Xtol(0.0), Xmin(0.0), Xavg(0.0)
					{MPI_Comm_rank(c, &Rank);};		// "c" will be used as PM_SimProxy->comm; avoid using MPI_COMM_NULL!
	SimProxyFile(const SimProxyFile &s) = delete;
	~SimProxyFile();
	std::string AddModel(const std::vector<std::string> &pname, const std::vector<double> &pval, const std::vector<std::string> &backval, const SimSMRY *smry);		// imports the model (parameter names, _EXTERNAL_ parameter values, eclipse summary)
															// "backval" [_EXTERNAL_/parameter_name] is from KW_parameters, lengths of "pname", "pval", "backval" should be the same
															// "pname" should be the superset of 'par_names'; "pname" should contain unique elements
															// smry.dates, smry.vecs should be supersets of last block of 'data_dates', 'data_vecs'; smry.dates, smry.vecs should contain unique elements
															// all input and "output" is only referenced on comm-RANKS-0
															// the function returns a non-empty message if the added model was popped out (NOTE: even in this case 'par_names' get updated)
															// the function also fills Xmin, Xavg
	std::string AddSimProxyFile(const SimProxyFile *smry_0);	// appends the proxy file 'smry_0' to 'this'
																// currently, both proxy files should contain 1 block, and have the same parameters names, same dates and vecs
																// all input and "output" is only referenced on comm-RANKS-0
																// models from 'smry_0' which are too close to the existing models, are skipped (similar to AddModel())
																// the function returns a message counting the added/skipped models
																// Xmin, Xavg are not updated

	void PopModel();								// pops the last added model; *NOTE* 'par_names' is not changed!; should be called on all ranks
	void SaveToBinary(const std::string &fname) const;			// saves to binary file (only comm-RANKS-0 is working, but call it on all ranks)
	std::string LoadFromBinary(const std::string &fname);		// loads from binary file, returns a short message; only comm-RANKS-0 is working, but "fname" should be sync on all ranks
	void SaveToAscii(FILE *fd) const;							// saves to ASCII file (only comm-RANKS-0 is working)
	PM_SimProxy *MakeProxy(const std::vector<Date> &dates, const std::vector<SimSMRY::pair> &vecs, const std::vector<double> &all_sigmas, Parser_1 *K, KW_item *kw, std::string cwd) const;		// creates a proxy model which resides on 'comm'
																// inputs 'dates', 'vecs', 'all_sigmas' (only comm-RANKS-0) are as in extract_proxy_vals()
																// inputs  'K', 'kw', 'cwd' (all comm-RANKS) are as in ModelFactory::Make
																// K->KW_parameters is used to transfer external "params" to the internal representation used by proxy
																// the returned model is freed automatically when necessary
	void ViewSmry(const std::string &fname, const std::vector<Date> &dates, const std::vector<SimSMRY::pair> &vecs, bool plain_order, Parser_1 *K) const;	// Writes data for the selected data points (dates x vecs) to ASCII file 'fname'
															// If 'plain_order' = true, models are listed in their direct order, otherwise they are reordered starting from the densely spaced models to more stand-alone models
															// K->KW_parameters is used to transfer external "params" to the internal representation used in the models ordering mentioned above
															// All input parameters are only used on comm-RANKS-0
															// This function is to be used for monitoring purposes

	std::string models_params_msg() const;			// message about number of models and parameters (comm-RANKS-0 only)
	void set_par_tran(const ParamsTransform *pt) {par_tran = pt;};
	int total_models() const {return block_ind.size();};	// current number of models in SimProxyFile
};
//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
}	// namespace HMMPI


#endif /* ECLSMRY_H_ */
