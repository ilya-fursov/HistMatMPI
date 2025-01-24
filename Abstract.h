
#ifndef ABSTRACT_H_
#define ABSTRACT_H_

//---------------------------------------------------------------------------
// some "defines", used mostly for debug purposes
//---------------------------------------------------------------------------

//#define WRITE_OBJ_FUNC
#define REGRESS_WITH_CONSTR
#define WRITE_RESAMPLES

//#define WRITE_LINREGRESS_FILES	//!
#define WRITE_WELLCOVAR_FILES	//!
#define WRITE_REGENTRYCONSTR	//!

#define ERROR_TO_FILE
//#define TESTCTOR
#define WRITE_PET_DATA			// write perturbed data for LIN, DATAPROXY, ECLIPSE
#define WRITE_PLOT_DATA			// write [y | X] to file in RUNPLOT
//#define WRITE_PROXY_VALS		// write proxy values before and after updates to PM_Proxy::dump_vals -- used mostly for debug and analysis
//#define PROXY_CAP_DROP			// enables ApplyValuesCap/ApplyValuesDrop; be careful: if this code was disabled for a long time, it may have become inconsistent with the surrounding code

#define obj_func_file "obj_func.txt"
#define resamples_file "resamples.txt"
#define lin_constr_file "lin_constr.txt"

//#define TEST_CACHE "Test_cache_%d.txt"

#include "mpi.h"
#include <vector>
#include <string>
#include <functional>

//---------------------------------------------------------------------------
// this header file declares small abstract classes which are used
// elsewhere to derive other concrete classes
//---------------------------------------------------------------------------

namespace HMMPI
{

//------------------------------------------------------------------------------------------
// base class for exceptions
//------------------------------------------------------------------------------------------
class ExceptionBase : public std::exception
{
protected:
	std::string msg;

public:
	ExceptionBase(std::string s) : msg(s) {};
	virtual ~ExceptionBase() noexcept {};
	const char* what() const noexcept { return msg.c_str(); };
};

//---------------------------------------------------------------------------
// descendants of ManagedObject created (and committed) within ModelFactory will be automatically destroyed
//---------------------------------------------------------------------------
class ManagedObject
{
public:
	virtual ~ManagedObject(){};
};
//------------------------------------------------------------------------------------------
// 'Constraints' - base class for checking constraints of a parameter vector, and working with constraints in other ways
//------------------------------------------------------------------------------------------
class Constraints
{
public:
	virtual ~Constraints(){};
	virtual std::string Check(const std::vector<double> &p) const = 0;					// "", if all constraints are satisfied for 'p'; or a message, if not
	virtual bool FindIntersect(const std::vector<double> &x0, const std::vector<double> &x1, std::vector<double> &xint, double &alpha, int &i) const = 0;		// 'true' if all constraints are satisfied for 'x1'
																						// otherwise returns 'false', in which case it finds the first intersection of x0 + t*(x1-x0) with the boundary;
																						// the intersection point is output as 'xint', corresponding t - as 'alpha', index of the constraint which was hit - as 'i'
																						// (it is supposed that all constraints are satisfied for 'x0')
};
//------------------------------------------------------------------------------------------
// 'CovAndDataCreator' - base class for the classes below
// 'CorrelCreator' - base class for creating array of correlation matrices for covariance "matrix" BDC
// 'StdCreator' - base class for creating vector of std's (for BDC)
// 'DataCreator' - base class for creating vector of observed data
///------------------------------------------------------------------------------------------
class Mat;
class CovAndDataCreator
{
public:
	virtual ~CovAndDataCreator(){};
};
///------------------------------------------------------------------------------------------
class CorrelCreator : public virtual CovAndDataCreator
{
public:
	virtual std::vector<Mat> CorrBlocks() const = 0;	// each Block[i] is a square {N x N} correlation matrix; if Block[i] is diagonal matrix, only its diagonal may be stored as {N x 1} matrix
};
//------------------------------------------------------------------------------------------
class StdCreator : public virtual CovAndDataCreator
{
public:
	virtual std::vector<double> Std() const = 0;
};
//------------------------------------------------------------------------------------------
class DataCreator : public virtual CovAndDataCreator
{
public:
	virtual std::vector<double> Data() const = 0;
};
//------------------------------------------------------------------------------------------
// 'SigmaMessage' - base class for getting information about sigmas of Eclipse vectors
//------------------------------------------------------------------------------------------
class SigmaMessage
{
public:
	virtual ~SigmaMessage(){};
	virtual std::string SigmaInfo(const std::string &wgname, const std::string &keyword) const = 0;		// returns info about sigma for <wgname, keyword>
};
//------------------------------------------------------------------------------------------
// 'ParamsTransform' - base class for transforming parameters between internal and external representations
//------------------------------------------------------------------------------------------
class ParamsTransform
{
public:
	virtual ~ParamsTransform(){};
	virtual std::vector<double> InternalToExternal(const std::vector<double> &in) const = 0;
	virtual std::vector<double> ExternalToInternal(const std::vector<double> &ex) const = 0;
	virtual std::vector<double> dxe_To_dxi(const std::vector<double> &dxe, const std::vector<double> &in) const = 0;		// transform gradient d/dxe -> d/dxi
};
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
void MsgToFileApp(const std::string &msg);						// output to TEST_CACHE file
//------------------------------------------------------------------------------------------
// 'Cache' - this class allows caching and reusing results of some calculation MyObj.Func(x)
// NOTE if state of MyObj changes, the stored cache may become irrelevant; in such situations do Reset()
//------------------------------------------------------------------------------------------
template <class Caller, class InType, class OutType>
class Cache
{
private:
	std::function<OutType(const Caller*, InType)> F;
	mutable OutType data;
	mutable InType last_x;
	mutable bool cache_valid;

public:
	Cache() : F(), cache_valid(false) {};															// empty ctor
	Cache(std::function<OutType(const Caller*, InType)> func) : F(func), cache_valid(false) {};		// here use something like &MyClass::Func
	const OutType &Get(const Caller *obj, const InType &x) const;									// here use something like (&MyObj, params) to call MyObj.Func(params)
	void Reset() {cache_valid = false;};
	void MsgToFile(const std::string &msg) const {MsgToFileApp(msg);};								// output to TEST_CACHE file
};
//------------------------------------------------------------------------------------------
template <class Caller, class InType, class OutType>
const OutType &Cache<Caller, InType, OutType>::Get(const Caller *obj, const InType &x) const
{
	if (x != last_x)
		cache_valid = false;
	if (!cache_valid)
	{
		if (!F)
			throw ExceptionBase("Calling empty target in Cache::Get");

		data = F(obj, x);
		last_x = x;
		cache_valid = true;

		MsgToFileApp(".....recalculating.....\n");
	}
	else
		MsgToFileApp("*****USING_CACHE!******\n");

	return data;
}
//------------------------------------------------------------------------------------------
} // namespace HMMPI


#endif /* ABSTRACT_H_ */
