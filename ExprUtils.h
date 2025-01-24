/*
 * ExprUtils.h
 *
 *  Created on: 06.11.2024
 *      Author: ilya fursov
 */

#ifndef EXPRUTILS_H_
#define EXPRUTILS_H_

#include "Abstract.h"
#include "Utils.h"
#include "EclSMRY.h"
#include <map>
#include <set>
#include <cassert>
#include <cmath>

namespace HMMPI
{
extern int count_val_Ctors;	// counts for debug purposes
extern int count_val_Dtors;

// Parse the printf format (that follows '%') into the respective parts
void parse_printf_fmt(const std::string &fmt, std::string &flags, std::string &width, std::string &prec, std::string &len, std::string &spec);

//------------------------------------------------------------------------------------------
// Classes and functions for formatted writing of
// strings with tags/expressions
//------------------------------------------------------------------------------------------

class ValBase					// Base class for wrappers Val<> of int, double, string, op[erator]
{
protected:
	static char buff[BUFFSIZE];
	const static inline std::vector<std::string> types = {"int", "double", "string", "operator"};
	static Date start_date;		// the start date and its format are used in date() operation

	static int Arity(std::string s);
	static std::string apply_width_to_format(std::string fmt, size_t &width); // Try to add 'width' to 'fmt', return the resulting 'fmt':
		// If 'width' > 0 and 'fmt' does not specify the width, then the 'width' provided is pushed to 'fmt', and '-' flag is added.
		// Otherwise 'fmt' is intact, and 'width' = 0 is set for output.

public:
	const int data_type;		// 0 - int, 1 - double, 2 - string (both data-string and op-string)
	const int arity;			// 0 - data, 1 - unary op, 2 - binary op
								// The operator strings are: + - * / ^ neg exp log date

	ValBase(int type, int ar) : data_type(type), arity(ar) {count_val_Ctors++;};
	ValBase(const ValBase &v) : data_type(v.data_type), arity(v.arity) {count_val_Ctors++;};
	virtual ~ValBase(){count_val_Dtors++;};
	static void set_start_date(Date D) {start_date = D;};

	virtual const ValBase *Copy() const = 0;					// For the following functions returning ValBase*, the result should be deleted by the caller
	virtual const ValBase *ToDouble() const = 0;				// promote to Val<double>
	virtual const ValBase *add(const ValBase *x) const = 0;		// *this + *x, data_type's should be the same
	virtual const ValBase *subtr(const ValBase *x) const = 0;	// *this - *x, data_type's should be the same
	virtual const ValBase *mult(const ValBase *x) const = 0;	// *this * *x, data_type's should be the same
	virtual const ValBase *div(const ValBase *x) const = 0;		// *this / *x, data_type's should be the same
	virtual const ValBase *pow(const ValBase *x) const = 0;		// *this ^ *x, data_type's should be the same
	virtual const ValBase *neg() const = 0;						// -(*this)
	virtual const ValBase *exp() const = 0;						// exp(*this), only for Val<double>
	virtual const ValBase *log() const = 0;						// log(*this), only for Val<double>, the natural logarithm
	virtual const ValBase *date() const = 0;					// date(*this), only for Val<double>, returns string

	std::string get_type() const {return types[arity ? 3 : data_type];};
	virtual std::string get_op_type() const = 0;		// returns 'Val<string>::val' for operators, "" otherwise
	virtual std::string ToString(std::string fmt, size_t &width) const = 0;	// Essentially, sprintf(fmt, this->val).
											// For 'fmt' = "", a format is taken which depends on Val<T>.
											// Besides, 'fmt' may be adjusted by 'width' as described in apply_width_to_format().
	std::string ToString(std::string fmt = "") const;	// Wrapper for the above function, does not handle the width.
};
//------------------------------------------------------------------------------------------
template <class T>
class Val : public ValBase							// T is expected to be: int, double, std::string
{
protected:
	T val;

	std::string get_fmt() const;					// format string to use by default; this function is 100% specialized for int, double, std::string
public:
	Val(T x);
	Val(const Val &v);

	virtual const ValBase *Copy() const;					// For the following functions returning ValBase*, the result should be deleted by the caller
	virtual const ValBase *ToDouble() const;				// promote to Val<double>
	virtual const ValBase *add(const ValBase *x) const;		// *this + *x, data_type's should be the same
	virtual const ValBase *subtr(const ValBase *x) const;	// *this - *x, data_type's should be the same
	virtual const ValBase *mult(const ValBase *x) const;	// *this * *x, data_type's should be the same
	virtual const ValBase *div(const ValBase *x) const;		// *this / *x, data_type's should be the same
	virtual const ValBase *pow(const ValBase *x) const;		// *this ^ *x, data_type's should be the same
	virtual const ValBase *neg() const;						// -(*this)
	virtual const ValBase *exp() const;						// exp(*this), only for Val<double>
	virtual const ValBase *log() const;						// log(*this), only for Val<double>, the natural logarithm
	virtual const ValBase *date() const;					// date(*this), only for Val<double>, returns string

	virtual std::string get_op_type() const;		// returns 'Val<string>::val' for operators, "" otherwise
	const T &get_val() const {return val;};
	virtual std::string ToString(std::string fmt, size_t &width) const;
};
//------------------------------------------------------------------------------------------
// Class that stores pairs <tag, value>, to be used in formatted output;
// essentially it's an std::map with some additional handy routines
class TagValMap : public std::map<std::string, ValBase*>
{
public:
	TagValMap();											// adds tags: MOD, PATH, RANK, SIZE, SMPL; for RANK, SIZE also sets values (global MPI rank and size)
	TagValMap(const std::vector<std::string> &tags, const std::vector<double> &vals);	// apart from 5 default tags, adds "tags" with "vals"
	TagValMap(const TagValMap &M) = delete;					// no copies so far
	const TagValMap &operator=(const TagValMap &M) = delete;
	~TagValMap();											// frees "vals"
	void SetModPath(std::string mod, std::string path);		// sets values for MOD and PATH tags
	void SetSize(int size);									// sets value for SIZE tag
	void SetSmpl(int smpl);									// sets value for SMPL tag
	void SetDoubles(const std::vector<std::string> &tags, const std::vector<double> &vals);	// sets "vals" for "tags", where "tags" is a subset of {this->first}
	std::set<std::string> get_tag_names() const;			// returns the set of all tag names (except MOD, PATH, RANK, SIZE, SMPL)
};
//------------------------------------------------------------------------------------------
std::vector<std::string> StringToInfix(const std::string &expr);		// Parses 'expr' to fill an infix expression stored as vector. Unary 'plus' is replaced by "", unary 'minus' is saved as "neg".
std::vector<const ValBase*> InfixToPostfix(const std::vector<std::string> &infix, const std::map<std::string, ValBase*> &tag_val, int &count, std::set<std::string> &tags_left,
										   const std::string &orig_expr, const std::string &comment = "", const std::string &msg_params = "");
			// Creates a postfix expression (values + operators). The output vector stores pointers which should be deleted by the caller.
			// 'tag_val' is used to substitute the variable values, updating the substitution 'count' and 'tags_left'.
			// 'orig_expr' is the original expression, to use in the error message.
			// 'comment' is an additional comment regarding the expression (e.g. its location).
			// 'msg_params' is an additional message regarding the parameters, to use in the error message.
const ValBase *CalcUnary(const ValBase *op, const ValBase *x, const std::string &orig_expr);					// Calculates op(x), creating a new object (to be deleted by the caller)
const ValBase *CalcBinary(const ValBase *op, const ValBase *x, const ValBase *y, const std::string &orig_expr);	// Calculates op(x, y), creating a new object (to be deleted by the caller)
const ValBase *CalcPostfix(const std::vector<const ValBase*> &expr, const std::string &orig_expr, const std::string &comment = "");	// Calculates postfix expression 'expr',
			// returns a new object (to be deleted by the caller). All the items (pointers) in 'expr' are deleted.
			// 'orig_expr' is the original expression, to use in the error message.
			// 'comment' is an additional comment regarding the expression (e.g. its location).
//------------------------------------------------------------------------------------------
std::string stringTagPrintf(const std::string &input_text, const std::map<std::string, ValBase*> &tag_val, int &count, std::set<std::string> &tags_left);
			// Writes values corresponding to 'tags' in "input_text". The 'tag' locations may be of the form $tag, $tag%fmt (e.g. format %fmt = %20.16g).
			// 'tag' is an arithmetic expression consisting of parameter names, numbers, operators +-*/^(), exp, log, date.
			// When one simply specifies: $tag, the default format is applied depending on the output value type of the expression.
			// The end of a tag location is marked by whitespace (excluded from the tag substring), or semicolon ";" (gets attached to tag, and then rejected).
			// If the format does not specify width ("", "%g", "%.10g") and the 'tag' is followed by many plain spaces marking the column width (space ASCII code = 32),
			// the format will be adjusted to preserve the column width, possibly by consuming/adding some spaces.
			// TagValMap object can be conveniently used as "tag_val", to provide the parameter values.
			// Output "count" shows how many parameter names were replaced.
			// Parameter names encountered in "input_text" are removed from the "tags_left" set.

std::vector<std::string> stringExtractTags(const std::string &input_text);	// Returns array of tags/expressions (without "$" and "%fmt" ) found in "input_text".
																			// Useful to check what expressions are present in string "input_text".
//------------------------------------------------------------------------------------------
// BELOW ARE THE TEMPLATE DEFINITIONS ******************************************************
//------------------------------------------------------------------------------------------
// class Val<T>
//------------------------------------------------------------------------------------------
template <class T>
Val<T>::Val(const Val &v) : ValBase(v), val(v.val)
{
}
//------------------------------------------------------------------------------------------
template <class T>
const ValBase *Val<T>::Copy() const
{
	return new Val<T>(*this);
}
//------------------------------------------------------------------------------------------
template <class T>
const ValBase *Val<T>::ToDouble() const					// only works for Val<int>, promotes it to Val<double>
{
	throw EObjFunc(stringFormatArr("Call Val<T>::ToDouble() is illegal for type '{0:%s}'", get_type()));
}
//------------------------------------------------------------------------------------------
template <class T>
const ValBase *Val<T>::add(const ValBase *x) const		// *this + *x, data_type's should be the same
{
	const Val<T> *xwork = dynamic_cast<const Val<T>*>(x);

	assert(data_type == x->data_type);
	assert(data_type <= 2);
	assert(xwork);

	return new Val<T>(val + xwork->val);
}
//------------------------------------------------------------------------------------------
template <class T>
const ValBase *Val<T>::subtr(const ValBase *x) const	// *this - *x, data_type's should be the same
{
	const Val<T> *xwork = dynamic_cast<const Val<T>*>(x);

	assert(data_type == x->data_type);
	assert(data_type <= 1);
	assert(xwork);

	return new Val<T>(val - xwork->val);
}
template <>
const ValBase *Val<std::string>::subtr(const ValBase *x) const;
//------------------------------------------------------------------------------------------
template <class T>
const ValBase *Val<T>::mult(const ValBase *x) const		// *this * *x, data_type's should be the same
{
	const Val<T> *xwork = dynamic_cast<const Val<T>*>(x);

	assert(data_type == x->data_type);
	assert(data_type <= 1);
	assert(xwork);

	return new Val<T>(val * xwork->val);
}
template <>
const ValBase *Val<std::string>::mult(const ValBase *x) const;
//------------------------------------------------------------------------------------------
template <class T>
const ValBase *Val<T>::div(const ValBase *x) const		// *this / *x, data_type's should be the same
{
	const Val<T> *xwork = dynamic_cast<const Val<T>*>(x);

	assert(data_type == x->data_type);
	assert(data_type <= 1);
	assert(xwork);

	return new Val<T>(val / xwork->val);
}
template <>
const ValBase *Val<std::string>::div(const ValBase *x) const;
//------------------------------------------------------------------------------------------
template <class T>
const ValBase *Val<T>::pow(const ValBase *x) const		// *this ^ *x, data_type's should be the same
{
	assert(data_type == x->data_type);
	throw EObjFunc(stringFormatArr("Call Val<T>::pow() is illegal for type '{0:%s}'", get_type()));
}
//------------------------------------------------------------------------------------------
template <class T>
const ValBase *Val<T>::neg() const						// -(*this)
{
	assert(data_type <= 1);
	return new Val<T>(-val);
}
template <>
const ValBase *Val<std::string>::neg() const;
//------------------------------------------------------------------------------------------
template <class T>
const ValBase *Val<T>::exp() const						// exp(*this), only for Val<double>
{
	throw EObjFunc(stringFormatArr("Call Val<T>::exp() is illegal for type '{0:%s}'", get_type()));
}
//------------------------------------------------------------------------------------------
template <class T>
const ValBase *Val<T>::log() const						// log(*this), only for Val<double>, the natural logarithm
{
	throw EObjFunc(stringFormatArr("Call Val<T>::log() is illegal for type '{0:%s}'", get_type()));
}
//------------------------------------------------------------------------------------------
template <class T>
const ValBase *Val<T>::date() const						// date(*this), only for Val<double>, returns string
{
	throw EObjFunc(stringFormatArr("Call Val<T>::date() is illegal for type '{0:%s}'", get_type()));
}
//------------------------------------------------------------------------------------------
template <class T>
std::string Val<T>::get_op_type() const					// returns 'Val<string>::val' for operators, "" otherwise
{
	return "";
}
template <>
std::string Val<std::string>::get_op_type() const;
//------------------------------------------------------------------------------------------
template <class T>														// Essentially, sprintf(fmt, this->val).
std::string Val<T>::ToString(std::string fmt, size_t &width) const		// For 'fmt' = "", a format is taken which depends on Val<T>.
{																		// Besides, 'fmt' may be adjusted by 'width' as described in apply_width_to_format().
	if (fmt == "")
		fmt = get_fmt();
	fmt = apply_width_to_format(fmt, width);

	int n = sprintf(buff, fmt.c_str(), val);
	if (n < 0 || n >= BUFFSIZE) {
		throw EObjFunc("Ошибка форматированной записи в Val<T>::ToString",
					   "Formatted output not successful in Val<T>::ToString");
	}
	return buff;
}
template <>
std::string Val<std::string>::ToString(std::string fmt, size_t &width) const;		// for T = std::string a specialization is used
//------------------------------------------------------------------------------------------
}	// namespace HMMPI
//------------------------------------------------------------------------------------------
#endif /* EXPRUTILS_H_ */
