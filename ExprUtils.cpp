/*
 * ExprUtils.cpp
 *
 *  Created on: 06.11.2024
 *      Author: ilya fursov
 */

#include "ExprUtils.h"
#include <stack>
#include <algorithm>

namespace HMMPI
{
int count_val_Ctors;	// counts for debug purposes
int count_val_Dtors;
char ValBase::buff[BUFFSIZE];
Date ValBase::start_date;

//------------------------------------------------------------------------------------------
// Check if the format 'fmt' contains not-handled symbols, starting from 'pos'
static void check_fmt_symbols(const std::string &fmt, size_t pos = 0)
{
	const static std::string symb_not_handled = "n*%"; 			// these symbols are not currently handled

	if (fmt.find_first_of(symb_not_handled, pos) != std::string::npos)
		throw EObjFunc(stringFormatArr("Format '{0:%s}' is not acceptable, symbols '{1:%s}' are not currently handled",
									   std::vector<std::string>{fmt, symb_not_handled}));
}
//------------------------------------------------------------------------------------------
// Parse the printf format (that follows '%') into the respective parts
void parse_printf_fmt(const std::string &fmt, std::string &flags, std::string &width, std::string &prec, std::string &len, std::string &spec)
{
	const static std::string msg  = "Format '{0:%s}' is not acceptable, the specifier is missing";
	const static std::string Flg  = "-+ #0";
	const static std::string Num  = "0123456789";
	const static std::string NumD = ".0123456789";
	const static std::string Len  = "hljztL";
	// specifiers: "iduoxXfFeEgGaAcsp";
	size_t pos1 = 0, pos2;

	check_fmt_symbols(fmt);

	// everything except specifier is optional
	pos2 = fmt.find_first_not_of(Flg, pos1);
	if (pos2 == std::string::npos) throw EObjFunc(stringFormatArr(msg, fmt));
	flags = std::string(fmt.data() + pos1, pos2 - pos1);
	pos1 = pos2;

	pos2 = fmt.find_first_not_of(Num, pos1);
	if (pos2 == std::string::npos) throw EObjFunc(stringFormatArr(msg, fmt));
	width = std::string(fmt.data() + pos1, pos2 - pos1);
	pos1 = pos2;

	pos2 = fmt.find_first_not_of(NumD, pos1);
	if (pos2 == std::string::npos) throw EObjFunc(stringFormatArr(msg, fmt));
	prec = std::string(fmt.data() + pos1, pos2 - pos1);
	pos1 = pos2;

	pos2 = fmt.find_first_not_of(Len, pos1);
	if (pos2 == std::string::npos) throw EObjFunc(stringFormatArr(msg, fmt));
	len = std::string(fmt.data() + pos1, pos2 - pos1);
	pos1 = pos2;

	spec = std::string(fmt.data() + pos1);
}
//------------------------------------------------------------------------------------------
// ValBase
//------------------------------------------------------------------------------------------
int ValBase::Arity(std::string s)	// The operator strings are: + - * / ^ neg exp log date
{
	if (s == "+" || s == "-" || s == "*" || s == "/" || s == "^") return 2;
	else if (s == "neg" || s == "exp" || s == "log" || s == "date") return 1;
	else return 0;					// data string
}
//------------------------------------------------------------------------------------------
std::string ValBase::apply_width_to_format(std::string fmt, size_t &width)	// Try to add 'width' to 'fmt', return the resulting 'fmt':
{			// If 'width' > 0 and 'fmt' does not specify the width, then the 'width' provided is pushed to 'fmt', and '-' flag is added.
	assert(fmt.size() > 0 && fmt[0] == '%');								// Otherwise 'fmt' is intact, and 'width' = 0 is set for output.

	if (width > 0) {
		std::string flg, wdth, prc, ln, spc;

		fmt = std::string(fmt.data() + 1);	// strip the leading '%'
		parse_printf_fmt(fmt, flg, wdth, prc, ln, spc);
		if (wdth.size() == 0) {	// fmt->width is not specified
			wdth = stringFormatArr("{0:%zu}", width);			// push the 'width' into the format
			if (flg.find_first_of("-") == std::string::npos) flg += "-";	// add '-' flag to left-justify
			fmt = flg + wdth + prc + ln + spc;
		} else width = 0;		// fmt->width is specified
		fmt = "%" + fmt;		// restore the leading '%'
	} else check_fmt_symbols(fmt, 1);

	return fmt;
}
//------------------------------------------------------------------------------------------
std::string ValBase::ToString(std::string fmt) const
{
	size_t width = 0;
	return ToString(fmt, width);
}
//------------------------------------------------------------------------------------------
// Val<T> specializations
//------------------------------------------------------------------------------------------
template <>
std::string Val<int>::get_fmt() const
{
	return "%d";
}
template <>
std::string Val<double>::get_fmt() const
{
	return "%g";
}
template <>
std::string Val<std::string>::get_fmt() const
{
	return "%s";
}
//------------------------------------------------------------------------------------------
template <>
Val<int>::Val(int x) : ValBase(0, 0), val(x)
{
}
template <>
Val<double>::Val(double x) : ValBase(1, 0), val(x)
{
}
template <>
Val<std::string>::Val(std::string x) : ValBase(2, Arity(x)), val(x)
{
}
//------------------------------------------------------------------------------------------
template <>
const ValBase *Val<int>::ToDouble() const			// Val<int> -> Val<double>
{
	return new Val<double>((double)val);
}
//------------------------------------------------------------------------------------------
template <>
const ValBase *Val<std::string>::subtr(const ValBase *x) const
{
	throw EObjFunc("Call Val<std::string>::subtr() is illegal");
}
//------------------------------------------------------------------------------------------
template <>
const ValBase *Val<std::string>::mult(const ValBase *x) const
{
	throw EObjFunc("Call Val<std::string>::mult() is illegal");
}
//------------------------------------------------------------------------------------------
template <>
const ValBase *Val<std::string>::div(const ValBase *x) const
{
	throw EObjFunc("Call Val<std::string>::div() is illegal");
}
//------------------------------------------------------------------------------------------
template <>
const ValBase *Val<int>::pow(const ValBase *x) const
{
	const Val<int> *xwork = dynamic_cast<const Val<int>*>(x);

	assert(data_type == x->data_type);
	assert(data_type == 0);
	assert(xwork);

	if (xwork->val < 0) throw EObjFunc("Val<int>::pow(x) doesn't work for x < 0");
	if (val == 0 && xwork->val == 0) throw EObjFunc("Cannot compute 0^0 in Val<int>::pow()");
	if (val == 0) return new Val<int>(0);					// 0^x for x > 0
	else {
		int res = 1;
		for (int i = 0; i < xwork->val; i++) res *= val;	// xwork->val >= 0
		return new Val<int>(res);
	}
}
template <>
const ValBase *Val<double>::pow(const ValBase *x) const
{
	const Val<double> *xwork = dynamic_cast<const Val<double>*>(x);

	assert(data_type == x->data_type);
	assert(data_type == 1);
	assert(xwork);

	return new Val<double>(::pow(val, xwork->val));
}
//------------------------------------------------------------------------------------------
template <>
const ValBase *Val<std::string>::neg() const
{
	throw EObjFunc("Call Val<std::string>::neg() is illegal");
}
//------------------------------------------------------------------------------------------
template <>
const ValBase *Val<double>::exp() const
{
	assert(data_type == 1);
	return new Val<double>(::exp(val));
}
//------------------------------------------------------------------------------------------
template <>
const ValBase *Val<double>::log() const
{
	assert(data_type == 1);
	return new Val<double>(::log(val));
}
//------------------------------------------------------------------------------------------
template <>
const ValBase *Val<double>::date() const
{
	assert(data_type == 1);
	Date D = start_date;
	D.add(val);
	return new Val<std::string>(D.ToString(D.get_fmt()));
}
//------------------------------------------------------------------------------------------
template <>
std::string Val<std::string>::get_op_type() const
{
	if (arity > 0) return val;
	else return "";
}
//------------------------------------------------------------------------------------------
template <>
std::string Val<std::string>::ToString(std::string fmt, size_t &width) const
{
	if (fmt == "")
		fmt = get_fmt();
	fmt = apply_width_to_format(fmt, width);

	char *dynbuff = new char[val.size() + BUFFSIZE];	// take some extra length; val = std::string
	int n = sprintf(dynbuff, fmt.c_str(), val.c_str());
	if (n < 0 || n >= (int)val.size() + BUFFSIZE) {
		delete [] dynbuff;
		throw EObjFunc("Ошибка форматированной записи в Val<string>::ToString",
					   "Formatted output not successful in Val<string>::ToString");
	}
	std::string res(dynbuff);
	delete [] dynbuff;
	return res;
}
//------------------------------------------------------------------------------------------
// class TagValMap
//------------------------------------------------------------------------------------------
TagValMap::TagValMap()
{
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	(*this)["MOD"] = new Val<std::string>("");
	(*this)["PATH"] = new Val<std::string>("");
	(*this)["RANK"] = new Val<int>(rank);
	(*this)["SIZE"] = new Val<int>(size);
	(*this)["SMPL"] = new Val<int>(-1);
}
//------------------------------------------------------------------------------------------
TagValMap::TagValMap(const std::vector<std::string> &tags, const std::vector<double> &vals) : TagValMap()	// calls default ctor
{
	const std::string op_symbol = "+-*/^()";		// these are not allowed in parameter names

	if (tags.size() != vals.size())
		throw EObjFunc("tags.size() != vals.size() in TagValMap ctor");

	for (size_t i = 0; i < tags.size(); i++) {
		const size_t aux = tags[i].find_first_of(op_symbol);
		if (aux != std::string::npos)
			throw EObjFunc(stringFormatArr("Недопустимый символ '{0:%s}' в имени параметра '",
										   "Inacceptable symbol '{0:%s}' in parameter name '", std::string(tags[i].data()+aux, 1)) + tags[i] + "'");
		iterator it = find(tags[i]);
		if (it != end())			// tag already exists
			throw EObjFunc("Повторное добавление тэга '" + tags[i] + "' в конструкторе TagValMap", "Duplicate tag '" + tags[i] + "' in TagValMap ctor");

		(*this)[tags[i]] = new Val<double>(vals[i]);
	}
}
//------------------------------------------------------------------------------------------
TagValMap::~TagValMap()
{
	for (auto &v : *this) {
		delete v.second;
		v.second = 0;
	}
}
//------------------------------------------------------------------------------------------
void TagValMap::SetModPath(std::string mod, std::string path)
{
	delete (*this)["MOD"];
	(*this)["MOD"] = new Val<std::string>(mod);

	delete (*this)["PATH"];
	(*this)["PATH"] = new Val<std::string>(path);
}
//------------------------------------------------------------------------------------------
void TagValMap::SetSize(int size)
{
	delete (*this)["SIZE"];
	(*this)["SIZE"] = new Val<int>(size);
}
//------------------------------------------------------------------------------------------
void TagValMap::SetSmpl(int smpl)
{
	delete (*this)["SMPL"];
	(*this)["SMPL"] = new Val<int>(smpl);
}
//------------------------------------------------------------------------------------------
void TagValMap::SetDoubles(const std::vector<std::string> &tags, const std::vector<double> &vals)	// sets "vals" for "tags", where "tags" is a subset of {this->first}
{
	if (tags.size() != vals.size())
		throw EObjFunc("tags.size() != vals.size() in TagValMap::SetDoubles");

	for (size_t i = 0; i < tags.size(); i++) {
		iterator it = find(tags[i]);
		if (it == end())			// tag does not exist
			throw EObjFunc("Не найден тэг '" + tags[i] + "' в TagValMap::SetDoubles", "Tag '" + tags[i] + "' was not found in TagValMap::SetDoubles");

		delete it->second;
		it->second = new Val<double>(vals[i]);
	}
}
//------------------------------------------------------------------------------------------
std::set<std::string> TagValMap::get_tag_names() const		// returns the set of all tag names (except MOD, PATH, RANK, SIZE, SMPL)
{
	std::vector<std::string> names;							// first, bring the names to the vector
	names.reserve(this->size());
	for (const auto &p : *this)
		names.push_back(p.first);

	std::set<std::string> res(names.begin(), names.end());	// second, create the set
	res.erase("MOD");										// third, erase the default tags
	res.erase("PATH");
	res.erase("RANK");
	res.erase("SIZE");
	res.erase("SMPL");

	return res;
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
std::vector<std::string> StringToInfix(const std::string &expr)		// Parses 'expr' to fill an infix expression stored as vector.
{																	// Unary 'plus' is replaced by "", unary 'minus' is saved as "neg".
	std::vector<std::string> res;
	static const std::string aux_unary = "+-*/^(";		// used for unary + - diagnostic
	static const std::string op        = "+-*/^()";
	std::string prev;

	tokenize(expr, res, op, true, true);
	for (size_t i = 0; i < res.size(); i++) {			// check the unary + -
		if ((res[i] == "+" || res[i] == "-") && (i == 0 || aux_unary.find(prev) != std::string::npos)) {
			prev = res[i];			// save for the next iteration
			if (res[i] == "+") res[i] = "";
			else res[i] = "neg";
		} else prev = res[i];		// save for the next iteration
	}
	return res;
}
//------------------------------------------------------------------------------------------
inline bool is_unary_op(const std::string &a)
{
	return a == "neg" || a == "exp" || a == "log" || a == "date";
}
//------------------------------------------------------------------------------------------
inline int op_prec(const std::string &a)	// operator's precedence
{
	if (is_unary_op(a)) return 4;
	else if (a == "^") return 3;
	else if (a == "*" || a == "/") return 2;
	else if (a == "+" || a == "-") return 1;
	else return 0;
}
//------------------------------------------------------------------------------------------
inline bool not_both_unary(const std::string &a, const std::string &b)
{
	return !is_unary_op(a) || !is_unary_op(b);
}
//------------------------------------------------------------------------------------------
std::vector<const ValBase*> InfixToPostfix(const std::vector<std::string> &infix, const std::map<std::string, ValBase*> &tag_val, int &count, std::set<std::string> &tags_left,
										   const std::string &orig_expr, const std::string &comment, const std::string &msg_params)
// Creates a postfix expression (values + operators). The output vector stores pointers which should be deleted by the caller.
{											// 'tag_val' is used to substitute the variable values, updating the substitution 'count' and 'tags_left'.
	std::stack<std::string> estack;			// 'orig_expr' is the original expression, to use in the error message.
	std::vector<std::string> res_str;		// 'comment' is an additional comment regarding the expression (e.g. its location).
	res_str.reserve(infix.size());			// 'msg_params' is an additional message regarding the parameters, to use in the error message.
	static const std::vector<std::string> ops = {"+", "-", "*", "/", "^", "neg", "exp", "log", "date"};

	// 1. Form the vector of strings 'res_str'
	for (const std::string &tok : infix) {
		if (tok == "") continue;					// unary '+'
		else if (tok == "(") estack.push(tok);
		else if (tok == ")") {
			bool found = false;
			while (estack.size() > 0) {
				std::string it = estack.top();
				estack.pop();
				if (it == "(") {
					found = true;
					break;
				}
				else res_str.push_back(it);
			}
			if (!found)
				throw EObjFunc(stringFormatArr(MessageRE("Синтаксическая ошибка, не хватает '(' в выражении '{0:%s}'{1:%s}",
											   	   	     "Syntax error, missing '(' in expression '{0:%s}'{1:%s}"),
														 std::vector<std::string>{orig_expr, comment}));
		}
		else if (std::find(ops.begin(), ops.end(), tok) != ops.end()) {			// process an operator
			while (estack.size() > 0 && op_prec(estack.top()) >= op_prec(tok) && not_both_unary(estack.top(), tok)) {
				res_str.push_back(estack.top());
				estack.pop();
			}
			estack.push(tok);
		} else res_str.push_back(tok);											// process a value/variable
	}
	while (estack.size() > 0) {						// pop the remaining 'estack' to 'res_str'
		res_str.push_back(estack.top());
		estack.pop();
	}

	// 2. Convert 'res_str' to the final output
	std::vector<const ValBase*> res;
	res.reserve(res_str.size());
	for (const std::string &it : res_str) {
		if (std::find(ops.begin(), ops.end(), it) != ops.end()) {
			res.push_back(new Val<std::string>(it));	// found operator -> add
			continue;
		}

		bool isnum = false;
		long L = StoL(it, isnum);
		if (isnum) {
			res.push_back(new Val<int>(L));				// found int -> add
			continue;
		}

		double D = StoD(it, isnum);
		if (isnum) {
			res.push_back(new Val<double>(D));			// found double -> add
			continue;
		}

		if (it == "(")
			throw EObjFunc(stringFormatArr(MessageRE("Синтаксическая ошибка, не хватает ')' в выражении '{0:%s}'{1:%s}",
										   	   	     "Syntax error, missing ')' in expression '{0:%s}'{1:%s}"),
													 std::vector<std::string>{orig_expr, comment}));

		auto var = tag_val.find(it);			// search for variable in the 'tag_val' list
		if (var == tag_val.end()) {				// variable not found!
			if (infix.size() == 1)				// expression = single parameter
				throw EObjFunc(stringFormatArr(MessageRE("Параметр '{0:%s}'{1:%s} не найден в списке параметров{2:%s}",
											   	   	     "Parameter '{0:%s}'{1:%s} was not found in the parameters list{2:%s}"),
														 std::vector<std::string>{it, comment, msg_params}));
			else
				throw EObjFunc(stringFormatArr(MessageRE("Входящий в выражение '{0:%s}' параметр '{1:%s}'{2:%s} не найден в списке параметров{3:%s}",
											   	   	   	 "Parameter '{1:%s}' from expression '{0:%s}'{2:%s} was not found in the parameters list{3:%s}"),
											   	   	   	 std::vector<std::string>{orig_expr, it, comment, msg_params}));
		}
		res.push_back(var->second->Copy());		// found variable -> add its value
		count++;
		tags_left.erase(it);
	}
	return res;
}
//------------------------------------------------------------------------------------------
const ValBase *CalcUnary(const ValBase *op, const ValBase *x, const std::string &orig_expr)		// Calculates op(x), creating a new object (to be deleted by the caller)
{
	assert(op->arity == 1);
	assert(op->data_type == 2);
	assert(x->arity == 0);
	std::string optype = op->get_op_type();

	if (x->data_type > 1)
		throw EObjFunc(stringFormatArr("Operation '{0:%s}' is illegal for data type '{1:%s}', in expression '{2:%s}'",
				       std::vector<std::string>{optype, x->get_type(), orig_expr}));

	if (optype == "neg") return x->neg();
	else {
		const ValBase *X1 = x, *Xdel = nullptr, *res = nullptr;
		if (x->data_type == 0) X1 = Xdel = x->ToDouble();			// promote x to double; to be deleted in the end!

		if (optype == "exp") res = X1->exp();
		else if (optype == "log") res = X1->log();
		else if (optype == "date") res = X1->date();
		else {
			delete Xdel;
			throw EObjFunc(stringFormatArr("Incorrect operation '{0:%s}' in CalcUnary()", optype));
		}
		delete Xdel;
		return res;
	}
}
//------------------------------------------------------------------------------------------
const ValBase *CalcBinary(const ValBase *op, const ValBase *x, const ValBase *y, const std::string &orig_expr)		// Calculates op(x, y), creating a new object (to be deleted by the caller)
{
	assert(op->arity == 2);
	assert(op->data_type == 2);
	assert(x->arity == 0 && y->arity == 0);
	std::string optype = op->get_op_type();

	const ValBase *X1 = x, *Xdel = nullptr, *res = nullptr;
	const ValBase *Y1 = y, *Ydel = nullptr;

	if (x->data_type == 0 && y->data_type == 1) X1 = Xdel = x->ToDouble();	// promote x to double; to be deleted in the end!
	if (x->data_type == 1 && y->data_type == 0) Y1 = Ydel = y->ToDouble();	// promote y to double; to be deleted in the end!

	if (X1->data_type != Y1->data_type)
		throw EObjFunc(stringFormatArr("Non-matching argument types '{0:%s}', '{1:%s}' for binary operation '{2:%s}', in expression '{3:%s}'",
					   std::vector<std::string>{X1->get_type(), Y1->get_type(), optype, orig_expr}));
	if (optype == "+") res = X1->add(Y1);
	else if (optype == "-" || optype == "*" || optype == "/" || optype == "^") {
		if (X1->data_type > 1)
			throw EObjFunc(stringFormatArr("Incorrect argument types '{0:%s}' for operation '{1:%s}', in expression '{2:%s}'",
						   std::vector<std::string>{X1->get_type(), optype, orig_expr}));
		if (optype == "-") res = X1->subtr(Y1);
		else if (optype == "*") res = X1->mult(Y1);
		else if (optype == "/") res = X1->div(Y1);
		else if (optype == "^") res = X1->pow(Y1);
	}
	else {
		delete Xdel;
		delete Ydel;
		throw EObjFunc(stringFormatArr("Incorrect operation '{0:%s}' in CalcBinary()", optype));
	}
	delete Xdel;
	delete Ydel;
	return res;
}
//------------------------------------------------------------------------------------------
const ValBase *CalcPostfix(const std::vector<const ValBase*> &expr, const std::string &orig_expr, const std::string &comment)	// Calculates postfix expression 'expr',
{										// returns a new object (to be deleted by the caller). All the items (pointers) in 'expr' are deleted.
	std::stack<const ValBase*> estack;	// 'orig_expr' is the original expression, to use in the error message.
										// 'comment' is an additional comment regarding the expression (e.g. its location).
	for (const ValBase *it : expr) {
		const ValBase *res = nullptr;

		if (it->arity == 0) estack.push(it);
		else if (it->arity == 1) {
			if (estack.size() < 1)
				throw EObjFunc(stringFormatArr(MessageRE(
					"Синтаксическая ошибка, не хватает аргумента(ов) в выражении '{1:%s}'{2:%s} // stack size < 1 on reaching unary operation '{0:%s}' in CalcPostfix()",
					"Syntax error, missing operand(s) in expression '{1:%s}'{2:%s} // stack size < 1 on reaching unary operation '{0:%s}' in CalcPostfix()"),
														 std::vector<std::string>{it->get_op_type(), orig_expr, comment}));
			const ValBase *X1 = estack.top();
			estack.pop();
			res = CalcUnary(it, X1, orig_expr);
			delete it;
			delete X1;
			estack.push(res);
		} else if (it->arity == 2) {
			if (estack.size() < 2)
				throw EObjFunc(stringFormatArr(MessageRE(
					"Синтаксическая ошибка, не хватает аргумента(ов) в выражении '{1:%s}'{2:%s} // stack size < 2 on reaching binary operation '{0:%s}' in CalcPostfix()",
					"Syntax error, missing operand(s) in expression '{1:%s}'{2:%s} // stack size < 2 on reaching binary operation '{0:%s}' in CalcPostfix()"),
											   	   	   	 std::vector<std::string>{it->get_op_type(), orig_expr, comment}));
			const ValBase *Y1 = estack.top();	// estack = {... X1 Y1}
			estack.pop();
			const ValBase *X1 = estack.top();
			estack.pop();
			res = CalcBinary(it, X1, Y1, orig_expr);
			delete it;
			delete X1;
			delete Y1;
			estack.push(res);
		} else throw EObjFunc(stringFormatArr("Wrong arity {0:%d} in CalcPostfix()", it->arity));
	}
	if (estack.size() != 1)
		throw EObjFunc(stringFormatArr("Финальный размер стека = {0:%zu} (требуется 1) в CalcPostfix(). Синтаксическая ошибка? Выражение: '",
									   "Final stack size = {0:%zu} (must be 1) in CalcPostfix(). Syntax error? Expression: '",
									   estack.size()) + orig_expr + "'" + comment);
	return estack.top();	// the single final element of estack is returned and should be deleted by the caller
}
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
std::string stringTagPrintf(const std::string &input_text, const std::map<std::string, ValBase*> &tag_val, int &count, std::set<std::string> &tags_left)
{
	size_t lastPos = 0;				// first position after the last bracket
	std::string res = "";
	std::string bracket = "$";		// bracket (tag delimiter) which is currently to be found
	bool finished = false;
	count = 0;

	while(!finished) {
		size_t pos = input_text.find_first_of(bracket, lastPos);
		if (pos == std::string::npos) {		// if bracket is not found, the "end()" element is considered as the bracket
			pos = input_text.size();
			finished = true;
		}

		size_t space_len = 0;		// additional spaces used for printing the value
		if (bracket == "$") {
			res += std::string(input_text.data()+lastPos, pos-lastPos);
			bracket = "; \r\n\t";
		} else {
			// width is only applied if the tag is followed by spaces or tabs
			bool apply_width = false;
			if (pos < input_text.size() && input_text[pos] == ' ') {
				apply_width = true;

				size_t pos2 = input_text.find_first_not_of(" ", pos);	// position of the after-space character
				char after_space = 0;									// 0 <-> the after-space character is not found
				if (pos2 < input_text.size()) after_space = input_text[pos2];
				if (after_space == 0) space_len = input_text.size() - pos;													// use all remaining spaces
				else if (after_space != '\t' && after_space != '\r' && after_space != '\n') space_len = pos2 - 1 - pos;		// use all spaces but one
				else space_len = pos2 - pos;																				// use all spaces
			}
			else if (pos < input_text.size() && input_text[pos] == '\t') apply_width = true;

			std::string tag = std::string(input_text.data()+lastPos, pos-lastPos);	// currently processed tag from "input_text"
			std::vector<std::string> tag_parts;
			tokenize(tag, tag_parts, "%", true);
			if (tag_parts.size() != 1 && tag_parts.size() != 2)
				throw EObjFunc("Некорректный формат тэга '" + tag + "', ожидается: $TAG $TAG; $TAG%FMT $TAG%FMT;",
							   "Incorrect tag format '" + tag + "', expected: $TAG $TAG; $TAG%FMT $TAG%FMT;");

			std::string fmt = "";
			if (tag_parts.size() == 2)
				fmt = "%" + tag_parts[1];

			size_t width = 1 + pos-lastPos + space_len;		// full width to apply to format: $ + tag_length + spaces_length
			if (!apply_width) width = 0;

			// tag_parts[0] is the expression
			std::vector<std::string> infix = StringToInfix(tag_parts[0]);
			std::vector<const ValBase*> postfix = InfixToPostfix(infix, tag_val, count, tags_left, tag_parts[0]);
			const ValBase *res_postfix = CalcPostfix(postfix, tag_parts[0]);
			res += res_postfix->ToString(fmt, width);
			if (!width) space_len = 0;		// if the width was not actually applied, no need to scroll the spaces
			delete res_postfix;

			bracket = "$";
		}

		lastPos = pos;
		if (lastPos < input_text.size() && (input_text[lastPos] == ';' || input_text[lastPos] == '$'))		// for white-space brackets lastPos is not incremented
			lastPos++;
		lastPos += space_len;				// scroll the spaces borrowed for printing
	}
	return res;
}
//------------------------------------------------------------------------------------------
// Returns array of tags/expressions (without "$" and "%fmt" ) found in "input_text".
// Useful to check what expressions are present in string "input_text"
std::vector<std::string> stringExtractTags(const std::string &input_text)
{
	size_t lastPos = 0;				// first position after the last bracket
	std::vector<std::string> res;
	std::string bracket = "$";		// bracket (tag delimiter) which is currently to be found
	bool finished = false;

	while(!finished) {
		size_t pos = input_text.find_first_of(bracket, lastPos);
		if (pos == std::string::npos) {		// if bracket is not found, the "end()" element is considered as the bracket
			pos = input_text.size();
			finished = true;
		}

		if (bracket == "$")
			bracket = "; \r\n\t";
		else {
			std::string tag = std::string(input_text.data()+lastPos, pos-lastPos);	// currently processed tag from "input_text"
			std::vector<std::string> tag_parts;
			tokenize(tag, tag_parts, "%", true);
			if (tag_parts.size() != 1 && tag_parts.size() != 2)
				throw EObjFunc("Некорректный формат тэга '" + tag + "', ожидается: $TAG $TAG; $TAG%FMT $TAG%FMT;",
							   "Incorrect tag format '" + tag + "', expected: $TAG $TAG; $TAG%FMT $TAG%FMT;");

			res.push_back(tag_parts[0]);
			bracket = "$";
		}

		lastPos = pos;
		if (lastPos < input_text.size() && (input_text[lastPos] == ';' || input_text[lastPos] == '$'))		// for white-space brackets lastPos is not incremented
			lastPos++;
	}
	return res;
}
//------------------------------------------------------------------------------------------
}	// namespace HMMPI
