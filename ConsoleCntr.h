/*
 * ConsoleCntr.h
 *
 *  Created on: 26 May 2016
 *      Author: ilya fursov
 */

#ifndef CONSOLECNTR_H_
#define CONSOLECNTR_H_

// http://www.cplusplus.com/forum/general/18200/

#include <string>
#include <cstdio>

namespace HMMPI
{
//------------------------------------------------------------------------------------------
enum class Color
{
	VT_BLACK = 0,
	VT_RED = 1,
	VT_GREEN = 2,
	VT_YELLOW = 3,
	VT_BLUE = 4,
	VT_MAGENTA = 5,
	VT_CYAN = 6,
	VT_WHITE = 7,
	VT_DEFAULT = 9
};
//------------------------------------------------------------------------------------------
class TextAttr
{
protected:
	bool is_b;		// here the "global state" is stored
	bool is_i;
	bool is_u;
	bool is_r;
	Color fg_c;
	Color bg_c;

	std::string set_b(bool b);		// "global state" is updated by these four functions
	std::string set_i(bool i);
	std::string set_u(bool u);
	std::string set_r(bool r);
	std::string col_to_str(Color c);

public:
	TextAttr();

	std::string set_color(Color f, Color b);
	std::string set_fg_color(Color f);
	std::string set_attr(bool b, bool i, bool u, bool r);
	std::string set_bold(bool v);
	std::string set_ital(bool v);
	std::string set_unds(bool v);
	std::string set_revs(bool v);
};
//------------------------------------------------------------------------------------------
}	// namespace HMMPI

#endif /* CONSOLECNTR_H_ */
