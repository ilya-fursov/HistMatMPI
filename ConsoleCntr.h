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
enum Color
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
	const int LEN = 100;

	bool is_b;		// here the "global state" is stored
	bool is_i;
	bool is_u;
	bool is_r;
	Color fg_c;
	Color bg_c;

	std::string set_b(bool b){is_b = b; return (b) ? "\33[1m" : "";};		// "global state" is updated by these four functions
	std::string set_i(bool i){is_i = i; return (i) ? "\33[3m" : "";};
	std::string set_u(bool u){is_u = u; return (u) ? "\33[4m" : "";};
	std::string set_r(bool r){is_r = r; return (r) ? "\33[7m" : "";};
	std::string col_to_str(Color c){char buff[LEN]; sprintf(buff, "%d", c); return buff;};

public:
	TextAttr(){is_b = is_i = is_u = is_r = false; fg_c = bg_c = VT_DEFAULT;};

	std::string set_color(Color f, Color b){fg_c = f; bg_c = b; return (std::string)"\33[3" + col_to_str(f) + ";4" + col_to_str(b) + "m";};
	std::string set_fg_color(Color f){return set_color(f, VT_DEFAULT);};
	std::string set_attr(bool b, bool i, bool u, bool r){return (std::string)"\33[0m" + set_b(b) + set_i(i) + set_u(u) + set_r(r) + set_color(fg_c, bg_c);};
	std::string set_bold(bool v){return set_attr(v, is_i, is_u, is_r);}
	std::string set_ital(bool v){return set_attr(is_b, v, is_u, is_r);}
	std::string set_unds(bool v){return set_attr(is_b, is_i, v, is_r);}
	std::string set_revs(bool v){return set_attr(is_b, is_i, is_u, v);}
};

}	// namespace HMMPI

#endif /* CONSOLECNTR_H_ */
