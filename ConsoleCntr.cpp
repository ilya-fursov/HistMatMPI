#include "ConsoleCntr.h"

namespace HMMPI
{

const int LEN = 100;
//------------------------------------------------------------------------------------------
std::string TextAttr::set_b(bool b) 
{ 
	is_b = b; 
	return (b) ? "\33[1m" : ""; 
}
//------------------------------------------------------------------------------------------
std::string TextAttr::set_i(bool i) 
{ 
	is_i = i; 
	return (i) ? "\33[3m" : ""; 
}
//------------------------------------------------------------------------------------------
std::string TextAttr::set_u(bool u) 
{ 
	is_u = u; 
	return (u) ? "\33[4m" : ""; 
}
//------------------------------------------------------------------------------------------
std::string TextAttr::set_r(bool r) 
{ 
	is_r = r; 
	return (r) ? "\33[7m" : ""; 
}
//------------------------------------------------------------------------------------------
std::string TextAttr::col_to_str(Color c) 
{ 
	char buff[LEN]; 
	sprintf(buff, "%d", c); 
	return buff; 
}
//------------------------------------------------------------------------------------------
TextAttr::TextAttr() 
{ 
	is_b = is_i = is_u = is_r = false; 
	fg_c = bg_c = Color::VT_DEFAULT;
}
//------------------------------------------------------------------------------------------
std::string TextAttr::set_color(Color f, Color b) 
{ 
	fg_c = f; 
	bg_c = b; 
	return (std::string)"\33[3" + col_to_str(f) + ";4" + col_to_str(b) + "m"; 
}
//------------------------------------------------------------------------------------------
std::string TextAttr::set_fg_color(Color f) 
{ 
	return set_color(f, Color::VT_DEFAULT); 
}
//------------------------------------------------------------------------------------------
std::string TextAttr::set_attr(bool b, bool i, bool u, bool r) 
{ 
	return (std::string)"\33[0m" + set_b(b) + set_i(i) + set_u(u) + set_r(r) + set_color(fg_c, bg_c); 
};
//------------------------------------------------------------------------------------------
std::string TextAttr::set_bold(bool v) 
{ 
	return set_attr(v, is_i, is_u, is_r); 
}
//------------------------------------------------------------------------------------------
std::string TextAttr::set_ital(bool v) 
{ 
	return set_attr(is_b, v, is_u, is_r); 
}
//------------------------------------------------------------------------------------------
std::string TextAttr::set_unds(bool v) 
{ 
	return set_attr(is_b, is_i, v, is_r); 
}
//------------------------------------------------------------------------------------------
std::string TextAttr::set_revs(bool v) 
{ 
	return set_attr(is_b, is_i, is_u, v); 
}
//------------------------------------------------------------------------------------------
}	// namespace HMMPI