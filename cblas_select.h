extern "C"
{

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
	#include "mkl_cblas.h"
#else
	#include "cblas.h"
#endif

}
