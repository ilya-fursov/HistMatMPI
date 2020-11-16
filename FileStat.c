/*
 * FileStat.c
 *
 *  Created on: 28 Nov 2016
 *      Author: ilya fursov
 */

#undef __cplusplus

#include <sys/stat.h>

int FileModTime(const char *file, time_t *time)
{
	int ret = 0;

	struct stat buf;
	ret = stat(file, &buf);
	if (ret == 0)
		*time = buf.st_mtime;

	return ret;
}

#define __cplusplus 201103L
