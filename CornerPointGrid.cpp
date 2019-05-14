/*
 * CornerPointGrid.cpp
 *
 *  Created on: May 12, 2019
 *      Author: ilya
 */

#include "CornerPointGrid.h"
#include "Utils.h"
#include <cassert>
#include <cstring>

namespace HMMPI
{
//------------------------------------------------------------------------------------------
void CornGrid::ReadGrids(const char *file, std::vector<size_t> len, std::vector<std::vector<double>> &data, std::vector<std::string> S1, std::string S2)
{															// reads a number of grids from "file"
	const int Buff = 4096;									// allocates and fills "data" of size S1.size(), with data[i].size() = len[i]
	char strmsg[Buff], str0[Buff];							// S1[i], S2 - are the start and end markers of "grid[i]" which is loaded to "data[i]"
	char *str = 0;
	FILE *File = fopen(file, "r");

	assert(len.size() == S1.size());
	data = std::vector<std::vector<double>>(S1.size());
	for (size_t i = 0; i < S1.size(); i++)
		data[i] = std::vector<double>(len[i]);

	size_t GridCount = 0;	// counts total grids already read
	size_t ValCount = 0;	// counts values read for the current grid
	size_t c = 0;			// index within currently read grid
	bool seek_beg = true;
	bool new_line = true;
	int ind = -1;			// index within the grid names array

	if (File == 0)
		throw Exception((std::string)"Cannot open " + file + "\n");

	bool expect_scan_two = true;
	while (ReadTokenComm(File, &str, new_line, str0, Buff))		// reads a token to "str", ignoring comments
	{
		std::string S = str;
		if (seek_beg)
		{
			S = ToUpper(S);
			ind = StrIndex(S, S1);
			if (ind != -1)					// "S" is the starting string for grid #ind
			{
				seek_beg = false;
				ValCount = 0;
				c = 0;
			}
			continue;
		}
		if (!seek_beg && S == S2)			// found the ending string
		{
			seek_beg = true;
			assert(ind != -1);
			if (ValCount < len[ind])
			{
				fclose(File);
				File = 0;
				sprintf(strmsg, " grid contains less values (%zu) than expected (%zu)\n", ValCount, len[ind]);	// TODO test
				throw Exception(std::string(file) + ": " + S1[ind] + std::string(strmsg));
			}

			GridCount++;
			if (GridCount >= S1.size())		// all grids have been read
				break;

			continue;
		}
		if (!seek_beg)						// reading the main data
		{
			size_t cnt;
			double d;
			bool err = false;
			assert(ind != -1);

			if (expect_scan_two)			// scan_two() and scan_one() are invoked based on the expectations (based on the previous successful scan)
			{
				if (!scan_two(str, cnt, d, expect_scan_two) && !scan_one(str, d, expect_scan_two))
					err = true;
			}
			else
			{
				if (!scan_one(str, d, expect_scan_two) && !scan_two(str, cnt, d, expect_scan_two))
					err = true;
			}

			if (!err)
			{
				if (!expect_scan_two)
					cnt = 1;

				ValCount += cnt;
				if (ValCount > len[ind])	// too many values encountered
				{
					fclose(File);
					File = 0;
					sprintf(strmsg, " grid contains more values than expected (%zu)\n", len[ind]);		// TODO test
					throw Exception(std::string(file) + ": " + S1[ind] + std::string(strmsg));
				}

				if (expect_scan_two)
					for (size_t i = 0; i < cnt; i++)
					{
						data[ind][c] = d;
						c++;
					}
				else
				{
					data[ind][c] = d;
					c++;
				}
			}
			else							// error reading the values
			{
				fclose(File);
				File = 0;
				sprintf(strmsg, " grid contains non-numeric symbol %s\n", str);				// TODO test
				throw Exception(std::string(file) + ": " + S1[ind] + std::string(strmsg));
			}
		}
	}
	fclose(File);
	File = 0;

	if (!seek_beg)
	{
		assert(ind != -1);
		if (ValCount < len[ind])
		{
			sprintf(strmsg, " grid contains less values (%zu) than expected (%zu)\n", ValCount, len[ind]);		// TODO test
			throw Exception(std::string(file) + ": " + S1[ind] + std::string(strmsg));
		}
		GridCount++;
	}

	if (GridCount < S1.size())
	{
		sprintf(strmsg, "Only %zu grid(s) found out of %zu\n", GridCount, S1.size());		// TODO test
		throw Exception(std::string(file) + ": " + std::string(strmsg));
	}
}
//------------------------------------------------------------------------------------------
bool CornGrid::ReadTokenComm(FILE *F, char **str, bool &new_line, char *str0, const int str0_len)
{																	// reads a token from the file (delimited by ' ', '\t', '\r', '\n'), dropping "--..." comments
	*str = 0;														// returns true on success, false on failure/EOF
																	// the token is saved to "str"
	static const char COMM[] = "--";		// comment beginning	// set "new_line" = true in the first call, then the function will manage it
	static const char DELIM[] = " \t\r\n";	// delimiters			// str0 is a working array (stores a line), it should have been allocated

	while (*str == 0)
	{
		if (new_line)
		{
			if (fgets(str0, str0_len, F) != 0)	// read the line
			{
				// remove the comment
				char *comm_ind = strstr(str0, COMM);
				if (comm_ind != 0)			// comment found
					comm_ind[0] = 0;		// set end-of-line at the comment start

				new_line = false;

				// get the first token
				*str = strtok(str0, DELIM);
			}
			else
				return false;
		}
		else
			*str = strtok(0, DELIM);			// TODO test more esp comments

		if (*str == 0)
			new_line = true;
	}

	return true;
}
//------------------------------------------------------------------------------------------
int CornGrid::StrIndex(const std::string &s, const std::vector<std::string> &vecs)	// index of "s" in vecs[], -1 if not found
{
	for (size_t i = 0; i < vecs.size(); i++)
		if (s == vecs[i])
			return i;

	return -1;
}
//------------------------------------------------------------------------------------------
inline bool CornGrid::scan_two(const char *str, size_t &cnt, double &d, bool &expect_scan_two)	// parses "cnt*d", returns 'true' on success, updates 'expect_scan_two'
{
	char swork[8];
	swork[0] = '\0';

	int read = sscanf(str, "%zu*%lg%5s", &cnt, &d, swork);
	if (read == 2 && swork[0] == '\0')				// RPT*VAL successfully read		TODO test more this template
	{
		expect_scan_two = true;
		return true;
	}
	else
		return false;
}
//------------------------------------------------------------------------------------------
inline bool CornGrid::scan_one(const char *str, double &d, bool &expect_scan_two)				// parses "d", returns 'true' on success, updates 'expect_scan_two'
{
	char swork[8];
	swork[0] = '\0';

	int read = sscanf(str, "%lg%5s", &d, swork);
	if (read == 1 && swork[0] == '\0') 				// VAL successfully read		TODO test more
	{
		expect_scan_two = false;
		return true;
	}
	else
		return false;
}
//------------------------------------------------------------------------------------------
std::string CornGrid::unify_pillar_z()		// sets z0_ij, z1_ij of the pillars to be const, corrects the corresponding x_ij, y_ij; returns a short message
{
	double mz0 = 0, mz1 = 0;
	size_t pcount = (Nx+1)*(Ny+1);

	assert(grid_loaded);

	for (size_t j = 0; j < Ny+1; j++)
		for (size_t i = 0; i < Nx+1; i++)	// consider pillar p = (i, j)
		{
			size_t p = j*(Nx+1) + i;

			mz0 += coord[p*6+2];
			mz1 += coord[p*6+5];
		}

	mz0 /= pcount;							// find the mean z0_ij, z1_ij
	mz1 /= pcount;

	for (size_t j = 0; j < Ny+1; j++)
		for (size_t i = 0; i < Nx+1; i++)	// consider pillar p = (i, j)
		{
			size_t p = j*(Nx+1) + i;

			double x0 = coord[p*6];
			double y0 = coord[p*6+1];
			double z0 = coord[p*6+2];

			double x1 = coord[p*6+3];
			double y1 = coord[p*6+4];
			double z1 = coord[p*6+5];

			if (z1 != z0)
			{
				coord[p*6] = x0 + (x1-x0)/(z1-z0)*(mz0 - z0);
				coord[p*6+1] = y0 + (y1-y0)/(z1-z0)*(mz0 - z0);
			}
			else
			{
				coord[p*6] = x0;
				coord[p*6+1] = y0;
			}
			coord[p*6+2] = mz0;

			if (z1 != z0)
			{
				coord[p*6+3] = x0 + (x1-x0)/(z1-z0)*(mz1 - z0);
				coord[p*6+4] = y0 + (y1-y0)/(z1-z0)*(mz1 - z0);
			}
			else
			{
				coord[p*6+3] = x0;
				coord[p*6+4] = y0;
			}
			coord[p*6+5] = mz1;
		}
	// TODO test by explicit output of pillars, and by creating a grid with/without pillar correction

	char msg[HMMPI::BUFFSIZE];
	sprintf(msg, "Processed %zu pillars, pillar starts unified to %g, pillar ends unified to %g", pcount, mz0, mz1);

	return msg;
}
//------------------------------------------------------------------------------------------
std::string CornGrid::LoadCOORD_ZCORN(std::string fname, int nx, int ny, int nz, double dx, double dy)	// loads "coord", "zcorn" for the grid (nx, ny, nz) from ASCII format (COORD, ZCORN)
{																										// [dx, dy] is the coordinates origin, it is added to COORD
	Nx = nx;																							// a small message is returned by this function
	Ny = ny;
	Nz = nz;

	const size_t coord_size = 6*(Nx+1)*(Ny+1);
	const size_t zcorn_size = 8*Nx*Ny*Nz;

	std::vector<std::vector<double>> data;

	// read the input file
	ReadGrids(fname.c_str(), std::vector<size_t>{coord_size, zcorn_size}, data, std::vector<std::string>{"COORD", "ZCORN"}, "/");

	assert(data.size() == 2);
	coord = std::move(data[0]);
	zcorn = std::move(data[1]);

	assert(coord.size() == coord_size);
	assert(zcorn.size() == zcorn_size);

	// shift the origin
	for (size_t j = 0; j < Ny+1; j++)
		for (size_t i = 0; i < Nx+1; i++)		// consider pillar p = (i, j)
		{
			size_t p = j*(Nx+1) + i;

			coord[p*6] += dx;
			coord[p*6+1] += dy;

			coord[p*6+3] += dx;
			coord[p*6+4] += dy;
		}

	grid_loaded = true;
	return stringFormatArr("Loaded {0:%zu} pillars and {1:%zu} ZCORN values", std::vector<size_t>{coord_size, zcorn_size});
}
//------------------------------------------------------------------------------------------
void CornGrid::fill_cell_coord()					// fills "cell_coord" from coord, zcorn, and grid dimensions
{
	const size_t coord_size = 6*(Nx+1)*(Ny+1);			// VTK_VOXEL:
	const size_t zcorn_size = 8*Nx*Ny*Nz;				// 2 --- 3				   6 --- 7
														// |	 |	+ another face |	 |
	assert(coord.size() == coord_size);					// 0 --- 1				   4 --- 5
	assert(zcorn.size() == zcorn_size);
	cell_coord = std::vector<double>(zcorn_size*3);		// ORDER: (x,y,z) for 8 vertices of the 1st cell, (x,y,z) for 8 vertices of the second cell,...
														// Vertex order in a cell: as in VTK_VOXEL
														// CELLS: i - fastest, k - slowest
	for (size_t k = 0; k < Nz; k++)
		for (size_t j = 0; j < Ny; j++)
			for (size_t i = 0; i < Nx; i++)			// consider cell (i, j, k)
			{
				size_t p[4];
				p[0] = j*(Nx+1) + i;				// global indices of the four pillars
				p[1] = j*(Nx+1) + i+1;
				p[2] = (j+1)*(Nx+1) + i;
				p[3] = (j+1)*(Nx+1) + i+1;
				assert(p[3]*6 + 5 < coord_size);

				size_t v[8];
				v[0] = 2*i + 4*Nx*j + 8*Nx*Ny*k;	// global indices in "zcorn" of the vertices
				v[1] = 2*i+1 + 4*Nx*j + 8*Nx*Ny*k;
				v[2] = 2*(i+Nx) + 4*Nx*j + 8*Nx*Ny*k;
				v[3] = 2*(i+Nx)+1 + 4*Nx*j + 8*Nx*Ny*k;

				v[4] = 2*i + 4*Nx*(j+Ny) + 8*Nx*Ny*k;
				v[5] = 2*i+1 + 4*Nx*(j+Ny) + 8*Nx*Ny*k;
				v[6] = 2*(i+Nx) + 4*Nx*(j+Ny) + 8*Nx*Ny*k;
				v[7] = 2*(i+Nx)+1 + 4*Nx*(j+Ny) + 8*Nx*Ny*k;
				assert(v[7] < zcorn_size);

				double xpill[8];					// order: 4 pillars x_up, 4 pillars x_down
				double ypill[8];
				double zpill[8];

				bool use_z[8];						// if pillar is not vertical, z values from COORD will be used
				for (int n = 0; n < 4; n++)			// n - pillar number
				{
					use_z[n] = use_z[n+4] = true;

					xpill[n] = coord[p[n]*6];
					ypill[n] = coord[p[n]*6+1];
					zpill[n] = coord[p[n]*6+2];

					xpill[n+4] = coord[p[n]*6+3];
					ypill[n+4] = coord[p[n]*6+4];
					zpill[n+4] = coord[p[n]*6+5];

					if (xpill[n] == xpill[n+4] && ypill[n] == ypill[n+4])
						use_z[n] = use_z[n+4] = false;
				}

				// fill the final cell vertices
				for (int n = 0; n < 4; n++)			// n - vertex in the upper plane, n+4 - in the lower plane
				{
					size_t ind = Nx*Ny*k + Nx*j + i;
					const double x0 = xpill[n];
					const double y0 = ypill[n];
					const double z0 = zpill[n];
					const double x1 = xpill[n+4];
					const double y1 = ypill[n+4];
					const double z1 = zpill[n+4];

					if (use_z[n])
					{
						cell_coord[24*ind + 3*n] = x0 + (x1-x0)/(z1-z0)*(zcorn[v[n]] - z0);
						cell_coord[24*ind + 3*n+1] = y0 + (y1-y0)/(z1-z0)*(zcorn[v[n]] - z0);

						cell_coord[24*ind + 3*n + 12] = x0 + (x1-x0)/(z1-z0)*(zcorn[v[n+4]] - z0);
						cell_coord[24*ind + 3*n + 13] = y0 + (y1-y0)/(z1-z0)*(zcorn[v[n+4]] - z0);
					}
					else
					{
						cell_coord[24*ind + 3*n] = x0;
						cell_coord[24*ind + 3*n+1] = y0;

						cell_coord[24*ind + 3*n + 12] = x0;
						cell_coord[24*ind + 3*n + 13] = y0;
					}
					cell_coord[24*ind + 3*n+2] = zcorn[v[n]];
					cell_coord[24*ind + 3*n+14] = zcorn[v[n+4]];
				}
			}
}
//------------------------------------------------------------------------------------------

}	// namespace HMMPI

