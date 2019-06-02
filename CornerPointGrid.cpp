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
#include <utility>
#include <iostream>
#include <cmath>

namespace HMMPI
{

//------------------------------------------------------------------------------------------
// NNC_point
//------------------------------------------------------------------------------------------
bool NNC_point::operator<=(const NNC_point &N2) const		// comparison is based on {i,j} only
{
	return (i < N2.i) || (i == N2.i && j <= N2.j);
}
//------------------------------------------------------------------------------------------
bool NNC_point::operator==(const NNC_point &N2) const		// comparison is based on {i,j} only
{
	return (i == N2.i && j == N2.j);
}
//------------------------------------------------------------------------------------------
// NNC
//------------------------------------------------------------------------------------------
NNC::NNC(int i0, int j0, int k0, int i1, int j1, int k1) : N0(i0, j0, k0), N1(i1, j1, k1)
{
	if (!(N0 <= N1))										// keep the points in the pair _ordered_
		std::swap(N0, N1);
}
//------------------------------------------------------------------------------------------
NNC NNC::incr(int di, int dj) const							// returns NNC where i and j are incremented by di, dj compared to "this" (same increment for both NNC_points)
{
	NNC res(*this);
	res.N0.i += di;
	res.N0.j += dj;

	res.N1.i += di;
	res.N1.j += dj;

	return res;
}
//------------------------------------------------------------------------------------------
bool NNC::operator==(const NNC &nnc2) const					// comparison is based on {i,j} of both points
{
	return N0 == nnc2.N0 && N1 == nnc2.N1;
}
//------------------------------------------------------------------------------------------
bool NNC::is_neighbour(const NNC &nnc2) const				// 'true' if the two NNCs are adjacent
{
	if (*this == nnc2)
		return true;
	if (N0 == nnc2.N0 || N0 == nnc2.N1 || N1 == nnc2.N0 || N1 == nnc2.N1)
		return true;
	if (incr(1,0) == nnc2 || incr(-1,0) == nnc2 || incr(0,1) == nnc2 || incr(0,-1) == nnc2)
		return true;

	return false;
}
//------------------------------------------------------------------------------------------
void NNC::add_NNC_to_array(std::vector<std::vector<NNC>> &NNC_array, NNC n)	// adds "n" to the NNC array, taking connectivity into account (each NNC_array[i] is a connected series of NNCs)
{
	int ind = -1;
	for (size_t i = 0; i < NNC_array.size(); i++)
	{
		for (size_t j = 0; j < NNC_array[i].size(); j++)
			if (NNC_array[i][j].is_neighbour(n))	// found a neighbour of "n" in NNC_array[i]
			{
				ind = i;
				break;
			}

		if (ind != -1)								// neighbour found - exit the loop
			break;
	}

	// add "n" to the appropriate NNC_array[i]
	if (ind != -1)
		NNC_array[ind].push_back(n);
	else
		NNC_array.push_back(std::vector<NNC>{n});	// start a new connectivity component
}
//------------------------------------------------------------------------------------------
// CornGrid
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
		throw Exception((std::string)"Cannot open " + file + "\n");		// TODO check

	try
	{
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
					sprintf(strmsg, " grid contains less values (%zu) than expected (%zu)\n", ValCount, len[ind]);
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
						sprintf(strmsg, " grid contains more values than expected (%zu)\n", len[ind]);
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
					sprintf(strmsg, " grid contains non-numeric symbol %s\n", str);
					throw Exception(std::string(file) + ": " + S1[ind] + std::string(strmsg));
				}
			}
		}
	}
	catch (...)
	{
		fclose(File);
		throw;
	}

	fclose(File);
	File = 0;

	if (!seek_beg)
	{
		assert(ind != -1);
		if (ValCount < len[ind])
		{
			sprintf(strmsg, " grid contains less values (%zu) than expected (%zu)\n", ValCount, len[ind]);
			throw Exception(std::string(file) + ": " + S1[ind] + std::string(strmsg));
		}
		GridCount++;
	}

	if (GridCount < S1.size())
	{
		sprintf(strmsg, "Only %zu grid(s) found out of %zu\n", GridCount, S1.size());
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
			*str = strtok(0, DELIM);

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
	if (read == 2 && swork[0] == '\0')				// RPT*VAL successfully read
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
	if (read == 1 && swork[0] == '\0') 				// VAL successfully read
	{
		expect_scan_two = false;
		return true;
	}
	else
		return false;
}
//------------------------------------------------------------------------------------------
bool CornGrid::faces_intersect(double a0, double b0, double c0, double d0, double a1, double b1, double c1, double d1)	// 'true' if two faces intersect;
{											// the faces are defined by their z-values for two shared pillars (0, 1): face_1 is [a0, b0; a1, b1], face_2 is [c0, d0; c1, d1]
	if ((a0 < c0 && c0 < b0)||(c0 < a0 && a0 < d0))		// intersection for pillar 0
		return true;
	if ((a1 < c1 && c1 < b1)||(c1 < a1 && a1 < d1))		// intersection for pillar 1
		return true;

	if ((b0 < c0 && d1 < a1)||(d0 < a0 && b1 < c1))		// higher/lower
		return true;

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

	char msg[HMMPI::BUFFSIZE];
	sprintf(msg, "Processed %zu pillars, pillar starts unified to %g, pillar ends unified to %g", pcount, mz0, mz1);

	return msg;
}
//------------------------------------------------------------------------------------------
std::string CornGrid::analyze()			// finds dx0, dy0, theta0, Q0; returns a short message
{
	assert(grid_loaded);

	// find dx0, theta0
	double sum = 0, sum_th0 = 0;
	double x0 = coord[0], y0 = coord[1];
	for (size_t i = 1; i < Nx+1; i++)
	{
		size_t p = i;
		double x = coord[p*6];			// only the top point of the pillar
		double y = coord[p*6+1];
		sum_th0 += atan2(y - y0, x - x0);

		x = (x - x0)/i;
		y = (y - y0)/i;
		sum += sqrt(x*x + y*y);
	}
	dx0 = sum/Nx;
	theta0 = sum_th0/Nx;

	double cos0 = cos(theta0);
	double sin0 = sin(theta0);
	Q0 = Mat(std::vector<double>{cos0, -sin0, sin0, cos0}, 2, 2);

	const bool take_cos = (fabs(cos0) >= fabs(sin0));

	// find dy0 (with sign)
	sum = 0;
	double sum2 = 0;
	for (size_t j = 1; j < Ny+1; j++)
	{
		size_t p = j*(Nx+1);
		double x = coord[p*6];			// only the top point of the pillar
		double y = coord[p*6+1];
		x = (x - x0)/j;
		y = (y - y0)/j;
		sum += sqrt(x*x + y*y);
		if (take_cos)
			sum2 += y/cos0;
		else
			sum2 -= x/sin0;
	}
	dy0 = sum/Ny;
	sum2 /= Ny;

	if (sum2 < 0)
		dy0 = -dy0;


	std::cout << "\n---------------------------\n";
	if (take_cos)	// DEBUG
		std::cout << "\n---------------------------\nanalyze: TAKE COS!!!\n";
	else
		std::cout << "analyze: take sine -!\n";	// DEBUG
	std::cout << "dy0 via sqrt: " << sum/Ny << ", signed version: " << sum2 << "\n-----------------------\n";	// DEBUG


	state_found = true;
	double shift = sqrt((coord[0] - coord[3])*(coord[0] - coord[3]) + (coord[1] - coord[4])*(coord[1] - coord[4]));
	char msg[HMMPI::BUFFSIZE*10];
	sprintf(msg, "The grid cell size DX, DY = (%g, %g), theta = %g degrees, the length of horizontal projection of the first pillar = %g\n",
				 dx0, dy0, theta0/acos(-1.0)*180, shift);

	return msg;
}
//------------------------------------------------------------------------------------------
bool CornGrid::point_between_pillars(double x, double y, int i, int j, double t) const	// 'true' if point (x,y) is between pillars [i,j]-[i+1,j]-[i+1,j+1]-[i,j+1] at depth "t" (fraction)
{
	assert(grid_loaded);
	assert(state_found);
	assert(i >= 0 && (size_t)i < Nx);
	assert(j >= 0 && (size_t)j < Ny);

	std::vector<size_t> p(4);
	p[0] = j*(Nx+1) + i;				// global indices of the four pillars
	p[1] = j*(Nx+1) + i+1;
	p[2] = (j+1)*(Nx+1) + i+1;
	p[3] = (j+1)*(Nx+1) + i;

	for (int n = 0; n < 4; n++)
	{
		size_t p0 = p[n];				// consider two pillars defining the line
		size_t p1 = p[(n+1)%4];
		size_t pt = p[(n+2)%4];			// test pillar which provides the proper test sign

		double x0 =
	}
}
//------------------------------------------------------------------------------------------
//void CornGrid::temp_out_pillars() const	// TODO temp!
//{
//	FILE *f = fopen("pillars_out.txt", "w");
//
//	for (size_t j = 0; j < Ny+1; j++)
//		for (size_t i = 0; i < Nx+1; i++)	// consider pillar p = (i, j)
//		{
//			size_t p = j*(Nx+1) + i;
//
//			fprintf(f, "%.12g\t%.12g\t%.12g\t\t%.12g\t%.12g\t%.12g\n", coord[p*6], coord[p*6+1], coord[p*6+2], coord[p*6+3], coord[p*6+4], coord[p*6+5]);
//		}
//
//	fclose(f);
//}
//void CornGrid::temp_out_zcorn() const	// TODO temp!
//{
//	FILE *f = fopen("zcorn_out.txt", "w");
//
//	for (size_t j = 0; j < zcorn.size(); j++)
//			fprintf(f, "%.12g\n", zcorn[j]);
//
//	fclose(f);
//}
//------------------------------------------------------------------------------------------
CornGrid::CornGrid() : grid_loaded(false), actnum_loaded(false), Nx(0), Ny(0), Nz(0), cell_coord_filled(false),
					   state_found(false), dx0(0), dy0(0), theta0(0)
{
}
//------------------------------------------------------------------------------------------
std::string CornGrid::LoadCOORD_ZCORN(std::string fname, int nx, int ny, int nz, double dx, double dy, bool y_positive)	// loads "coord", "zcorn" for the grid (nx, ny, nz)
{													// from ASCII format (COORD, ZCORN), returning a small message;
	Nx = nx;										// [dx, dy] is the coordinates origin, it is added to COORD; "y_positive" indicates positive/negative direction of the Y axis
	Ny = ny;										// [dx, dy] is [X2, Y2] from the 'MAPAXES', similarly "y_positive" = sign(Y1 - Y2)
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

	// shift the origin / transform
	for (size_t j = 0; j < Ny+1; j++)
		for (size_t i = 0; i < Nx+1; i++)		// consider pillar p = (i, j)
		{
			size_t p = j*(Nx+1) + i;

			coord[p*6] += dx;
			coord[p*6+3] += dx;

			if (y_positive)
			{
				coord[p*6+1] += dy;
				coord[p*6+4] += dy;
			}
			else
			{
				coord[p*6+1] = dy - coord[p*6+1];
				coord[p*6+4] = dy - coord[p*6+4];
			}
		}

	grid_loaded = true;
	actnum_loaded = false;
	cell_coord_filled = false;
	state_found = false;
	std::string msg1 = stringFormatArr("Loaded {0:%zu} COORD values and {1:%zu} ZCORN values\n", std::vector<size_t>{coord_size, zcorn_size});
	std::string msg2 = unify_pillar_z() + "\n";
	std::string msg3 = analyze();

	return msg1 + msg2 + msg3;
}
//------------------------------------------------------------------------------------------
std::string CornGrid::LoadACTNUM(std::string fname)		// loads ACTNUM, should be called after "grid_loaded", returns a small message
{														// treats positive real values as 'active'
	assert(grid_loaded);

	const size_t grid_size = Nx*Ny*Nz;
	std::vector<std::vector<double>> data;

	// read the input file
	ReadGrids(fname.c_str(), std::vector<size_t>{grid_size}, data, std::vector<std::string>{"ACTNUM"}, "/");

	assert(data.size() == 1);
	assert(data[0].size() == grid_size);
	actnum = std::vector<int>(grid_size, 0);

	size_t count = 0;
	for (size_t i = 0; i < grid_size; i++)
		if (data[0][i] > 0)
		{
			actnum[i] = 1;
			count++;			// count the active cells
		}

	actnum_loaded = true;

	return stringFormatArr("Active cells: {0:%zu} / {1:%zu}", std::vector<size_t>{count, grid_size});
}
//------------------------------------------------------------------------------------------
void CornGrid::fill_cell_coord()					// fills "cell_coord" from coord, zcorn, and grid dimensions
{
	assert(grid_loaded);
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

	cell_coord_filled = true;
}
//------------------------------------------------------------------------------------------
std::vector<std::vector<NNC>> CornGrid::get_same_layer_NNC(std::string &out_msg)		// based on the mesh and ACTNUM, generates NNCs (where the logically connected cells are not connected in the mesh)
{																	// only the cells with the same "k" are taken for such NNCs
	assert(grid_loaded);											// the PURPOSE is to form NNCs across the faults

	if (!cell_coord_filled)
		fill_cell_coord();
	assert(cell_coord.size() > 0);

	std::vector<std::vector<NNC>> res;
	size_t NxNy = Nx*Ny;
	size_t count = 0;
	char msg[HMMPI::BUFFSIZE*10];

	for (size_t i = 0; i < Nx; i++)
		for (size_t j = 0; j < Ny; j++)		// consider the first cell (i,j) in the potential NNC
			for (int k = 1; k <= 2; k++)
			{
				size_t i1 = i + k%2;		// consider the second cell (i1,j1) in the potential NNC
				size_t j1 = j + k/2;

				if (i1 >= Nx || j1 >= Ny)
					continue;

				for (size_t z = 0; z < Nz; z++)
				{
					size_t m = i + j*Nx + z*NxNy;		// first cell
					size_t m1 = i1 + j1*Nx + z*NxNy;	// second cell

					if (actnum_loaded && (actnum[m] == 0 || actnum[m1] == 0))
						continue;			// inactive cells do not participate in NNC 	TODO check on model with/without ACTNUM

					int p11, p12;	// pillars of the first cell
					int p21, p22;	// pillars of the second cell
					if (i1 > i)
					{
						p11 = 1; p12 = 3;		// #3 2#
						p21 = 0; p22 = 2;		// #1 0#
					}
					else						// ##
					{							// 01
						p11 = 2; p12 = 3;		//
						p21 = 0; p22 = 1;		// 23
					}							// ##

					double a0 = cell_coord[m*24 + p11*3 + 2];
					double b0 = cell_coord[m*24 + (p11+4)*3 + 2];
					double a1 = cell_coord[m*24 + p12*3 + 2];
					double b1 = cell_coord[m*24 + (p12+4)*3 + 2];

					double c0 = cell_coord[m1*24 + p21*3 + 2];
					double d0 = cell_coord[m1*24 + (p21+4)*3 + 2];
					double c1 = cell_coord[m1*24 + p22*3 + 2];
					double d1 = cell_coord[m1*24 + (p22+4)*3 + 2];
					if (!faces_intersect(a0, b0, c0, d0, a1, b1, c1, d1))
					{
						NNC::add_NNC_to_array(res, NNC(i, j, z, i1, j1, z));
						count++;
					}
				}
			}

	sprintf(msg, "Found %zu NNCs grouped into %zu chain(s)", count, res.size());
	out_msg = msg;
	if (actnum_loaded)
		out_msg += " (ACTNUM was used)";
	else
		out_msg += " (ACTNUM was not found)";

	return res;
}
//------------------------------------------------------------------------------------------
void CornGrid::find_cell(double x, double y, double z, int &i, int &j, int &k) 		// find cell [i,j,k] containing the point [x,y,z]
{
	assert(grid_loaded);
	if (!cell_coord_filled)
		fill_cell_coord();
	if (!state_found)
		analyze();

	double x0 = coord[0];			// the first pillar
	double y0 = coord[1];
	double z0 = coord[2];
	double x1 = coord[3];
	double y1 = coord[4];
	double z1 = coord[5];

	double t = 0;
	if (z1 != z0)
		t = (z - z0)/(z1 - z0);		// for vertical pillars (z1 == z0) --> t = 0

	double xbar = x - (1-t)*x0 - t*x1;
	double ybar = y - (1-t)*y0 - t*y1;

	Mat rhs(std::vector<double>{xbar, ybar}, 2, 1);

	Mat sol = Q0.Tr()*rhs;
	i = int(sol(0,0)/dx0);
	j = int(sol(1,0)/dy0);

	// check that the found (i,j) is correct



	//double x0 =
	//double shift = sqrt((coord[0] - coord[3])*(coord[0] - coord[3]) + (coord[1] - coord[4])*(coord[1] - coord[4]));
}
//------------------------------------------------------------------------------------------
void CornGrid::temp_coord_from_cell(int i, int j, int k, double &x, double &y) const	// (i,j,k) -> (x,y)	// TODO it's a temp stuff
{
	assert(grid_loaded);
	assert(cell_coord_filled);

	size_t m = i + j*Nx + k*Nx*Ny;
	int v1 = 0;
	int v2 = 3;
	int v3 = 4;
	int v4 = 7;

	if (i < 0 || i >= Nx)
		throw HMMPI::Exception("i out of range");
	if (j < 0 || j >= Ny)
		throw HMMPI::Exception("j out of range");
	if (k < 0 || k >= Nz)
		throw HMMPI::Exception("k out of range");

	double x0 = (cell_coord[m*24 + v1*3 + 0] + cell_coord[m*24 + v2*3 + 0])/2;
	double y0 = (cell_coord[m*24 + v1*3 + 1] + cell_coord[m*24 + v2*3 + 1])/2;

	double x1 = (cell_coord[m*24 + v3*3 + 0] + cell_coord[m*24 + v4*3 + 0])/2;
	double y1 = (cell_coord[m*24 + v3*3 + 1] + cell_coord[m*24 + v4*3 + 1])/2;

	x = (x0+x1)/2;
	y = (y0+y1)/2;
}
//------------------------------------------------------------------------------------------


}	// namespace HMMPI

