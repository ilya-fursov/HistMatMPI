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
#include <algorithm>
#include <functional>
#include <climits>
#include <chrono>

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
		throw Exception((std::string)"Cannot open " + file + "\n");

	try
	{
		bool expect_scan_two = true;
		while (ReadTokenComm(File, &str, new_line, str0, Buff))		// reads a token to "str", ignoring comments
		{
			std::string S = str;
			if (seek_beg)
			{
				//S = ToUpper(S);
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
		if (File) fclose(File);
		throw;
	}

	if (File) fclose(File);
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
		std::string gr_list = ToString(S1, "%s", ", ");
		gr_list.pop_back();			// pop the '\n'
		sprintf(strmsg, "Only %zu grid(s) found out of %zu (%s)\n", GridCount, S1.size(), gr_list.c_str());
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
	if ((a0 <= c0 && c0 < b0)||(c0 <= a0 && a0 < d0))		// intersection for pillar 0
		return true;
	if ((a1 <= c1 && c1 < b1)||(c1 <= a1 && a1 < d1))		// intersection for pillar 1
		return true;

	if ((b0 <= c0 && d1 <= a1)||(d0 <= a0 && b1 <= c1))		// higher/lower
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

	state_found = true;
	double shift = sqrt((coord[0] - coord[3])*(coord[0] - coord[3]) + (coord[1] - coord[4])*(coord[1] - coord[4]));
	char msg[HMMPI::BUFFSIZE*10];
	sprintf(msg, "The grid cell size DX, DY = (%g, %g), theta = %g degrees, the length of horizontal projection of the first pillar = %g\n",
				 dx0, dy0, theta0/acos(-1.0)*180, shift);

	return msg;
}
//------------------------------------------------------------------------------------------
std::string CornGrid::fill_cell_height()	// fills "cell_height", returns a short message
{
	assert(cell_coord_filled);
	cell_height = std::vector<double>(Nx*Ny*Nz);

	const size_t cell_height_size_local = Nx*Ny*Nz_local;
	std::vector<double> cell_height_local(cell_height_size_local);

	size_t count_empty = 0;							// to be reduced
	for (size_t k = 0; k < Nz_local; k++)
		for (size_t j = 0; j < Ny; j++)
			for (size_t i = 0; i < Nx; i++)			// consider cell (i, j, k)
			{
				size_t ind = Nx*Ny*k + Nx*j + i;
				double h = 0;
				for (size_t p = 0; p < 4; p++)		// pillars
				{
					Mat Top(std::vector<double>(cell_coord_local.begin() + 24*ind + 3*p, cell_coord_local.begin() + 24*ind + 3*p + 3));			// vertex at the cell top
					Mat Bot(std::vector<double>(cell_coord_local.begin() + 24*ind + 3*(p+4), cell_coord_local.begin() + 24*ind + 3*(p+4) + 3));	// vertex at the cell bottom
					h += (Top-Bot).Norm2();			// length along this pillar
				}
				h /= 4;
				cell_height_local[ind] = h;

				if (h <= min_cell_height)
					count_empty++;
			}

	// gather 'cell_height'
	std::vector<int> counts = counts_Nz, displs = displs_Nz;	// two arrays specially for 'cell_height'
	const size_t mult = Nx*Ny;
	if (Nx*Ny*Nz >= (size_t)INT_MAX)
		throw Exception("Array size exceeds INT_MAX in CornGrid::fill_cell_height");

	for (auto &v : counts)
		v *= mult;
	for (auto &v : displs)
		v *= mult;

	MPI_Allgatherv(cell_height_local.data(), cell_height_size_local, MPI_DOUBLE, cell_height.data(), counts.data(), displs.data(), MPI_DOUBLE, comm);

	MPI_Allreduce(MPI_IN_PLACE, &count_empty, 1, MPI_LONG_LONG, MPI_SUM, comm);

	return stringFormatArr("Empty cells: {0:%zu} / {1:%zu}", std::vector<size_t>{count_empty, Nx*Ny*Nz});
}
//------------------------------------------------------------------------------------------
void CornGrid::fill_cell_center()					// fills "cell_center"
{
	assert(cell_coord_filled);
	cell_center = std::vector<double>(3*Nx*Ny*Nz);

	const size_t cell_center_size_local = 3*Nx*Ny*Nz_local;
	std::vector<double> cell_center_local(cell_center_size_local);

	for (size_t k = 0; k < Nz_local; k++)
		for (size_t j = 0; j < Ny; j++)
			for (size_t i = 0; i < Nx; i++)			// consider cell (i, j, k)
			{
				size_t ind = 3*(Nx*Ny*k + Nx*j + i);
				xyz_from_cell_ijk_local(i, j, k, cell_center_local[ind], cell_center_local[ind+1], cell_center_local[ind+2]);
			}

	// gather 'cell_height'
	std::vector<int> counts = counts_Nz, displs = displs_Nz;	// two arrays specially for 'cell_center'
	const size_t mult = 3*Nx*Ny;
	if (3*Nx*Ny*Nz >= (size_t)INT_MAX)
		throw Exception("Array size exceeds INT_MAX in CornGrid::fill_cell_center");

	for (auto &v : counts)
		v *= mult;
	for (auto &v : displs)
		v *= mult;

	MPI_Allgatherv(cell_center_local.data(), cell_center_size_local, MPI_DOUBLE, cell_center.data(), counts.data(), displs.data(), MPI_DOUBLE, comm);
}
//------------------------------------------------------------------------------------------
void CornGrid::xyz_from_cell_ijk(int i, int j, int k, double &x, double &y, double &z) const		// (i,j,k) -> (x,y,z) [RANK-0]
{
	assert(grid_loaded);
	assert(cell_coord_filled);

	size_t m = i + j*Nx + k*Nx*Ny;
	int v1 = 0;
	int v2 = 3;
	int v3 = 4;
	int v4 = 7;

	if (i < 0 || i >= (int)Nx)
		throw HMMPI::Exception("i out of range in CornGrid::xyz_from_cell_ijk");
	if (j < 0 || j >= (int)Ny)
		throw HMMPI::Exception("j out of range in CornGrid::xyz_from_cell_ijk");
	if (k < 0 || k >= (int)Nz)
		throw HMMPI::Exception("k out of range in CornGrid::xyz_from_cell_ijk");

	double x0 = (cell_coord[m*24 + v1*3 + 0] + cell_coord[m*24 + v2*3 + 0])/2;
	double y0 = (cell_coord[m*24 + v1*3 + 1] + cell_coord[m*24 + v2*3 + 1])/2;
	double z0 = (cell_coord[m*24 + v1*3 + 2] + cell_coord[m*24 + v2*3 + 2])/2;

	double x1 = (cell_coord[m*24 + v3*3 + 0] + cell_coord[m*24 + v4*3 + 0])/2;
	double y1 = (cell_coord[m*24 + v3*3 + 1] + cell_coord[m*24 + v4*3 + 1])/2;
	double z1 = (cell_coord[m*24 + v3*3 + 2] + cell_coord[m*24 + v4*3 + 2])/2;

	x = (x0+x1)/2;
	y = (y0+y1)/2;
	z = (z0+z1)/2;
}
//------------------------------------------------------------------------------------------
void CornGrid::xyz_from_cell_ijk_local(int i, int j, int k, double &x, double &y, double &z) const		// (i,j,k_local) -> (x,y,z)
{
	assert(grid_loaded);
	assert(cell_coord_filled);

	size_t m = i + j*Nx + k*Nx*Ny;
	int v1 = 0;
	int v2 = 3;
	int v3 = 4;
	int v4 = 7;

	if (i < 0 || i >= (int)Nx)
		throw HMMPI::Exception("i out of range in CornGrid::xyz_from_cell_ijk_local");
	if (j < 0 || j >= (int)Ny)
		throw HMMPI::Exception("j out of range in CornGrid::xyz_from_cell_ijk_local");
	if (k < 0 || k >= (int)Nz_local)
		throw HMMPI::Exception("k out of range in CornGrid::xyz_from_cell_ijk_local");

	double x0 = (cell_coord_local[m*24 + v1*3 + 0] + cell_coord_local[m*24 + v2*3 + 0])/2;
	double y0 = (cell_coord_local[m*24 + v1*3 + 1] + cell_coord_local[m*24 + v2*3 + 1])/2;
	double z0 = (cell_coord_local[m*24 + v1*3 + 2] + cell_coord_local[m*24 + v2*3 + 2])/2;

	double x1 = (cell_coord_local[m*24 + v3*3 + 0] + cell_coord_local[m*24 + v4*3 + 0])/2;
	double y1 = (cell_coord_local[m*24 + v3*3 + 1] + cell_coord_local[m*24 + v4*3 + 1])/2;
	double z1 = (cell_coord_local[m*24 + v3*3 + 2] + cell_coord_local[m*24 + v4*3 + 2])/2;

	x = (x0+x1)/2;
	y = (y0+y1)/2;
	z = (z0+z1)/2;
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

	const Mat Xt(std::vector<double>{x, y, 0});
	bool is_inside = true;
	for (int n = 0; n < 4; n++)
	{
		size_t p0 = p[n];				// consider two pillars defining the line
		size_t p1 = p[(n+1)%4];
		size_t p2 = p[(n+2)%4];			// test pillar which provides the proper test sign

		Mat X0(std::vector<double>{coord[p0*6]*(1-t) + coord[p0*6+3]*t, coord[p0*6+1]*(1-t) + coord[p0*6+4]*t, 0});
		Mat X1(std::vector<double>{coord[p1*6]*(1-t) + coord[p1*6+3]*t, coord[p1*6+1]*(1-t) + coord[p1*6+4]*t, 0});
		Mat X2(std::vector<double>{coord[p2*6]*(1-t) + coord[p2*6+3]*t, coord[p2*6+1]*(1-t) + coord[p2*6+4]*t, 0});
		Mat V1 = X1 - X0;
		Mat V2 = X2 - X0;
		Mat Vt = Xt - X0;
		Mat prod_pill = VecProd(V1, V2);
		Mat prod_xy = VecProd(V1, Vt);	// may be zero
		if (prod_pill(2,0)*prod_xy(2,0) < 0)
		{
			is_inside = false;
			break;
		}
	}

	pbp_call_count++;
	return is_inside;
}
//------------------------------------------------------------------------------------------
bool CornGrid::point_inside_boundary_pillars(double x, double y, double t) const 		// 'true' if point (x,y) is inside all the boundary pillars at depth "t" (fraction)
{
	assert(grid_loaded);
	assert(state_found);

	const size_t len = 2*Nx + 2*Ny;
	std::vector<size_t> p(len);	// global indices of the boundary pillars

	size_t c = 0;
	for (size_t i = 0; i < Nx; i++)
	{
		p[c] = i;				// j = 0
		c++;
	}
	for (size_t j = 0; j < Ny; j++)
	{
		p[c] = j*(Nx+1) + Nx;	// i = Nx
		c++;
	}
	for (size_t i = Nx; i > 0; i--)
	{
		p[c] = Ny*(Nx+1) + i;	// j = Ny
		c++;
	}
	for (size_t j = Ny; j > 0; j--)
	{
		p[c] = j*(Nx+1);		// i = 0
		c++;
	}

	const Mat Xt(std::vector<double>{x, y, 0});
	int sign = 0;
	bool is_inside = true;

	for (size_t n = 0; n < len; n++)	// go through all boundary pillars
	{
		size_t p0 = p[n];				// consider two pillars defining the line
		size_t p1 = p[(n+1)%len];

		Mat X0(std::vector<double>{coord[p0*6]*(1-t) + coord[p0*6+3]*t, coord[p0*6+1]*(1-t) + coord[p0*6+4]*t, 0});
		Mat X1(std::vector<double>{coord[p1*6]*(1-t) + coord[p1*6+3]*t, coord[p1*6+1]*(1-t) + coord[p1*6+4]*t, 0});

		Mat V1 = X1 - X0;
		Mat Vt = Xt - X0;
		Mat prod_xy = VecProd(V1, Vt);	// may be zero
		const double Prod = prod_xy(2,0);
		if (Prod != 0)
		{
			if (sign*Prod < 0)
			{
				is_inside = false;
				break;
			}
			if (sign == 0)
				sign = (Prod > 0 ? 1: -1);
		}
	}

	pinside_call_count++;
	return is_inside;
}
//------------------------------------------------------------------------------------------
bool CornGrid::find_cell_in_window(double x, double y, int i0, int i1, int j0, int j1, double t, int &ii, int &jj)	// iteratively searches the cell index window [i0, i1)*[j0, j1)
{											// for the first encounter of cell [ii, jj] containing the point (x, y); uses point_between_pillars() test; returns "true" on success
	assert(i0 >= 0 && (size_t)i1 <= Nx);
	assert(j0 >= 0 && (size_t)j1 <= Ny);

	for (int i = i0; i < i1; i++)
		for (int j = j0; j < j1; j++)
			if (point_between_pillars(x, y, i, j, t))
			{
				ii = i;
				jj = j;
				return true;
			}

	return false;
}
//------------------------------------------------------------------------------------------
bool CornGrid::point_in_same_semispace(double x, double y, double z, int i, int j, int k, int v0, int v1, int v2, int vt, double shift) const	// [RANK-0]
{										// for cell (i,j,k) consider the voxel vertices v0, v1, v2, vt = [0, 8)
	assert(grid_loaded);				// return "true" if (x,y,z) is non-strictly in the same semispace relative to the plane span{v0,v1,v2} as "vt"
	assert(cell_coord_filled);			// if v0, v1, v2, vt are all in one plane, vt.z += shift is used for testing
	assert(i >= 0 && (size_t)i < Nx);
	assert(j >= 0 && (size_t)j < Ny);
	assert(k >= 0 && (size_t)k < Nz);
	assert(v0 >= 0 && v0 < 8);
	assert(v1 >= 0 && v1 < 8);
	assert(v2 >= 0 && v2 < 8);
	assert(vt >= 0 && vt < 8);

	size_t ind = Nx*Ny*k + Nx*j + i;
	// points in R^3
	Mat X0(std::vector<double>{cell_coord[24*ind + 3*v0], cell_coord[24*ind + 3*v0 + 1], cell_coord[24*ind + 3*v0 + 2]});
	Mat X1(std::vector<double>{cell_coord[24*ind + 3*v1], cell_coord[24*ind + 3*v1 + 1], cell_coord[24*ind + 3*v1 + 2]});
	Mat X2(std::vector<double>{cell_coord[24*ind + 3*v2], cell_coord[24*ind + 3*v2 + 1], cell_coord[24*ind + 3*v2 + 2]});
	Mat Xt(std::vector<double>{cell_coord[24*ind + 3*vt], cell_coord[24*ind + 3*vt + 1], cell_coord[24*ind + 3*vt + 2]});
	Mat A(std::vector<double>{x, y, z});

	Mat U1 = X1 - X0;
	Mat U2 = X2 - X0;
	Mat Ut = Xt - X0;
	Mat UA = A - X0;
	Mat prod = VecProd(U1, U2);

	// check_t*check_A < 0 -- strictly different semispace
	// check_t*check_A > 0 -- strictly same semispace
	// check_t == 0		   -- degenerate case: v0, v1, v2, vt are all in one plane, perturbation to be made
	// check_A == 0		   -- non-strictly same semispace (same plane)

	double check_t = InnerProd(prod, Ut);
	double check_A = InnerProd(prod, UA);

	if (check_t == 0)			// degenerate case: v0, v1, v2, vt are all in one plane
	{
		Xt(2, 0) += shift;
		Ut = Xt - X0;
		check_t = InnerProd(prod, Ut);
		assert(check_t != 0);
	}

	psspace_call_count++;
	if (check_t*check_A < 0)
		return false;
	else
		return true;
}
//------------------------------------------------------------------------------------------
bool CornGrid::point_below_lower_plane(const pointT &X0, int i, int j, int k, const CornGrid *grid)	// [RANK-0] "true" if X0=(x,y,z) is strictly below the lower plane of cell (i,j,k)
{
	double x, y, z;
	std::tie(x, y, z) = X0;
	return !grid->point_in_same_semispace(x, y, z, i, j, k, 4, 5, 7, 0, - grid->delta_Z);
}
//------------------------------------------------------------------------------------------
bool CornGrid::point_below_upper_plane(const pointT &X0, int i, int j, int k, const CornGrid *grid)	// [RANK-0] "true" if X0=(x,y,z) is non-strictly below the upper plane of cell (i,j,k)
{
	double x, y, z;
	std::tie(x, y, z) = X0;
	return grid->point_in_same_semispace(x, y, z, i, j, k, 0, 1, 3, 4, + grid->delta_Z);
}
//------------------------------------------------------------------------------------------
int CornGrid::find_k_lower_bound(int i, int j, double x, double y, double z) const		// [RANK-0] for column (i,j) find the smallest "k" such that
{										// (x,y,z) is above the lower plane of cell (i,j,k), returns Nz if not found; binary search is used here
	using namespace std::placeholders;

	std::vector<int> k_range(Nz);		// iterator for "k"
	std::iota(k_range.begin(), k_range.end(), 0);
	pointT X0(x, y, z);

	std::vector<int>::iterator res = std::lower_bound(k_range.begin(), k_range.end(), X0,
													  std::bind(CornGrid::point_below_lower_plane, _2, i, j, _1, this));
	return res - k_range.begin();
}
//------------------------------------------------------------------------------------------
void CornGrid::calc_strat_dist(int i1, int j1, int k1, int i2, int j2, int k2, double &dx, double &dy, double &dz) const		// calculates 3D radius-vector (dx, dy, dz)
{								// between cells (i1, j1, k1) and (i2, j2, k2), flattening the stratigraphy
								// uses 'cell_center', and can be called on any RANK
	assert(i1 >=0 and i1 < (int)Nx);
	assert(j1 >=0 and j1 < (int)Ny);
	assert(k1 >=0 and k1 < (int)Nz);

	assert(i2 >=0 and i2 < (int)Nx);
	assert(j2 >=0 and j2 < (int)Ny);
	assert(k2 >=0 and k2 < (int)Nz);

	const size_t ind1 = Nx*Ny*k1 + Nx*j1 + i1;			//		p1	------------ p2_cross
	const size_t ind2 = Nx*Ny*k2 + Nx*j2 + i2;			//		|				 |
														//		|				 |
	const size_t ind1_cross = Nx*Ny*k2 + Nx*j1 + i1;	//		|				 |
	const size_t ind2_cross = Nx*Ny*k1 + Nx*j2 + i2;	//		p1_cross ------- p2

	dx = cell_center[3*ind2] - cell_center[3*ind1];
	dy = cell_center[3*ind2+1] - cell_center[3*ind1+1];

	const double dz1 = cell_center[3*ind1_cross+2] - cell_center[3*ind1+2];
	const double dz2 = cell_center[3*ind2+2] - cell_center[3*ind2_cross+2];
	dz = (dz1 + dz2)/2;
}
//------------------------------------------------------------------------------------------
double CornGrid::calc_scaled_dist(int i1, int j1, int k1, int i2, int j2, int k2, double Rmaj, double rmin, double rz, double cosx, double sinx) const	// calculates the scaled distance for use in 1D covariance function
{																// uses calc_strat_dist(), the 3D variogram radii Rmaj, rmin, rz, and cos/sin of angle chi
	double dx, dy, dz;
	calc_strat_dist(i1, j1, k1, i2, j2, k2, dx, dy, dz);

	double dr1 = (cosx*dx + sinx*dy)/Rmaj;
	double dr2 = (-sinx*dx + cosx*dy)/rmin;
	double dr3 = dz/rz;

	return sqrt(dr1*dr1 + dr2*dr2 + dr3*dr3);
}
//------------------------------------------------------------------------------------------
CornGrid::CornGrid(MPI_Comm c) : comm(c), grid_loaded(false), actnum_loaded(false), Nx(0), Ny(0), Nz(0), Nz_local(0), pbp_call_count(0), pinside_call_count(0), psspace_call_count(0),
					   delta_Z(10.0), min_cell_height(1e-3), actnum_name("ACTNUM"), actnum_min(0), actnum_count(0), cell_coord_filled(false),
					   state_found(false), dx0(0), dy0(0), theta0(0)
{
	MPI_Comm_rank(c, &rank);
}
//------------------------------------------------------------------------------------------
std::string CornGrid::LoadCOORD_ZCORN(std::string fname, int nx, int ny, int nz, double dx, double dy, bool y_positive, std::string aname, double amin)
{											// loads "coord", "zcorn" for the grid (nx, ny, nz) from ASCII format (COORD, ZCORN), returning a small message;
	std::string msg;						// [dx, dy] is the coordinates origin, it is added to COORD; "y_positive" indicates positive/negative direction of the Y axis
	RANK0_SYNCERR_BEGIN(comm);				// [dx, dy] is [X2, Y2] from the 'MAPAXES', similarly "y_positive" = sign(Y1 - Y2)
	{										// aname - ACTNUM name, amin - ACTNUM min
		Nx = nx;							// all reading is done by comm-rank-0
		Ny = ny;
		Nz = nz;

		actnum_name = aname;
		actnum_min = amin;

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
		std::string msg1 = stringFormatArr("Loaded {0:%zu} COORD values and {1:%zu} ZCORN values\n", std::vector<size_t>{coord_size, zcorn_size});
		std::string msg2 = unify_pillar_z() + "\n";
		std::string msg3 = analyze();

		msg = msg1 + msg2 + msg3;
	}
	RANK0_SYNCERR_END(comm);
	Bcast_string(msg, 0, comm);

	// sync all the data loaded
	MPI_Bcast(&Nx, 1, MPI_LONG_LONG, 0, comm);
	MPI_Bcast(&Ny, 1, MPI_LONG_LONG, 0, comm);
	MPI_Bcast(&Nz, 1, MPI_LONG_LONG, 0, comm);

	Bcast_string(actnum_name, 0, comm);
	MPI_Bcast(&actnum_min, 1, MPI_DOUBLE, 0, comm);
	Bcast_vector(coord, 0, comm);

	MPI_Bcast(&dx0, 1, MPI_DOUBLE, 0, comm);
	MPI_Bcast(&dy0, 1, MPI_DOUBLE, 0, comm);
	MPI_Bcast(&theta0, 1, MPI_DOUBLE, 0, comm);
	Q0.Bcast(0, comm);

	pbp_call_count = 0;
	pinside_call_count = 0;
	psspace_call_count = 0;
	grid_loaded = true;
	actnum_loaded = false;
	cell_coord_filled = false;
	MPI_Bcast(&state_found, 1, MPI_BYTE, 0, comm);

	// scatter 'zcorn' to 'zcorn_local'
	if (8*Nx*Ny*Nz >= (size_t)INT_MAX)
		throw Exception("Array size exceeds INT_MAX in CornGrid::LoadCOORD_ZCORN");

	MPI_count_displ(comm, Nz, counts_Nz, displs_Nz);			// all inputs and outputs are sync on "comm"
	assert(rank < (int)counts_Nz.size());
	Nz_local = counts_Nz[rank];

	std::vector<int> counts = counts_Nz, displs = displs_Nz;	// two arrays specially for zcorn
	const size_t mult = 8*Nx*Ny;
	for (auto &v : counts)
		v *= mult;
	for (auto &v : displs)
		v *= mult;

	assert(rank < (int)counts.size());
	zcorn_local = std::vector<double>(counts[rank]);

	MPI_Scatterv(zcorn.data(), counts.data(), displs.data(), MPI_DOUBLE, zcorn_local.data(), counts[rank], MPI_DOUBLE, 0, comm);

	return msg;
}
//------------------------------------------------------------------------------------------
std::string CornGrid::LoadACTNUM(std::string fname)		// loads ACTNUM, should be called after "grid_loaded", returns a small message
{														// treats real values > "actnum_min" as 'active'
	assert(grid_loaded);								// all reading is done by comm-rank-0
	std::string msg;

	RANK0_SYNCERR_BEGIN(comm);
	{
		const size_t grid_size = Nx*Ny*Nz;
		std::vector<std::vector<double>> data;

		// read the input file
		ReadGrids(fname.c_str(), std::vector<size_t>{grid_size}, data, std::vector<std::string>{actnum_name}, "/");

		assert(data.size() == 1);
		assert(data[0].size() == grid_size);
		actnum = std::vector<int>(grid_size, 0);

		actnum_count = 0;
		for (size_t i = 0; i < grid_size; i++)
			if (data[0][i] > actnum_min)
			{
				actnum[i] = 1;
				actnum_count++;			// count the active cells
			}

		// fill 'act_cell_ind'
		act_cell_ind = std::vector<int>(actnum_count);
		size_t c = 0;
		for (size_t i = 0; i < grid_size; i++)
			if (actnum[i])
			{
				assert(c < actnum_count);
				act_cell_ind[c] = i;
				c++;
			}

		msg = stringFormatArr("Active cells: {0:%zu} / {1:%zu}", std::vector<size_t>{actnum_count, grid_size});
	}
	RANK0_SYNCERR_END(comm);
	actnum_loaded = true;

	// sync all the data loaded
	Bcast_string(msg, 0, comm);
	Bcast_vector(actnum, 0, comm);
	MPI_Bcast(&actnum_count, 1, MPI_LONG_LONG, 0, comm);

	if (actnum_count >= (size_t)INT_MAX)
		throw Exception("Array size exceeds INT_MAX in CornGrid::LoadACTNUM");

	MPI_count_displ(comm, actnum_count, counts_act, displs_act);
	assert(rank < (int)counts_act.size());
	assert(rank < (int)displs_act.size());

	act_cell_ind_local = std::vector<int>(counts_act[rank]);
	MPI_Scatterv(act_cell_ind.data(), counts_act.data(), displs_act.data(), MPI_INT, act_cell_ind_local.data(), act_cell_ind_local.size(), MPI_INT, 0, comm);
	HMMPI::Bcast_vector(act_cell_ind, 0, comm);

	return msg;
}
//------------------------------------------------------------------------------------------
void CornGrid::SavePropertyToFile(std::string fname, std::string prop_name, const std::vector<double> &prop)	// saves "prop" in ECLIPSE format
{
	assert(prop.size() > 0);
	const int max_items = 12;
	const char *fmt1 = "      %-11.6g";
	const char *fmt = " %4d*%-11.6g";

	FILE *F = fopen(fname.c_str(), "w");
	if (F == 0)
		throw Exception((std::string)"Cannot open " + fname + "\n");

	fprintf(F, "%s\n", prop_name.c_str());
	double last_val = prop[0];

	int count = 0;			// counts repeated values already read
	int count_str = 0;		// counts items in a string
	for (size_t i = 0; i < prop.size()+1; i++)		// note the "+1"
	{
		if (i == prop.size() || prop[i] != last_val)
		{
			if (count == 1)
				fprintf(F, fmt1, last_val);
			else
				fprintf(F, fmt, count, last_val);
			count = 0;
			if (i < prop.size())
				last_val = prop[i];

			count_str++;
			if (count_str >= max_items)
			{
				fprintf(F, "\n");
				count_str = 0;
			}
		}
		count++;
	}

	fprintf(F, "/");
	if (F) fclose(F);
}
//------------------------------------------------------------------------------------------
// ZCORN numbering scheme for grid X*Y*Z cells
// For each layer k in [0, Z):
// A) First number the upper plane of nodes
//     columns below = (j-)lines of pillars
//     the square box highlights a single pillar
//          	1   	3   	5									-
//     0    	2   	4   	...			-- 1st line of pillars	|
//																	| 1st (i-)line of cells
//      				______										|
//          	2X+1 	|2X+3| 	2X+5								|
//     2X    	2X+2 	|2X+4| 	...			-- 2nd line of pillars	-
//     -------------------------------
//          	4X+1 	|4X+3| 	4X+5								-
//     4X    	4X+2 	|4X+4| 	...			-- 2nd line of pillars	|
//	    				------										| 2nd (i-)line of cells
//																	|
//          	6X+1 	6X+3 	6X+5								|
//     6X    	6X+2 	6X+4 	...			-- 3rd line of pillars	-
//     -------------------------------								-
//     ...															| 3rd (i-)line of cells
//     -------------------------------								-
//     ...
// B) Then number the lower plane of nodes in the same way
//     ....
//     ....
//     ....
std::string CornGrid::fill_cell_coord()				// fills "cell_coord", "cell_coord_local" from coord, zcorn_local, and grid dimensions; returns a short message
{
	assert(grid_loaded);
	const size_t coord_size = 6*(Nx+1)*(Ny+1);			// VOXEL (as taken in HistMatMPI):
	const size_t zcorn_size = 8*Nx*Ny*Nz;				// 0 --- 1				 4 --- 5		0 ---> I
														// |	 |	+ lower face |	   |		|
	assert(coord.size() == coord_size);					// 2 --- 3				 6 --- 7	    |
	if (rank == 0)										//										J
	{
		assert(zcorn.size() == zcorn_size);
		cell_coord = std::vector<double>(zcorn_size*3);	// ORDER: (x,y,z) for 8 vertices of the 1st cell, (x,y,z) for 8 vertices of the second cell,...
	}													// Vertex order in a cell: as in VOXEL above
														// CELLS: i - fastest, k - slowest
	const size_t zcorn_size_local = 8*Nx*Ny*Nz_local;
	assert(zcorn_local.size() == zcorn_size_local);
	cell_coord_local = std::vector<double>(zcorn_size_local*3);		// this array is filled in parallel and then is gathered to 'cell_coord'

	for (size_t k = 0; k < Nz_local; k++)
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
				assert(v[7] < zcorn_size_local);

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
						cell_coord_local[24*ind + 3*n] = x0 + (x1-x0)/(z1-z0)*(zcorn_local[v[n]] - z0);
						cell_coord_local[24*ind + 3*n+1] = y0 + (y1-y0)/(z1-z0)*(zcorn_local[v[n]] - z0);

						cell_coord_local[24*ind + 3*n + 12] = x0 + (x1-x0)/(z1-z0)*(zcorn_local[v[n+4]] - z0);
						cell_coord_local[24*ind + 3*n + 13] = y0 + (y1-y0)/(z1-z0)*(zcorn_local[v[n+4]] - z0);
					}
					else
					{
						cell_coord_local[24*ind + 3*n] = x0;
						cell_coord_local[24*ind + 3*n+1] = y0;

						cell_coord_local[24*ind + 3*n + 12] = x0;
						cell_coord_local[24*ind + 3*n + 13] = y0;
					}
					cell_coord_local[24*ind + 3*n+2] = zcorn_local[v[n]];
					cell_coord_local[24*ind + 3*n+14] = zcorn_local[v[n+4]];
				}
			}

	// gather 'cell_coord'
	std::vector<int> counts = counts_Nz, displs = displs_Nz;	// two arrays specially for 'cell_coord'
	const size_t mult = 24*Nx*Ny;
	if (zcorn_size*3 >= (size_t)INT_MAX)
		throw Exception("Array size exceeds INT_MAX in CornGrid::fill_cell_coord");

	for (auto &v : counts)
		v *= mult;
	for (auto &v : displs)
		v *= mult;
	MPI_Gatherv(cell_coord_local.data(), cell_coord_local.size(), MPI_DOUBLE, cell_coord.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, comm);

	cell_coord_filled = true;

	fill_cell_center();
	return fill_cell_height();
}
//------------------------------------------------------------------------------------------
std::vector<std::vector<NNC>> CornGrid::get_same_layer_NNC(std::string &out_msg)		// based on the mesh and ACTNUM, generates NNCs (where the logically connected cells are not connected in the mesh)
{																	// only the cells with the same "k" are taken for such NNCs
	assert(grid_loaded);											// the PURPOSE is to form NNCs across the faults
																	// the result is significant on comm-rank-0
	out_msg = "";
	if (!cell_coord_filled)
		out_msg += fill_cell_coord() + "\n";

	std::vector<std::vector<NNC>> res;
	if (rank == 0)
	{
		assert(cell_coord.size() > 0);

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
		out_msg += msg;
		if (actnum_loaded)
			out_msg += " (ACTNUM was used)";
		else
			out_msg += " (ACTNUM was not found)";
	}
	Bcast_string(out_msg, 0, comm);

	return res;
}
//------------------------------------------------------------------------------------------
std::vector<double> CornGrid::MarkPinchBottoms() const	// returns Nx*Ny*Nz array, with values =0 or >0, where >0 is for the cells that don't have an active adjacent cell below, in which case
{														// the value is the gap size (sum of heights) between the current cell and the nearest active cell below;
	assert(grid_loaded);								// the result is significant on comm-rank-0
	assert(actnum_loaded);
	assert(cell_coord_filled);
	assert(cell_height.size() > 0);
	assert(cell_center.size() > 0);
	assert(cell_center.size() == cell_height.size()*3);

	std::vector<double> res;							// NOTE: it's assumed there is no empty space between the cells!
	if (rank == 0)
	{
		res = std::vector<double>(Nx*Ny*Nz, 0.0);

		for (size_t j = 0; j < Ny; j++)
			for (size_t i = 0; i < Nx; i++)			// consider cell (i, j, k)
				for (size_t k = 0; k < Nz; k++)
				{
					const size_t ind = Nx*Ny*k + Nx*j + i;													//
																											//	 active		 inactive	 active
					if (actnum[ind] == 0)			// inactive cells are not suitable candidates			//		k	-----	k1	----	k2
						continue;																			//

					size_t k1 = k + 1;				// find the next non-empty cell below
					while (k1 < Nz && cell_height[Nx*Ny*k1 + Nx*j + i] <= min_cell_height)
						k1++;

					if (k1 >= Nz)					// non-empty cell not found
					{
						k = k1;
						continue;
					}

					// now, k1 is the next non-empty cell
					if (actnum[Nx*Ny*k1 + Nx*j + i] == 0)		// case when 'k1' is inactive: more processing
					{
						size_t k2 = k1 + 1;			// find the next non-empty ACTIVE cell below
						while (k2 < Nz && (cell_height[Nx*Ny*k2 + Nx*j + i] <= min_cell_height || actnum[Nx*Ny*k2 + Nx*j + i] == 0))
							k2++;

						if (k2 >= Nz)				// non-empty ACTIVE cell not found
						{
							k = k2;
							continue;
						}

						// now, k and k2 are two active cells separated by a bunch of inactive cells; find the gap size |k - k2|:
						const size_t ind2 = Nx*Ny*k2 + Nx*j + i;
						const double dx = cell_center[3*ind2] - cell_center[3*ind];
						const double dy = cell_center[3*ind2+1] - cell_center[3*ind+1];
						const double dz = cell_center[3*ind2+2] - cell_center[3*ind+2];
						double dist = sqrt(dx*dx + dy*dy + dz*dz) - cell_height[ind]/2 - cell_height[ind2]/2;	// distance between the cell centers minus half the cell thicknesses
						if (dist < 0)
							dist = 0;

						res[ind] = dist;

						k = k2 - 1;					// -1 because of the "for" increment
					}
					else										// case when 'k1' is active: keep going
						k = k1 - 1;								// -1 because of the "for" increment
				}
	}

	return res;
}
//------------------------------------------------------------------------------------------
std::vector<double> CornGrid::ord_krig_final_mult(const Vector2<int> &pts, double R, double r, double rz, double chirad, const HMMPI::Func1D_corr *corr, const Mat &invK_Ys) const
{															// performs (in parallel) the final multiplication needed by ordinary kriging:
	const size_t count_loc = act_cell_ind_local.size();		// [c(x,x1)...c(x,xn),1]*invK_Ys; should be called on all ranks;
	const size_t NG = invK_Ys.JCount();						// the result (NG*actnum_count matrix in row-major order) is significant on rank-0
	const size_t Np = pts.ICount();							// all inputs should be sync on all ranks; "invK_Ys" is (Np+1)*NG matrix
	assert(invK_Ys.ICount() == Np+1);
	assert(cell_coord_filled);

	const double cosx = cos(chirad);
	const double sinx = sin(chirad);

	Mat c_loc(count_loc, Np+1, 0);					// correlation part

	// first, make the correlation part
	for (size_t c = 0; c < count_loc; c++)			// go through all local active points
	{
		const size_t n = act_cell_ind_local[c];
		const size_t i = n % Nx;					// get the (i,j,k) of the cell
		const size_t j = ((n-i)/Nx) % Ny;
		const size_t k = (n-i-Nx*j)/(Nx*Ny);
		assert(i < Nx);
		assert(j < Ny);
		assert(k < Nz);

		c_loc(c, Np) = 1;							// trend part
		for (size_t p = 0; p < Np; p++)				// correlation part
		{
			double dist = calc_scaled_dist(i, j, k, pts(p,0), pts(p,1), pts(p,2), R, r, rz, cosx, sinx);
			c_loc(c, p) = corr->f(dist);
		}
	}

	// second, multiply and gather on rank-0
	Mat res_loc	= c_loc*invK_Ys;					// count_loc*NG;
	assert(res_loc.Length() == count_loc*NG);

	std::vector<double> res, resTr;					// res stores actnum_count*NG row-major matrix
	if (rank == 0)									// resTr stores NG*actnum_count row-major matrix
		resTr = res = std::vector<double>(NG*actnum_count);

	std::vector<int> counts = counts_act, displs = displs_act;		// two arrays specially for 'res'
	if (actnum_count*NG >= (size_t)INT_MAX)
		throw Exception("Array size exceeds INT_MAX in CornGrid::ord_krig_final_mult");

	for (auto &v : counts)
		v *= NG;
	for (auto &v : displs)
		v *= NG;
	MPI_Gatherv(res_loc.Serialize(), res_loc.Length(), MPI_DOUBLE, res.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, comm);

	if (rank == 0)
		for (size_t n = 0; n < NG; n++)
			for (size_t i = 0; i < actnum_count; i++)
				resTr[n*actnum_count + i] = res[i*NG + n];

	return resTr;
}
//------------------------------------------------------------------------------------------
std::vector<double> CornGrid::krig_result_prop(const std::vector<double> &krig_res_loc, int n_loc) const
{														// [DISTR] fills a full property (Nx*Ny*Nz), by extracting grid with local index 'n_loc'
	size_t NG_loc = krig_res_loc.size()/actnum_count;	// from "krig_res_loc" - (NG_loc*actnum_count) local part of output of 'ord_krig_final_mult()'
	assert(krig_res_loc.size()%actnum_count == 0);
	assert((size_t)n_loc < NG_loc);

	std::vector<double> res(Nx*Ny*Nz, 0.0);
	for (size_t i = 0; i < actnum_count; i++)			// go through all active cells
		res[act_cell_ind[i]] = krig_res_loc[n_loc*actnum_count + i];

	return res;
}
//------------------------------------------------------------------------------------------
void CornGrid::find_cell(const double x, const double y, const double z, int &i, int &j, int &k) 	// find cell [i,j,k] containing the point [x,y,z];
{								// call on all ranks; the result is significant on comm-rank-0
	const int delta_i = 5;		// if the CELL with analytical coords (i,j) does not contain the point, window
	const int delta_j = 5;		// [i - delta_i, i + delta_i]*[j - delta_j, j + delta_j] is searched iteratively;
								// if the windowed search fails, the whole grid is searched iteratively
	assert(grid_loaded);
	if (!cell_coord_filled)
		fill_cell_coord();
	if (!state_found)
		analyze();

	bool iserror = false;
	char errmsg[BUFFSIZE];
	if (rank == 0)
	{
		try
		{
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

			const Mat rhs(std::vector<double>{xbar, ybar});

			Mat sol = Q0.Tr()*rhs;
			i = int(sol(0,0)/dx0);			// find analytic (i,j), a.k.a. quick estimate
			j = int(sol(1,0)/dy0);

			if (i < 0 || i >= (int)Nx || j < 0 || j >= (int)Ny)		// quick estimate led to (i,j) outside the grid
			{
				char msg[BUFFSIZE];
				if (!point_inside_boundary_pillars(x, y, t))		// the point is definitely outside
				{
					sprintf(msg, "Point (%.0f, %.0f, %.2f) is laterally outside the grid", x, y, z);
					throw Exception(msg);
				}
				else if (!find_cell_in_window(x, y, 0, Nx, 0, Ny, t, i, j))		// attempt the full grid search
				{
					sprintf(msg, "Fatal error in full grid lateral search: despite identifying the point between the boundary pillars, "
								 "failed to find a grid cell containing the point (%.0f, %.0f, %.2f)", x, y, z);
					throw Exception(msg);
				}
			}

			if (!point_between_pillars(x, y, i, j, t))
			{
				int i0 = i - delta_i;
				int i1 = i + delta_i + 1;
				int j0 = j - delta_j;
				int j1 = j + delta_j + 1;

				if (i0 < 0) i0 = 0;
				if ((size_t)i1 > Nx) i1 = Nx;
				if (j0 < 0) j0 = 0;
				if ((size_t)j1 > Ny) j1 = Ny;

				if (!find_cell_in_window(x, y, i0, i1, j0, j1, t, i, j))	// windowed search
					if (!find_cell_in_window(x, y, 0, Nx, 0, Ny, t, i, j))	// full grid search
					{
						char msg[BUFFSIZE];
						sprintf(msg, "Full grid lateral search: failure to find a grid cell containing the point (%.0f, %.0f, %.2f)", x, y, z);
						throw Exception(msg);
					}
			}

			// now (i,j) is the correct cell, need to find "k"
			k = find_k_lower_bound(i, j, x, y, z);							// [RANK-0]
			if (k == (int)Nz)
			{
				char msg[BUFFSIZE];
				sprintf(msg, "Point (%.0f, %.0f, %.2f) is in column [%d, %d], but is located below the grid", x, y, z, i+1, j+1);
				throw Exception(msg);
			}
			if (!point_below_upper_plane(pointT(x, y, z), i, j, k, this))	// [RANK-0]
			{
				char msg[BUFFSIZE];
				sprintf(msg, "Point (%.0f, %.0f, %.2f) is in column [%d, %d], but cannot be located within the grid cells", x, y, z, i+1, j+1);
				throw Exception(msg);
			}

			// all tests passed, (i,j,k) is correct
		}
		catch (const std::exception &e)
		{
			iserror = true;
			sprintf(errmsg, "%.490s", e.what());
		}
	}

	MPI_Bcast(&pbp_call_count, 1, MPI_LONG_LONG, 0, comm);
	MPI_Bcast(&pinside_call_count, 1, MPI_LONG_LONG, 0, comm);
	MPI_Bcast(&psspace_call_count, 1, MPI_LONG_LONG, 0, comm);

	MPI_Bcast(&iserror, 1, MPI_BYTE, 0, comm);
	MPI_Bcast(errmsg, BUFFSIZE, MPI_CHAR, 0, comm);
	if (iserror)
		throw Exception(errmsg);		// sync
}
//------------------------------------------------------------------------------------------
std::string CornGrid::report_find_cell_stats() const		// info on the auxiliary function call counts within find_cell()
{
	char msg_eng[BUFFSIZE];
	char msg_rus[BUFFSIZE];

	sprintf(msg_eng, "Calls of 'point_between_pillars': %zu, calls of 'point_inside_boundary_pillars': %zu, calls of 'point_in_same_semispace': %zu", pbp_call_count, pinside_call_count, psspace_call_count);
	sprintf(msg_rus, "Проверок 'point_between_pillars': %zu, проверок 'point_inside_boundary_pillars': %zu, проверок 'point_in_same_semispace': %zu", pbp_call_count, pinside_call_count, psspace_call_count);

	return MessageRE(msg_rus, msg_eng);
}
//------------------------------------------------------------------------------------------


}	// namespace HMMPI

