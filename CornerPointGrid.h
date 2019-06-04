/*
 * CornerPointGrid.h
 *
 *  Created on: May 12, 2019
 *      Author: ilya
 */

#ifndef CORNERPOINTGRID_H_
#define CORNERPOINTGRID_H_

#include <string>
#include <vector>
#include "MathUtils.h"

namespace HMMPI
{

//------------------------------------------------------------------------------------------
// a small helper class for NNC
class NNC_point
{
public:
	int i, j, k;

	NNC_point(int i0, int j0, int k0) : i(i0), j(j0), k(k0){};
	bool operator<=(const NNC_point &N2) const;		// comparison is based on {i,j} only
	bool operator==(const NNC_point &N2) const;		// comparison is based on {i,j} only
};
//------------------------------------------------------------------------------------------
// another small helper class for NNC
class NNC
{
public:
	NNC_point N0, N1;

	NNC(int i0, int j0, int k0, int i1, int j1, int k1);
	NNC incr(int di, int dj) const;					// returns NNC where i and j are incremented by di, dj compared to "this" (same increment for both NNC_points)
	bool operator==(const NNC &nnc2) const;			// comparison is based on {i,j} of both points
	bool is_neighbour(const NNC &nnc2) const;		// 'true' if the two NNCs are adjacent
	static void add_NNC_to_array(std::vector<std::vector<NNC>> &NNC_array, NNC n);	// adds "n" to the NNC array, taking connectivity into account (each NNC_array[i] is a connected series of NNCs)
};
//------------------------------------------------------------------------------------------
// class for working with corner point grids: I/O, grid cell location, other operations
class CornGrid
{
private:
	bool grid_loaded;		// 'true' if the grid has been loaded
	bool actnum_loaded;		// 'true' if ACTNUM has been loaded
	size_t Nx, Ny, Nz;		// grid dimensions

	std::vector<double> coord;		// read from COORD
	std::vector<double> zcorn;		// read from ZCORN
	std::vector<int> actnum;		// read from ACTNUM
	std::string actnum_name;		// name of the grid serving as ACTNUM
	double actnum_min;				// when ACTNUM is loaded from the "double" array, values <= "actnum_min" are assigned ACTNUM=0

	bool cell_coord_filled;
	std::vector<double> cell_coord;		// filled by fill_cell_coord() (see for details), contains vertex coords for all cells;
										// ORDER: (x,y,z) for 8 vertices of the 1st cell, (x,y,z) for 8 vertices of the second cell,...

	bool state_found;		// 'true' <=> dx0, dy0, theta0 have been found
	double dx0, dy0;		// grid horizontal cell size
	double theta0;			// grid rotation angle
	Mat Q0;					// grid rotation matrix

	static void ReadGrids(const char *file, std::vector<size_t> len, std::vector<std::vector<double>> &data, std::vector<std::string> S1, std::string S2);
																// reads a number of grids from "file"
																// allocates and fills "data" of size S1.size(), with data[i].size() = len[i]
																// S1[i], S2 - are the start and end markers of "grid[i]" which is loaded to "data[i]"
	static bool ReadTokenComm(FILE *F, char **str, bool &new_line, char *str0, const int str0_len);
																// reads a token from the file (delimited by ' ', '\t', '\r', '\n'), dropping "--..." comments
																// returns true on success, false on failure/EOF
																// the token is saved to "str"
																// set "new_line" = true in the first call, then the function will manage it
																// str0 is a working array, it should have been allocated
	static int StrIndex(const std::string &s, const std::vector<std::string> &vecs);	// index of "s" in vecs[], -1 if not found
	static inline bool scan_two(const char *str, size_t &cnt, double &d, bool &expect_scan_two);	// parses "cnt*d", returns 'true' on success, updates 'expect_scan_two'
	static inline bool scan_one(const char *str, double &d, bool &expect_scan_two);					// parses "d", returns 'true' on success, updates 'expect_scan_two'
	static bool faces_intersect(double a0, double b0, double c0, double d0, double a1, double b1, double c1, double d1);	// 'true' if two faces intersect, the faces are defined by their
																// z-values for two shared pillars (0, 1): face_1 is [a0, b0; a1, b1], face_2 is [c0, d0; c1, d1]
	std::string unify_pillar_z();	// sets z0_ij, z1_ij of the pillars to be const, corrects the corresponding x_ij, y_ij; returns a short message
	std::string analyze();			// finds dx0, dy0, theta0, Q0; returns a short message

	bool point_between_pillars(double x, double y, int i, int j, double t) const;	// 'true' if point (x,y) is between pillars [i,j]-[i+1,j]-[i+1,j+1]-[i,j+1] at depth "t" (fraction)

public:
	CornGrid();
	std::string LoadCOORD_ZCORN(std::string fname, int nx, int ny, int nz, double dx, double dy, bool y_positive, std::string aname, double amin);
								// loads "coord", "zcorn" for the grid (nx, ny, nz) from ASCII format (COORD, ZCORN), returning a small message;
								// [dx, dy] is the coordinates origin, it is added to COORD; "y_positive" indicates positive/negative direction of the Y axis
								// [dx, dy] is [X2, Y2] from the 'MAPAXES', similarly "y_positive" = sign(Y1 - Y2)
								// aname - ACTNUM name, amin - ACTNUM min
	std::string LoadACTNUM(std::string fname);		// loads ACTNUM, should be called after "grid_loaded", returns a small message
													// treats real values > "actnum_min" as 'active'
	void fill_cell_coord();			// fills "cell_coord" from coord, zcorn, and grid dimensions
	std::vector<std::vector<NNC>> get_same_layer_NNC(std::string &out_msg);		// based on the mesh and ACTNUM, generates NNCs (where the logically connected cells are not connected in the mesh)
															// only the cells with the same "k" are taken for such NNCs
															// the PURPOSE is to form NNCs across the faults
													// TODO this function was not thoroughly tested
	void find_cell(double x, double y, double z, int &i, int &j, int &k);		// find cell [i,j,k] containing the point [x,y,z]

	void temp_coord_from_cell(int i, int j, int k, double &x, double &y) const;	// (i,j,k) -> (x,y)	// TODO it's a temp stuff
};
//------------------------------------------------------------------------------------------

}	// namespace HMMPI

#endif /* CORNERPOINTGRID_H_ */
