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
protected:
	NNC_point N0, N1;

public:
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
	std::vector<double> cell_coord;		// filled by fill_cell_coord() (see for details), contains vertex coords for all cells;
										// ORDER: (x,y,z) for 8 vertices of the 1st cell, (x,y,z) for 8 vertices of the second cell,...

	bool state_found;		// 'true' <=> dx0, dy0, theta0 have been found
	double dx0, dy0;		// grid horizontal cell size
	double theta0;			// grid rotation angle


	void ReadGrids(const char *file, std::vector<size_t> len, std::vector<std::vector<double>> &data, std::vector<std::string> S1, std::string S2);
																// reads a number of grids from "file"
																// allocates and fills "data" of size S1.size(), with data[i].size() = len[i]
																// S1[i], S2 - are the start and end markers of "grid[i]" which is loaded to "data[i]"
	bool ReadTokenComm(FILE *F, char **str, bool &new_line, char *str0, const int str0_len);
																// reads a token from the file (delimited by ' ', '\t', '\r', '\n'), dropping "--..." comments
																// returns true on success, false on failure/EOF
																// the token is saved to "str"
																// set "new_line" = true in the first call, then the function will manage it
																// str0 is a working array, it should have been allocated
	int StrIndex(const std::string &s, const std::vector<std::string> &vecs);	// index of "s" in vecs[], -1 if not found
	inline bool scan_two(const char *str, size_t &cnt, double &d, bool &expect_scan_two);	// parses "cnt*d", returns 'true' on success, updates 'expect_scan_two'
	inline bool scan_one(const char *str, double &d, bool &expect_scan_two);				// parses "d", returns 'true' on success, updates 'expect_scan_two'
	bool faces_intersect(double a0, double b0, double c0, double d0, double a1, double b1, double c1, double d1);	// 'true' if two faces intersect, the faces are defined by their
																// z-values for two shared pillars (0, 1): face_1 is [a0, b0; a1, b1], face_2 is [c0, d0; c1, d1]



		public:	// TODO temp
		std::string unify_pillar_z();	// sets z0_ij, z1_ij of the pillars to be const, corrects the corresponding x_ij, y_ij; returns a short message


		void temp_out_pillars() const;	// TODO temp
		void temp_out_zcorn() const;	// TODO temp


public:
	CornGrid();
	std::string LoadCOORD_ZCORN(std::string fname, int nx, int ny, int nz, double dx, double dy);	// loads "coord", "zcorn" for the grid (nx, ny, nz) from ASCII format (COORD, ZCORN)
																									// [dx, dy] is the coordinates origin, it is added to COORD
																									// a small message is returned by this function
	std::string LoadACTNUM(std::string fname);		// loads ACTNUM, should be called after "grid_loaded", returns a small message
													// treats positive real values as 'active'
	void fill_cell_coord();			// fills "cell_coord" from coord, zcorn, and grid dimensions
};
//------------------------------------------------------------------------------------------

}	// namespace HMMPI

#endif /* CORNERPOINTGRID_H_ */
