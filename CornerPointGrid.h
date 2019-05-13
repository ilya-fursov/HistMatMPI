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
// class for working with corner point grids: I/O, grid cell location, other operations
class CornGrid
{
private:
	size_t Nx, Ny, Nz;		// grid dimensions

	std::vector<double> coord;		// read from COORD
	std::vector<double> zcorn;		// read from ZCORN
	std::vector<double> cell_coord;		// filled by fill_cell_coord(), contains vertex coords for all cells;
										// ORDER: (x,y,z) for 8 vertices of the 1st cell, (x,y,z) for 8 vertices of the second cell,...





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


public:
	std::string LoadCOORD_ZCORN(std::string fname, int nx, int ny, int nz, double dx, double dy);	// loads "coord", "zcorn" for the grid (nx, ny, nz) from ASCII format (COORD, ZCORN)
																									// [dx, dy] is the coordinates origin, it is added to COORD
																									// a small message is returned by this function
	void fill_cell_coord();			// fills "cell_coord" from coord, zcorn, and grid dimensions
};
//------------------------------------------------------------------------------------------

}	// namespace HMMPI

#endif /* CORNERPOINTGRID_H_ */
