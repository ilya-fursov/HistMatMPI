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
#include <tuple>
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
// MPI principles:
// all data are sync on all ranks, except those marked by:
// [DISTR] - data member is different on all ranks (e.g. distributed array)
// [RANK-0] - data member is only significant on comm-ranks-0
//
class CornGrid
{
private:
	typedef std::tuple<double, double, double> pointT;		// a helper type for cell search within a column

	std::vector<double> cell_coord_local;		// [DISTR] local part of 'cell_coord', distr w.r.t. "k", for reuse by fill_cell_height(), fill_cell_center()
	std::vector<double> zcorn_local;			// [DISTR] local part of 'zcorn', distr w.r.t. "k"
	std::vector<int> counts_Nz, displs_Nz;		// counts and displs for the distribution w.r.t "k", filled by LoadCOORD_ZCORN()
	std::vector<int> counts_act, displs_act;	// counts and displs for the distribution of active cells, filled by LoadACTNUM()

	MPI_Comm comm;
	int rank;
	bool grid_loaded;		// 'true' if the grid has been loaded
	bool actnum_loaded;		// 'true' if ACTNUM has been loaded
	size_t Nx, Ny, Nz;		// grid dimensions
	size_t Nz_local;		// [DISTR] Nz on a particular rank, after distribution
	mutable size_t pbp_call_count;		// counts point_between_pillars() calls since the last grid loading
	mutable size_t psspace_call_count;	// counts point_in_same_semispace() calls since the last grid loading
	const double delta_Z;	// perturbation for Z in point_in_same_semispace()

	std::vector<double> coord;		// read from COORD
	std::vector<double> zcorn;		// read from ZCORN [RANK-0] -- although, this global array is not used anywhere
	std::vector<int> actnum;		// read from ACTNUM
	std::vector<int> act_cell_ind;	// full indices of active cells, i.e. a map: [0, actnum_count) -> [0, Nx*Ny*Nz)
	std::vector<int> act_cell_ind_local;	// [DISTR] distributed version of 'act_cell_ind'
	std::vector<double> cell_height;	// Nx*Ny*Nz array with cell heights (taken as average height along 4 pillars)
	std::vector<double> cell_center;	// 3*Nx*Ny*Nz array with cell centers (x,y,z); ORDER: coord - fastest, X, Y, Z - slowest
	const double min_cell_height;		// cells with heights <= this value are considered empty
	std::string actnum_name;		// name of the grid serving as ACTNUM
	double actnum_min;				// when ACTNUM is loaded from the "double" array, values <= "actnum_min" are assigned ACTNUM=0
	size_t actnum_count;

	bool cell_coord_filled;
	std::vector<double> cell_coord;		// [RANK-0] filled by fill_cell_coord() (see for details), contains vertex coords for all cells;
										// ORDER: (x,y,z) for 8 vertices of the 1st cell, (x,y,z) for 8 vertices of the second cell,...

	bool state_found;		// 'true' <=> dx0, dy0, theta0 have been found
	double dx0, dy0;		// grid horizontal cell size
	double theta0;			// grid rotation angle
	Mat Q0;					// grid rotation matrix

public:
	static void ReadGrids(const char *file, std::vector<size_t> len, std::vector<std::vector<double>> &data, std::vector<std::string> S1, std::string S2);
																// reads a number of grids from "file"
																// allocates and fills "data" of size S1.size(), with data[i].size() = len[i]
																// S1[i], S2 - are the start and end markers of "grid[i]" which is loaded to "data[i]"
protected:
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
	std::string fill_cell_height();	// fills "cell_height", returns a short message
	void fill_cell_center();		// fills "cell_center"

public:
	void xyz_from_cell_ijk(int i, int j, int k, double &x, double &y, double &z) const;				// (i,j,k) -> (x,y,z) [RANK-0]

protected:
	void xyz_from_cell_ijk_local(int i, int j, int k, double &x, double &y, double &z) const;		// (i,j,k_local) -> (x,y,z)

	bool point_between_pillars(double x, double y, int i, int j, double t) const;	// 'true' if point (x,y) is between pillars [i,j]-[i+1,j]-[i+1,j+1]-[i,j+1] at depth "t" (fraction)
	bool find_cell_in_window(double x, double y, int i0, int i1, int j0, int j1, double t, int &ii, int &jj);	// iteratively searches the cell index window [i0, i1)*[j0, j1)
									// for the first encounter of cell [ii, jj] containing the point (x, y); uses point_between_pillars() test; returns "true" on success

	bool point_in_same_semispace(double x, double y, double z, int i, int j, int k, int v0, int v1, int v2, int vt, double shift) const;	// [RANK-0]
									// for cell (i,j,k) consider the voxel vertices v0, v1, v2, vt = [0, 8)
									// return "true" if (x,y,z) is non-strictly in the same semispace relative to the plane span{v0,v1,v2} as "vt"
									// if v0, v1, v2, vt are all in one plane, vt.z += shift is used for testing
	static bool point_below_lower_plane(const pointT &X0, int i, int j, int k, const CornGrid *grid);	// [RANK-0] "true" if X0=(x,y,z) is strictly below the lower plane of cell (i,j,k)
	static bool point_below_upper_plane(const pointT &X0, int i, int j, int k, const CornGrid *grid);	// [RANK-0] "true" if X0=(x,y,z) is non-strictly below the upper plane of cell (i,j,k)
	int find_k_lower_bound(int i, int j, double x, double y, double z) const;		// [RANK-0] for column (i,j) find the smallest "k" such that
									// (x,y,z) is above the lower plane of cell (i,j,k), returns Nz if not found; binary search is used here
	void calc_strat_dist(int i1, int j1, int k1, int i2, int j2, int k2, double &dx, double &dy, double &dz) const;		// calculates 3D radius-vector (dx, dy, dz)
									// between cells (i1, j1, k1) and (i2, j2, k2), flattening the stratigraphy
									// uses 'cell_center', and can be called on any RANK
public:
	double calc_scaled_dist(int i1, int j1, int k1, int i2, int j2, int k2, double Rmaj, double rmin, double rz, double cosx, double sinx) const;	// calculates the scaled distance for use in 1D covariance function
									// uses calc_strat_dist(), the 3D variogram radii Rmaj, rmin, rz, and cos/sin of angle chi
public:
	CornGrid(MPI_Comm c);
	std::string LoadCOORD_ZCORN(std::string fname, int nx, int ny, int nz, double dx, double dy, bool y_positive, std::string aname, double amin);
								// loads "coord", "zcorn" for the grid (nx, ny, nz) from ASCII format (COORD, ZCORN), returning a small message;
								// [dx, dy] is the coordinates origin, it is added to COORD; "y_positive" indicates positive/negative direction of the Y axis
								// [dx, dy] is [X2, Y2] from the 'MAPAXES', similarly "y_positive" = sign(Y1 - Y2)
								// aname - ACTNUM name, amin - ACTNUM min
								// all reading is done by comm-rank-0
	std::string LoadACTNUM(std::string fname);		// loads ACTNUM, should be called after "grid_loaded", returns a small message
													// treats real values > "actnum_min" as 'active'
													// all reading is done by comm-rank-0
	static void SavePropertyToFile(std::string fname, std::string prop_name, const std::vector<double> &prop);		// saves "prop" in ECLIPSE format
	std::string fill_cell_coord();			// fills "cell_coord", "cell_coord_local" from coord, zcorn_local, and grid dimensions; returns a short message
	std::vector<std::vector<NNC>> get_same_layer_NNC(std::string &out_msg);		// based on the mesh and ACTNUM, generates NNCs (where the logically connected cells are not connected in the mesh)
															// only the cells with the same "k" are taken for such NNCs
															// the PURPOSE is to form NNCs across the faults
															// the result is significant on comm-rank-0
													// TODO this function was not thoroughly tested

	std::vector<double> MarkPinchBottoms() const;	// returns Nx*Ny*Nz array, with values = 0 or 1, where 1 is for the cells which don't have active adjacent cell below
													// the result is significant on comm-rank-0
	std::vector<double> ord_krig_final_mult(const Vector2<int> &pts, double R, double r, double rz, double chirad, const HMMPI::Func1D_corr *corr, const Mat &invK_Ys) const;
													// performs (in parallel) the final multiplication needed by ordinary kriging:
													// [c(x,x1)...c(x,xn),1]*invK_Ys; should be called on all ranks;
													// the result (NG*actnum_count matrix in row-major order) is significant on rank-0
													// all inputs should be sync on all ranks; "invK_Ys" is (Np+1)*NG matrix
	std::vector<double> krig_result_prop(const std::vector<double> &krig_res_loc, int n_loc) const;	// [DISTR] fills a full property (Nx*Ny*Nz), by extracting grid with local index 'n_loc'
																									// from "krig_res_loc" - (NG_loc*actnum_count) local part of output of 'ord_krig_final_mult()'
	void find_cell(const double x, const double y, const double z, int &i, int &j, int &k);	// find cell [i,j,k] containing the point [x,y,z]; the result is significant on comm-rank-0
	std::string report_find_cell_stats() const;		// info on the auxiliary function call counts within find_cell()
	bool IsCellCoordFilled() const {return cell_coord_filled;};
	bool IsActnumLoaded() const {return actnum_loaded;};
	size_t GetActnumCount() const {return actnum_count;};
};
//------------------------------------------------------------------------------------------

}	// namespace HMMPI

#endif /* CORNERPOINTGRID_H_ */
