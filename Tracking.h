/*
 * Tracking.h
 *
 *  Created on: Mar 20, 2013
 *      Author: ilya
 */

#ifndef TRACKING_H_
#define TRACKING_H_

#include <string>
#include "Vectors.h"

// some legacy stuff for 2D grids

//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
class Grid2D
{
private:
	std::string delim;
	int countX;
	int countY;
public:
	double **data;
	int **flag;          // 0 - íåàêòèâíàÿ ÿ÷åéêà, 1 - àêòèâíàÿ ÿ÷åéêà
private:
	double dx;
	double dy;
	double x0;
	double y0;
	std::string undef_val;

	void ClearData();                   // î÷èùàåò data è flag
    void CopySmallFrom(const Grid2D &g);// êîïèðóåò âñå, êðîìå data, flag; íàäî âûçûâàòü ClearData ïåðåä çàïóñêîì ýòîé ôóíêöèè (ò.ê. ïðåæíèå countX, countY èçìåíÿòñÿ)
public:
	// äâå ôàéëîâûå ôóíêöèè, ðàáîòàþùèå ñ ôîðìàòîì CPS-3 (ASCII)
	void LoadFromFile(std::string fname);    // çàãðóçêà äàííûõ èç ôàéëà (íåóïðàâëÿåìûå äàííûå ïðè íåîáõîäèìîñòè óäàëÿþòñÿ)
	static HMMPI::Vector2<double> LoadPilotPoints(std::string fname);  // çàãðóçêà ïèëîòíûõ òî÷åê èç ôàéëà
	void ReadActnum(std::string fegrid);     // ÷òåíèå ACTNUM èç ôàéëà fegrid; this ä.á. èíèöèàëèçèðîâàí, à åãî ïàðàìåòðû ãðèäà - îïðåäåëåíû
	void ReadProp(std::string funrst, std::string prop, int step, const Grid2D &act);	// ÷òåíèå êóáà prop èç funrst íà øàãå step; this èíèöèàëèçèðóåòñÿ ïî ãðèäó àêò. ÿ÷ååê act
									    // prop ì.á. íàïðèìåð 'PRESSURE', 'SWAT    ' è ò.ä.
	void SaveToFile(std::string fname);
	void SaveToTextFile(std::string fname, double undef);   // ñîõðàíåíèå â âèäå òàáëèöû â òåêñòîâûé ôàéë
	void SaveProp(std::string fname, std::string propname, double undef);		// ñîõðàíåíèå äàííûõ â ôîðìàòå êóáîâ ýêëèïñà
	void SaveProp3D(std::string fname, std::string propname, double undef, int Nz);		// (eng) copies 2D grid for Nz layers
	void SaveIntProp(std::string fname, std::string propname, double undef);	// ñîõðàíåíèå öåëî÷èñëåííûõ äàííûõ â ôîðìàòå êóáîâ ýêëèïñà
	void VectorToFile(std::string fname);  	// çàïèñûâàåò â ôàéë ïîñëåäîâàòåëüíûå çíà÷åíèÿ (âêëþ÷àÿ íåàêòèâíûå)
	void VectorToFileAct(std::string fname); // çàïèñûâàåò â ôàéë ïîñëåäîâàòåëüíûå çíà÷åíèÿ (âêëþ÷àÿ íåàêòèâíûå) (eng) only active!

	~Grid2D();     						// äåñòðóêòîð
	Grid2D();						    // êîíñòðóêòîð
	Grid2D(const Grid2D &g);            // êîíñòðóêòîð, êîïèðóåò âñå, êðîìå data, flag (îíè áåðóòñÿ = 0)
	Grid2D(Grid2D &&g);
	const Grid2D &operator=(const Grid2D &g);
	const Grid2D &operator=(Grid2D &&g);
	Grid2D(const double *d, int cX, int cY, double DX, double DY, double X0, double Y0, std::string U);	// data[i][j] = d[i + j*cX]
	//Grid2D ^Copy();                   // êîïèðóåò âñå, âêëþ÷àÿ äàííûå â data, flag

	int CountX() const {return countX;};
	int CountY() const {return countY;};
	std::vector<double> GetParams();         // âîçâðàùàåò ìàññèâ {x0, y0, dx, dy}
	void SetGeom(double x0_, double y0_, double dx_, double dy_);

	void InitData(int cX, int cY);      // î÷èùàåò ñòàðûå äàííûå (ClearData), èíèöèàëèçèðóåò data è flags (âûäåëÿåò ïàìÿòü, íî íå çàäàåò çíà÷åíèÿ)
	void SetVal(double d);              // çàäàåò äëÿ âñåõ ÿ÷ååê data = d, flag = 1
	void SetUndefVal();                 // çàäàåò äëÿ âñåõ ÿ÷ååê flag = 0

	double min();  		// ìèíèìóì â data[]
	double max();  		// ìàêñèìóì â data[]
	double min_all();  	// ìèíèìóì â data[], (eng) all cells
	double DX();   		// dx
	double DY();   		// dy

	static bool CheckSizes(std::vector<const Grid2D*> GS);    // GS - ìàññèâ îáúåêòîâ òèïà Grid2D
	static void SynchronizeActive(std::vector<Grid2D*> GS);   // GS - ìàññèâ îáúåêòîâ òèïà Grid2D

	HMMPI::Vector2<double> KrigingCoeffs(const HMMPI::Vector2<double> &pts, double Vchi, double VR, double Vr, double sill, double nugget, std::string Vtype, std::string KRIGtype);
	             // âû÷èñëåíèå äëÿ òî÷åê äàííîãî ãðèäà êîýôôèöèåíòîâ êðèãèíãà
	             // ñ èñõîäíûìè òî÷êàìè pts, âàðèîãðàììîé (Vchi, VR, Vr, Vtype, KRIGtype) (eng)
	static double VarSpher(double h, double range, double sill, double n);  // ñôåðè÷åñêàÿ âàðèîãðàììà
	static double VarExp(double h, double range, double sill, double n);    // ýêñïîíåíöèàëüíàÿ âàðèîãðàììà
	static double VarGauss(double h, double range, double sill, double n);  // ãàóññîâà âàðèîãðàììà
	static double EllipseTransform(double Dx, double Dy, double chi, double R, double r);  // ïåðåâîä (Dx, Dy) â ñèñòåìó êîîðäèíàò ýëëèïñà (chi, R, r)
	Grid2D Kriging(const HMMPI::Vector2<double> &pts, const std::vector<double> &vals, const HMMPI::Vector2<double> &coeffs, int K_type); // êðèãèíã ïî òî÷êàì pts
	             // ñî çíà÷åíèÿìè vals, êîýôôèöèåíòàìè êðèãèíãà coeffs, êàðòîé ñðåäíèõ, õðàíÿùåéñÿ â òåêóùåì ãðèäå;
	             // åñëè êàêèå-òî òî÷êè pts âûõîäÿò çà ãðàíèöó ãðèäà, â íèõ ïðåäïîëàãàåòñÿ ñðåäíåå = 0
				 // K_type = 0 (SIM), 1(ORD)
	std::vector<double> KrigingMatch(const HMMPI::Vector2<double> &coeffs, int K_type);		// (eng) returns vals which best match this grid
	double SumValsAll() const;			// ñóììà çíà÷åíèé (èç data), âêëþ÷àÿ íåàêòèâíûå ÿ÷åéêè
	void Subtract(const Grid2D &g);
	void Mult(const Grid2D &g);
	void Plus(const Grid2D &g);
	int SignStats();	// -1 if all negative, 1 if all positive, 0 if both or none
	void Round();
	static double *Serialize(const std::vector<Grid2D> &V);				// serializes only data; delete result after use!
	static void Deserialize(std::vector<Grid2D> &V, const double *d);	// deserializes only data; sizes must be correct
};
//------------------------------------------------------------------------------------------

#endif /* TRACKING_H_ */
