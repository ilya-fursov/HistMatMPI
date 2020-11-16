/*
 * Tracking.cpp
 *
 *  Created on: Mar 21, 2013
 *      Author: ilya
 */

#include "MathUtils.h"
#include "Tracking.h"
#include "Parsing.h"
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>


const double BIGNUM = 1e308;
const int BIGINT = 2147483647;
const double Var_a = 0.33;
const double GRAV = 9.80665;   // óñêîðåíèå ñâîáîäíîãî ïàäåíèÿ ì/ñ2
const int MAXINT = 2147483647;
//------------------------------------------------------------------------------------------
// Grid2D
//------------------------------------------------------------------------------------------
Grid2D::Grid2D()
{
	delim = " \t\r";

	countX = 0;
	countY = 0;
	data = 0;
	flag = 0;
	undef_val = "0.1E+31";
	x0 = y0 = dx = dy = 0;
}
//------------------------------------------------------------------------------------------
Grid2D::Grid2D(const double *d, int cX, int cY, double DX, double DY, double X0, double Y0, std::string U)
{
	delim = " \t\r";

	countX = cX;
	countY = cY;
	dx = DX;
	dy = DY;
	x0 = X0;
	y0 = Y0;
	undef_val = U;

	data = new double*[countX];
	flag = new int*[countX];
	for (int i = 0; i < countX; i++)
	{
		data[i] = new double[countY];
		flag[i] = new int[countY];
		for (int j = 0; j < countY; j++)
		{
			data[i][j] = d[i + j*countX];
			flag[i][j] = 1;
		}
	}
}
//------------------------------------------------------------------------------------------
Grid2D::Grid2D(const Grid2D &g) : data(0), flag(0)
{
	*this = g;
}
//------------------------------------------------------------------------------------------
Grid2D::Grid2D(Grid2D &&g) : data(0), flag(0)
{
	*this = std::move(g);
}
//------------------------------------------------------------------------------------------
void Grid2D::SetGeom(double x0_, double y0_, double dx_, double dy_)
{
	x0 = x0_;
	y0 = y0_;
	dx = dx_;
	dy = dy_;
}
//------------------------------------------------------------------------------------------
const Grid2D &Grid2D::operator=(const Grid2D &g)
{
	if (this != &g)
	{
		ClearData();
		CopySmallFrom(g);

		data = new double*[countX];
		flag = new int*[countX];
		for (int i = 0; i < countX; i++)
		{
			data[i] = new double[countY];
			flag[i] = new int[countY];
			for (int j = 0; j < countY; j++)
			{
				data[i][j] = g.data[i][j];
				flag[i][j] = g.flag[i][j];
			}
		}
	}
	return *this;
}
//------------------------------------------------------------------------------------------
const Grid2D &Grid2D::operator=(Grid2D &&g)
{
	if (this != &g)
	{
		ClearData();
		CopySmallFrom(g);

		data = g.data;
		flag = g.flag;

		g.countX = 0;
		g.countY = 0;
		g.data = 0;
		g.flag = 0;
	}
	return *this;
}
//------------------------------------------------------------------------------------------
// âûçâàòü ClearData ïåðåä çàïóñêîì ýòîé ôóíêöèè!
void Grid2D::CopySmallFrom(const Grid2D &g)
{
	delim = g.delim;
	countX = g.countX;
	countY = g.countY;
	dx = g.dx;
	dy = g.dy;
	x0 = g.x0;
	y0 = g.y0;
	undef_val = g.undef_val;
}
//------------------------------------------------------------------------------------------
void Grid2D::InitData(int cX, int cY)
{
	ClearData();

	countX = cX;
	countY = cY;

	data = new double*[countX];
	flag = new int*[countX];
	for (int i = 0; i < countX; i++)
	{
		data[i] = new double[countY];
		flag[i] = new int[countY];
	}
}
//------------------------------------------------------------------------------------------
void Grid2D::SetVal(double d)
{
	if ((data != 0)&&(flag != 0)&&(countX > 0)&&(countY > 0))
	{
		for (int i = 0; i < countX; i++)
		{
			for (int j = 0; j < countY; j++)
			{
				data[i][j] = d;
				flag[i][j] = 1;
			}
		}
	}
}
//------------------------------------------------------------------------------------------
void Grid2D::SetUndefVal()
{
	if ((data != 0)&&(flag != 0)&&(countX > 0)&&(countY > 0))
	{
		for (int i = 0; i < countX; i++)
			for (int j = 0; j < countY; j++)
				flag[i][j] = 0;
	}
}
//------------------------------------------------------------------------------------------
Grid2D::~Grid2D()   // äåñòðóêòîð
{
	ClearData();
}
//------------------------------------------------------------------------------------------
void Grid2D::ClearData()
{
	if (data != 0)
	{
		for (int i = 0; i < countX; i++)
		{
			delete [] data[i];
		}
		delete [] data;

		data = 0;
	}
	if (flag != 0)
	{
		for (int i = 0; i < countX; i++)
		{
			delete [] flag[i];
		}
		delete [] flag;

		flag = 0;
	}
}
//------------------------------------------------------------------------------------------
double Grid2D::min()
{
	double min = BIGNUM;
	if ((data != 0)&&(countX > 0)&&(countY > 0))
	{
		for (int i = 0; i < countX; i++)
			for (int j = 0; j < countY; j++)
				if ((flag[i][j])&&(data[i][j] < min))
					min = data[i][j];
	}
	return min;
}
//------------------------------------------------------------------------------------------
double Grid2D::max()
{
	double max = -BIGNUM;
	if ((data != 0)&&(countX > 0)&&(countY > 0))
	{
		for (int i = 0; i < countX; i++)
			for (int j = 0; j < countY; j++)
				if ((flag[i][j])&&(data[i][j] > max))
					max = data[i][j];
	}
	return max;
}
//------------------------------------------------------------------------------------------
double Grid2D::min_all()
{
	double min = BIGNUM;
	if ((data != 0)&&(countX > 0)&&(countY > 0))
	{
		for (int i = 0; i < countX; i++)
			for (int j = 0; j < countY; j++)
				if (data[i][j] < min)
					min = data[i][j];
	}
	return min;
}
//------------------------------------------------------------------------------------------
double Grid2D::DX()
{
	return dx;
}
//------------------------------------------------------------------------------------------
double Grid2D::DY()
{
	return dy;
}
//------------------------------------------------------------------------------------------
bool Grid2D::CheckSizes(std::vector<const Grid2D*> GS)
{
	int L = GS.size();
	if (L > 0)
	{
		int cX = GS[0]->countX;
		int cY = GS[0]->countY;
		for (int i = 1; i < L; i++)
		{
			if ((GS[i]->countX != cX)||(GS[i]->countY != cY))
				return false;
		}
		return true;
	}
	else
		return true;
}
//------------------------------------------------------------------------------------------
void Grid2D::SynchronizeActive(std::vector<Grid2D*> GS)
{
	int L = GS.size();
	if (L > 0)
	{
		int cX = GS[0]->countX;
		int cY = GS[0]->countY;

		for (int i = 0; i < cX; i++)
		{
			for (int j = 0; j < cY; j++)
			{
				int FL = 1;
				for (int g = 0; (g < L)&&(FL); g++)   // îïðåäåëÿåì îáùèé ôëàã FL
					FL &= GS[g]->flag[i][j];

				for (int g = 0; g < L; g++)           // ìåíÿåì âñå ôëàãè íà FL
					GS[g]->flag[i][j] = FL;
			}
		}
	}
}
//------------------------------------------------------------------------------------------
std::vector<double> Grid2D::GetParams()
{
	return std::vector<double>{x0, y0, dx, dy};
}
//------------------------------------------------------------------------------------------
// pts[i, j] -> i - íîìåð òî÷êè, {j = 0} - êîîðäèíàòà x, {j = 1} - êîîðäèíàòà y
HMMPI::Vector2<double> Grid2D::KrigingCoeffs(const HMMPI::Vector2<double> &pts, double Vchi, double VR, double Vr, double sill, double nugget, std::string Vtype, std::string KRIGtype)
{
	int pcount = pts.ICount();
	int Gsize = pcount;
	if (KRIGtype == "ORD")
		Gsize += 1;
	long NM = countX*countY;

	HMMPI::Vector2<double> res(Gsize, NM);				// ðåøåíèå
	HMMPI::Vector2<double> Gamma(Gsize, Gsize);		 	// ìàòðèöà
	HMMPI::Vector2<double> A(Gsize, NM);				// ïðàâàÿ ÷àñòü

	// çàïîëíåíèå ìàòðèö
	for (int i = 0; i < Gsize; i++)
	{
		for (int j = 0; j < Gsize; j++)
		{
			if (i < pcount && j < pcount)
			{
				double val;
				double h = EllipseTransform(pts(i, 0) - pts(j, 0), pts(i, 1) - pts(j, 1), Vchi, VR, Vr);

				if (Vtype == "SPHER")
					val = VarSpher(h, 1, sill, nugget);
				else if (Vtype == "EXP")
					val = VarExp(h, 1, sill, nugget);
				else if (Vtype == "GAUSS")
					val = VarGauss(h, 1, sill, nugget);
				else
					throw HMMPI::Exception("Íåêîððåêòíûé òèï âàðèîãðàììû â Grid2D::KrigingCoeffs",
									"Incorrect variogram type in Grid2D::KrigingCoeffs");

				Gamma(i, j) = val - sill;
			}
			else if (i < pcount || j < pcount)
				Gamma(i, j) = 1;
			else
				Gamma(i, j) = 0;
		}
	}

	for (int i = 0; i < Gsize; i++)
	{
		if (i < pcount)
		{
			for (long j = 0; j < NM; j++)
			{
				int indx = j % countX;
				int indy = j / countX;
				double x = x0 + dx*(double(indx) + 0.5);
				double y = y0 + dy*(double(indy) + 0.5);

				double val;
				double h = EllipseTransform(pts(i, 0) - x, pts(i, 1) - y, Vchi, VR, Vr);

				if (Vtype == "SPHER")
					val = VarSpher(h, 1, sill, nugget);
				else if (Vtype == "EXP")
					val = VarExp(h, 1, sill, nugget);
				else if (Vtype == "GAUSS")
					val = VarGauss(h, 1, sill, nugget);
				else
					throw HMMPI::Exception("Íåêîððåêòíûé òèï âàðèîãðàììû â Grid2D::KrigingCoeffs",
							 	    "Incorrect variogram type in Grid2D::KrigingCoeffs");

				A(i, j) = val - sill;
				res(i, j) = 0;
			}
		}
		else
			for (long j = 0; j < NM; j++)
			{
				A(i, j) = 1;
				res(i, j) = 0;
			}
	}

	// ðåøåíèå - âïåðåä
	for (int i = 0; i < Gsize-1; i++)
	{
		// ïîèñê ìàêñ. çíà÷åíèÿ
		int max_ind = i;
		double max_val = fabs(Gamma(i, i));
		for (int k = i+1; k < Gsize; k++)
		{
			if (fabs(Gamma(k, i)) > max_val)
			{
				max_val = fabs(Gamma(k, i));
				max_ind = k;
			}
		}

		// ïåðåñòàíîâêà
		if (max_ind != i)
		{
			for (int k = i; k < Gsize; k++)
			{
				double swp = Gamma(i, k);
				Gamma(i, k) = Gamma(max_ind, k);
				Gamma(max_ind, k) = swp;
			}
			for (long k = 0; k < NM; k++)
			{
				double swp = A(i, k);
				A(i, k) = A(max_ind, k);
				A(max_ind, k) = swp;
			}
		}

		// ñëîæåíèå ñòðîê
		for (int k = i+1; k < Gsize; k++)
		{
			double r = -Gamma(k, i)/Gamma(i, i);
			Gamma(k, i) = 0;
			for (int n = i+1; n < Gsize; n++)
			{
				Gamma(k, n) += r * Gamma(i, n);
			}
			for (long n = 0; n < NM; n++)
			{
				A(k, n) += r * A(i, n);
			}
		}
	}

	// ðåøåíèå - íàçàä
	for (long j = 0; j < NM; j++)
	{
		for (int i = Gsize-1; i >= 0; i--)
		{
			double sum = 0;
			for (int k = i+1; k < Gsize; k++)
			{
				sum += Gamma(i, k)*res(k, j);
			}
			res(i, j) = (A(i, j) - sum)/Gamma(i, i);
		}
	}

	return res;
}
//------------------------------------------------------------------------------------------
double Grid2D::VarSpher(double h, double range, double sill, double n)
{
	if (h < range)
	{
		double hr = h/range;
		return (sill - n)*(1.5*hr - 0.5*hr*hr*hr) + n;
	}
	else
		return sill;
}
//------------------------------------------------------------------------------------------
double Grid2D::VarExp(double h, double range, double sill, double n)
{
	return (sill - n)*(1 - exp(-h/(range*Var_a))) + n;
}
//------------------------------------------------------------------------------------------
double Grid2D::VarGauss(double h, double range, double sill, double n)
{
	double hr = h/range;
	return (sill - n)*(1 - exp(-hr*hr/Var_a)) + n;
}
//------------------------------------------------------------------------------------------
double Grid2D::EllipseTransform(double Dx, double Dy, double chi, double R, double r)
{
	double Cos = cos(chi);
	double Sin = sin(chi);
	double x1 = Cos/R*Dx + Sin/R*Dy;
	double y1 = -Sin/r*Dx + Cos/r*Dy;
	return sqrt(x1*x1 + y1*y1);
}
//------------------------------------------------------------------------------------------
Grid2D Grid2D::Kriging(const HMMPI::Vector2<double> &pts, const std::vector<double> &vals, const HMMPI::Vector2<double> &coeffs, int K_type)
{
	size_t len = vals.size();
	if (K_type != 1)
		throw HMMPI::Exception("Â Grid2D::Kriging äîïóñêàåòñÿ òîëüêî K_type == 1",
						"Only K_type == 1 is allowed in Grid2D::Kriging");

	if ((len != pts.ICount())||(len + 1 != coeffs.ICount()))
		throw HMMPI::Exception("Íå ñîâïàäàþò ðàçìåðû ìàññèâîâ âõîäíûõ äàííûõ â Grid2D::Kriging",
						"Incorrect dimensions of input arrays in Grid2D::Kriging");

	// âû÷èòàåì ñðåäíèå èç ïèëîòíûõ òî÷åê
	std::vector<double> valsnew = vals;

	// âû÷èñëåíèå êîíå÷íîãî ðåçóëüòàòà
	Grid2D res = *this;

	for (int i = 0; i < countX; i++)
	{
		for (int j = 0; j < countY; j++)
		{
			res.flag[i][j] = flag[i][j];
			if (res.flag[i][j])
			{
				long n = long(j)*long(countX) + long(i);
				double sum = 0;
				for (size_t k = 0; k < len; k++)
					sum += coeffs(k, n) * valsnew[k];

				if (K_type == 0)
					res.data[i][j] = sum + data[i][j];
				else
					res.data[i][j] = sum;
			}
			else
				res.data[i][j] = 0;
		}
	}

 	return res;
}
//------------------------------------------------------------------------------------------
std::vector<double> Grid2D::KrigingMatch(const HMMPI::Vector2<double> &coeffs, int K_type)
{
	int Psize = coeffs.ICount();
	int count = Psize;
	if (K_type == 1)	// ORD
		count -= 1;

	size_t NM = countX*countY;
	if (NM != coeffs.JCount())
		throw HMMPI::Exception("Ðàçìåðû ãðèäà è ìàòðèöû êîýôôèöèåíòîâ íå ñîîòâåòñòâóþò äðóã äðóãó â Grid2D::KrigingMatch",
						"Mismatch of arrays dimensions in Grid2D::KrigingMatch");

	HMMPI::Vector2<double> Gamma(count, count);
	std::vector<double> A(count);
	std::vector<double> res(count);

	// çàïîëíÿåì ìàòðèöó è ïðàâóþ ÷àñòü
	for (int i = 0; i < count; i++)
	{
		for (int j = 0; j < count; j++)
		{
			double sum = 0;
			for (size_t k = 0; k < NM; k++)
				sum += coeffs(i, k)*coeffs(j, k);

			Gamma(i, j) = sum;
		}

		double aux = 0;
		for (size_t l = 0; l < NM; l++)
		{
			int ni = l % countX;
			int nj = l / countX;
			aux += coeffs(i, l)*data[ni][nj];
		}
		A[i] = aux;
	}

	// ðåøàåì ñèñòåìó
	for (int i = 0; i < count-1; i++)
	{
		int max_ind = i;
		double max_val = fabs(Gamma(i, i));
		for (int k = i+1; k < count; k++)
		{
			if (fabs(Gamma(k, i)) > max_val)
			{
				max_val = fabs(Gamma(k, i));
				max_ind = k;
			}
		}

		if (max_ind != i)
		{
			double swp;
			for (int k = i; k < count; k++)
			{
				swp = Gamma(i, k);
				Gamma(i, k) = Gamma(max_ind, k);
				Gamma(max_ind, k) = swp;
			}
			swp = A[i];
			A[i] = A[max_ind];
			A[max_ind] = swp;
		}

		for (int k = i+1; k < count; k++)
		{
			double r = -Gamma(k, i)/Gamma(i, i);
			Gamma(k, i) = 0;
			for (int n = i+1; n < count; n++)
				Gamma(k, n) += r * Gamma(i, n);

			A[k] += r * A[i];
		}
	}

	for (int i = count-1; i >= 0; i--)
	{
		double sum = 0;
		for (int k = i+1; k < count; k++)
			sum += Gamma(i, k)*res[k];

		res[i] = (A[i] - sum)/Gamma(i, i);
	}

	return res;
}
//------------------------------------------------------------------------------------------
// Grid2D IO
//------------------------------------------------------------------------------------------
void Grid2D::LoadFromFile(std::string fname)
{
	std::ifstream sr;
	sr.exceptions(std::ios_base::badbit);
	CheckFileOpen(fname);

	try // ÷òåíèå
	{
		sr.open(fname.c_str());
		ClearData();

		// çàãîëîâîê
		std::string line;
		std::vector<std::string> line_el;

		std::getline(sr, line);	// 1
		HMMPI::tokenize(line, line_el, delim, true);
		undef_val = line_el[5];

		std::getline(sr, line);  // 2
		std::getline(sr, line);  // 3
		HMMPI::tokenize(line, line_el, delim, true);
		x0 = HMMPI::StoD(line_el[1]);
		y0 = HMMPI::StoD(line_el[3]);

		std::getline(sr, line);  // 4
		HMMPI::tokenize(line, line_el, delim, true);
		countY = HMMPI::StoL(line_el[1]);
		countX = HMMPI::StoL(line_el[2]);

		std::getline(sr, line);  // 5
		HMMPI::tokenize(line, line_el, delim, true);
		dx = HMMPI::StoD(line_el[1]);
		dy = HMMPI::StoD(line_el[2]);

		std::getline(sr, line);  // 6

		x0 = x0 - dx/2;
		y0 = y0 - dy/2;

		// âûäåëåíèå ïàìÿòè ïîä ìàññèâû
		data = new double*[countX];
		flag = new int*[countX];
		for (int i = 0; i < countX; i++)
		{
			data[i] = new double[countY];
			flag[i] = new int[countY];
		}

		// ÷òåíèå äàííûõ
		int count = 0;
		while (!sr.eof())
		{
			std::getline(sr, line);
			HMMPI::tokenize(line, line_el, delim, true);
			int line_len = line_el.size();
            for (int i = 0; i < line_len; i++)
		    {
			    int y = count % countY;       // y ìåíÿåòñÿ áûñòðåå
			    int x = count / countY;
				double val = HMMPI::StoD(line_el[i]);
			    if (line_el[i] != undef_val)
				{
				    data[x][countY-1-y] = val;
					flag[x][countY-1-y] = 1;
				}
  			    else
				{
					data[x][countY-1-y] = 0;
				    flag[x][countY-1-y] = 0;
				}
			    count++;
		    }
		}
		sr.close();
	}
	catch (...)
	{
		if (sr.is_open())
			sr.close();
		throw;
	}
}
//------------------------------------------------------------------------------------------
HMMPI::Vector2<double> Grid2D::LoadPilotPoints(std::string fname)
{
	std::ifstream sr;
	sr.exceptions(std::ios_base::badbit);
	CheckFileOpen(fname);
	std::string DELIM = " \t\r";	// cannot use member 'delim' because this function is static

	try // ÷òåíèå
	{
		sr.open(fname.c_str());
		std::vector<std::vector<double>> pts;

		while (!sr.eof())
		{
			std::string line;
			std::vector<std::string> line_el;

			std::getline(sr, line);
			HMMPI::tokenize(line, line_el, DELIM, true);

			if (line_el.size() > 0)
			{
				std::vector<double> pp(2);
				for (int i = 0; i < 2; i++)
					pp[i] = HMMPI::StoD(line_el[i]);

				pts.push_back(pp);
			}
		}

		HMMPI::Vector2<double> res(pts.size(), 2);
		for (size_t i = 0; i < pts.size(); i++)
			for (int j = 0; j < 2; j++)
				res(i, j) = pts[i][j];

		sr.close();
		return res;
	}
	catch (...)
	{
		if (sr.is_open())
			sr.close();
		throw;
	}
}
//------------------------------------------------------------------------------------------
void Grid2D::ReadActnum(std::string fegrid)
{
	std::ifstream sr;
	sr.exceptions(std::ios_base::badbit);
	CheckFileOpen(fegrid);

	std::string HDR = "'ACTNUM  '";

	try
	{
		sr.open(fegrid);
		int i = 0, j = 0;
		while (!sr.eof())
		{
			std::string line;
			std::vector<std::string> line_aux;
			std::getline(sr, line);

			size_t i_ = line.find(HDR);
			if (i_ != std::string::npos)
			{
				// ÷èòàåì çàãîëîâîê äëÿ ACTNUM
				HMMPI::tokenize(HMMPI::Trim(line.substr(i_+HDR.length()), delim), line_aux, delim, true);

				int count = HMMPI::StoL(line_aux[0]);
				if (count != countX*countY)
					throw HMMPI::Exception(HMMPI::stringFormatArr(HMMPI::MessageRE("×èñåë â ACTNUM: {0:%d}, ÿ÷ååê â ãðèäå: {1:%d}",
															  "Elements in ACTNUM: {0:%d}, elements in Grid2D: {1:%d}"), std::vector<int>{count, countX*countY}));
				int c = 0;
				while ((c < count)&&(!sr.eof()))
				{
					std::getline(sr, line);
					HMMPI::tokenize(line, line_aux, delim, true);
					for (size_t k = 0; k < line_aux.size(); k++)
					{
						data[i][countY-1-j] = HMMPI::StoD(line_aux[k]);
						flag[i][countY-1-j] = 1;
						c++;
						i++;
						if (i >= countX)
						{
							i = 0;
							j++;
						}
					}
				}
				break;
			}
		}

		sr.close();
	}
	catch (...)
	{
		if (sr.is_open())
			sr.close();
		throw;
	}
}
//------------------------------------------------------------------------------------------
void Grid2D::ReadProp(std::string funrst, std::string prop, int step, const Grid2D &act)
{
	std::ifstream sr;
	sr.exceptions(std::ios_base::badbit);
	CheckFileOpen(funrst);

	std::string STEPHD = "SEQNUM";
	std::string HDR = prop;

	try
	{
		this->CopySmallFrom(act);
		this->InitData(countX, countY);
		this->SetUndefVal();

		sr.open(funrst);
		int i = 0, j = 0;
		int cur_step = 0;
		bool found = false;
		while (!sr.eof())
		{
			std::string line;
			std::vector<std::string> line_aux;
			std::getline(sr, line);
			if (line.find(STEPHD) != std::string::npos)
			{
				// ÷èòàåì çàãîëîâîê äëÿ âðåìåííîãî øàãà
				std::getline(sr, line);
				line = HMMPI::Trim(line, delim);
				cur_step = HMMPI::StoL(line);
			}
			size_t i_ = line.find(HDR);
			if (i_ != std::string::npos)
			{
				// ÷èòàåì çàãîëîâîê äëÿ prop
				if (cur_step == step)
				{
					found = true;
					HMMPI::tokenize(HMMPI::Trim(line.substr(i_+HDR.length()), delim), line_aux, delim, true);
					int count = HMMPI::StoL(line_aux[0]);
					if (count > countX*countY)
						throw HMMPI::Exception(HMMPI::stringFormatArr(HMMPI::MessageRE("Èìååòñÿ áîëüøå çíà÷åíèé ñâîéñòâà ({0:%d}), ÷åì ÿ÷ååê â ãðèäå ({1:%d})",
																  "Number of values ({0:%d}) exceeds the number of Grid2D elements ({1:%d})"), std::vector<int>{count, countX*countY}));
					int c = 0;
					while ((c < count)&&(!sr.eof()))
					{
						std::getline(sr, line);
						HMMPI::tokenize(line, line_aux, delim, true);
						for (size_t k = 0; k < line_aux.size(); k++)
						{
							while (act.data[i][countY-1-j] == 0)
							{
								if (j >= countY)
									throw HMMPI::Exception(HMMPI::stringFormatArr(HMMPI::MessageRE("Èìååòñÿ áîëüøå çíà÷åíèé ñâîéñòâà ({0:%d}), ÷åì àêòèâíûõ ÿ÷ååê ({1:%d})",
																			  "Number of values ({0:%d}) exceeds the number of active cells ({1:%d})"), std::vector<int>{count, (int)act.SumValsAll()}));
								data[i][countY-1-j] = 0;
								flag[i][countY-1-j] = 0;
								i++;
								if (i >= countX)
								{
									i = 0;
									j++;
								}
							}

							if (j >= countY)
								throw HMMPI::Exception(HMMPI::stringFormatArr(HMMPI::MessageRE("Èìååòñÿ áîëüøå çíà÷åíèé ñâîéñòâà ({0:%d}), ÷åì àêòèâíûõ ÿ÷ååê ({1:%d})",
																		  "Number of values ({0:%d}) exceeds the number of active cells ({1:%d})"), std::vector<int>{count, (int)act.SumValsAll()}));
							data[i][countY-1-j] = HMMPI::StoD(line_aux[k]);
							flag[i][countY-1-j] = 1;
							c++;
							i++;
							if (i >= countX)
							{
								i = 0;
								j++;
							}
						}
					}
					break;
				}
			}
		}
		if (!found)
			throw HMMPI::Exception(HMMPI::stringFormatArr("Âðåìåííîé øàã {0:%d} íå íàéäåí â ôàéëå",
													  	  "Data for step {0:%d} not found", step));
		sr.close();
	}
	catch(...)
	{
		if (sr.is_open())
			sr.close();
		throw;
	}
}
//------------------------------------------------------------------------------------------
void Grid2D::SaveToFile(std::string fname)
{
	std::ofstream sw;
	sw.exceptions(std::ios_base::badbit | std::ios_base::failbit);

	try
	{
		sw.open(fname);

		double x1 = x0 + dx/2;
		double y1 = y0 + dy/2;
		double x2 = x0 + dx*countX - dx/2;
		double y2 = y0 + dy*countY - dy/2;
		double M = max();
		double m = min();
		std::vector<double> a_params1 = {x1, x2, y1, y2, m, M};
		std::vector<double> a_params2 = {dx, dy};
		std::string s_params1 = HMMPI::stringFormatArr("FSLIMI {0:%.6f} {1:%.6f} {2:%.6f} {3:%.6f} {4:%g} {5:%g}", a_params1);
		std::string s_params2 = HMMPI::stringFormatArr("FSXINC {0:%.6f} {1:%.6f}", a_params2);

		// çàãîëîâîê
		sw << "FSASCI 0 1 COMPUTED 0 " + undef_val + "\n";
		sw << "FSATTR 0 0\n";
		sw << s_params1 << "\n";
		sw << HMMPI::stringFormatArr("FSNROW {0:%d} {1:%d}\n", std::vector<int>{countY, countX});
		sw << s_params2 << "\n";
	    sw << "->MSMODL: Surface of z1\n";

		// äàííûå
	    int count = 0;
 	    for (int i = 0; i < countX; i++)
		{
		    for (int j = 0; j < countY; j++)
			{
  			  if ((flag[i][countY-1-j])&&(!HMMPI::IsNaN(data[i][countY-1-j])))
				  sw << HMMPI::stringFormatArr("{0:%.6f}", std::vector<double>{data[i][countY-1-j]});
			  else
				  sw << undef_val;

			  if ((count % 5 == 4)||(j == countY-1))
				  sw << "\n";
			  else
				  sw << " ";

			  count++;
			  if (j == countY-1)
				  count = 0;
			}
		}
        sw.close();
	}
	catch (...)
	{
		if (sw.is_open())
			sw.close();
		throw;
	}
}
//------------------------------------------------------------------------------------------
void Grid2D::SaveToTextFile(std::string fname, double undef)
{
	std::ofstream sw;
	sw.exceptions(std::ios_base::badbit | std::ios_base::failbit);

	try
	{
		sw.open(fname);

		// çàãîëîâîê íå ïèøåòñÿ, òîëüêî äàííûå
 	    for (int j = 0; j < countY; j++)
		{
		    for (int i = 0; i < countX; i++)
			{
				if ((flag[i][countY-1-j])&&(!HMMPI::IsNaN(data[i][countY-1-j])))
					sw << HMMPI::stringFormatArr("{0}", std::vector<double>{data[i][countY-1-j]});
				else
					sw << HMMPI::stringFormatArr("{0}", std::vector<double>{undef});
				sw << "\t";
			}
			sw << "\n";
		}
        sw.close();
	}
	catch (...)
	{
		if (sw.is_open())
			sw.close();
		throw;
	}
}
//------------------------------------------------------------------------------------------
void Grid2D::SaveProp(std::string fname, std::string propname, double undef)
{
	std::ofstream sw;
	sw.exceptions(std::ios_base::badbit | std::ios_base::failbit);

	try
	{
		sw.open(fname);

		sw << propname << "\n";
 	    for (int j = 0; j < countY; j++)
		{
			int c = 0;
		    for (int i = 0; i < countX; i++)
			{
				if ((flag[i][countY-1-j])&&(!HMMPI::IsNaN(data[i][countY-1-j])))
					sw << HMMPI::stringFormatArr("{0:%.6f}", std::vector<double>{data[i][countY-1-j]});
				else
					sw << HMMPI::stringFormatArr("{0:%.6f}", std::vector<double>{undef});

				if ((c+1)%4 == 0)
					sw << "\n";
				else
					sw << "\t";
				c++;
			}
			sw << "\n";
		}
		sw << "/";
        sw.close();
	}
	catch (...)
	{
		if (sw.is_open())
			sw.close();
		throw;
	}
}
//------------------------------------------------------------------------------------------
void Grid2D::SaveProp3D(std::string fname, std::string propname, double undef, int Nz)
{
	std::ofstream sw;
	sw.exceptions(std::ios_base::badbit | std::ios_base::failbit);

	try
	{
		sw.open(fname);

		sw << propname << "\n";
		for (int k = 0; k < Nz; k++)
		{
 			for (int j = 0; j < countY; j++)
			{
				int c = 0;
				for (int i = 0; i < countX; i++)
				{
					if ((flag[i][countY-1-j])&&(!HMMPI::IsNaN(data[i][countY-1-j])))
						sw << HMMPI::stringFormatArr("{0:%.6e}", std::vector<double>{data[i][countY-1-j]});
					else
						sw << HMMPI::stringFormatArr("{0:%.6e}", std::vector<double>{undef});

					if ((c+1)%4 == 0)
						sw << "\n";
					else
						sw << "\t";
					c++;
				}
				sw << "\n";
			}
		}
		sw << "/";
        sw.close();
	}
	catch (...)
	{
		if (sw.is_open())
			sw.close();
		throw;
	}
}
//------------------------------------------------------------------------------------------
void Grid2D::SaveIntProp(std::string fname, std::string propname, double undef)
{
	std::ofstream sw;
	sw.exceptions(std::ios_base::badbit | std::ios_base::failbit);

	try
	{
		sw.open(fname);

		sw << propname << "\n";
 	    for (int j = 0; j < countY; j++)
		{
			int c = 0;
		    for (int i = 0; i < countX; i++)
			{
				if ((flag[i][countY-1-j])&&(!HMMPI::IsNaN(data[i][countY-1-j])))
					sw << HMMPI::stringFormatArr("{0:%d}", std::vector<int>{int(data[i][countY-1-j])});
				else
					sw << HMMPI::stringFormatArr("{0:%g}", std::vector<double>{undef});

				if ((c+1)%4 == 0)
					sw << "\n";
				else
					sw << "\t";
				c++;
			}
			sw << "\n";
		}
		sw << "/";
        sw.close();
	}
	catch (...)
	{
		if (sw.is_open())
			sw.close();
		throw;
	}
}
//------------------------------------------------------------------------------------------
void Grid2D::VectorToFile(std::string fname)
{
	std::ofstream sw;
	sw.exceptions(std::ios_base::badbit | std::ios_base::failbit);

	try
	{
		if (data == nullptr)
			throw HMMPI::Exception("Îòñóòñòâóþò äàííûå â Grid2D::VectorToFile",
									"No data in Grid2D::VectorToFile");
		sw.open(fname);
	    for (int i = 0; i < countX; i++)
 			for (int j = 0; j < countY; j++)
				sw << HMMPI::stringFormatArr("{0}\n", std::vector<double>{data[i][j]});

        sw.close();
	}
	catch (...)
	{
		if (sw.is_open())
			sw.close();
		throw;
	}
}
//------------------------------------------------------------------------------------------
void Grid2D::VectorToFileAct(std::string fname)
{
	std::ofstream sw;
	sw.exceptions(std::ios_base::badbit | std::ios_base::failbit);

	try
	{
		if (data == nullptr)
			throw HMMPI::Exception("Îòñóòñòâóþò äàííûå â Grid2D::VectorToFileAct",
									"No data in Grid2D::VectorToFileAct");

		sw.open(fname);
	    for (int i = 0; i < countX; i++)
 			for (int j = 0; j < countY; j++)
				if (flag[i][j])
					sw << HMMPI::stringFormatArr("{0}\n", std::vector<double>{data[i][j]});

        sw.close();
	}
	catch (...)
	{
		if (sw.is_open())
			sw.close();
		throw;
	}
}
//------------------------------------------------------------------------------------------
double Grid2D::SumValsAll() const
{
	double res = 0;
	for (int i = 0; i < countX; i++)
		for (int j = 0; j < countY; j++)
			res += data[i][j];

	return res;
}
//------------------------------------------------------------------------------------------
void Grid2D::Subtract(const Grid2D &g)
{
	if (!CheckSizes(std::vector<const Grid2D*>{const_cast<const Grid2D*>(this),
										  const_cast<const Grid2D*>(&g)}))
		throw HMMPI::Exception("Íå ñîâïàäàþò ðàçìåðû ãðèäîâ â Grid2D::Subtract",
						"Grid size mismatch in Grid2D::Subtract");

	for (int i = 0; i < countX; i++)
	{
		for (int j = 0; j < countY; j++)
		{
			data[i][j] -= g.data[i][j];
			flag[i][j] &= g.flag[i][j];
		}
	}
}
//------------------------------------------------------------------------------------------
void Grid2D::Mult(const Grid2D &g)
{
	if (!CheckSizes(std::vector<const Grid2D*>{const_cast<const Grid2D*>(this),
										  const_cast<const Grid2D*>(&g)}))
		throw HMMPI::Exception("Íå ñîâïàäàþò ðàçìåðû ãðèäîâ â Grid2D::Mult",
						"Grid size mismatch in Grid2D::Mult");

	for (int i = 0; i < countX; i++)
	{
		for (int j = 0; j < countY; j++)
		{
			data[i][j] *= g.data[i][j];
			flag[i][j] &= g.flag[i][j];
		}
	}
}
//------------------------------------------------------------------------------------------
void Grid2D::Plus(const Grid2D &g)
{
	if (!CheckSizes(std::vector<const Grid2D*>{const_cast<const Grid2D*>(this),
										  const_cast<const Grid2D*>(&g)}))
		throw HMMPI::Exception("Íå ñîâïàäàþò ðàçìåðû ãðèäîâ â Grid2D::Plus",
						"Grid size mismatch in Grid2D::Plus");

	for (int i = 0; i < countX; i++)
	{
		for (int j = 0; j < countY; j++)
		{
			data[i][j] += g.data[i][j];
			flag[i][j] &= g.flag[i][j];
		}
	}
}
//------------------------------------------------------------------------------------------
int Grid2D::SignStats()
{
	int pos = 0;
	int neg = 0;
	for (int i = 0; i < countX; i++)
	{
		for (int j = 0; j < countY; j++)
		{
			if (flag[i][j])
			{
				if (data[i][j] > 0)
					pos = 1;
				if (data[i][j] < 0)
					neg = -1;
			}
		}
	}

	return pos + neg;
}
//------------------------------------------------------------------------------------------
void Grid2D::Round()
{
	for (int i = 0; i < countX; i++)
		for (int j = 0; j < countY; j++)
			data[i][j] = floor(data[i][j] + 0.5);
}
//------------------------------------------------------------------------------------------
double *Grid2D::Serialize(const std::vector<Grid2D> &V)	// delete result after use!
{
	if (V.size() == 0)
		return 0;

	size_t count = V.size() * V[0].countX * V[0].countY;
	if (count == 0)
		return 0;

	double *res = new double[count];
	size_t k = 0;
	for (size_t c = 0; c < V.size(); c++)
		for (int i = 0; i < V[0].countX; i++)
			for (int j = 0; j < V[0].countY; j++)
			{
				res[k] = V[c].data[i][j];
				k++;
			}

	return res;
}
//------------------------------------------------------------------------------------------
void Grid2D::Deserialize(std::vector<Grid2D> &V, const double *d)
{
	size_t k = 0;
	for (size_t c = 0; c < V.size(); c++)
		for (int i = 0; i < V[0].countX; i++)
			for (int j = 0; j < V[0].countY; j++)
			{
				V[c].data[i][j] = d[k];
				k++;
			}
}
//------------------------------------------------------------------------------------------
