/*
 * OptPhysModels.cpp
 *
 *  Created on: Mar 26, 2013
 *      Author: ilya
 */

#include "Abstract.h"
#include "Utils.h"
#include "MathUtils.h"
#include "Parsing.h"
#include "Parsing2.h"
#include "Tracking.h"
#include "ConcretePhysModels.h"
#include "LinRegress.h"
#include "mpi.h"
#include <cmath>
#include <cassert>
#include <limits>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sys/stat.h>

const double NA = -1;
const double MIN_DIST = 1e-10;
const double MAX_INIT = 1e300;
const double SATEPS = 1e-6;
const double PI = acos(-1.0);

std::string PhysModelHM::log_file;
std::string PhysModelHM::uncert_dir;
HMMPI::Vector2<double> PhysModelHM::COEFFS;
int PhysModelHM::K_type;
HMMPI::Vector2<double> PhysModelHM::pts;

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
void PhysModelHM::RunSimulation(int i, int par_size)
{
	DECLKWD(file, KW_datafile, "DATAFILE");
	DECLKWD(simcmd, KW_simcmd, "SIMCMD");

	if (file->GetState() != "")
		throw HMMPI::EObjFunc(HMMPI::MessageRE("Корректный DATA-файл не задан в DATAFILE",
											   "No correct DATA-file specified in DATAFILE"));
	if (simcmd->cmd.size() == 0)
		throw HMMPI::EObjFunc(HMMPI::MessageRE("В SIMCMD должна быть задана хотя бы одна строка",
											   "At least one line should be defined in SIMCMD"));

	std::string fn = file->base_name + HMMPI::stringFormatArr("{0:%d}", std::vector<int>{i});
	std::string par = HMMPI::stringFormatArr("{0:%d}", std::vector<int>{par_size});
	std::string cmd = HMMPI::stringFormatArr(simcmd->cmd[0], std::vector<std::string>{file->path, fn, par});

#ifdef TESTCTOR
	std::ofstream testf(HMMPI::stringFormatArr("TESTCTOR_out_{0:%d}.txt", std::vector<int>{RNK}), std::ios::app);
	testf << cmd << "\n";
	testf.close();
#endif

	system(cmd.c_str());
}
//---------------------------------------------------------------------------
void PhysModelHM::GetParamsKrig() const
{
	DECLKWD(limits, KW_limits, "LIMITS");

	int krig_count = index_arr(2, 1) - index_arr(2, 0);
	params = std::vector<double>(krig_count);
	for (int i = index_arr(2, 0); i < index_arr(2, 1); i++)
	{
		if (limits->func[i] == "I")
			params[i - index_arr(2, 0)] = params_all[i] * limits->norm[i];
		else if (limits->func[i] == "LIN")
			params[i - index_arr(2, 0)] = params_all[i] * limits->norm[i] + limits->dh[i];
		else if (limits->func[i] == "EXP")
			params[i - index_arr(2, 0)] = pow(10, params_all[i] * limits->norm[i]);
		else
			throw HMMPI::EObjFunc(HMMPI::MessageRE("Некорректное функциональное преобразование в LIMITS",
									 	 	 	   "Incorrect function transform in LIMITS"));
	}
}
//---------------------------------------------------------------------------
void PhysModelHM::WriteData(int i)
{
	DECLKWD(file, KW_datafile, "DATAFILE");

#ifdef PUNQADJ
	file->WriteDataFile(i, adjoint_run);
#else
	file->WriteDataFile(i);
#endif
}
//---------------------------------------------------------------------------
void PhysModelHM::WriteModel(int i)
{
	DECLKWD(file, KW_datafile, "DATAFILE");
	DECLKWD(DIMS, KW_griddims, "GRIDDIMS");

	Grid2D aux;
	aux.InitData(DIMS->Nx, DIMS->Ny);
	aux.SetGeom(-0.5, -0.5, 1, 1);
	aux.SetVal(0);

	if (params.size() != COEFFS.ICount() - K_type)
		throw HMMPI::EObjFunc(HMMPI::MessageRE("Неправильное число модельных параметров (params) в PhysModelHM::WriteModel",
								 "Incorrect number of model parameters (params) in PhysModelHM::WriteModel"));
	if ((int)COEFFS.JCount() != DIMS->Nx*DIMS->Ny)
		throw HMMPI::EObjFunc(HMMPI::MessageRE("Неправильное число коэфф-ов кригинга (COEFFS) в PhysModelHM::WriteModel",
								 "Incorrect number of kriging coefficients (COEFFS) in PhysModelHM::WriteModel"));

	Grid2D krig = aux.Kriging(pts, params, COEFFS, K_type);
	krig.SaveProp3D(file->path + "/" + file->base_name + HMMPI::stringFormatArr("{0:%d}", std::vector<int>{i}) +
					DIMS->krig_file + ".inc", DIMS->krig_prop, 0, DIMS->Nz);

	grid = std::vector<double>(DIMS->Nx*DIMS->Ny);
	for (int i0 = 0; i0 < DIMS->Nx; i0++)
		for (int j0 = 0; j0 < DIMS->Ny; j0++)
			grid[i0 + (DIMS->Ny - j0 - 1)*DIMS->Nx] = krig.data[i0][j0];
}
//---------------------------------------------------------------------------
void PhysModelHM::WriteSWOF(int i)
{
	DECLKWD(swof, KW_SWOFParams, "SWOFPARAMS");
	DECLKWD(file, KW_datafile, "DATAFILE");
	DECLKWD(DIMS, KW_griddims, "GRIDDIMS");

	std::vector<double> par = swof->VarParams(params_all, index_arr(0, 0), index_arr(0, 1));
	std::string fn = file->path + "/" + file->base_name + HMMPI::stringFormatArr("{0:%d}", std::vector<int>{i}) +
				DIMS->swof_file + ".inc";
	swof->WriteSWOF(fn, par);
}
//---------------------------------------------------------------------------
void PhysModelHM::WriteSGOF(int i)
{
	DECLKWD(sgof, KW_SGOFParams, "SGOFPARAMS");
	DECLKWD(gas, KW_gas, "GAS");
	DECLKWD(file, KW_datafile, "DATAFILE");
	DECLKWD(DIMS, KW_griddims, "GRIDDIMS");

	if (gas->on == "ON")
	{
		std::vector<double> par = sgof->VarParams(params_all, index_arr(1, 0), index_arr(1, 1));
		std::string fn = file->path + "/" + file->base_name + HMMPI::stringFormatArr("{0:%d}", std::vector<int>{i}) +
					DIMS->sgof_file + ".inc";
		sgof->WriteSWOF(fn, par);
	}
}
//---------------------------------------------------------------------------
void PhysModelHM::WriteINC(int i)
{
	DECLKWD(incfiles, KW_incfiles, "INCFILES");
	DECLKWD(file, KW_datafile, "DATAFILE");
	DECLKWD(limits, KW_limits, "LIMITS");

	if (incfiles->GetState() != "")
		throw HMMPI::EObjFunc(HMMPI::MessageRE("Не удалось прочитать некоторые INCFILES-файлы",
								 "Failed to read some of the INCFILES"));

	if (incfiles->file.size() != 0)
	{
		size_t count = incfiles->file.size();
		if (count + 3 != index_arr.ICount())
			throw HMMPI::EObjFunc(HMMPI::MessageRE("Не совпадают размеры INCFILES и массива индексов в PhysModelHM::WriteINC",
									 "Inconsistent dimensions of INCFILES and index array in PhysModelHM::WriteINC"));
		for (size_t c = 0; c < count; c++)
		{
			int len = index_arr(c+3, 1) - index_arr(c+3, 0);
			if (len != incfiles->pcount[c])
				throw HMMPI::EObjFunc(HMMPI::MessageRE("Ошибка числа параметров в PhysModelHM::WriteINC",
										 	 	 	   "Parameter count error in PhysModelHM::WriteINC"));
			std::vector<double> p(len);
			for (int k = 0; k < len; k++)
			{
				int index = k + index_arr(c+3, 0);
				if (limits->func[index] == "I")
					p[k] = params_all[index] * limits->norm[index];
				else if (limits->func[index] == "LIN")
					p[k] = params_all[index] * limits->norm[index] + limits->dh[index];
				else if (limits->func[index] == "EXP")
					p[k] = pow(10, params_all[index] * limits->norm[index]);
				else
					throw HMMPI::EObjFunc(HMMPI::MessageRE("Некорректное функциональное преобразование в LIMITS",
											 "Incorrect function transform in LIMITS"));
			}

			// writing
			std::string fwrite = file->path + "/" + file->base_name + HMMPI::stringFormatArr("{0:%d}", std::vector<int>{i}) +
							incfiles->mod[c] + ".inc";
			std::ofstream sW;
			sW.exceptions(std::ios_base::badbit | std::ios_base::failbit);

			try
			{
				sW.open(fwrite);
				sW << HMMPI::stringFormatArr(incfiles->Buffer[c], p);
				sW.close();
			}
			catch (...)
			{
				if (sW.is_open())
					sW.close();
				throw HMMPI::Exception(HMMPI::stringFormatArr("ОШИБКА записи в файл {0:%s} в PhysModelHM::WriteINC\n",
												  "ERROR of writing to file {0:%s} in PhysModelHM::WriteINC\n", fwrite));
			}
		}
	}
}
//---------------------------------------------------------------------------
bool PhysModelHM::check_limits_swgof() const
{
	DECLKWD(swof, KW_SWOFParams, "SWOFPARAMS");
	DECLKWD(sgof, KW_SGOFParams, "SGOFPARAMS");
	DECLKWD(gas, KW_gas, "GAS");
	DECLKWD(SWCO, KW_Swco, "SWCO");

	bool res = true;
	std::string dlm = "";

	double Swco, Swcr, Soi;
	double Sgcr, Sgmax;
	std::vector<double> par_W = swof->VarParams(params_all, index_arr(0, 0), index_arr(0, 1));
	swof->SwcSor(par_W, Swcr, Soi);
	Swco = SWCO->Swco;
	if (Swco > Swcr)
		Swco = Swcr;
	if (Swcr >= 1-Soi)
	{
		res = false;
		if (RNK == 0)
		{
			limits_msg += "Swcr >= 1-Soi";
			dlm = ", ";
		}
	}

	if (gas->on == "ON")
	{
		std::vector<double> par_G = sgof->VarParams(params_all, index_arr(1, 0), index_arr(1, 1));
		sgof->SwcSor(par_G, Sgcr, Sgmax);
		if (Sgcr < 0)
		{
			res = false;
			if (RNK == 0)
			{
				limits_msg += dlm + "Sgcr < 0";
				dlm = ", ";
			}
		}
		if (Sgcr >= Sgmax)
		{
			res = false;
			if (RNK == 0)
			{
				limits_msg += dlm + "Sgcr >= Sgmax";
				dlm = ", ";
			}
		}
		if (Sgmax > 1-Swco)
		{
			res = false;
			if (RNK == 0)
				limits_msg += dlm + "Sgmax > 1-Swco";
		}
	}

	if (RNK == 0)
		limits_msg += "\n";

	return res;
}
//---------------------------------------------------------------------------
bool PhysModelHM::check_limits_krig() const
{
	DECLKWD(DIMS, KW_griddims, "GRIDDIMS");

	Grid2D aux;
	aux.InitData(DIMS->Nx, DIMS->Ny);
	aux.SetGeom(-0.5, -0.5, 1, 1);
	aux.SetVal(0);

	int krig_count = index_arr(2, 1) - index_arr(2, 0);
	if (krig_count == 0)
		return true;

	if (params.size() != COEFFS.ICount() - K_type)
		throw HMMPI::EObjFunc(HMMPI::MessageRE("Некорректное число модельных параметров (params) в PhysModelHM::check_limits_krig",
								 "Incorrect number of model parameters (params) in PhysModelHM::check_limits_krig"));
	if ((int)COEFFS.JCount() != DIMS->Nx*DIMS->Ny)
		throw HMMPI::EObjFunc(HMMPI::MessageRE("Некорректное число коэффициентов кригинга (COEFFS) в PhysModelHM::check_limits_krig",
								 "Incorrect number of kriging coefficients (COEFFS) in PhysModelHM::check_limits_krig"));

	GetParamsKrig();
	Grid2D krig = aux.Kriging(pts, params, COEFFS, K_type);

	bool res = true;
	for (int i = 0; i < krig.CountX(); i++)
	{
		for (int j = 0; j < krig.CountY(); j++)
		{
			if (krig.data[i][j] < 0)
			{
				res = false;
				if (RNK == 0)
					limits_msg += HMMPI::stringFormatArr(HMMPI::MessageRE("отрицательное значение в кригинге в позиции ({0:%d}, {1:%d})\n",
															"negative value for kriging in location ({0:%d}, {1:%d})\n"), std::vector<int>{i, j});
				return res;
			}
		}
	}

	return res;
}
//---------------------------------------------------------------------------
void PhysModelHM::add_limits_def(std::vector<std::vector<double>> &C, std::vector<double> &b)
{
	DECLKWD(limits, KW_limits, "LIMITS");
	const std::vector<double> min = limits->fullmin();
	const std::vector<double> max = limits->fullmax();

	size_t dim = limits->init.size();
	for (size_t i = 0; i < dim; i++)
	{
		std::vector<double> aux(dim);
		for (size_t j = 0; j < dim; j++)
			aux[j] = 0;

		aux[i] = -1;
		C.push_back(aux);
		b.push_back(-min[i]);

		aux = std::vector<double>(dim);
		for (size_t j = 0; j < dim; j++)
			aux[j] = 0;

		aux[i] = 1;
		C.push_back(aux);
		b.push_back(max[i]);
	}
}
//---------------------------------------------------------------------------
void PhysModelHM::add_limits_swgof(std::vector<std::vector<double>> &C, std::vector<double> &b)
{
	DECLKWD(swof, KW_SWOFParams, "SWOFPARAMS");
	DECLKWD(sgof, KW_SGOFParams, "SGOFPARAMS");
	DECLKWD(gas, KW_gas, "GAS");
	DECLKWD(SWCO, KW_Swco, "SWCO");
	DECLKWD(limits, KW_limits, "LIMITS");

	size_t dim = params_all.size();
	std::vector<double> line(dim);
	for (size_t i = 0; i < dim; i++)
		line[i] = 0;

	double Swco, Swcr, Soi;
	double Sgcr, Sgmax;
	std::vector<double> par_W = swof->VarParams(params_all, index_arr(0, 0), index_arr(0, 1));
	swof->SwcSor(par_W, Swcr, Soi);
	std::vector<int> par_ind = swof->SwcSorIndex(index_arr(0, 0));
	Swco = SWCO->Swco;

	if (par_ind[0] != -1)
	{
		std::vector<double> eqn1 = line;
		eqn1[par_ind[0]] = -(limits->norm[par_ind[0]]);

		C.push_back(eqn1);
		b.push_back(-Swco);
	}

	if (par_ind[0] != -1 || par_ind[1] != -1)
	{
		std::vector<double> eqn2 = line;
		double r = 1;

		if (par_ind[0] != -1)
			eqn2[par_ind[0]] = limits->norm[par_ind[0]];
		else
			r -= Swcr;

		if (par_ind[1] != -1)
			eqn2[par_ind[1]] = limits->norm[par_ind[1]];
		else
			r -= Soi;

		C.push_back(eqn2);
		b.push_back(r);
	}

	if (gas->on == "ON")
	{
		std::vector<double> par_G = sgof->VarParams(params_all, index_arr(1, 0), index_arr(1, 1));
		sgof->SwcSor(par_G, Sgcr, Sgmax);
		par_ind = sgof->SwcSorIndex(index_arr(1, 0));

		if (par_ind[0] != -1)
		{
			std::vector<double> eqn3 = line;
			eqn3[par_ind[0]] = -(limits->norm[par_ind[0]]);

			C.push_back(eqn3);
			b.push_back(0);
		}

		if (par_ind[0] != -1 || par_ind[1] != -1)
		{
			std::vector<double> eqn4 = line;
			double r = 0;

			if (par_ind[0] != -1)
				eqn4[par_ind[0]] = limits->norm[par_ind[0]];
			else
				r -= Sgcr;

			if (par_ind[1] != -1)
				eqn4[par_ind[1]] = -limits->norm[par_ind[1]];
			else
				r += Sgmax;

			C.push_back(eqn4);
			b.push_back(r);
		}

		if (par_ind[1] != -1)
		{
			std::vector<double> eqn5 = line;
			eqn5[par_ind[1]] = limits->norm[par_ind[1]];

			C.push_back(eqn5);
			b.push_back(1 - Swco);
		}
	}
}
//---------------------------------------------------------------------------
void PhysModelHM::add_limits_krig(std::vector<std::vector<double>> &C, std::vector<double> &b)
{
	DECLKWD(limits, KW_limits, "LIMITS");

	size_t dim = params_all.size();
	int krig_count = index_arr(2, 1) - index_arr(2, 0);
	if (krig_count == 0)
		return;

	if (K_type != 1)
		throw HMMPI::EObjFunc(HMMPI::MessageRE("PhysModelHM::add_limits_krig допускает только K_type == 1",
								 "Only K_type == 1 is allowed in PhysModelHM::add_limits_krig"));
	if (krig_count != (int)COEFFS.ICount()-1)
		throw HMMPI::EObjFunc(HMMPI::MessageRE("Не совпадают размеры массивов в PhysModelHM::add_limits_krig",
								 "Array size mismatch in PhysModelHM::add_limits_krig"));
	size_t N = COEFFS.JCount();

	for (size_t n = 0; n < N; n++)
	{
		std::vector<double> aux(dim);
		bool found_pos = false;
		for (size_t i = 0; i < dim; i++)
		{
			aux[i] = 0;
		}
		for (int i = 0; i < krig_count; i++)
		{
			aux[i + index_arr(2, 0)] = -COEFFS(i, n) * limits->norm[i + index_arr(2, 0)];
			if (aux[i + index_arr(2, 0)] > 0)
				found_pos = true;
		}
		if (found_pos)
		{
			C.push_back(aux);
			b.push_back(0);
		}
	}
}
//---------------------------------------------------------------------------
//void PhysModelHM::FillParamsAll(const std::vector<double> &p)		// removed 01.09.2016
//{
//	DECLKWD(limits, KW_limits, "LIMITS");
//
//	size_t len = params_all.size();
//	int act_par = 0;
//	for (size_t i = 0; i < len; i++)
//	{
//		if (limits->act[i] == "A")
//		{
//			params_all[i] = p[act_par];
//			act_par++;
//		}
//		else
//			params_all[i] = limits->init[i];
//	}
//}
//---------------------------------------------------------------------------
void PhysModelHM::write_params(std::ofstream &SW)
{
	DECLKWD(limits, KW_limits, "LIMITS");

	SW << std::string(HMMPI::MessageRE("Параметры:\n", "Parameters:\n"));
	SW << HMMPI::stringFormatArr("внутр.    \tвнешн.    \tактивн.   \tвсего: {0:%d}\n",
								 "inner     \touter     \tactive    \ttotal: {0:%d}\n", (int)params_all.size());
	for (size_t i = 0; i < params_all.size(); i++)
	{
		if (limits->func[i] == "I")
			SW << HMMPI::stringFormatArr("{0:%-10.8g}\t{1:%-10.8g}\t", std::vector<double>{params_all[i], params_all[i] * limits->norm[i]}) + limits->act[i] + "\n";
		else if (limits->func[i] == "LIN")
			SW << HMMPI::stringFormatArr("{0:%-10.8g}\t{1:%-10.8g}\t", std::vector<double>{params_all[i], params_all[i] * limits->norm[i] + limits->dh[i]}) + limits->act[i] + "\n";
		else if (limits->func[i] == "EXP")
			SW << HMMPI::stringFormatArr("{0:%-10.8g}\t{1:%-10.8g}\t", std::vector<double>{params_all[i], pow(10, params_all[i] * limits->norm[i])}) + limits->act[i] + "\n";
	}
	SW << "\n";
}
//---------------------------------------------------------------------------
void PhysModelHM::write_smry(std::ofstream &SW, const HMMPI::Vector2<double> &smry_mod, const HMMPI::Vector2<double> &smry_hist, const std::vector<double> &of1_full, bool text_sigma)
{
	DECLKWD(datesW, KW_dates, "DATES");
	DECLKWD(vect, KW_eclvectors, "ECLVECTORS");

	size_t Nsteps = smry_mod.ICount();
	size_t Nvect = smry_mod.JCount();

	std::string headerM1 = "          \t", headerM2 = "          \t", headerH1 = "", headerH2 = "", headerH3 = "";
	std::string headerM3 = HMMPI::MessageRE("ДД/ММ/ГГГГ\t", "DD/MM/YYYY\t");
	std::string headerS1 = "", headerS2 = "", headerS3 = "";
	for (size_t i = 0; i < Nvect; i++)
	{
		headerM1 += HMMPI::stringFormatArr("мод{0:%d}      \t", "mod{0:%d}      \t", (int)i+1);
		headerH1 += HMMPI::stringFormatArr("ист{0:%d} ", "hist{0:%d}", (int)i+1);
		headerS1 += HMMPI::stringFormatArr("сигма{0:%d}", "sigma{0:%d}", (int)i+1);

		int pads2 = 10 - vect->WGname[i].length();
		if (pads2 < 0)
			pads2 = 0;
		headerM2 += vect->WGname[i] + std::string(pads2, ' ') + "\t";
		headerH2 += vect->WGname[i];
		headerS2 += vect->WGname[i];

		int pads3 = 10 - vect->vect[i].length();
		if (pads3 < 0)
			pads3 = 0;
		int pads3h = pads3-1;
		if (pads3h < 0)
			pads3h = 0;
		headerM3 += vect->vect[i] + std::string(pads3, ' ') + "\t";
		headerH3 += vect->vect[i] + "H";
		headerS3 += vect->vect[i] + "S";

		if (i < Nvect-1 || text_sigma)
		{
			headerH1 += "     \t";
			headerH2 += std::string(pads2, ' ') + "\t";
			headerH3 += std::string(pads3h, ' ') + "\t";
		}
		if (i < Nvect-1)
		{
			headerS1 += "    \t";
			headerS2 += std::string(pads2, ' ') + "\t";
			headerS3 += std::string(pads3h, ' ') + "\t";
		}
	}

	SW << std::string(HMMPI::MessageRE("Вектора (модельные и исторические):\n",
						   	   	   	   "Vectors summary (modelled and historic):\n"));
	if (!text_sigma)
		SW << headerM1 + headerH1 + "\n" + headerM2 + headerH2 + "\n" + headerM3 + headerH3 + "\n";
	else
		SW << headerM1 + headerH1 + headerS1 + "\n" + headerM2 + headerH2 + headerS2 + "\n" + headerM3 + headerH3 + headerS3 + "\n";
	for (size_t i = 0; i < Nsteps; i++)
	{
		SW << HMMPI::stringFormatArr("{0:%.2d}/{1:%.2d}/{2:%.4d}\t", std::vector<int>{datesW->D[i], datesW->M[i], datesW->Y[i]});
		for (size_t j = 0; j < Nvect; j++)
			SW << HMMPI::stringFormatArr("{0:%-10.8g}\t", std::vector<double>{smry_mod(i, j)});

		for (size_t j = 0; j < Nvect; j++)
		{
			SW << HMMPI::stringFormatArr("{0:%-10.8g}", std::vector<double>{smry_hist(i, j)});
			if (j < Nvect-1)
				SW << "\t";
		}
		if (text_sigma)
		{
			for (size_t j = 0; j < Nvect; j++)
				SW << HMMPI::stringFormatArr("\t{0:%-10.8g}", std::vector<double>{smry_hist(i, j + Nvect)});
		}

		SW << "\n";
	}

	// of1_full
	SW << "          \t";
	for (size_t j = 0; j < Nvect; j++)
	{
		SW << HMMPI::stringFormatArr("{0:%-10.8g}", std::vector<double>{of1_full[j]});
		if (j < Nvect-1)
			SW << "\t";
	}

	// spaces
	for (size_t j = 0; j < Nvect; j++)
		SW << "          \t";

	// correlation radii (below sigmas)
	for (size_t j = 0; j < Nvect; j++)
	{
		SW << HMMPI::stringFormatArr("{0:%-10.8g}", std::vector<double>{vect->R[j]});
		if (j < Nvect-1)
			SW << "\t";
	}

	SW << "\n\n";
}
//---------------------------------------------------------------------------
void PhysModelHM::write_smry(std::ofstream &SW, const HMMPI::Vector2<double> &smry_mod)
{
	DECLKWD(datesW, KW_dates, "DATES");
	DECLKWD(vect, KW_eclvectors, "ECLVECTORS");

	size_t Nsteps = smry_mod.ICount();
	size_t Nvect = smry_mod.JCount();

	std::string headerM1 = "          \t", headerM2 = "          \t";
	std::string headerM3 = HMMPI::MessageRE("ДД/ММ/ГГГГ\t", "DD/MM/YYYY\t");
	for (size_t i = 0; i < Nvect; i++)
	{
		headerM1 += HMMPI::stringFormatArr(HMMPI::MessageRE("мод{0:%d}", "mod{0:%d}"), std::vector<int>{(int)i+1});

		int pads2 = 10 - vect->WGname[i].length();
		if (pads2 < 0)
			pads2 = 0;
		headerM2 += vect->WGname[i];

		int pads3 = 10 - vect->vect[i].length();
		if (pads3 < 0)
			pads3 = 0;
		headerM3 += vect->vect[i];

		if (i < Nvect-1)
		{
			headerM1 += "      \t";
			headerM2 += std::string(pads2, ' ') + "\t";
			headerM3 += std::string(pads3, ' ') + "\t";
		}
	}

	SW << std::string(HMMPI::MessageRE("Вектора (модельные):\n",
						   "Vectors summary (modelled):\n"));
	SW << headerM1 + "\n" + headerM2 + "\n" + headerM3 + "\n";

	for (size_t i = 0; i < Nsteps; i++)
	{
		SW << HMMPI::stringFormatArr("{0:%.2d}/{1:%.2d}/{2:%.4d}\t", std::vector<int>{datesW->D[i], datesW->M[i], datesW->Y[i]});
		for (size_t j = 0; j < Nvect; j++)
		{
			SW << HMMPI::stringFormatArr("{0:%-10.8g}", std::vector<double>{smry_mod(i, j)});
			if (j < Nvect-1)
				SW << "\t";
		}
		SW << "\n";
	}
}
//---------------------------------------------------------------------------
void PhysModelHM::write_smry_hist(std::ofstream &SW, const HMMPI::Vector2<double> &smry_hist, bool text_sigma)
{
	DECLKWD(datesW, KW_dates, "DATES");
	DECLKWD(vect, KW_eclvectors, "ECLVECTORS");

	size_t Nsteps = smry_hist.ICount();
	size_t Nvect = smry_hist.JCount()/2;

	std::string headerH2 = "          \t", headerH3 = "          \t";
	std::string headerS2 = "", headerS3 = "";
	for (size_t i = 0; i < Nvect; i++)
	{
		int pads2 = 10 - vect->WGname[i].length();
		if (pads2 < 0)
			pads2 = 0;
		headerH2 += vect->WGname[i];
		headerS2 += vect->WGname[i];

		int pads3 = 10 - vect->vect[i].length();
		if (pads3 < 0)
			pads3 = 0;
		int pads3h = pads3-1;
		if (pads3h < 0)
			pads3h = 0;
		headerH3 += vect->vect[i] + "H";
		headerS3 += vect->vect[i] + "S";

		if (i < Nvect-1 || text_sigma)
		{
			headerH2 += std::string(pads2, ' ') + "\t";
			headerH3 += std::string(pads3h, ' ') + "\t";
		}
		if (i < Nvect-1)
		{
			headerS2 += std::string(pads2, ' ') + "\t";
			headerS3 += std::string(pads3h, ' ') + "\t";
		}
	}

	if (!text_sigma)
		SW << headerH2 + "\n" + headerH3 + "\n";
	else
		SW << headerH2 + headerS2 + "\n" + headerH3 + headerS3 + "\n";
	for (size_t i = 0; i < Nsteps; i++)
	{
		SW << HMMPI::stringFormatArr("{0:%.2d}/{1:%.2d}/{2:%.4d}\t", std::vector<int>{datesW->D[i], datesW->M[i], datesW->Y[i]});
		for (size_t j = 0; j < Nvect; j++)
		{
			SW << HMMPI::stringFormatArr("{0:%-10.7g}", std::vector<double>{smry_hist(i, j)});
			if (j < Nvect-1)
				SW << "\t";
		}
		if (text_sigma)
		{
			for (size_t j = 0; j < Nvect; j++)
				SW << HMMPI::stringFormatArr("\t{0:%-10.7g}", std::vector<double>{smry_hist(i, j + Nvect)});
		}
		SW << "\n";
	}

	// spaces
	SW << "-- corr.  \t";
	for (size_t j = 0; j < Nvect; j++)
		SW << "          \t";

	// correlation radii (below sigmas)
	for (size_t j = 0; j < Nvect; j++)
	{
		SW << HMMPI::stringFormatArr("{0:%-10.8g}", std::vector<double>{vect->R[j]});
		if (j < Nvect-1)
			SW << "\t";
	}
	SW << "\n";
}
//---------------------------------------------------------------------------
std::vector<Grid2D> PhysModelHM::SubtractBase(const std::vector<Grid2D> &grids)
{
	size_t len = grids.size();
	std::vector<Grid2D> res(len-1);

	for (size_t i = 1; i < len; i++)
	{
		Grid2D B = grids[i];
		B.Subtract(grids[0]);
		res[i-1] = std::move(B);
	}
	return res;
}
//---------------------------------------------------------------------------
HMMPI::Vector2<std::vector<double>> PhysModelHM::calc_derivatives_dRdP(std::string mod_name) const
{
	const int Pcount = 45;
	const std::vector<std::string> validR = {"WBHP", "WOPR", "WLPR", "WGPR"};
	const double dk = log(10)*9.03;
	const double dkz = 0.306*dk;

	DECLKWD(rst, KW_funrst, "FUNRST");
	DECLKWD(vect, KW_eclvectors, "ECLVECTORS");

	HMMPI::Vector2<KW_funrst::grad> GRADS = rst->ReadGrads(mod_name);
	std::vector<double> permx = rst->ReadDataInit(mod_name + ".FINIT", "'PERMX   '");
	std::vector<double> multnum = rst->ReadDataInit(mod_name + ".FINIT", "'MULTNUM '");

	HMMPI::Vector2<std::vector<double>> res(GRADS.ICount(), GRADS.JCount());
	for (size_t i = 0; i < res.ICount(); i++)
		for (size_t j = 0; j < res.JCount(); j++)
			res(i, j) = std::vector<double>(Pcount);

	for (size_t t = 0; t < res.ICount(); t++)		// time step
		for (size_t v = 0; v < res.JCount(); v++)		// ecl vector
			if (std::find(validR.begin(), validR.end(), vect->vect[v]) != validR.end() && GRADS(t, v).count("PORO") > 0)		// found a valid vector, and valid date
				for (size_t l = 0; l < multnum.size(); l++)	// go through grid
					if (!HMMPI::IsNaN(multnum[l]))				// take active cell
					{
						double perm = permx[l];
						int ind = multnum[l];		// 1-based region number
						if (ind < 1 || ind > Pcount)
							throw HMMPI::Exception("ind < 1 || ind > Pcount in PhysModelHM::calc_derivatives_dRdP");

						res(t, v)[ind-1] += GRADS(t, v)["PORO"][l] + GRADS(t, v)["PERMXY"][l]*perm*dk + GRADS(t, v)["PERMZ"][l]*perm*dkz;
//						if (vect->WGname[v] == "PRO-5" && vect->vect[v] == "WBHP" && (ind == 30 || ind == 21) && t == 1)	// DEBUG
//							std::cout << "multnum size " << multnum.size() << ", ind " << ind << ", l = " << l << ", PRO-5 WBHP grad PORO " << GRADS(t, v)["PORO"][l] <<
//							", PRO-5 WBHP grad PERMXY " << GRADS(t, v)["PERMXY"][l] <<
//							", PRO-5 WBHP grad PERMZ " << GRADS(t, v)["PERMZ"][l] <<
//							", perm*dk = " << perm*dk << ", res = " << res(t, v)[ind-1] <<
//							"\n";	// DEBUG
					}

	return res;
}
//---------------------------------------------------------------------------
void PhysModelHM::calc_derivatives_dRdP2(HMMPI::Vector2<std::vector<double>> &drdp, const HMMPI::Vector2<double> &smry) const
{
	const int Pcount = 45;
	DECLKWD(vect, KW_eclvectors, "ECLVECTORS");

	assert(drdp.ICount() == smry.ICount() && drdp.JCount() == smry.JCount() && vect->vect.size() == drdp.JCount());
	for (size_t v = 0; v < drdp.JCount(); v++)
		if (vect->vect[v] == "WWCT")
		{
			HMMPI::SimSMRY::pair wwct{vect->WGname[v], (std::string)"WWCT"};
			HMMPI::SimSMRY::pair wopr{vect->WGname[v], (std::string)"WOPR"};
			HMMPI::SimSMRY::pair wlpr{vect->WGname[v], (std::string)"WLPR"};
			int i_wwct = std::find(vect->vecs.begin(), vect->vecs.end(), wwct) - vect->vecs.begin();
			int i_wopr = std::find(vect->vecs.begin(), vect->vecs.end(), wopr) - vect->vecs.begin();
			int i_wlpr = std::find(vect->vecs.begin(), vect->vecs.end(), wlpr) - vect->vecs.begin();
			if (i_wwct == (int)vect->vecs.size() || i_wopr == (int)vect->vecs.size() || i_wlpr == (int)vect->vecs.size())
				throw HMMPI::Exception("Vector not found in PhysModelHM::calc_derivatives_dRdP2");

			for (size_t t = 0; t < drdp.ICount(); t++)		// all time steps
			{
				double WOPR = smry(t, i_wopr);
				double WLPR = smry(t, i_wlpr);
				for (int p = 0; p < Pcount; p++)
					if (WLPR != 0)
						drdp(t, i_wwct)[p] = -drdp(t, i_wopr)[p]/WLPR + drdp(t, i_wlpr)[p]*WOPR/(WLPR*WLPR);
			}
		}
		else if (vect->vect[v] == "WGOR")
		{
			HMMPI::SimSMRY::pair wgor{vect->WGname[v], "WGOR"};
			HMMPI::SimSMRY::pair wopr{vect->WGname[v], "WOPR"};
			HMMPI::SimSMRY::pair wgpr{vect->WGname[v], "WGPR"};
			int i_wgor = std::find(vect->vecs.begin(), vect->vecs.end(), wgor) - vect->vecs.begin();
			int i_wopr = std::find(vect->vecs.begin(), vect->vecs.end(), wopr) - vect->vecs.begin();
			int i_wgpr = std::find(vect->vecs.begin(), vect->vecs.end(), wgpr) - vect->vecs.begin();
			if (i_wgor == (int)vect->vecs.size() || i_wopr == (int)vect->vecs.size() || i_wgpr == (int)vect->vecs.size())
				throw HMMPI::Exception("Vector not found in PhysModelHM::calc_derivatives_dRdP2");

			for (size_t t = 0; t < drdp.ICount(); t++)		// all time steps
			{
				double WOPR = smry(t, i_wopr);
				double WGPR = smry(t, i_wgpr);
				for (int p = 0; p < Pcount; p++)
					if (WOPR != 0)
						drdp(t, i_wgor)[p] = drdp(t, i_wgpr)[p]/WOPR - drdp(t, i_wopr)[p]*WGPR/(WOPR*WOPR);
			}
		}
}
//---------------------------------------------------------------------------
PhysModelHM::PhysModelHM(Parser_1 *k, KW_item *kw, std::string cwd, MPI_Comm c) : Sim_small_interface(k, kw, c), K(k), CWD(cwd), modelled_data_size(0)
{
	DECLKWD(wghts, KW_ofweights, "OFWEIGHTS");		// declare and add to prerequisites
	DECLKWD(gas, KW_gas, "GAS");
	DECLKWD(smry, KW_funsmry, "FUNSMRY");
	DECLKWD(textsmry, KW_textsmry, "TEXTSMRY");
	DECLKWD(fegrid, KW_fegrid, "FEGRID");
	DECLKWD(incfiles, KW_incfiles, "INCFILES");
	DECLKWD(swof, KW_SWOFParams, "SWOFPARAMS");
	DECLKWD(sgof, KW_SGOFParams, "SGOFPARAMS");
	DECLKWD(pilot, KW_pilot, "PILOT");
	DECLKWD(var, KW_variogram, "VARIOGRAM");
	DECLKWD(varCs, KW_variogram_Cs, "VARIOGRAM_CS");
	DECLKWD(dims, KW_griddims, "GRIDDIMS");
	DECLKWD(limits, KW_limits, "LIMITS");
	DECLKWD(regConstr, KW_regressConstr, "REGRESSCONSTR");
	DECLKWD(rstA, KW_funrstA, "FUNRSTA");
	DECLKWD(mapreg, KW_mapreg, "MAPREG");
	DECLKWD(vect, KW_eclvectors, "ECLVECTORS");
	DECLKWD(datesW, KW_dates, "DATES");

	name = "PhysModelHM";
#ifdef PUNQADJ
	adjoint_run = false;
#endif

	kw->Start_pre();
	kw->Add_pre("OFWEIGHTS");
	kw->Add_pre("LIMITS");
	kw->Add_pre("PCAPILL");
	kw->Add_pre("DATAFILE");
	kw->Add_pre("GAS");
	if (wghts->w1 > 0)			// depending on what objective function parts (weights) are active, add the necessary keywords to the prerequisites
	{
		kw->Add_pre("ECLVECTORS");
		kw->Add_pre("DATES");
	}
	if (wghts->w2 > 0)
		kw->Add_pre("FUNRST");
	if (wghts->w3 > 0)
		kw->Add_pre("FUNRSTG");
	if (wghts->w4 > 0)
	{
		kw->Add_pre("REFMAP");
		kw->Add_pre("REFMAP_W");
	}
	if (wghts->w5 > 0)
	{
		kw->Add_pre("FUNRSTA");
		kw->Add_pre("MAPREG");
		kw->Add_pre("MAPSEISSCALE");
		kw->Add_pre("REGRESSCONSTR");
	}
	kw->Finish_pre();

	if ((wghts->w3 > 0)&&(gas->on == "OFF"))
		throw HMMPI::Exception("OFWEIGHTS->w3 != 0, но опция газа выключена (OFF)", "OFWEIGHTS->w3 != 0, but gas is OFF");

	if ((wghts->w2 > 0 || wghts->w3 > 0 || wghts->w4 > 0 || wghts->w5 > 0) && fegrid->data.size() == 0)
	{
		K->AppText(HMMPI::MessageRE("ПРЕДУПРЕЖДЕНИЕ: FEGRID не задан корректно\n",
									"WARNING: FEGRID is not defined properly\n"));
		K->TotalWarnings++;
	}
	if (textsmry->data.Length() != 0)
		K->AppText(HMMPI::MessageRE("TEXTSMRY задан -> история берется из TEXTSMRY\n",
									"TEXTSMRY is defined -> history is taken from TEXTSMRY\n"));
	else if (smry->data.Length() == 0)
		K->AppText(HMMPI::MessageRE("TEXTSMRY и FUNSMRY не заданы -> история берется из модельного *.FUNSMRY\n",
									"TEXTSMRY and FUNSMRY are not defined -> history is taken from model *.FUNSMRY\n"));
	else
		K->AppText(HMMPI::MessageRE("TEXTSMRY не задан, FUNSMRY задан -> история берется из FUNSMRY\n",
									"TEXTSMRY is not defined, FUNSMRY is defined -> history is taken from FUNSMRY\n"));
	RLS = 0;
	VCL = 0;

	s_error = s_warning = 0;
	s_echo = true;

	sign = 0;
	f1 = f2 = f3 = f4 = f5 = 0;

	ignore_small_errors = true;
	uncert_dir = this->CWD + "/uncert";
	log_file = this->CWD + "/ObjFuncLog.txt";
	RegEntry::CWD = CWD;

	if (wghts->w5 > 0)
	{
		regConstr->fill_ind();
		RegEntryConstr::regcon = regConstr;
		LinRegressConstr::regcon = regConstr;
	}

	// fill index_arr from SWOF, SGOF, KRIG, INC{}
	int swof_count = swof->VarCount();
	int sgof_count = sgof->VarCount();
	if (gas->on == "OFF")
		sgof_count = 0;

	int krig_count = pilot->x.size();
	int inc_count = incfiles->file.size();	// number of inc-files
	int index_count = 3 + inc_count;
	index_arr = HMMPI::Vector2<int>(index_count, 2);

	index_arr(0, 0) = 0;
	index_arr(0, 1) = swof_count;

	index_arr(1, 0) = index_arr(0, 1);
	index_arr(1, 1) = index_arr(1, 0) + sgof_count;

	index_arr(2, 0) = index_arr(1, 1);
	index_arr(2, 1) = index_arr(2, 0) + krig_count;

	int inc_params_count = 0;		// number of inc-params
	for (int i = 3; i < index_count; i++)
	{
		index_arr(i, 0) = index_arr(i-1, 1);
		index_arr(i, 1) = index_arr(i, 0) + incfiles->pcount[i-3];
		inc_params_count += incfiles->pcount[i-3];
	}

	// define params_all, params
	int tot_count = index_arr(index_count-1, 1);
	if (tot_count != (int)limits->init.size())
	{
		std::string par_str = HMMPI::stringFormatArr("(SWOF[{0:%d}], SGOF[{1:%d}], KRIG[{2:%d}], INC[{3:%d}])", std::vector<int>{swof_count, sgof_count, krig_count, inc_params_count});
		throw HMMPI::EObjFunc(HMMPI::stringFormatArr("Число заданных параметров " + par_str + " не согласуется с LIMITS[{0:%d}]",
								 	  "Number of parameters specified " + par_str + " is not consistent with LIMITS[{0:%d}]", (int)limits->init.size()));
	}
	params_all = std::vector<double>(tot_count);
	params = std::vector<double>(krig_count);

	if (krig_count != 0)
	{
		K_type = (var->krig_type == "SIM")?(0):(1);
		pts = HMMPI::Vector2<double>(krig_count, 3);	// old version: pts = pilot->points;
		for (int i = 0; i < krig_count; i++)
		{
			pts(i, 0) = pilot->x[i];
			pts(i, 1) = pilot->y[i];
			pts(i, 2) = pilot->z[i];
		}

		Grid2D aux;
		aux.InitData(dims->Nx, dims->Ny);
		aux.SetGeom(-0.5, -0.5, 1, 1);
		aux.SetVal(0);
		COEFFS = aux.KrigingCoeffs(pts, var->chi/180*PI, var->R, var->r, var->sill, var->nugget, var->type, var->krig_type);
	}
	else
	{
		pts = HMMPI::Vector2<double>();
		COEFFS = HMMPI::Vector2<double>();
	}

	// initialize ModelledData
	modelled_data = std::vector<double>();

	// initialize VectCorrList
	if (wghts->w1 > 0 && textsmry->data.Length() != 0)
	{
		VCL = new VectCorrList;
		VCL->ownerCount = 1;
		VCL->LoadData(textsmry->data, datesW->zeroBased(), vect->R, vect->corr);
	}

	// initialize pet_seis, RegListSpat
	pet_seis_len = 0;
	if (wghts->w5 > 0)
	{
		size_t count = rstA->data.size() - 1;
		pet_seis = std::vector<Grid2D>(count);
		for (size_t c = 0; c < count; c++)
		{
			Grid2D aux;
			aux.InitData(dims->Nx, dims->Ny);
			aux.SetGeom(-0.5, -0.5, 1, 1);
			aux.SetVal(0);
			pet_seis[c] = aux;
		}

		pet_seis_len = count * dims->Nx * dims->Ny;

		// make RegListSpat
		RLS = new RegListSpat;
		RLS->ownerCount = 1;

		Grid2D Reg = mapreg->GetGrid2D(dims->Nx, dims->Ny);
		Reg.Round();
		RLS->ReadSpatial(Reg, varCs, (int)count);
	}

#ifdef TESTCTOR
	std::ofstream testf(HMMPI::stringFormatArr("TESTCTOR_out_{0:%d}.txt", std::vector<int>{RNK}), std::ios::app);
	testf << "rank " << RNK << ", PhysModelHM easy CTOR, this = " << this << "\n";
	testf.close();
#endif
}
//---------------------------------------------------------------------------
PhysModelHM::PhysModelHM(const PhysModelHM &PM) : Sim_small_interface(PM)
{
	K = PM.K;
	grid = PM.grid;

	index_arr = PM.index_arr;
	pet_seis_len = PM.pet_seis_len;
	modelled_data_size = PM.modelled_data_size;
#ifdef PUNQADJ
	adjoint_run = PM.adjoint_run;
#endif

	VCL = PM.VCL;
	if (VCL != 0)
		VCL->ownerCount++;

	RLS = PM.RLS;
	if (RLS != 0)
		RLS->ownerCount++;

	params = PM.params;
	f1 = PM.f1;
	f2 = PM.f2;
	f3 = PM.f3;
	f4 = PM.f4;
	f5 = PM.f5;
	sign = PM.sign;

	params_all = PM.params_all;
	ignore_small_errors = PM.ignore_small_errors;
	limits_msg = PM.limits_msg;
	pet_seis = PM.pet_seis;

	s_error = PM.s_error;
	s_warning = PM.s_warning;
	s_echo = PM.s_echo;
}
//---------------------------------------------------------------------------
int PhysModelHM::ParamsDim() const noexcept
{
	DECLKWD(limits, KW_limits, "LIMITS");
	return limits->init.size();
}
//---------------------------------------------------------------------------
void PhysModelHM::SavePMState()
{
	s_error = K->TotalErrors;
	s_warning = K->TotalWarnings;
	s_echo = K->echo;
	K->echo = false;
}
//---------------------------------------------------------------------------
void PhysModelHM::RestorePMState()
{
	K->TotalErrors = s_error;
	K->TotalWarnings = s_warning;
	K->echo = s_echo;
}
//---------------------------------------------------------------------------
PhysModelHM::~PhysModelHM()
{
	if (VCL != 0)
	{
		VCL->ownerCount--;
		if (VCL->ownerCount == 0)
		{
			delete VCL;
			VCL = 0;
		}
		else if (VCL->ownerCount < 0)
			throw HMMPI::Exception("VCL->ownerCount < 0");
	}

	if (RLS != 0)
	{
		RLS->ownerCount--;
		if (RLS->ownerCount == 0)
		{
			delete RLS;
			RLS = 0;
		}
		else if (RLS->ownerCount < 0)
			throw HMMPI::Exception("RLS->ownerCount < 0");
	}

#ifdef TESTCTOR
	std::ofstream testf(HMMPI::stringFormatArr("TESTCTOR_out_{0:%d}.txt", std::vector<int>{RNK}), std::ios::app);
	testf << "rank " << RNK << ", PhysModelHM -- DTOR --, this = " << this << "\n";
	testf.close();
#endif
}
//---------------------------------------------------------------------------
std::string PhysModelHM::IndexMsg() const
{
	DECLKWD(gas, KW_gas, "GAS");
	DECLKWD(limits, KW_limits, "LIMITS");

	std::string res = "";
	std::string aux = "";
	if (gas->on == "OFF")
		aux = HMMPI::MessageRE("(eng)", "Gas is INACTIVE");

	if (index_arr(0, 1) > index_arr(0, 0))
		res += HMMPI::stringFormatArr(HMMPI::MessageRE("Параметры SWOF: {0:%d} ({1:%d}..{2:%d})\n",
										 "SWOF parameters: {0:%d} ({1:%d}..{2:%d})\n"), std::vector<int>{index_arr(0, 1) - index_arr(0, 0), index_arr(0, 0), index_arr(0, 1)-1});
	else
		res += HMMPI::stringFormatArr(HMMPI::MessageRE("Параметры SWOF: {0:%d}\n",
										 "SWOF parameters: {0:%d}\n"), std::vector<int>{index_arr(0, 1) - index_arr(0, 0)});

	if (index_arr(1, 1) > index_arr(1, 0))
		res += HMMPI::stringFormatArr(HMMPI::MessageRE("Параметры SGOF: {0:%d} ({1:%d}..{2:%d})\n",
										 "SGOF parameters: {0:%d} ({1:%d}..{2:%d})\n"), std::vector<int>{index_arr(1, 1) - index_arr(1, 0), index_arr(1, 0), index_arr(1, 1)-1});
	else
		res += HMMPI::stringFormatArr(HMMPI::MessageRE("Параметры SGOF: {0:%d} (",
										 "SGOF parameters: {0:%d} ("), std::vector<int>{index_arr(1, 1) - index_arr(1, 0)}) + aux + ")\n";

	if (index_arr(2, 1) > index_arr(2, 0))
		res += HMMPI::stringFormatArr(HMMPI::MessageRE("Параметры KRIG: {0:%d} ({1:%d}..{2:%d})\n",
										 "KRIG parameters: {0:%d} ({1:%d}..{2:%d})\n"), std::vector<int>{index_arr(2, 1) - index_arr(2, 0), index_arr(2, 0), index_arr(2, 1)-1});
	else
		res += HMMPI::stringFormatArr(HMMPI::MessageRE("Параметры KRIG: {0:%d}\n",
										 "KRIG parameters: {0:%d}\n"), std::vector<int>{index_arr(2, 1) - index_arr(2, 0)});

	for (size_t i = 3; i < index_arr.ICount(); i++)
	{
		if (index_arr(i, 1) > index_arr(i, 0))
			res += HMMPI::stringFormatArr(HMMPI::MessageRE("Параметры INC{0:%d}: {1:%d}",
											 "INC{0:%d} parameters: {1:%d}"), std::vector<int>{(int)i-3, index_arr(i, 1) - index_arr(i, 0)}) +
				   HMMPI::stringFormatArr(" ({0:%d}..{1:%d})\n", std::vector<int>{index_arr(i, 0), index_arr(i, 1)-1});
		else
			res += HMMPI::stringFormatArr(HMMPI::MessageRE("Параметры INC{0:%d}: {1:%d}\n",
											 "INC{0:%d} parameters: {1:%d}\n"), std::vector<int>{(int)i-3, index_arr(i, 1) - index_arr(i, 0)});
	}

	res += HMMPI::stringFormatArr(HMMPI::MessageRE("Активные: {0:%zu}, всего: {1:%zu}\n",
									 	 	 	   "Active: {0:%zu}, total: {1:%zu}\n"), std::vector<size_t>{limits->get_act_ind().size(), limits->init.size()});

	return res;
}
//---------------------------------------------------------------------------
bool PhysModelHM::CheckLimits(const std::vector<double> &p) const
{
	// DEBUG
	//return true;	// DEBUG to avoid problems with fin diff close to boundary TODO
	// DEBUG

	if (p.size() != (size_t)ParamsDim())
		throw HMMPI::Exception("p.size() != ParamsDim() in PhysModelHM::CheckLimits");

	DECLKWD(limits, KW_limits, "LIMITS");
	if (RNK == 0)
		limits_msg = HMMPI::MessageRE("Нарушены ограничения:\n", "Bounds violated:\n");

	//FillParamsAll(p);		// before 01.09.2016
	params_all = p;

	size_t len = params_all.size();
	if (len != limits->init.size())
		throw HMMPI::EObjFunc(HMMPI::MessageRE("Не совпадает число параметров в LIMITS и общее число параметров и PhysModelHM::CheckLimits",
								 "Limits number and total parameters number do not match in PhysModelHM::CheckLimits"));

	bool res = true;
	const std::vector<double> min = limits->fullmin();
	const std::vector<double> max = limits->fullmax();
	for (size_t i = 0; i < len; i++)
	{
		if ((params_all[i] < min[i])||(params_all[i] > max[i]))
		{
			res = false;
			if (RNK == 0)
			{
				limits_msg += HMMPI::stringFormatArr(HMMPI::MessageRE("для параметра {0:%d} ",
														"for parameter {0:%d} "), std::vector<int>{(int)i + 1});
				limits_msg += HMMPI::stringFormatArr(HMMPI::MessageRE("нарушены границы [{0}, {1}]\n",
														"bounds [{0}, {1}] are violated\n"), std::vector<double>{min[i], max[i]});
			}
			break;
		}
	}

	if (res)
		res = check_limits_krig();
	if (res)
		res = check_limits_swgof();

	return res;
}
//---------------------------------------------------------------------------
double PhysModelHM::ObjFunc(const std::vector<double> &p)
{
	DECLKWD(incfiles, KW_incfiles, "INCFILES");
	DECLKWD(file, KW_datafile, "DATAFILE");
	DECLKWD(gas, KW_gas, "GAS");
	DECLKWD(wghts, KW_ofweights, "OFWEIGHTS");
#ifdef ECLFORMATTED
	DECLKWD(spec, KW_fsmspec, "FSMSPEC");
#else
	DECLKWD(datesW, KW_dates, "DATES");
#endif
	DECLKWD(vect, KW_eclvectors, "ECLVECTORS");
	DECLKWD(DIMS, KW_griddims, "GRIDDIMS");
	DECLKWD(smry, KW_funsmry, "FUNSMRY");			// f1 - well data
	DECLKWD(textsmry, KW_textsmry, "TEXTSMRY");		// f1 - well data
	DECLKWD(rst, KW_funrst, "FUNRST");				// f2 - SWAT
	DECLKWD(rstG, KW_funrstG, "FUNRSTG");			// f3 - SGAS
	DECLKWD(refmap, KW_refmap, "REFMAP");			// f4 - k
	DECLKWD(refmap_w, KW_refmap_w, "REFMAP_W");		// f4 - k
	DECLKWD(rstA, KW_funrstA, "FUNRSTA");				// f5 - R2
	DECLKWD(mapreg, KW_mapreg, "MAPREG");				// f5 - R2
	DECLKWD(mapss, KW_mapseisscale, "MAPSEISSCALE");	// f5 - R2
	DECLKWD(quadr, KW_regressquadr, "REGRESSQUADR");	// f5 - R2
	DECLKWD(regRs, KW_regressRs, "REGRESSRS");			// f5 - R2
	DECLKWD(regConstr, KW_regressConstr, "REGRESSCONSTR");	// f5 - R2

	int parallel_size = 0;				// parallel simulation size
	int parallel_rank = -1;
	if (comm != MPI_COMM_NULL)
	{
		MPI_Comm_size(comm, &parallel_size);
		MPI_Comm_rank(comm, &parallel_rank);
	}

	double res = 0;
	if (parallel_rank == 0)		// simulation is only done for comm-RANKS-0
	{
		int f_ind = RNK;		// = K->MPI_rank; -- before 29.07.2016

		if (!CheckLimits(p))
			throw HMMPI::EObjFunc(HMMPI::MessageRE("Параметры выходят за допустимый диапазон",
												   "Parameters are out of range"));
		if (file->GetState() != "")
			throw HMMPI::EObjFunc(HMMPI::MessageRE("Не задан корректный DATA-файл в EKDATAFILE",
									 "No correct DATA-file specified in EKDATAFILE"));

		//FillParamsAll(p);	// before 01.09.2016
		params_all = p;

		bool complete = false;
		while (!complete)
		{
			f1 = 0, f2 = 0, f3 = 0, f4 = 0, f5 = 0;

			std::ofstream sw;
			sw.exceptions(std::ios_base::badbit | std::ios_base::failbit);
			std::string mod_name = file->path + "/" + file->base_name + HMMPI::stringFormatArr("{0:%d}", std::vector<int>{f_ind});
			try
			{
				if (RNK == 0)
				{
					sw.open(log_file);
					write_params(sw);
				}

				WriteData(f_ind);
				GetParamsKrig();
				if ((pts.Length() != 0)&&(COEFFS.Length() != 0))
					WriteModel(f_ind);		// krig
				WriteSWOF(f_ind);
				WriteSGOF(f_ind);
				WriteINC(f_ind);
				RunSimulation(f_ind, parallel_size);

				// calculation of f1, f2, f3, f4, f5
				// f1 - well data
				if (wghts->w1 != 0)
				{
					std::string msg_vect = "";

#ifdef ECLFORMATTED
					spec->ind = spec->ReadData(mod_name + ".FSMSPEC", spec->Y, spec->M, spec->D, &msg_vect);
#else
					std::string msg_dat_short, msg_vec_short, msg_dat_full, msg_vec_full;
					HMMPI::EclSMRY SMRY_mod;			// load whole summary
					SMRY_mod.ReadFiles(mod_name);
#endif

#ifdef ECLFORMATTED
					if (HMMPI::FileModCompare(mod_name + ".FUNSMRY", file->GetDataFileName(f_ind)) < 0)		// FUNSMRY is younger than DATA
						throw HMMPI::Exception(HMMPI::stringFormatArr("Файл FUNSMRY был изменен до изменения файла DATA ({0:%d})",
																	  "File FUNSMRY was changed before DATA file ({0:%d})", f_ind));		// "small error" - i.e. not EObjFunc

	#ifndef PUNQADJ
					HMMPI::Vector2<double> SMRY = smry->ReadData(mod_name + ".FUNSMRY");	// model, UNIFOUT
	#else	// 	DERIVATIVES FROM ECLIPSE - ADJOINT
					HMMPI::Vector2<double> SMRY = smry->ReadData(mod_name, 1, (adjoint_run ? 41 : 83));			// model data from multiple output
					HMMPI::Vector2<std::vector<double>> grads2;
					if (adjoint_run)
					{
						grads2 = calc_derivatives_dRdP(mod_name);				// gradients from multiple output
						calc_derivatives_dRdP2(grads2, SMRY);					// gradients of WWCT, WGOR
					}
	#endif


	#ifdef PUNQGRADS		// abandoned piece of code
					// create indices for gradients
					int i0_wbhp, i1_wbhp, i0_wgor, i1_wgor, i0_wwct, i1_wwct;
					const int expected = 5*45*6;	// 5 regs, 45 subregs, 6 wells
					spec->GetKeywordIndRange(mod_name + ".FSMSPEC", "QWBHP", i0_wbhp, i1_wbhp);
					spec->GetKeywordIndRange(mod_name + ".FSMSPEC", "QWGOR", i0_wgor, i1_wgor);
					spec->GetKeywordIndRange(mod_name + ".FSMSPEC", "QWWCT", i0_wwct, i1_wwct);
					std::cout << i0_wbhp << "\t" << i0_wgor << "\t" << i0_wwct << "\n";
					std::cout << i1_wbhp << "\t" << i1_wgor << "\t" << i1_wwct << "\n";
					assert(i0_wbhp != -1 && i0_wgor != -1 && i0_wwct != -1);
					assert(i1_wbhp - i0_wbhp == expected && i1_wgor - i0_wgor == expected && i1_wwct - i0_wwct == expected);

					// read the gradients from eclipse
					std::vector<int> cache = spec->ind;
					spec->ind = std::vector<int>(expected);
					std::iota(spec->ind.begin(), spec->ind.end(), i0_wbhp);
					//std::cout << HMMPI::ToString(spec->ind, "%d") << "\n";	// DEBUG
					HMMPI::Vector2<double> Qwbhp = smry->ReadData(mod_name + ".FUNSMRY");

					std::iota(spec->ind.begin(), spec->ind.end(), i0_wgor);
					HMMPI::Vector2<double> Qwgor = smry->ReadData(mod_name + ".FUNSMRY");

					std::iota(spec->ind.begin(), spec->ind.end(), i0_wwct);
					HMMPI::Vector2<double> Qwwct = smry->ReadData(mod_name + ".FUNSMRY");

					// output gradients to ASCII for debug
					FILE *f = fopen("GRADIENTS_QWBHP.txt", "w");
					HMMPI::Mat(std::vector<double>(Qwbhp.Serialize(), Qwbhp.Serialize() + Qwbhp.ICount()*Qwbhp.JCount()), Qwbhp.ICount(), Qwbhp.JCount()).SaveASCII(f);
					fclose(f);
					f = fopen("GRADIENTS_QWGOR.txt", "w");
					HMMPI::Mat(std::vector<double>(Qwgor.Serialize(), Qwgor.Serialize() + Qwgor.ICount()*Qwgor.JCount()), Qwgor.ICount(), Qwgor.JCount()).SaveASCII(f);
					fclose(f);
					f = fopen("GRADIENTS_QWWCT.txt", "w");
					HMMPI::Mat(std::vector<double>(Qwwct.Serialize(), Qwwct.Serialize() + Qwwct.ICount()*Qwwct.JCount()), Qwwct.ICount(), Qwwct.JCount()).SaveASCII(f);
					fclose(f);

					// Gradients for PunqS3 turned out to have bugs; so this code is not developed further

	#endif
#else
					if (HMMPI::FileModCompare(mod_name + ".UNSMRY", file->GetDataFileName(f_ind)) < 0)		// UNSMRY is younger than DATA
						throw HMMPI::Exception(HMMPI::stringFormatArr("Файл UNSMRY был изменен до изменения файла DATA ({0:%d})",
																	  "File UNSMRY was changed before DATA file ({0:%d})", f_ind));		// "small error" - i.e. not EObjFunc

					HMMPI::Vector2<double> SMRY = SMRY_mod.ExtractSummary(datesW->dates, vect->vecs, msg_dat_short, msg_vec_short, msg_dat_full, msg_vec_full, K->StrListN());
					if (msg_dat_full != "")
					{
						msg_vect += msg_dat_full + "\n";
						K->AppText((std::string)HMMPI::MessageRE("ПРЕДУПРЕЖДЕНИЕ: ", "WARNING: ") + msg_dat_short + "\n");
						K->TotalWarnings++;
					}
					if (msg_vec_full != "")
					{
						msg_vect += msg_vec_full + "\n";
						K->AppText((std::string)HMMPI::MessageRE("ПРЕДУПРЕЖДЕНИЕ: ", "WARNING: ") + msg_vec_short + "\n");
						K->TotalWarnings++;
					}

#endif
					size_t Nsteps = SMRY.ICount();
					size_t Npars = SMRY.JCount();
					if (Npars != vect->sigma.size())
						throw HMMPI::EObjFunc(HMMPI::MessageRE("Неправильная длина массива в PhysModelHM::ObjFunc",
												 	 	 	  "Incorrect array length in PhysModelHM::ObjFunc"));

					HMMPI::Vector2<double> smry_hist;										// history!
					bool text_sigma = false;
					if (textsmry->data.Length() != 0)	// option 2
					{
						text_sigma = true;
						smry_hist = textsmry->pet_dat;	// 30.06.2013
					}
					else if (smry->data.Length() == 0)	// option 3
					{
#ifdef ECLFORMATTED
						// read model's FSMSPEC and FUNSMRY (history)
						spec->indH = spec->ReadDataH(mod_name + ".FSMSPEC", &msg_vect);
						smry_hist = smry->ReadData(mod_name + ".FUNSMRY", true);	// history!
#else
						smry_hist = SMRY_mod.ExtractSummary(datesW->dates, vect->vecs, msg_dat_short, msg_vec_short, msg_dat_full, msg_vec_full, K->StrListN(), "H");
						if (msg_vec_full != "")
						{
							msg_vect += msg_vec_full + "\n";
							K->AppText((std::string)HMMPI::MessageRE("ПРЕДУПРЕЖДЕНИЕ: ", "WARNING: ") + msg_vec_short + "\n");
							K->TotalWarnings++;
						}
#endif
					}
					else								// option 1
					{
#ifdef ECLFORMATTED
						smry_hist = smry->data;
#else
						throw HMMPI::Exception("This option for reading eclipse summary is not working for BINARY case");
#endif
					}

					std::vector<double> of1_full(Npars);
					int count_undef = 0;
					modelled_data = std::vector<double>();
					for (size_t p = 0; p < Npars; p++)		// legacy code (but it fills "ModelledData"); if correlations are used, of1_full and f1 will be re-written below
					{
						double of1 = 0;
						int count_steps = 0;
						for (size_t t = 0; t < Nsteps; t++)
						{
							double sigma = 0;
							if (text_sigma)
								sigma = smry_hist(t, p + Npars);
							else
								sigma = vect->sigma[p];

							if (!HMMPI::IsNaN(smry_hist(t, p)) && sigma != 0)	// skip checking SMRY, to avoid dependence on the current model
							{
								of1 += pow((SMRY(t, p) - smry_hist(t, p))/sigma, 2);
								count_steps++;

								modelled_data.push_back(SMRY(t, p));
							}
							else
								count_undef++;
						}
						f1 += of1;
						of1_full[p] = of1;
					}
					if (text_sigma)			// obj. func. with covariances, of1_full and f1 are redefined here
					{
						bool cov_is_diag;	// this flag is filled, but not used at the moment
						f1 = VCL->ObjFunc(SMRY, smry_hist, cov_is_diag);
						of1_full = VCL->of1;
					}

#ifdef PUNQADJ	// 	DERIVATIVES FROM ECLIPSE - ADJOINT
					if (adjoint_run)
					{
						const int Pdim = 45;
						assert(ParamsDim() == Pdim);
						assert(text_sigma);
						DataSens = HMMPI::Mat(modelled_data.size(), Pdim, 0);
						gradient = std::vector<double>(Pdim);

						for (int i0 = 0; i0 < Pdim; i0++)		// model parameters - 45 regions
						{
							HMMPI::Vector2<double> grads_of_region = KW_funrst::GradsOfRegion(grads2, i0);		// sensitivity of modelled data w.r.t. one parameter

							int count = 0;
							for (size_t p = 0; p < Npars; p++)	// fill modelled data sensitivities
								for (size_t t = 0; t < Nsteps; t++)
									if (!HMMPI::IsNaN(smry_hist(t, p)) && smry_hist(t, p + Npars) != 0)
										DataSens(count++, i0) = grads_of_region(t, p);

							gradient[i0] = 2 * VCL->ObjFunc(SMRY, smry_hist, &grads_of_region);
						}
					}
#endif

					if (RNK == 0)
					{
						sw << msg_vect;
						write_smry(sw, SMRY, smry_hist, of1_full, text_sigma);
						sw << HMMPI::stringFormatArr("f1 = {0}\n", std::vector<double>{f1});
						if (text_sigma)
							sw << (std::string)HMMPI::MessageRE("Была использована полная ковариационная матрица (при маленьких R она становится диагональной)\n",
																"Full covariance matrix was used (it becomes diagonal for small R)\n");
						sw << HMMPI::stringFormatArr(HMMPI::MessageRE("найдено неопределенных значений / нулевых сигм: {0:%d}\nвсего значений: {1:%d}\n\n",
														"found undefined values / zero sigmas: {0:%d}\ntotal values: {1:%d}\n\n"), std::vector<int>{count_undef, int(Npars*Nsteps)});
					}
				}

				// f2 - SWAT
				if (wghts->w2 != 0)
				{
					std::string funrst = mod_name + ".FUNRST";
					std::vector<std::vector<double>> RST = rst->ReadData(funrst, "'SWAT    '", false);

					size_t Nsteps = RST.size();
					if (Nsteps != rst->data.size())
						throw HMMPI::EObjFunc(HMMPI::MessageRE("Не совпадает число шагов для насыщенности в PhysModelHM::ObjFunc",
												 "Number of steps for saturation does not match in PhysModelHM::ObjFunc"));

					size_t Nvals = RST[0].size();
					if (Nvals != rst->data[0].size())
						throw HMMPI::EObjFunc(HMMPI::MessageRE("Не совпадает число значений в ячейках грида в PhysModelHM::ObjFunc",
												 "Number of values in grid cells does not match in PhysModelHM::ObjFunc"));

					int count_undef = 0;
					for (size_t i = 0; i < Nsteps; i++)
					{
						double of2 = 0;
						int count_cells = 0;
						for (size_t j = 0; j < Nvals; j++)
						{
							if (!HMMPI::IsNaN(RST[i][j]) && !HMMPI::IsNaN(rst->data[i][j]))
							{
								of2 += (RST[i][j] - rst->data[i][j])*(RST[i][j] - rst->data[i][j]);
								count_cells++;
							}
							else
								count_undef++;
						}
						if (count_cells != 0)
							of2 = sqrt(of2/count_cells);
						f2 += of2;
					}
					f2 /= Nsteps;

					if (RNK == 0)
					{
						sw << HMMPI::stringFormatArr("f2 = {0}\n", std::vector<double>{f2});
						sw << HMMPI::stringFormatArr(HMMPI::MessageRE("найдено неопределенных значений: {0:%d}\n\n",
														"found undefined values: {0:%d}\n\n"), std::vector<int>{count_undef});
					}
				}

				// f3 - SGAS
				if (wghts->w3 != 0)
				{
					std::string funrstg = mod_name + ".FUNRST";
					std::vector<std::vector<double>> RSTG = rstG->ReadData(funrstg, "'SGAS    '", false);

					size_t Nsteps = RSTG.size();
					if (Nsteps != rstG->data.size())
						throw HMMPI::EObjFunc(HMMPI::MessageRE("Не совпадает число шагов для насыщенности SGAS в PhysModelHM::ObjFunc",
												 "Number of steps for saturation SGAS does not match in PhysModelHM::ObjFunc"));

					size_t Nvals = RSTG[0].size();
					if (Nvals != rstG->data[0].size())
						throw HMMPI::EObjFunc(HMMPI::MessageRE("Не совпадает число значений в ячейках грида в PhysModelHM::ObjFunc",
												 "Number of values in grid cells does not match in PhysModelHM::ObjFunc"));

					int count_undef = 0;
					for (size_t i = 0; i < Nsteps; i++)
					{
						double of3 = 0;
						int count_cells = 0;
						for (size_t j = 0; j < Nvals; j++)
						{
							if (!HMMPI::IsNaN(RSTG[i][j]) && !HMMPI::IsNaN(rstG->data[i][j]))
							{
								of3 += (RSTG[i][j] - rstG->data[i][j])*(RSTG[i][j] - rstG->data[i][j]);
								count_cells++;
							}
							else
								count_undef++;
						}
						if (count_cells != 0)
							of3 = sqrt(of3/count_cells);
						f3 += of3;
					}
					f3 /= Nsteps;

					if (RNK == 0)
					{
						sw << HMMPI::stringFormatArr("f3 = {0}\n", std::vector<double>{f3});
						sw << HMMPI::stringFormatArr(HMMPI::MessageRE("найдено неопределенных значений: {0:%d}\n\n",
														"found undefined values: {0:%d}\n\n"), std::vector<int>{count_undef});
					}
				}

				// f4 - k_APRIORI
				if (wghts->w4 != 0)
				{
					grid = rst->ReadDataInit(mod_name + ".FINIT", "'PERMX   '");
					size_t Nvals = grid.size();
					if ((Nvals != refmap->data.size())||(Nvals != refmap_w->data.size()))
						throw HMMPI::EObjFunc(HMMPI::MessageRE("Не совпадает число значений в ячейках грида в EKREFMAP/EKREFMAP_W и текущей модели",
												 "Number of values in grid cells does not match in EKREFMAP/EKREFMAP_W and current model"));
					f4 = 0;
					double sumw = 0;
					int count_undef = 0;
					for (size_t j = 0; j < Nvals; j++)
					{
						if (!HMMPI::IsNaN(grid[j]) && !HMMPI::IsNaN(refmap->data[j]) && !HMMPI::IsNaN(refmap_w->data[j]))
						{
							f4 += pow((grid[j] - refmap->data[j])*refmap_w->data[j], 2);
							sumw += refmap_w->data[j]*refmap_w->data[j];
						}
						else
							count_undef++;
					}
					if (sumw != 0)
						f4 = sqrt(f4/sumw);

					if (RNK == 0)
					{
						sw << HMMPI::stringFormatArr("f4 = {0}\n", std::vector<double>{f4});
						sw << HMMPI::stringFormatArr(HMMPI::MessageRE("найдено неопределенных значений: {0:%d}\n\n",
														"found undefined values: {0:%d}\n\n"), std::vector<int>{count_undef});
					}
				}

				// f5 - seismic
				if (wghts->w5 != 0)
				{
					// PART 1
					// reading P, Sw, Sg, Rs...
					std::string fn_init = mod_name + ".FINIT";
					std::string funrst = mod_name + ".FUNRST";
					std::vector<Grid2D> Pres = rst->ReadDataGrid2D(funrst, "'PRESSURE'", DIMS->Nx, DIMS->Ny, fn_init);
					std::vector<Grid2D> Swat = rst->ReadDataGrid2D(funrst, "'SWAT    '", DIMS->Nx, DIMS->Ny, fn_init);
					std::vector<Grid2D> Sgas;
					std::vector<Grid2D> Rs;
					std::vector<Grid2D> Attr = rstA->GetGrid2D(DIMS->Nx, DIMS->Ny, fn_init);
					Grid2D Reg = mapreg->GetGrid2D(DIMS->Nx, DIMS->Ny, fn_init);
					Reg.Round();
					Grid2D SScale = mapss->GetGrid2D(DIMS->Nx, DIMS->Ny, fn_init);

					if (SScale.SignStats() == 0)
						throw HMMPI::EObjFunc(HMMPI::MessageRE("Масштабирующая карта должна иметь постоянный знак",
												 "Scaling map should have constant sign"));

					Pres = PhysModelHM::SubtractBase(Pres);
					Swat = PhysModelHM::SubtractBase(Swat);
					Attr = PhysModelHM::SubtractBase(Attr);

					size_t len = Attr.size();
					// add pet_seis for RML
					for (size_t c = 0; c < len; c++)
						Attr[c].Plus(pet_seis[c]);

					if ((len != Pres.size())||(len != Swat.size()))
						throw HMMPI::EObjFunc(HMMPI::MessageRE("Разное число гридов в PRESSURE, SWAT, ATTR",
												 "Different number of grids in PRESSURE, SWAT, ATTR"));

					std::vector<Grid2D*> SyncList;
					SyncList.push_back(&Reg);
					SyncList.push_back(&SScale);
					for (size_t i = 0; i < len; i++)
					{
						SyncList.push_back(&Pres[i]);
						SyncList.push_back(&Swat[i]);
						SyncList.push_back(&Attr[i]);
					}

					int cM = 2;
					if (gas->on == "ON")
					{
						Sgas = rst->ReadDataGrid2D(funrst, "'SGAS    '", DIMS->Nx, DIMS->Ny, fn_init);
						Sgas = PhysModelHM::SubtractBase(Sgas);
						cM = 3;
						if (len != Sgas.size())
							throw HMMPI::EObjFunc(HMMPI::MessageRE("Разное число гридов в SGAS, ATTR",
													 "Different number of grids in SGAS, ATTR"));
						for (size_t i = 0; i < len; i++)
						{
							SyncList.push_back(&Sgas[i]);
						}
					}

					std::vector<int> indRs = regRs->IndActive(cM == 3);
					if (indRs.size() > 0)		// Rs
					{
						Rs = rst->ReadDataGrid2D(funrst, "'RS      '", DIMS->Nx, DIMS->Ny, fn_init);
						Rs = PhysModelHM::SubtractBase(Rs);
						if (len != Rs.size())
							throw HMMPI::EObjFunc(HMMPI::MessageRE("Разное число гридов в RS, ATTR",
													 "Different number of grids in RS, ATTR"));
						for (size_t i = 0; i < len; i++)
						{
							SyncList.push_back(&Rs[i]);
						}
					}

					Grid2D::SynchronizeActive(SyncList);

					// PART 2
					// processing...
					std::vector<int> cQuad = quadr->IndActive(cM == 3);
					size_t cVar = cM + cQuad.size();
					size_t RsVar = indRs.size();

					//  [P, Sw, Sg | P2, Sw2, Sg2, PSw, PSg, SwSg | Rs, Rs2, RsP, RsSw, RsSg]
					//= [    cM	   |			cQuad			  |			 RsVar		    ]
					//= [					cVar				  |			 RsVar		    ]
					HMMPI::Vector2<Grid2D> dRES(len, cVar + RsVar);
					for (size_t t = 0; t < len; t++)
					{
						dRES(t, 0) = Pres[t];
						dRES(t, 1) = Swat[t];
						if (cM > 2)
							dRES(t, 2) = Sgas[t];

						for (size_t j = cM; j < cVar; j++)
						{
							int v1, v2;
							v1 = cQuad[j-cM] % 3;
							v2 = cQuad[j-cM] / 3;
							Grid2D aux = dRES(t, v1);
							aux.Mult(dRES(t, v2));
							dRES(t, j) = aux;
						}

						for (size_t j = 0; j < RsVar; j++)	// Rs
						{
							int v = indRs[j];
							Grid2D aux = Rs[t];
							if (v == 1)
								aux.Mult(Rs[t]);
							else if (v > 1)
								aux.Mult(dRES(t, v-2));

							dRES(t, cVar + j) = aux;
						}

						for (size_t p = 0; p < cVar + RsVar; p++)
							dRES(t, p).Mult(SScale);
					}

					//LinRegress::eps = 1e-20;	// 31.12.2012
					RegList RL;
					RL.ReadAllData(Attr, dRES, Reg, SScale, RLS, wghts->w5);
					std::vector<Grid2D> coeff_maps = RL.Regression(Attr, dRES, Reg);

					if (RNK == 0)
					{
						KW_undef *undef = dynamic_cast<KW_undef*>(K->GetKW_item("UNDEF"));
						for (size_t t = 0; t < len; t++)
						{
							std::string fn = this->CWD + HMMPI::stringFormatArr("/dA_{0:%d}.txt", std::vector<int>{(int)t+1});
							std::string fn_input = this->CWD + HMMPI::stringFormatArr("/dA_input_{0:%d}.txt", std::vector<int>{(int)t+1});
							std::string prop = HMMPI::stringFormatArr("dA_{0:%d}", std::vector<int>{(int)t+1});
							coeff_maps[t + cVar + RsVar + 1].SaveProp3D(fn, prop, undef->Ugrid, DIMS->Nz);
							Attr[t].SaveProp3D(fn_input, prop, undef->Ugrid, DIMS->Nz);
						}
						for (size_t c = 0; c < cVar + RsVar; c++)
						{
							std::string fn = this->CWD + HMMPI::stringFormatArr("/coeff_{0:%d}.txt", std::vector<int>{(int)c+1});
							std::string prop = HMMPI::stringFormatArr("COEF_{0:%d}", std::vector<int>{(int)c+1});
							coeff_maps[c + 1].SaveProp3D(fn, prop, undef->Ugrid, DIMS->Nz);
						}

						sw << "dA/dP  =" + regConstr->getConstrStr(0) + "\n";
						sw << "dA/dSw =" + regConstr->getConstrStr(1) + "\n";
						sw << "dA/dSg =" + regConstr->getConstrStr(2) + "\n";
						sw << "dA/dRs =" + regConstr->getConstrStr(3) + "\n";

						sw << HMMPI::stringFormatArr(HMMPI::MessageRE("Регрессия завершена для {0:%d} врем. шага(ов), {1:%d} карт(ы)\n",
														"Regression complete for {0:%d} time step(s), {1:%d} map(s)\n"), std::vector<int>{(int)len, (int)(cVar + RsVar)});
						sw << HMMPI::stringFormatArr(HMMPI::MessageRE("Средний R2 = {0}\n", "Average R2 = {0}\n"), std::vector<double>{RL.avgR2});
						sw << HMMPI::stringFormatArr(HMMPI::MessageRE("Составной R2 = {0}\n", "Composite R2 = {0}\n"), std::vector<double>{RL.composR2});
						sw << HMMPI::stringFormatArr(HMMPI::MessageRE("SSerr = {0}\n", "SSerr = {0}\n"), std::vector<double>{RL.SSerr});
						sw << HMMPI::stringFormatArr(HMMPI::MessageRE("SStot = {0}\n", "SStot = {0}\n"), std::vector<double>{RL.VA});
						sw << RL.ReportLog();
					}

					// PART 3
					// printing f5
					// f5 = 1 - RL.avgR2;
					f5 = RL.f5 / wghts->w5;				// 29.06.2013 | 12.12.2013
					if (RNK == 0)
					{
						sw << HMMPI::stringFormatArr("f5 = {0}\n\n", std::vector<double>{f5});
					}
				}

				res = wghts->w1*f1 + wghts->w2*f2 + wghts->w3*f3 + wghts->w4*f4 + wghts->w5*f5;
				if (RNK == 0)
				{
					sw << HMMPI::stringFormatArr(HMMPI::MessageRE(
										  "ц.ф. = {0}\n", "o.f. = {0}\n"), std::vector<double>{res});
					sw << HMMPI::stringFormatArr("     = {0} + {1} + {2}", std::vector<double>{wghts->w1*f1, wghts->w2*f2, wghts->w3*f3}) +
						  HMMPI::stringFormatArr(" + {0} + {1}\n", std::vector<double>{wghts->w4*f4, wghts->w5*f5});
					sw << HMMPI::stringFormatArr("       f1 = {0}\n", std::vector<double>{f1});
					sw << HMMPI::stringFormatArr("       f2 = {0}\n", std::vector<double>{f2});
					sw << HMMPI::stringFormatArr("       f3 = {0}\n", std::vector<double>{f3});
					sw << HMMPI::stringFormatArr("       f4 = {0}\n", std::vector<double>{f4});
					sw << HMMPI::stringFormatArr("       f5 = {0}\n", std::vector<double>{f5});
				}

				complete = true;
			}
			catch (const HMMPI::EObjFunc &e)	// immediate termination
			{
				if (sw.is_open())
					sw.close();

				throw e;
			}
			catch (const std::exception &e)
			{
				if (sw.is_open())
					sw.close();

				if (!ignore_small_errors)
				{
					complete = true;
					throw HMMPI::Exception(e.what());
				}
			}

			if (sw.is_open())
				sw.close();

			// deleting files
			try
			{
				if (RNK != 0)
				{
					if (f_ind != 0)
						remove(std::string(mod_name + ".DATA").c_str());

					remove(std::string(mod_name + ".FEGRID").c_str());
					remove(std::string(mod_name + ".FGRID").c_str());
					remove(std::string(mod_name + ".FINIT").c_str());
					remove(std::string(mod_name + ".FINSPEC").c_str());
					remove(std::string(mod_name + ".FRSSPEC").c_str());
					remove(std::string(mod_name + ".FSMSPEC").c_str());
					remove(std::string(mod_name + ".FUNRST").c_str());
					remove(std::string(mod_name + ".FUNSMRY").c_str());
					remove(std::string(mod_name + ".PRT").c_str());
					remove(std::string(mod_name + ".DBG").c_str());
					remove(std::string(mod_name + ".ECLEND").c_str());
					remove(std::string(mod_name + ".MSG").c_str());
					remove(std::string(mod_name + ".RSM").c_str());

					remove(std::string(mod_name + ".EGRID").c_str());
					remove(std::string(mod_name + ".GRID").c_str());
					remove(std::string(mod_name + ".INIT").c_str());
					remove(std::string(mod_name + ".INSPEC").c_str());
					remove(std::string(mod_name + ".RSSPEC").c_str());
					remove(std::string(mod_name + ".SMSPEC").c_str());
					remove(std::string(mod_name + ".UNRST").c_str());
					remove(std::string(mod_name + ".UNSMRY").c_str());

					remove(std::string(mod_name + DIMS->krig_file + ".inc").c_str());
					remove(std::string(mod_name + DIMS->swof_file + ".inc").c_str());
					remove(std::string(mod_name + DIMS->sgof_file + ".inc").c_str());
					for (size_t c = 0; c < incfiles->mod.size(); c++)
						remove(std::string(mod_name + incfiles->mod[c] + ".inc").c_str());;
				}
			}
			catch (...)
			{
			}
		}
	}

	// Bcast res and modelled_data
	if (comm != MPI_COMM_NULL)
	{
		MPI_Bcast(&res, 1, MPI_DOUBLE, 0, comm);
		HMMPI::Bcast_vector(modelled_data, 0, comm);
	}

	if (sign == -1)
		return -res;
	else
		return res;
}
//---------------------------------------------------------------------------
std::vector<double> PhysModelHM::ObjFuncGrad(const std::vector<double> &params)
{
#ifdef PUNQADJ
	adjoint_run = true;
	ObjFunc(params);
	adjoint_run = false;

	return gradient;		// filled in ObjFunc
#else
	throw HMMPI::Exception("Illegal call to PhysModelHM::ObjFuncGrad");
#endif
}
//---------------------------------------------------------------------------
size_t PhysModelHM::ModelledDataSize() const
{
	if (modelled_data_size == 0)
		modelled_data_size = Data().size();

	return modelled_data_size;
}
//---------------------------------------------------------------------------
std::string PhysModelHM::ObjFuncMsg() const
{
	DECLKWD(regCon, KW_regressConstr, "REGRESSCONSTR");
	DECLKWD(wghts, KW_ofweights, "OFWEIGHTS");

	std::string msg = IndexMsg();
	std::string quadr_t = "";
	if (wghts->w5 > 0)
	{
		if (regCon->hasQuadr(0))
			quadr_t += " P";

		if (regCon->hasQuadr(1))
			quadr_t += " Sw";

		if (regCon->hasQuadr(2))
			quadr_t += " Sg";

		if (regCon->hasQuadr(3))
			quadr_t += " Rs";
	}

	if (quadr_t != "")
		msg += std::string(HMMPI::MessageRE("\nКвадратичные члены заданы для:", "\nQuadratic terms present for:")) + quadr_t + "\n\n";

	return msg;
}
//---------------------------------------------------------------------------
void PhysModelHM::Constraints(HMMPI::Vector2<double> &matrC, std::vector<double> &vectb)
{
	std::vector<std::vector<double>> C;
	std::vector<double> b;

	add_limits_def(C, b);
	add_limits_swgof(C, b);
	add_limits_krig(C, b);

	size_t count = C.size();
	size_t dim = C[0].size();
	matrC = HMMPI::Vector2<double>(count, dim);
	vectb = move(b);

	for (size_t i = 0; i < count; i++)
	{
		for (size_t j = 0; j < dim; j++)
		{
			matrC(i, j) = C[i][j];
		}
	}

	// no synchronization so far

	std::ofstream sw;
	sw.exceptions(std::ios_base::badbit | std::ios_base::failbit);
	try
	{
		sw.open(CWD + "/" + lin_constr_file);
		for (size_t i = 0; i < count; i++)
		{
			for (size_t j = 0; j < dim; j++)
			{
				sw << HMMPI::stringFormatArr("{0}\t", std::vector<double>{matrC(i, j)});
			}
			sw << HMMPI::stringFormatArr("{0}\n", std::vector<double>{vectb[i]});
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
//---------------------------------------------------------------------------
void PhysModelHM::PerturbWell(double w1)
{
	DECLKWD(textsmry, KW_textsmry, "TEXTSMRY");

	if (w1 > 0)
	{
		double sgm = 1/sqrt(w1);
		std::string stat_check = HMMPI::MessageRE("Статистическая проверка хи-2 для возмущенных скв. данных:",
									  "Chi-2 statistical check for perturbed well data:");
		double chi2 = 0;
		if (VCL != 0)
			chi2 = VCL->PerturbData(textsmry->pet_dat, &textsmry->randn, sgm);

		stat_check += HMMPI::stringFormatArr("\t{0:%g}\n", std::vector<double>{chi2});
		K->AppText(stat_check);
	}
}
//---------------------------------------------------------------------------
void PhysModelHM::PerturbSeis(double w5)
{
	DECLKWD(DIMS, KW_griddims, "GRIDDIMS");
	DECLKWD(undef, KW_undef, "UNDEF");
	DECLKWD(textsmry, KW_textsmry, "TEXTSMRY");

	size_t count = pet_seis.size();	// time steps
	if (w5 > 0)
	{
//		initial version - no correlations
//		double sigma = 1/sqrt(w5);
//		for (size_t c = 0; c < count; c++)
//		{
//			int cx = pet_seis[0].CountX();
//			int cy = pet_seis[0].CountY();
//			for (int i = 0; i < cx; i++)
//				for (int j = 0; j < cy; j++)
//					pet_seis[c].data[i][j] = sigma * textsmry->randn.get();
//		}
//		new version
		double sigma = 1/sqrt(w5);
		std::string stat_check = HMMPI::MessageRE("Статистическая проверка хи-2 для возмущенной сейсмики\n",
									  "Chi-2 statistical check for perturbed seismic\n");
		for (int r = 0; r < RLS->CountReg(); r++)	// regions
		{
			int SZ = RLS->VecSize(r);
			std::vector<double> v(SZ);
			std::vector<double> w(SZ);
			stat_check += HMMPI::stringFormatArr(HMMPI::MessageRE("Регион {0:%d}", "Region {0:%d}"), std::vector<int>{RLS->RegNum(r)});
			for (size_t c = 0; c < count; c++)		// time steps
			{
				for (int i = 0; i < SZ; i++)
					v[i] = textsmry->randn.get() * sigma;
				RLS->L_v(v, w, r);
				RLS->vec2grid(w, pet_seis[c], r);

				// statistical checks
				std::vector<double> z(SZ);
				RLS->Linv_v(w, z, r);
				double sum = 0;
				for (int j = 0; j < SZ; j++)
					sum += z[j]*z[j];
				sum *= w5;
				stat_check += HMMPI::stringFormatArr("\t{0:%g}", std::vector<double>{sum});
			}
			stat_check += "\n";
		}
		K->AppText(stat_check);

#ifdef WRITE_PET_DATA
		for (size_t c = 0; c < count; c++)
		{
			std::string fn = HMMPI::stringFormatArr(CWD + "/pet_seis_{0:%d}.txt", std::vector<int>{(int)c});
			pet_seis[c].SaveProp3D(fn, HMMPI::stringFormatArr("ATTR_{0:%d}", std::vector<int>{(int)c}), undef->Ugrid, DIMS->Nz);
		}
#endif
	}
}
//---------------------------------------------------------------------------
void PhysModelHM::PerturbData()
{
	DECLKWD(wghts, KW_ofweights, "OFWEIGHTS");
	DECLKWD(textsmry, KW_textsmry, "TEXTSMRY");
	DECLKWD(RML, KW_RML, "RML");

	if (wghts->w1 > 0 && textsmry->data.Length() == 0 && RML->on == "ON")
		throw HMMPI::Exception("For ECLIPSE model, RML is only possible for history from TEXTSMRY");

	// randomization + MPI synchronization
	if (K->MPI_rank == 0)		// master branch
	{
		std::string errmsg = "";
		if (wghts->w1 > 0)
		{
			int size = 0;
			const double *dat = 0;
			try
			{
				if (textsmry->data.Length() != 0)
					PerturbWell(wghts->w1);

				dat = textsmry->pet_dat.Serialize();
				size = (int)textsmry->pet_dat.ICount()*textsmry->pet_dat.JCount();

				// save perturbed data
				std::ofstream sw("TextSMRY_RML.txt", std::ios::out);
				write_smry_hist(sw, textsmry->pet_dat, true);
				sw.close();
			}
			catch (std::exception &e)
			{
				errmsg = e.what();
			}

			for (int i = 1; i < K->MPI_size; i++)
				MPI_Ssend((void *)dat, size, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);

			// don't delete dat
		}
		if (wghts->w5 > 0)
		{
			int size = 0;
			const double *dat = 0;
			try
			{
				PerturbSeis(wghts->w5);
				dat = Grid2D::Serialize(pet_seis);
				size = PetSeisLen();
			}
			catch (std::exception &e)
			{
				errmsg = e.what();
			}

			for (int i = 1; i < K->MPI_size; i++)
				MPI_Ssend((void *)dat, size, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);

			delete [] dat;
		}
		if (errmsg != "")
			throw HMMPI::Exception(errmsg);
	}
	else				// worker branches
	{
		if (wghts->w1 > 0)
		{
			int size = (int)textsmry->pet_dat.ICount()*textsmry->pet_dat.JCount();
			double *dat = new double[size];
			MPI_Status stat;

			MPI_Recv(dat, size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &stat);
			textsmry->pet_dat.Deserialize(dat);
			delete [] dat;
		}
		if (wghts->w5 > 0)
		{
			int size = PetSeisLen();
			double *dat = new double[size];
			MPI_Status stat;

			MPI_Recv(dat, size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &stat);
			Grid2D::Deserialize(pet_seis, dat);
			delete [] dat;
		}
	}
}
//---------------------------------------------------------------------------
std::vector<HMMPI::Mat> PhysModelHM::CorrBlocks() const
{
	if (VCL == 0)
		throw HMMPI::Exception("VCL == 0 in PhysModelHM::CorrBlocks");

	std::vector<HMMPI::Mat> res(VCL->vec_vce().size());
	bool cov_is_diag = true;
	size_t totsz = 0;							// total size of entire matrix
	for (size_t i = 0; i < res.size(); i++)
	{
		size_t sz = VCL->vec_vce()[i].sz;
		totsz += sz;
		if (VCL->vec_vce()[i].R0 > 0.01)
		{
			res[i] = HMMPI::Mat(std::vector<double>(VCL->vec_vce()[i].C, VCL->vec_vce()[i].C + sz*sz), sz, sz);		// full matrix
			cov_is_diag = false;
		}
		else
			res[i] = HMMPI::Mat(sz, 1, 1.0);	// only diagonal
	}

#ifdef DIAGCOV_1X1
	if (cov_is_diag)
		res = std::vector<HMMPI::Mat>(totsz, HMMPI::Mat(1, 1, 1.0));	// elementary 1x1 blocks
#endif

	return res;
}
//---------------------------------------------------------------------------
std::vector<double> PhysModelHM::Std() const
{
	if (VCL == 0)
		throw HMMPI::Exception("VCL == 0 in PhysModelHM::Std");

	std::vector<double> res;
	for (size_t i = 0; i < VCL->vec_vce().size(); i++)
	{
		res.reserve(res.size() + VCL->vec_vce()[i].sz);
		res.insert(res.end(), VCL->vec_vce()[i].sigma.begin(), VCL->vec_vce()[i].sigma.end());
	}

	return res;
}
//---------------------------------------------------------------------------
std::vector<double> PhysModelHM::Data() const
{
	DECLKWD(textsmry, KW_textsmry, "TEXTSMRY");
	if (textsmry->data.Length() == 0)
		throw HMMPI::Exception("TEXTSMRY not defined in PhysModelHM::Data");

	HMMPI::Vector2<double> smry_hist = textsmry->pet_dat;
	size_t Nsteps = smry_hist.ICount();
	size_t Npars = smry_hist.JCount() / 2;

	std::vector<double> res;
	res.reserve(Npars*Nsteps);		// some extra space may be reserved
	for (size_t p = 0; p < Npars; p++)
		for (size_t t = 0; t < Nsteps; t++)
		{
			double sigma = smry_hist(t, p + Npars);
			if (!HMMPI::IsNaN(smry_hist(t, p)) && sigma != 0)
				res.push_back(smry_hist(t, p));
		}

	return res;
}
//---------------------------------------------------------------------------
const HMMPI::SimSMRY *PhysModelHM::get_smry() const
{
	throw HMMPI::Exception("Illegal call to PhysModelHM::get_smry()");
}
//---------------------------------------------------------------------------
// PMEclipse
//---------------------------------------------------------------------------
void PMEclipse::write_smry(std::ofstream &sw, const HMMPI::Vector2<double> &smry_mod, const HMMPI::Vector2<double> &smry_hist, const std::vector<double> &of_vec, bool text_sigma, bool only_hist)
{
	DECLKWD(datesW, KW_dates, "DATES");
	DECLKWD(vect, KW_eclvectors, "ECLVECTORS");
	DECLKWD(params, KW_parameters, "PARAMETERS");		// kw->Add_pre("PARAMETERS") is in CTOR

	size_t Nsteps = datesW->D.size();
	size_t Nvect = vect->sigma.size();

	const int Width = 14;			// each column's min width
	const int DateWidth = 21;		// 'date' column min width
	char buffM[HMMPI::BUFFSIZE], buffH[HMMPI::BUFFSIZE], buffS[HMMPI::BUFFSIZE];
	char workM[HMMPI::BUFFSIZE], workH[HMMPI::BUFFSIZE], workS[HMMPI::BUFFSIZE];

	// a) form the headers
	std::string headerD1, headerD2, headerD3;
	std::string headerM1, headerM2, headerM3;																	// D1 [M1] H1 S1
	std::string headerH1, headerH2, headerH3;																	// D2 [M2] H2 S2
	std::string headerS1, headerS2, headerS3;																	// D3 [M3] H3 S3

	const std::string date_hdr = HMMPI::MessageRE("дата/время", "date/time");
	const int date_hdr_add = (int)date_hdr.length() - HMMPI::StrLen(date_hdr);
	assert(date_hdr_add >= 0);
	sprintf(buffM, "%-*s", DateWidth, "");
	sprintf(buffH, "%-*s", DateWidth + date_hdr_add, date_hdr.c_str());
	const std::string date_gap = buffM;
	headerD1 += buffM;
	headerD2 += buffM;
	headerD3 += buffH;

	for (size_t i = 0; i < Nvect; i++)
	{
		sprintf(workM, "mod%zu", i+1);
		sprintf(workH, "hist%zu", i+1);
		sprintf(workS, "sigma%zu", i+1);
		sprintf(buffM, "\t%-*.*s", Width, HMMPI::BUFFSIZE-5, workM);
		sprintf(buffH, "\t%-*.*s", Width, HMMPI::BUFFSIZE-5, workH);
		sprintf(buffS, "\t%-*.*s", Width, HMMPI::BUFFSIZE-5, workS);
		headerM1 += buffM;
		headerH1 += buffH;
		headerS1 += buffS;

		sprintf(buffM, "\t%-*s", Width, vect->WGname[i].c_str());
		headerM2 += buffM;
		headerH2 += buffM;
		headerS2 += buffM;

		sprintf(buffM, "\t%-*s", Width, vect->vect[i].c_str());
		sprintf(buffH, "\t%-*s", Width, (vect->vect[i] + "H").c_str());
		sprintf(buffS, "\t%-*s", Width, (vect->vect[i] + "S").c_str());
		headerM3 += buffM;
		headerH3 += buffH;
		headerS3 += buffS;
	}
	if (!only_hist)
	{
		sw << std::string(HMMPI::MessageRE("Вектора (модельные, исторические, сигмы):\n", "Vectors summary (modelled, historical, sigmas):\n"));
		sw << headerD1 + headerM1 + headerH1 + headerS1 + "\n" + headerD2 + headerM2 + headerH2 + headerS2 + "\n" + headerD3 + headerM3 + headerH3 + headerS3 + "\n";
	}
	else
	{
		sw << std::string(HMMPI::MessageRE("Вектора (исторические [RML], сигмы):\n", "Vectors summary (historical [RML], sigmas):\n"));
		sw << headerD1 + headerH1 + headerS1 + "\n" + headerD2 + headerH2 + headerS2 + "\n" + headerD3 + headerH3 + headerS3 + "\n";
	}

	// b) output vectors
	for (size_t i = 0; i < Nsteps; i++)
	{
		sprintf(buffM, "%-*s", DateWidth, datesW->dates[i].ToString().c_str());

		sw << buffM;
		if (!only_hist)
			for (size_t j = 0; j < Nvect; j++)
			{
				sprintf(buffM, "\t%-*.9g", Width, smry_mod(i, j));
				sw << buffM;
			}
		for (size_t j = 0; j < Nvect; j++)
		{
			sprintf(buffM, "\t%-*.9g", Width, smry_hist(i, j));
			sw << buffM;
		}
		for (size_t j = 0; j < Nvect; j++)
		{
			double sigma = 0;
			if (text_sigma)
				sigma = smry_hist(i, j + Nvect);
			else
				sigma = vect->sigma[j];

			sprintf(buffM, "\t%-*.9g", Width, sigma);
			sw << buffM;
		}
		sw << "\n";
	}

	// c) output "of_vec" etc
	sw << date_gap;
	if (!only_hist)
		for (size_t j = 0; j < Nvect; j++)
		{
			sprintf(buffM, "\t%-*.9g", Width, of_vec[j]);
			sw << buffM;
		}

	for (size_t j = 0; j < Nvect; j++)		// spaces
	{
		sprintf(buffM, "\t%-*s", Width, "");
		sw << buffM;
	}

	for (size_t j = 0; j < Nvect; j++)		// correlation radii (below sigmas)
	{
		sprintf(buffM, "\t%-*.9g", Width, vect->R[j]);
		sw << buffM;
	}
	sw << "\n";

	if (!only_hist)
	{
		int MaxLen = Nvect;					// number of rows to be written
		const int WidthWell = 10;			// min width for well, property columns
		std::vector<int> perm = HMMPI::SortPermutation(of_vec.begin(), of_vec.end());
		std::vector<std::string> wgname = HMMPI::Reorder(vect->WGname, perm);			// form the sorted arrays
		std::vector<std::string> propname = HMMPI::Reorder(vect->vect, perm);
		std::vector<double> ofval = HMMPI::Reorder(of_vec, perm);

		int Nparams = 0;
		std::vector<std::string> parnames;												// sorted params names
		std::vector<double> priorof;													// sorted prior o.f.
		if (outer_post_diag != nullptr)
		{
			Nparams = outer_post_diag->prior_contrib.size();
			if (Nparams > MaxLen)
				MaxLen = Nparams;
			std::vector<int> perm2 = HMMPI::SortPermutation(outer_post_diag->prior_contrib.begin(), outer_post_diag->prior_contrib.end());
			parnames = HMMPI::Reorder(params->name, perm2);
			priorof = HMMPI::Reorder(outer_post_diag->prior_contrib, perm2);
		}

		sprintf(buffM, "\n%-*s\t%-*s\t%-*s\t\t\t\t\t%-*s\t%-*s\t%-*s", WidthWell, "Well", WidthWell, "property", Width, "o.f.", WidthWell, "well", WidthWell, "property", Width, "o.f. (decreasing order)");
		sprintf(buffH, "\n%-*s\t%-*s\t%-*s\t\t\t\t\t%-*s\t%-*s\t%-*s", WidthWell, "Скважина", WidthWell, "свойство", Width, "ц.ф.", WidthWell, "скважина", WidthWell, "свойство", Width, "ц.ф. (по убыванию)");

		sprintf(workM, "\t\t\t\t\t%-*s\t%-*s\t\t\t\t\t%-*s\t%-*s\n", WidthWell+4, "parameter", Width, "prior o.f.", WidthWell+4, "parameter", Width, "prior o.f. (decreasing order)");
		sprintf(workH, "\t\t\t\t\t\t%-*s\t%-*s\t\t\t\t\t\t%-*s\t%-*s\n", WidthWell+12, "параметр", Width, "апр. ц.ф.", WidthWell+12, "параметр", Width, "апр. ц.ф. (по убыванию)");

		sw << (std::string)HMMPI::MessageRE(buffH, buffM);
		if (outer_post_diag != nullptr)
			sw << (std::string)HMMPI::MessageRE(workH, workM);
		else
			sw << "\n";

		for (int j = 0; j < MaxLen; j++)
		{
			// Likelihood
			if (j < (int)Nvect)
				sprintf(buffM, "%-*s\t%-*s\t%-*.2f\t\t\t\t\t%-*s\t%-*s\t%-*.2f", WidthWell, vect->WGname[j].c_str(), WidthWell, vect->vect[j].c_str(), Width, of_vec[j],
																				   WidthWell, wgname[Nvect-1-j].c_str(), WidthWell, propname[Nvect-1-j].c_str(), Width, ofval[Nvect-1-j]);
			else
				sprintf(buffM, "%-*s\t%-*s\t%-*s\t\t\t\t\t%-*s\t%-*s\t%-*s", WidthWell, "", WidthWell, "", Width, "", WidthWell, "", WidthWell, "", Width, "");

			sw << buffM;
			if (outer_post_diag == nullptr)
				sw << "\n";
			else					// Prior
			{
				if (j < Nparams)
					sprintf(buffM, "\t\t\t\t\t\t\t%-*s\t%-*.2f\t\t\t\t\t%-*s\t%-*.2f\n", WidthWell+4, params->name[j].c_str(), Width, outer_post_diag->prior_contrib[j],
																			 	 	 	 WidthWell+4, parnames[Nparams-1-j].c_str(), Width, priorof[Nparams-1-j]);
				else
					sprintf(buffM, "\t\t\t\t\t\t\t%-*s\t%-*s\t\t\t\t\t%-*s\t%-*s\n", WidthWell+4, "", Width, "", WidthWell+4, "", Width, "");
				sw << buffM;
			}
		}
	}
}
//---------------------------------------------------------------------------
std::string PMEclipse::form_prior(double &pr_of) const		// if prior info is available, returns a message (and fills its value 'pr_of'); returns "" otherwise
{
	if (outer_post_diag != nullptr)
	{
		int Nparams = outer_post_diag->prior_contrib.size();
		pr_of = 0;
		for (int i = 0; i < Nparams; i++)
			pr_of += outer_post_diag->prior_contrib[i];

		char buffEN[HMMPI::BUFFSIZE], buffRU[HMMPI::BUFFSIZE];
		sprintf(buffEN, "Prior for %d parameter(s) = %.8g\n", Nparams, pr_of);
		sprintf(buffRU, "Априорное распределение для %d параметр(ов) = %.8g\n", Nparams, pr_of);
		return HMMPI::MessageRE(buffRU, buffEN);
	}
	else
		return "";
}
//---------------------------------------------------------------------------
void PMEclipse::perturb_well()
{
	DECLKWD(textsmry, KW_textsmry, "TEXTSMRY");

	double chi2 = 0;
	if (VCL != 0)
	{
		textsmry->pet_dat = textsmry->data;
		chi2 = VCL->PerturbData(textsmry->pet_dat, &textsmry->randn, 1.0);
	}

	K->AppText(HMMPI::stringFormatArr("Статистическая проверка хи-2 для возмущенных скв. данных: {0:%g}\n",
	  	  	  	  	  	  	  	  	  "Chi-2 statistical check for perturbed well data: {0:%g}\n", chi2));
}
//---------------------------------------------------------------------------
PMEclipse::PMEclipse(Parser_1 *k, KW_item *kw, std::string cwd, MPI_Comm c) : Sim_small_interface(k, kw, c), K(k), CWD(cwd), log_file(cwd + "/ObjFuncLog.txt"), modelled_data_size(0), cov_is_diag(false), VCL(0)
{
	DECLKWD(mod, KW_model, "MODEL");
	DECLKWD(datesW, KW_dates, "DATES");
	DECLKWD(groups, KW_groups, "GROUPS");
	DECLKWD(Sdate, KW_startdate, "STARTDATE");
	DECLKWD(vect, KW_eclvectors, "ECLVECTORS");
	DECLKWD(textsmry, KW_textsmry, "TEXTSMRY");
	DECLKWD(parameters, KW_parameters, "PARAMETERS");

	kw->Start_pre();
	kw->Add_pre("PARAMETERS");
	kw->Add_pre("TEMPLATES");
	kw->Add_pre("MODEL");
	kw->Add_pre("DATES");
	kw->Add_pre("GROUPS");
	kw->Add_pre("STARTDATE");
	kw->Add_pre("ECLVECTORS");
	kw->Add_pre("SIMCMD");
	kw->Finish_pre();

	con = parameters;
	bool print = true;
	if (dynamic_cast<KW_runOptProxy*>(kw) != 0)		// no print for RUNOPTPROXY
		print = false;

	if (mod->simulator == "ECL")
		smry = new HMMPI::EclSMRY;
	else if (mod->simulator == "TNAV")
		smry = new HMMPI::tNavSMRY(groups->sec_obj, Sdate->start);
	else
		throw HMMPI::Exception("Не распознан тип симулятора " + mod->simulator, "Unrecognized simulator type " + mod->simulator);

	if (textsmry->data.Length() != 0)
	{
		kw->Start_pre();
		kw->Add_pre("TEXTSMRY");
		kw->Finish_pre();

		VCL = new VectCorrList;
		VCL->ownerCount = 1;
		VCL->LoadData(textsmry->data, datesW->zeroBased(), vect->R, vect->corr);
		if (print)
			K->AppText(HMMPI::MessageRE("TEXTSMRY задан -> история берется из TEXTSMRY\n",
										"TEXTSMRY is defined -> history is taken from TEXTSMRY\n"));
	}
	else if (print)
		K->AppText(std::string(HMMPI::MessageRE("TEXTSMRY не задан -> история берется из модельного ",
												"TEXTSMRY is not defined -> history is taken from model ")) + HMMPI::EraseSubstr(smry->data_file(), "./") + "\n");
	name = "SIM";
	ignore_small_errors = true;

#ifdef TESTCTOR
	std::ofstream testf(HMMPI::stringFormatArr("TESTCTOR_out_{0:%d}.txt", std::vector<int>{RNK}), std::ios::app);
	testf << "rank " << RNK << ", PMEclipse easy CTOR, this = " << this << "\n";
	testf.close();
#endif
}
//---------------------------------------------------------------------------
PMEclipse::PMEclipse(const PMEclipse &PM) : Sim_small_interface(PM), K(PM.K), CWD(PM.CWD), log_file(PM.log_file), obj_func_msg(PM.obj_func_msg), cov_is_diag(PM.cov_is_diag), VCL(PM.VCL), smry(PM.smry->Copy())
{
	modelled_data_size = PM.modelled_data_size;
	ignore_small_errors = PM.ignore_small_errors;

	if (VCL != 0)
		VCL->ownerCount++;
}
//---------------------------------------------------------------------------
int PMEclipse::ParamsDim() const noexcept
{
	DECLKWD(parameters, KW_parameters, "PARAMETERS");
	return parameters->init.size();
}
//---------------------------------------------------------------------------
PMEclipse::~PMEclipse()
{
	delete smry;

	if (VCL != 0)
	{
		VCL->ownerCount--;
		if (VCL->ownerCount == 0)
		{
			delete VCL;
			VCL = 0;
		}
		else if (VCL->ownerCount < 0)
			throw HMMPI::Exception("VCL->ownerCount < 0");
	}

#ifdef TESTCTOR
	std::ofstream testf(HMMPI::stringFormatArr("TESTCTOR_out_{0:%d}.txt", std::vector<int>{RNK}), std::ios::app);
	testf << "rank " << RNK << ", PMEclipse -- DTOR --, this = " << this << "\n";
	testf.close();
#endif
}
//---------------------------------------------------------------------------
//	PMEclipse::ObjFunc() outline, showing how execution control is managed in MPI:
//
//	if (comm_rank == 0)
//	{A
//		while (complete == 0)
//		{B
//			try
//			{C
//				MPI_BarrierSleepy(comm);
//				simcmd->RunCmd(comm);
//				...
//				...
//			}C
//			catch (-> immediate termination, fill err_msg)
//			catch (-> terminate or re-run, fill err_msg)
//
//			MPI_BarrierSleepy(comm);
//			MPI_Bcast(complete);
//		}B
//	}A
//	else	  	  // comm_rank != 0
//	{A'
//		while (complete == 0)
//		{B'
//			try
//			{C'
//				MPI_BarrierSleepy(comm);
//				simcmd->RunCmd(comm);
//			}C'
//			catch (...)
//
//			MPI_BarrierSleepy(comm);
//			MPI_Bcast(complete);
//		}B'
//	}A'
//
//	MPI_BarrierSleepy(comm);
//	MPI_Bcast(warning_count);
//	MPI_Bcast(err_msg);
//	if (err_msg[0] != 0)
//		throw Exception(err_msg);		// sync exception
//
double PMEclipse::ObjFunc(const std::vector<double> &params)
{
	DECLKWD(parameters, KW_parameters, "PARAMETERS");
	DECLKWD(templ, KW_templates, "TEMPLATES");
	DECLKWD(datesW, KW_dates, "DATES");
	DECLKWD(vect, KW_eclvectors, "ECLVECTORS");
	DECLKWD(textsmry, KW_textsmry, "TEXTSMRY");
	DECLKWD(simcmd, KW_simcmd, "SIMCMD");

	int comm_size = 0;		// parallel simulation size
	int comm_rank = -1;
	if (comm != MPI_COMM_NULL)
	{
		MPI_Comm_size(comm, &comm_size);
		MPI_Comm_rank(comm, &comm_rank);
	}

	std::vector<double> of_vec;	// o.f. contribution for each vector, its sum is the final result
	double res = 0;
	int warning_count = 0;
	char err_msg[HMMPI::BUFFSIZE];
	err_msg[0] = 0;
	int complete = 0;			// while-loop controller: 0 - running, 1 - finished, 2 - error

	if (comm_rank == 0)			// processing is done on comm-RANKS-0, simulation - on all ranks
	{
		while (complete == 0)	// make several attempts to calculate obj. func.
		{
			std::ofstream sw;	// ObjFuncLog.txt
			sw.exceptions(std::ios_base::badbit | std::ios_base::failbit);
			try					// this try-catch block is intended to separate severe errors (EObjFunc) from everything else
			{
				// 1. Substitute the parameters and run the model
				if (!CheckLimits(params))
					throw HMMPI::EObjFunc(HMMPI::MessageRE("Параметры выходят за допустимый диапазон",
														   "Parameters are out of range"));

				HMMPI::TagPrintfMap *tmap = parameters->get_tag_map();		// object handling the parameters - fill it!
				tmap->SetSize(comm_size);
				std::vector<double> par_external = parameters->InternalToExternal(params);
				tmap->SetDoubles(parameters->name, par_external);
				std::string templ_msg = templ->WriteFiles(*tmap);			// MOD and PATH for "tmap" are set here, simcmd->cmd_work is also filled here

				HMMPI::MPI_BarrierSleepy(comm);
				simcmd->RunCmd(comm);

				// 2. Get modelled data
				std::string model_name = (*tmap)["PATH"]->ToString() + "/" + (*tmap)["MOD"]->ToString();
				std::string msg_vect_file = "", msg_vect_stdout = "";		// ultimate "vectors" messages
				std::string msg_dat_short, msg_vec_short, msg_dat_full, msg_vec_full;		// work messages; the full versions are for file output

				smry->ReadFiles(model_name);						// load whole summary
				HMMPI::Vector2<double> SMRY = smry->ExtractSummary(datesW->dates, vect->vecs, msg_dat_short, msg_vec_short, msg_dat_full, msg_vec_full, K->StrListN());

				std::vector<std::string> smry_files{smry->data_file(), smry->vecs_file(), smry->dates_file()};		// check files modification time
				for (const auto &f : smry_files)
					if (HMMPI::FileModCompare(f, templ->DataFileSubst()) < 0)
						throw HMMPI::Exception(HMMPI::stringFormatArr("[{0:%d}] Файл " + f + " был изменен до изменения DATA-файла",
																	  "[{0:%d}] File " + f + " was changed before DATA-file", RNK));	// "small error" - i.e. not EObjFunc
				if (msg_dat_full != "")									// the warnings below are issued after potential error with files modification time
				{
					msg_vect_file += msg_dat_full + "\n";
					msg_vect_stdout += (std::string)HMMPI::MessageRE("ПРЕДУПРЕЖДЕНИЕ: ", "WARNING: ") + msg_dat_short + "\n";
					warning_count++;
				}
				if (msg_vec_full != "")
				{
					msg_vect_file += msg_vec_full + "\n";
					msg_vect_stdout += (std::string)HMMPI::MessageRE("ПРЕДУПРЕЖДЕНИЕ: ", "WARNING: ") + msg_vec_short + "\n";
					warning_count++;
				}

				// 3. Get historical data
				HMMPI::Vector2<double> smry_hist;					// historical data
				bool text_sigma = false;
				if (textsmry->data.Length() != 0)					// option with history from TEXTSMRY
				{
					text_sigma = true;
					smry_hist = textsmry->pet_dat;
				}
				else 												// option with history from model's UNSMRY
				{
					smry_hist = smry->ExtractSummary(datesW->dates, vect->vecs, msg_dat_short, msg_vec_short, msg_dat_full, msg_vec_full, K->StrListN(), "H");
					if (msg_vec_full != "")
					{
						msg_vect_file += msg_vec_full + "\n";
						msg_vect_stdout += (std::string)HMMPI::MessageRE("ПРЕДУПРЕЖДЕНИЕ: ", "WARNING: ") + msg_vec_short + "\n";
						warning_count++;
					}
				}

				// 4. Calculate the objective function
				size_t Nsteps = SMRY.ICount();
				size_t Nvecs = SMRY.JCount();
				assert(Nvecs == vect->sigma.size());

				of_vec = std::vector<double>(Nvecs);
				int count_undef = 0;
				modelled_data = std::vector<double>();
				for (size_t p = 0; p < Nvecs; p++)		// fill modelled_data (and calculate o.f. for NOT-TEXTSMRY case)
				{
					double of1 = 0;						// o.f. contribution of vector "p"
					for (size_t t = 0; t < Nsteps; t++)
					{
						double sigma = 0;
						if (text_sigma)
							sigma = smry_hist(t, p + Nvecs);
						else
							sigma = vect->sigma[p];

						if (!HMMPI::IsNaN(smry_hist(t, p)) && sigma != 0)
						{
							of1 += pow((SMRY(t, p) - smry_hist(t, p))/sigma, 2);
							modelled_data.push_back(SMRY(t, p));
						}
						else
							count_undef++;
					}
					of_vec[p] = of1;
				}
				if (text_sigma)							// of_vec with covariances is redefined here
				{
					VCL->ObjFunc(SMRY, smry_hist, cov_is_diag);
					of_vec = VCL->of1;
				}
				res = HMMPI::Mat(of_vec).Sum();

				// 5. Messaging
				// printing to file
				if (RNK == 0 && !ignore_small_errors)
				{
					sw.open(log_file);
					sw << parameters->msg(-1) << "\n";	// printing all lines to the file

					// limits_msg is not reported, since problems with parameter bounds lead to an exception and immediate termination

					sw << templ_msg << "\n";
					sw << msg_vect_file;
					write_smry(sw, SMRY, smry_hist, of_vec, text_sigma);

					double prior = 0;
					std::string prior_msg = form_prior(prior);
					sw << HMMPI::stringFormatArr("\nf = {0:%.8g}\n", std::vector<double>{res + prior});
					sw << prior_msg;
					if (prior_msg != "")
						sw << HMMPI::stringFormatArr("Likelihood = {0:%.8g}\n", std::vector<double>{res});
					if (text_sigma)
					{
						if (!cov_is_diag)
							sw << (std::string)HMMPI::stringFormatArr("Была использована полная ковариационная матрица (при маленьких R в {0:%s} она становится диагональной)\n",
																	  "Full covariance matrix was used (it becomes diagonal for small R in {0:%s})\n", vect->name);
						else
							sw << (std::string)HMMPI::MessageRE("Была использована диагональная ковариационная матрица\n",
																"Diagonal covariance matrix was used\n");
					}
					sw << HMMPI::stringFormatArr(HMMPI::MessageRE("Использовано точек данных: {0:%d}\nНеиспользованных значений / нулевых сигм: {1:%d}\nВсего точек данных: {2:%d}",
																  "Used data points: {0:%d}\nUnused values / zero sigmas: {1:%d}\nTotal data points: {2:%d}"),
																  std::vector<int>{int(Nvecs*Nsteps) - count_undef, count_undef, int(Nvecs*Nsteps)});
					sw.close();
				}
				obj_func_msg = templ_msg + "\n" + msg_vect_stdout;

				// 6. Clear files
				templ->ClearFiles();
				templ->ClearFilesEcl();

				complete = 1;
			}
			catch (const HMMPI::EObjFunc &e)	// immediate termination
			{
				if (sw.is_open())
					sw.close();

				sprintf(err_msg, "%.*s", HMMPI::BUFFSIZE-50, e.what());
				complete = 2;
			}
			catch (const std::exception &e)
			{
				if (sw.is_open())
					sw.close();

				if (!ignore_small_errors)
				{
					sprintf(err_msg, "%.*s", HMMPI::BUFFSIZE-50, e.what());
					complete = 2;
				}
				else
					K->AppText(HMMPI::stringFormatArr("*** ошибка: {0:%s}, симулятор будет перезапущен\n",
													  "*** error: {0:%s}, simulator will be re-run\n", (std::string)e.what()));
			}

			HMMPI::MPI_BarrierSleepy(comm);
			MPI_Bcast(&complete, 1, MPI_INT, 0, comm);

		} // while (complete == 0)
	}
	else	  	  // comm_rank != 0
	{
		while (complete == 0)
		{
			try	  // try-catch block should be sync across all ranks
			{
				HMMPI::MPI_BarrierSleepy(comm);
				simcmd->RunCmd(comm);
			}
			catch (...)
			{

			}

			HMMPI::MPI_BarrierSleepy(comm);
			MPI_Bcast(&complete, 1, MPI_INT, 0, comm);
		}
	}

	if (comm != MPI_COMM_NULL)
	{
		HMMPI::MPI_BarrierSleepy(comm);

		MPI_Bcast(&warning_count, 1, MPI_INT, 0, comm);
		if (!ignore_small_errors)
			K->TotalWarnings += warning_count;

		MPI_Bcast(err_msg, HMMPI::BUFFSIZE, MPI_CHAR, 0, comm);		// sync the error
		if (err_msg[0] != 0)
			throw HMMPI::Exception(err_msg);
	}

	return res;
}
//---------------------------------------------------------------------------
size_t PMEclipse::ModelledDataSize() const
{
	if (modelled_data_size == 0)
		modelled_data_size = Data().size();

	return modelled_data_size;
}
//---------------------------------------------------------------------------
void PMEclipse::PerturbData()
{
	DECLKWD(textsmry, KW_textsmry, "TEXTSMRY");
	DECLKWD(RML, KW_RML, "RML");

	if (RML->on == "OFF")
		throw HMMPI::Exception("Call to PMEclipse::PerturbData for RML=OFF");
	if (textsmry->data.Length() == 0)
		throw HMMPI::Exception("For SIMECL model, RML is only possible for history from TEXTSMRY");

	perturb_well();		// VCL perturbation

	int size = textsmry->pet_dat.ICount()*textsmry->pet_dat.JCount();
	std::vector<double> dat(textsmry->pet_dat.Serialize(), textsmry->pet_dat.Serialize() + size);
	if (K->MPI_rank == 0)											// save perturbed data
	{
		std::ofstream sw(CWD + "/TextSMRY_RML.txt", std::ios::out);
		write_smry(sw, HMMPI::Vector2<double>(), textsmry->pet_dat, std::vector<double>(), true, true);		// write only history
		sw.close();
	}

	HMMPI::Bcast_vector(dat, 0, MPI_COMM_WORLD);					// MPI synchronization
	textsmry->pet_dat.Deserialize(dat.data());
}
//---------------------------------------------------------------------------
std::vector<HMMPI::Mat> PMEclipse::CorrBlocks() const
{
	if (VCL == 0)
		throw HMMPI::Exception("VCL == 0 in PMEclipse::CorrBlocks");

	std::vector<HMMPI::Mat> res(VCL->vec_vce().size());
	cov_is_diag = true;
	size_t totsz = 0;			// total size of entire matrix
	for (size_t i = 0; i < res.size(); i++)
	{
		size_t sz = VCL->vec_vce()[i].sz;
		totsz += sz;
		if (VCL->vec_vce()[i].R0 > VCL->vec_vce()[i].R_threshold)
		{
			res[i] = HMMPI::Mat(std::vector<double>(VCL->vec_vce()[i].C, VCL->vec_vce()[i].C + sz*sz), sz, sz);		// full matrix
			cov_is_diag = false;
		}
		else
			res[i] = HMMPI::Mat(sz, 1, 1.0);	// only diagonal
	}

#ifdef DIAGCOV_1X1
	if (cov_is_diag)
		res = std::vector<HMMPI::Mat>(totsz, HMMPI::Mat(1, 1, 1.0));	// elementary 1x1 blocks
#endif

	return res;
}
//---------------------------------------------------------------------------
std::vector<double> PMEclipse::Std() const
{
	if (VCL == 0)
		throw HMMPI::Exception("VCL == 0 in PMEclipse::Std");

	std::vector<double> res;
	for (size_t i = 0; i < VCL->vec_vce().size(); i++)
	{
		res.reserve(res.size() + VCL->vec_vce()[i].sz);
		res.insert(res.end(), VCL->vec_vce()[i].sigma.begin(), VCL->vec_vce()[i].sigma.end());
	}

	return res;
}
//---------------------------------------------------------------------------
std::vector<double> PMEclipse::Data() const
{
	DECLKWD(textsmry, KW_textsmry, "TEXTSMRY");
	if (textsmry->data.Length() == 0)
		throw HMMPI::Exception("TEXTSMRY not defined in PMEclipse::Data");

	HMMPI::Vector2<double> smry_hist = textsmry->pet_dat;
	size_t Nsteps = smry_hist.ICount();
	size_t Npars = smry_hist.JCount() / 2;

	std::vector<double> res;
	res.reserve(Npars*Nsteps);		// some extra space may be reserved
	for (size_t p = 0; p < Npars; p++)
		for (size_t t = 0; t < Nsteps; t++)
		{
			double sigma = smry_hist(t, p + Npars);
			if (!HMMPI::IsNaN(smry_hist(t, p)) && sigma != 0)
				res.push_back(smry_hist(t, p));
		}

	return res;
}
//---------------------------------------------------------------------------
// PMpConnect
//---------------------------------------------------------------------------
#define TEST_CACHES
void PMpConnect::run_simulation(const std::vector<double> &params)
{
	DECLKWD(parameters, KW_parameters2, "PARAMETERS2");
	DECLKWD(templ, KW_templates, "TEMPLATES");
	DECLKWD(simcmd, KW_simcmd, "SIMCMD");

	const std::vector<std::string> Fnames_templ = {"ObjFunc_%d.bin", "Data_mod_%d.bin", "Data_hist_%d.bin", "Data_sigmas_%d.bin", "ObjFuncGrad_%d.bin"};
	std::vector<std::string> Fnames(Fnames_templ.size());
	for (size_t i = 0; i < Fnames.size(); i++)
	{
		char buff[HMMPI::BUFFSIZE];
		sprintf(buff, Fnames_templ[i].c_str(), RNK);
		Fnames[i] = buff;
	}

	int parallel_size = 0;		// parallel simulation size
	int parallel_rank = -1;
	if (comm != MPI_COMM_NULL)
	{
		MPI_Comm_size(comm, &parallel_size);
		MPI_Comm_rank(comm, &parallel_rank);
	}

	char err_msg[HMMPI::BUFFSIZE];
	err_msg[0] = 0;
	if (parallel_rank == 0)		// processing is done on comm-RANKS-0, simulation - on all ranks
	{
#ifdef TEST_CACHES
		std::cout << "		[" << RNK << "] * RUN SIMULATION *\n";
#endif

		std::vector<bool> file_ok(5, true);
		std::vector<FILE*> Files(5, NULL);							// if Files[i] is not open, Files[i] = NULL

		try						// this try-catch block is intended for MPI synchronisation of errors
		{
			if (!CheckLimits(params))
				throw HMMPI::EObjFunc((std::string)HMMPI::MessageRE("Параметры выходят за допустимый диапазон: ",
													   	   	   	    "Parameters are out of range: ") + HMMPI::ToString(params));

			// 1. Substitute the parameters and run the model
			HMMPI::TagPrintfMap *tmap = parameters->get_tag_map();		// object handling the parameters - fill it!
			tmap->SetSize(parallel_size);
			tmap->SetSmpl(smpl_tag);
			const std::vector<double> par_external = parameters->InternalToExternal(params);
			tmap->SetDoubles(parameters->name, par_external);
			obj_func_msg = templ->WriteFiles(*tmap);					// MOD and PATH for "tmap" are set here, simcmd->cmd_work is also filled here
			obj_func_msg += "\n";

			HMMPI::MPI_BarrierSleepy(comm);
			simcmd->RunCmd(comm);

			// 2. Get objective_function, modelled_data, historical_data, sigmas, gradient
			for (size_t i = 0; i < Files.size(); i++)
			{
				Files[i] = fopen(Fnames[i].c_str(), "rb");
				if (Files[i] == NULL)																// check if file-i exists
					file_ok[i] = false;
				if (file_ok[i] && HMMPI::FileModCompare(Fnames[i], templ->DataFileSubst()) < 0)		// check if file-i was updated
					file_ok[i] = false;
				if (i < 4 && !file_ok[i])															// check that all files (except gradients) are ok
					throw HMMPI::Exception(HMMPI::stringFormatArr("[{0:%d}] Файл " + Fnames[i] + " отсутствует, либо был изменен до изменения дата-файла",
																  "[{0:%d}] File " + Fnames[i] + " is missing, or was changed before data file", RNK));
			}

			if (file_ok[0])
			{
				assert(file_ok[1] && file_ok[2] && file_ok[3]);

				fread(&of_cache, sizeof(double), 1, Files[0]);
				par_of_cache = params;

				std::vector<double> data_work = fread_vector(Files[1]);
				std::vector<double> hist_work = fread_vector(Files[2]);
				std::vector<double> sigma_work = fread_vector(Files[3]);
				if (data_work.size() != hist_work.size() || data_work.size() != sigma_work.size())
					throw HMMPI::Exception("Inconsistent sizes of data_work, hist_work, sigma_work in PMpConnect::run_simulation");

				data_cache.clear();
				hist_cache.clear();
				sigma_cache.clear();

				data_cache.reserve(data_work.size());
				hist_cache.reserve(data_work.size());
				sigma_cache.reserve(data_work.size());

				for (size_t i = 0; i < data_work.size(); i++)			// only data points with non-zero sigmas are taken
					if (sigma_work[i] != 0)
					{
						data_cache.push_back(data_work[i]);
						hist_cache.push_back(hist_work[i]);
						sigma_cache.push_back(sigma_work[i]);
					}

#ifdef TEST_CACHES
				if (!hist_sigmas_ok)
					std::cout << "== Full data vector size: " << data_work.size() << ", used data vector size: " << data_cache.size() << " ==\n";
#endif
				hist_sigmas_ok = true;
			}

			if (file_ok[4])
			{
				grad_cache = fread_vector(Files[4]);
				if (grad_cache.size() != params.size())
					throw HMMPI::Exception(HMMPI::stringFormatArr(HMMPI::MessageRE("Длина вектора градиентов из файла ({0:%zu}) не совпадает с размерностью параметров ({1:%zu}) в PMpConnect::run_simulation",
																				   "Gradient vector size from the file ({0:%zu}) does not match the parameters dimension ({1:%zu}) in PMpConnect::run_simulation"),
																					std::vector<size_t>{grad_cache.size(), params.size()}));
				// transform from d/dxe -> d/dxi
				grad_cache = parameters->dxe_To_dxi(grad_cache, params);
				par_grad_cache = params;
			}

			for (size_t i = 0; i < Files.size(); i++)
				fclose(Files[i]);
		}
		catch (const std::exception &e)
		{
			for (size_t i = 0; i < Files.size(); i++)
				fclose(Files[i]);

			sprintf(err_msg, "%.*s", HMMPI::BUFFSIZE-50, e.what());
		}
	}
	else	// parallel_rank != 0
	{
		try		// try-catch block is sync across all ranks
		{
			HMMPI::MPI_BarrierSleepy(comm);
			simcmd->RunCmd(comm);
		}
		catch (...)
		{

		}
	}

	// sync everything from comm-rank-0
	MPI_Bcast(&of_cache, 1, MPI_DOUBLE, 0, comm);
	HMMPI::Bcast_vector(par_of_cache, 0, comm);

	HMMPI::Bcast_vector(data_cache, 0, comm);
	HMMPI::Bcast_vector(hist_cache, 0, comm);
	HMMPI::Bcast_vector(sigma_cache, 0, comm);
	MPI_Bcast(&hist_sigmas_ok, 1, MPI_BYTE, 0, comm);

	HMMPI::Bcast_vector(grad_cache, 0, comm);
	HMMPI::Bcast_vector(par_grad_cache, 0, comm);

	if (comm != MPI_COMM_NULL)
	{
		MPI_Bcast(err_msg, HMMPI::BUFFSIZE, MPI_CHAR, 0, comm);		// sync the error
		if (err_msg[0] != 0)
			throw HMMPI::Exception(err_msg);
	}
}
//---------------------------------------------------------------------------
std::vector<double> PMpConnect::fread_vector(FILE *file)
{
	int len;
	fread(&len, sizeof(int), 1, file);

	if (len < 0 || len > 1000000)
		throw HMMPI::Exception("Inappropriate array length read from file in PMpConnect::fread_vector");

	std::vector<double> res(len);
	fread(res.data(), sizeof(double), len, file);

	return res;
}
//---------------------------------------------------------------------------
PMpConnect::PMpConnect(Parser_1 *k, KW_item *kw, std::string cwd, MPI_Comm c) : PhysModel(k, kw, c), of_cache(-1.0), hist_sigmas_ok(false), smpl_tag(-1), K(k), CWD(cwd)			// all data are taken from keywords of "k"; "kw" is used only to handle prerequisites
{
	DECLKWD(parameters, KW_parameters2, "PARAMETERS2");
	DECLKWD(pcon, KW_pConnect_config, "PCONNECT_CONFIG");

	kw->Start_pre();
	kw->Add_pre("PARAMETERS2");
	kw->Add_pre("TEMPLATES");
	kw->Add_pre("SIMCMD");
	kw->Add_pre("PCONNECT_CONFIG");
	kw->Finish_pre();

	con = parameters;
	name = "PMpConnect";
	scale = pcon->scale;

#ifdef TEST_CACHES
	std::cout << "[" << RNK << "] * Recalculate PMpConnect hist_data and sigma cache *\n";
#endif
	run_simulation(parameters->init);		// simulation with a dummy parameters vector, to get hist_data and sigmas

#ifdef TESTCTOR
	std::ofstream testf(HMMPI::stringFormatArr("TESTCTOR_out_{0:%d}.txt", std::vector<int>{RNK}), std::ios::app);
	testf << "rank " << RNK << ", PMpConnect easy CTOR, this = " << this << "\n";
	testf.close();
#endif
}
//---------------------------------------------------------------------------
PMpConnect::PMpConnect(const PMpConnect &PM) : PhysModel(PM), of_cache(PM.of_cache), data_cache(PM.data_cache), par_of_cache(PM.par_of_cache),
		grad_cache(PM.grad_cache), par_grad_cache(PM.par_grad_cache),
		hist_cache(PM.hist_cache), sigma_cache(PM.sigma_cache), hist_sigmas_ok(PM.hist_sigmas_ok), smpl_tag(PM.smpl_tag),
		K(PM.K), CWD(PM.CWD), obj_func_msg(PM.obj_func_msg), scale(PM.scale)
{
}
//---------------------------------------------------------------------------
PMpConnect::~PMpConnect()
{
#ifdef TESTCTOR
	std::ofstream testf(HMMPI::stringFormatArr("TESTCTOR_out_{0:%d}.txt", std::vector<int>{RNK}), std::ios::app);
	testf << "rank " << RNK << ", PMpConnect -- DTOR --, this = " << this << "\n";
	testf.close();
#endif
}
//---------------------------------------------------------------------------
int PMpConnect::ParamsDim() const noexcept
{
	DECLKWD(parameters, KW_parameters2, "PARAMETERS2");
	return parameters->init.size();
}
//---------------------------------------------------------------------------
size_t PMpConnect::ModelledDataSize() const
{
	if (!hist_sigmas_ok)
		throw HMMPI::Exception("hist_sigmas_ok == false in PMpConnect::ModelledDataSize");

	return hist_cache.size();
}
//---------------------------------------------------------------------------
double PMpConnect::ObjFunc(const std::vector<double> &params)					// calculates objective function [and gradient] by running the simulation model;
{																				// modelled_data is also filled; simulation is only done on comm-RANKS-0
	if (params != par_of_cache || smpl_tag != -1)		// Note: non-trivial (!= -1) smpl_tag also causes running the simulation
	{
#ifdef TEST_CACHES
		std::cout << "[" << RNK << "] * Recalculate PMpConnect obj. func. cache *\n";
#endif
		run_simulation(params);
	}
	else
	{
#ifdef TEST_CACHES
		std::cout << "[" << RNK << "] +++ REUSE PMpConnect obj. func. cache +++\n";
#endif
	}

	if (params != par_of_cache)
		throw HMMPI::Exception("Невозможно рассчитать целевую функцию", "Objective function cannot be calculated");

	modelled_data = data_cache;

	return of_cache*scale;
}
//---------------------------------------------------------------------------
std::vector<double> PMpConnect::ObjFuncGrad(const std::vector<double> &params)	// gradient of objective function; internally, run_simulation() is called
{
	if (params != par_grad_cache)
	{
#ifdef TEST_CACHES
		std::cout << "[" << RNK << "] * Recalculate PMpConnect Gradient cache *\n";
#endif
		run_simulation(params);
	}
	else
	{
#ifdef TEST_CACHES
		std::cout << "[" << RNK << "] +++ REUSE PMpConnect Gradient cache +++\n";
#endif
	}

	if (params != par_grad_cache)
		throw HMMPI::Exception("Невозможно рассчитать градиент целевой функции", "Objective function gradient cannot be calculated");

	std::vector<double> res = grad_cache;
	for (auto &x : res)
		x *= scale;

	return res;
}
//---------------------------------------------------------------------------
std::vector<HMMPI::Mat> PMpConnect::CorrBlocks() const		// 1 x 1 blocks
{
	return std::vector<HMMPI::Mat>(ModelledDataSize(), HMMPI::Mat(1, 1, 1.0));
}
//---------------------------------------------------------------------------
std::vector<double> PMpConnect::Std() const					// sigmas (only non-zero sigmas are used)
{
	if (!hist_sigmas_ok)
		throw HMMPI::Exception("hist_sigmas_ok == false in PMpConnect::Std");

	return sigma_cache;
}
//---------------------------------------------------------------------------
std::vector<double> PMpConnect::Data() const				// historical data
{
	if (!hist_sigmas_ok)
		throw HMMPI::Exception("hist_sigmas_ok == false in PMpConnect::Data");

	return hist_cache;
}
//---------------------------------------------------------------------------
// PMConc
//---------------------------------------------------------------------------
void PMConc::run_simulation(const std::vector<double> &params, std::vector<double> &out_t, std::vector<double> &out_conc)			// runs simulation, filling the output out_t, out_conc (which are sync)
{
	DECLKWD(parameters, KW_parameters, "PARAMETERS");
	DECLKWD(templ, KW_templates, "TEMPLATES");
	DECLKWD(simcmd, KW_simcmd, "SIMCMD");

	char buff[HMMPI::BUFFSIZE];
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	sprintf(buff, "conc_%d.out", rank);
	const std::string fname = buff;				// currently the simulation output should be in this file

	int parallel_size = 0;						// parallel simulation size
	int parallel_rank = -1;
	if (comm != MPI_COMM_NULL)
	{
		MPI_Comm_size(comm, &parallel_size);
		MPI_Comm_rank(comm, &parallel_rank);
	}

	char err_msg[HMMPI::BUFFSIZE];
	err_msg[0] = 0;
	if (parallel_rank == 0)		// processing is done on comm-RANKS-0, simulation - on all ranks
	{
		FILE *file = 0;
		try						// this try-catch block is intended for MPI synchronisation of errors
		{
			if (!CheckLimits(params))
				throw HMMPI::EObjFunc(HMMPI::MessageRE("Параметры выходят за допустимый диапазон",
													   "Parameters are out of range"));

			// 1. Substitute the parameters and run the model
			HMMPI::TagPrintfMap *tmap = parameters->get_tag_map();		// object handling the parameters - fill it!
			tmap->SetSize(parallel_size);
			const std::vector<double> par_external = parameters->InternalToExternal(params);
			tmap->SetDoubles(parameters->name, par_external);
			obj_func_msg = templ->WriteFiles(*tmap);					// MOD and PATH for "tmap" are set here, simcmd->cmd_work is also filled here
			obj_func_msg += "\n";

			HMMPI::MPI_BarrierSleepy(comm);
			simcmd->RunCmd(comm);

			// 2. Get objective_function, modelled_data, historical_data, sigmas, gradient
			bool file_ok = true;
			file = fopen(fname.c_str(), "r");
			if (file == NULL)																// check if file exists
				file_ok = false;
			fclose(file);
			file = 0;

			if (file_ok && HMMPI::FileModCompare(fname, templ->DataFileSubst()) < 0)		// check if file was updated
				file_ok = false;
			if (!file_ok)
				throw HMMPI::Exception(HMMPI::stringFormatArr("[{0:%d}] Файл " + fname + " отсутствует, либо был изменен до изменения дата-файла",
															  "[{0:%d}] File " + fname + " is missing, or was changed before data file", RNK));

			const std::vector<std::vector<double>> work = KW_Dtable::ReadTableFromFile(fname);
			if (work.size() == 0)
				throw HMMPI::Exception("Concentration file '" + fname + "' has no data");
			if (work[0].size() != 2)
				throw HMMPI::Exception("Concentration file '" + fname + "' should have 2 columns");

			out_t = out_conc = std::vector<double>(work.size());
			for (size_t i = 0; i < work.size(); i++)
			{
				out_t[i] = work[i][0];
				out_conc[i] = work[i][1];
			}

			if (out_t != tt)
				throw HMMPI::Exception("Time from concentration file '" + fname + "' should be the same as in CONC_DATA file");
		}
		catch (const std::exception &e)
		{
			fclose(file);
			sprintf(err_msg, "%.*s", HMMPI::BUFFSIZE-50, e.what());
		}
	}
	else	// parallel_rank != 0
	{
		try
		{
			HMMPI::MPI_BarrierSleepy(comm);
			simcmd->RunCmd(comm);
		}
		catch (...)
		{

		}
	}

	// sync everything from comm-rank-0
	HMMPI::Bcast_vector(out_t, 0, comm);
	HMMPI::Bcast_vector(out_conc, 0, comm);

	if (comm != MPI_COMM_NULL)
	{
		MPI_Bcast(err_msg, HMMPI::BUFFSIZE, MPI_CHAR, 0, comm);		// sync the error
		if (err_msg[0] != 0)
			throw HMMPI::Exception(err_msg);
	}
}
//---------------------------------------------------------------------------
PMConc::PMConc(Parser_1 *k, KW_item *kw, std::string cwd, MPI_Comm c) : PhysModel(k, kw, c), K(k), CWD(cwd)			// all data are taken from keywords of "k"; "kw" is used only to handle prerequisites
{
	DECLKWD(parameters, KW_parameters, "PARAMETERS");
	DECLKWD(conc, KW_conc_data, "CONC_DATA");

	kw->Start_pre();
	kw->Add_pre("PARAMETERS");
	kw->Add_pre("TEMPLATES");
	kw->Add_pre("SIMCMD");
	kw->Add_pre("CONC_DATA");
	kw->Finish_pre();

	con = parameters;
	name = "PMConc";

	// fill the observed data, sigmas, and indices for nonzero sigmas
	tt.resize(conc->data.size());
	c_hist.resize(conc->data.size());
	sigma.resize(conc->data.size());

	for (size_t i = 0; i < tt.size(); i++)
	{
		assert(conc->data[i].size() == 3);
		tt[i] = conc->data[i][0];
		c_hist[i] = conc->data[i][1];
		sigma[i] = conc->data[i][2];

		if (sigma[i] != 0)
			nonzero_sigma_ind.push_back(i);
	}
}
//---------------------------------------------------------------------------
int PMConc::ParamsDim() const noexcept
{
	DECLKWD(parameters, KW_parameters, "PARAMETERS");
	return parameters->init.size();
}
//---------------------------------------------------------------------------
double PMConc::ObjFunc(const std::vector<double> &params)		// calculates objective function by running the simulation model;
{																// modelled_data is also filled; simulation is only done on comm-RANKS-0
	std::vector<double> out_t, out_conc;
	run_simulation(params, out_t, out_conc);							// output is sync

	modelled_data = HMMPI::Reorder(out_conc, nonzero_sigma_ind);		// actually, modelled_data is sync here
	const std::vector<double> hist_data = Data();
	const std::vector<double> sgm = Std();
	assert(modelled_data.size() == hist_data.size() && modelled_data.size() == sgm.size());

	double of = 0;
	for (size_t i = 0; i < modelled_data.size(); i++)
		of += pow((modelled_data[i] - hist_data[i])/sgm[i], 2);

	return of;
}
//---------------------------------------------------------------------------
std::vector<HMMPI::Mat> PMConc::CorrBlocks() const		// 1 x 1 blocks
{
	return std::vector<HMMPI::Mat>(ModelledDataSize(), HMMPI::Mat(1));
}
//---------------------------------------------------------------------------
std::vector<double> PMConc::Std() const					// sigmas (only non-zero sigmas are used)
{
	return HMMPI::Reorder(sigma, nonzero_sigma_ind);
}
//---------------------------------------------------------------------------
std::vector<double> PMConc::Data() const				// historical data
{
	return HMMPI::Reorder(c_hist, nonzero_sigma_ind);
}
//---------------------------------------------------------------------------
// PM_Rosenbrock
//---------------------------------------------------------------------------
PM_Rosenbrock::PM_Rosenbrock(Parser_1 *K, KW_item *kw, MPI_Comm c) : PhysModel(K, kw, c)
{
	name = "PM_Rosenbrock";
	dim = init.size();

	kw->Start_pre();
	kw->Add_pre("LIMITS");
	kw->Finish_pre();

#ifdef TESTCTOR
	std::ofstream testf(HMMPI::stringFormatArr("TESTCTOR_out_{0:%d}.txt", std::vector<int>{RNK}), std::ios::app);
	testf << "rank " << RNK << ", PM_Rosenbrock easy CTOR, this = " << this << "\n";
	testf.close();
#endif
}
//---------------------------------------------------------------------------
PM_Rosenbrock::~PM_Rosenbrock()
{
#ifdef TESTCTOR
	std::ofstream testf(HMMPI::stringFormatArr("TESTCTOR_out_{0:%d}.txt", std::vector<int>{RNK}), std::ios::app);
	testf << "rank " << RNK << ", PM_Rosenbrock -- DTOR --, this = " << this << "\n";
	testf.close();
#endif
}
//---------------------------------------------------------------------------
double PM_Rosenbrock::ObjFunc(const std::vector<double> &params)
{
	if ((int)params.size() != dim)
		throw HMMPI::Exception("Неправильная размерность в PM_Rosenbrock::ObjFunc", "Wrong dimension in PM_Rosenbrock::ObjFunc");

	double res = 0;
	modelled_data.resize(2*dim - 2);
	for (int i = 0; i < dim-1; i++)
	{
		double d1 = params[i+1] - params[i]*params[i];
		double d2 = 1 - params[i];
		modelled_data[i] = d1;
		modelled_data[i+dim-1] = params[i];
		res += 100*d1*d1 + d2*d2;
	}

	return res;
}
//---------------------------------------------------------------------------
std::vector<double> PM_Rosenbrock::ObjFuncGrad(const std::vector<double> &params)
{
	if ((int)params.size() != dim)
		throw HMMPI::Exception("Неправильная размерность в PM_Rosenbrock::ObjFuncGrad", "Wrong dimension in PM_Rosenbrock::ObjFuncGrad");

	std::vector<double> res(dim);
	for (int i = 0; i < dim; i++)
	{
		res[i] = 0;
		if (i < dim-1)
			res[i] += (-400*(params[i+1] - params[i]*params[i])*params[i]) + 2*(params[i] - 1);
		if (i > 0)
			res[i] += 200*(params[i] - params[i-1]*params[i-1]);
	}

	DataSens = HMMPI::Mat(2*dim-2, dim, 0);
	for (int i = 0; i < dim-1; i++)
	{
		DataSens(i, i) = -2*params[i];
		DataSens(i, i+1) = 1;
		DataSens(i+dim-1, i) = 1;
	}

	return res;
}
//---------------------------------------------------------------------------
HMMPI::Mat PM_Rosenbrock::ObjFuncHess(const std::vector<double> &params)
{
	if ((int)params.size() != dim)
		throw HMMPI::Exception("Неправильная размерность в PM_Rosenbrock::ObjFuncHess", "Wrong dimension in PM_Rosenbrock::ObjFuncHess");

	HMMPI::Mat res(dim, dim, 0.0);
	for (int i = 0; i < dim; i++)
	{
		if (i < dim-1)
		{
			res(i, i+1) = -400*params[i];
			res(i, i) += -400*(params[i+1] - 3*params[i]*params[i]) + 2;
		}
		if (i > 0)
		{
			res(i, i-1) = -400*params[i-1];
			res(i, i) += 200;
		}
	}

	return res;
}
//---------------------------------------------------------------------------
HMMPI::Mat PM_Rosenbrock::ObjFuncFisher(const std::vector<double> &params)
{
	std::vector<double> cov = Std();
	for (auto &x : cov)
		x = 1/x;

	ObjFuncGrad(params);						// calculate DataSens
	HMMPI::Mat res = cov % DataSens;
	return res.Tr() * res;
}
//---------------------------------------------------------------------------
HMMPI::Mat PM_Rosenbrock::ObjFuncFisher_dxi(const std::vector<double> &params, const int i, int r)
{
	HMMPI::Mat res(dim, dim, 0);
	if (i < dim-1)
	{
		res(i, i) = 4*params[i];
		res(i+1, i) = -2;
	}

	return 100*(res + res.Tr());
}
//---------------------------------------------------------------------------
std::vector<HMMPI::Mat> PM_Rosenbrock::CorrBlocks() const				// a single block with diagonal matrix stored as {N x 1} array, where N = 2*(dim-1)
{
	return std::vector<HMMPI::Mat>{HMMPI::Mat(2*(dim-1), 1, 1.0)};
}
//---------------------------------------------------------------------------
std::vector<double> PM_Rosenbrock::Std() const
{
	std::vector<double> res(2*(dim-1));
	for (int i = 0; i < dim-1; i++)
	{
		res[i] = 0.1;
		res[i+dim-1] = 1;
	}

	return res;
}
//---------------------------------------------------------------------------
std::vector<double> PM_Rosenbrock::Data() const
{
	std::vector<double> res(2*(dim-1));
	for (int i = 0; i < dim-1; i++)
	{
		res[i] = 0;
		res[i+dim-1] = 1;
	}

	return res;
}
//---------------------------------------------------------------------------
// PM_Linear
//---------------------------------------------------------------------------
std::vector<double> PM_Linear::cov_diag()
{
	std::vector<double> res;
	if (DiagCov.size() > 0)
		res = DiagCov;
	else
	{
		assert(FullCov.ICount() == FullCov.JCount());
		res = std::vector<double>(FullCov.ICount());
		for (size_t i = 0; i < res.size(); i++)
			res[i] = FullCov(i, i);
	}

	return res;
}
//---------------------------------------------------------------------------
PM_Linear::PM_Linear(Parser_1 *K, KW_item *kw, MPI_Comm c) : PhysModel(K, kw, c), holding_chol(false)
{
	DECLKWD(corr, KW_corrstruct, "CORRSTRUCT");
	DECLKWD(mat, KW_matvecvec, "MATVECVEC");
	DECLKWD(textsmry, KW_textsmry, "TEXTSMRY");

	name = "PM_Linear";

	kw->Start_pre();
	kw->Add_pre("CORRSTRUCT");
	kw->Add_pre("MATVECVEC");
	kw->Add_pre("LIMITS");
	kw->Finish_pre();

	bool R_small = true;		// check if all correlation radii are small
	for (auto r : corr->R)
		if (r > 0.01)
		{
			R_small = false;
			break;
		}

	RndN = &textsmry->randn;
	G = mat->M;
	if (G.ICount() != (size_t)corr->size())
		throw HMMPI::Exception("Не совпадают размерности матрицы G и CORRSTRUCT в модели LINEAR",
							   "Inconsistent dimensions of matrix G and CORRSTRUCT in LINEAR model");
	if (G.JCount() != init.size())
		throw HMMPI::Exception("Не совпадают размерности матрицы G и LIMITS в модели LINEAR",
							   "Inconsistent dimensions of matrix G and LIMITS in LINEAR model");
	d0_orig = mat->v1;
	d0 = d0_orig;
	if (R_small)
		DiagCov = mat->v2;
	else
	{
		std::vector<double> big_diag = mat->v2;
		std::transform(big_diag.begin(), big_diag.end(), big_diag.begin(), HMMPI::_sqrt);
		FullCov = big_diag % corr->Corr() % big_diag;
	}

#ifdef TESTCTOR
	std::ofstream testf(HMMPI::stringFormatArr("TESTCTOR_out_{0:%d}.txt", std::vector<int>{RNK}), std::ios::app);
	testf << "rank " << RNK << ", PM_Linear easy CTOR, FullCovariance = " << (R_small ? "FALSE" : "TRUE") << ", this = " << this << "\n";
	testf.close();

	if (!R_small && RNK == 0)
	{
		FILE *f0 = fopen("LINEAR_FullCovariance.txt", "w");
		FullCov.SaveASCII(f0, "%20.16g");
		fclose(f0);
	}
#endif
}
//---------------------------------------------------------------------------
PM_Linear::~PM_Linear()
{
#ifdef TESTCTOR
	std::ofstream testf(HMMPI::stringFormatArr("TESTCTOR_out_{0:%d}.txt", std::vector<int>{RNK}), std::ios::app);
	testf << "rank " << RNK << ", PM_Linear -- DTOR --, this = " << this << "\n";
	testf.close();
#endif
}
//---------------------------------------------------------------------------
double PM_Linear::ObjFunc(const std::vector<double> &params)
{
	HMMPI::Mat Gx = G * params;
	modelled_data = Gx.ToVector();

	HMMPI::Mat Gxd0 = Gx - d0;
	size_t sz = Gxd0.ICount();
	if (DiagCov.size() != 0 && sz != DiagCov.size())
		throw HMMPI::Exception("Неправильный размер DiagCov в PM_Linear::ObjFunc", "Wrong size of DiagCov in PM_Linear::ObjFunc");

	double res = 0;
	if (DiagCov.size() != 0)	// diagonal covariance
		for (size_t i = 0; i < sz; i++)
			res += Gxd0(i, 0)*Gxd0(i, 0)/DiagCov[i];
	else						// dense covariance
		res = InnerProd(FullCov / Gxd0, Gxd0);

	return res;
}
//---------------------------------------------------------------------------
std::vector<double> PM_Linear::ObjFuncGrad(const std::vector<double> &params)
{
	HMMPI::Mat Gxd0 = HMMPI::Mat(G * params) - d0;
	size_t sz = Gxd0.ICount();
	if (DiagCov.size() != 0 && sz != DiagCov.size())
		throw HMMPI::Exception("Неправильный размер DiagCov в PM_Linear::ObjFuncGrad", "Wrong size of DiagCov in PM_Linear::ObjFuncGrad");

	if (DiagCov.size() != 0)	// diagonal covariance
		for (size_t i = 0; i < sz; i++)
			Gxd0(i, 0) *= 2/DiagCov[i];
	else						// dense covariance
		Gxd0 = 2*(FullCov / std::move(Gxd0));

	//std::vector<int> data_ind(ModelledDataSize());		// [0, 1, 2,...] - take all indices for data
	//std::iota(data_ind.begin(), data_ind.end(), 0);
	//DataSens = G.Reorder(data_ind, act_ind);
	DataSens = G;		// full-dim sensitivity!

	return (G.Tr()*Gxd0).ToVector();
}
//---------------------------------------------------------------------------
HMMPI::Mat PM_Linear::ObjFuncHess(const std::vector<double> &params)
{
	size_t Ni = G.ICount();
	size_t Nj = G.JCount();
	if (DiagCov.size() != 0 && Ni != DiagCov.size())
		throw HMMPI::Exception("Неправильный размер DiagCov в PM_Linear::ObjFuncHess", "Wrong size of DiagCov in PM_Linear::ObjFuncHess");

	HMMPI::Mat CG = G;
	if (DiagCov.size() != 0)	// diagonal covariance
		for (size_t i = 0; i < Ni; i++)
			for (size_t j = 0; j < Nj; j++)
				CG(i, j) *= 2/DiagCov[i];
	else						// dense covariance
		CG = 2*(FullCov / std::move(CG));

	return G.Tr() * CG;
}
//---------------------------------------------------------------------------
HMMPI::Mat PM_Linear::ObjFuncFisher(const std::vector<double> &params)
{
	// full DataSens = G

	HMMPI::Mat CinvSens;				// Cov^(-1) * DataSens
	if (DiagCov.size() != 0)			// diagonal covariance
	{
		std::vector<double> Cinv(DiagCov.size());
		std::transform(DiagCov.begin(), DiagCov.end(), Cinv.begin(), [](double x){return 1/x;});	// invert the diagonal
		CinvSens = Cinv % G;			// multiply by diagonal from left
	}
	else								// dense covariance
		CinvSens = FullCov / G;

	return G.Tr() * CinvSens;
}
//---------------------------------------------------------------------------
HMMPI::Mat PM_Linear::ObjFuncFisher_dxi(const std::vector<double> &params, const int i, int r)
{
	return HMMPI::Mat(params.size(), params.size(), 0.0);		// zero matrix
}
//---------------------------------------------------------------------------
size_t PM_Linear::ModelledDataSize() const
{
	return G.ICount();
}
//---------------------------------------------------------------------------
void PM_Linear::PerturbData()
{
	size_t Ni = G.ICount();				// data size
	if (RndN == 0)
		throw HMMPI::Exception("RndN == 0 in PM_Linear::PerturbData");
	std::vector<double> stdNorm = RndN->get(Ni);

	if (DiagCov.size() != 0)			// diagonal case
	{
		if (Ni != DiagCov.size())
			throw HMMPI::Exception("Неправильный размер DiagCov в PM_Linear::PerturbData", "Wrong size of DiagCov in PM_Linear::PerturbData");

		for (size_t i = 0; i < Ni; i++)
			d0(i, 0) = d0_orig(i, 0) + sqrt(DiagCov[i])*stdNorm[i];
	}
	else								// dense case
	{
		if (Ni != FullCov.ICount() || Ni != FullCov.JCount())
			throw HMMPI::Exception("Неправильный размер FullCov в PM_Linear::PerturbData", "Wrong size of FullCov in PM_Linear::PerturbData");
		if (!holding_chol)
		{
			chol_FullCov = FullCov.Chol();
			holding_chol = true;
		}
		d0 = d0_orig + chol_FullCov*stdNorm;
	}

	if (RNK == 0)
	{
		FILE *fpet = 0;
#ifdef WRITE_PET_DATA
		fpet = fopen("pet_data_LIN.txt", "w");		// output only perturbed data
		if (fpet != NULL)
		{
			d0.SaveASCII(fpet);
			fclose(fpet);
		}
#endif
		fpet = fopen("MatVecVec_RML.txt", "w");		// output perturbed data within MATVECVEC
		if (fpet != NULL)
		{
			fprintf(fpet, "MATVECVEC\n");
			(G && d0 && cov_diag()).SaveASCII(fpet);
			fclose(fpet);
		}
	}
}
//---------------------------------------------------------------------------
// PM_Func
//---------------------------------------------------------------------------
PM_Func::PM_Func(int param_dim, const std::vector<double> &data, const std::vector<double> &c) : Npar(param_dim), d0(data), C(c)
{
	assert(d0.size() == C.size());

	for (auto &i : C)		// invert the matrix
		i = 1.0/i;
}
//---------------------------------------------------------------------------
PM_Func::~PM_Func()
{
#ifdef TESTCTOR

	std::ofstream testf(HMMPI::stringFormatArr("TESTCTOR_out_{0:%d}.txt", std::vector<int>{RNK}), std::ios::app);
	testf << "rank " << RNK << ", PM_Func -- DTOR --, this = " << this << "\n";
	testf.close();
#endif
}
//---------------------------------------------------------------------------
double PM_Func::ObjFunc(const std::vector<double> &params)
{
	assert(params.size() == (size_t)Npar);

	modelled_data = F(params);
	HMMPI::Mat v = HMMPI::Mat(modelled_data) - HMMPI::Mat(d0);
	return InnerProd(v, C%v);
}
//---------------------------------------------------------------------------
std::vector<double> PM_Func::ObjFuncGrad(const std::vector<double> &params)
{
	assert(params.size() == (size_t)Npar);

	HMMPI::Mat f = F(params);
	f = 2*(f - HMMPI::Mat(d0));
	DataSens = dF(params);
	return ((f.Tr() % C)*DataSens).ToVector();
}
//---------------------------------------------------------------------------
HMMPI::Mat PM_Func::ObjFuncHess(const std::vector<double> &params)
{
	HMMPI::Mat res;
	HMMPI::Mat f = HMMPI::Mat(F(params)) - HMMPI::Mat(d0);
	HMMPI::Mat Jac = dF(params);

	for (int i = 0; i < Npar; i++)
	{
		HMMPI::Mat Ji = Jac.Reorder(0, Jac.ICount(), i, i+1);			// i-th column of Jac
		HMMPI::Mat Hi = (Ji.Tr() % C)*Jac + (f.Tr() % C)*dJk(params, i);
		res = std::move(res) || (2*Hi);									// accumulate the Hessian line by line
	}

	return res;
}
//---------------------------------------------------------------------------
// PM_Func_lin
//---------------------------------------------------------------------------
std::vector<double> PM_Func_lin::F(const std::vector<double> &par) const	// forward operator
{
	return (G * HMMPI::Mat(par)).ToVector();
}
//---------------------------------------------------------------------------
HMMPI::Mat PM_Func_lin::dF(const std::vector<double> &par) const			// Jacobian
{
	return G;
}
//---------------------------------------------------------------------------
HMMPI::Mat PM_Func_lin::dJk(const std::vector<double> &par, int k) const	// derivatives of k-th column of Jacobian
{
	return HMMPI::Mat(G.ICount(), G.JCount(), 0.0);
}
//---------------------------------------------------------------------------
PM_Func_lin::PM_Func_lin(Parser_1 *K, KW_item *kw, MPI_Comm c) : PM_Func(K, kw, c)	// easy constructor; all data are taken from keywords of "K"; "kw" is used only to handle prerequisites
{
	DECLKWD(mat, KW_matvecvec, "MATVECVEC");
	name = "PM_Func_lin";

	kw->Start_pre();
	kw->Add_pre("MATVECVEC");
	kw->Add_pre("LIMITS");
	kw->Finish_pre();

	G = mat->M;
	if (G.JCount() != init.size())
		throw HMMPI::Exception("Не совпадают размерности матрицы G и LIMITS в модели FUNC_LIN",
							   "Inconsistent dimensions of matrix G and LIMITS in FUNC_LIN model");	// TODO
	Npar = init.size();

	d0 = mat->v1;
	C = mat->v2;
	for (auto &i : C)		// invert the matrix
		i = 1.0/i;

#ifdef TESTCTOR
	std::ofstream testf(HMMPI::stringFormatArr("TESTCTOR_out_{0:%d}.txt", std::vector<int>{RNK}), std::ios::app);
	testf << "rank " << RNK << ", PM_Func_lin easy CTOR, this = " << this << "\n";
	testf.close();
#endif
}
//---------------------------------------------------------------------------
// PM_Func_pow
//---------------------------------------------------------------------------
std::vector<double> PM_Func_pow::F(const std::vector<double> &par) const	// forward operator
{
	// F(par)_i = ln(a*(Si - S0)^b) = ln(a) + b*ln(Si - S0), where {a, b, S0} = par

	std::vector<double> res(Si.size());
	for (size_t i = 0; i < res.size(); i++)
		res[i] = log(par[0]) + par[1]*log(Si[i] - par[2]);

	return res;
}
//---------------------------------------------------------------------------
HMMPI::Mat PM_Func_pow::dF(const std::vector<double> &par) const			// Jacobian
{
	HMMPI::Mat res(Si.size(), 3, 0.0);
	for (size_t i = 0; i < Si.size(); i++)
	{
		res(i, 0) = 1/par[0];
		res(i, 1) = log(Si[i] - par[2]);
		res(i, 2) = -par[1]/(Si[i] - par[2]);
	}

	return res;
}
//---------------------------------------------------------------------------
HMMPI::Mat PM_Func_pow::dJk(const std::vector<double> &par, int k) const	// derivatives of k-th column of Jacobian
{
	HMMPI::Mat res(Si.size(), 3, 0.0);
	if (k == 0)
	{
		for (size_t i = 0; i < Si.size(); i++)
		{
			res(i, 0) = -1/(par[0]*par[0]);
			res(i, 1) = 0;
			res(i, 2) = 0;
		}
	}
	else if (k == 1)
	{
		for (size_t i = 0; i < Si.size(); i++)
		{
			res(i, 0) = 0;
			res(i, 1) = 0;
			res(i, 2) = -1/(Si[i] - par[2]);
		}
	}
	else if (k == 2)
	{
		for (size_t i = 0; i < Si.size(); i++)
		{
			res(i, 0) = 0;
			res(i, 1) = -1/(Si[i] - par[2]);
			res(i, 2) = -par[1]/((Si[i] - par[2])*(Si[i] - par[2]));
		}
	}
	else
		throw HMMPI::Exception("Incorrect k in PM_Func_pow::dJk");

	return res;
}
//---------------------------------------------------------------------------
PM_Func_pow::PM_Func_pow(Parser_1 *K, KW_item *kw, MPI_Comm c, int j) : PM_Func(K, kw, c), small(1e-5), big(1e5)	// easy constructor
{
	DECLKWD(mat, KW_matvec, "MATVEC");
	name = "PM_Func_pow";

	kw->Start_pre();
	kw->Add_pre("MATVEC");
	kw->Add_pre("LIMITS");
	kw->Finish_pre();

	HMMPI::Mat colj = mat->M.Reorder(0, mat->M.ICount(), j, j+1);
	Si = colj.ToVector();
	int i0, j0;
	const double minSi = colj.Min(i0, j0);

	if (init.size() != 3)
		throw HMMPI::Exception("Размерность LIMITS в модели FUNC_POW должна быть равна 3",
							   "Dimension of LIMITS in FUNC_POW model should equal 3");
	if (minSi <= small)
		throw HMMPI::Exception(HMMPI::stringFormatArr("В модели FUNC_POW величины Si должны быть > {0:%g}",
													  "In model FUNC_POW, Si should be > {0:%g}", small));
	Npar = 3;
	d0 = mat->v1;
	for (auto &x : d0)			// data are treated in log scale
		x = log(x);
	C = std::vector<double>(d0.size(), 1.0);

	min = {small, -big, 0};
	max = {big, -small, minSi-small};

#ifdef TESTCTOR
	std::ofstream testf(HMMPI::stringFormatArr("TESTCTOR_out_{0:%d}.txt", std::vector<int>{RNK}), std::ios::app);
	testf << "rank " << RNK << ", PM_Func_pow easy CTOR, this = " << this << "\n";
	testf.close();
#endif
}
//---------------------------------------------------------------------------
