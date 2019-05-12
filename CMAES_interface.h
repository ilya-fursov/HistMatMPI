/*
 * CMAES_interface.h
 *
 *  Created on: Mar 30, 2013
 *      Author: ilya
 */

#ifndef CMAES_INTERFACE_H_
#define CMAES_INTERFACE_H_

//------------------------------------------------------------------------------------------
// File: cmaes_interface.h - Author: Nikolaus Hansen
// last modified: IV 2007
// by: Nikolaus Hansen
//------------------------------------------------------------------------------------------

#include "cmaes.h"
#include "PhysModels.h"
#include "Parsing.h"
#include "GradientOpt.h"

//------------------------------------------------------------------------------------------
// Interface
//------------------------------------------------------------------------------------------
class OptCMAES : public Optimizer
{
public:
	double of;					// filled with ObjFunc() value after optimization

	OptCMAES() : of(0){};
	virtual std::vector<double> RunOpt(PhysModel *pm, std::vector<double> x0, const OptContext *ctx);	// Optimises "pm" starting from "x0" (full-dim);
																				// "ctx" should be Parser_1*, "pm" should be PhysModMPI*
																				// MPI: inputs and outputs are sync on all pm-comm-ranks
	virtual std::string ReportMsg();
};
//------------------------------------------------------------------------------------------
struct CWD_holder
{
	static std::string N;
};
//------------------------------------------------------------------------------------------
double *optimize(PhysModMPI *PM, int nrestarts, double incpopsize, const long maxResample, const char *filename, Parser_1 *kw, double &fbestever);
					// maxResample - ìàêñ. èòåðàöèé íà îäèí öèêë ïåðåìîäåëèðîâàíèÿ ñë. âåë., ïðè ïðåâûøåíèè âûäàåòñÿ îøèáêà
					// the returned array should be deleted manually;
					// worker processes return 0
					// fbestever - minimum found o.f. value

// --- initialization, constructors, destructors ---
double * cmaes_init(cmaes_t *, int dimension , double *xstart,
		double *stddev, long seed, int lambda,
		const char *input_parameter_filename);
void cmaes_resume_distribution(cmaes_t *evo_ptr, char *filename);
void cmaes_exit(cmaes_t *);

// --- core functions ---
double * const * cmaes_SamplePopulation(cmaes_t *);
double *         cmaes_UpdateDistribution(cmaes_t *,
					  const double *rgFitnessValues);
const char *     cmaes_TestForTermination(cmaes_t *);

// --- additional functions ---
double * const * cmaes_ReSampleSingle( cmaes_t *t, int index);
double const *   cmaes_ReSampleSingle_old(cmaes_t *, double *rgx);
double *         cmaes_SampleSingleInto( cmaes_t *t, double *rgx);
void             cmaes_UpdateEigensystem(cmaes_t *, int flgforce);

// --- getter functions ---
double         cmaes_Get(cmaes_t *, char const *keyword);
const double * cmaes_GetPtr(cmaes_t *, char const *keyword); /* e.g. "xbestever" */
double *       cmaes_GetNew( cmaes_t *t, char const *keyword);
double *       cmaes_GetInto( cmaes_t *t, char const *keyword, double *mem);

// --- online control and output ---
void           cmaes_ReadSignals(cmaes_t *, char const *filename);
void           cmaes_WriteToFile(cmaes_t *, const char *szKeyWord,
                                 const char *output_filename);
char *         cmaes_SayHello(cmaes_t *);
// --- misc ---
double *       cmaes_NewDouble(int n);
void           cmaes_FATAL(char const *s1, char const *s2, char const *s3,
			   char const *s4);

//------------------------------------------------------------------------------------------
// read limits from file - [dim] pairs
double **ReadLimits(const char *fname, int dim);

// release memory - [dim] pairs
void ReleaseLimits(double **mem, int dim);

// checks boundaries
int is_feasible(double *par, double **lim, int dim);
double sgn(double d);
//------------------------------------------------------------------------------------------

#endif /* CMAES_INTERFACE_H_ */
