//------------------------------------------------------------------------------------------
// File: cmaes.h - Author: Nikolaus Hansen
// last modified: IX 2010
// by: Nikolaus Hansen
//------------------------------------------------------------------------------------------

#ifndef NH_cmaes_h		// only include ones
#define NH_cmaes_h

#include <time.h>

//------------------------------------------------------------------------------------------
typedef struct
// random_t
// sets up a pseudo random number generator instance
{
  // Variables for Uniform()
  long int startseed;
  long int aktseed;
  long int aktrand;
  long int *rgrand;

  // Variables for Gauss()
  short flgstored;
  double hold;
} random_t;
//------------------------------------------------------------------------------------------
long   random_init(random_t *, long unsigned seed);	 // 0==clock
void   random_exit(random_t *);
double random_Gauss(random_t *);	// (0,1)-normally distributed
//------------------------------------------------------------------------------------------
typedef struct
// timings_t
// time measurement, used to time eigendecomposition
{
  // for outside use
  double totaltime;		// zeroed by calling re-calling timings_start
  double totaltotaltime;
  double tictoctime;
  double lasttictoctime;

  // local fields
  clock_t lastclock;
  time_t lasttime;
  clock_t ticclock;
  time_t tictime;
  short istic;
  short isstarted;

  double lastdiff;
  double tictoczwischensumme;
} timings_t;
//------------------------------------------------------------------------------------------
typedef struct
// readpara_t
// collects all parameters, in particular those that are read from
// a file before to start. This should split in future?
{
  // input parameter
  int N;				// problem dimension, must stay constant
  unsigned int seed;
  double * xstart;
  double * typicalX;
  int typicalXcase;
  double * rgInitialStds;
  double * rgDiffMinChange;

  // termination parameters
  double stopMaxFunEvals;
  double facmaxeval;
  double stopMaxIter;
  struct { int flg; double val; } stStopFitness;
  double stopTolFun;
  double stopTolFunHist;
  double stopTolX;
  double stopTolUpXFactor;

  // internal evolution strategy parameters
  int lambda;				// -> mu, <- N
  int mu;					// -> weights, (lambda)
  double mucov, mueff;		// <- weights
  double *weights;			// <- mu, -> mueff, mucov, ccov
  double damps;				// <- cs, maxeval, lambda
  double cs;				// -> damps, <- N
  double ccumcov;			// <- N
  double ccov;				// <- mucov, <- N
  double diagonalCov;		// number of initial iterations
  struct {int flgalways; double modulo; double maxtime;} updateCmode;
  double facupdateCmode;

  // supplementary variables
  char *weigkey;
  char resumefile[99];
  const char **rgsformat;
  void **rgpadr;
  const char **rgskeyar;
  double ***rgp2adr;
  int n1para, n1outpara;
  int n2para;
} readpara_t;
//------------------------------------------------------------------------------------------
typedef struct
// cmaes_t
// CMA-ES "object"
{
  const char *version;
  readpara_t sp;
  random_t rand;		// random number generator

  double sigma;			// step size

  double *rgxmean;		// mean x vector, "parent"
  double *rgxbestever;
  double **rgrgx;		// range of x-vectors, lambda offspring
  int *index;			// sorting index of sample pop.
  double *arFuncValueHist;

  short flgIniphase;	// not really in use anymore
  short flgStop;

  double chiN;
  double **C;			// lower triangular matrix: i>=j for C[i][j]
  double **B;			// matrix with normalize eigenvectors in columns
  double *rgD;			// axis lengths

  unsigned int tot_models;	// total models: sampled and re-sampled
  unsigned int tot_ignored;	// total models ignored because of small dens_eval (dens_eval set = 0)
  double eps_i;			// models with exp(...) < eps_i will be ignored; eps_i depends on dimension
  double *dens_val;		// density value for models, length = lambda	19.04.2013
  double *x_mu;			// x - mu, auxiliary array, length = N			19.04.2013
  double **BiBj;		// Gram matrix (NxN) for B columns				19.04.2013

  double *rgpc;
  double *rgps;
  double *rgxold;
  double *rgout;
  double *rgBDz;			// for B*D*z
  double *rgdTmp;			// temporary (random) vector used in different places
  double *rgFuncValue;
  double *publicFitness;	// returned by cmaes_init()

  double gen;				// Generation number
  double countevals;
  double state;				// 1 == sampled, 2 == not in use anymore, 3 == updated

  double maxdiagC;			// repeatedly used for output
  double mindiagC;
  double maxEW;
  double minEW;

  char sOutString[330];		// 4x80

  short flgEigensysIsUptodate;
  short flgCheckEigen;		// control via signals.par
  double genOfEigensysUpdate;
  timings_t eigenTimings;

  double dMaxSignifKond;
  double dLastMinEWgroesserNull;

  short flgresumedone;

  time_t printtime;
  time_t writetime;			// ideally should keep track for each output file
  time_t firstwritetime;
  time_t firstprinttime;

} cmaes_t;
//------------------------------------------------------------------------------------------

#endif /* NH_cmaes_h */
