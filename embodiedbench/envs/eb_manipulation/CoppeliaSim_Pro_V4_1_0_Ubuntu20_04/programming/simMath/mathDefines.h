#pragma once

#include <math.h>
#include <limits>
#include <float.h>
#include <cstdlib>
#include <cmath>

#define SIM_MAX_FLOAT (0.01f*FLT_MAX)
#define SIM_MAX_DOUBLE (0.01*DBL_MAX)

#ifdef SIM_MATH_DOUBLE
typedef double simMathReal;
#define REAL_MAX DBL_MAX
#define SIM_MAX_REAL SIM_MAX_DOUBLE
#else
typedef float simMathReal;
#define REAL_MAX FLT_MAX
#define SIM_MAX_REAL SIM_MAX_FLOAT
#endif

#define piValue_f 3.14159265359f
#define piValD2_f 1.570796326794f
#define piValTimes2_f 6.28318530718f
#define radToDeg_f 57.2957795130785499f
#define degToRad_f 0.017453292519944444f

#define piValue simMathReal(3.14159265359)
#define piValD2 simMathReal(1.570796326794)
#define piValTimes2 simMathReal(6.28318530718)
#define radToDeg simMathReal(57.2957795130785499)
#define degToRad simMathReal(0.017453292519944444)

#define simZero simMathReal(0.0)
#define simHalf simMathReal(0.5)
#define simOne simMathReal(1.0)
#define simTwo simMathReal(2.0)

#define SIM_MAX_INT INT_MAX
#define SIM_RAND_FLOAT (static_cast<simMathReal>(rand())/static_cast<simMathReal>(RAND_MAX))
#define SIM_IS_NAN(x) ((std::isnan)(x))
#define SIM_IS_FINITE(x) ((std::isfinite)(x))
