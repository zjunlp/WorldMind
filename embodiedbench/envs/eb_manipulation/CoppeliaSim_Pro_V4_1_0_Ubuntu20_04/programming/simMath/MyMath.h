#pragma once

#include "mathDefines.h"
#include <vector>
#include "3Vector.h"
#include "4Vector.h"
#include "6Vector.h"
#include "7Vector.h"
#include "Vector.h"
#include "3X3Matrix.h"
#include "4X4Matrix.h"
#include "6X6Matrix.h"
#include "4X4FullMatrix.h"
#include "MMatrix.h"

class CMath  
{
public:
    CMath();
    virtual ~CMath();

    static void limitValue(simMathReal minValue,simMathReal maxValue,simMathReal &value);
    static void limitValue(int minValue,int maxValue,int &value);


    static simMathReal robustAsin(simMathReal v);
    static simMathReal robustAcos(simMathReal v);
    static simMathReal robustFmod(simMathReal v,simMathReal w);
    static double robustmod(double v,double w);
    static bool isFloatNumberOk(float v);
    static bool isDoubleNumberOk(double v);
    static bool isRealNumberOk(simMathReal v);
};
