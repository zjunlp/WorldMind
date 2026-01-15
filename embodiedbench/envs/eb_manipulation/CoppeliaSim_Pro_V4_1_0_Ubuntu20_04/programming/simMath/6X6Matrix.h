#pragma once

#include "mathDefines.h"
#include "6Vector.h"
#include "3X3Matrix.h"

class C6X6Matrix  
{
public:
    C6X6Matrix();
    C6X6Matrix(const C6X6Matrix& m);
    C6X6Matrix(const C3X3Matrix& m00,const C3X3Matrix& m01,const C3X3Matrix& m10,const C3X3Matrix& m11);
    ~C6X6Matrix();

    void clear();
    void setMultResult(const C6X6Matrix& m1,const C6X6Matrix& m2);
    void setIdentity();

    C6X6Matrix operator* (const C6X6Matrix& m) const;
    C6X6Matrix operator+ (const C6X6Matrix& m) const;
    C6X6Matrix operator- (const C6X6Matrix& m) const;

    void operator*= (const C6X6Matrix& m);
    void operator+= (const C6X6Matrix& m);
    void operator-= (const C6X6Matrix& m);

    C6Vector operator* (const C6Vector& v) const;
    C6X6Matrix& operator= (const C6X6Matrix& m);

    inline void getInternalData(simMathReal d[36]) const
    {
        M[0][0].getInternalData(d+0);
        M[0][1].getInternalData(d+9);
        M[1][0].getInternalData(d+18);
        M[1][1].getInternalData(d+27);
    }
    inline void setInternalData(const simMathReal d[36])
    {
        M[0][0].setInternalData(d+0);
        M[0][1].setInternalData(d+9);
        M[1][0].setInternalData(d+18);
        M[1][1].setInternalData(d+27);
    }
    inline simMathReal& operator() (size_t i,size_t j)
    {
        return(M[i/3][j/3](i%3,j%3));
    }
    inline const simMathReal& operator() (size_t i,size_t j) const
    {
        return(M[i/3][j/3](i%3,j%3));
    }

    C3X3Matrix M[2][2];
};

