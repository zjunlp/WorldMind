#pragma once

#include "mathDefines.h"
#include "3Vector.h"
#include "4Vector.h"
#include "6Vector.h"
#include "7Vector.h"

class CVector  
{
public:
    CVector();
    CVector(size_t nElements);
    CVector(const C3Vector& v);
    CVector(const C4Vector& v);
    CVector(const C6Vector& v);
    CVector(const C7Vector& v);
    CVector(const CVector& v);
    ~CVector();

    CVector operator* (simMathReal d) const;
    CVector operator/ (simMathReal d) const;
    CVector operator+ (const CVector& v) const;
    CVector operator- (const CVector& v) const;

    void operator*= (simMathReal d);
    void operator/= (simMathReal d);
    void operator+= (const CVector& v);
    void operator-= (const CVector& v);
    
    simMathReal operator* (const C3Vector& v) const;
    simMathReal operator* (const C4Vector& v) const;
    simMathReal operator* (const C6Vector& v) const;
    simMathReal operator* (const C7Vector& v) const;
    simMathReal operator* (const CVector& v) const;

    CVector& operator= (const C3Vector& v);
    CVector& operator= (const C4Vector& v);
    CVector& operator= (const C6Vector& v);
    CVector& operator= (const C7Vector& v);
    CVector& operator= (const CVector& v);

inline simMathReal& operator() (size_t i)
{
    return(data[i]);
}

inline const simMathReal& operator() (size_t i) const
{
    return(data[i]);
}

    size_t elements;
private:
    simMathReal* data;
};

