#include "Vector.h"

CVector::CVector()
{
}

CVector::CVector(size_t nElements)
{
    data=new simMathReal[nElements];
    elements=nElements;
}

CVector::CVector(const C3Vector& v)
{
    data=new simMathReal[3];
    elements=3;
    (*this)=v;
}

CVector::CVector(const C4Vector& v)
{
    data=new simMathReal[4];
    elements=4;
    (*this)=v;
}

CVector::CVector(const C6Vector& v)
{
    data=new simMathReal[6];
    elements=6;
    (*this)=v;
}

CVector::CVector(const C7Vector& v)
{
    data=new simMathReal[7];
    elements=7;
    (*this)=v;
}

CVector::CVector(const CVector& v)
{
    data=new simMathReal[v.elements];
    elements=v.elements;
    (*this)=v;
}
 
CVector::~CVector()
{
   delete[] data;
} 

CVector& CVector::operator= (const C3Vector& v)
{
    for (size_t i=0;i<3;i++)
        data[i]=v(i);
    return(*this);
}

CVector& CVector::operator= (const C4Vector& v)
{
    for (size_t i=0;i<4;i++)
        data[i]=v(i);
    return(*this);
}

CVector& CVector::operator= (const C6Vector& v)
{
    for (size_t i=0;i<6;i++)
        data[i]=v(i);
    return(*this);
}

CVector& CVector::operator= (const C7Vector& v)
{
    for (size_t i=0;i<7;i++)
        data[i]=v(i);
    return(*this);
}

CVector& CVector::operator= (const CVector& v)
{
    for (size_t i=0;i<elements;i++)
        data[i]=v.data[i];
    return(*this);
}

CVector CVector::operator* (simMathReal d) const
{
    CVector retV(elements);
    for (size_t i=0;i<elements;i++)
        retV.data[i]=data[i]*d;
    return(retV);
}

CVector CVector::operator/ (simMathReal d) const
{
    CVector retV(elements);
    for (size_t i=0;i<elements;i++)
        retV.data[i]=data[i]/d;
    return(retV);
}

CVector CVector::operator+ (const CVector& v) const
{
    CVector retV(elements);
    for (size_t i=0;i<elements;i++)
        retV.data[i]=data[i]+v.data[i];
    return(retV);
}

CVector CVector::operator- (const CVector& v) const
{
    CVector retV(elements);
    for (size_t i=0;i<elements;i++)
        retV.data[i]=data[i]-v.data[i];
    return(retV);
}
    
simMathReal CVector::operator* (const C3Vector& v) const
{
    simMathReal retVal=simZero;
    for (size_t i=0;i<3;i++)
        retVal+=(data[i]*v(i));
    return(retVal);
}

simMathReal CVector::operator* (const C4Vector& v) const
{
    simMathReal retVal=simZero;
    for (size_t i=0;i<4;i++)
        retVal+=(data[i]*v(i));
    return(retVal);
}

simMathReal CVector::operator* (const C6Vector& v) const
{
    simMathReal retVal=simZero;
    for (size_t i=0;i<6;i++)
        retVal+=(data[i]*v(i));
    return(retVal);
}

simMathReal CVector::operator* (const C7Vector& v) const
{
    simMathReal retVal=simZero;
    for (size_t i=0;i<7;i++)
        retVal+=(data[i]*v(i));
    return(retVal);
}

simMathReal CVector::operator* (const CVector& v) const
{
    simMathReal retVal=simZero;
    for (size_t i=0;i<elements;i++)
        retVal+=(data[i]*v.data[i]);
    return(retVal);
}

void CVector::operator*= (simMathReal d)
{
    for (size_t i=0;i<elements;i++)
        data[i]*=d;
}

void CVector::operator/= (simMathReal d)
{
    for (size_t i=0;i<elements;i++)
        data[i]/=d;
}

void CVector::operator+= (const CVector& v) 
{
    for (size_t i=0;i<elements;i++)
        data[i]+=v.data[i];
}

void CVector::operator-= (const CVector& v) 
{
    for (size_t i=0;i<elements;i++)
        data[i]-=v.data[i];
}
