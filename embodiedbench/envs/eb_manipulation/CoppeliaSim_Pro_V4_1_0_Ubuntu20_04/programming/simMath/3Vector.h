#pragma once

#include "mathDefines.h"

class C3X3Matrix;
class C4X4Matrix;
class C7Vector;

class C3Vector  
{
public:

    C3Vector();
    C3Vector(simMathReal v0,simMathReal v1,simMathReal v2);
    C3Vector(const simMathReal v[3]);
    C3Vector(const C3Vector& v);
    ~C3Vector();

    void buildInterpolation(const C3Vector& fromThis,const C3Vector& toThat,simMathReal t);
    simMathReal getAngle(const C3Vector& v) const;
    C3X3Matrix getProductWithStar() const;

    void operator*= (const C4X4Matrix& m);
    void operator*= (const C3X3Matrix& m);
    void operator*= (const C7Vector& transf);

    inline void getInternalData(simMathReal d[3]) const
    {
        d[0]=data[0];
        d[1]=data[1];
        d[2]=data[2];
    }
    inline void setInternalData(const simMathReal d[3])
    {
        data[0]=d[0];
        data[1]=d[1];
        data[2]=d[2];
    }
    inline simMathReal* ptr()
    {
        return(data);
    }
    inline bool isColinear(const C3Vector& v,simMathReal precision) const
    {
        simMathReal scalProdSq=(*this)*v;
        scalProdSq=scalProdSq*scalProdSq;
        simMathReal l1=(*this)*(*this);
        simMathReal l2=v*v;
        return((scalProdSq/(l1*l2))>=precision);
    }
    inline simMathReal& operator() (size_t i)
    {
        return(data[i]);
    }
    inline const simMathReal& operator() (size_t i) const
    {
        return(data[i]);
    }
    inline simMathReal getLength() const
    {
        return(sqrt(data[0]*data[0]+data[1]*data[1]+data[2]*data[2]));
    }
    inline void copyTo(simMathReal v[3]) const
    {
        v[0]=data[0];
        v[1]=data[1];
        v[2]=data[2];
    }
    inline void set(const simMathReal v[3])
    {
        data[0]=v[0];
        data[1]=v[1];
        data[2]=v[2];
    }
    inline void get(simMathReal v[3]) const
    {
        v[0]=data[0];
        v[1]=data[1];
        v[2]=data[2];
    }
    inline C3Vector getNormalized() const
    {
        C3Vector retV;
        simMathReal l=sqrt(data[0]*data[0]+data[1]*data[1]+data[2]*data[2]);
        if (l!=simZero)
        {
            retV(0)=data[0]/l;
            retV(1)=data[1]/l;
            retV(2)=data[2]/l;
            return(retV);
        }
        return(C3Vector::zeroVector);
    }
    inline void keepMax(const C3Vector& v)
    {
        if (v(0)>data[0])
            data[0]=v(0);
        if (v(1)>data[1])
            data[1]=v(1);
        if (v(2)>data[2])
            data[2]=v(2);
    }
    inline void keepMin(const C3Vector& v)
    {
        if (v(0)<data[0])
            data[0]=v(0);
        if (v(1)<data[1])
            data[1]=v(1);
        if (v(2)<data[2])
            data[2]=v(2);
    }
    inline bool isValid() const
    {
        return((SIM_IS_FINITE(data[0])!=0)&&(SIM_IS_FINITE(data[1])!=0)&&(SIM_IS_FINITE(data[2])!=0)&&(SIM_IS_NAN(data[0])==0)&&(SIM_IS_NAN(data[1])==0)&&(SIM_IS_NAN(data[2])==0));
    }
    inline void set(simMathReal v0,simMathReal v1,simMathReal v2)
    {
        data[0]=v0;
        data[1]=v1;
        data[2]=v2;
    }
    inline void normalize()
    {
        simMathReal l=sqrt(data[0]*data[0]+data[1]*data[1]+data[2]*data[2]);
        if (l!=simZero)
        {
            data[0]=data[0]/l;
            data[1]=data[1]/l;
            data[2]=data[2]/l;
        }
    }
    inline void clear()
    {
        data[0]=simZero;
        data[1]=simZero;
        data[2]=simZero;
    }
    inline C3Vector operator/ (simMathReal d) const
    {
        C3Vector retV;
        retV(0)=data[0]/d;
        retV(1)=data[1]/d;
        retV(2)=data[2]/d;
        return(retV);
    }
    inline void operator/= (simMathReal d)
    {
        data[0]/=d;
        data[1]/=d;
        data[2]/=d;
    }
    inline C3Vector operator* (simMathReal d) const
    {
        C3Vector retV;
        retV(0)=data[0]*d;
        retV(1)=data[1]*d;
        retV(2)=data[2]*d;
        return(retV);
    }
    inline void operator*= (simMathReal d)
    {
        data[0]*=d;
        data[1]*=d;
        data[2]*=d;
    }
    inline C3Vector& operator= (const C3Vector& v)
    {
        data[0]=v(0);
        data[1]=v(1);
        data[2]=v(2);
        return(*this);
    }
    inline bool operator!= (const C3Vector& v)
    {
        return( (data[0]!=v(0))||(data[1]!=v(1))||(data[2]!=v(2)) );
    }
    inline C3Vector operator+ (const C3Vector& v) const
    {
        C3Vector retV;
        retV(0)=data[0]+v(0);
        retV(1)=data[1]+v(1);
        retV(2)=data[2]+v(2);
        return(retV);
    }
    inline void operator+= (const C3Vector& v)
    {
        data[0]+=v(0);
        data[1]+=v(1);
        data[2]+=v(2);
    }
    inline C3Vector operator- (const C3Vector& v) const
    {
        C3Vector retV;
        retV(0)=data[0]-v(0);
        retV(1)=data[1]-v(1);
        retV(2)=data[2]-v(2);
        return(retV);
    }
    inline void operator-= (const C3Vector& v)
    {
        data[0]-=v(0);
        data[1]-=v(1);
        data[2]-=v(2);
    }
    inline C3Vector operator^ (const C3Vector& v) const
    { // Cross product
        C3Vector retV;
        retV(0)=data[1]*v(2)-data[2]*v(1);
        retV(1)=data[2]*v(0)-data[0]*v(2);
        retV(2)=data[0]*v(1)-data[1]*v(0);
        return(retV);
    }
    inline void operator^= (const C3Vector& v)
    { // Cross product
        C3Vector retV;
        retV(0)=data[1]*v(2)-data[2]*v(1);
        retV(1)=data[2]*v(0)-data[0]*v(2);
        retV(2)=data[0]*v(1)-data[1]*v(0);
        data[0]=retV(0);
        data[1]=retV(1);
        data[2]=retV(2);
    }
    inline simMathReal operator* (const C3Vector& v) const
    { // Scalar product
        return(data[0]*v.data[0]+data[1]*v.data[1]+data[2]*v.data[2]);
    }

    static const C3Vector oneOneOneVector;
    static const C3Vector unitXVector;
    static const C3Vector unitYVector;
    static const C3Vector unitZVector;
    static const C3Vector zeroVector;

    simMathReal data[3];
};




