#pragma once

#include "mathDefines.h"
#include "3Vector.h"
#include "3X3Matrix.h"

class C4Vector  
{
public:
    C4Vector();
    C4Vector(simMathReal v0,simMathReal v1,simMathReal v2,simMathReal v3);
    C4Vector(const simMathReal v[4]);
    C4Vector(const C3Vector& v);
    C4Vector(const C4Vector& q);
    C4Vector(simMathReal a,simMathReal b,simMathReal g);
    C4Vector(simMathReal angle,const C3Vector& axis);
    C4Vector(const C3Vector& startV,const C3Vector& endV);
    ~C4Vector();

    void setEulerAngles(const C3Vector& v);
    void setEulerAngles(simMathReal a,simMathReal b,simMathReal g);
    void setAngleAndAxis(simMathReal angle,const C3Vector& axis);
    void setVectorMapping(const C3Vector& startV,const C3Vector& endV);
    void buildInterpolation(const C4Vector& fromThis,const C4Vector& toThat,simMathReal t);
    void buildInterpolation_otherWayRound(const C4Vector& fromThis,const C4Vector& toThat,simMathReal t);
    void buildRandomOrientation();

    simMathReal getAngleBetweenQuaternions(const C4Vector& q) const;
    C4Vector getAngleAndAxis() const;
    C4Vector getAngleAndAxisNoChecking() const;
    
    C3Vector getEulerAngles() const;

    inline void getInternalData(simMathReal d[4]) const
    {
        d[0]=data[0];
        d[1]=data[1];
        d[2]=data[2];
        d[3]=data[3];
    }
    inline void setInternalData(const simMathReal d[4])
    {
        data[0]=d[0];
        data[1]=d[1];
        data[2]=d[2];
        data[3]=d[3];
    }
    inline simMathReal& operator() (size_t i)
    {
        return(data[i]);
    }
    inline const simMathReal& operator() (size_t i) const
    {
        return(data[i]);
    }
    inline void normalize()
    {
        simMathReal l=sqrt(data[0]*data[0]+data[1]*data[1]+data[2]*data[2]+data[3]*data[3]);
        data[0]/=l;
        data[1]/=l;
        data[2]/=l;
        data[3]/=l;
    }
    inline void clear()
    {
        data[0]=simZero;
        data[1]=simZero;
        data[2]=simZero;
        data[3]=simZero;
    }
    inline void setIdentity()
    {
        data[0]=simOne;
        data[1]=simZero;
        data[2]=simZero;
        data[3]=simZero;
    }
    inline C4Vector getInverse() const
    {
        return(C4Vector(data[0],-data[1],-data[2],-data[3]));
    }
    inline void inverse()
    {
        data[1]=-data[1];
        data[2]=-data[2];
        data[3]=-data[3];
    }
    inline C4Vector operator/ (simMathReal d) const
    {
        C4Vector retV;
        retV(0)=data[0]/d;
        retV(1)=data[1]/d;
        retV(2)=data[2]/d;
        retV(3)=data[3]/d;
        return(retV);
    }
    inline C4Vector operator* (simMathReal d) const
    {
        C4Vector retV;
        retV(0)=data[0]*d;
        retV(1)=data[1]*d;
        retV(2)=data[2]*d;
        retV(3)=data[3]*d;
        return(retV);
    }
    inline C4Vector& operator= (const C4Vector& v)
    {
        data[0]=v(0);
        data[1]=v(1);
        data[2]=v(2);
        data[3]=v(3);
        return(*this);
    }
    inline bool operator!= (const C4Vector& v)
    {
        return( (data[0]!=v(0))||(data[1]!=v(1))||(data[2]!=v(2))||(data[3]!=v(3)) );
    }
    inline C4Vector operator* (const C4Vector& v) const
    { // Quaternion multiplication.
        C4Vector retV;
        retV(0)=data[0]*v(0)-data[1]*v(1)-data[2]*v(2)-data[3]*v(3);
        retV(1)=data[0]*v(1)+v(0)*data[1]+data[2]*v(3)-data[3]*v(2);
        retV(2)=data[0]*v(2)+v(0)*data[2]+data[3]*v(1)-data[1]*v(3);
        retV(3)=data[0]*v(3)+v(0)*data[3]+data[1]*v(2)-data[2]*v(1);
        //  retV.normalize(); // NOOOOOOO!!!!!! We might compute the rotation of a vector which should be (q*v*qI).normalize and not q*((v*qi).normalize).normalize !!
        return(retV);
    }
    inline C3Vector getAxis(size_t index) const
    {
        C3X3Matrix m=getMatrix();
        return(m.axis[index]);
    }
    inline C3Vector operator* (const C3Vector& v) const
    { // Rotation of a vector.
        C4Vector tmpV(simOne,v(0),v(1),v(2));
        tmpV=(*this)*(tmpV*getInverse());
        return(C3Vector(tmpV(1),tmpV(2),tmpV(3)));
    }
    inline C4Vector operator+ (const C4Vector& v) const
    {
        C4Vector retV;
        retV(0)=data[0]+v(0);
        retV(1)=data[1]+v(1);
        retV(2)=data[2]+v(2);
        retV(3)=data[3]+v(3);
        return(retV);
    }
    inline C3X3Matrix getMatrix() const
    {
        C3X3Matrix retM;
        simMathReal xx=data[1]*data[1];
        simMathReal xy=data[1]*data[2];
        simMathReal xz=data[1]*data[3];
        simMathReal xw=data[1]*data[0];
        simMathReal yy=data[2]*data[2];
        simMathReal yz=data[2]*data[3];
        simMathReal yw=data[2]*data[0];
        simMathReal zz=data[3]*data[3];
        simMathReal zw=data[3]*data[0];

        retM(0,0)=simOne-simTwo*(yy+zz);
        retM(0,1)=simTwo*(xy-zw);
        retM(0,2)=simTwo*(xz+yw);
        retM(1,0)=simTwo*(xy+zw);
        retM(1,1)=simOne-simTwo*(xx+zz);
        retM(1,2)=simTwo*(yz-xw);
        retM(2,0)=simTwo*(xz-yw);
        retM(2,1)=simTwo*(yz+xw);
        retM(2,2)=simOne-simTwo*(xx+yy);
        return(retM);
    }
    inline void operator/= (simMathReal d)
    {
        data[0]/=d;
        data[1]/=d;
        data[2]/=d;
        data[3]/=d;
    }
    inline void operator*= (simMathReal d)
    {
        data[0]*=d;
        data[1]*=d;
        data[2]*=d;
        data[3]*=d;
    }
    inline void operator*= (const C4Vector& v)
    {
        (*this)=(*this)*v;
        // Already normalized through * operator
    }
    inline void operator+= (const C4Vector& v)
    {
        data[0]+=v(0);
        data[1]+=v(1);
        data[2]+=v(2);
        data[3]+=v(3);
    }

    static const C4Vector identityRotation;

    simMathReal data[4];
};
