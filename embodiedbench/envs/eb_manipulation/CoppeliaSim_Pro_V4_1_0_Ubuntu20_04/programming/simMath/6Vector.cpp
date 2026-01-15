#include "6Vector.h"
#include "6X6Matrix.h"

C6Vector::C6Vector()
{
}

C6Vector::C6Vector(simMathReal v0,simMathReal v1,simMathReal v2,simMathReal v3,simMathReal v4,simMathReal v5)
{
    V[0](0)=v0;
    V[0](1)=v1;
    V[0](2)=v2;
    V[1](0)=v3;
    V[1](1)=v4;
    V[1](2)=v5;
}

C6Vector::C6Vector(const simMathReal v[6])
{
    V[0](0)=v[0];
    V[0](1)=v[1];
    V[0](2)=v[2];
    V[1](0)=v[3];
    V[1](1)=v[4];
    V[1](2)=v[5];
}

C6Vector::C6Vector(const C3Vector& v0,const C3Vector& v1)
{
    V[0]=v0;
    V[1]=v1;
}

C6Vector::C6Vector(const C6Vector& v)
{
    V[0]=v.V[0];
    V[1]=v.V[1];
}

C6Vector::~C6Vector()
{

}

void C6Vector::clear()
{
    V[0].clear();
    V[1].clear();
}

C6Vector C6Vector::operator* (simMathReal d) const
{
    C6Vector retV;
    retV.V[0]=V[0]*d;
    retV.V[1]=V[1]*d;
    return(retV);
}

C6Vector C6Vector::operator/ (simMathReal d) const
{
    C6Vector retV;
    retV.V[0]=V[0]/d;
    retV.V[1]=V[1]/d;
    return(retV);
}

C6Vector& C6Vector::operator= (const C6Vector& v)
{
    V[0]=v.V[0];
    V[1]=v.V[1];
    return(*this);
}
C6Vector C6Vector::operator+ (const C6Vector& v) const
{
    C6Vector retV;
    retV.V[0]=V[0]+v.V[0];
    retV.V[1]=V[1]+v.V[1];
    return(retV);
}
C6Vector C6Vector::operator- (const C6Vector& v) const
{
    C6Vector retV;
    retV.V[0]=V[0]-v.V[0];
    retV.V[1]=V[1]-v.V[1];
    return(retV);
}
simMathReal C6Vector::operator* (const C6Vector& v) const
{ // Scalar product of spatial vectors:
    return(V[0](0)*v.V[1](0)+V[0](1)*v.V[1](1)+V[0](2)*v.V[1](2)+V[1](0)*v.V[0](0)+V[1](1)*v.V[0](1)+V[1](2)*v.V[0](2));
}

C6X6Matrix C6Vector::getSpatialCross() const
{ // Not very efficiently implemented. Rewrite later!
    C6X6Matrix retM;
    retM.M[0][1].clear();
    retM.M[0][0]=V[0].getProductWithStar();
    retM.M[1][0]=V[1].getProductWithStar();
    retM.M[1][1]=V[0].getProductWithStar();
    return(retM);
}

void C6Vector::operator*= (simMathReal d)
{
    V[0]*=d;
    V[1]*=d;
}

void C6Vector::operator/= (simMathReal d)
{
    V[0]/=d;
    V[1]/=d;
}

void C6Vector::operator+= (const C6Vector& v)
{
    V[0]+=v.V[0];
    V[1]+=v.V[1];
}

void C6Vector::operator-= (const C6Vector& v)
{
    V[0]-=v.V[0];
    V[1]-=v.V[1];
}
