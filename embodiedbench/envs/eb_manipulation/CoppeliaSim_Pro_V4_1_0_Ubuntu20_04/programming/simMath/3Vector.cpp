#include "MyMath.h"
#include "3Vector.h"
#include "3X3Matrix.h"
#include "4X4Matrix.h"
#include "7Vector.h"

C3Vector::C3Vector()
{
}

C3Vector::C3Vector(simMathReal v0,simMathReal v1,simMathReal v2)
{
    data[0]=v0;
    data[1]=v1;
    data[2]=v2;
}

C3Vector::C3Vector(const simMathReal v[3])
{
    data[0]=v[0];
    data[1]=v[1];
    data[2]=v[2];
}

C3Vector::C3Vector(const C3Vector& v)
{
    (*this)=v;
}

C3Vector::~C3Vector()
{
}

simMathReal C3Vector::getAngle(const C3Vector& v) const
{ // Return value is in radian!!
    C3Vector a(getNormalized());
    C3Vector b(v.getNormalized());
    return(CMath::robustAcos(a*b));
}


C3X3Matrix C3Vector::getProductWithStar() const
{
    C3X3Matrix retM;
    retM(0,0)=simZero;
    retM(0,1)=-data[2];
    retM(0,2)=data[1];
    retM(1,0)=data[2];
    retM(1,1)=simZero;
    retM(1,2)=-data[0];
    retM(2,0)=-data[1];
    retM(2,1)=data[0];
    retM(2,2)=simZero;
    return(retM);
}

void C3Vector::operator*= (const C4X4Matrix& m)
{
//  (*this)=m*(*this);
    simMathReal x=data[0];
    simMathReal y=data[1];
    simMathReal z=data[2];
    data[0]=m.M.axis[0].data[0]*x+m.M.axis[1].data[0]*y+m.M.axis[2].data[0]*z+m.X.data[0];
    data[1]=m.M.axis[0].data[1]*x+m.M.axis[1].data[1]*y+m.M.axis[2].data[1]*z+m.X.data[1];
    data[2]=m.M.axis[0].data[2]*x+m.M.axis[1].data[2]*y+m.M.axis[2].data[2]*z+m.X.data[2];
}
void C3Vector::operator*= (const C3X3Matrix& m)
{
//  (*this)=m*(*this);
    simMathReal x=data[0];
    simMathReal y=data[1];
    simMathReal z=data[2];
    data[0]=m.axis[0].data[0]*x+m.axis[1].data[0]*y+m.axis[2].data[0]*z;
    data[1]=m.axis[0].data[1]*x+m.axis[1].data[1]*y+m.axis[2].data[1]*z;
    data[2]=m.axis[0].data[2]*x+m.axis[1].data[2]*y+m.axis[2].data[2]*z;
}


void C3Vector::operator*= (const C7Vector& transf)
{
    (*this)=transf*(*this);
}

void C3Vector::buildInterpolation(const C3Vector& fromThis,const C3Vector& toThat,simMathReal t)
{
    (*this)=fromThis+((toThat-fromThis)*t);
}

const C3Vector C3Vector::oneOneOneVector(simOne,simOne,simOne);
const C3Vector C3Vector::unitXVector(simOne,simZero,simZero);
const C3Vector C3Vector::unitYVector(simZero,simOne,simZero);
const C3Vector C3Vector::unitZVector(simZero,simZero,simOne);
const C3Vector C3Vector::zeroVector(simZero,simZero,simZero);
