#include "7Vector.h"
#include "4X4Matrix.h"

C7Vector::C7Vector()
{
}

C7Vector::C7Vector(const C7Vector& v)
{
    (*this)=v;
}

C7Vector::C7Vector(const C4Vector& q)
{
    Q=q;
    X.clear();
}

C7Vector::C7Vector(const C3Vector& x)
{
    X=x;
    Q.setIdentity();
}

C7Vector::C7Vector(const C4Vector& q,const C3Vector& x)
{
    Q=q;
    X=x;
}

C7Vector::C7Vector(const simMathReal m[4][4])
{
    set(m);
}

C7Vector::C7Vector(const C4X4Matrix& m)
{
    set(m);
}

C7Vector::C7Vector(simMathReal angle,const C3Vector& pos,const C3Vector& dir)
{ // Builds a rotation around dir at position pos of angle angle (in radians)
    C7Vector shift1;
    shift1.setIdentity();
    shift1.X(0)=-pos(0);
    shift1.X(1)=-pos(1);
    shift1.X(2)=-pos(2);
    C7Vector shift2;
    shift2.setIdentity();
    shift2.X=pos;
    C7Vector rot;
    rot.setIdentity();
    rot.Q.setAngleAndAxis(angle,dir);
    (*this)=shift2*rot*shift1;
}


C7Vector::~C7Vector()
{

}

void C7Vector::set(simMathReal m[4][4])
{
    C4X4Matrix tr(m);
    (*this)=tr.getTransformation();
}

void C7Vector::set(const C4X4Matrix& m)
{
    (*this)=m.getTransformation();
}

C3Vector C7Vector::getAxis(size_t index) const
{
    return(Q.getAxis(index));
}

void C7Vector::setIdentity()
{
    Q.setIdentity();
    X.clear();
}

C4X4Matrix C7Vector::getMatrix() const
{
    return(C4X4Matrix(Q.getMatrix(),X));
}
void C7Vector::copyTo(simMathReal m[4][4]) const
{ // Temporary routine. Remove later!
    C4X4Matrix tmp(getMatrix());
    for (size_t i=0;i<3;i++)
    {
        for (size_t j=0;j<3;j++)
            m[i][j]=tmp.M(i,j);
        m[i][3]=tmp.X(i);
    }
    m[3][0]=simZero;
    m[3][1]=simZero;
    m[3][2]=simZero;
    m[3][3]=simOne;
}

C7Vector& C7Vector::operator= (const C7Vector& v)
{
    Q=v.Q;
    X=v.X;
    return(*this);
}

void C7Vector::setMultResult(const C7Vector& v1,const C7Vector& v2)
{
    X=v1.X+(v1.Q*v2.X);
    Q=v1.Q*v2.Q;
}

C7Vector C7Vector::operator* (const C7Vector& v) const
{ // Transformation multiplication
    C7Vector retV;
    retV.X=X+(Q*v.X);
    retV.Q=Q*v.Q;
    retV.Q.normalize();
    return(retV);
}

void C7Vector::operator*= (const C7Vector& v)
{ 
    X+=(Q*v.X);
    Q*=v.Q;
}

C3Vector C7Vector::operator* (const C3Vector& v) const
{ // Vector transformation
    return(X+(Q*v)); 
}

void C7Vector::inverse()
{
    (*this)=getInverse();
}

C7Vector C7Vector::getInverse() const
{
    C7Vector retV;
    retV.Q=Q.getInverse();
    retV.X=(retV.Q*X)*-simOne;
    return(retV);
}

void C7Vector::buildInterpolation(const C7Vector& fromThis,const C7Vector& toThat,simMathReal t)
{   // Builds the interpolation (based on t) from 'fromThis' to 'toThat'
    Q.buildInterpolation(fromThis.Q,toThat.Q,t);
    X.buildInterpolation(fromThis.X,toThat.X,t);
}

const C7Vector C7Vector::identityTransformation(C4Vector(simOne,simZero,simZero,simZero),C3Vector(simZero,simZero,simZero));
