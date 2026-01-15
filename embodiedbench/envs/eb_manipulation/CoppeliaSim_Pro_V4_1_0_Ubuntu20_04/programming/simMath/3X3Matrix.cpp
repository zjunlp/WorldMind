#include "3X3Matrix.h"
#include "4Vector.h"
#include "MyMath.h"

C3X3Matrix::C3X3Matrix()
{

}

C3X3Matrix::C3X3Matrix(const C3Vector& xAxis,const C3Vector& yAxis,const C3Vector& zAxis)
{
    (*this).axis[0]=xAxis;
    (*this).axis[1]=yAxis;
    (*this).axis[2]=zAxis;
}

C3X3Matrix::C3X3Matrix(const C4Vector& q)
{
    (*this)=q.getMatrix();
}

C3X3Matrix::C3X3Matrix(const C3X3Matrix& m)
{
    (*this).axis[0]=m.axis[0];
    (*this).axis[1]=m.axis[1];
    (*this).axis[2]=m.axis[2];
}

C3X3Matrix::~C3X3Matrix()
{

}

void C3X3Matrix::setEulerAngles(simMathReal a,simMathReal b,simMathReal g)
{ // a,b anf g are in radian!
    simMathReal A=cos(a);
    simMathReal B=sin(a);
    simMathReal C=cos(b);
    simMathReal D=sin(b);
    simMathReal E=cos(g);
    simMathReal F=sin(g);
    simMathReal AD=A*D;
    simMathReal BD=B*D;
    axis[0](0)=C*E;
    axis[1](0)=-C*F;
    axis[2](0)=D;
    axis[0](1)=BD*E+A*F;
    axis[1](1)=-BD*F+A*E;
    axis[2](1)=-B*C;
    axis[0](2)=-AD*E+B*F;
    axis[1](2)=AD*F+B*E;
    axis[2](2)=A*C;
}

void C3X3Matrix::setEulerAngles(const C3Vector& v)
{ // v(0), v(1), v(2) are in radian!
    setEulerAngles(v(0),v(1),v(2));
}

C3Vector C3X3Matrix::getEulerAngles() const
{ // Angles are in radians!! // THERE IS ANOTHER SUCH ROUTINE IN C4X4MATRIX
    C3Vector retV;
    simMathReal m02=axis[2](0);
    if (m02>simOne)
        m02=simOne;   // Just in case
    if (m02<-simOne)
        m02=-simOne;  // Just in case
    
    retV(1)=CMath::robustAsin(m02);
    if (m02<simZero)
        m02=-m02;
    if (m02<simMathReal(0.999995))
    {   // No gimbal lock
        retV(0)=atan2(-axis[2](1),axis[2](2));
        retV(2)=atan2(-axis[1](0),axis[0](0));
    }
    else
    {   // Gimbal lock has occured
        retV(0)=simZero;
        retV(2)=atan2(axis[0](1),axis[1](1));
    }
    return(retV);
}

C4Vector C3X3Matrix::getQuaternion() const
{   
    C4Vector retV;
    simMathReal trace=axis[0](0)+axis[1](1)+axis[2](2);
    if (trace>=simZero)    // trace>0.00000001)
    {
        simMathReal s=sqrt(trace+simOne)*simTwo;
        retV(1)=(axis[1](2)-axis[2](1))/s;
        retV(2)=(axis[2](0)-axis[0](2))/s;
        retV(3)=(axis[0](1)-axis[1](0))/s;
        retV(0)=simMathReal(0.25)*s;
    }
    else
    {
        if ((axis[0](0)>=axis[1](1))&&(axis[0](0)>=axis[2](2)))
        {
            simMathReal s=sqrt(simOne+axis[0](0)-axis[1](1)-axis[2](2))*simTwo;
            retV(1)=simMathReal(0.25)*s;
            retV(2)=(axis[0](1)+axis[1](0))/s;
            retV(3)=(axis[2](0)+axis[0](2))/s;
            retV(0)=(axis[1](2)-axis[2](1))/s;
        }
        else
        {
            if ((axis[1](1)>=axis[0](0))&&(axis[1](1)>=axis[2](2)))
            {
                simMathReal s=sqrt(simOne+axis[1](1)-axis[0](0)-axis[2](2))*simTwo;
                retV(1)=(axis[0](1)+axis[1](0))/s;
                retV(2)=simMathReal(0.25)*s;
                retV(3)=(axis[1](2)+axis[2](1))/s;
                retV(0)=(axis[2](0)-axis[0](2))/s;
            }
            else
            {
                simMathReal s=sqrt(simOne+axis[2](2)-axis[0](0)-axis[1](1))*simTwo;
                retV(1)=(axis[2](0)+axis[0](2))/s;
                retV(2)=(axis[1](2)+axis[2](1))/s;
                retV(3)=simMathReal(0.25)*s;
                retV(0)=(axis[0](1)-axis[1](0))/s;
            }
        }
    }
    retV.normalize(); // Really needed?
    return(retV);
}

void C3X3Matrix::buildInterpolation(const C3X3Matrix& fromThis,const C3X3Matrix& toThat,simMathReal t)
{   // Builds the interpolation (based on t) from 'fromThis' to 'toThat'
    C4Vector out;
    out.buildInterpolation(fromThis.getQuaternion(),toThat.getQuaternion(),t);
    (*this)=out;
}

void C3X3Matrix::buildXRotation(simMathReal angle)
{
    simMathReal c=cos(angle);
    simMathReal s=sin(angle);
    axis[0](0)=simOne;
    axis[1](0)=simZero;
    axis[2](0)=simZero;
    axis[0](1)=simZero;
    axis[1](1)=c;
    axis[2](1)=-s;
    axis[0](2)=simZero;
    axis[1](2)=s;
    axis[2](2)=c;
}

void C3X3Matrix::buildYRotation(simMathReal angle)
{
    simMathReal c=cos(angle);
    simMathReal s=sin(angle);
    axis[0](0)=c;
    axis[1](0)=simZero;
    axis[2](0)=s;
    axis[0](1)=simZero;
    axis[1](1)=simOne;
    axis[2](1)=simZero;
    axis[0](2)=-s;
    axis[1](2)=simZero;
    axis[2](2)=c;
}

void C3X3Matrix::buildZRotation(simMathReal angle)
{
    simMathReal c=cos(angle);
    simMathReal s=sin(angle);
    axis[0](0)=c;
    axis[1](0)=-s;
    axis[2](0)=simZero;
    axis[0](1)=s;
    axis[1](1)=c;
    axis[2](1)=simZero;
    axis[0](2)=simZero;
    axis[1](2)=simZero;
    axis[2](2)=simOne;
}

C3Vector C3X3Matrix::getNormalVector() const
{ // returns the normal vector to the plane described by axis[0],axis[1],axis[2]
    C3Vector v0(axis[0]-axis[1]);
    C3Vector v1(axis[1]-axis[2]);
    return((v0^v1).getNormalized());
}
