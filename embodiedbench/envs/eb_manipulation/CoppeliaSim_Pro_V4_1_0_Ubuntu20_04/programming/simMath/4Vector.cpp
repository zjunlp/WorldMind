#include "4Vector.h"
#include "MyMath.h"

C4Vector::C4Vector()
{
}

C4Vector::C4Vector(simMathReal v0,simMathReal v1,simMathReal v2,simMathReal v3)
{
    data[0]=v0;
    data[1]=v1;
    data[2]=v2;
    data[3]=v3;
    // We don't normalize here
}

C4Vector::C4Vector(const simMathReal v[4])
{
    data[0]=v[0];
    data[1]=v[1];
    data[2]=v[2];
    data[3]=v[3];
    // We don't normalize here
}

C4Vector::C4Vector(const C4Vector& q)
{
    data[0]=q(0);
    data[1]=q(1);
    data[2]=q(2);
    data[3]=q(3);
    // We don't normalize here
}

C4Vector::C4Vector(const C3Vector& v)
{ // Alpha, beta and gamma are in radians!
    setEulerAngles(v);
}

C4Vector::C4Vector(simMathReal a,simMathReal b,simMathReal g)
{ // Alpha, beta and gamma are in radians!
    setEulerAngles(a,b,g);
}

C4Vector::C4Vector(simMathReal angle,const C3Vector& axis)
{ // Builds a rotation quaternion around axis (angle in radian!)
    setAngleAndAxis(angle,axis);
}

C4Vector::C4Vector(const C3Vector& startV,const C3Vector& endV)
{
    setVectorMapping(startV,endV);
}

C4Vector::~C4Vector()
{

}

void C4Vector::setEulerAngles(simMathReal a,simMathReal b,simMathReal g)
{ // a,b anf g are in radian!
    C4Vector vx(a,C3Vector(simOne,simZero,simZero));
    C4Vector vy(b,C3Vector(simZero,simOne,simZero));
    C4Vector vz(g,C3Vector(simZero,simZero,simOne));
    (*this)=vx*vy*vz;
}

void C4Vector::setEulerAngles(const C3Vector& v)
{ // v(0), v(1) and v(2) are in radian!
    setEulerAngles(v(0),v(1),v(2));
}

void C4Vector::setAngleAndAxis(simMathReal angle,const C3Vector& axis)
{ // angle in radian!
    C3Vector axisTmp=axis;
    axisTmp.normalize();
    simMathReal sinA=sin(angle/simTwo);
    data[1]=axisTmp(0)*sinA;
    data[2]=axisTmp(1)*sinA;
    data[3]=axisTmp(2)*sinA;
    data[0]=cos(angle/simTwo);
}

void C4Vector::setVectorMapping(const C3Vector& startV,const C3Vector& endV)
{
    C3Vector v0(startV.getNormalized());
    C3Vector v1(endV.getNormalized());
    C3Vector cross(v0^v1);
    simMathReal cosAngle=v0*v1;
    if (cosAngle>simOne)
        setIdentity();
    else
        setAngleAndAxis(CMath::robustAcos(cosAngle),cross);
}

C4Vector C4Vector::getAngleAndAxis() const
{ // Returned vector is (angle,x,y,z) (angle is in radians)
    C4Vector retV;
    C4Vector d(*this);
    if (d(0)<simZero)  // Condition added on 2009/02/26
        d=d*-simOne;
    simMathReal l=sqrt(d(0)*d(0)+d(1)*d(1)+d(2)*d(2)+d(3)*d(3));
    simMathReal cosA=d(0)/l; // Quaternion needs to be normalized
    if (cosA>simOne) // Just make sure..
        cosA=simOne;
    retV(0)=CMath::robustAcos(cosA)*simTwo;
    simMathReal sinA=sqrt(simOne-cosA*cosA);
    if (fabs(sinA)<simMathReal(0.00005))
        sinA=simOne;
    else
        sinA*=l; // Quaternion needs to be normalized
    retV(1)=d(1)/sinA;
    retV(2)=d(2)/sinA;  
    retV(3)=d(3)/sinA;
    return(retV);
}

C4Vector C4Vector::getAngleAndAxisNoChecking() const
{ // Returned vector is (angle,x,y,z) (angle is in radians)
    C4Vector retV;
    C4Vector d(*this);
    if (d(0)<simZero)  // Condition added on 2009/02/26
        d=d*-simOne;
    simMathReal l=sqrt(d(0)*d(0)+d(1)*d(1)+d(2)*d(2)+d(3)*d(3));
    simMathReal cosA=d(0)/l; // Quaternion needs to be normalized
    if (cosA>simOne) // Just make sure..
        cosA=simOne;
    retV(0)=acos(cosA)*simTwo;
    simMathReal sinA=sqrt(simOne-cosA*cosA);
    if (fabs(sinA)<simMathReal(0.00005))
        sinA=simOne;
    else
        sinA*=l; // Quaternion needs to be normalized
    retV(1)=d(1)/sinA;
    retV(2)=d(2)/sinA;  
    retV(3)=d(3)/sinA;
    return(retV);
}


C3Vector C4Vector::getEulerAngles() const
{ // angles are in radians!
    return(getMatrix().getEulerAngles());
}


simMathReal C4Vector::getAngleBetweenQuaternions(const C4Vector& q) const
{
    simMathReal angle=fabs(data[0]*q(0)+data[1]*q(1)+data[2]*q(2)+data[3]*q(3));
    return(CMath::robustAcos(angle)*simTwo);
}

void C4Vector::buildInterpolation(const C4Vector& fromThis,const C4Vector& toThat,simMathReal t)
{
    C4Vector AA(fromThis);
    C4Vector BB(toThat);
    if (AA(0)*BB(0)+AA(1)*BB(1)+AA(2)*BB(2)+AA(3)*BB(3)<simZero)
        AA=AA*-simOne;
    C4Vector r((AA.getInverse()*BB).getAngleAndAxis());
    (*this)=(AA*C4Vector(r(0)*t,C3Vector(r(1),r(2),r(3))));
    // Already normalized through * operator
}

void C4Vector::buildInterpolation_otherWayRound(const C4Vector& fromThis,const C4Vector& toThat,simMathReal t)
{
    C4Vector AA(fromThis);
    C4Vector BB(toThat);
    if (AA(0)*BB(0)+AA(1)*BB(1)+AA(2)*BB(2)+AA(3)*BB(3)<simZero)
        AA=AA*-simOne;
    C4Vector r((AA.getInverse()*BB).getAngleAndAxis());

    // r(0) is the rotation angle
    // r(1),r(2),r(3) is the rotation axis
    // Here, since we want to rotate the other way round, we inverse the axis and rotate by 2*pi-r(0) instead:
    (*this)=(AA*C4Vector((piValTimes2-r(0))*t,C3Vector(r(1)*-simOne,r(2)*-simOne,r(3)*-simOne)));
    // Already normalized through * operator
}

void C4Vector::buildRandomOrientation()
{
    C3Vector u(SIM_RAND_FLOAT,SIM_RAND_FLOAT,SIM_RAND_FLOAT);
    data[0]=sqrt(simOne-u(0))*sin(piValTimes2*u(1));
    data[1]=sqrt(simOne-u(0))*cos(piValTimes2*u(1));
    data[2]=sqrt(u(0))*sin(piValTimes2*u(2));
    data[3]=sqrt(u(0))*cos(piValTimes2*u(2));
}

const C4Vector C4Vector::identityRotation(simOne,simZero,simZero,simZero);
