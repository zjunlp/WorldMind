#include "MyMath.h"
#include "4X4FullMatrix.h"

C4X4FullMatrix::C4X4FullMatrix()
{
}

C4X4FullMatrix::C4X4FullMatrix(const C4X4Matrix& m)
{
    (*this)=m;
}

C4X4FullMatrix::C4X4FullMatrix(const C4X4FullMatrix& m)
{
    (*this)=m;
}
 
C4X4FullMatrix::~C4X4FullMatrix()
{
} 

void C4X4FullMatrix::clear()
{
    data[0][0]=simZero;
    data[1][0]=simZero;
    data[2][0]=simZero;
    data[3][0]=simZero;
    data[0][1]=simZero;
    data[1][1]=simZero;
    data[2][1]=simZero;
    data[3][1]=simZero;
    data[0][2]=simZero;
    data[1][2]=simZero;
    data[2][2]=simZero;
    data[3][2]=simZero;
    data[0][3]=simZero;
    data[1][3]=simZero;
    data[2][3]=simZero;
    data[3][3]=simZero;
}

void C4X4FullMatrix::setIdentity()
{
    data[0][0]=simOne;
    data[1][0]=simZero;
    data[2][0]=simZero;
    data[3][0]=simZero;
    data[0][1]=simZero;
    data[1][1]=simOne;
    data[2][1]=simZero;
    data[3][1]=simZero;
    data[0][2]=simZero;
    data[1][2]=simZero;
    data[2][2]=simOne;
    data[3][2]=simZero;
    data[0][3]=simZero;
    data[1][3]=simZero;
    data[2][3]=simZero;
    data[3][3]=simOne;
}

C4X4FullMatrix C4X4FullMatrix::operator* (const C4X4FullMatrix& m) const
{
    C4X4FullMatrix retM;
    retM(0,0)=data[0][0]*m(0,0)+data[0][1]*m(1,0)+data[0][2]*m(2,0)+data[0][3]*m(3,0);
    retM(1,0)=data[1][0]*m(0,0)+data[1][1]*m(1,0)+data[1][2]*m(2,0)+data[1][3]*m(3,0);
    retM(2,0)=data[2][0]*m(0,0)+data[2][1]*m(1,0)+data[2][2]*m(2,0)+data[2][3]*m(3,0);
    retM(3,0)=data[3][0]*m(0,0)+data[3][1]*m(1,0)+data[3][2]*m(2,0)+data[3][3]*m(3,0);

    retM(0,1)=data[0][0]*m(0,1)+data[0][1]*m(1,1)+data[0][2]*m(2,1)+data[0][3]*m(3,1);
    retM(1,1)=data[1][0]*m(0,1)+data[1][1]*m(1,1)+data[1][2]*m(2,1)+data[1][3]*m(3,1);
    retM(2,1)=data[2][0]*m(0,1)+data[2][1]*m(1,1)+data[2][2]*m(2,1)+data[2][3]*m(3,1);
    retM(3,1)=data[3][0]*m(0,1)+data[3][1]*m(1,1)+data[3][2]*m(2,1)+data[3][3]*m(3,1);

    retM(0,2)=data[0][0]*m(0,2)+data[0][1]*m(1,2)+data[0][2]*m(2,2)+data[0][3]*m(3,2);
    retM(1,2)=data[1][0]*m(0,2)+data[1][1]*m(1,2)+data[1][2]*m(2,2)+data[1][3]*m(3,2);
    retM(2,2)=data[2][0]*m(0,2)+data[2][1]*m(1,2)+data[2][2]*m(2,2)+data[2][3]*m(3,2);
    retM(3,2)=data[3][0]*m(0,2)+data[3][1]*m(1,2)+data[3][2]*m(2,2)+data[3][3]*m(3,2);

    retM(0,3)=data[0][0]*m(0,3)+data[0][1]*m(1,3)+data[0][2]*m(2,3)+data[0][3]*m(3,3);
    retM(1,3)=data[1][0]*m(0,3)+data[1][1]*m(1,3)+data[1][2]*m(2,3)+data[1][3]*m(3,3);
    retM(2,3)=data[2][0]*m(0,3)+data[2][1]*m(1,3)+data[2][2]*m(2,3)+data[2][3]*m(3,3);
    retM(3,3)=data[3][0]*m(0,3)+data[3][1]*m(1,3)+data[3][2]*m(2,3)+data[3][3]*m(3,3);

    return(retM);
}

C4X4FullMatrix C4X4FullMatrix::operator+ (const C4X4FullMatrix& m) const
{
    C4X4FullMatrix retM;
    retM(0,0)=data[0][0]+m(0,0);
    retM(1,0)=data[1][0]+m(1,0);
    retM(2,0)=data[2][0]+m(2,0);
    retM(3,0)=data[3][0]+m(3,0);
    retM(0,1)=data[0][1]+m(0,1);
    retM(1,1)=data[1][1]+m(1,1);
    retM(2,1)=data[2][1]+m(2,1);
    retM(3,1)=data[3][1]+m(3,1);
    retM(0,2)=data[0][2]+m(0,2);
    retM(1,2)=data[1][2]+m(1,2);
    retM(2,2)=data[2][2]+m(2,2);
    retM(3,2)=data[3][2]+m(3,2);
    retM(0,3)=data[0][3]+m(0,3);
    retM(1,3)=data[1][3]+m(1,3);
    retM(2,3)=data[2][3]+m(2,3);
    retM(3,3)=data[3][3]+m(3,3);
    return(retM);
}

C4X4FullMatrix C4X4FullMatrix::operator- (const C4X4FullMatrix& m) const
{
    C4X4FullMatrix retM;
    retM(0,0)=data[0][0]-m(0,0);
    retM(1,0)=data[1][0]-m(1,0);
    retM(2,0)=data[2][0]-m(2,0);
    retM(3,0)=data[3][0]-m(3,0);
    retM(0,1)=data[0][1]-m(0,1);
    retM(1,1)=data[1][1]-m(1,1);
    retM(2,1)=data[2][1]-m(2,1);
    retM(3,1)=data[3][1]-m(3,1);
    retM(0,2)=data[0][2]-m(0,2);
    retM(1,2)=data[1][2]-m(1,2);
    retM(2,2)=data[2][2]-m(2,2);
    retM(3,2)=data[3][2]-m(3,2);
    retM(0,3)=data[0][3]-m(0,3);
    retM(1,3)=data[1][3]-m(1,3);
    retM(2,3)=data[2][3]-m(2,3);
    retM(3,3)=data[3][3]-m(3,3);
    return(retM);
}

C4X4FullMatrix C4X4FullMatrix::operator* (simMathReal d) const
{
    C4X4FullMatrix retM;
    retM(0,0)=data[0][0]*d;
    retM(1,0)=data[1][0]*d;
    retM(2,0)=data[2][0]*d;
    retM(3,0)=data[3][0]*d;
    retM(0,1)=data[0][1]*d;
    retM(1,1)=data[1][1]*d;
    retM(2,1)=data[2][1]*d;
    retM(3,1)=data[3][1]*d;
    retM(0,2)=data[0][2]*d;
    retM(1,2)=data[1][2]*d;
    retM(2,2)=data[2][2]*d;
    retM(3,2)=data[3][2]*d;
    retM(0,3)=data[0][3]*d;
    retM(1,3)=data[1][3]*d;
    retM(2,3)=data[2][3]*d;
    retM(3,3)=data[3][3]*d;
    return(retM);
}

C4X4FullMatrix C4X4FullMatrix::operator/ (simMathReal d) const
{
    C4X4FullMatrix retM;
    retM(0,0)=data[0][0]/d;
    retM(1,0)=data[1][0]/d;
    retM(2,0)=data[2][0]/d;
    retM(3,0)=data[3][0]/d;
    retM(0,1)=data[0][1]/d;
    retM(1,1)=data[1][1]/d;
    retM(2,1)=data[2][1]/d;
    retM(3,1)=data[3][1]/d;
    retM(0,2)=data[0][2]/d;
    retM(1,2)=data[1][2]/d;
    retM(2,2)=data[2][2]/d;
    retM(3,2)=data[3][2]/d;
    retM(0,3)=data[0][3]/d;
    retM(1,3)=data[1][3]/d;
    retM(2,3)=data[2][3]/d;
    retM(3,3)=data[3][3]/d;
    return(retM);
}

void C4X4FullMatrix::operator*= (const C4X4FullMatrix& m)
{
    (*this)=(*this)*m;
}

void C4X4FullMatrix::operator+= (const C4X4FullMatrix& m)
{
    data[0][0]+=m(0,0);
    data[1][0]+=m(1,0);
    data[2][0]+=m(2,0);
    data[3][0]+=m(3,0);
    data[0][1]+=m(0,1);
    data[1][1]+=m(1,1);
    data[2][1]+=m(2,1);
    data[3][1]+=m(3,1);
    data[0][2]+=m(0,2);
    data[1][2]+=m(1,2);
    data[2][2]+=m(2,2);
    data[3][2]+=m(3,2);
    data[0][3]+=m(0,3);
    data[1][3]+=m(1,3);
    data[2][3]+=m(2,3);
    data[3][3]+=m(3,3);
}

void C4X4FullMatrix::operator-= (const C4X4FullMatrix& m)
{
    data[0][0]-=m(0,0);
    data[1][0]-=m(1,0);
    data[2][0]-=m(2,0);
    data[3][0]-=m(3,0);
    data[0][1]-=m(0,1);
    data[1][1]-=m(1,1);
    data[2][1]-=m(2,1);
    data[3][1]-=m(3,1);
    data[0][2]-=m(0,2);
    data[1][2]-=m(1,2);
    data[2][2]-=m(2,2);
    data[3][2]-=m(3,2);
    data[0][3]-=m(0,3);
    data[1][3]-=m(1,3);
    data[2][3]-=m(2,3);
    data[3][3]-=m(3,3);
}

void C4X4FullMatrix::operator*= (simMathReal d)
{
    data[0][0]*=d;
    data[1][0]*=d;
    data[2][0]*=d;
    data[3][0]*=d;
    data[0][1]*=d;
    data[1][1]*=d;
    data[2][1]*=d;
    data[3][1]*=d;
    data[0][2]*=d;
    data[1][2]*=d;
    data[2][2]*=d;
    data[3][2]*=d;
    data[0][3]*=d;
    data[1][3]*=d;
    data[2][3]*=d;
    data[3][3]*=d;
}

void C4X4FullMatrix::operator/= (simMathReal d)
{
    data[0][0]/=d;
    data[1][0]/=d;
    data[2][0]/=d;
    data[3][0]/=d;
    data[0][1]/=d;
    data[1][1]/=d;
    data[2][1]/=d;
    data[3][1]/=d;
    data[0][2]/=d;
    data[1][2]/=d;
    data[2][2]/=d;
    data[3][2]/=d;
    data[0][3]/=d;
    data[1][3]/=d;
    data[2][3]/=d;
    data[3][3]/=d;
}

C4X4FullMatrix& C4X4FullMatrix::operator= (const C4X4Matrix& m)
{
    data[0][0]=m.M.axis[0](0);
    data[1][0]=m.M.axis[0](1);
    data[2][0]=m.M.axis[0](2);
    data[3][0]=simZero;

    data[0][1]=m.M.axis[1](0);
    data[1][1]=m.M.axis[1](1);
    data[2][1]=m.M.axis[1](2);
    data[3][1]=simZero;

    data[0][2]=m.M.axis[2](0);
    data[1][2]=m.M.axis[2](1);
    data[2][2]=m.M.axis[2](2);
    data[3][2]=simZero;

    data[0][3]=m.X(0);
    data[1][3]=m.X(1);
    data[2][3]=m.X(2);
    data[3][3]=simOne;

    return(*this);
}

C4X4FullMatrix& C4X4FullMatrix::operator= (const C4X4FullMatrix& m)
{
    data[0][0]=m(0,0);
    data[1][0]=m(1,0);
    data[2][0]=m(2,0);
    data[3][0]=m(3,0);
    data[0][1]=m(0,1);
    data[1][1]=m(1,1);
    data[2][1]=m(2,1);
    data[3][1]=m(3,1);
    data[0][2]=m(0,2);
    data[1][2]=m(1,2);
    data[2][2]=m(2,2);
    data[3][2]=m(3,2);
    data[0][3]=m(0,3);
    data[1][3]=m(1,3);
    data[2][3]=m(2,3);
    data[3][3]=m(3,3);
    return(*this);
}

void C4X4FullMatrix::invert()
{ // Write a faster routine later!  
    C4X4FullMatrix n;

    n(0,0)=data[0][0];
    n(1,0)=data[0][1];
    n(2,0)=data[0][2];
    n(3,0)=data[0][3];

    n(0,1)=data[1][0];
    n(1,1)=data[1][1];
    n(2,1)=data[1][2];
    n(3,1)=data[1][3];

    n(0,2)=data[2][0];
    n(1,2)=data[2][1];
    n(2,2)=data[2][2];
    n(3,2)=data[2][3];

    n(0,3)=data[3][0];
    n(1,3)=data[3][1];
    n(2,3)=data[3][2];
    n(3,3)=data[3][3];

    (*this)=n;
}

void C4X4FullMatrix::buildZRotation(simMathReal angle)
{
    simMathReal c=cos(angle);
    simMathReal s=sin(angle);
    data[0][0]=c;
    data[0][1]=-s;
    data[0][2]=simZero;
    data[0][3]=simZero;
    data[1][0]=s;
    data[1][1]=c;
    data[1][2]=simZero;
    data[1][3]=simZero;
    data[2][0]=simZero;
    data[2][1]=simZero;
    data[2][2]=simOne;
    data[2][3]=simZero;
    data[3][0]=simZero;
    data[3][1]=simZero;
    data[3][2]=simZero;
    data[3][3]=simOne;
}

void C4X4FullMatrix::buildTranslation(simMathReal x, simMathReal y, simMathReal z)
{
    data[0][0]=simOne;
    data[0][1]=simZero;
    data[0][2]=simZero;
    data[0][3]=x;
    data[1][0]=simZero;
    data[1][1]=simOne;
    data[1][2]=simZero;
    data[1][3]=y;
    data[2][0]=simZero;
    data[2][1]=simZero;
    data[2][2]=simOne;
    data[2][3]=z;
    data[3][0]=simZero;
    data[3][1]=simZero;
    data[3][2]=simZero;
    data[3][3]=simOne;
}

C3Vector C4X4FullMatrix::getEulerAngles() const
{ // Angles are in radians!! // THERE IS ANOTHER SUCH ROUTINE IN C4X4MATRIX
    C3Vector retV;
    simMathReal m02=data[0][2];
    if (m02>simOne)
        m02=simOne;   // Just in case
    if (m02<-simOne)
        m02=-simOne;  // Just in case

    retV(1)=CMath::robustAsin(m02);
    if (m02<simZero)
        m02=-m02;
    if (m02<simMathReal(0.999995))
    {   // No gimbal lock
        retV(0)=atan2(-data[1][2],data[2][2]);
        retV(2)=atan2(-data[0][1],data[0][0]);
    }
    else
    {   // Gimbal lock has occured
        retV(0)=simZero;
        retV(2)=atan2(data[1][0],data[1][1]);
    }
    return(retV);
}
