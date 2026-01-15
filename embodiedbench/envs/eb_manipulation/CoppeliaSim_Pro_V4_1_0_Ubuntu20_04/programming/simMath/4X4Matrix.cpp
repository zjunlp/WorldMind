#include "4X4Matrix.h"
#include "4X4FullMatrix.h"
#include "MMatrix.h"

C4X4Matrix::C4X4Matrix()
{
}

C4X4Matrix::C4X4Matrix(const C3X3Matrix& m,const C3Vector& x)
{
    M=m;
    X=x;
}

C4X4Matrix::C4X4Matrix(const C7Vector& transf)
{
    M=transf.Q.getMatrix();
    X=transf.X;
}

C4X4Matrix::C4X4Matrix(const C4X4Matrix& m)
{
    (*this)=m;
}

C4X4Matrix::C4X4Matrix(const CMatrix& m)
{
    (*this)=m;
}

C4X4Matrix::C4X4Matrix(const simMathReal m[4][4])
{
    for (size_t i=0;i<3;i++)
    {
        for (size_t j=0;j<3;j++)
            M(i,j)=m[i][j];
        X(i)=m[i][3];
    }
}

C4X4Matrix::C4X4Matrix(const C3Vector& x,const C3Vector& y,const C3Vector& z,const C3Vector& pos)
{
    M.set(x,y,z);
    X=pos;
}

C4X4Matrix::~C4X4Matrix()
{

}
void C4X4Matrix::setIdentity()
{
    M.setIdentity();
    X.clear();
}

C4X4Matrix C4X4Matrix::operator* (const CMatrix& m) const
{
    C4X4Matrix retM((*this)*C4X4Matrix(m));
    return(retM);
}

C4X4Matrix& C4X4Matrix::operator= (const CMatrix& m)
{
    M.axis[0](0)=m(0,0);
    M.axis[0](1)=m(1,0);
    M.axis[0](2)=m(2,0);
    M.axis[1](0)=m(0,1);
    M.axis[1](1)=m(1,1);
    M.axis[1](2)=m(2,1);
    M.axis[2](0)=m(0,2);
    M.axis[2](1)=m(1,2);
    M.axis[2](2)=m(2,2);
    X(0)=m(0,3);
    X(1)=m(1,3);
    X(2)=m(2,3);
    return(*this);
}

C4X4Matrix& C4X4Matrix::operator= (const C4X4FullMatrix& m)
{
    M.axis[0](0)=m(0,0);
    M.axis[0](1)=m(1,0);
    M.axis[0](2)=m(2,0);
    M.axis[1](0)=m(0,1);
    M.axis[1](1)=m(1,1);
    M.axis[1](2)=m(2,1);
    M.axis[2](0)=m(0,2);
    M.axis[2](1)=m(1,2);
    M.axis[2](2)=m(2,2);
    X(0)=m(0,3);
    X(1)=m(1,3);
    X(2)=m(2,3);
    return(*this);
}

C7Vector C4X4Matrix::getTransformation() const
{
    return(C7Vector(M.getQuaternion(),X));
}


void C4X4Matrix::buildInterpolation(const C4X4Matrix& fromThis,const C4X4Matrix& toThat,simMathReal t)
{   // Builds the interpolation (based on t) from 'fromThis' to 'toThat'
    C7Vector out;
    out.buildInterpolation(fromThis.getTransformation(),toThat.getTransformation(),t);
    (*this)=out;
}

void C4X4Matrix::rotateAroundX(simMathReal angle)
{
    C4X4Matrix rot;
    rot.setIdentity();
    rot.M.buildXRotation(angle);
    (*this)=rot*(*this);
}

void C4X4Matrix::rotateAroundY(simMathReal angle)
{
    C4X4Matrix rot;
    rot.setIdentity();
    rot.M.buildYRotation(angle);
    (*this)=rot*(*this);
}

void C4X4Matrix::rotateAroundZ(simMathReal angle)
{
    C4X4Matrix rot;
    rot.setIdentity();
    rot.M.buildZRotation(angle);
    (*this)=rot*(*this);
}

void C4X4Matrix::buildXRotation(simMathReal angle)
{
    setIdentity();
    M.buildXRotation(angle);
}

void C4X4Matrix::buildYRotation(simMathReal angle)
{
    setIdentity();
    M.buildYRotation(angle);
}

void C4X4Matrix::buildZRotation(simMathReal angle)
{
    setIdentity();
    M.buildZRotation(angle);
}

void C4X4Matrix::buildTranslation(simMathReal x,simMathReal y,simMathReal z)
{
    setIdentity();
    X(0)=x;
    X(1)=y;
    X(2)=z;
}

void C4X4Matrix::translate(simMathReal x,simMathReal y,simMathReal z)
{
    X(0)+=x;
    X(1)+=y;
    X(2)+=z;
}
