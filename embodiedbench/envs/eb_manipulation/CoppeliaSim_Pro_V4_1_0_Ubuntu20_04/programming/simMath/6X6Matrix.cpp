#include "6X6Matrix.h"

C6X6Matrix::C6X6Matrix()
{
}

C6X6Matrix::~C6X6Matrix()
{
}

C6X6Matrix::C6X6Matrix(const C6X6Matrix& m)
{
    M[0][0]=m.M[0][0];
    M[0][1]=m.M[0][1];
    M[1][0]=m.M[1][0];
    M[1][1]=m.M[1][1];
}

C6X6Matrix::C6X6Matrix(const C3X3Matrix& m00,const C3X3Matrix& m01,const C3X3Matrix& m10,const C3X3Matrix& m11)
{
    M[0][0]=m00;
    M[0][1]=m01;
    M[1][0]=m10;
    M[1][1]=m11;
}

void C6X6Matrix::clear()
{
    M[0][0].clear();
    M[0][1].clear();
    M[1][0].clear();
    M[1][1].clear();
}
void C6X6Matrix::setIdentity()
{
    M[0][0].setIdentity();
    M[0][1].clear();
    M[1][0].clear();
    M[1][1].setIdentity();
}

C6X6Matrix C6X6Matrix::operator* (const C6X6Matrix& m) const
{
    C6X6Matrix retM;
    retM.M[0][0]=(M[0][0]*m.M[0][0])+(M[0][1]*m.M[1][0]);
    retM.M[0][1]=(M[0][0]*m.M[0][1])+(M[0][1]*m.M[1][1]);
    retM.M[1][0]=(M[1][0]*m.M[0][0])+(M[1][1]*m.M[1][0]);
    retM.M[1][1]=(M[1][0]*m.M[0][1])+(M[1][1]*m.M[1][1]);
    return(retM);   
}

void C6X6Matrix::setMultResult(const C6X6Matrix& m1,const C6X6Matrix& m2)
{
    M[0][0]=(m1.M[0][0]*m2.M[0][0])+(m1.M[0][1]*m2.M[1][0]);
    M[0][1]=(m1.M[0][0]*m2.M[0][1])+(m1.M[0][1]*m2.M[1][1]);
    M[1][0]=(m1.M[1][0]*m2.M[0][0])+(m1.M[1][1]*m2.M[1][0]);
    M[1][1]=(m1.M[1][0]*m2.M[0][1])+(m1.M[1][1]*m2.M[1][1]);
}

C6X6Matrix C6X6Matrix::operator+ (const C6X6Matrix& m) const
{
    C6X6Matrix retM(M[0][0]+m.M[0][0],M[0][1]+m.M[0][1],M[1][0]+m.M[1][0],M[1][1]+m.M[1][1]);
    return(retM);   
}

C6X6Matrix C6X6Matrix::operator- (const C6X6Matrix& m) const
{
    C6X6Matrix retM(M[0][0]-m.M[0][0],M[0][1]-m.M[0][1],M[1][0]-m.M[1][0],M[1][1]-m.M[1][1]);
    return(retM);   
}

C6Vector C6X6Matrix::operator* (const C6Vector& v) const
{
    C6Vector retV;
    retV.V[0]=(M[0][0]*v.V[0])+(M[0][1]*v.V[1]);
    retV.V[1]=(M[1][0]*v.V[0])+(M[1][1]*v.V[1]);
    return(retV);   
}

C6X6Matrix& C6X6Matrix::operator= (const C6X6Matrix& m)
{
    M[0][0]=m.M[0][0];
    M[0][1]=m.M[0][1];
    M[1][0]=m.M[1][0];
    M[1][1]=m.M[1][1];
    return(*this);
}

void C6X6Matrix::operator*= (const C6X6Matrix& m)
{
    (*this)=(*this)*m;
}

void C6X6Matrix::operator+= (const C6X6Matrix& m)
{
    M[0][0]+=m.M[0][0];
    M[0][1]+=m.M[0][1];
    M[1][0]+=m.M[1][0];
    M[1][1]+=m.M[1][1];
}

void C6X6Matrix::operator-= (const C6X6Matrix& m)
{
    M[0][0]-=m.M[0][0];
    M[0][1]-=m.M[0][1];
    M[1][0]-=m.M[1][0];
    M[1][1]-=m.M[1][1];
}
