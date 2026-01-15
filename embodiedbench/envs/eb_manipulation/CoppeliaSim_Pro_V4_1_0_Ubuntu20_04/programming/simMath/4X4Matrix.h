#pragma once

#include "mathDefines.h"
#include "3Vector.h"
#include "4Vector.h"
#include "7Vector.h"
#include "3X3Matrix.h"

class CMatrix;
class C4X4FullMatrix;

class C4X4Matrix  
{
public:

    C4X4Matrix();
    C4X4Matrix(const C4X4Matrix& m);
    C4X4Matrix(const CMatrix& m);
    C4X4Matrix(const simMathReal m[4][4]);
    C4X4Matrix(const C3X3Matrix& m,const C3Vector& x);
    C4X4Matrix(const C3Vector& x,const C3Vector& y,const C3Vector& z,const C3Vector& pos);
    C4X4Matrix(const C7Vector& transf);
    ~C4X4Matrix();

    void setIdentity();
    void buildInterpolation(const C4X4Matrix& fromThis,const C4X4Matrix& toThat,simMathReal t);
    C7Vector getTransformation() const;
    void buildXRotation(simMathReal angle);
    void buildYRotation(simMathReal angle);
    void buildZRotation(simMathReal angle);
    void buildTranslation(simMathReal x,simMathReal y,simMathReal z);
    void rotateAroundX(simMathReal angle);
    void rotateAroundY(simMathReal angle);
    void rotateAroundZ(simMathReal angle);
    void translate(simMathReal x,simMathReal y,simMathReal z);

    C4X4Matrix operator* (const CMatrix& m) const;
    C4X4Matrix& operator= (const CMatrix& m);
    C4X4Matrix& operator= (const C4X4FullMatrix& m);

    inline void getInternalData(simMathReal d[12]) const
    { // temporary routine, remove later
        X.getInternalData(d+0);
        M.getInternalData(d+3);
    }
    inline void setInternalData(const simMathReal d[12])
    {
        X.setInternalData(d+0);
        M.setInternalData(d+3);
    }
    inline void inverse()
    {
        // Speed optimized version:
        simMathReal tmp=M.axis[0](1);
        M.axis[0](1)=M.axis[1](0);
        M.axis[1](0)=tmp;;
        tmp=M.axis[0](2);
        M.axis[0](2)=M.axis[2](0);
        M.axis[2](0)=tmp;;
        tmp=M.axis[1](2);
        M.axis[1](2)=M.axis[2](1);
        M.axis[2](1)=tmp;;
        simMathReal v[3]={-X(0),-X(1),-X(2)};
        X(0)=M.axis[0](0)*v[0]+M.axis[1](0)*v[1]+M.axis[2](0)*v[2];
        X(1)=M.axis[0](1)*v[0]+M.axis[1](1)*v[1]+M.axis[2](1)*v[2];
        X(2)=M.axis[0](2)*v[0]+M.axis[1](2)*v[1]+M.axis[2](2)*v[2];
        // Normal version:
        //  (*this)=getInverse();
    }
    inline void copyTo(simMathReal m[4][4]) const
    {
        for (size_t i=0;i<3;i++)
        {
            for (size_t j=0;j<3;j++)
                m[i][j]=M(i,j);
            m[i][3]=X(i);
        }
        m[3][0]=simZero;
        m[3][1]=simZero;
        m[3][2]=simZero;
        m[3][3]=simOne;
    }
    inline void set(simMathReal m[4][4])
    {
        for (size_t i=0;i<3;i++)
        {
            for (size_t j=0;j<3;j++)
                M.axis[j](i)=m[i][j];
            X(i)=m[i][3];
        }
    }
    inline void copyToInterface(simMathReal* m) const
    { // Temporary routine. Remove later!
        for (size_t i=0;i<3;i++)
        {
            m[4*i+0]=M.axis[0](i);
            m[4*i+1]=M.axis[1](i);
            m[4*i+2]=M.axis[2](i);
            m[4*i+3]=X(i);
        }
    }
    inline void copyFromInterface(const simMathReal* m)
    { // Temporary routine. Remove later!
        for (size_t i=0;i<3;i++)
        {
            M.axis[0](i)=m[4*i+0];
            M.axis[1](i)=m[4*i+1];
            M.axis[2](i)=m[4*i+2];
            X(i)=m[4*i+3];
        }
    }
    inline bool isValid() const
    {
        if (!M.isValid())
            return(false);
        if (!X.isValid())
            return(false);
        return(true);
    }
    inline C4X4Matrix getInverse() const
    {
        C4X4Matrix retM(*this);
        retM.inverse();
        return(retM);
    }
    inline void setMultResult(const C4X4Matrix& m1,const C4X4Matrix& m2)
    { // Use this routine to avoid temporary variables
        M.axis[0](0)=m1.M.axis[0](0)*m2.M.axis[0](0)+m1.M.axis[1](0)*m2.M.axis[0](1)+m1.M.axis[2](0)*m2.M.axis[0](2);
        M.axis[0](1)=m1.M.axis[0](1)*m2.M.axis[0](0)+m1.M.axis[1](1)*m2.M.axis[0](1)+m1.M.axis[2](1)*m2.M.axis[0](2);
        M.axis[0](2)=m1.M.axis[0](2)*m2.M.axis[0](0)+m1.M.axis[1](2)*m2.M.axis[0](1)+m1.M.axis[2](2)*m2.M.axis[0](2);
        M.axis[1](0)=m1.M.axis[0](0)*m2.M.axis[1](0)+m1.M.axis[1](0)*m2.M.axis[1](1)+m1.M.axis[2](0)*m2.M.axis[1](2);
        M.axis[1](1)=m1.M.axis[0](1)*m2.M.axis[1](0)+m1.M.axis[1](1)*m2.M.axis[1](1)+m1.M.axis[2](1)*m2.M.axis[1](2);
        M.axis[1](2)=m1.M.axis[0](2)*m2.M.axis[1](0)+m1.M.axis[1](2)*m2.M.axis[1](1)+m1.M.axis[2](2)*m2.M.axis[1](2);
        M.axis[2](0)=m1.M.axis[0](0)*m2.M.axis[2](0)+m1.M.axis[1](0)*m2.M.axis[2](1)+m1.M.axis[2](0)*m2.M.axis[2](2);
        M.axis[2](1)=m1.M.axis[0](1)*m2.M.axis[2](0)+m1.M.axis[1](1)*m2.M.axis[2](1)+m1.M.axis[2](1)*m2.M.axis[2](2);
        M.axis[2](2)=m1.M.axis[0](2)*m2.M.axis[2](0)+m1.M.axis[1](2)*m2.M.axis[2](1)+m1.M.axis[2](2)*m2.M.axis[2](2);
        X(0)=m1.M.axis[0](0)*m2.X(0)+m1.M.axis[1](0)*m2.X(1)+m1.M.axis[2](0)*m2.X(2)+m1.X(0);
        X(1)=m1.M.axis[0](1)*m2.X(0)+m1.M.axis[1](1)*m2.X(1)+m1.M.axis[2](1)*m2.X(2)+m1.X(1);
        X(2)=m1.M.axis[0](2)*m2.X(0)+m1.M.axis[1](2)*m2.X(1)+m1.M.axis[2](2)*m2.X(2)+m1.X(2);
    }
    inline C4X4Matrix operator* (const C4X4Matrix& m) const
    {   // Speed optimized version:
        C4X4Matrix retM;
        retM.M.axis[0](0)=M.axis[0](0)*m.M.axis[0](0)+M.axis[1](0)*m.M.axis[0](1)+M.axis[2](0)*m.M.axis[0](2);
        retM.M.axis[0](1)=M.axis[0](1)*m.M.axis[0](0)+M.axis[1](1)*m.M.axis[0](1)+M.axis[2](1)*m.M.axis[0](2);
        retM.M.axis[0](2)=M.axis[0](2)*m.M.axis[0](0)+M.axis[1](2)*m.M.axis[0](1)+M.axis[2](2)*m.M.axis[0](2);
        retM.M.axis[1](0)=M.axis[0](0)*m.M.axis[1](0)+M.axis[1](0)*m.M.axis[1](1)+M.axis[2](0)*m.M.axis[1](2);
        retM.M.axis[1](1)=M.axis[0](1)*m.M.axis[1](0)+M.axis[1](1)*m.M.axis[1](1)+M.axis[2](1)*m.M.axis[1](2);
        retM.M.axis[1](2)=M.axis[0](2)*m.M.axis[1](0)+M.axis[1](2)*m.M.axis[1](1)+M.axis[2](2)*m.M.axis[1](2);
        retM.M.axis[2](0)=M.axis[0](0)*m.M.axis[2](0)+M.axis[1](0)*m.M.axis[2](1)+M.axis[2](0)*m.M.axis[2](2);
        retM.M.axis[2](1)=M.axis[0](1)*m.M.axis[2](0)+M.axis[1](1)*m.M.axis[2](1)+M.axis[2](1)*m.M.axis[2](2);
        retM.M.axis[2](2)=M.axis[0](2)*m.M.axis[2](0)+M.axis[1](2)*m.M.axis[2](1)+M.axis[2](2)*m.M.axis[2](2);
        retM.X(0)=M.axis[0](0)*m.X(0)+M.axis[1](0)*m.X(1)+M.axis[2](0)*m.X(2)+X(0);
        retM.X(1)=M.axis[0](1)*m.X(0)+M.axis[1](1)*m.X(1)+M.axis[2](1)*m.X(2)+X(1);
        retM.X(2)=M.axis[0](2)*m.X(0)+M.axis[1](2)*m.X(1)+M.axis[2](2)*m.X(2)+X(2);
        return(retM);
        /*  // Normal version:
        C4X4Matrix retM;
        retM.M=M*m.M;
        retM.X=(M*m.X)+X;
        return(retM);
        //  */
    }
    inline C3Vector operator* (const C3Vector& v) const
    {
        // Speed optimized version:
        return(C3Vector(M.axis[0](0)*v(0)+M.axis[1](0)*v(1)+M.axis[2](0)*v(2)+X(0),
            M.axis[0](1)*v(0)+M.axis[1](1)*v(1)+M.axis[2](1)*v(2)+X(1),
            M.axis[0](2)*v(0)+M.axis[1](2)*v(1)+M.axis[2](2)*v(2)+X(2)));
        // Normal version:
        //  return(C3Vector((M*v)+X));
    }
    inline C4X4Matrix& operator= (const C4X4Matrix& m)
    {
        // Speed optimized version:
        M.axis[0](0)=m.M.axis[0](0);
        M.axis[0](1)=m.M.axis[0](1);
        M.axis[0](2)=m.M.axis[0](2);
        M.axis[1](0)=m.M.axis[1](0);
        M.axis[1](1)=m.M.axis[1](1);
        M.axis[1](2)=m.M.axis[1](2);
        M.axis[2](0)=m.M.axis[2](0);
        M.axis[2](1)=m.M.axis[2](1);
        M.axis[2](2)=m.M.axis[2](2);
        X(0)=m.X(0);
        X(1)=m.X(1);
        X(2)=m.X(2);
        return(*this);
        /*  // Normal version:  
        M=m.M;
        X=m.X;
        return(*this);
        //  */
    }
    inline void operator*= (const C4X4Matrix& m)
    {
        C4X4Matrix retM((*this)*m);
        (*this)=retM;
    }

    C3X3Matrix M;
    C3Vector X;
};
