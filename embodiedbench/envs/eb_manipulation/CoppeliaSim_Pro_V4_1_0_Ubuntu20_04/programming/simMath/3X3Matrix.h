#pragma once

#include "mathDefines.h"
#include "3Vector.h"

class C4Vector;

class C3X3Matrix  
{
public:
    C3X3Matrix();
    C3X3Matrix(const C4Vector& q);
    C3X3Matrix(const C3X3Matrix& m);
    C3X3Matrix(const C3Vector& xAxis,const C3Vector& yAxis,const C3Vector& zAxis);
    ~C3X3Matrix();

    void buildInterpolation(const C3X3Matrix& fromThis,const C3X3Matrix& toThat,simMathReal t);
    C4Vector getQuaternion() const;
    void setEulerAngles(simMathReal a,simMathReal b,simMathReal g);
    void setEulerAngles(const C3Vector& v);
    C3Vector getEulerAngles() const;
    void buildXRotation(simMathReal angle);
    void buildYRotation(simMathReal angle);
    void buildZRotation(simMathReal angle);
    C3Vector getNormalVector() const;

    inline void getInternalData(simMathReal d[9]) const
    {
        axis[0].getInternalData(d+0);
        axis[1].getInternalData(d+3);
        axis[2].getInternalData(d+6);
    }
    inline void setInternalData(const simMathReal d[9])
    {
        axis[0].setInternalData(d+0);
        axis[1].setInternalData(d+3);
        axis[2].setInternalData(d+6);
    }
    inline simMathReal& operator() (size_t i,size_t j)
    {
        return(axis[j](i));
    }
    inline const simMathReal& operator() (size_t i,size_t j) const
    {
        return(axis[j](i));
    }
    inline void clear()
    {
        axis[0](0)=simZero;
        axis[0](1)=simZero;
        axis[0](2)=simZero;
        axis[1](0)=simZero;
        axis[1](1)=simZero;
        axis[1](2)=simZero;
        axis[2](0)=simZero;
        axis[2](1)=simZero;
        axis[2](2)=simZero;
    }
    inline void setIdentity()
    {
        axis[0](0)=simOne;
        axis[0](1)=simZero;
        axis[0](2)=simZero;
        axis[1](0)=simZero;
        axis[1](1)=simOne;
        axis[1](2)=simZero;
        axis[2](0)=simZero;
        axis[2](1)=simZero;
        axis[2](2)=simOne;
    }
    inline void transpose()
    {
        (*this)=getTranspose();
    }
    inline void set(const C3Vector& xAxis,const C3Vector& yAxis,const C3Vector& zAxis)
    {
        axis[0]=xAxis;
        axis[1]=yAxis;
        axis[2]=zAxis;
    }
    inline void set(const simMathReal m[3][3])
    {
        axis[0](0)=m[0][0];
        axis[0](1)=m[1][0];
        axis[0](2)=m[2][0];
        axis[1](0)=m[0][1];
        axis[1](1)=m[1][1];
        axis[1](2)=m[2][1];
        axis[2](0)=m[0][2];
        axis[2](1)=m[1][2];
        axis[2](2)=m[2][2];
    }
    inline void copyTo(simMathReal m[3][3]) const
    {
        m[0][0]=axis[0](0);
        m[1][0]=axis[0](1);
        m[2][0]=axis[0](2);
        m[0][1]=axis[1](0);
        m[1][1]=axis[1](1);
        m[2][1]=axis[1](2);
        m[0][2]=axis[2](0);
        m[1][2]=axis[2](1);
        m[2][2]=axis[2](2);
    }
    inline void copyToInterface(simMathReal* m) const
    { // Temporary routine. Remove later!
        for (size_t i=0;i<3;i++)
        {
            m[3*i+0]=axis[0](i);
            m[3*i+1]=axis[1](i);
            m[3*i+2]=axis[2](i);
        }
    }
    inline void copyFromInterface(const simMathReal* m)
    { // Temporary routine. Remove later!
        for (size_t i=0;i<3;i++)
        {
            axis[0](i)=m[3*i+0];
            axis[1](i)=m[3*i+1];
            axis[2](i)=m[3*i+2];
        }
    }
    inline bool isValid() const
    {
        for (int i=0;i<3;i++)
        {
            if (!axis[i].isValid())
                return(false);
        }
        return(true);
    }
    inline C3X3Matrix getTranspose() const
    {
        C3X3Matrix retM;
        retM(0,0)=axis[0](0);
        retM(0,1)=axis[0](1);
        retM(0,2)=axis[0](2);
        retM(1,0)=axis[1](0);
        retM(1,1)=axis[1](1);
        retM(1,2)=axis[1](2);
        retM(2,0)=axis[2](0);
        retM(2,1)=axis[2](1);
        retM(2,2)=axis[2](2);
        return(retM);
    }
    inline C3X3Matrix operator* (const C3X3Matrix& m) const
    {
        C3X3Matrix retM;
        retM(0,0)=axis[0](0)*m(0,0)+axis[1](0)*m(1,0)+axis[2](0)*m(2,0);
        retM(0,1)=axis[0](0)*m(0,1)+axis[1](0)*m(1,1)+axis[2](0)*m(2,1);
        retM(0,2)=axis[0](0)*m(0,2)+axis[1](0)*m(1,2)+axis[2](0)*m(2,2);
        retM(1,0)=axis[0](1)*m(0,0)+axis[1](1)*m(1,0)+axis[2](1)*m(2,0);
        retM(1,1)=axis[0](1)*m(0,1)+axis[1](1)*m(1,1)+axis[2](1)*m(2,1);
        retM(1,2)=axis[0](1)*m(0,2)+axis[1](1)*m(1,2)+axis[2](1)*m(2,2);
        retM(2,0)=axis[0](2)*m(0,0)+axis[1](2)*m(1,0)+axis[2](2)*m(2,0);
        retM(2,1)=axis[0](2)*m(0,1)+axis[1](2)*m(1,1)+axis[2](2)*m(2,1);
        retM(2,2)=axis[0](2)*m(0,2)+axis[1](2)*m(1,2)+axis[2](2)*m(2,2);
        return(retM);   
    }
    inline C3X3Matrix operator+ (const C3X3Matrix& m) const
    {
        C3X3Matrix retM;
        retM(0,0)=axis[0](0)+m(0,0);
        retM(0,1)=axis[1](0)+m(0,1);
        retM(0,2)=axis[2](0)+m(0,2);
        retM(1,0)=axis[0](1)+m(1,0);
        retM(1,1)=axis[1](1)+m(1,1);
        retM(1,2)=axis[2](1)+m(1,2);
        retM(2,0)=axis[0](2)+m(2,0);
        retM(2,1)=axis[1](2)+m(2,1);
        retM(2,2)=axis[2](2)+m(2,2);
        return(retM);   
    }
    inline C3X3Matrix operator- (const C3X3Matrix& m) const
    {
        C3X3Matrix retM;
        retM(0,0)=axis[0](0)-m(0,0);
        retM(0,1)=axis[1](0)-m(0,1);
        retM(0,2)=axis[2](0)-m(0,2);
        retM(1,0)=axis[0](1)-m(1,0);
        retM(1,1)=axis[1](1)-m(1,1);
        retM(1,2)=axis[2](1)-m(1,2);
        retM(2,0)=axis[0](2)-m(2,0);
        retM(2,1)=axis[1](2)-m(2,1);
        retM(2,2)=axis[2](2)-m(2,2);
        return(retM);   
    }
    inline C3X3Matrix operator* (simMathReal f) const
    {
        C3X3Matrix retM;
        retM(0,0)=axis[0](0)*f;
        retM(0,1)=axis[1](0)*f;
        retM(0,2)=axis[2](0)*f;
        retM(1,0)=axis[0](1)*f;
        retM(1,1)=axis[1](1)*f;
        retM(1,2)=axis[2](1)*f;
        retM(2,0)=axis[0](2)*f;
        retM(2,1)=axis[1](2)*f;
        retM(2,2)=axis[2](2)*f;
        return(retM);   
    }
    inline C3X3Matrix operator/ (simMathReal f) const
    {
        C3X3Matrix retM;
        retM(0,0)=axis[0](0)/f;
        retM(0,1)=axis[1](0)/f;
        retM(0,2)=axis[2](0)/f;
        retM(1,0)=axis[0](1)/f;
        retM(1,1)=axis[1](1)/f;
        retM(1,2)=axis[2](1)/f;
        retM(2,0)=axis[0](2)/f;
        retM(2,1)=axis[1](2)/f;
        retM(2,2)=axis[2](2)/f;
        return(retM);   
    }
    inline void operator*= (const C3X3Matrix& m)
    {
        C3X3Matrix retM;
        retM(0,0)=axis[0](0)*m(0,0)+axis[1](0)*m(1,0)+axis[2](0)*m(2,0);
        retM(0,1)=axis[0](0)*m(0,1)+axis[1](0)*m(1,1)+axis[2](0)*m(2,1);
        retM(0,2)=axis[0](0)*m(0,2)+axis[1](0)*m(1,2)+axis[2](0)*m(2,2);
        retM(1,0)=axis[0](1)*m(0,0)+axis[1](1)*m(1,0)+axis[2](1)*m(2,0);
        retM(1,1)=axis[0](1)*m(0,1)+axis[1](1)*m(1,1)+axis[2](1)*m(2,1);
        retM(1,2)=axis[0](1)*m(0,2)+axis[1](1)*m(1,2)+axis[2](1)*m(2,2);
        retM(2,0)=axis[0](2)*m(0,0)+axis[1](2)*m(1,0)+axis[2](2)*m(2,0);
        retM(2,1)=axis[0](2)*m(0,1)+axis[1](2)*m(1,1)+axis[2](2)*m(2,1);
        retM(2,2)=axis[0](2)*m(0,2)+axis[1](2)*m(1,2)+axis[2](2)*m(2,2);
        (*this)=retM;
    }
    inline void operator+= (const C3X3Matrix& m)
    {
        axis[0](0)+=m(0,0);
        axis[1](0)+=m(0,1);
        axis[2](0)+=m(0,2);
        axis[0](1)+=m(1,0);
        axis[1](1)+=m(1,1);
        axis[2](1)+=m(1,2);
        axis[0](2)+=m(2,0);
        axis[1](2)+=m(2,1);
        axis[2](2)+=m(2,2);
    }
    inline void operator-= (const C3X3Matrix& m)
    {
        axis[0](0)-=m(0,0);
        axis[1](0)-=m(0,1);
        axis[2](0)-=m(0,2);
        axis[0](1)-=m(1,0);
        axis[1](1)-=m(1,1);
        axis[2](1)-=m(1,2);
        axis[0](2)-=m(2,0);
        axis[1](2)-=m(2,1);
        axis[2](2)-=m(2,2);
    }
    inline void operator*= (simMathReal f)
    {
        axis[0](0)*=f;
        axis[1](0)*=f;
        axis[2](0)*=f;
        axis[0](1)*=f;
        axis[1](1)*=f;
        axis[2](1)*=f;
        axis[0](2)*=f;
        axis[1](2)*=f;
        axis[2](2)*=f;
    }
    inline void operator/= (simMathReal f)
    {
        axis[0](0)/=f;
        axis[1](0)/=f;
        axis[2](0)/=f;
        axis[0](1)/=f;
        axis[1](1)/=f;
        axis[2](1)/=f;
        axis[0](2)/=f;
        axis[1](2)/=f;
        axis[2](2)/=f;
    }
    inline C3Vector operator* (const C3Vector& v) const
    {
        C3Vector retV;
        retV(0)=axis[0](0)*v(0)+axis[1](0)*v(1)+axis[2](0)*v(2);
        retV(1)=axis[0](1)*v(0)+axis[1](1)*v(1)+axis[2](1)*v(2);
        retV(2)=axis[0](2)*v(0)+axis[1](2)*v(1)+axis[2](2)*v(2);
        return(retV);   
    }
    inline C3X3Matrix& operator= (const C3X3Matrix& m)
    {
        axis[0](0)=m(0,0);
        axis[1](0)=m(0,1);
        axis[2](0)=m(0,2);
        axis[0](1)=m(1,0);
        axis[1](1)=m(1,1);
        axis[2](1)=m(1,2);
        axis[0](2)=m(2,0);
        axis[1](2)=m(2,1);
        axis[2](2)=m(2,2);
        return(*this);
    }

    C3Vector axis[3];
};
