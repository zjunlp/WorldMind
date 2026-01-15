#pragma once

#include "mathDefines.h"
#include "4X4Matrix.h"
#include "3Vector.h"

class C4X4FullMatrix  
{
public:
    C4X4FullMatrix();   // Needed for serialization
    C4X4FullMatrix(const C4X4Matrix& m);
    C4X4FullMatrix(const C4X4FullMatrix& m);
    ~C4X4FullMatrix();

    void invert();
    void clear();
    void setIdentity();
    void buildZRotation(simMathReal angle);
    void buildTranslation(simMathReal x, simMathReal y, simMathReal z);
    C3Vector getEulerAngles() const;

    C4X4FullMatrix operator* (const C4X4FullMatrix& m) const;
    C4X4FullMatrix operator* (simMathReal d) const;
    C4X4FullMatrix operator/ (simMathReal d) const;
    C4X4FullMatrix operator+ (const C4X4FullMatrix& m) const;
    C4X4FullMatrix operator- (const C4X4FullMatrix& m) const;
    
    void operator*= (const C4X4FullMatrix& m);
    void operator+= (const C4X4FullMatrix& m);
    void operator-= (const C4X4FullMatrix& m);
    void operator*= (simMathReal d);
    void operator/= (simMathReal d);

    C4X4FullMatrix& operator= (const C4X4Matrix& m);
    C4X4FullMatrix& operator= (const C4X4FullMatrix& m);

    inline simMathReal& operator() (size_t row,size_t col)
    {
        return(data[row][col]);
    }
    inline const simMathReal& operator() (size_t row,size_t col) const
    {
        return(data[row][col]);
    }
        
private:
    simMathReal data[4][4];
};

