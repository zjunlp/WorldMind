#ifndef EXTERNAL_IK
#else
//PUT_EXTERNALIK_COPYRIGHT_NOTICE_HERE
#endif

#pragma once

#include "mathDefines.h"

class VPoint
{
public:
    VPoint()    {}
    VPoint(int initX,int initY) { x=initX; y=initY; }
    virtual ~VPoint()   {}
    int x,y;
};
