#include "octave/oct.h"

extern "C" {
    #include "extApi.h"
    #include <string>
}

#define emptyArg 2147483600
#define stringArg 2147483601
#define anyArg 2147483602

bool checkOneArg(const octave_value_list& args,int index,int size)
{ // size: 0 for any size, except 0. >0: for the size, or bigger (unless it is 1, then it is 1). <0: for the -size, or smaller (unless it is -1, then it is -1) or empty
    bool error=false;
    if (size==emptyArg)
        error=true;
    else
    {
        if (size==stringArg)
            error=(!args(index).is_string());
        else
        {
            if (size==-stringArg)
            {
                if (args(index).length()!=0)
                    error=(!args(index).is_string());
            }
            else
            {
                if (size!=anyArg)
                {
                    if (size==0)
                        error=(args(index).length()==0);
                    else
                    {
                        if (size>0)
                        {
                            if (size==1)
                                error=(args(index).length()!=1);
                            else
                                error=(args(index).length()<size);
                        }
                        else
                        {
                            if (args(index).length()!=0)
                            {
                                if (size==-1)
                                    error=(args(index).length()!=1);
                                else
                                    error=(args(index).length()<-size);
                            }
                        }
                    }
                }
            }
        }
    }
    return(error);
}

bool checkInputArgs(const char* funcName,const octave_value_list& args,int requiredCnt,int size1=emptyArg,int size2=emptyArg,int size3=emptyArg,int size4=emptyArg,int size5=emptyArg,int size6=emptyArg,int size7=emptyArg,int size8=emptyArg,int size9=emptyArg)
{ // size: 0 for any size, except 0. >0: for the size, or bigger (unless it is 1, then it is 1). <0: for the -size, or smaller (unless it is -1, then it is -1) or empty
    if (requiredCnt>args.length())
    {
        octave_stdout << "Error in remote API function ";
        octave_stdout << funcName;
        octave_stdout << ":\n--> invalid number of arguments\n";
        return(false);
    }
    bool error=false;
    error|=((requiredCnt>0)&&checkOneArg(args,0,size1));
    error|=((requiredCnt>1)&&checkOneArg(args,1,size2));
    error|=((requiredCnt>2)&&checkOneArg(args,2,size3));
    error|=((requiredCnt>3)&&checkOneArg(args,3,size4));
    error|=((requiredCnt>4)&&checkOneArg(args,4,size5));
    error|=((requiredCnt>5)&&checkOneArg(args,5,size6));
    error|=((requiredCnt>6)&&checkOneArg(args,6,size7));
    error|=((requiredCnt>7)&&checkOneArg(args,7,size8));
    error|=((requiredCnt>8)&&checkOneArg(args,8,size9));
    if (error)
    {
        octave_stdout << "Error in remote API function ";
        octave_stdout << funcName;
        octave_stdout << ":\n--> invalid arguments\n";
        return(false);
    }
    return(true);
}


DEFUN_DLD (simxStart, args, nargout,"simxStart")
{
    const char* funcName="simxStart";
    octave_value ret=-1;
    if (!checkInputArgs(funcName,args,6,stringArg,1,1,1,1,1))
        return(ret);

    std::string server = args(0).string_value();
    simxInt port = args(1).int_value();
    simxUChar val1 = args(2).bool_value();
    simxUChar val2 = args(3).bool_value();
    simxInt timeout = args(4).int_value();
    simxInt t2 = args(5).int_value();
    const simxChar *srv = server.c_str();
    ret = simxStart(srv,port,val1,val2,timeout,t2);
    return ret;
}

DEFUN_DLD (simxFinish, args, nargout,"simxFinish")
{
    const char* funcName="simxFinish";
    octave_value_list retVallist;
    if (!checkInputArgs(funcName,args,1,1))
        return retVallist;

    int clientID = args(0).int_value();
    simxFinish(clientID);
    return retVallist;
}

DEFUN_DLD (simxAddStatusbarMessage,args,nargout,"simxAddStatusbarMessage")
{
    const char* funcName="simxAddStatusbarMessage";
// simxAddStatusbarMessage(simxInt clientID,const simxChar* message,simxInt operationMode)

    octave_value ret=simx_return_local_error_flag;
    if (!checkInputArgs(funcName,args,3,1,stringArg,1))
        return(ret);

    simxInt clientID = args(0).int_value() ;
    std::string msg = args(1).string_value();
    const simxChar* message = msg.c_str();
    simxInt operationMode = args(2).int_value();
    ret = simxAddStatusbarMessage (clientID, message, operationMode);
    charNDArray test="Marc";
    ret = octave_value(test,true,'\'');
    return ret;
}

DEFUN_DLD (simxAppendStringSignal,args,nargout,"simxAppendStringSignal")
{
    const char* funcName="simxAppendStringSignal";
// simxAppendStringSignal(simxInt clientID,const simxChar* signalName,const simxUChar* signalValue,simxInt signalLength,simxInt operationMode)

    octave_value retVal=simx_return_local_error_flag;
    if (!checkInputArgs(funcName,args,4,1,stringArg,stringArg,1))
        return(retVal);

    simxInt clientID = args(0).int_value();
    std::string msg = args(1).string_value();
    const simxChar* signalName = msg.c_str();
    charNDArray sigval = args(2).char_array_value();
    const simxUChar* signalValue = (const simxUChar*)sigval.data();
    simxInt signalLength = sigval.length();
    simxInt operationMode = args(3).int_value();
    retVal = simxAppendStringSignal ( clientID, signalName, signalValue, signalLength, operationMode);
    return retVal;
}

DEFUN_DLD (simxWriteStringStream,args,nargout,"simxWriteStringStream")
{
    const char* funcName="simxWriteStringStream";
// simxWriteStringStream(simxInt clientID,const simxChar* signalName,const simxUChar* signalValue,simxInt signalLength,simxInt operationMode)

    octave_value retVal=simx_return_local_error_flag;
    if (!checkInputArgs(funcName,args,4,1,stringArg,stringArg,1))
        return(retVal);

    simxInt clientID = args(0).int_value();
    std::string msg = args(1).string_value();
    const simxChar* signalName = msg.c_str();
    charNDArray sigval = args(2).char_array_value();
    const simxUChar* signalValue = (const simxUChar*)sigval.data();
    simxInt signalLength = sigval.length();
    simxInt operationMode = args(3).int_value();
    retVal = simxWriteStringStream ( clientID, signalName, signalValue, signalLength, operationMode);
    return retVal;
}

DEFUN_DLD (simxAuxiliaryConsoleClose,args,nargout,"simxAuxiliaryConsoleClose")
{
    const char* funcName="simxAuxiliaryConsoleClose";
// simxAuxiliaryConsoleClose(simxInt clientID,simxInt consoleHandle,simxInt operationMode)

    octave_value retVal=simx_return_local_error_flag;
    if (!checkInputArgs(funcName,args,3,1,1,1))
        return(retVal);

    simxInt clientID = args(0).int_value();
    simxInt consoleHandle = args(1).int_value();
    simxInt operationMode = args(2).int_value();
    retVal = simxAuxiliaryConsoleClose ( clientID, consoleHandle, operationMode);
    return retVal;
}

DEFUN_DLD (simxAuxiliaryConsoleOpen,args,nargout,"simxAuxiliaryConsoleOpen")
{
    const char* funcName="simxAuxiliaryConsoleOpen";
// simxAuxiliaryConsoleOpen(simxInt clientID,const simxChar* title,simxInt maxLines,simxInt mode,simxInt* position,simxInt* size,simxFloat* textColor,simxFloat* backgroundColor,simxInt* consoleHandle,simxInt operationMode)
    octave_value_list retVallist;
    octave_value retVal=simx_return_local_error_flag;
    if (!checkInputArgs(funcName,args,9,1,stringArg,1,1,-2,-2,-3,-3,1))
    {
        retVallist(1) = -1;
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value();
    std::string msg = args(1).string_value();
    const simxChar* title = msg.c_str();
    simxInt maxLines = args(2).int_value();
    simxInt mode = args(3).int_value();

    simxInt position[2];
    simxInt* positionP=NULL;
    int32NDArray pos = args(4).int32_array_value();
    if (pos.length() >= 2)
    {
        positionP=position;
        position[0] = pos(0);
        position[1] = pos(1);
    }

    simxInt size[2];
    simxInt* sizeP=NULL;
    int32NDArray sz = args(5).int32_array_value();
    if (sz.length() >= 2)
    {
        sizeP=size;
        size[0] = sz(0);
        size[1] = sz(1);
    }

    simxFloat textColor[3];
    simxFloat* textColorP=NULL;
    FloatNDArray tc = args(6).float_array_value();
    if (tc.length() >= 3)
    {
        textColorP=textColor;
        textColor[0] = tc(0);
        textColor[1] = tc(1);
        textColor[2] = tc(2);
    }

    simxFloat backgroundColor[3];
    simxFloat* backgroundColorP=NULL;
    FloatNDArray bc = args(7).float_array_value();
    if (bc.length() >= 3)
    {
        backgroundColorP=backgroundColor;
        backgroundColor[0] = bc(0);
        backgroundColor[1] = bc(1);
        backgroundColor[2] = bc(2);
    }

    simxInt consoleHandle=-1;
    simxInt operationMode = args(8).int_value();
    retVal = simxAuxiliaryConsoleOpen (clientID, title, maxLines, mode, positionP, sizeP, textColorP, backgroundColorP, &consoleHandle, operationMode);
    retVallist(1) = consoleHandle;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxAuxiliaryConsolePrint,args,nargout,"simxAuxiliaryConsolePrint")
{
    const char* funcName="simxAuxiliaryConsolePrint";
// simxAuxiliaryConsolePrint(simxInt clientID,simxInt consoleHandle,const simxChar* txt,simxInt operationMode)

    octave_value retVal=simx_return_local_error_flag;
    if (!checkInputArgs(funcName,args,4,1,1,-stringArg,1))
        return(retVal);

    simxInt clientID = args(0).int_value();
    simxInt consoleHandle = args(1).int_value();

    const simxChar* txt=NULL;
    std::string msg;
    if (args(2).length()!=0)
    {
        msg = args(2).string_value();
        txt = msg.c_str();
    }
    simxInt operationMode = args(3).int_value();
    retVal = simxAuxiliaryConsolePrint ( clientID, consoleHandle, txt, operationMode);
    return retVal;
}

DEFUN_DLD (simxAuxiliaryConsoleShow,args,nargout,"simxAuxiliaryConsoleShow")
{
    const char* funcName="simxAuxiliaryConsoleShow";
// simxAuxiliaryConsoleShow(simxInt clientID,simxInt consoleHandle,simxUChar showState,simxInt operationMode)

    octave_value retVal=simx_return_local_error_flag;
    if (!checkInputArgs(funcName,args,4,1,1,1,1))
        return(retVal);

    simxInt clientID = args(0).int_value();
    simxInt consoleHandle = args(1).int_value();
    simxUChar showState = args(2).bool_value();
    simxInt operationMode = args(3).int_value();
    retVal = simxAuxiliaryConsoleShow ( clientID, consoleHandle, showState, operationMode);
    return retVal;
}

DEFUN_DLD (simxBreakForceSensor,args,nargout,"simxBreakForceSensor")
{
    const char* funcName="simxBreakForceSensor";
// simxBreakForceSensor(simxInt clientID,simxInt forceSensorHandle,simxInt operationMode)

    octave_value retVal=simx_return_local_error_flag;
    if (!checkInputArgs(funcName,args,3,1,1,1))
        return(retVal);

    simxInt clientID = args(0).int_value();
    simxInt forceSensorHandle = args(1).int_value();
    simxInt operationMode = args(2).int_value();
    retVal = simxBreakForceSensor ( clientID, forceSensorHandle, operationMode);
    return retVal;
}

DEFUN_DLD (simxClearFloatSignal,args,nargout,"simxClearFloatSignal")
{
    const char* funcName="simxClearFloatSignal";
// simxClearFloatSignal(simxInt clientID,const simxChar* signalName,simxInt operationMode)

    octave_value retVal=simx_return_local_error_flag;
    if (!checkInputArgs(funcName,args,3,1,stringArg,1))
        return(retVal);

    simxInt clientID = args(0).int_value();
    std::string tmp = args(1).string_value();
    const simxChar* signalName = tmp.c_str();
    simxInt operationMode = args(2).int_value();
    retVal = simxClearFloatSignal ( clientID, signalName, operationMode);
    return retVal;
}

DEFUN_DLD (simxClearIntegerSignal,args,nargout,"simxClearIntegerSignal")
{
    const char* funcName="simxClearIntegerSignal";
// simxClearIntegerSignal(simxInt clientID,const simxChar* signalName,simxInt operationMode)

    octave_value retVal=simx_return_local_error_flag;
    if (!checkInputArgs(funcName,args,3,1,stringArg,1))
        return(retVal);

    simxInt clientID = args(0).int_value() ;
    std::string tmp = args(1).string_value();
    const simxChar* signalName = tmp.c_str();
    simxInt operationMode = args(2).int_value();
    retVal = simxClearIntegerSignal ( clientID, signalName, operationMode);
    return retVal;
}

DEFUN_DLD (simxClearStringSignal,args,nargout,"simxClearStringSignal")
{
    const char* funcName="simxClearStringSignal";
// simxClearStringSignal(simxInt clientID,const simxChar* signalName,simxInt operationMode)

    octave_value retVal=simx_return_local_error_flag;
    if (!checkInputArgs(funcName,args,3,1,stringArg,1))
        return(retVal);

    simxInt clientID = args(0).int_value() ;
    std::string tmp = args(1).string_value();
    const simxChar* signalName = tmp.c_str();
    simxInt operationMode = args(2).int_value();
    retVal = simxClearStringSignal ( clientID, signalName, operationMode);
    return retVal;
}

DEFUN_DLD (simxCloseScene,args,nargout,"simxCloseScene")
{
    const char* funcName="simxCloseScene";
// simxCloseScene(simxInt clientID,simxInt operationMode)

    octave_value retVal=simx_return_local_error_flag;
    if (!checkInputArgs(funcName,args,2,1,1))
        return(retVal);

    simxInt clientID = args(0).int_value();
    simxInt operationMode = args(1).int_value();
    retVal = simxCloseScene ( clientID, operationMode);
    return retVal;
}

DEFUN_DLD (simxCopyPasteObjects,args,nargout,"simxCopyPasteObjects")
{
    const char* funcName="simxCopyPasteObjects";
// simxCopyPasteObjects(simxInt clientID,const simxInt* objectHandles,simxInt objectCount,simxInt** newObjectHandles,simxInt* newObjectCount,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;
    int32NDArray newHandles;

     if (!checkInputArgs(funcName,args,3,1,0,1))
    {
        retVallist(1) = newHandles;
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value();
    int32NDArray objh = args(1).int32_array_value();
    const simxInt* objectHandles = (const simxInt*)objh.data();
    simxInt objectCount = objh.length();
    simxInt* newObjectHandles;
    simxInt newObjectCount;
    simxInt operationMode = args(2).int_value();
    retVal = simxCopyPasteObjects (clientID,objectHandles,objectCount,&newObjectHandles,&newObjectCount,operationMode);
    if (retVal == 0)
    {

        newHandles.resize(dim_vector(newObjectCount,1));
        for (int i=0; i<newObjectCount; i++)
        {
            newHandles(i) = newObjectHandles[i];
        }
    }
    retVallist(1) = newHandles;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxCreateDummy,args,nargout,"simxCreateDummy")
{
    const char* funcName="simxCreateDummy";
// simxCreateDummy(simxInt clientID,simxFloat size,const simxUChar* colors,simxInt* objectHandle,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal = simx_return_local_error_flag;

    if (!checkInputArgs(funcName,args,4,1,1,-12,1))
    {
        retVallist(1) = -1;
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value();
    simxFloat size = args(1).float_scalar_value();
    const simxUChar* colors = NULL;
    charNDArray colors_=args(2).char_array_value();
    if (colors_.length() >= 12 )
        colors = (const simxUChar*)colors_.data() ;
    simxInt objectHandle=-1;
    simxInt operationMode = args(3).int_value();
    retVal = simxCreateDummy ( clientID, size, colors, &objectHandle, operationMode);
    retVallist(1) = objectHandle;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxDisplayDialog,args,nargout,"simxDisplayDialog")
{
    const char* funcName="simxDisplayDialog";
// simxDisplayDialog(simxInt clientID,const simxChar* titleText,const simxChar* mainText,
    //simxInt dialogType,const simxChar* initialText,simxFloat* titleColors,simxFloat*
    //dialogColors,simxInt* dialogHandle,simxInt* uiHandle,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;

    if (!checkInputArgs(funcName,args,8,1,stringArg,stringArg,1,stringArg,-6,-6,1))
    {
        retVallist(2) = -1;
        retVallist(1) = -1;
        retVallist(0) = simx_return_local_error_flag;
        return retVallist;
    }

    simxInt clientID = args(0).int_value();
    std::string title_=args(1).string_value();
    const simxChar* titleText = title_.c_str();
    std::string mainText_=args(2).string_value();
    const simxChar* mainText = mainText_.c_str();
    simxInt dialogType = args(3).int_value();
    std::string initialText_=args(4).string_value();
    const simxChar* initialText = initialText_.c_str();

    const simxFloat* titleColors =NULL;
    FloatNDArray titleColors_=args(5).float_array_value();
    if (args(5).length() == 6)
        titleColors = titleColors_.data();

    const simxFloat* dialogColors =NULL;
    FloatNDArray dialogColors_=args(6).float_array_value();
    if (args(6).length() == 6)
        dialogColors = dialogColors_.data();

    simxInt dialogHandle=-1;
    simxInt uiHandle=-1;
    simxInt operationMode = args(7).int_value();
    retVal = simxDisplayDialog ( clientID, titleText, mainText, dialogType, initialText, titleColors, dialogColors, &dialogHandle, &uiHandle, operationMode);
    retVallist(2) = uiHandle;
    retVallist(1) = dialogHandle;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxEndDialog,args,nargout,"simxEndDialog")
{
    const char* funcName="simxEndDialog";
// simxEndDialog(simxInt clientID,simxInt dialogHandle,simxInt operationMode)

    octave_value retVal = simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,3,1,1,1))
        return retVal;

    simxInt clientID = args(0).int_value();
    simxInt dialogHandle = args(1).int_value();
    simxInt operationMode = args(2).int_value();
    retVal = simxEndDialog ( clientID, dialogHandle, operationMode);
    return retVal;
}

DEFUN_DLD (simxEraseFile,args,nargout,"simxEraseFile")
{
    const char* funcName="simxEraseFile";
// simxEraseFile(simxInt clientID,const simxChar* fileName_serverSide,simxInt operationMode)

    octave_value retVal = simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,3,1,stringArg,1))
        return retVal;

    simxInt clientID = args(0).int_value() ;
    std::string fileName_serverSide_=args(1).string_value();
    const simxChar* fileName_serverSide = fileName_serverSide_.c_str();
    simxInt operationMode = args(2).int_value();
    retVal = simxEraseFile ( clientID, fileName_serverSide, operationMode);
    return retVal;
}

DEFUN_DLD (simxGetAndClearStringSignal,args,nargout,"simxGetAndClearStringSignal")
{
    const char* funcName="simxGetAndClearStringSignal";
// simxGetAndClearStringSignal(simxInt clientID,const simxChar* signalName,simxUChar** signalValue,simxInt* signalLength,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal = simx_return_local_error_flag;
    charNDArray sigvals;

     if (!checkInputArgs(funcName,args,3,1,stringArg,1))
    {
        retVallist(1) = octave_value(sigvals,true,'\'');
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value();
    std::string signalName_= args(1).string_value();
    const simxChar* signalName = signalName_.c_str();
    simxUChar* signalValue;
    simxInt signalLength;
    simxInt operationMode = args(2).int_value();
    retVal = simxGetAndClearStringSignal ( clientID, signalName, &signalValue, &signalLength, operationMode);
    if (retVal == 0)
    {
        sigvals.resize(dim_vector(signalLength,1));
        for (int i=0; i<signalLength; i++)
            sigvals(i) = signalValue[i];
    }
    retVallist(1) = octave_value(sigvals,true,'\'');
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxReadStringStream,args,nargout,"simxReadStringStream")
{
    const char* funcName="simxReadStringStream";
// simxReadStringStream(simxInt clientID,const simxChar* signalName,simxUChar** signalValue,simxInt* signalLength,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal = simx_return_local_error_flag;
    charNDArray sigvals;

     if (!checkInputArgs(funcName,args,3,1,stringArg,1))
    {
        retVallist(1) = octave_value(sigvals,true,'\'');
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value();
    std::string signalName_= args(1).string_value();
    const simxChar* signalName = signalName_.c_str();
    simxUChar* signalValue;
    simxInt signalLength;
    simxInt operationMode = args(2).int_value();
    retVal = simxReadStringStream ( clientID, signalName, &signalValue, &signalLength, operationMode);
    if (retVal == 0)
    {
        sigvals.resize(dim_vector(signalLength,1));
        for (int i=0; i<signalLength; i++)
            sigvals(i) = signalValue[i];
    }
    retVallist(1) = octave_value(sigvals,true,'\'');
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxGetArrayParameter,args,nargout,"simxGetArrayParameter")
{
    const char* funcName="simxGetArrayParameter";
// simxGetArrayParameter(simxInt clientID,simxInt paramIdentifier,simxFloat* paramValues,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;
    FloatNDArray pv;

     if (!checkInputArgs(funcName,args,3,1,1,1))
    {
        retVallist(1) = pv;
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value();
    simxInt paramIdentifier = args(1).int_value();
    simxFloat paramValues[3];
    simxInt operationMode = args(2).int_value();
    retVal = simxGetArrayParameter ( clientID, paramIdentifier, paramValues, operationMode);
    if (retVal == 0)
    {
        pv.resize(dim_vector(3,1));
        pv(0) = paramValues[0];
        pv(1) = paramValues[1];
        pv(2) = paramValues[2];

    }
    retVallist(1) = pv;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxGetBooleanParameter,args,nargout,"simxGetBooleanParameter")
{
    const char* funcName="simxGetBooleanParameter";
// simxGetBooleanParameter(simxInt clientID,simxInt paramIdentifier,simxUChar* paramValue,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;

     if (!checkInputArgs(funcName,args,3,1,1,1))
    {
        retVallist(1) = false;
        retVallist(0) = retVal;
        return retVallist;
    }
    simxInt clientID = args(0).int_value();
    simxInt paramIdentifier = args(1).int_value();
    simxUChar paramValue;
    simxInt operationMode = args(2).int_value();
    retVal = simxGetBooleanParameter ( clientID, paramIdentifier, &paramValue, operationMode);
    retVallist(1) = false;
    if (retVal == 0)
        retVallist(1)=(paramValue != 0);
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxGetCollisionHandle,args,nargout,"simxGetCollisionHandle")
{
    const char* funcName="simxGetCollisionHandle";
// simxGetCollisionHandle(simxInt clientID,const simxChar* collisionObjectName,simxInt* handle,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;

     if (!checkInputArgs(funcName,args,3,1,stringArg,1))
    {
        retVallist(1) = -1;
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value();
    std::string collisionObjectName_=args(1).string_value();
    const simxChar* collisionObjectName = collisionObjectName_.c_str();
    simxInt handle=-1;
    simxInt operationMode = args(2).int_value();
    retVal = simxGetCollisionHandle ( clientID, collisionObjectName, &handle, operationMode);
    retVallist(1) = handle;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxGetCollectionHandle,args,nargout,"simxGetCollectionHandle")
{
    const char* funcName="simxGetCollectionHandle";
// simxGetCollectionHandle(simxInt clientID,const simxChar* collectionName,simxInt* handle,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;

     if (!checkInputArgs(funcName,args,3,1,stringArg,1))
    {
        retVallist(1) = -1;
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value();
    std::string collectionName_=args(1).string_value();
    const simxChar* collectionName = collectionName_.c_str();
    simxInt handle=-1;
    simxInt operationMode = args(2).int_value();
    retVal = simxGetCollectionHandle ( clientID, collectionName, &handle, operationMode);
    retVallist(1) = handle;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxGetConnectionId,args,nargout,"simxGetConnectionId")
{
    const char* funcName="simxGetConnectionId";
// simxGetConnectionId(simxInt clientID)

    octave_value retVal = -1;
     if (!checkInputArgs(funcName,args,1,1))
        return retVal;

    simxInt clientID = args(0).int_value();
    retVal = simxGetConnectionId (clientID);
    return retVal;
}

DEFUN_DLD (simxGetDialogInput,args,nargout,"simxGetDialogInput")
{
    const char* funcName="simxGetDialogInput";
// simxGetDialogInput(simxInt clientID,simxInt dialogHandle,simxChar** inputText,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;
    charNDArray arr;

    if (!checkInputArgs(funcName,args,3,1,1,1))
    {
        retVallist(1) = octave_value(arr,true,'\'');
        retVallist(0) = retVal;
        return retVallist;
    }
    simxInt clientID = args(0).int_value();
    simxInt dialogHandle = args(1).int_value();
    simxChar* inputText;
    simxInt operationMode = args(2).int_value();
    retVal = simxGetDialogInput ( clientID, dialogHandle, &inputText, operationMode);
    if (retVal == 0)
        arr=inputText;
    retVallist(1) = octave_value(arr,true,'\'');
    retVallist(0) = retVal;
    return retVallist;

}

DEFUN_DLD (simxGetDialogResult,args,nargout,"simxGetDialogResult")
{
    const char* funcName="simxGetDialogResult";
// simxGetDialogResult(simxInt clientID,simxInt dialogHandle,simxInt* result,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;
    if (!checkInputArgs(funcName,args,3,1,1,1))
    {
        retVallist(1) = -1;
        retVallist(0) = retVal;
        return retVallist;
    }
    simxInt clientID = args(0).int_value();
    simxInt dialogHandle = args(1).int_value();
    simxInt result=0;
    simxInt operationMode = args(2).int_value();
    retVal = simxGetDialogResult ( clientID, dialogHandle, &result, operationMode);
    retVallist(1) = result;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxGetDistanceHandle,args,nargout,"simxGetDistanceHandle")
{
    const char* funcName="simxGetDistanceHandle";
// simxGetDistanceHandle(simxInt clientID,const simxChar* distanceObjectName,simxInt* handle,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;

     if (!checkInputArgs(funcName,args,3,1,stringArg,1))
    {
        retVallist(1) = -1;
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value();
    std::string distanceObjectName_=args(1).string_value();
    const simxChar* distanceObjectName = distanceObjectName_.c_str();
    simxInt handle=-1;
    simxInt operationMode = args(2).int_value();
    retVal = simxGetDistanceHandle (clientID, distanceObjectName, &handle, operationMode);
    retVallist(1) = handle;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxGetFloatingParameter,args,nargout,"simxGetFloatingParameter")
{
    const char* funcName="simxGetFloatingParameter";
// simxGetFloatingParameter(simxInt clientID,simxInt paramIdentifier,simxFloat* paramValue,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;

     if (!checkInputArgs(funcName,args,3,1,1,1))
    {
        retVallist(1) = 0.0f;
        retVallist(0) = retVal;
        return retVallist;
    }
    simxInt clientID = args(0).int_value();
    simxInt paramIdentifier = args(1).int_value();
    simxFloat paramValue=0.0f;
    simxInt operationMode = args(2).int_value();
    retVal = simxGetFloatingParameter ( clientID, paramIdentifier, &paramValue, operationMode);
    retVallist(1) = paramValue;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxGetFloatSignal,args,nargout,"simxGetFloatSignal")
{
    const char* funcName="simxGetFloatSignal";
// simxGetFloatSignal(simxInt clientID,const simxChar* signalName,simxFloat* signalValue,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;

     if (!checkInputArgs(funcName,args,3,1,stringArg,1))
    {
        retVallist(1) = 0.0f;
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value();
    std::string signalName_ = args(1).string_value();
    const simxChar* signalName = signalName_.c_str();
    simxFloat signalValue=0.0f;
    simxInt operationMode = args(2).int_value();
    retVal = simxGetFloatSignal ( clientID, signalName, &signalValue, operationMode);
    retVallist(1) = signalValue;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxGetInMessageInfo,args,nargout,"simxGetInMessageInfo")
{
    const char* funcName="simxGetInMessageInfo";
// simxGetInMessageInfo(simxInt clientID,simxInt infoType,simxInt* info)
    octave_value_list retVallist;
    simxInt retVal=-1;
     if (!checkInputArgs(funcName,args,2,1,1))
    {
        retVallist(1) = -1;
        retVallist(0) = retVal;
        return retVallist;
    }
    simxInt clientID = args(0).int_value();
    simxInt infoType = args(1).int_value();
    simxInt info=0;
    retVal = simxGetInMessageInfo ( clientID, infoType, &info);
    retVallist(1) = info;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxGetIntegerParameter,args,nargout,"simxGetIntegerParameter")
{
    const char* funcName="simxGetIntegerParameter";
// simxGetIntegerParameter(simxInt clientID,simxInt paramIdentifier,simxInt* paramValue,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;

     if (!checkInputArgs(funcName,args,3,1,1,1))
    {
        retVallist(1) = -1;
        retVallist(0) = retVal;
        return retVallist;
    }
    simxInt clientID = args(0).int_value();
    simxInt paramIdentifier = args(1).int_value();
    simxInt paramValue=0;
    simxInt operationMode = args(2).int_value();
    retVal = simxGetIntegerParameter ( clientID, paramIdentifier, &paramValue, operationMode);
    retVallist(1) = paramValue;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxGetIntegerSignal,args,nargout,"simxGetIntegerSignal")
{
    const char* funcName="simxGetIntegerSignal";
// simxGetIntegerSignal(simxInt clientID,const simxChar* signalName,simxInt* signalValue,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;

     if (!checkInputArgs(funcName,args,3,1,stringArg,1))
    {
        retVallist(1) = -1;
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value();
    std::string signalName_ = args(1).string_value();
    const simxChar* signalName = signalName_.c_str();
    simxInt signalValue=0;
    simxInt operationMode = args(2).int_value();
    retVal = simxGetIntegerSignal ( clientID, signalName, &signalValue, operationMode);
    retVallist(1) = signalValue;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxGetJointMatrix,args,nargout,"simxGetJointMatrix")
{
    const char* funcName="simxGetJointMatrix";
// simxGetJointMatrix(simxInt clientID,simxInt jointHandle,simxFloat* matrix,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;
    FloatNDArray mtx;

     if (!checkInputArgs(funcName,args,3,1,1,1))
    {
        retVallist(1) = mtx;
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value();
    simxInt jointHandle = args(1).int_value();
    simxFloat matrix[12];
    simxInt operationMode = args(2).int_value();
    retVal = simxGetJointMatrix ( clientID, jointHandle, matrix, operationMode);
    if (retVal == 0)
    {
        mtx.resize(dim_vector(12,1));
        for (int i=0;i<12; i++)
            mtx(i) = matrix[i];
    }
    retVallist(1) = mtx;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxGetJointPosition,args,nargout,"simxGetJointPosition")
{
    const char* funcName="simxGetJointPosition";
// simxGetJointPosition(simxInt clientID,simxInt jointHandle,simxFloat* position,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;

     if (!checkInputArgs(funcName,args,3,1,1,1))
    {
        retVallist(1) = 0.0f;
        retVallist(0) = retVal;
        return retVallist;
    }
    simxInt clientID = args(0).int_value();
    simxInt jointHandle = args(1).int_value();
    simxFloat position=0.0f;
    simxInt operationMode = args(2).int_value();
    retVal = simxGetJointPosition ( clientID, jointHandle, &position, operationMode);
    retVallist(1) = position;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxGetLastCmdTime,args,nargout,"simxGetLastCmdTime")
{
    const char* funcName="simxGetLastCmdTime";
// simxGetLastCmdTime(simxInt clientID)

    octave_value retVal = -1;
     if (!checkInputArgs(funcName,args,1,1))
        return retVal;

    simxInt clientID = args(0).int_value();
    retVal = simxGetLastCmdTime (clientID);
    return retVal;
}

DEFUN_DLD (simxGetLastErrors,args,nargout,"simxGetLastErrors")
{
    const char* funcName="simxGetLastErrors";
// simxGetLastErrors(simxInt clientID,simxInt* errorCnt,simxChar** errorStrings,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;
    Cell cellstr;

     if (!checkInputArgs(funcName,args,2,1,1))
    {
        retVallist(1) = cellstr;
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value();
    simxInt errorCnt;
    simxChar* errorStrings;
    simxInt operationMode = args(1).int_value();
    retVal = simxGetLastErrors ( clientID, &errorCnt, &errorStrings, operationMode);

    if (retVal == 0)
    {
        int off=0;
        cellstr.resize(dim_vector(errorCnt,1));
        for (int i=0; i<errorCnt; i++)
        {
            cellstr(i) = errorStrings+off;
            off+=strlen(errorStrings+off)+1;
        }
    }
    retVallist(1) = cellstr;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxGetModelProperty,args,nargout,"simxGetModelProperty")
{
    const char* funcName="simxGetModelProperty";
// simxGetModelProperty(simxInt clientID,simxInt objectHandle,simxInt* prop,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;

     if (!checkInputArgs(funcName,args,3,1,1,1))
    {
        retVallist(1) = 0;
        retVallist(0) = retVal;
        return retVallist;
    }
    simxInt clientID = args(0).int_value();
    simxInt objectHandle = args(1).int_value();
    simxInt prop=0;
    simxInt operationMode = args(2).int_value();
    retVal = simxGetModelProperty ( clientID, objectHandle, &prop, operationMode);
    retVallist(1) = prop;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxGetObjectChild,args,nargout,"simxGetObjectChild")
{
    const char* funcName="simxGetObjectChild";
// simxGetObjectChild(simxInt clientID,simxInt parentObjectHandle,simxInt childIndex,simxInt* childObjectHandle,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;

     if (!checkInputArgs(funcName,args,4,1,1,1,1))
    {
        retVallist(1) = -1;
        retVallist(0) = retVal;
        return retVallist;
    }
    simxInt clientID = args(0).int_value();
    simxInt parentObjectHandle = args(1).int_value();
    simxInt childIndex = args(2).int_value();
    simxInt childObjectHandle=-1;
    simxInt operationMode = args(3).int_value();
    retVal = simxGetObjectChild ( clientID, parentObjectHandle, childIndex, &childObjectHandle, operationMode);
    retVallist(1) = childObjectHandle;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxGetObjectFloatParameter,args,nargout,"simxGetObjectFloatParameter")
{
    const char* funcName="simxGetObjectFloatParameter";
// simxGetObjectFloatParameter(simxInt clientID,simxInt objectHandle,simxInt parameterID,simxFloat* parameterValue,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;

     if (!checkInputArgs(funcName,args,4,1,1,1,1))
    {
        retVallist(1) = 0.0f;
        retVallist(0) = retVal;
        return retVallist;
    }
    simxInt clientID = args(0).int_value();
    simxInt objectHandle = args(1).int_value();
    simxInt parameterID = args(2).int_value();
    simxFloat parameterValue=0.0f;
    simxInt operationMode = args(3).int_value();
    retVal = simxGetObjectFloatParameter ( clientID, objectHandle, parameterID, &parameterValue, operationMode);
    retVallist(1) = parameterValue;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxGetObjectGroupData,args,nargout,"simxGetObjectGroupData")
{
    const char* funcName="simxGetObjectGroupData";
// simxGetObjectGroupData(simxInt clientID,simxInt objectType,simxInt dataType,simxInt* handlesCount,simxInt** handles,simxInt* intDataCount,simxInt** intData,simxInt* floatDataCount,simxFloat** floatData,simxInt* stringDataCount,simxChar** stringData,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;
    int32NDArray hndls;
    int32NDArray intd;
    FloatNDArray fltd;
    Cell strd;

     if (!checkInputArgs(funcName,args,4,1,1,1,1))
    {
        octave_stdout << "Invalid number of inputs \n";
        retVallist(4) = strd;
        retVallist(3) = fltd;
        retVallist(2) = intd;
        retVallist(1) = hndls;
        retVallist(0) = retVal;
        return retVallist;
    }
    simxInt clientID = args(0).int_value();
    simxInt objectType = args(1).int_value();
    simxInt dataType = args(2).int_value();
    simxInt handlesCount;
    simxInt* handles;
    simxInt intDataCount;
    simxInt* intData;
    simxInt floatDataCount;
    simxFloat* floatData;
    simxInt stringDataCount;
    simxChar* stringData;
    simxInt operationMode = args(3).int_value();
    retVal = simxGetObjectGroupData ( clientID, objectType, dataType, &handlesCount, &handles,
                                      &intDataCount, &intData, &floatDataCount, &floatData,
                                      &stringDataCount, &stringData, operationMode);
    if (retVal == 0)
    {
        hndls.resize(dim_vector(handlesCount,1));
        intd.resize(dim_vector(intDataCount,1));
        fltd.resize(dim_vector(floatDataCount,1));
        strd.resize(dim_vector(stringDataCount,1));
        for (int i=0; i< handlesCount; i++)
        {
            hndls(i) = handles[i];
        }
        for (int i=0; i< intDataCount; i++)
        {
            intd(i) = intData[i];
        }
        for (int i=0; i< floatDataCount; i++)
        {
            fltd(i) = floatData[i];
        }
        int off=0;
        for (int i=0; i< stringDataCount; i++)
        {
            strd(i) = stringData+off;
            off+=strlen(stringData+off)+1;
        }
    }
    retVallist(4) = strd;
    retVallist(3) = fltd;
    retVallist(2) = intd;
    retVallist(1) = hndls;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxCallScriptFunction,args,nargout,"simxCallScriptFunction")
{
    const char* funcName="simxCallScriptFunction";

    // simxInt simxCallScriptFunction(simxInt clientID,const simxChar* scriptDescription,simxInt options,const simxChar* functionName,simxInt inIntCnt,const simxInt* inInt,simxInt inFloatCnt,const simxFloat* inFloat,simxInt inStringCnt,const simxChar* inString,simxInt inBufferSize,const simxUChar* inBuffer,simxInt* outIntCnt,simxInt** outInt,simxInt* outFloatCnt,simxFloat** outFloat,simxInt* outStringCnt,simxChar** outString,simxInt* outBufferSize,simxUChar** outBuffer,simxInt operationMode);

    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;
    int32NDArray intd;
    FloatNDArray fltd;
    charNDArray strd;
    charNDArray buffd;

     if (!checkInputArgs(funcName,args,9,1,stringArg,1,stringArg,anyArg,anyArg,anyArg,anyArg,1))
    {
        retVallist(4) = buffd;
        retVallist(3) = strd;
        retVallist(2) = fltd;
        retVallist(1) = intd;
        retVallist(0) = retVal;
        return retVallist;
    }
    simxInt clientID = args(0).int_value();
    std::string scriptDescription = args(1).string_value();
    simxInt options = args(2).int_value();
    std::string functionName = args(3).string_value();

    int32NDArray inti = args(4).int32_array_value();
    simxInt intic = inti.length();
    const simxInt* intip = (const simxInt*)inti.data();

    FloatNDArray floati = args(5).float_array_value();
    simxInt floatic = floati.length();
    const simxFloat* floatip = (const simxFloat*)floati.data();

    charNDArray stringi = args(6).char_array_value();
    int l=stringi.length();
    std::string strdat;
    simxInt stringic = 0;
    const simxChar* stringip=NULL;
    if (l>0)
    {
        strdat=std::string(stringi.data(),stringi.data()+l);
        if (strdat[strdat.size()-1]!=0)
            strdat+='\0';
        for (size_t i=0;i<strdat.size();i++)
        {
            if (strdat[i]==0)
             stringic++;
        }
        stringip=strdat.c_str();
    }

    charNDArray buffi = args(7).char_array_value();
    simxInt buffis = buffi.length();
    const simxUChar* buffip = (const simxUChar*)buffi.data();

    simxInt operationMode = args(8).int_value();

    simxInt intoc;
    simxInt* intop;
    simxInt floatoc;
    simxFloat* floatop;
    simxInt stringoc;
    simxChar* stringop;
    simxInt buffos;
    simxUChar* buffop;
    retVal = simxCallScriptFunction ( clientID, scriptDescription.c_str(), options, functionName.c_str(),
                intic,intip,floatic,floatip,stringic,stringip,buffis,buffip,&intoc,&intop,&floatoc,&floatop,&stringoc,&stringop,&buffos,&buffop,operationMode);

    if (retVal == 0)
    {
        intd.resize(dim_vector(intoc,1));
        fltd.resize(dim_vector(floatoc,1));
        buffd.resize(dim_vector(buffos,1));

        for (int i=0; i< intoc; i++)
            intd(i) = intop[i];
        for (int i=0; i< floatoc; i++)
            fltd(i) = floatop[i];
        int cnt=0;
        for (int i=0; i< stringoc; i++)
            cnt+=strlen(stringop+cnt)+1;
        strd.resize(dim_vector(cnt,1));
        for (int i=0; i<cnt; i++)
            strd(i) = stringop[i];
        for (int i=0; i<buffos; i++)
            buffd(i) = buffop[i];
    }
    retVallist(4) = buffd;
    retVallist(3) = strd;
    retVallist(2) = fltd;
    retVallist(1) = intd;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxGetObjectHandle,args,nargout,"simxGetObjectHandle")
{
    const char* funcName="simxGetObjectHandle";
// simxGetObjectHandle(simxInt clientID,const simxChar* objectName,simxInt* handle,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;

     if (!checkInputArgs(funcName,args,3,1,stringArg,1))
    {
        retVallist(1) = -1;
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value();
    std::string objectName_ = args(1).string_value();
    const simxChar* objectName = objectName_.c_str();
    simxInt handle=-1;
    simxInt operationMode = args(2).int_value();
    retVal = simxGetObjectHandle ( clientID, objectName, &handle, operationMode);
    retVallist(1) = handle;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxGetObjectIntParameter,args,nargout,"simxGetObjectIntParameter")
{
    const char* funcName="simxGetObjectIntParameter";
// simxGetObjectIntParameter(simxInt clientID,simxInt objectHandle,simxInt parameterID,simxInt* parameterValue,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;

     if (!checkInputArgs(funcName,args,4,1,1,1,1))
    {
        retVallist(1) = 0;
        retVallist(0) = retVal;
        return retVallist;
    }
    simxInt clientID = args(0).int_value();
    simxInt objectHandle = args(1).int_value();
    simxInt parameterID = args(2).int_value();
    simxInt parameterValue=-1;
    simxInt operationMode = args(3).int_value();
    retVal = simxGetObjectIntParameter ( clientID, objectHandle, parameterID, &parameterValue, operationMode);
    retVallist(1) = parameterValue;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxGetObjectOrientation,args,nargout,"simxGetObjectOrientation")
{
    const char* funcName="simxGetObjectOrientation";
// simxGetObjectOrientation(simxInt clientID,simxInt objectHandle,simxInt relativeToObjectHandle,simxFloat* eulerAngles,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;
    FloatNDArray eang;

     if (!checkInputArgs(funcName,args,4,1,1,1,1))
    {
        retVallist(1) = eang;
        retVallist(0) = retVal;
        return retVallist;

    }

    simxInt clientID = args(0).int_value();
    simxInt objectHandle = args(1).int_value();
    simxInt relativeToObjectHandle = args(2).int_value();
    simxFloat eulerAngles[3];
    simxInt operationMode = args(3).int_value();
    retVal = simxGetObjectOrientation ( clientID, objectHandle, relativeToObjectHandle, eulerAngles, operationMode);
    if (retVal == 0)
    {
        eang.resize(dim_vector(3,1));
        eang(0) = eulerAngles[0];
        eang(1) = eulerAngles[1];
        eang(2) = eulerAngles[2];
    }
    retVallist(1) = eang;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxGetObjectQuaternion,args,nargout,"simxGetObjectQuaternion")
{
    const char* funcName="simxGetObjectQuaternion";
// simxGetObjectQuaternion(simxInt clientID,simxInt objectHandle,simxInt relativeToObjectHandle,simxFloat* quaternion,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;
    FloatNDArray quat;

     if (!checkInputArgs(funcName,args,4,1,1,1,1))
    {
        retVallist(1) = quat;
        retVallist(0) = retVal;
        return retVallist;

    }

    simxInt clientID = args(0).int_value();
    simxInt objectHandle = args(1).int_value();
    simxInt relativeToObjectHandle = args(2).int_value();
    simxFloat quaternion[4];
    simxInt operationMode = args(3).int_value();
    retVal = simxGetObjectQuaternion ( clientID, objectHandle, relativeToObjectHandle, quaternion, operationMode);
    if (retVal == 0)
    {
        quat.resize(dim_vector(4,1));
        quat(0) = quaternion[0];
        quat(1) = quaternion[1];
        quat(2) = quaternion[2];
        quat(3) = quaternion[3];
    }
    retVallist(1) = quat;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxGetObjectParent,args,nargout,"simxGetObjectParent")
{
    const char* funcName="simxGetObjectParent";
// simxGetObjectParent(simxInt clientID,simxInt childObjectHandle,simxInt* parentObjectHandle,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;

     if (!checkInputArgs(funcName,args,3,1,1,1))
    {
        retVallist(1) =  -1;
        retVallist(0) =  retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value();
    simxInt childObjectHandle = args(1).int_value();
    simxInt parentObjectHandle=-1;
    simxInt operationMode = args(2).int_value();
    retVal = simxGetObjectParent ( clientID, childObjectHandle, &parentObjectHandle, operationMode);
    retVallist(1) =  parentObjectHandle;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxGetObjectPosition,args,nargout,"simxGetObjectPosition")
{
    const char* funcName="simxGetObjectPosition";
// simxGetObjectPosition(simxInt clientID,simxInt objectHandle,simxInt relativeToObjectHandle,simxFloat* position,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;
    FloatNDArray pos;

     if (!checkInputArgs(funcName,args,4,1,1,1,1))
    {
        retVallist(1) = pos;
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value();
    simxInt objectHandle = args(1).int_value();
    simxInt relativeToObjectHandle = args(2).int_value();
    simxFloat position[3];
    simxInt operationMode = args(3).int_value();
    retVal = simxGetObjectPosition ( clientID, objectHandle, relativeToObjectHandle, position, operationMode);
    if (retVal == 0)
    {
        pos.resize(dim_vector(3,1));
        pos(0) = position[0];
        pos(1) = position[1];
        pos(2) = position[2];
    }
    retVallist(1) = pos;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxGetObjects,args,nargout,"simxGetObjects")
{
    const char* funcName="simxGetObjects";
// simxGetObjects(simxInt clientID,simxInt objectType,simxInt* objectCount,simxInt** objectHandles,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;
    int32NDArray objhn;

     if (!checkInputArgs(funcName,args,3,1,1,1))
    {
        retVallist(1) = objhn;
        retVallist(0) = retVal;
        return retVallist;
    }
    simxInt clientID = args(0).int_value();
    simxInt objectType = args(1).int_value();
    simxInt objectCount;
    simxInt* objectHandles;
    simxInt operationMode = args(2).int_value();
    retVal = simxGetObjects ( clientID, objectType, &objectCount, &objectHandles, operationMode);
    if (retVal == 0)
    {
        objhn.resize(dim_vector(objectCount,1));
        for (int i=0; i< objectCount; i++)
        {
            objhn(i) = objectHandles[i];
        }
    }
    retVallist(1) = objhn;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxGetObjectSelection,args,nargout,"simxGetObjectSelection")
{
    const char* funcName="simxGetObjectSelection";
// simxGetObjectSelection(simxInt clientID,simxInt** objectHandles,simxInt* objectCount,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;
    int32NDArray objhn;

     if (!checkInputArgs(funcName,args,2,1,1))
    {
        retVallist(1) = objhn;
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value();
    simxInt* objectHandles;
    simxInt objectCount;
    simxInt operationMode = args(1).int_value();
    retVal = simxGetObjectSelection ( clientID, &objectHandles, &objectCount, operationMode);
    if (retVal == 0)
    {
        objhn.resize(dim_vector(objectCount,1));
        for (int i=0; i<objectCount; i++)
        {
            objhn(i) = objectHandles[i];
        }
    }
    retVallist(1) = objhn;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxGetObjectVelocity,args,nargout,"simxGetObjectVelocity")
{
    const char* funcName="simxGetObjectVelocity";
// simxGetObjectVelocity(simxInt clientID,simxInt objectHandle,simxFloat* linearVelocity,simxFloat* angularVelocity,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;
    FloatNDArray lv,av;

     if (!checkInputArgs(funcName,args,3,1,1,1))
    {
        retVallist(2) = av;
        retVallist(1) = lv;
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value();
    simxInt objectHandle = args(1).int_value();
    simxFloat linearVelocity[3];
    simxFloat angularVelocity[3];
    simxInt operationMode = args(2).int_value();
    retVal = simxGetObjectVelocity ( clientID, objectHandle, linearVelocity, angularVelocity, operationMode);
    if (retVal == 0)
    {
        lv.resize(dim_vector(3,1));
        av.resize(dim_vector(3,1));
        for (int i=0; i<3; i++)
        {
            lv(i) = linearVelocity[i];
            av(i) = angularVelocity[i];
        }
    }
    retVallist(2) = av;
    retVallist(1) = lv;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxGetOutMessageInfo,args,nargout,"simxGetOutMessageInfo")
{
    const char* funcName="simxGetOutMessageInfo";
// simxGetOutMessageInfo(simxInt clientID,simxInt infoType,simxInt* info)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;

     if (!checkInputArgs(funcName,args,2,1,1))
    {
        retVallist(1) = 0;
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value();
    simxInt infoType = args(1).int_value();
    simxInt info;
    retVal = simxGetOutMessageInfo ( clientID, infoType, &info);
    retVallist(1) = info;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxGetPingTime,args,nargout,"simxGetPingTime")
{
    const char* funcName="simxGetPingTime";
// simxGetPingTime(simxInt clientID,simxInt* pingTime)
    octave_value_list retVallist;
    simxInt retVal=-1;

     if (!checkInputArgs(funcName,args,1,1))
    {
        retVallist(1) = -1;
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value();
    simxInt pingTime=-1;
    retVal = simxGetPingTime ( clientID, &pingTime);
    retVallist(1) = pingTime;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxGetStringParameter,args,nargout,"simxGetStringParameter")
{
    const char* funcName="simxGetStringParameter";
// simxGetStringParameter(simxInt clientID,simxInt paramIdentifier,simxChar** paramValue,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;
    charNDArray pval;

     if (!checkInputArgs(funcName,args,3,1,1,1))
    {
        retVallist(1) = pval;
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value();
    simxInt paramIdentifier = args(1).int_value();
    simxChar* paramValue;
    simxInt operationMode = args(2).int_value();
    retVal = simxGetStringParameter ( clientID, paramIdentifier, &paramValue, operationMode);
    if (retVal == 0)
        pval = paramValue;

    retVallist(1) = pval;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxGetStringSignal,args,nargout,"simxGetStringSignal")
{
    const char* funcName="simxGetStringSignal";
// simxGetStringSignal(simxInt clientID,const simxChar* signalName,simxUChar** signalValue,simxInt* signalLength,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;
    charNDArray sigval;

     if (!checkInputArgs(funcName,args,3,1,stringArg,1))
    {
        retVallist(1) = octave_value(sigval,true,'\'');
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value();
    std::string signalName_ = args(1).string_value();
    const simxChar* signalName = signalName_.c_str();
    simxUChar* signalValue;
    simxInt signalLength;
    simxInt operationMode = args(2).int_value();
    retVal = simxGetStringSignal ( clientID, signalName, &signalValue, &signalLength, operationMode);
    if (retVal == 0)
    {
        sigval.resize(dim_vector(signalLength,1));
        for (int i=0; i<signalLength; i++)
            sigval(i) = signalValue[i];
    }
    retVallist(1) = octave_value(sigval,true,'\'');
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxGetUIButtonProperty,args,nargout,"simxGetUIButtonProperty")
{
    const char* funcName="simxGetUIButtonProperty";
// simxGetUIButtonProperty(simxInt clientID,simxInt uiHandle,simxInt uiButtonID,simxInt* prop,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;

     if (!checkInputArgs(funcName,args,4,1,1,1,1))
    {
        retVallist(1) = 0;
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value();
    simxInt uiHandle = args(1).int_value();
    simxInt uiButtonID = args(2).int_value();
    simxInt prop;
    simxInt operationMode = args(3).int_value();
    retVal = simxGetUIButtonProperty ( clientID, uiHandle, uiButtonID, &prop, operationMode);
    retVallist(1) = prop;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxGetUIEventButton,args,nargout,"simxGetUIEventButton")
{
    const char* funcName="simxGetUIEventButton";
// simxGetUIEventButton(simxInt clientID,simxInt uiHandle,simxInt* uiEventButtonID,simxInt* auxValues,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;
    int32NDArray auxval;

     if (!checkInputArgs(funcName,args,3,1,1,1))
    {
        retVallist(2) = auxval;
        retVallist(1) = -1;
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value();
    simxInt uiHandle = args(1).int_value();
    simxInt uiEventButtonID;
    simxInt auxValues[2];
    simxInt operationMode = args(2).int_value();
    retVal = simxGetUIEventButton ( clientID, uiHandle, &uiEventButtonID, auxValues, operationMode);
    if (retVal == 0)
    {
        auxval.resize(dim_vector(2,1));
        auxval(0) = auxValues[0];
        auxval(1) = auxValues[1];
    }
    retVallist(2) = auxval;
    retVallist(1) = uiEventButtonID;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxGetUIHandle,args,nargout,"simxGetUIHandle")
{
    const char* funcName="simxGetUIHandle";
// simxGetUIHandle(simxInt clientID,const simxChar* uiName,simxInt* handle,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;

     if (!checkInputArgs(funcName,args,3,1,stringArg,1))
    {
        retVallist(1) = -1;
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value();
    std::string uiName_ = args(1).string_value();
    const simxChar* uiName = uiName_.c_str();
    simxInt handle=-1;
    simxInt operationMode = args(2).int_value();
    retVal = simxGetUIHandle ( clientID, uiName, &handle, operationMode);
    retVallist(1) = handle;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxGetUISlider,args,nargout,"simxGetUISlider")
{
    const char* funcName="simxGetUISlider";
// simxGetUISlider(simxInt clientID,simxInt uiHandle,simxInt uiButtonID,simxInt* position,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;

     if (!checkInputArgs(funcName,args,4,1,1,1,1))
    {
        retVallist(1) = 0;
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value();
    simxInt uiHandle = args(1).int_value();
    simxInt uiButtonID = args(2).int_value();
    simxInt position=0;
    simxInt operationMode = args(3).int_value();
    retVal = simxGetUISlider ( clientID, uiHandle, uiButtonID, &position, operationMode);
    retVallist(1) = position;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxGetVisionSensorDepthBuffer,args,nargout,"simxGetVisionSensorDepthBuffer")
{
    const char* funcName="simxGetVisionSensorDepthBuffer";
// simxGetVisionSensorDepthBuffer(simxInt clientID,simxInt sensorHandle,simxInt* resolution,simxFloat** buffer,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;
    int32NDArray res;
    Matrix mat;

     if (!checkInputArgs(funcName,args,3,1,1,1))
    {
        retVallist(2) = mat;
        retVallist(1) = res;
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value();
    simxInt sensorHandle = args(1).int_value();
    simxInt resolution[2];
    simxFloat* buffer;
    simxInt operationMode = args(2).int_value();
    retVal = simxGetVisionSensorDepthBuffer ( clientID, sensorHandle, resolution, &buffer, operationMode);
    if (retVal == 0)
    {
        res.resize(dim_vector(2,1));
        res(0)=resolution[1];
        res(1)=resolution[0];
        mat.resize(resolution[1],resolution[0]);
        int pos = 0;
        for (int i=resolution[1]-1; i>=0; i--)
            for (int j=0; j<resolution[0]; j++)
                mat(i,j) = buffer[pos++];
    }
    retVallist(2) = mat;
    retVallist(1) = res;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxGetVisionSensorImage,args,nargout,"simxGetVisionSensorImage")
{
    const char* funcName="simxGetVisionSensorImage";
// simxGetVisionSensorImage(simxInt clientID,simxInt sensorHandle,simxInt* resolution,simxUChar** image,simxUChar options,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;
    int32NDArray res;
    Matrix matDummy;

     if (!checkInputArgs(funcName,args,3,1,1,1))
    {
        retVallist(1) = matDummy;
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value();
    simxInt sensorHandle = args(1).int_value();
    simxInt resolution[2];
    simxUChar* image;
    simxUChar options = args(2).uint8_scalar_value().value();
    simxInt operationMode = args(3).int_value();
    retVal = simxGetVisionSensorImage ( clientID, sensorHandle, resolution, &image, options, operationMode);


    if (retVal == 0)
    {
        res.resize(dim_vector(2,1));
        res(0)=resolution[0];
        res(1)=resolution[1];
        int pos=0;
        if ((options&1)==0)
        { // RGB
            dim_vector dv (resolution[1],resolution[0],3);
            uint8NDArray mat(dv);
            for (int i=resolution[1]-1; i>=0; i--)
            {
                for (int j=0; j<resolution[0]; j++)
                {
                    for (int k=0;k<3;k++)
                        mat(i,j,k) = image[pos++];
                }
            }
            retVallist(1) = mat;
            retVallist(0) = retVal;
            return retVallist;
        }
        else
        { // Greyscale
            dim_vector dv (2,1);
            dv(0) = resolution[1];
            dv(1) = resolution[0];
            Matrix mat(dv);
            for (int i=resolution[1]-1; i>=0; i--)
            {
                for (int j=0; j<resolution[0]; j++)
                    mat(i,j) = image[pos++];
            }
            retVallist(1) = mat;
            retVallist(0) = retVal;
            return retVallist;
        }
    }

    retVallist(1) = matDummy;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxJointGetForce,args,nargout,"simxJointGetForce")
{
    const char* funcName="simxJointGetForce";
// simxJointGetForce(simxInt clientID,simxInt jointHandle,simxFloat* force,simxInt operationMode)
    octave_value_list retVallist;

    simxInt retVal=simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,3,1,1,1))
    {
        retVallist(1) = 0.0f;
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value();
    simxInt jointHandle = args(1).int_value();
    simxFloat force;
    simxInt operationMode = args(2).int_value();
    retVal = simxGetJointForce ( clientID, jointHandle, &force, operationMode);
    retVallist(1) = force;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxGetJointForce,args,nargout,"simxGetJointForce")
{
    const char* funcName="simxGetJointForce";
// simxGetJointForce(simxInt clientID,simxInt jointHandle,simxFloat* force,simxInt operationMode)
    octave_value_list retVallist;

    simxInt retVal=simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,3,1,1,1))
    {
        retVallist(1) = 0.0f;
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value();
    simxInt jointHandle = args(1).int_value();
    simxFloat force;
    simxInt operationMode = args(2).int_value();
    retVal = simxGetJointForce ( clientID, jointHandle, &force, operationMode);
    retVallist(1) = force;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxGetJointMaxForce,args,nargout,"simxGetJointMaxForce")
{
    const char* funcName="simxGetJointMaxForce";
// simxGetJointMaxForce(simxInt clientID,simxInt jointHandle,simxFloat* force,simxInt operationMode)
    octave_value_list retVallist;

    simxInt retVal=simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,3,1,1,1))
    {
        retVallist(1) = 0.0f;
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value();
    simxInt jointHandle = args(1).int_value();
    simxFloat force;
    simxInt operationMode = args(2).int_value();
    retVal = simxGetJointMaxForce ( clientID, jointHandle, &force, operationMode);
    retVallist(1) = force;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxLoadModel,args,nargout,"simxLoadModel")
{
    const char* funcName="simxLoadModel";
// simxLoadModel(simxInt clientID,const simxChar* modelPathAndName,simxUChar options,simxInt* baseHandle,simxInt operationMode)
    octave_value_list retVallist;

    simxInt retVal = simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,4,1,stringArg,1,1))
    {
        retVallist(1) = -1;
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value();
    std::string modelPathAndName_ = args(1).string_value();
    const simxChar* modelPathAndName = modelPathAndName_.c_str();
    simxUChar options = args(2).uint8_scalar_value().value();
    simxInt baseHandle=-1;
    simxInt operationMode = args(3).int_value();
    retVal = simxLoadModel ( clientID, modelPathAndName, options, &baseHandle, operationMode);
    retVallist(1) = baseHandle;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxLoadScene,args,nargout,"simxLoadScene")
{
    const char* funcName="simxLoadScene";
// simxLoadScene(simxInt clientID,const simxChar* scenePathAndName,simxUChar options,simxInt operationMode)

    octave_value retVal = simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,4,1,stringArg,1,1))
        return retVal;

    simxInt clientID = args(0).int_value();
    std::string scenePathAndName_ = args(1).string_value();
    const simxChar* scenePathAndName = scenePathAndName_.data();
    simxUChar options = args(2).uint8_scalar_value().value();
    simxInt operationMode = args(3).int_value();
    retVal = simxLoadScene ( clientID, scenePathAndName, options, operationMode);
    return retVal;
}

DEFUN_DLD (simxLoadUI,args,nargout,"simxLoadUI")
{
    const char* funcName="simxLoadUI";
// simxLoadUI(simxInt clientID,const simxChar* uiPathAndName,simxUChar options,simxInt* count,simxInt** uiHandles,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;
    int32NDArray uh;

     if (!checkInputArgs(funcName,args,4,1,stringArg,1,1))
    {
        retVallist(1) = uh;
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value();
    std::string uiPathAndName_ = args(1).string_value();
    const simxChar* uiPathAndName = uiPathAndName_.data();
    simxUChar options = args(2).uint8_scalar_value().value();
    simxInt count;
    simxInt* uiHandles;
    simxInt operationMode = args(3).int_value();
    retVal = simxLoadUI ( clientID, uiPathAndName, options, &count, &uiHandles, operationMode);
    if (retVal == 0)
    {
        uh.resize(dim_vector(count,1));
        for (int i=0; i< count; i++)
            uh(i) = uiHandles[i];
    }
    retVallist(1) = uh;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxPauseCommunication,args,nargout,"simxPauseCommunication")
{
    const char* funcName="simxPauseCommunication";
// simxPauseCommunication(simxInt clientID,simxUChar pause)

    octave_value retVal = simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,2,1,1))
        return retVal;

    simxInt clientID = args(0).int_value();
    simxUChar pause = args(1).bool_value();
    retVal = simxPauseCommunication ( clientID, pause);
    return retVal;
}

DEFUN_DLD (simxPauseSimulation,args,nargout,"simxPauseSimulation")
{
    const char* funcName="simxPauseSimulation";
// simxPauseSimulation(simxInt clientID,simxInt operationMode)

    octave_value retVal =simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,2,1,1))
        return retVal;

    simxInt clientID = args(0).int_value();
    simxInt operationMode = args(1).int_value();
    retVal = simxPauseSimulation ( clientID, operationMode);
    return retVal;
}

DEFUN_DLD (simxQuery,args,nargout,"simxQuery")
{
    const char* funcName="simxQuery";
// simxQuery(simxInt clientID,const simxChar* signalName,const simxUChar* signalValue,simxInt signalLength,const simxChar* retSignalName,simxUChar** retSignalValue,simxInt* retSignalLength,simxInt timeOutInMs)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;
    charNDArray sigval;

     if (!checkInputArgs(funcName,args,5,1,stringArg,stringArg,stringArg,1))
    {
        retVallist(1) = octave_value(sigval,true,'\'');
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value();
    std::string signalName_ = args(1).string_value();
    const simxChar* signalName = signalName_.c_str();
    charNDArray signalValue_=args(2).char_array_value();
    const simxUChar* signalValue = (simxUChar*)signalValue_.data();
    simxInt signalLength = args(2).length();
    std::string retSignalName_ = args(3).string_value();
    const simxChar* retSignalName = retSignalName_.c_str();
    simxUChar* retSignalValue;
    simxInt retSignalLength;
    simxInt timeOutInMs = args(4).int_value();
    retVal = simxQuery ( clientID, signalName, signalValue, signalLength,retSignalName, &retSignalValue, &retSignalLength, timeOutInMs);
    if (retVal == 0)
    {
        sigval.resize(dim_vector(retSignalLength,1));
        for (int i=0; i<retSignalLength; i++)
            sigval(i) = retSignalValue[i];
    }
    retVallist(1) = octave_value(sigval,true,'\'');
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxReadCollision,args,nargout,"simxReadCollision")
{
    const char* funcName="simxReadCollision";
// simxReadCollision(simxInt clientID,simxInt collisionObjectHandle,simxUChar* collisionState,simxInt operationMode)
    octave_value_list retVallist;

    simxInt retVal=simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,3,1,1,1))
    {
        retVallist(1) = 0;
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value();
    simxInt collisionObjectHandle = args(1).int_value();
    simxUChar collisionState;
    simxInt operationMode = args(2).int_value();
    retVal = simxReadCollision ( clientID, collisionObjectHandle, &collisionState, operationMode);
    retVallist(1) = collisionState;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxReadDistance,args,nargout,"simxReadDistance")
{
// simxReadDistance(simxInt clientID,simxInt distanceObjectHandle,simxFloat* minimumDistance,simxInt operationMode)
    const char* funcName="simxReadDistance";
    octave_value_list retVallist;

    simxInt retVal=simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,3,1,1,1))
    {
        retVallist(1) = 0.0f;
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value();
    simxInt distanceObjectHandle = args(1).int_value();
    simxFloat minimumDistance;
    simxInt operationMode = args(2).int_value();
    retVal = simxReadDistance ( clientID, distanceObjectHandle, &minimumDistance, operationMode);
    retVallist(1) = minimumDistance;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxReadForceSensor,args,nargout,"simxReadForceSensor")
{
    const char* funcName="simxReadForceSensor";
// simxReadForceSensor(simxInt clientID,simxInt forceSensorHandle,simxUChar* state,simxFloat* forceVector,simxFloat* torqueVector,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;
    FloatNDArray fv,tv;
    simxUChar state=0;

     if (!checkInputArgs(funcName,args,3,1,1,1))
    {
        retVallist(3) = tv;
        retVallist(2) = fv;
        retVallist(1) = state;
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value() ;
    simxInt forceSensorHandle = args(1).int_value();
    simxFloat forceVector[3];
    simxFloat torqueVector[3];
    simxInt operationMode = args(2).int_value();
    retVal = simxReadForceSensor ( clientID, forceSensorHandle, &state, forceVector, torqueVector, operationMode);
    if (retVal == 0)
    {
        fv.resize(dim_vector(3,1));
        tv.resize(dim_vector(3,1));
        for (int i=0; i<3; i++)
        {
            fv(i) = forceVector[i];
            tv(i) = torqueVector[i];
        }
    }
    retVallist(3) = tv;
    retVallist(2) = fv;
    retVallist(1) = state;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxReadProximitySensor,args,nargout,"simxReadProximitySensor")
{
    const char* funcName="simxReadProximitySensor";
// simxReadProximitySensor(simxInt clientID,simxInt sensorHandle,simxUChar* detectionState,simxFloat* detectedPoint,simxInt* detectedObjectHandle,simxFloat* detectedSurfaceNormalVector,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;
    FloatNDArray dp,dv;

     if (!checkInputArgs(funcName,args,3,1,1,1))
    {
        retVallist(4) = dv;
        retVallist(3) = -1;
        retVallist(2) = dp;
        retVallist(1) = 0;
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value();
    simxInt sensorHandle = args(1).int_value();
    simxUChar detectionState = 0;
    simxFloat detectedPoint[3];
    simxInt detectedObjectHandle = -1;
    simxFloat detectedSurfaceNormalVector[3];
    simxInt operationMode = args(2).int_value();
    retVal = simxReadProximitySensor ( clientID, sensorHandle, &detectionState, detectedPoint, &detectedObjectHandle, detectedSurfaceNormalVector, operationMode);
    if (retVal == 0)
    {
        dp.resize(dim_vector(3,1));
        dv.resize(dim_vector(3,1));
        for (int i=0; i<3; i++)
        {
            dp(i) = detectedPoint[i];
            dv(i) = detectedSurfaceNormalVector[i];
        }
    }
    retVallist(4) = dv;
    retVallist(3) = detectedObjectHandle;
    retVallist(2) = dp;
    retVallist(1) = detectionState;
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxReadVisionSensor,args,nargout,"simxReadVisionSensor")
{
    const char* funcName="simxReadVisionSensor";
// simxReadVisionSensor(simxInt clientID,simxInt sensorHandle,simxUChar* detectionState,simxFloat** auxValues,simxInt** auxValuesCount,simxInt operationMode)
    octave_value_list retVallist;
    simxInt retVal=simx_return_local_error_flag;

    simxUChar detectionState = 0;
    FloatNDArray packets;
    int32NDArray packetSizes;

     if (!checkInputArgs(funcName,args,3,1,1,1))
    {
        retVallist(3)= packetSizes;
        retVallist(2)= packets;
        retVallist(1)= (detectionState!=0);
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value();
    simxInt sensorHandle = args(1).int_value();
    simxFloat* auxValues;
    simxInt* auxValuesCount;
    simxInt operationMode = args(2).int_value();
    retVal = simxReadVisionSensor ( clientID, sensorHandle, &detectionState, &auxValues, &auxValuesCount, operationMode);
    if (retVal == 0)
    {
        int count = auxValuesCount[0];
        packetSizes.resize(dim_vector(count,1));
        int packetTotSize=0;
        for (int i=1; i<=count; i++)
        {
            packetSizes(i-1) = auxValuesCount[i];
            packetTotSize+=auxValuesCount[i];
        }
        packets.resize(dim_vector(packetTotSize,1));
        for (int i=0; i<packetTotSize; i++)
            packets(i)=auxValues[i];
        simxReleaseBuffer((simxUChar*)auxValues);
        simxReleaseBuffer((simxUChar*)auxValuesCount);
    }
    retVallist(3)= packetSizes;
    retVallist(2)= packets;
    retVallist(1)= (detectionState!=0);
    retVallist(0) = retVal;
    return retVallist;
}

DEFUN_DLD (simxRemoveObject,args,nargout,"simxRemoveObject")
{
    const char* funcName="simxRemoveObject";
// simxRemoveObject(simxInt clientID,simxInt objectHandle,simxInt operationMode)

    octave_value retVal =simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,3,1,1,1))
        return retVal;

    simxInt clientID = args(0).int_value();
    simxInt objectHandle = args(1).int_value();
    simxInt operationMode = args(2).int_value();
    retVal = simxRemoveObject ( clientID, objectHandle, operationMode);
    return retVal;
}

DEFUN_DLD (simxRemoveModel,args,nargout,"simxRemoveModel")
{
    const char* funcName="simxRemoveModel";
// simxRemoveModel(simxInt clientID,simxInt objectHandle,simxInt operationMode)

    octave_value retVal =simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,3,1,1,1))
        return retVal;

    simxInt clientID = args(0).int_value();
    simxInt objectHandle = args(1).int_value();
    simxInt operationMode = args(2).int_value();
    retVal = simxRemoveModel ( clientID, objectHandle, operationMode);
    return retVal;
}

DEFUN_DLD (simxRemoveUI,args,nargout,"simxRemoveUI")
{
    const char* funcName="simxRemoveUI";
// simxRemoveUI(simxInt clientID,simxInt uiHandle,simxInt operationMode)

    octave_value retVal =simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,3,1,1,1))
        return retVal;

    simxInt clientID = args(0).int_value();
    simxInt uiHandle = args(1).int_value();
    simxInt operationMode = args(2).int_value();
    retVal = simxRemoveUI ( clientID, uiHandle, operationMode);
    return retVal;
}

DEFUN_DLD (simxSetArrayParameter,args,nargout,"simxSetArrayParameter")
{
    const char* funcName="simxSetArrayParameter";
// simxSetArrayParameter(simxInt clientID,simxInt paramIdentifier,const simxFloat* paramValues,simxInt operationMode)

    octave_value retVal=simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,4,1,1,3,1))
        return retVal;

    simxInt clientID = args(0).int_value();
    simxInt paramIdentifier = args(1).int_value();
    simxFloat paramValues[3];

    FloatNDArray pv = args(2).float_array_value();
    paramValues[0] = pv(0);
    paramValues[1] = pv(1);
    paramValues[2] = pv(2);
    simxInt operationMode = args(3).int_value();
    retVal = simxSetArrayParameter ( clientID, paramIdentifier, paramValues, operationMode);
    return retVal;
}

DEFUN_DLD (simxSetBooleanParameter,args,nargout,"simxSetBooleanParameter")
{
    const char* funcName="simxSetBooleanParameter";
// simxSetBooleanParameter(simxInt clientID,simxInt paramIdentifier,simxUChar paramValue,simxInt operationMode)

    octave_value retVal=simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,4,1,1,1,1))
        return retVal;

    simxInt clientID = args(0).int_value();
    simxInt paramIdentifier = args(1).int_value();
    simxUChar paramValue = args(2).bool_value();
    simxInt operationMode = args(3).int_value();
    retVal = simxSetBooleanParameter ( clientID, paramIdentifier, paramValue, operationMode);
    return retVal;
}

DEFUN_DLD (simxSetFloatingParameter,args,nargout,"simxSetFloatingParameter")
{
    const char* funcName="simxSetFloatingParameter";
// simxSetFloatingParameter(simxInt clientID,simxInt paramIdentifier,simxFloat paramValue,simxInt operationMode)

    octave_value retVal=simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,4,1,1,1,1))
        return retVal;

    simxInt clientID = args(0).int_value();
    simxInt paramIdentifier = args(1).int_value();
    simxFloat paramValue = args(2).float_scalar_value();
    simxInt operationMode = args(3).int_value();
    retVal = simxSetFloatingParameter ( clientID, paramIdentifier, paramValue, operationMode);
    return retVal;
}

DEFUN_DLD (simxSetFloatSignal,args,nargout,"simxSetFloatSignal")
{
    const char* funcName="simxSetFloatSignal";
// simxSetFloatSignal(simxInt clientID,const simxChar* signalName,simxFloat signalValue,simxInt operationMode)

    octave_value retVal=simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,4,1,stringArg,1,1))
        return retVal;

    simxInt clientID = args(0).int_value();
    std::string signalName_ = args(1).string_value();
    const simxChar* signalName = signalName_.c_str();
    simxFloat signalValue = args(2).float_scalar_value();
    simxInt operationMode = args(3).int_value();
    retVal = simxSetFloatSignal ( clientID, signalName, signalValue, operationMode);
    return retVal;
}

DEFUN_DLD (simxSetIntegerParameter,args,nargout,"simxSetIntegerParameter")
{
    const char* funcName="simxSetIntegerParameter";
//simxSetIntegerParameter(simxInt clientID,simxInt paramIdentifier,simxInt paramValue,simxInt operationMode)
    octave_value_list retVallist;

    octave_value retVal=simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,4,1,1,1,1))
    {
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value() ;
    simxInt paramIdentifier = args(1).int_value();
    simxInt paramValue = args(2).int_value();
    simxInt operationMode = args(3).int_value();

    retVal = simxSetIntegerParameter(clientID,paramIdentifier,paramValue,operationMode);

    retVallist(0) = retVal;

    return retVallist;
}

DEFUN_DLD (simxSetIntegerSignal,args,nargout,"simxSetIntegerSignal")
{
    const char* funcName="simxSetIntegerSignal";
// simxSetIntegerSignal(simxInt clientID,const simxChar* signalName,simxInt signalValue,simxInt operationMode)
    octave_value_list retVallist;

    octave_value retVal=simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,4,1,stringArg,1,1))
    {
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value() ;
    std::string signalName_ = args(1).string_value();
    const simxChar* signalName = signalName_.c_str();
    simxInt signalValue = args(2).int_value();
    simxInt operationMode = args(3).int_value();

    retVal = simxSetIntegerSignal(clientID,signalName,signalValue,operationMode);

    retVallist(0) = retVal;

    return retVallist;
}

DEFUN_DLD (simxSetJointForce,args,nargout,"simxSetJointForce")
{
    const char* funcName="simxSetJointMaxForce";
// simxSetJointMaxForce(simxInt clientID,simxInt jointHandle,simxFloat force,simxInt operationMode)
    octave_value_list retVallist;

    octave_value retVal=simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,4,1,1,1,1))
    {
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value() ;
    simxInt jointHandle = args(1).int_value();
    simxFloat force = args(2).float_value();
    simxInt operationMode = args(3).int_value();

    retVal = simxSetJointMaxForce(clientID,jointHandle,force,operationMode);

    retVallist(0) = retVal;

    return retVallist;
}

DEFUN_DLD (simxSetJointMaxForce,args,nargout,"simxSetJointMaxForce")
{
    const char* funcName="simxSetJointMaxForce";
// simxSetJointMaxForce(simxInt clientID,simxInt jointHandle,simxFloat force,simxInt operationMode)
    octave_value_list retVallist;

    octave_value retVal=simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,4,1,1,1,1))
    {
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value() ;
    simxInt jointHandle = args(1).int_value();
    simxFloat force = args(2).float_value();
    simxInt operationMode = args(3).int_value();

    retVal = simxSetJointMaxForce(clientID,jointHandle,force,operationMode);

    retVallist(0) = retVal;

    return retVallist;
}

DEFUN_DLD (simxSetJointPosition,args,nargout,"simxSetJointPosition")
{
    const char* funcName="simxSetJointPosition";
// simxSetJointPosition(simxInt clientID,simxInt jointHandle,simxFloat position,simxInt operationMode)
    octave_value_list retVallist;

    octave_value retVal=simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,4,1,1,1,1))
    {
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value() ;
    simxInt jointHandle = args(1).int_value();
    simxFloat position = args(2).float_value();
    simxInt operationMode = args(3).int_value();

    retVal = simxSetJointPosition(clientID,jointHandle,position,operationMode);

    retVallist(0) = retVal;

    return retVallist;
}

DEFUN_DLD (simxSetJointTargetPosition,args,nargout,"simxSetJointTargetPosition")
{
    const char* funcName="simxSetJointTargetPosition";
// simxSetJointTargetPosition(simxInt clientID,simxInt jointHandle,simxFloat targetPosition,simxInt operationMode)
    octave_value_list retVallist;

    octave_value retVal=simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,4,1,1,1,1))
    {
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value() ;
    simxInt jointHandle = args(1).int_value();
    simxFloat targetPosition = args(2).float_value();
    simxInt operationMode = args(3).int_value();

    retVal = simxSetJointTargetPosition(clientID,jointHandle,targetPosition,operationMode);

    retVallist(0) = retVal;

    return retVallist;
}

DEFUN_DLD (simxSetJointTargetVelocity,args,nargout,"simxSetJointTargetVelocity")
{
    const char* funcName="simxSetJointTargetVelocity";
//simxSetJointTargetVelocity(simxInt clientID,simxInt jointHandle,simxFloat targetVelocity,simxInt operationMode)
    octave_value_list retVallist;

    octave_value retVal=simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,4,1,1,1,1))
    {
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value() ;
    simxInt jointHandle = args(1).int_value();
    simxFloat targetVelocity = args(2).float_value();
    simxInt operationMode = args(3).int_value();

    retVal = simxSetJointTargetVelocity(clientID,jointHandle,targetVelocity,operationMode);

    retVallist(0) = retVal;

    return retVallist;
}

DEFUN_DLD (simxSetModelProperty,args,nargout,"simxSetModelProperty")
{
    const char* funcName="simxSetModelProperty";
//simxSetModelProperty(simxInt clientID,simxInt objectHandle,simxInt prop,simxInt operationMode)
    octave_value_list retVallist;

    octave_value retVal=simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,4,1,1,1,1))
    {
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value() ;
    simxInt objectHandle = args(1).int_value();
    simxInt prop = args(2).int_value();
    simxInt operationMode = args(3).int_value();

    retVal = simxSetModelProperty(clientID,objectHandle,prop,operationMode);

    retVallist(0) = retVal;

    return retVallist;
}

DEFUN_DLD (simxSetObjectFloatParameter,args,nargout,"simxSetObjectFloatParameter")
{
    const char* funcName="simxSetObjectFloatParameter";
//simxSetObjectFloatParameter(simxInt clientID,simxInt objectHandle,simxInt parameterID,simxFloat parameterValue,simxInt operationMode)
    octave_value_list retVallist;

    octave_value retVal=simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,5,1,1,1,1,1))
    {
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value() ;
    simxInt objectHandle = args(1).int_value();
    simxInt parameterID = args(2).int_value();
    simxFloat parameterValue = args(3).float_value();
    simxInt operationMode = args(4).int_value();

    retVal = simxSetObjectFloatParameter(clientID,objectHandle,parameterID,parameterValue,operationMode);

    retVallist(0) = retVal;

    return retVallist;
}

DEFUN_DLD (simxSetObjectIntParameter,args,nargout,"simxSetObjectIntParameter")
{
    const char* funcName="simxSetObjectIntParameter";
//simxSetObjectIntParameter(simxInt clientID,simxInt objectHandle,simxInt parameterID,simxInt parameterValue,simxInt operationMode)
    octave_value_list retVallist;

    octave_value retVal=simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,5,1,1,1,1,1))
    {
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value() ;
    simxInt objectHandle = args(1).int_value();
    simxInt parameterID = args(2).int_value();
    simxInt parameterValue = args(3).int_value();
    simxInt operationMode = args(4).int_value();

    retVal = simxSetObjectIntParameter(clientID,objectHandle,parameterID,parameterValue,operationMode);

    retVallist(0) = retVal;

    return retVallist;
}

DEFUN_DLD (simxSetObjectOrientation,args,nargout,"simxSetObjectOrientation")
{
    const char* funcName="simxSetObjectOrientation";
//simxSetObjectOrientation(simxInt clientID,simxInt objectHandle,simxInt relativeToObjectHandle,const simxFloat* eulerAngles,simxInt operationMode)
    octave_value_list retVallist;

    octave_value retVal=simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,5,1,1,1,3,1))
    {
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value() ;
    simxInt objectHandle = args(1).int_value();
    simxInt relativeToObjectHandle = args(2).int_value();

    FloatNDArray fArr = args(3).float_array_value();

    float eulerAngles[3];
    eulerAngles[0] = fArr(0);
    eulerAngles[1] = fArr(1);
    eulerAngles[2] = fArr(2);

    simxInt operationMode = args(4).int_value();

    retVal = simxSetObjectOrientation(clientID,objectHandle,relativeToObjectHandle,eulerAngles,operationMode);

    retVallist(0) = retVal;

    return retVallist;
}

DEFUN_DLD (simxSetObjectQuaternion,args,nargout,"simxSetObjectQuaternion")
{
    const char* funcName="simxSetObjectQuaternion";
//simxSetObjectQuaternion(simxInt clientID,simxInt objectHandle,simxInt relativeToObjectHandle,const simxFloat* quaternion,simxInt operationMode)
    octave_value_list retVallist;

    octave_value retVal=simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,5,1,1,1,4,1))
    {
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value() ;
    simxInt objectHandle = args(1).int_value();
    simxInt relativeToObjectHandle = args(2).int_value();

    FloatNDArray fArr = args(3).float_array_value();

    float quaternion[4];
    quaternion[0] = fArr(0);
    quaternion[1] = fArr(1);
    quaternion[2] = fArr(2);
    quaternion[3] = fArr(3);

    simxInt operationMode = args(4).int_value();

    retVal = simxSetObjectQuaternion(clientID,objectHandle,relativeToObjectHandle,quaternion,operationMode);

    retVallist(0) = retVal;

    return retVallist;
}

DEFUN_DLD (simxSetObjectParent,args,nargout,"simxSetObjectParent")
{
    const char* funcName="simxSetObjectParent";
//simxSetObjectParent(simxInt clientID,simxInt objectHandle,simxInt parentObject,simxUChar keepInPlace,simxInt operationMode)
    octave_value_list retVallist;

    octave_value retVal=simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,5,1,1,1,1,1))
    {
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value() ;
    simxInt objectHandle = args(1).int_value();
    simxInt parentObject = args(2).int_value();
    simxUChar keepInPlace = args(3).bool_value();
    simxInt operationMode = args(4).int_value();

    retVal = simxSetObjectParent(clientID,objectHandle,parentObject,keepInPlace,operationMode);

    retVallist(0) = retVal;

    return retVallist;
}

DEFUN_DLD (simxSetObjectPosition,args,nargout,"simxSetObjectPosition")
{
    const char* funcName="simxSetObjectPosition";
//simxSetObjectPosition(simxInt clientID,simxInt objectHandle,simxInt relativeToObjectHandle,const simxFloat* position,simxInt operationMode)
    octave_value_list retVallist;

    octave_value retVal=simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,5,1,1,1,3,1))
    {
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value() ;
    simxInt objectHandle = args(1).int_value();
    simxInt relativeToObjectHandle = args(2).int_value();

    FloatNDArray fArr = args(3).float_array_value();

    float position[3];
    position[0] = fArr(0);
    position[1] = fArr(1);
    position[2] = fArr(2);

    simxInt operationMode = args(4).int_value();

    retVal = simxSetObjectPosition(clientID,objectHandle,relativeToObjectHandle,position,operationMode);

    retVallist(0) = retVal;

    return retVallist;
}

DEFUN_DLD (simxSetObjectSelection,args,nargout,"simxSetObjectSelection")
{
    const char* funcName="simxSetObjectSelection";
//simxSetObjectSelection(simxInt clientID,const simxInt* objectHandles,simxInt objectCount,simxInt operationMode)
    octave_value_list retVallist;

    octave_value retVal=simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,3,1,anyArg,1))
    {
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value() ;
    int32NDArray objectHandles_;
    const simxInt* objectHandles=NULL;
    if (args(1).length()!=0)
    {
        objectHandles_=args(1).int32_array_value();
        objectHandles=(const simxInt*)objectHandles_.data();
    }
    simxInt objectCount = args(1).length();
    simxInt operationMode = args(2).int_value();

    retVal = simxSetObjectSelection(clientID,objectHandles,objectCount,operationMode);

    retVallist(0) = retVal;
    retVallist(1) = objectHandles;

    return retVallist;
}

DEFUN_DLD (simxSetSphericalJointMatrix,args,nargout,"simxSetSphericalJointMatrix")
{
    const char* funcName="simxSetSphericalJointMatrix";
//simxSetSphericalJointMatrix(simxInt clientID,simxInt jointHandle,simxFloat* matrix,simxInt operationMode)
    octave_value_list retVallist;

    octave_value retVal=simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,4,1,1,12,1))
    {
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value() ;
    simxInt jointHandle = args(1).int_value();

    FloatNDArray fArr = args(2).float_array_value();

    float matrix[12];
    for(int i=0;i<12;i++)
        matrix[i] = fArr(i);

    simxInt operationMode = args(3).int_value();

    retVal = simxSetSphericalJointMatrix(clientID,jointHandle,matrix,operationMode);

    retVallist(0) = retVal;

    return retVallist;
}

DEFUN_DLD (simxSetStringSignal,args,nargout,"simxSetStringSignal")
{
    const char* funcName="simxSetStringSignal";
//simxSetStringSignal(simxInt clientID,const simxChar* signalName,const simxUChar* signalValue,simxInt signalLength,simxInt operationMode)
    octave_value_list retVallist;

    octave_value retVal=simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,4,1,stringArg,stringArg,1))
    {
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value() ;
    std::string signalName_ = args(1).string_value();
    const simxChar* signalName = signalName_.c_str();

    charNDArray signalValue_ = args(2).char_array_value();
    const simxUChar* signalValue = (const simxUChar*)signalValue_.data();
    simxInt signalLength = signalValue_.length();


    simxInt operationMode = args(3).int_value();

    retVal = simxSetStringSignal(clientID,signalName,signalValue,signalLength,operationMode);

    retVallist(0) = retVal;

    return retVallist;
}

DEFUN_DLD (simxSetUIButtonLabel,args,nargout,"simxSetUIButtonLabel")
{
    const char* funcName="simxSetUIButtonLabel";
//simxSetUIButtonLabel(simxInt clientID,simxInt uiHandle,simxInt uiButtonID,const simxChar* upStateLabel,const simxChar* downStateLabel,simxInt operationMode)
    octave_value_list retVallist;

    octave_value retVal=simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,6,1,1,1,stringArg,stringArg,1))
    {
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value() ;
    simxInt uiHandle = args(1).int_value();
    simxInt uiButtonID = args(2).int_value();
    std::string upStateLabel_ = args(3).string_value();
    const simxChar* upStateLabel = upStateLabel_.c_str();
    std::string downStateLabel_ = args(4).string_value();
    const simxChar* downStateLabel = downStateLabel_.c_str();
    simxInt operationMode = args(5).int_value();

    retVal = simxSetUIButtonLabel(clientID,uiHandle,uiButtonID,upStateLabel,downStateLabel,operationMode);

    retVallist(0) = retVal;

    return retVallist;
}

DEFUN_DLD (simxSetUIButtonProperty,args,nargout,"simxSetUIButtonProperty")
{
    const char* funcName="simxSetUIButtonProperty";
//simxSetUIButtonProperty(simxInt clientID,simxInt uiHandle,simxInt uiButtonID,simxInt prop,simxInt operationMode)
    octave_value_list retVallist;

    octave_value retVal=simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,5,1,1,1,1,1))
    {
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value() ;
    simxInt uiHandle = args(1).int_value();
    simxInt uiButtonID = args(2).int_value();
    simxInt prop = args(3).int_value();
    simxInt operationMode = args(4).int_value();

    retVal = simxSetUIButtonProperty(clientID,uiHandle,uiButtonID,prop,operationMode);

    retVallist(0) = retVal;

    return retVallist;
}

DEFUN_DLD (simxSetUISlider,args,nargout,"simxSetUISlider")
{
    const char* funcName="simxSetUISlider";
//simxSetUISlider(simxInt clientID,simxInt uiHandle,simxInt uiButtonID,simxInt position,simxInt operationMode)
    octave_value_list retVallist;

    octave_value retVal;
     if (!checkInputArgs(funcName,args,5,1,1,1,1,1))
    {
        retVallist(0) = retVal;
        return retVallist;
    }

    simxInt clientID = args(0).int_value() ;
    simxInt uiHandle = args(1).int_value();
    simxInt uiButtonID = args(2).int_value();
    simxInt position = args(3).int_value();
    simxInt operationMode = args(4).int_value();

    retVal = simxSetUISlider(clientID,uiHandle,uiButtonID,position,operationMode);

    retVallist(0) = retVal;

    return retVallist;
}

DEFUN_DLD (simxStartSimulation,args,nargout,"simxStartSimulation")
{
    const char* funcName="simxStartSimulation";
//simxStartSimulation(simxInt clientID,simxInt operationMode)

    octave_value retVal=simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,2,1,1))
        return retVal;

    simxInt clientID = args(0).int_value() ;
    simxInt operationMode = args(1).int_value();

    retVal = simxStartSimulation(clientID,operationMode);

    return retVal;
}

DEFUN_DLD (simxStopSimulation,args,nargout,"simxStopSimulation")
{
    const char* funcName="simxStopSimulation";
//simxStopSimulation(simxInt clientID,simxInt operationMode)

    octave_value retVal=simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,2,1,1))
        return retVal;

    simxInt clientID = args(0).int_value() ;
    simxInt operationMode = args(1).int_value();

    retVal = simxStopSimulation(clientID,operationMode);

    return retVal;
}

DEFUN_DLD (simxSynchronous,args,nargout,"simxSynchronous")
{
    const char* funcName="simxSynchronous";
//simxSynchronous(simxInt clientID,simxUChar enable)

    octave_value retVal=simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,2,1,1))
        return retVal;

    simxInt clientID = args(0).int_value() ;
    simxUChar enable = args(1).bool_value();

    retVal = simxSynchronous(clientID,enable);

    return retVal;
}

DEFUN_DLD (simxSynchronousTrigger,args,nargout,"simxSynchronousTrigger")
{
    const char* funcName="simxSynchronousTrigger";
//simxSynchronousTrigger(simxInt clientID)

    octave_value retVal=simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,1,1))
        return retVal;

    simxInt clientID = args(0).int_value() ;

    retVal = simxSynchronousTrigger(clientID);

    return retVal;
}

DEFUN_DLD (simxTransferFile,args,nargout,"simxTransferFile")
{
    const char* funcName="simxTransferFile";
//simxTransferFile(simxInt clientID,const simxChar* filePathAndName,const simxChar* fileName_serverSide,simxInt timeOut,simxInt operationMode)

    octave_value retVal=simx_return_local_error_flag;
     if (!checkInputArgs(funcName,args,5,1,stringArg,stringArg,1,1))
        return retVal;

    simxInt clientID = args(0).int_value() ;
    std::string filePathAndName_ = args(1).string_value();
    const simxChar* filePathAndName = filePathAndName_.c_str();
    std::string fileName_serverSide_ = args(2).string_value();
    const simxChar* fileName_serverSide = fileName_serverSide_.c_str();
    simxInt timeOut = args(3).int_value();
    simxInt operationMode = args(4).int_value();

    retVal = simxTransferFile(clientID,filePathAndName,fileName_serverSide,timeOut,operationMode);

    return retVal;
}

DEFUN_DLD (simxSetVisionSensorImage,args,nargout,"simxSetVisionSensorImage")
{
    const char* funcName="simxSetVisionSensorImage";
// simxSetVisionSensorImage(simxInt clientID,simxInt sensorHandle,simxUChar* image,simxInt bufferSize,simxUChar options,simxInt operationMode)

    octave_value retVal=simx_return_local_error_flag;

     if (!checkInputArgs(funcName,args,4,1,1,anyArg,1))
        return retVal;

    simxInt clientID = args(0).int_value();
    simxInt sensorHandle = args(1).int_value();
    simxInt operationMode = args(3).int_value();
    charNDArray mat=args(2).char_array_value();
    dim_vector dim=mat.dims();
    int resolution[2]={dim(1),dim(0)};
    bool err=false;
    err|=((dim.length()!=2)&&(dim.length()!=3));
    bool isRgb=(dim.length()==3);
    if (!err)
    {
        if (isRgb)
            err|=((dim(0)==0)||(dim(1)==0)||(dim(2)!=3));
        else
            err|=((dim(0)==0)||(dim(1)==0));
    }
    if (err)
    {
        octave_stdout << "Error in remote API function ";
        octave_stdout << funcName;
        octave_stdout << ":\n--> invalid arguments\n";
        return retVal;
    }
    simxUChar* buffer;
    int bufferSize;
    int pos=0;
    int options=0;
    if (isRgb)
    { // RGB
        bufferSize=resolution[0]*resolution[1]*3;
        buffer=new simxUChar[bufferSize];
        for (int i=resolution[1]-1; i>=0; i--)
        {
            for (int j=0; j<resolution[0]; j++)
            {
                for (int k=0;k<3;k++)
                    buffer[pos++] = (simxUChar)mat(i,j,k);
            }
        }
    }
    else
    { // greyscale
        bufferSize=resolution[0]*resolution[1];
        buffer=new simxUChar[bufferSize];
        options|=1;
        for (int i=resolution[1]-1; i>=0; i--)
        {
            for (int j=0; j<resolution[0]; j++)
                buffer[pos++] = (simxUChar)mat(i,j);
        }
    }
    retVal=simxSetVisionSensorImage(clientID,sensorHandle,buffer,bufferSize,options,operationMode);
    delete buffer;
    return retVal;
}

DEFUN_DLD (simxPackFloats,args,nargout,"simxPackFloats")
{
    const char* funcName="simxPackFloats";
    charNDArray charArray;
     if (!checkInputArgs(funcName,args,1,0))
        return octave_value(charArray,true,'\'');

    FloatNDArray floatarr = args(0).float_array_value();
    charArray.resize(dim_vector(floatarr.length()*4,1));
    for (int i=0; i<floatarr.length(); i++)
    {
        float floatV=floatarr(i);
        charArray(4*i+0)=((char*)&floatV)[0];
        charArray(4*i+1)=((char*)&floatV)[1];
        charArray(4*i+2)=((char*)&floatV)[2];
        charArray(4*i+3)=((char*)&floatV)[3];
    }

    return octave_value(charArray,true,'\'');
}

DEFUN_DLD (simxPackInts,args,nargout,"simxPackInts")
{
    const char* funcName="simxPackInts";
    charNDArray charArray;

     if (!checkInputArgs(funcName,args,1,0))
        return octave_value(charArray,true,'\'');

    int32NDArray intarr = args(0).int32_array_value();
    charArray.resize(dim_vector(intarr.length()*4,1));
    for (int i=0; i<intarr.length(); i++)
    {
        int intV=intarr(i);
        charArray(4*i+0)=((char*)&intV)[0];
        charArray(4*i+1)=((char*)&intV)[1];
        charArray(4*i+2)=((char*)&intV)[2];
        charArray(4*i+3)=((char*)&intV)[3];
    }
    return octave_value(charArray,true,'\'');
}

DEFUN_DLD (simxUnpackFloats,args,nargout,"simxUnpackFloats")
{
    const char* funcName="simxUnpackFloats";
    octave_value retVal;
    FloatNDArray floatArray;
    retVal=floatArray;

     if (!checkInputArgs(funcName,args,1,stringArg))
        return retVal;

    charNDArray txt_=args(0).char_array_value();
    const simxUChar* txt=(const simxUChar*)txt_.data();
    simxInt txtLength = args(0).length();

    floatArray.resize(dim_vector(txtLength/4,1));
    for (int i=0;i<int(txtLength/4);i++)
        floatArray(i)=((float*)(txt+4*i))[0];

    retVal=floatArray;
    return retVal;
}

DEFUN_DLD (simxUnpackInts,args,nargout,"simxUnpackInts")
{
    const char* funcName="simxUnpackInts";
    octave_value retVal;
    int32NDArray intArray;
    retVal=intArray;

     if (!checkInputArgs(funcName,args,1,stringArg))
        return retVal;

    charNDArray txt_=args(0).char_array_value();
    const simxUChar* txt=(const simxUChar*)txt_.data();
    simxInt txtLength = args(0).length();

    intArray.resize(dim_vector(txtLength/4,1));
    for (int i=0;i<int(txtLength/4);i++)
        intArray(i)=((int*)(txt+4*i))[0];

    retVal=intArray;
    return retVal;
}
