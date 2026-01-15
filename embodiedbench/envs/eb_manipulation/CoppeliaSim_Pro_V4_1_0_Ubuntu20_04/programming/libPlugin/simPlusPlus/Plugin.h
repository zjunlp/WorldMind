#ifndef SIMPLUSPLUS_PLUGIN_H_INCLUDED
#define SIMPLUSPLUS_PLUGIN_H_INCLUDED

#if __cplusplus <= 199711L
    #error simPlusPlus needs at least a C++11 compliant compiler
#endif

#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <boost/format.hpp>

#include "simLib.h"

#ifdef _WIN32
	#define SIM_DLLEXPORT extern "C" __declspec(dllexport)
#endif /* _WIN32 */
#if defined (__linux) || defined (__APPLE__)
	#define SIM_DLLEXPORT extern "C"
#endif /* __linux || __APPLE__ */

#ifdef _WIN32
    #ifdef QT_COMPIL
        #include <direct.h>
    #else
        #include <shlwapi.h>
        #pragma comment(lib, "Shlwapi.lib")
    #endif
#endif /* _WIN32 */
#if defined (__linux) || defined (__APPLE__)
    #include <unistd.h>
#define _stricmp strcasecmp
#endif /* __linux || __APPLE__ */

namespace sim
{
    struct InstancePassFlags
    {
        bool objectsErased;
        bool objectsCreated;
        bool modelLoaded;
        bool sceneLoaded;
        bool undoCalled;
        bool redoCalled;
        bool sceneSwitched;
        bool editModeActive;
        bool objectsScaled;
        bool selectionStateChanged;
        bool keyPressed;
        bool simulationStarted;
        bool simulationEnded;
        bool scriptCreated;
        bool scriptErased;
    };

    class Plugin
    {
    public:
        void setName(const std::string &name);
        std::string name() const;
        void setExtVersion(const std::string &s);
        void setExtVersion(int i);
        void setBuildDate(const std::string &s);
        void setVerbosity(int i);
        int getVerbosity();
        void init();
        virtual void onStart();
        virtual void onEnd();
        virtual void * onMessage(int message, int *auxData, void *customData, int *replyData) final;
        virtual LIBRARY loadSimLibrary();

        virtual void onInstancePass(const InstancePassFlags &flags, bool first);
        virtual void onInstancePass(const InstancePassFlags &flags);
        virtual void onFirstInstancePass(const InstancePassFlags &flags);
        virtual void onLastInstancePass();
        virtual void onInstanceSwitch(int sceneID);
        virtual void onInstanceAboutToSwitch(int sceneID);
        virtual void onMenuItemSelected(int itemHandle, int itemState);
        virtual void onBroadcast(int header, int messageID);
        virtual void onSceneSave();
        virtual void onModelSave();
        virtual void onModuleOpen(char *name);
        virtual void onModuleHandle(char *name);
        virtual void onModuleHandleInSensingPart(char *name);
        virtual void onModuleClose(char *name);
        virtual void onRenderingPass();
        virtual void onBeforeRendering();
        virtual void onImageFilterEnumReset();
        virtual void onImageFilterEnumerate(int &headerID, int &filterID, std::string &name);
        virtual void onImageFilterAdjustParams(int headerID, int filterID, int bufferSize, void *buffer, int &editedBufferSize, void *&editedBuffer);
        virtual std::vector<simFloat> onImageFilterProcess(int headerID, int filterID, int resX, int resY, int visionSensorHandle, simFloat *inputImage, simFloat *depthImage, simFloat *workImage, simFloat *bufferImage1, simFloat *bufferImage2, simFloat *outputImage, void *filterParamBuffer, int &triggerDetectionn);
        virtual void onAboutToUndo();
        virtual void onUndo();
        virtual void onAboutToRedo();
        virtual void onRedo();
        virtual void onScriptIconDblClick(int objectHandle, int &dontOpenEditor);
        virtual void onSimulationAboutToStart();
        virtual void onSimulationAboutToEnd();
        virtual void onSimulationEnded();
        virtual void onKeyPress(int key, int mods);
        virtual void onBannerClicked(int bannerID);
        virtual void onRefreshDialogs(int refreshDegree);
        virtual void onSceneLoaded();
        virtual void onModelLoaded();
        virtual void onGuiPass();
        virtual void onMainScriptAboutToBeCalled(int &out);
        virtual void onRMLPos();
        virtual void onRMLVel();
        virtual void onRMLStep();
        virtual void onRMLRemove();
        virtual void onPathPlanningPlugin();
        virtual void onColladaPlugin();
        virtual void onOpenGL(int programIndex, int renderingAttributes, int cameraHandle, int viewIndex);
        virtual void onOpenGLFrame(int sizeX, int sizeY, int &out);
        virtual void onOpenGLCameraView(int sizeX, int sizeY, int viewIndex, int &out);
        virtual void onProxSensorSelectDown(int objectID, simFloat *clickedPoint, simFloat *normalVector);
        virtual void onProxSensorSelectUp(int objectID, simFloat *clickedPoint, simFloat *normalVector);
        virtual void onPickSelectDown(int objectID);

    private:
        bool firstInstancePass = true;
        std::string name_;
    };
}

#define SIM_PLUGIN(pluginName, pluginVersion, className) \
LIBRARY simLib; \
className *simPlugin; \
SIM_DLLEXPORT unsigned char simStart(void *reservedPointer, int reservedInt) \
{ \
    try \
    { \
        simPlugin = new className; \
        simPlugin->setName(pluginName); \
        simLib = simPlugin->loadSimLibrary(); \
        simPlugin->onStart(); \
        return pluginVersion; \
    } \
    catch(std::exception &ex) \
    { \
        simAddLog(pluginName, sim_verbosity_errors, ex.what()); \
        return 0; \
    } \
} \
SIM_DLLEXPORT void simEnd() \
{ \
    try \
    { \
        if(simPlugin) \
        { \
            simPlugin->onEnd(); \
            delete simPlugin; \
            simPlugin = nullptr; \
        } \
    } \
    catch(std::exception &ex) \
    { \
        simAddLog(pluginName, sim_verbosity_errors, ex.what()); \
    } \
    unloadSimLibrary(simLib); \
} \
SIM_DLLEXPORT void * simMessage(int message, int *auxiliaryData, void *customData, int *replyData) \
{ \
    try \
    { \
        if(simPlugin) \
        { \
            return simPlugin->onMessage(message, auxiliaryData, customData, replyData); \
        } \
    } \
    catch(std::exception &ex) \
    { \
        simAddLog(pluginName, sim_verbosity_errors, ex.what()); \
    } \
    return 0L; \
}

#endif // SIMPLUSPLUS_PLUGIN_H_INCLUDED
