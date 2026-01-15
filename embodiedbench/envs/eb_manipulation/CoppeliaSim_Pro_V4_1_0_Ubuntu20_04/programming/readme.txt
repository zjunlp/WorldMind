The various source code items can be found on https://github.com/CoppeliaRobotics
Clone each required repository with:

git clone --recursive https://github.com/CoppeliaRobotics/repositoryName

Use following directory structure:

coppeliaRobotics
    |__ coppeliaSimLib (CoppeliaSim main library)
    |__ programming
                  |__ include
                  |__ common
                  |__ simMath
                  |__ coppeliaGeometricRoutines
                  |__ coppeliaKinematicsRoutines
                  |__ simExtGeometric
                  |__ simExtIK
                  |__ simExtDynamics
                  |__ libPlugin
                  |__ simExtCodeEditor
                  |__ remoteApi
                  |__ remoteApiBindings
                  |__ b0RemoteApiBindings
                  |__ simExtRemoteApi
                  |__ simExtJoystick
                  |__ simExtCam
                  |__ simExtURDF
                  |__ simExtSDF
                  |__ simExtCollada
                  |__ simExtRmlType2
                  |__ simExtRRS1
                  |__ simExtMtb
                  |__ simExtCustomUI
                  |__ simExtOMPL
                  |__ simExtICP
                  |__ simExtSurfaceReconstruction
                  |__ simExtLuaCommander
                  |__ simExtPluginSkeleton
                  |__ simExtPluginSkeletonNG
                  |__ simExtCHAI3D
                  |__ simExtConvexDecompose
                  |__ simExtPovRay
                  |__ simExtQhull
                  |__ simExtVision
                  |__ simExtExternalRenderer
                  |__ simExtLuaRemoteApiClient
                  |__ simExtBlueZero
                  |__ simExtImage
                  |__ simExtBubbleRob
                  |__ simExtK3
                  |__ simExtAssimp
                  |__ simExtOpenMesh
                  |__ simExtOpenGL3Renderer
                  |__ simExtGLTF
                  |__ simExtURLDrop
                  |__ standAloneKinematicsDemo1
                  |__ standAloneKinematicsDemo2
                  |__ standAloneKinematicsDemo3
                  |__ bubbleRobClient
                  |__ bubbleRobServer
                  |__ b0_bubbleRob
                  |__ rcsServer
                  |__ mtbServer
                  |__ simLuaLibrary
                  |__ bluezero

ros_packages
           |__ simExtROSInterface
           |__ ros_bubble_rob

ros2_packages
           |__ simExtROS2Interface
           |__ ros2_bubble_rob

           
Following are the main Items:
-----------------------------

-   'coppeliaSimLib' (requires 'include', 'common' and 'simMath'):         
    https://github.com/CoppeliaRobotics/coppeliaSimLib

-   'coppeliaSimClientApplication' (requires 'include' and 'common'):
    https://github.com/CoppeliaRobotics/coppeliaSimClientApplication


Various common items:
---------------------

-   'simMath':
    https://github.com/CoppeliaRobotics/simMath

-   'common' (requires 'include'):
    https://github.com/CoppeliaRobotics/common

-   'include' (requires 'common'):
    https://github.com/CoppeliaRobotics/include

-   'remoteApi' (requires 'include' and 'common'):
    https://github.com/CoppeliaRobotics/remoteApi

-   'libPlugin':
    https://github.com/CoppeliaRobotics/libPlugin

-   'remoteApiBindings' (requires 'remoteApi' if libs need to be rebuilt)
    https://github.com/CoppeliaRobotics/remoteApiBindings

-   'b0RemoteApiBindings' (requires 'bluezero' if libs need to be rebuilt)
    https://github.com/CoppeliaRobotics/b0RemoteApiBindings

-   'coppeliaGeometricRoutines' (requires 'include', 'common' and 'simMath'):
    https://github.com/CoppeliaRobotics/coppeliaGeometricRoutines

-   'coppeliaKinematicsRoutines' (requires 'include', 'common' and 'simMath'):
    https://github.com/CoppeliaRobotics/coppeliaKinematicsRoutines
    
Major plugins:
--------------

-   'simExtDynamics' (requires 'include', 'common' and 'simMath'):
    https://github.com/CoppeliaRobotics/simExtDynamics

-   'simExtGeometric' (requires 'include', 'common', 'simMath' and coppeliaGeometricRoutines):
    https://github.com/CoppeliaRobotics/simExtGeometric

-   'simExtIK' (requires 'include', 'common', 'simMath' and coppeliaKinematicsRoutines):
    https://github.com/CoppeliaRobotics/simExtIK

-   'simExtCodeEditor' (requires 'include', 'common' and 'QScintilla'):
    https://github.com/CoppeliaRobotics/simExtCodeEditor


Various plugins:
----------------

-   'simExtJoystick' (requires 'include' and 'common'):
    https://github.com/CoppeliaRobotics/simExtJoystick (Windows only)

-   'simExtCam' (requires 'include' and 'common'):
    https://github.com/CoppeliaRobotics/simExtCam (Windows only)

-   'simExtUrdf' (requires 'include', 'common' and 'simMath'):
    https://github.com/CoppeliaRobotics/simExtUrdf

-   'simExtCollada' (requires 'include', 'common' and 'simMath'):
    https://github.com/CoppeliaRobotics/simExtCollada

-   'simExtSDF' (requires 'include' and 'common', 'simMath' and 'libPlugin'):
    https://github.com/CoppeliaRobotics/simExtSDF

-   'simExtRmlType2' (requires 'include' and 'common'):
    https://github.com/CoppeliaRobotics/simExtRmlType2

-   'simExtRRS1' (requires 'include' and 'common'):
    https://github.com/CoppeliaRobotics/simExtRRS1

-   'simExtMtb' (requires 'include' and 'common'):
    https://github.com/CoppeliaRobotics/simExtMtb

-   'simExtCustomUI' (requires 'include', 'common' and 'libPlugin'):
    https://github.com/CoppeliaRobotics/simExtCustomUI

-   'simExtOMPL' (requires 'include', 'common' and 'libPlugin'):
    https://github.com/CoppeliaRobotics/simExtOMPL

-   'simExtICP' (requires 'include', 'common' and 'libPlugin'):
    https://github.com/CoppeliaRobotics/simExtICP

-   'simExtSurfaceReconstruction' (requires 'include', 'common' and 'libPlugin'):
    https://github.com/CoppeliaRobotics/simExtSurfaceReconstruction

-   'simExtROSInterface' (requires 'include', 'common' and 'libPlugin'):
    https://github.com/CoppeliaRobotics/simExtROSInterface

-   'simExtROS2Interface' (requires 'include', 'common' and 'libPlugin'):
    https://github.com/CoppeliaRobotics/simExtROS2Interface

-   'simExtLuaCommander' (requires 'include', 'common' and 'libPlugin'):
    https://github.com/CoppeliaRobotics/simExtLuaCommander

-   'simExtCHAI3D' (requires 'include' and 'common'):
    https://github.com/CoppeliaRobotics/simExtCHAI3D

-   'simExtConvexDecompose' (requires 'include' and 'common'):
    https://github.com/CoppeliaRobotics/simExtConvexDecompose

-   'simExtPovRay' (requires 'include' and 'common'):
    https://github.com/CoppeliaRobotics/simExtPovRay

-   'simExtQhull' (requires 'include' and 'common'):
    https://github.com/CoppeliaRobotics/simExtQhull

-   'simExtOpenMesh' (requires 'include' and 'common'):
    https://github.com/CoppeliaRobotics/simExtOpenMesh

-   'simExtRemoteApi' (requires 'include' and 'common'):
    https://github.com/CoppeliaRobotics/simExtRemoteApi

-   'simExtVision' (requires 'include' and 'common'):
    https://github.com/CoppeliaRobotics/simExtVision

-   'simExtExternalRenderer' (requires 'include' and 'common'):
    https://github.com/CoppeliaRobotics/simExtExternalRenderer

-   'simExtLuaRemoteApiClient' (requires 'include' and 'common'):
    https://github.com/CoppeliaRobotics/simExtLuaRemoteApiClient

-   'simExtBlueZero' (requires 'include', 'common' and 'libPlugin'):
    https://github.com/CoppeliaRobotics/simExtBlueZero

-   'simExtImage' (requires 'include', 'common' and 'libPlugin'):
    https://github.com/CoppeliaRobotics/simExtImage

-   'simExtBubbleRob' (requires 'include' and 'common'):
    https://github.com/CoppeliaRobotics/simExtBubbleRob

-   'simExtK3' (requires 'include' and 'common'):
    https://github.com/CoppeliaRobotics/simExtK3

-   'simExtOpenGL3Renderer' (requires 'include' and 'common'):
    https://github.com/stepjam/simExtOpenGL3Renderer or https://github.com/CoppeliaRobotics/simExtOpenGL3Renderer

-   'simExtGLTF' (requires 'include' and 'common'):
    https://github.com/CoppeliaRobotics/simExtGLTF

-   'simExtURLDrop' (requires 'include' and 'common'):
    https://github.com/CoppeliaRobotics/simExtURLDrop

Various other repositories:		
---------------------------

-   'standAloneKinematicsDemo1' (requires 'include', 'common', 'coppeliaKinematicsRoutines' and 'remoteApi'):
    https://github.com/CoppeliaRobotics/standAloneKinematicsDemo1

-   'standAloneKinematicsDemo2' (requires 'include', 'common', 'coppeliaKinematicsRoutines' and 'remoteApi'):
    https://github.com/CoppeliaRobotics/standAloneKinematicsDemo2

-   'standAloneKinematicsDemo3' (requires 'include', 'common', 'coppeliaKinematicsRoutines' and 'remoteApi'):
    https://github.com/CoppeliaRobotics/standAloneKinematicsDemo3

-   'bubbleRobClient' (requires 'include', 'common' and 'remoteApi'):
    https://github.com/CoppeliaRobotics/bubbleRobClient

-   'bubbleRobServer' (requires 'include' and 'common'):
    https://github.com/CoppeliaRobotics/bubbleRobServer
    
-   'b0_bubbleRob' (requires 'include' and 'common'):
    https://github.com/CoppeliaRobotics/b0_bubbleRob

-   'rcsServer' (requires 'include' and 'common'):
    https://github.com/CoppeliaRobotics/rcsServer

-   'mtbServer' (requires 'include' and 'common'):
    https://github.com/CoppeliaRobotics/mtbServer

-   'ros_bubble_rob'
    https://github.com/CoppeliaRobotics/ros_bubble_rob

-   'ros2_bubble_rob'
    https://github.com/CoppeliaRobotics/ros2_bubble_rob

-   'simLuaLibrary':
    https://github.com/CoppeliaRobotics/simLuaLibrary

-   'PyRep':
    https://github.com/stepjam/PyRep
