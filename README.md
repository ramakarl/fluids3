Fluids v.3.1
------------
Rama Hoetzlein (c) 2012-2014, https://ramakarl.com/fluids3

Source code based on the technique from:<br>
2014, Hoetzlein. <a href="https://ramakarl.com/fluids3">Fast Fixed-Radius Nearest Neighbors: Interactive Million-Particle Fluids</a>. GPU Technology Conference. Santa Clara, CA. 2014.

UPDATE: March 7, 2014
Fluids v.3.1 is now available on github and on project website.
This version is essentially identical, but is much easier to build. It has the following new features:
- Easier to build. No more dependency on freetype or glut.
- Interactive on-screen GUIs
- Upgraded to run on CUDA 5.0 and 5.5
- Profile timings. Now includes GPU markers for profiling on the GPU using NVIDIA’s free NSight tool.
- Bug fixes noticed by users


Requirements:
-----------

Fluids v.3.1 requires a CUDA-capable NVIDIA graphics card /w Compute capability 2.1 or higher. 
Builds tested on Windows 7 with Visual Studio 2010

Start Fluids by running fluids_v3.exe
See below for License terms.
See website for software details.

Fluids 3.1 makes use of the following libraries:
 CUDA - http://www.nvidia.com/object/cuda_home_new.html
 OpenGL - http://www.opengl.org/
 TinyXML - http://www.grinninglizard.com/tinyxml/index.html

Keyboard commands:
------------------
H		Turn scene info on/off
N, M		Change number of particles
[, ]		Change example scene (loaded from scene.xml)
F, G		Change algorithm (CPU, GPU, etc.)
J		Change rendering mode (points, sprites, spheres)
C		Adjust camera (using mouse)
L		Adjust light (using mouse)
A,S,W,D	Move camera target
1		Draw acceleration grid
2		Draw particle IDs (be sure # < 4096 first)
~		Start video capture to disk (tilde key)
-, +		Change grid density

Scene parameters:
-----------------
* Example scenes are loaded from the scene.xml file
All parameters are permitted in either Fluid or Scene sections.
DT			Simulation time step
SimScale		Simulation scale (see website)
Viscosity		Fluid viscosity coefficient
RestDensity	Fluid rest density
Mass			Fluid particle mass
Radius		Fluid particle radius, only for boundary tests
IntStiff		Fluid internal stiffness (non-boundary)
BoundStiff		Fluid stiffness at boundary
BoundDamp		Fluid damping at boundary
AccelLimit		Acceleration limit (for stability)
VelLimit		Velocity limit (for stability)
PointGravAmt	Strength of point gravity
PointGravPos	Position of point gravity
PlaneGravDir	Direction of plane gravity
GroundSlope	Slope of the ground (Y- plane)
WaveForceFre	Frequency of wave forcing
WaveForceMin	Amplitude of wave forcing from X- plane
WaveForceMax	Amplitude of wave forcing from X+ plane
Name			Name of scene example
Num			Number of particles to simulate
VolMin		Start corner of Domain Volume
VolMax		End corner of Domain Volume
InitMin		Start corner of Initial Particle placement
InitMax		End corner of Initial Particle placement





-----------
FLUIDS v.3 - SPH Fluid Simulator for CPU and GPU
Copyright (C) 2012-2013. Rama Hoetzlein

Z-Lib License
  
