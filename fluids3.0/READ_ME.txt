
Fluids v.3.1, R. Hoetzlein (c) 2012-213, http://fluids3.com

-----------
UPDATE: March 7, 2014
Fluids v.3.1 is now available on github and on project website.
This version is essentially identical, but is much easier to build. It has the following new features:
- Easier to build. No more dependency on freetype or glut.
- Interactive on-screen GUIs
- Upgraded to run on CUDA 5.0 and 5.5
- Profile timings. Now includes GPU markers for profiling on the GPU using NVIDIA’s free NSight tool.
- Bug fixes noticed by users
-----------

Requirements:
Fluids v.3 requires a CUDA-capable NVIDIA graphics card /w Compute capability 2.1 or higher. 
Builds tested on Windows 7 with Visual Studio 2010

Start Fluids by running fluids_v3.exe
See below for License terms.
See website for software details.

Fluids makes use of the following libraries:
 CUDA - http://www.nvidia.com/object/cuda_home_new.html
 OpenGL - http://www.opengl.org/
 FreeGlut - http://freeglut.sourceforge.net/
 TinyXML - http://www.grinninglizard.com/tinyxml/index.html
 Glee - http://elf-stone.com/glee.php

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


FLUIDS v.3 - SPH Fluid Simulator for CPU and GPU
Copyright (C) 2012-2013. Rama Hoetzlein, http://fluids3.com
=======================================================

  Attribute-ZLib license (* See additional part 4)


  This software is provided 'as-is', without any express or implied
  warranty. In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
  4. Any published work based on this code must include public acknowledgement
     of the origin. This includes following when applicable:
	   - Journal/Paper publications. Credited by reference to work in text & citation.
	   - Public presentations. Credited in at least one slide.
	   - Distributed Games/Apps. Credited as single line in game or app credit page.	 
	 Retaining this additional license term is required in derivative works.
	 Acknowledgement may be provided as:
	   Publication version:  
	      2012-2013, Hoetzlein, Rama C. Fluids v.3 - A Large-Scale, Open Source
	 	  Fluid Simulator. Published online at: http://fluids3.com
	   Single line (slides or app credits):
	      GPU Fluids: Rama C. Hoetzlein (Fluids v3 2013)

 Notes on Clause 4:
  The intent of this clause is public attribution for this contribution, not code use restriction. 
  Both commerical and open source projects may redistribute and reuse without code release.
  However, clause #1 of ZLib indicates that "you must not claim that you wrote the original software". 
  Clause #4 makes this more specific by requiring public acknowledgement to be extended to 
  derivative licenses. 