## Fluids:<br> A fast, GPU-based, smooth particle hydrodynamics (SPH) fluid simulator.
Rama Hoetzlein (c) 2007-2025<br>

👉 Project Website: <a href="https://ramakarl.com/fluids3">https://ramakarl.com/fluids3</a><br>

Fluids v.3 was the first very fast, interactive >1 million particle, Smooth Particle Hydrodynamic simulator.<br>
This repository contains the complete history of Fluids.<br>

🚀🚀 **NEW (2025)**: The latest Fluids 5.0 was updated in 2023, and first published here in Nov 2025.<br>
The current version is even faster, uses the CUDA Driver API, can handle >4 million particles, uses full smart CPU/GPU pointers, buffer-based structure of arrays, integrated CPU/GPU switching, with cleaner, simplified code.<br>

## Latest Performance
Date: 11/23/2025<br>
Hardware: **GeForce RTX 4090** <br>
\# Particles: **4,000,000** <br>
- Fluids ver 3.2 ➔ 32 fps
- Fluids ver 3.9 ➔ 71 fps
- Fluids ver 4.0 ➔ 98 fps
- Fluids ver 5.0 ➔ 160 fps <br>
(same hardware & num particles for all versions)<br>

## History

**Fluids 5.0**
- Modernized, 2023
- Cuda Driver API (cuLaunchKernel), no longer requires CUDA Run-time linkage
- Large-scale prefixSums, custom code, >4 million particles
- Full Smart Pointers, CPU & GPU (DataPtr)
- Buffer-based kernels
- Abstract Buffer management (DataX)
- Removed run modes
- Integrated CPU & GPU switching per func (instead of ver 4.0 Run modes)
- Modern folder struct (src, assets)
- Major code cleanup and reduction
<img src="https://github.com/ramakarl/fluids3/blob/master/gallery/fluids_5.0.png" width="900" />

---

**Fluids 4.0**
- Rewrite, 2022
- Fluid struct removed
- Fluid as buffers (SofA, see fluid.h)
- Buffer-based kernels
- Allows for arbitrary per-particle properties
- New kernels: sampleParticles, computeQuery
- Write to disk (SavePoints)
- Equivalent CPU & GPU code
- Run modes: SEARCH, CPU_SLOW, CPU_GRID, VALIDATE, GPU_FULL, PLAYBACK
<img src="https://github.com/ramakarl/fluids3/blob/master/gallery/fluids_4.0.png" width="900" />

---

**Fluids 3.9**
- Major update, 2021
- Switched to Cuda Driver API (cuLaunchKernel), no need to link to Cuda Run-time
- Large-scale prefix sums, custom code, >4 million particles
- Allocation helper, cpu & gpu (AllocateBuffer)
- Source at top level
<img src="https://github.com/ramakarl/fluids3/blob/master/gallery/fluids_3.9.png" width="900" />

---

**Fluids 3.2**
- Minor update, 2013
- Application code (abstracted main)
- GUI added
- Still using Cuda Run-time API (cudaLaunchKernel)
- Identical kernels to 3.0

---

**Fluids 3.0**
- Major update, 2012
- New Insertion-sort based
- Entirely on GPU
- Explicit memory allocs
- Cuda Run-time API (cudaLaunchKernel)
- Static/mixed host code (fluid_system_host.cu)
- 3rd party Prefix scan
- Fluid as struct (AofS)
- Kernels: insertParticles, countingSort, computePressure, computeForce, advanceParticles
- Introduced insertion-sort and prefix scan, as faster alternatives to radix-sort for particle search
- GTC 2014 Talk: 2014, Hoetzlein. <a href="https://ramakarl.com/fluids3">Fast Fixed-Radius Nearest Neighbors: Interactive Million-Particle Fluids</a>. GPU Technology Conference. Santa Clara, CA. 2014.
<img src="https://github.com/ramakarl/fluids3/blob/master/gallery/fluids_3.2.png" width="900" />

---

**Fluids 2**
- First GPU version, 2011
- Radix-sort based (Simon Green, 2010)
- CPU-GPU transfers per frame (pipeline bottleneck)
<img src="https://github.com/ramakarl/fluids3/blob/master/gallery/fluids_2.0.png" width="700" />

---

**Fluids 1**
- First CPU version, 2007
- Smoothed Particle Hydrodynamics (SPH)
- Proof of concept
- Self learning
<img src="https://github.com/ramakarl/fluids3/blob/master/gallery/fluids_1.0.png" width="400" />

---

## How to Build
Fluids 1, 2, and 3.0 did not use CMake. VS solutions are provided.<br>
Fluids >3.2 use Cmake.<br>
Each Fluids version has its own CMakeLists.txt.<br>
Run Cmake on the version you want. 

## Keyboard commands (Fluids 3.0):
```
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
```

## Scene parameters (Fluids 3.0)
* Example scenes are loaded from the scene.xml file<br>
All parameters are permitted in either Fluid or Scene sections.
```
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
```

## Contact
Feel free to contact me if you have any questions, comments or suggestions:<br>
**Rama Hoetzlein** <br>
Website: <a href="https://ramakarl.com">ramakarl.com</a><br>
Email: ramahoetzlein@gmail.com<br>

## License & Copyright
MIT License<br>
FLUIDS - SPH Fluid Simulator for CPU and GPU<br>
Copyright (c) 2007-2023. Rama Hoetzlein<br>


  
