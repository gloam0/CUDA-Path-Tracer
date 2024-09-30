# CUDA Path Tracer

This repo contains my progress thus far in implementing a path tracer in CUDA, designed 
to simulate the physical behavior of light to produce realistic renders. I'm making an
effort to emphasize **physically-based rendering**, ensuring that light-material interactions 
adhere closely to real-world physics.


## Contact

If you have suggestions, critiques, or just want to chat about computer graphics, 
you can find my email in the bio of my GitHub profile!


## Development
### Milestones
- [X] __Graphics Pipeline and Base Infrastructure__
  - CUDA-OpenGL interop, double-buffering
  - GLFW window management
  - Logging and timing utilities
- [x] __Core Rendering Pipeline__
  - GLFW window and OpenGL context initialization
  - PBO and texture management
  - CUDA kernel integration
  - Rendering loop (render frame, poll input, update camera, repeat...)
- [X] __Ray Tracing Implementation__
  - Camera-viewport ray generation and tracing
  - Geometry-specific intersection testing
  - Material-specific ray scattering
  - HDR environment mapping / indirect environment lighting
- [X] __Materials and Shading__
  - Diffuse (Lambertian)
  - Conductor (Fresnel equations)
  - Dielectric (Fresnel equations, Schlick's approximation)
  - Predefined materials
- [X] __User Input__
  - Keyboard and mouse input handling implemented as GLFW callbacks
- [X] __Camera Mechanics__
  - Free-movement camera system with quaternion-based orientation and natural movement acceleration/deceleration.
- [X] __Frame Saving__
  - Save rendered frames as JPEG images using libjpeg directly.
- [X] __Free Mode and Render Mode__
  - Dual-mode rendering system with a 'free' mode for real-time camera operations and a 'render' mode for fixed, long-term rendering of a single frame.

### TODO
    
- [ ] Advanced Geometries
  - Additional primitives
  - Mesh importing
- [ ] Acceleration Structures
  - BVH
  - Spatial partitioning?
- [ ] Enhanced Lighting Models
  - Global illumination (path tracing, photon mapping?)
  - Area lights, soft shadows
- [ ] Additional Material Types
  - Emmissive materials
  - Anisotropic materials
  - Subsurface scattering?
- [ ] Texture Mapping, Bump Mapping, Normal Mapping?
- [ ] Optimize Sampling
  - Switch to quasi-random low-discrepancy sequences (Sobol?) over uniform random sampling
  - Adaptive sampling of areas with higher variance (especially as a means of firefly reduction for scenes with high-variance lighting)
- [ ] Denoising
- [ ] Improved Management and Configuration of Data in GPU Memory
- [ ] Volumetric Rendering and Atmospheric Phenomena?
- [ ] Real-Time Configurability
  - UI for updating global configuration items during runtime
- [ ] Post-Processing Effects?


## Thanks to...

### Resources
- [_Ray Tracing in One Weekend_](https://raytracing.github.io/books/RayTracingInOneWeekend.html)
- [_Physically Based Rendering: From Theory to Implementation_](https://www.pbr-book.org/4ed/contents)
- [_"Artist Friendly Metallic Fresnel", Ole Gulbrandsen (2014)_](https://jcgt.org/published/0003/04/03/)
- [_"Understanding Quaternions", Jeremiah (2012)_](https://www.3dgep.com/understanding-quaternions/)
- [_Inigo Quilez's Compter Graphics Articles_](https://iquilezles.org/articles/)
- [_"Color Part 2: Color Spaces and Color Perception", Roger N. Clark_](https://clarkvision.com/imagedetail/color-spaces/)
- [_RefractiveIndex.INFO_](https://refractiveindex.info/)
- _...and many more_

### HDR Images
HDR environment images ("Day Sky HDRI 008 B", "Night Environment HDRI 008", 
"Night Environment HDRI 002", "Day Sky HDRI 027 B", "Indoor Environment HDRI 005") sourced 
from [ambientCG.com](https://ambientcg.com/), licensed under the Creative Commons CC0 
1.0 Universal License. 
