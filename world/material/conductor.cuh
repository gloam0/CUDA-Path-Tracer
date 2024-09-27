#ifndef CONDUCTOR_CUH
#define CONDUCTOR_CUH

#include "common.hpp"
#include "math_utils.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////
struct conductor_params {
    color3 eta;         /* re part of complex refractive index for each sRGB channel */
    color3 k;           /* im part of complex refractive index for each sRGB channel */
    float roughness;    /* surface roughness [0.0, 1.0] */
};

////////////////////////////////////////////////////////////////////////////////////////////////
/// Approximate Fresnel reflectance of a conductor material characterized by its complex
/// refractive index (eta + i*kappa).
/// @param cos_theta Cosine of incident angle.
/// @param eta Real part of complex refractive index for each sRGB channel.
/// @param kappa Complex part of complex refractive index for each sRGB channel.
__device__ inline float3 fresnel_conductor(float cos_theta, const float3 &eta, const float3 &kappa) {
    /*
     * Using the approach described in "Artist Friendly Metallic Fresnel", Gulbrandsen, 2014,
     * adapted for float3 eta and kappa values representing the complex refractive index
     * of the sRGB primaries (see predefined_materials.cuh).
     *
     * 2.3 Unpolarized Approximate Fresnel Equations
     *
     * Perpendicular polarized reflection
     *     f_perpendicular = (n^2 + k^2 - 2*n*cos(theta) + cos^2(theta))
     *                     / (n^2 + k^2 + 2*n*cos(theta) + cos^2(theta))
     *
     * Parallel polarized reflection
     *     f_parallel      = ((n^2 + k^2)*cos^2(theta) - 2*n*cos(theta) + 1)
     *                     / ((n^2 + k^2)*cos^2(theta) + 2*n*cos(theta) + 1)
     *
     *  f(cos_theta, eta, k) = (1/2) * (f_perpendicular + f_parallel)
     */
    cos_theta = clamp(cos_theta, 0.0f, 1.0f);

    float  cos2_theta          = cos_theta * cos_theta;
    color3 two_eta_cos_theta   = 2 * eta * cos_theta;
    color3 eta2_p_kappa2       = elem_square(eta) + elem_square(kappa);

    color3 f_perpendicular_common = eta2_p_kappa2 + cos2_theta;
    color3 f_perpendicular = elem_divide(f_perpendicular_common - two_eta_cos_theta,
                                         f_perpendicular_common + two_eta_cos_theta);

    color3 f_parallel_common = eta2_p_kappa2 * cos2_theta + 1.f;
    color3 f_parallel = elem_divide(f_parallel_common - two_eta_cos_theta,
                                    f_parallel_common + two_eta_cos_theta);

    color3 reflectance = 0.5f * (f_perpendicular + f_parallel);
    return elem_clamp(reflectance, 0.0f, 1.0f);
}

/// Material-specific scatter function for a conductor material.
__device__ inline color3 scatter_conductor(ray3* r, const hit* h, const conductor_params* params, unsigned int* seed) {
    vec3 incident = r->direction;
    vec3 normal = h->normal;

    /* Apply roughness */
    if (params->roughness > 0.0f) {
        vec3 random_vector = random_unit_vector(seed);
        normal = normalize(normal + params->roughness * random_vector);
    }

    /* Reflect incident ray about the normal */
    vec3 reflected = reflect(incident, normal);
    r->origin = h->loc;
    r->direction = reflected;

    float cos_theta =  cos_theta_from_incident_and_normal(incident, normal);

    /* Return fresnel reflectance as attenuation */
    return fresnel_conductor(cos_theta, params->eta, params->k);
}
////////////////////////////////////////////////////////////////////////////////////////////////

#endif //CONDUCTOR_CUH
