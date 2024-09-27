#ifndef DIELECTRIC_CUH
#define DIELECTRIC_CUH

#include "common.hpp"
#include "math_utils.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////
enum class dielectric_render_method {
    FRESNEL,
    SHLICK
};

struct dielectric_params {
    float ior;         /* index of refraction */
    float roughness;   /* surface roughness [0.0, 1.0] */
    dielectric_render_method render_method;
};

////////////////////////////////////////////////////////////////////////////////////////////////
/// Calculate Fresnel reflectance for a dielectric material.
/// @param cos_theta_i Cosine of incident angle.
/// @param eta_i Refractive index of incident medium.
/// @param eta_t Refractive index of the transmitted medium.
__device__ inline float fresnel_dielectric(float cos_theta_i, float eta_i, float eta_t) {
    /*
     * https://www.pbr-book.org/3ed-2018/Reflection_Models/Specular_Reflection_and_Transmission
     *
     * f_parallel      = (eta_t * cos(theta_i) - eta_i * cos(theta_t))
     *                 / (eta_t * cos(theta_i) + eta_i * cos(theta_t))
     *
     * f_perpendicular = (eta_i * cos(theta_i) - eta_t * cos(theta_t))
     *                 / (eta_i * cos(theta_i) + eta_t * cos(theta_t))
     *
     * f(cos_theta_i, eta_i, eta_t) = (1/2) * (f_parallel^2 + f_perpendicular^2)
     *
     * with cos_theta_t derived from cos_theta_i via Snell's law
     */
    cos_theta_i = clamp(cos_theta_i, 0.0f, 1.0f);

    float sin_theta_t = (eta_i / eta_t) * sqrtf(fmaxf(0.f, 1.f - cos_theta_i * cos_theta_i));
    if (sin_theta_t >= 1.f) return 1.f; /* Total internal reflection */
    float cos_theta_t = sqrtf(fmaxf(0.f, 1.f - sin_theta_t * sin_theta_t));

    float eta_t_cos_theta_i = eta_t * cos_theta_i;
    float eta_i_cos_theta_t = eta_i * cos_theta_t;
    float f_parallel = (eta_t_cos_theta_i - eta_i_cos_theta_t)
                     / (eta_t_cos_theta_i + eta_i_cos_theta_t);

    float eta_i_cos_theta_i = eta_i * cos_theta_i;
    float eta_t_cos_theta_t = eta_t * cos_theta_t;
    float f_perpendicular = (eta_i_cos_theta_i - eta_t_cos_theta_t)
                          / (eta_i_cos_theta_i + eta_t_cos_theta_t);

    float reflectance = 0.5f * (f_parallel * f_parallel + f_perpendicular * f_perpendicular);
    return reflectance;
}

/// Material-specific scatter function for a conductor material.
/// Assumes encompassing medium has an IOR of 1.0. The IOR of a material encompassed by
/// a medium other than air should be expressed as a ratio of material_IOR / encompassing_IOR.
__device__ inline color3 scatter_dielectric_fresnel(ray3* r, const hit* h, const dielectric_params* params, unsigned int* seed) {
    vec3 incident = r->direction;
    vec3 normal = h->normal;

    /* Apply roughness */
    if (params->roughness > 0.0f) {
        vec3 random_vector = random_unit_vector(seed);
        normal = normalize(normal + params->roughness * random_vector);
    }

    /* Swap eta_i and eta_t if the hit came from the back face */
    float eta_i = h->is_front_face ? 1.0f : params->ior;
    float eta_t = h->is_front_face ? params->ior : 1.0f;
    float refraction_ratio = eta_i / eta_t;

    float cos_theta_i = cos_theta_from_incident_and_normal(incident, normal);

    float fresnel_reflectance = fresnel_dielectric(cos_theta_i, eta_i, eta_t);

    /* Probabilistic reflection / refraction */
    vec3 direction;
    if (fresnel_reflectance > xorshift32_f_norm(seed)) {
        direction = reflect(incident, normal);
    } else {
        direction = refract(incident, normal, refraction_ratio);
    }

    r->origin = h->loc;
    r->direction = normalize(direction);

    /*
     * Assumes perfect transmission / no absorption, non-dispersion of component
     * colors / wavelength independence.
     * TODO: Consider this, and extensions thereof such as fluorescence/phosphorescence,
     *       subsurface scattering / subsurface interactions, etc.
     */
    return color3{1.f,1.f,1.f};
}

////////////////////////////////////////////////////////////////////////////////////////////////
/// Schlick's approximation for reflectance
/// @param cos_theta Cosine of incident angle.
/// @param ior Index of refraction.
__device__ inline float shlick_reflectance(float cos_theta, float ior) {
    /*
     * https://en.wikipedia.org/wiki/Schlick%27s_approximation
     *
     * r(theta) = r0 + (1 - r0) * (1 - cos(theta))^5
     *
     * with r0 = ((eta_i - eta_t) / (eta_i + eta_t))^2
     */
    float r0_sqrt = (1.0f - ior) / (1.0f + ior);
    float r0 = r0_sqrt * r0_sqrt;
    return r0 + (1.0f - r0) * powf(1.0f - cos_theta, 5.0f);
}

/// Material-specific scatter function for a conductor material.
/// Assumes encompassing medium has an IOR of 1.0. The IOR of a material encompassed by
/// a medium other than air should be expressed as a ratio of material_IOR / encompassing_IOR.
__device__ inline color3 scatter_dielectric_shlick(ray3* r, const hit* h, const dielectric_params* params, unsigned int* seed) {
    vec3 incident = r->direction;
    vec3 normal = h->normal;

    if (params->roughness > 0.0f) {
        vec3 random_vector = random_unit_vector(seed);
        normal = normalize(normal + params->roughness * random_vector);
    }

    /* Swap eta_i and eta_t if the hit came from the back face */
    float refraction_ratio = h->is_front_face ? (1.0f / params->ior) : params->ior;
    float cos_theta = cos_theta_from_incident_and_normal(incident, normal);
    float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));

    bool cannot_refract = refraction_ratio * sin_theta > 1.f;
    float reflect_prob = shlick_reflectance(cos_theta, refraction_ratio);

    /* Probabilistic reflection / refraction */
    vec3 direction;
    if (cannot_refract || reflect_prob > xorshift32_f_norm(seed)) {
        direction = reflect(incident, normal);
    } else {
        direction = refract(incident, normal, refraction_ratio);
    }

    r->origin = h->loc;
    r->direction = normalize(direction);

    /*
     * Assumes perfect transmission / no absorption, non-dispersion of component
     * colors / wavelength independence.
     * TODO: Consider this, and extensions thereof such as fluorescence/phosphorescence,
     *       subsurface scattering / subsurface interactions, etc.
     */
    return color3{1.f,1.f,1.f};
}
////////////////////////////////////////////////////////////////////////////////////////////////
///
#endif //DIELECTRIC_CUH
