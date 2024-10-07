#ifndef DIFFUSE_CUH
#define DIFFUSE_CUH

#include "common.hpp"
#include "math_utils.cuh"

struct diffuse_params {
    color3 albedo;
    float roughness;
};

/// Material-specific scatter function for a diffuse (Oren-Nayar) material.
__device__ inline color3 scatter_diffuse(
    ray3* r,
    const hit* h,
    const diffuse_params& params,
    unsigned int* seed)
{
    /*
     * https://www.cs.cmu.edu/afs/cs/academic/class/16823-s16/www/pdfs/appearance-modeling-5.pdf
     *
     * E0 = 1
     * A = 1.0 - 0.5 * (roughness_sq / (roughness_sq + 0.09))
     * B = 0.45 * (roughness_sq / (roughness_sq + 0.09))
     * alpha = max(theta_i, theta_o)
     * beta = min(theta_i, theta_o)
     *
     * L() = (albedo / pi) * E0 * cos_theta_i(A + B * max(0, cos_phi_diff) * sin(alpha) * tan(beta))
     *
     * PDF = cos_theta_i / pi
     *
     * L() / PDF = albedo * (A + B * max(0, cos_phi_diff) * sin(alpha) * tan(beta))
     */
    vec3 normal = h->normal;
    vec3 incident = r->direction;
    vec3 scattered = random_cosine_weighted_in_hemisphere(normal, seed);

    r->origin = h->loc;
    r->direction = scattered;

    float cos_theta_i = cos_theta_inner(-incident, normal);
    float cos_theta_o = cos_theta_inner(scattered, normal);

    if (!h->is_front_face) return color3{0.f, 0.f, 0.f};

    float alpha = acosf(cos_theta_i);
    float beta = acosf(cos_theta_o);
    if (alpha < beta) {
        float tmp = alpha;
        alpha = beta;
        beta = tmp;
    }

    float roughess_sq = params.roughness * params.roughness;
    float cos_dphi = clamp(cos_phi_diff_from_system(incident, normal, scattered), 0.f, 1.f);

    float A = 1.f - (roughess_sq / (2.f * (roughess_sq + 0.33f)));
    float B = 0.45f * roughess_sq / (roughess_sq + 0.09f);
    float brdf = (A + B * cos_dphi * sinf(alpha) * tanf(beta));

    // cos(theta_i) and 1/pi term cancel with the PDF of random_cosine_weighted_in_hemisphere()
    return params.albedo * brdf;
}
#endif //DIFFUSE_CUH
