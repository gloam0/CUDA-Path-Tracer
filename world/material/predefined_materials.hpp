#ifndef PREDEFINED_MATERIALS_H
#define PREDEFINED_MATERIALS_H

#include "material.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////
/* ---------------------------------------------------------------------
 * sRGB primaries (https://clarkvision.com/imagedetail/color-spaces/)
 * - red:   612nm
 * - green: 549nm
 * - blue:  465nm
 *
 * ---------------------------------------------------------------------
 * Sampled at sRGB primaries from https://refractiveindex.info/
 * ---------------------------------------------------------------------*/

namespace predefined_materials {

    ////////////////////////////////////////////////////////////////////////////////////////
    // Conductors

    /* Magnozzi et al. 2019 via refractiveindex.info */
    constexpr auto GOLD = [](
        color3 albedo = {1.f,1.f,1.f},
        float roughness = 0.05f
    ) -> conductor_params {
        return conductor_params{
            .albedo = albedo,
            .eta = color3{0.18535f, 0.33746f, 1.3760f},
            .kappa = color3{3.2633f, 2.5039f, 1.7987f},
            .roughness = roughness
        };
    };

    /* Cheng et al. 2016 via refractiveindex.info */
    constexpr auto ALUMINUM = [](
            color3 albedo = {1.f,1.f,1.f},
            float roughness = 0.05f
        ) -> conductor_params {
            return conductor_params{
                .albedo = albedo,
                .eta = color3{0.77045, 0.58815f, 0.41547f},
                .kappa = color3{5.8703f, 5.2666f, 4.4449f},
                .roughness = roughness
            };
        };

    ////////////////////////////////////////////////////////////////////////////////////////
    // Dielectrics

    /* N-BK7 (SCHOTT) via refractiveindex.info */
    constexpr auto GLASS_BK7 = [](
            color3 albedo = {1.f,1.f,1.f},
            float roughness = 0.f,
            dielectric_render_method render_method = dielectric_render_method::FRESNEL
        ) -> dielectric_params {
            return dielectric_params{
                .albedo = albedo,
                .eta = color3{1.5158f, 1.5186f, 1.5240f},
                .c_absorption = color3{0.23310f, 0.16407f, 0.27558f},
                .roughness = roughness,
                .render_method = render_method
            };
        };

    /* Ciddor 1996 via refractiveindex.info */
    constexpr auto AIR = [](
            color3 albedo = {1.f,1.f,1.f},
            float roughness = 0.f,
            dielectric_render_method render_method = dielectric_render_method::SCHLICK
        ) -> dielectric_params {
            return dielectric_params{
                .albedo = albedo,
                .eta = color3{1.00027681f, 1.00027786f, 1.00028001f},
                .c_absorption = color3{0.f, 0.f, 0.f},
                .roughness = roughness,
                .render_method = render_method
            };
        };

}
////////////////////////////////////////////////////////////////////////////////////////////////

#endif //PREDEFINED_MATERIALS_H
