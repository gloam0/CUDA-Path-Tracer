#ifndef PREDEFINED_MATERIALS_H
#define PREDEFINED_MATERIALS_H

#include "material.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////
/* ---------------------------------------------------------------------
 * sRGB primaries (https://clarkvision.com/imagedetail/color-spaces/)
 * - red:   612nm
 * - green: 549nm
 * - blue:  465nm
 * - middle: 539nm (used for dielectrics)
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
        ) -> material {
            return material{
                material_type::Conductor,
                albedo,
                mat_properties{
                    .conductor = conductor_params{
                        .eta = color3{0.18535f, 0.33746f, 1.3760f},
                        .k = color3{3.2633f, 2.5039f, 1.7987f},
                        .roughness = roughness
                    }
                }
            };
        };

    /* Cheng et al. 2016 via refractiveindex.info */
    constexpr auto ALUMINUM = [](
            color3 albedo = {1.f,1.f,1.f},
            float roughness = 0.05f
        ) -> material {
            return material{
                material_type::Conductor,
                albedo,
                mat_properties{
                    .conductor = conductor_params{
                        .eta = color3{0.77045, 0.58815f, 0.41547f},
                        .k = color3{5.8703f, 5.2666f, 4.4449f},
                        .roughness = roughness
                    }
                }
            };
        };

    ////////////////////////////////////////////////////////////////////////////////////////
    // Dielectrics

    /* N-BK7 (SCHOTT) via refractiveindex.info */
    constexpr auto GLASS_BK7 = [](
            color3 albedo = {1.f,1.f,1.f},
            float roughness = 0.f,
            dielectric_render_method render_method = dielectric_render_method::FRESNEL
        ) -> material {
            return material{
                material_type::Dielectric,
                albedo,
                mat_properties{
                    .dielectric = dielectric_params{
                        .ior = 1.5191,
                        .roughness = roughness,
                        .render_method = render_method
                    }
                }
            };
        };

    /* Ciddor 1996 via refractiveindex.info */
    constexpr auto AIR = [](
            color3 albedo = {1.f,1.f,1.f},
            float roughness = 0.f,
            dielectric_render_method render_method = dielectric_render_method::FRESNEL
        ) -> material {
            return material{
                material_type::Dielectric,
                albedo,
                mat_properties{
                    .dielectric = dielectric_params{
                        .ior = 1.00027806,
                        .roughness = roughness,
                        .render_method = render_method
                    }
                }
            };
        };

}
////////////////////////////////////////////////////////////////////////////////////////////////

#endif //PREDEFINED_MATERIALS_H
