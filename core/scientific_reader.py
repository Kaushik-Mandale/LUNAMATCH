"""Scientific Raster Reader & Product Loader for LunaMatch V3."""
from core.sr_spec import (
    ProductType,
    ScientificRasterSpec,
    scientific_raster_spec,
    check_file_size_consistency,
    apply_lroc_nac_edr_pds3_profile,
    is_lroc_nac_edr_product,
    LROC_NAC_EDR_PDS3_PROFILE_ID,
)
from core.sr_load import (
    classify_product,
    classify_product_display_label,
    inspect_scientific_product,
    load_scientific_preview,
    load_product,
    get_raster_shape,
    get_raster_dtype,
)

__all__ = [
    "ProductType",
    "ScientificRasterSpec",
    "scientific_raster_spec",
    "check_file_size_consistency",
    "classify_product",
    "classify_product_display_label",
    "inspect_scientific_product",
    "load_scientific_preview",
    "load_product",
    "get_raster_shape",
    "get_raster_dtype",
    "apply_lroc_nac_edr_pds3_profile",
    "is_lroc_nac_edr_product",
    "LROC_NAC_EDR_PDS3_PROFILE_ID",
]
