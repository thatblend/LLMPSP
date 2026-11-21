/* Benchmark-only copies of the FHQ4 version-2 four-row Q4 tile kernel
 * (fp16 block scales), kept so the section 2.8 benchmark still measures
 * exactly what it measured: falcon_q4_tile4_vtfm (the v1.3 app kernel)
 * and falcon_q4_tile4_vdot, the same arithmetic with eight vdot.q instead
 * of two vtfm4.q per tile, to cross-check the documented vtfm4 timing on
 * hardware. The version-2 tile format and kernel are documented in
 * tools/q4_tile_v2_psp.h; the app's version-3 kernel is in
 * src/falcon_q4_tile_psp.h. */
#include <stdint.h>

#if defined(_PSP_FW_VERSION)

#include "q4_tile_v2_psp.h"

DEFINE_TILE_KERNEL(falcon_q4_tile4_vtfm, MUL_VTFM)
DEFINE_TILE_KERNEL(falcon_q4_tile4_vdot, MUL_VDOT)

#else
/* ISO C forbids an empty translation unit, hence the typedef. */
typedef int falcon_q4_tile_psp_unused;
#endif
