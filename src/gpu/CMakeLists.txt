# Metal GPU compute backend. One Objective-C++ translation unit (MetalContext.mm)
# owns all the ObjC/Metal interop behind a pure-C++ header, so the rest of the
# codebase stays C++20. Shaders are compiled from embedded MSL at runtime, so no
# offline `metal` toolchain is required -- only the Metal + Foundation frameworks.
add_library(bochner_gpu MetalContext.mm)
target_link_libraries(bochner_gpu PUBLIC bochner_grid)
target_link_libraries(bochner_gpu PRIVATE ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
# Consumers get the BOCHNER_WITH_METAL define so they can guard GPU-only code.
target_compile_definitions(bochner_gpu PUBLIC BOCHNER_WITH_METAL)
# ARC for the .mm (automatic retain/release of the ObjC Metal objects).
set_source_files_properties(MetalContext.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
