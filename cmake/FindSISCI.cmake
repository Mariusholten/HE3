include(FindPackageHandleStandardArgs)

set(DIS_PATH "/opt/DIS"
    CACHE
    PATH "Path to dolphin installation directory"
)

find_library(SISCI_LIBRARY sisci
             HINTS ${DIS_PATH}/lib64)

find_path(SISCI_INCLUDE_DIR
          NAMES sisci_api.h
          HINTS ${DIS_PATH}/include)

find_package_handle_standard_args(SISCI REQUIRED_VARS SISCI_LIBRARY SISCI_INCLUDE_DIR)

if (SISCI_FOUND)
    mark_as_advanced(SISCI_LIBRARY)
    mark_as_advanced(SISCI_INCLUDE_DIR)
endif()

if (SISCI_FOUND AND NOT TARGET SISCI)
    add_library(SISCI SHARED IMPORTED)
    set_property(TARGET SISCI PROPERTY IMPORTED_LOCATION ${SISCI_LIBRARY})
    target_include_directories(SISCI INTERFACE
                               ${SISCI_INCLUDE_DIR}
                               ${SISCI_INCLUDE_DIR}/dis)
endif()


