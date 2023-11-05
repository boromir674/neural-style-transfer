#!/bin/bash

# Function to print the help message
show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -d, --debug            Enable debug mode (verbose output)"
    # echo "  -b, --build-location   Specify the build location (default: 'dist')"
    echo "  -s, --pkg-semver       Specify the package semantic version (required)"
    echo "  -h, --help             Show this help message"
}

# Initialize variables
DEBUG=false
BUILD_LOCATION=""
PKG_SEMVER=""

# Default build location
DEFAULT_BL="dist"

        # -b|--build-location)
        #     BUILD_LOCATION="$2"
        #     shift 2
        #     ;;

# Process command-line options
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -d|--debug)
            DEBUG=true
            shift
            ;;
        -s|--pkg-semver)
            PKG_SEMVER="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Error: Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# # Use DIST_DIR if BUILD_LOCATION is not specified
# if [ -z "$BUILD_LOCATION" ]; then
#     if [ -n "$DIST_DIR" ]; then
#         BUILD_LOCATION="$DIST_DIR"
#     else
#         BUILD_LOCATION="$DEFAULT_BL"
#     fi
# fi

# # Verify that BUILD_LOCATION is an existing directory
# if [ ! -d "$BUILD_LOCATION" ]; then
#     echo "Error: The specified build location '$BUILD_LOCATION' does not exist."
#     exit 1
# fi

# Check if PKG_SEMVER is provided
if [ -z "$PKG_SEMVER" ]; then
    echo "Error: Package semantic version is not specified. Please provide a semantic version using the -s or --pkg-semver option."
    show_help
    exit 1
fi


# Here we Support the below:

# We support official Sem Ver 2.0 to the extend possible
# If input version string follows sem ver 2.0 for the required starting M.m.p, we support parsing it

# for the rest we only support
# optional pre-release metadata, which if present must:
# - separate with dash (-) from M.m.p (ie 1.0.0-dev)
# - only include characters from [A-Za-z]

# PyPI: POETRY
# sdist and wheel through poetry back end:

## TAR GZ .tar.gz
# 1.0.1     --> artifact-1.0.1.tar.gz
# 1.0.1-dev -->  artificial_artwork-1.0.1.dev0.tar.gz
# 1.0.1-rc  -->  artificial_artwork-1.0.1.rc0.tar.gz

## WHEEL .whl
# 1.0.1-dev -->  artificial_artwork-1.0.1.dev0-py3-none-any.whl

# PyPI: BUILD
# sdist and wheel through build back end:
# 1.0.1-dev -->  artificial_artwork-1.0.1.dev0.tar.gz
# 1.0.1-dev -->  artificial_artwork-1.0.1.dev0-py3-none-any.whl

# Input should match regex below
# ^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)(?:-(?P<prerelease>(?:[a-z]+)))?$

# Convert the user-provided semantic version
converted_semver=""

# 1. if dash (-) is found right after the M.m.p then store the M.m.p, remove - and concat potential rest of string
if [[ "$PKG_SEMVER" =~ ^([0-9]+\.[0-9]+\.[0-9]+)-(.*)$ ]]; then
    converted_semver="${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
else
    converted_semver="$PKG_SEMVER"
fi

# 2. Hard append 0
converted_semver="${converted_semver}0"


# Set the package semantic version
pkg_semver="$converted_semver"

# Debug mode
if [ "$DEBUG" = true ]; then
    echo "[DEBUG] Debug mode enabled"
    echo "[DEBUG] BUILD_LOCATION: $BUILD_LOCATION"
    echo "[DEBUG] pkg_semver: $pkg_semver"
fi

echo "$pkg_semver"

# # Search for tar.gz and whl files in the specified build location
# artifacts=($(find "$BUILD_LOCATION" -type f -name "*$pkg_semver*"))

# # Print the artifacts as a space-separated string
# for artifact in "${artifacts[@]}"; do
#     echo -n "$artifact "
# done
