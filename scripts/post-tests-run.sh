#!/usr/bin/env bash

# MUST run from within the Repo Root Directory
# DESIGNED to run after tests have been run

# READ 1st Positional Arguments
destination_xml_file_path=$1


# Script that creates a Coverage XML file, by Aggregating all discovered Coverage
# data found at runtime (ie generated from Test Suite Run(s), that produced a
# coverage file per Run).

# This script should cover cases, where, for a given OS and Python version
# (ie os: Ubuntu, python: 3.10), the Test Suite ran against one or more
# 'python installation' modes (possible modes: 'in edit mode', or as
# sdist, as wheel).

# Running this script after Test Suite ran against all possible modes
# (at least one), will

# 1. Aggregate all coverage data files found in the 'coverage' directory

# 2. Create a single Coverage XML file, that contains all coverage data

# 3. Put the file into PWD directory and return the path to it


# gather individual coverage data produced by Test Suite
# runs against potentially multiple 'package' installations
# such as 'in edit mode', or as sdist (source distribution),
# or as wheel (potentially binary/compiled distribution)

# We gather all that info and export 2 files with same info
# but in different format (xml, html)

# Combine Coverage (ie dev, sdist, wheel) & make Reports (ie in xml, html)
# capture stdout stderr and exit code
# tox -e coverage --sitepackages -vv -s false
tox -e coverage --sitepackages -vv -s false 2>&1 | tee coverage-tox-run.log

# get exit code of tox run
TOX_RUN_EXIT_CODE=$?

# if tox run failed, exit with same exit code
if [ $TOX_RUN_EXIT_CODE -ne 0 ]; then
    echo "[ERROR] Tox run failed with exit code: $TOX_RUN_EXIT_CODE"
    exit $TOX_RUN_EXIT_CODE
fi

# START - Rename Coverage Files (POC Version)
platform="linux"
python_version="3.8"

# get coverage data file path
# try to copy coverage data file to destination, else print error and tox coverage run log
mv ./.tox/coverage.xml "${destination_xml_file_path}"

if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to copy coverage data file to destination: ${destination_xml_file_path}"
    echo "[DEBUG] Dumping tox -e coverage run output:"
    cat coverage-tox-run.log
    exit 1
fi

# END - Rename Coverage Files (POC Version)

echo " --- COVERAGE XML: ${destination_xml_file_path} --- "

# Github Actions original code
# mv ./.tox/coverage.xml ./coverage-${{ matrix.platform }}-${{ matrix.python-version }}.xml
