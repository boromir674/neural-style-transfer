#!/usr/bin/env python

# 'Assumptions': input string follows Sem Ver 2.0
# with 2 'Limitations' on pre-release metadata and build metadata:

# 1) if user wants to include pre-release info, they must
#     - separate with dash (-) from M.m.p (ie 1.0.0-dev)
#     - only include characters from [a-z]
# 2) no build metadata are supported and string MUST end with patch or pre-release metadata

import sys

if len(sys.argv) != 2:
    print("Usage: process_sem_ver.py <version>")
    print(f"Example: process_sem_ver.py 1.0.0-dev")
    sys.exit(1)

semver: str = sys.argv[1]

# Verify input string meets our 'Hard Requirements', otherwise show message:
# - indicating what happened and crashed the program
# - what caused the crash
# - how the user input is part of the cause
# -  what the user should try to do, to fix the issue

# 'Hard Requirements', based on 'Assumptions' and 'Limitations' (see above):
# - string must be at least 5 characters long
# - string mush have 2 dots
# - string must end with a patch or pre-release metadata
# - string must have a dash (-) if pre-release metadata is included
# - if prerelase metadata is included, it must be only characheters from [a-z]

# Verify string is at least 5 characters long
if len(semver) < 5:
    print("[ERROR]: Sem Ver Version string must be at least 5 characters long")
    print(f"Your input: {semver}")
    print(f"Your input is only {len(semver)} characters long")
    print("Please try again with a version string that is at least 5 characters long")
    sys.exit(1)

# Verify string has 2 dots
if semver.count(".") != 2:
    print("[ERROR]: Version string must have 2 dots")
    print(f"Your input: {semver}")
    print(f"Your input has {semver.count('.')} dots")
    print("Please try again with a version string that has 2 dots")
    sys.exit(1)


# Interact with the Sem Ver 2.0 Regular Expression at https://regex101.com/r/Ly7O1x/3/
# Sem Ver 2.0 Docs related section at: https://semver.org/#is-there-a-suggested-regular-expression-regex-to-check-a-semver-string

# Note: Sem Ver 2.0 requires dash (-) to separate pre-release metadata

# Verify string ends with a patch or pre-release metadata
## Sem Ver 2.0 requires dash (-) to separate pre-release metadata

if '-' not in semver:  # if no Sem Ver 2.0 prerelease separator found in the string
    # for us now it is impossible to have pre-release metadata
    # and the string MUST be in Major.Minor.Patch format only

    if semver[-1] not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        # ERROR: Should be dealing with M.m.p case, but last character is not digit
        print("[ERROR]: Version string must end with a patch or pre-release metadata")
        print(f"Your input: {semver}")
        print(f"Your input ends with {semver[-0]}")
        print("Please try again with a version string that ends with a patch or pre-release metadata")
        print(f"EXPLANATION: Since we did not find a dash (-) in the input, we expect to",
        " the Input Version String to be of 'Major.Minor.Patch' format."
        " So, 'Patch' must be the last part of the string, thus the last digit must be a number."
        f"But we found {semver[-0]} instead\n."
        "If you intended to include 'pre-release' metadata,"
        " please concatenate a dash (-) to the mandatory starting 'Major.Minor.Patch' part"
        " and then add your 'pre-release' metadata, ie '1.0.0-dev'.")
        sys.exit(1)

    # more reg ex checks can go here, but that not really the purpose of this script

else:  # if Sem Ver 2.0 prerelease separator found in the string
    
    # 1) Given the above condition
    # 2) Given scripts 'Limitations':
    #    - only characters [a-z] are allowed for prerelease metadata
    #    - we do not support 'build metadata' of Sem Ver 2.0
    
    # Then
    #  - expect to find exactly 1 dash (-), 
    #  - the dash is right after Patch (ie Major.Minor.Patch-Prerelaese)
    #  - only [a-z] characters are found in prerelease substring
    #  - prerelease substring is not empty

    if semver.count("-") != 1:
        # ERROR: Should be dealing with M.m.p-prerelase case, but more than 1 dash found
        print("[ERROR]: Version string must have exactly 1 dash (-)")
        print(f"Your input: {semver}")
        print(f"Your input has {semver.count('-')} dashes")
        print("Please try again with a version string that has exactly 1 dash (-)")
        print(f"EXPLANATION: Since, we found a dash (-) in the input, and given the script 'Limitation' that we do not support build-metadata, we expect",
        " the Input Version String to be of 'Major.Minor.Patch-Prerelease' format.")
        sys.exit(1)

    prerelease: str = semver.split('-')[1]  # get the prerelease substring

    if prerelease == '':
        # ERROR: Should be dealing with M.m.p-prerelase case, but prerelease substring is empty
        print("[ERROR]: Version string must have a non-empty prerelease substring")
        print(f"Your input: {semver}")
        print(f"Your input has an empty prerelease substring")
        print("Please try again with a version string that has a non-empty prerelease substring")
        print(f"EXPLANATION: Since, we found a dash (-) in the input, and given the script 'Limitation' that we do not support build-metadata, we expect",
        " the Input Version String to be of 'Major.Minor.Patch-Prerelease' format.")
        sys.exit(1)

    # english alphabet has 26 characters
    lowercase_chars = set(chr(ord('a') + i) for i in range(26))
    if lowercase_chars.issuperset(prerelease) is False:
        # ERROR: Should be dealing with M.m.p-prerelase case, but found non [a-z] characters in prerelease substring
        print("[ERROR]: Version string's prerelease must have only [a-z] characters")
        print(f"Your input: {semver}")
        print(f"Your input has {prerelease}")
        print("Please try again with a version string that has only [a-z] characters in prerelease substring")
        print(f"EXPLANATION: Since we found a dash (-) in the input, we expect to",
        " the Input Version String to be of 'Major.Minor.Patch-Prerelease' format.")
        sys.exit(1)

    # more reg ex checks can go here, but that not really the purpose of this script


# If we got here, then the input string is a valid Sem Ver 2.0 string
# And valid as input to the rest of the script

assert sys.argv[1] == semver

# CRITICAL to be in par with Pip sdist /wheel and python -m build operations

# here it safe to implement the logic simply as:
# if there is a dash then convert to dot and add trailing zero (0), else return as it is
print(
    sys.argv[1] if "-" not in sys.argv[1] else sys.argv[1].replace("-", ".") + "0"
)
