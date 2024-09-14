#!/bin/bash

# Get the last 10 commit hashes
commits=$(git log -n 10 --pretty=format:'%h')

# Convert the string of commits into an array
IFS=$'\n' read -d '' -r -a commit_array <<< "$commits"

# Print commit history
echo "Commit History:"
git log -n 10 --pretty=format:'%h - %an, %ad : %s' --date=short

# Print diff between the second and first commits with more verbose output
echo -e "\nDiff between the second and first commits:"
git diff --stat --patch -U5 ${commit_array[1]} ${commit_array[0]}

# Print diff between the third and second commits with more verbose output
echo -e "\nDiff between the third and second commits:"
git diff --stat --patch -U5 ${commit_array[2]} ${commit_array[1]}
