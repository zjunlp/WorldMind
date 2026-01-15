#!/bin/sh
GIT_CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
set -x
set -e
git checkout gh-pages
git fetch --all
git reset --hard origin/gh-pages
rm -rf $GIT_CURRENT_BRANCH && mkdir $GIT_CURRENT_BRANCH
cp -r docs/* $GIT_CURRENT_BRANCH/
git add .
git commit --amend -m 'update docs'
git push -f
git checkout $GIT_CURRENT_BRANCH
