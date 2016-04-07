#!/bin/bash

python setup.py build_docs

# get git hash for commit message
GITHASH=$(git rev-parse HEAD)
MSG="doc build for commit $GITHASH"

cd build

# clone the repo if needed
if test -d lombscargle;
then echo "using existing cloned lombscargle directory";
else git clone https://github.com/jakevdp/lombscargle.git;
fi

# change to the correct branch
cd lombscargle
git checkout gh-pages

# sync the website
rsync -r ../sphinx/html/* ./
git add .
git commit -m MSG
git push origin gh-pages
