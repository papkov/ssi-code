#!/bin/bash

# Move to the working directory
if ${PWD##*/} == "experiments" ; then
    cd ..
fi

echo "$PWD"

for image in ./ssi/benchmark/images/generic_2d_all/gt/*.png; do
  [ -e "$image" ] || continue
  echo "${image##*/}"
  python main.py image="${image##*/}" two_pass=True
done
