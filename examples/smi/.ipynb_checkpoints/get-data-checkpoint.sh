#!/usr/bin/env sh
mkdir -p data
cd data

wget -q http://bigwww.epfl.ch/smlm/challenge/datasets/Bundled_Tubes_Long_Sequence/sequence.zip
unzip -q sequence.zip
rm sequence.zip
