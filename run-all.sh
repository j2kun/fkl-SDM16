#!/bin/bash

python3 experimetn-baselines.py > results/baseline-results.txt &
python3 experiment-SDB.py > results/SDB-results.txt &
python3 experiment-RR.py > results/RR-results.txt &
python3 experiment-RM.py > results/RM-results.txt &
python3 experiment-FWL.py > results/FWL-results.txt &

