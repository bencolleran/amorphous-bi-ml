#!/bin/bash
  
while true; do
    jf job rerun -s CHECKED_OUT -fid e54fff74-7840-4e43-a7d9-10c219cba44c 
    jf job rerun -s REMOTE_ERROR -fid e54fff74-7840-4e43-a7d9-10c219cba44c
    jf job rerun -s FAILED -fid e54fff74-7840-4e43-a7d9-10c219cba44c
    echo "Command executed. Waiting 3 mins before the next run."
 
    seconds=180
    while [ $seconds -gt 0 ]; do
        mins=$((seconds / 60))
        printf "\rTime remaining: %02d min" $mins
        sleep 60
        seconds=$((seconds - 60))
    done
    echo
done