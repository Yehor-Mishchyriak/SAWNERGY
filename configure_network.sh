#!/usr/bin/bash

# Required arguments
topology_file=$1
trajectory_file=$2
number_frames=$3
cpptraj_analysis_command=$4
cpptraj_output_type=$5

# Optional arguments with default values
start_frame=""
batch_size=""
in_one_batch=""
output_directory=""

# Parse optional arguments
shift 5
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --start_frame) start_frame=$2; shift ;;
        --batch_size) batch_size=$2; shift ;;
        --in_one_batch) in_one_batch=$2; shift ;;
        --output_directory) output_directory=$2; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Construct the command with optional arguments if provided
cmd="python3 FramesAnalyser.py $topology_file $trajectory_file $number_frames $cpptraj_analysis_command $cpptraj_output_type"

if [[ -n $start_frame ]]; then
    cmd="$cmd --start_frame $start_frame"
fi

if [[ -n $batch_size ]]; then
    cmd="$cmd --batch_size $batch_size"
fi

if [[ -n $in_one_batch ]]; then
    cmd="$cmd --in_one_batch $in_one_batch"
fi

if [[ -n $output_directory ]]; then
    cmd="$cmd --output_directory $output_directory"
fi

# Pipe the command to the other scripts
$cmd | python3 AnalysesProcessor.py | python3 AtToResConverter.py | python3 InteractionsToProbsConverter.py
