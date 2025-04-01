# rlab-conversion

## Description
Lightweight utilities to convert OSort outputs to NWB format.

## Features
- NWB file initialization
- Conversion of OSort outputs to NWB format

## Installation
1. Clone the repository
2. Install dependencies via package manager

## Usage
The two components you're probably most interested in are `make_nwb_file`, which
will initialize an NWB file in the correct format for one of our epilepsy patients,
and `OsortSortingInterface`, which reads an OSort output directory and adds 
all the cell files to the NWB file. The examples directory includes a demonstration 
of how to use these together. In total you will need:
1. Directory with final OSort outputs (probably `sort/final`).
2. Events file, which should be your `events.Ncs` file saved as a `.csv`. This
can be done easily using Ueli's matlab Nlx utils and Matlab's `writematrix`
function. 
3. Config file with experiment metadata (task, experimenter, etc.). This exists
to fill in metadata required by the NWB format. See examples for what this should 
include.

The `trials` object you add to the NWB file will be experiment specific and is 
best contained within your project folder. Basically process your events into a 
DataFrame and add them to the NWB file. Or add the events in some other format.
Totally up to you.

## License
Distributed under the MIT license. Check LICENSE file for details.