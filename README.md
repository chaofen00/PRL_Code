# Trial-wise fMRI Pattern-Change Analysis Pipeline

A portable MATLAB and Python workflow for trial-wise fMRI analysis in probabilistic reversal learning. The repository covers cue- and outcome-locked least squares separate (LSS) estimation, searchlight-based multivoxel pattern change mapping, learning stage organization, ROI-informed voxel-wise regression, reinforcement learning analyses, and SPM12 second-level statistics.

## Overview

The pipeline is organized around trial-wise beta maps and adjacent-trial representational change. It supports the following analyses:

* cue-locked and outcome-locked LSS estimation;
* spherical searchlight pattern change mapping;
* organization of trials into LN, LE, RN, and RE stages;
* participant- and stage-level mean pattern change maps;
* paired and one-sample second-level tests in SPM12;
* ROI to pattern change regression with whole-gray-matter control;
* RN analyses separated by positive and negative value prediction errors;
* outcome-matched dual learning rate reinforcement learning regressions;
* voxel-wise regression linking pattern change to `delta\_policy\_dual`.

The scripts contain no machine-specific paths. Input and output locations are supplied through command-line arguments or MATLAB name-value arguments.

## Analysis workflow

```text
Preprocessed fMRI data
        |
        v
Cue- and outcome-locked LSS beta maps
        |
        v
Adjacent-trial searchlight pattern-change maps
        |
        +---------------------------+
        |                           |
        v                           v
Stage labeling                 ROI/RL regressions
LN / LE / RN / RE              seed, RPE, absPE,
        |                       interaction, policy change
        v                           |
Stage means and contrasts          v
        |                       Participant-level slope maps
        +-------------+-------------+
                      |
                      v
               SPM12 group analysis
```

