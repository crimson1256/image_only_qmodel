#!/bin/sh
lli=${LLVMINTERP-lli}
exec $lli \
    /home/hisky/DEEPCALO/deepcalo-with-hls-4-ml-v0.2_2022_5/demos/atlas_specific_usecases/train_recommended_electron_models_for_hls4ml/hls_qmodel/imgonly_qmodel_bnfold_4/myproject_prj/solution1/.autopilot/db/a.g.bc ${1+"$@"}