# TMIP-EMAT and VisionEval

This repository contains a demo for creating a files-based core model interface
to run VisionEval's Regional Strategic Planning Model (RSPM) using TMIP-EMAT.
To use this demo, you must also install [VisionEval](https://visioneval.org) as well
as [TMIP-EMAT](https://tmip-emat.github.io) itself, following the [instructions](https://tmip-emat.github.io/source/emat.conda.html) 
provided.

The interface for this demo is defined in the python module `emat_verspm.py`. This
heavily commented file includes all of the various parts of an interface needed to
run RSPM automatically and store the results in a database.

The Jupyter notebook `verspm-walkthrough.ipynb` provides a short demo that walks through
developing and using the interface, including single runs, multiple experiments, and 
parallel processes.

After the model runs have been completed and stored in a database, you can do a 
variety of analysis with the results, including building metamodels.  Some of this 
analysis is shown in the `versp-interactive.ipynb` notebook.
