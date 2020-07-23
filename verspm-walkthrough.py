# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # VERSPM Model Interface

# %%
import emat
import os
import pandas as pd
import numpy as np
import gzip
from emat.util.show_dir import show_dir, show_file_contents

# %% [markdown]
# This notebook is meant to illustrate the use of TMIP-EMAT's
# various modes of operation.  It provides an illustration of how to use 
# TMIP-EMAT and the demo interface to run the command line version
# of the [Road Test](https://tmip-emat.github.io/source/emat.examples/RoadTest/road_test_yaml.html) 
# model. A similar approach can be developed to run
# any transportation model that can be run from the command line, including
# for proprietary modeling tools that are typically run from a graphical
# user interface (GUI) but that provide command line access also.

# %% [markdown]
# In this example notebook, we will activate some logging features.  The 
# same logging utility is written directly into the EMAT and the
# `core_files_demo.py` module. This will give us a view of what's happening
# inside the code as it runs.

# %%
import logging
from emat.util.loggers import log_to_stderr
log = log_to_stderr(logging.INFO)

# %% [markdown]
# ## Connecting to the Model

# %% [markdown]
# The interface for this model is located in the `core_files_demo.py`
# module, which we will import into this notebook.  This file is extensively
# documented in comments, and is a great starting point for new users
# who want to write an interface for a new bespoke travel demand model.

# %%
import emat_verspm

# %% [markdown]
# Within this module, you will find a definition for the 
# `RoadTestFileModel` class.
#
# We initialize an instance of the model interface object.
# If you look at the module code, you'll note the `__init__` function
# does a number of things, including creating a temporary directory
# to work in, copying the needed files into this temporary directory,
# loading the scope, and creating a SQLite database to work within.
# For your implementation, you might or might not do any of these steps.
# In particular, you'll probably want to use a database that is
# not in a temporary location, so that the results will be available
# after this notebook is closed.

# %%
fx = emat_verspm.VERSPModel()

# %%

# %% [markdown]
# Once we have loaded the `RoadTestFileModel` class, we have
# a number of files available in the "master_directory" that 
# was created as that temporary directory:

# %% [markdown]
# ## Understanding Directories
#
# The TMIP-EMAT interface design for files-based bespoke models uses
# pointers for several directories to control the operation of the 
# model.
#
# - **local_directory**  
#     This is the working directory for this instance of TMIP-EMAT,
#     not that for the core model itself. Typically it can be Python's 
#     usual current working directory, accessible via `os.getcwd()`.
#     In this directory typically you'll have a TMIP-EMAT model 
#     configuration *yaml* file, a scope definition *yaml* file, and
#     a sub-directory containing the files needed to run the core model
#     itself.
#
# - **model_path**  
#     The relative path from the `local_directory` to the directory where
#     the core model files are located.  When the core model itself is actually
#     run, this should be to the "current working directory" for that run.
#     The `model_path` must be given in the model config *yaml* file.
#
# - **rel_output_path**  
#     The relative path from the `model_path` to the directory where
#     the core model output files are located. The default value of this 
#     path is "./Outputs" but this can be overridden by setting 
#     `rel_output_path` in the model config *yaml* file. If the outputs
#     are comingled with other input files in the core model directory,
#     this can be set to "." (just a dot).
#
# - **archive_path**  
#     The path where model archive directories can be found. This path
#     must be given in the model config *yaml* file. It can be given as
#     an absolute path, or a relative path. If it is a relative path, 
#     it should be relative to the `local_directory`.
#     
# These directories, especially the ones other than the `local_directory`,
# are defined in a model configuration *yaml* file. This makes it easy to
# change the directory pointers when moving TMIP-EMAT between different
# machines that may have different file system structures.

# %% [markdown]
# ## Single Run Operation for Development and Debugging

# %% [markdown]
# Before we take on the task of running this model in exploratory mode, we'll
# want to make sure that our interface code is working correctly. To check each
# of the components of the interface (setup, run, post-process, load-measures,
# and archive), we can run each individually in sequence, and inspect the results
# to make sure they are correct.

# %% [markdown]
# ### setup
#
# This method is the place where the core model *set up* takes place,
# including creating or modifying files as necessary to prepare
# for a core model run.  When running experiments, this method
# is called once for each core model experiment, where each experiment
# is defined by a set of particular values for both the exogenous
# uncertainties and the policy levers.  These values are passed to
# the experiment only here, and not in the `run` method itself.
# This facilitates debugging, as the `setup` method can be used 
# without the `run` method, as we do here. This allows us to manually
# inspect the prepared files and ensure they are correct before
# actually running a potentially expensive model.
#
# Each input exogenous uncertainty or policy lever can potentially
# be used to manipulate multiple different aspects of the underlying
# core model.  For example, a policy lever that includes a number of
# discrete future network "build" options might trigger the replacement
# of multiple related network definition files.  Or, a single uncertainty
# relating to the cost of fuel might scale both a parameter linked to
# the modeled per-mile cost of operating an automobile and the
# modeled total cost of fuel used by transit services.

# %% [markdown]
# For this demo model, running the core model itself in files mode 
# requires two configuration files to be available, one for levers and
# another for uncertainties.  These two files are provided in the demo
# in two ways: as a runnable base file (for the levers) and as a template
# file (for the uncertainties).

# %% [markdown]
# The levers file is a *ready-to-use* file (for this demo, in YAML format,
# although your model may use a different file format for input files).
# It has default values pre-coded into the file, and to modify this 
# file for use by EMAT the `setup` method needs to parse and edit this
# file to swap out the default values for new ones in each experiment.
# This can be done using regular expressions (as in this demo), or any other method you
# like to edit the file appropriately.  The advantage of this approach
# is that the base file is ready to use with the core model as-is, facilitating
# the use of this file outside the EMAT context.

# %%
#show_file_contents(fx.master_directory.name, 'road-test-files', 'demo-inputs-l.yml')

# %% [markdown]
# By contrast, the uncertainties file is in a *template* format. The
# values of the parameters that will be manipulated by EMAT for each 
# experiment are not given by default values, but instead 
# each value to be set is indicated in the file by a unique token that is easy to
# search and replace, and definitely not something that appear in any script otherwise.
# This approach makes the text-substitution code that is used in this module much
# simpler and less prone to bugs.  But there is a small downside of this approach:
# every parameter must definitely be replaced in this process, as the template file
# is unusable outside the EMAT context, and also every unique token needs to be replaced. 

# %%
#show_file_contents(fx.master_directory.name, 'road-test-files', 'demo-inputs-x.yml.template')

# %% [markdown]
# Regardless of which file management system you use, the `setup` method
# is the place to make edits to these input files and write them into 
# your working directory.  To do so,
# the `setup` method takes one argument: a dictionary containing key-value
# pairs that assign a particular value to each input (exogenous uncertainty 
# or policy lever) that is defined in the model scope.  The keys must match 
# exactly with the names of the parameters given in the scope. 
#
# If you have written your `setup` method to call the super-class `setup`,
# you will find that if you give keys as input that are not defined in
# the scope, you'll get a KeyError.

# %%
# bad_params = {
#     'name_not_in_scope': 'is_a_problem',
# }

# try:
#     fx.setup(bad_params)
# except KeyError as error:
#     log.error(repr(error))

# %% [markdown]
# On the other hand, your custom model may or may not allow you to leave out
# some parameters.  It is up to you to decide how to handle missing values, 
# either by setting them at their default values or raising an error. In 
# normal operation, parameters typically won't be left out from the design
# of experiments, so it is not usually important to monitor this carefully.
#
# In our example module's `setup`, all of the uncertainty values must be given,
# because the template file would be unusable otherwise. But the policy levers 
# can be omitted, and if so they are left at their default values in the 
# original file.  Note that the default values in that file are not strictly
# consistent with the default values in the scope file, and TMIP-EMAT does 
# nothing on its own to address this discrepancy.

# %%
params = {
    'ValueOfTime': 13,
    'Income': 46300,
} 

fx.setup(params)

# %% [markdown]
# After running `setup` successfully, we will have overwritten the 
# "demo-inputs-l.yml" file with new values, and written a new 
# "demo-inputs-x.yml" file into the model working directory with those
# values.

# %%
show_dir(fx.local_directory)

# %%
show_file_contents(fx.local_directory, 'VERSPM', 'defs', 'model_parameters.json')

# %% [markdown]
# ### run

# %% [markdown]
# The `run` method is the place where the core model run takes place.
# Note that this method takes no arguments; all the input
# exogenous uncertainties and policy levers are delivered to the
# core model in the `setup` method, which will be executed prior
# to calling this method. This facilitates debugging, as the `setup`
# method can be used without the `run` method as we did above, allowing
# us to manually inspect the prepared files and ensure they
# are correct before actually running a potentially expensive model.

# %%
fx.run()

# %% [markdown]
# The `RoadTestFileModel` class includes a custom `last_run_logs` method,
# which displays both the "stdout" and "stderr" logs generated by the 
# model executable during the most recent call to the `run` method.
# We can use this method for debugging purposes, to identify why the 
# core model crashes (if it does crash).  In this first test it did not
# crash, and the logs look good.

# %%
fx.last_run_logs()

# %%
show_dir(os.path.join(fx.master_directory.name, 'VERSPM', 'output'))

# %%
os.path.join(fx.master_directory.name, 'VERSPM', 'output')

# %% [markdown]
# ### post-process
#
# There is an (optional) `post_process` step that is separate from the `run` step.
#
# Post-processing differs from the main model run in two important ways:
#
# - It can be run to efficiently generate a subset of performance measures.
# - It can be run based on archived model main-run core model results.
#
# Both features are designed to support workflows where new performance 
# measures are added to the exploratory scope after the main model run(s)
# are completed. By allowing the `post_process` method to be run only for a 
# subset of measures, we can avoid replicating possibly expensive 
# post-processing steps when we have already completed them, or when they
# are not needed for a particular application.  
#
# For example, consider an exploratory modeling activity where the scope 
# at the time of the initial model run experiments was focused on highway
# measures, and transit usage was not explored extensively, and no 
# network assignment was done for transit trips when the experiments were
# initially run.  By creating a post-process step to run the transit 
# network assignment, we can apply that step to existing archived results,
# as well as have it run automatically for future model experients
# where transit usage is under study, but continue to omit it for future 
# model experients where we do not need it.
#
# An optional `measure_names` argument allows the post-processor to
# identify which measures need additional computational effort to generate,
# and to skip excluded measures that are not currently of interest, or
# which have already been computed and do not need to be computed again.
#
# The post processing is isolated from the main model run to allow it to
# be run later using archived model results.  When executed directly 
# after a core model run, it will operate on the results of the model
# stored in the local working directory.  However, it can also be
# used with an optional `output_path` argument, which can be pointed at
# a model archive directory instead of the local working directory.
#
# A consequence of this (and an intentional limitation) is that the 
# `post_process` method should only use files from the set of files 
# that are or will be archived from the core model run, and not attempt
# to use other non-persistent temporary or intermediate files that 
# will not be archived.

# %%
fx.post_process()

# %% [markdown]
# At this point, the model's output performance measures should be available in one
# or more output files that can be read in the next step.  For this example, the
# results are written to two separate files: 'output_1.csv.gz' and 'output.yaml'.

# %%
show_file_contents(fx.model_path, "output", "ComputedMeasures.json")

# %% [markdown]
# ### load-measures
#
# The `load_measures` method is the place to actually reach into
# files in the core model's run results and extract performance
# measures, returning a dictionary of key-value pairs for the 
# various performance measures. It takes an optional list giving a 
# subset of performance measures to load, and like the `post_process` 
# method also can be pointed at an archive location instead of loading 
# measures from the local working directory (which is the default).
# The `load_measures` method should not do any post-processing
# of results (i.e. it should read from but not write to the model
# outputs directory).

# %%
fx.load_measures()

# %% [markdown]
# You may note that the implementation of `RoadTestFileModel` in the `core_files_demo` module
# does not actually include a `load_measures` method itself, but instead inherits this method
# from the `FilesCoreModel` superclass. The instructions on how to actually find the relevant
# performance measures for this file are instead loaded into table parsers, which are defined
# in the `RoadTestFileModel.__init__` constructor.  There are [details and illustrations
# of how to write and use parsers in the file parsing examples page of the TMIP-EMAT documentation.](https://tmip-emat.github.io/source/emat.models/table_parse_example.html)

# %% [markdown]
# ### archive
#
# The `archive` method copies the relevant model output files to an archive location for 
# longer term storage.  The particular archive location is based on the experiment id
# for a particular experiment, and can be customized if desired by overloading the 
# `get_experiment_archive_path` method.  This customization is not done in this demo,
# so the default location is used.

# %%
fx.get_experiment_archive_path(parameters=params)

# %% [markdown]
# Actually running the `archive` method should copy any relevant output files
# from the `model_path` of the current active model into a subdirectory of `archive_path`.

# %%
fx.archive(params)

# %%
show_dir(fx.local_directory)

# %%
STOP

# %% [markdown]
# It is permissible, but not required, to simply copy the entire contents of the 
# former to the latter, as is done in this example. However, if the current active model
# directory has a lot of boilerplate files that don't change with the inputs, or
# if it becomes full of intermediate or temporary files that definitely will never
# be used to compute performance measures, it can be advisable to selectively copy
# only relevant files. In that case, those files and whatever related sub-directory
# tree structure exists in the current active model should be replicated within the
# experiments archive directory.

# %% [markdown]
# ## Normal Operation for Running Multiple Experiments

# %% [markdown]
# For this demo, we'll create a design of experiments with only 8 experiments.
# The `design_experiments` method of the `RoadTestFileModel` object is not defined
# in the custom `core_files_demo` written for this model, but rather is a generic
# function provide by the TMIP-EMAT main library.
# Real applications will typically use a larger number of experiments, but this small number
# is sufficient to demonstrate the operation of the tools.

# %%
design1 = fx.design_experiments(design_name='lhs_1', n_samples=8)
design1

# %% [markdown]
# The `run_experiments` command will automatically run the model once for each experiment in the named design.
# The demo command line version of the road test model is (intentionally) a little bit slow, so will take a few
# seconds to conduct these eight model experiment runs.

# %%
# fx.run_experiments(design_name='lhs_1')

# %% [markdown]
# Much better!  Now we can see we have a more complete set of outputs, without the NaN's.  Hooray!

# %%
# results = fx.db.read_experiment_all(scope_name=fx.scope.name, design_name='lhs_1')
# results

# %% [markdown]
# ## Multiprocessing for Running Multiple Experiments
#
# The examples above are all single-process demonstrations of using TMIP-EMAT to run core model
# experiments.  If your core model itself is multi-threaded or otherwise is designed to make 
# full use of your multi-core CPU, or if a single core model run will otherwise max out some
# computational resource (e.g. RAM, disk space) then single process operation should be sufficient.
#
# If, on the other hand, your core model is such that you can run multiple independent instances of
# the model side-by-side on the same machine, then you could benefit from a multiprocessing 
# approach.  This can be accomplished by splitting a design of experiments over several
# processes that you start manually, or by using an automatic multiprocessing library such as 
# `dask.distributed`.

# %% [markdown]
# ### Automatic Multiprocessing for Running Multiple Experiments
#
# The examples above are all essentially single-process demonstrations of using TMIP-EMAT to run core model
# experiments, either by running all in one single process, or by having a user manually instantiate a number 
# of single processes.  If your core model itself is multi-threaded or otherwise is designed to make 
# full use of your multi-core CPU, or if a single core model run will otherwise max out some
# computational resource (e.g. RAM, disk space) then single process operation should be sufficient.
#
# If, on the other hand, your model is such that you can run multiple independent instances of
# the model side-by-side on the same machine, but you don't want to manage the process of manually, 
# then you could benefit from a multiprocessing approach that uses the `dask.distributed` library.  To
# demonstrate this, we'll create yet another small design of experiments to run.

# %%
design3 = fx.design_experiments(design_name='lhs_3', n_samples=8, random_seed=3)
design3

# %% [markdown]
# The demo module is set up to facilitate distributed multiprocessing. During the `setup`
# step, the code detects if it is being run in a distributed "worker" environment instead of
# in a normal Python environment.  If the "worker" environment is detected, then a copy
# of the entire files-based model is made into the worker's local workspace, and the model
# is run there instead of in the master workspace.  This allows each worker to edit the files
# independently and simultaneously, without disturbing other parallel workers.
#
# With this small modification, we are ready to run this demo model in parallel subprocesses.
# to do, we simply import the `get_client` function, and use that for the `evaluator` argument
# in the `run_experiments` method.

# %%
from emat.util.distributed import get_client # for multi-process operation
fx.run_experiments(design=design3, evaluator=get_client())

# %%
os.path.exists(fx.get_experiment_archive_path(1)[:-6])

# %%
fx.local_directory

# %%

# %%

# %%

# %%

# %%
fx.archive_path = os.path.expanduser( "~/sandbox/ve8-temp/archive")
fx.archive_path

# %%
fx.load_archived_measures(1)

# %%
