#############################
 neu_weight_util User Manual
#############################
**********
 Overview
**********

This document is about using neu_weight_util.

**************
 Requirements
**************

Python
========

Version: 3.10 or above

Required Package:
 - sentencepiece
 - transformers
 - numpy
 - tqdm

Operating System
================

Currently only supports Ubuntu (Debian)distribution


**************
Example usage
**************

Use Case: Use CodeLlama-7b-hf from huggingface

Clone model repository from huggingface
=============================================

.. code-block:: console
    :linenos:

    $ git lfs install # make sure your system support git-lfs
    $ git clone https://huggingface.co/codellama/CodeLlama-7b-hf


Convert model and generate model weight files
==============================================

.. code-block:: console
    :linenos:

    $ ./neu_weight_util --outfile CodeLlama-7b-hf.bin --model <your path>/CodeLlama-7b-hf/

 model weight files is in output_weight/CodeLlama-7b-hf

Download weight to target N3000 device
=======================================

.. code-block:: console
    :linenos:

    # /dev/neuchips_ai_epr-<DEV>
    # need root privileges to download weight
    $ sudo ./neu_weight_util --weight output_weight/CodeLlama-7b-hf --n3kid <DEV>

Convert model, generate model weight files, and download weight to target N3000 device
========================================================================================

.. code-block:: console
    :linenos:

    $ sudo ./neu_weight_util --outfile CodeLlama-7b-hf.bin --model <your path>/CodeLlama-7b-hf/ --n3kid <DEV>

