# How to set up ReBeL (for 2021)

## Installation Guide

1. Setting up ReBeL using `conda` is **HIGHLY** recommended. Both Mukesh and I built it on a `conda` environment, so if you try to build it on `pip` virtual environment, we might not know how to solve the errors.
2. When creating a conda environment, make sure to install with python v3.7.0
3. Run `pip install -r requirements.txt`
4. Next, run `conda install cmake`
5. `git submodule update --init` did not work for us, so we directly git cloned from the github repo. Instead first create a directory called `third_party` and then run the following two commands:

```
git clone https://github.com/google/googletest.git
git clone https://github.com/pybind/pybind11.git
```

6. According to https://github.com/facebookresearch/hanabi_SAD/issues/8, pybind11 on the master branch for pytorch v1.3.0 does not work, so here is the quick workaround:
   1. First run `git checkout a1b71df`
   2. Now run:
   ```
   git config --unset core.bare
   git reset --hard
   ```
7. Once you run `make`, you might get an error in regards to Caffe trying to locate the CuDNN file. The solution to this problem is to modify the paths in `build/CMakeCache.txt`. Particularly, you want to specify `CUDNN_INCLUDE_DIR:PATH`, `CUDNN_INCLUDE_PATH:PATH`, `CUDNN_LIBRARY:PATH`, `CUDNN_LIBRARY:PATH`, and `CUDNN_ROOT:PATH`

Once you have completed all these steps, your ReBeL environment should be able to successfully build and be able to run.
