# Sequential Neural Likelihood

## How to run the experiments

Each file in folder `exps` contains the description of one or more experiments. Using Lotka-Volterra as an example, the files are organized like this:

File           | Experiments
---------------|------------
`lv_nl.txt`    | Neural Likelihood
`lv_seq.txt`   | Sequential methods: SNL, SNPE-A and SNPE-B
`lv_smc.txt`   | SMC-ABC
`lv_sl.txt`    | Synthetic Likelihood
`lv_calib.txt` | Calibration experiment for SNL

Replace `lv` with `gauss` for the toy Gaussian model, `mg1` for the M/G/1 queue, or `hh` for Hodgkin-Huxley.  Note that to run the Hodgkin-Huxley simulation, it is necessary to install NEURON; see [How to install NEURON](#how-to-install-neuron).

You can run all experiments in file `lv_nl.txt` by:
```
python main.py run exps/lv_nl.txt
```
Depending on the experiment, this can take from a couple of hours to a couple of weeks.
After the experiments are finished, the results will be saved in folder `data/experiments`.
After running the experiments, you can view the results by:
```
python main.py view exps/lv_nl.txt
```
This will produce several plots, depending on the experiment.
You can also print the experiment log by:
```
python main.py log exps/lv_nl.txt
```
This will print the experiment log on the screen.

For the calibration test, you need to run a particular experiment multiple times, each time with different parameters randomly drawn from the prior. You can do this by:
```
python main.py trials 1 200 exps/lv_calib.txt
```
This runs the experiment(s) described in `lv_calib.txt` for 200 trials, and labels the trials 1..200. Each trial is independent, so you can save time by issuing multiple commands in paralell, for example:
```
python main.py trials 1 10 exps/lv_calib.txt
python main.py trials 11 20 exps/lv_calib.txt
...
python main.py trials 191 200 exps/lv_calib.txt
```
This will produce the same result as above.

## How to reproduce the figures

After having run all the experiments in folder `exps`, you can reproduce the figures as follows:

Command                                | Figure it reproduces
---------------------------------------|----------------------------
`python plot_results_mmd.py <sim>`     | MMD vs simulation cost (only works for the Gaussian model)
`python plot_results_lprob.py <sim>`   | Minus log probability vs simulation cost
`python plot_results_dist.py <sim>`    | Distance vs number of rounds
`python plot_results_gof.py <sim>`     | Likelihood goodness-of-fit vs simulation cost
`python plot_results_calib.py <sim>`   | Histograms for the calibration test
`python plot_results_post.py <sim>`    | Posterior histograms and pairwise scatter plots for SNL

Here, `<sim>` can be any of `gauss`, `mg1`, `lv`, or `hh`. The first time you run one of the above it may take a long time, because the code will be calculating lots of intermediate results, such as MCMC samples, MMDs, etc. All intermediate results will be saved in folder `data/results` for future use. The second time you run it, the figures should appear immediately.

## How to install NEURON

The SNL code has been tested with NEURON on Scientific Linux 7.3. The following steps should allow the code to run on multiple versions of Linux:

1. Install NEURON 7.5 from using the 64 bit .deb (Debian or Ubuntu) or .rpm (Fedora derivatives) precompiled installer from https://neuron.yale.edu/neuron/download/precompiled-installers

2. Set `PYTHONPATH=/usr/local/nrn/lib/python:$PYTHONPATH`

3. In this directory run `nrnivmodl`, which will compile the `.mod` files into an executable file in a new directory, `x86_64`.
