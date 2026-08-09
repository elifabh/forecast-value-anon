# Forecast error is not forecast value

Anonymous code release for a double-blind workshop submission.

## What this is

A model predictive controller dispatches a single storage unit on a
distribution feeder to keep bus voltages inside the statutory band. Three
arms differ only in the solar forecast supplied to the planner: a day-ahead
persistence forecast, a day-ahead numerical weather prediction post-processed
by a gradient-boosted model, and perfect solar foresight. Every dispatch is
projected onto a full AC power flow before it is applied.

The question is not which forecast has the lower error, but which produces
the better control outcome.

## Layout

    mpc_econ_voltage_pareto.py   planner, safe layer wiring, shared state
    diag_mpc_forecastvolt.py     linearised distribution power flow
    safe_layer.py                AC projection (first feeder)
    gg2/safe_layer_ob.py         AC projection, masked (second feeder)
    gg2/mpc_ob.py                parametric window solver (second feeder)
    align_clean.py               half-hourly alignment of the input series
    fetch_*.py, load_demand.py   data loaders

    gg2/gg2_ob1..ob11_*.py       second feeder: calibration, arms, bootstrap,
                                 curtailment, thermal audit
    gg2/gg2_ob12_seasonfc.py     leakage-free forecaster for the summer block
    gg2/gg2_ob13_make.py         derives the summer-block runner from ob8
    gg2/gg2_ob14_make.py         derives the parametric bootstrap from ob9
    gg2/*.log                    run logs for the second-feeder experiments

## How the numbers were verified

Each script prints its anchors before it runs. A new pipeline is not trusted
until it reproduces a previously published number: the summer-block runner
reproduces the test-block levels exactly, and the parametric bootstrap
reproduces the first feeder's headline difference to the reported precision.
The derivation scripts (`*_make.py`) apply a fixed list of textual edits to
the original file and stop if any pattern fails to match exactly once, so the
derived scripts cannot silently drift from the originals.

Thresholds and the sentence to be written under each outcome were fixed
before each comparison was run; the relevant text sits in the docstring of
the script that runs it.

## Data

Raw data is not redistributed here. The input series are demand and
day-ahead price for the Republic of Ireland, solar and wind generation from
the Irish system operator, and day-ahead irradiance forecasts from a public
weather API. The loader scripts above fetch each series from its source; see
the paper for the full citations and access dates. Comments and docstrings were originally written in Turkish and machine
translated; the numbers in the logs are untouched.

## Environment

Python 3.13, pandapower, cvxpy, LightGBM, numpy, pandas. Results were
reproduced across two pandapower versions, two operating systems, and with
and without the JIT backend.
