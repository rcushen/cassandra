# cassandra
Code files for the Cassandra monitoring tool

## Getting started

To develop Cassandra, it is strongly recommended that you use a virtual environment. The full dependency list is in `requirements.txt`, and can be installed with `pip install -r requirements.txt` once this environment has been configured and activated.

## Running the tests

To run the tests, run `pytest` from the root directory of the project. This will run all tests in the `tests` directory.

## Running the experiments

The experiment scripts are best run as modules, using the `-m` flag. For example, to run the scaling experiment, run `python -m experiments.scaling`.
