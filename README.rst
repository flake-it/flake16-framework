================================================================================================
Evaluating Features for Machine Learning Detection of Order- and Non-Order-Dependent Flaky Tests
================================================================================================

Replication Package

Contents
========

- ``showflakes/`` Our ShowFlakes `pytest <https://docs.pytest.org/en/6.2.x/>`_ plugin.
- ``subjects/`` Contains the dependencies of each of our 26 subject Python projects.
- ``testinspect/`` Our TestInspect `pytest <https://docs.pytest.org/en/6.2.x/>`_ plugin.
- ``Dockerfile`` Describes our Docker image used to create containers during experiments.
- ``experient.py`` Our experiment automation script.
- ``LICENSE`` The MIT License.
- ``requirements.txt`` Dependencies for our experiment automation script.
- ``subjects.txt`` CSV file containing information about our 26 subject Python projects.
- ``test_experiment.py`` Test suite for our experiment automation script.

Prerequisites
=============

- Install Python 3.8 and Docker on your system.
- ``pip install -r requirements.txt`` Install the dependencies for our experiment automation script.
- ``docker build -t flake16framework .`` Build the Docker image. (A prebuilt image can be found `here <https://drive.google.com/file/d/1yl8qQcsy6rXPq6uhrcFABC8LXgAtggDP/view?usp=sharing>`_)

Usage
=====

Our experiment automation script can be executed with ``python experiment.py COMMAND``, where ``COMMAND`` is one of the following:

- ``run *MODES`` Run the data collection phase of the experiment. This involves executing the 26 test suites of our subjects 5,000 times each to identify flaky tests and once more to collect feature data. A separate Docker container is created for each test suite run. Raw data is stored in ``data/`` and the standard output streams of the containers are recorded in ``stdout/``. ``*MODES`` is one or more modes, separated by spaces. There are three possible modes to choose from:

    - ``baseline`` Run the test suites in their original order.
    - ``shuffle`` Run the test suites in a randomized order each time.
    - ``testinspect`` Run the test suites with TestInspect to collect feature data.

- ``tests`` Collates the raw data from the previous command, producing a JSON file named ``tests.json``. Requires ``data/``. See the Data section for more details.
- ``scores`` Trains and evaluates machine learning models, producing a pickle file named ``scores.json``. Requires ``tests.json``. See the Data section for more details.
- ``shap`` Calculates SHAP values, producing a pickle file named ``shap.json``. Requires ``tests.json``. See the Data section for more details.
- ``figures`` Produces LaTeX figures. Requires ``tests.json``, ``scores.json``, and ``shap.json``.

Data
====

Our compressed experiment data can be found `here <https://drive.google.com/file/d/1mGHXdGxjt0zdjYJCUJRNunRh9vsBYzAd/view?usp=sharing>`_. It contains the following items:

- ``data/`` Output from ``python experiment.py run baseline shuffle testinspect``.
- ``tests.json`` Output from ``python experiment.py tests``.
- ``scores.pkl`` Output from ``python experiment.py scores``.
- ``shap.pkl`` Output from ``python experiment.py shap``.

``tests.json``
--------------

This is a JSON file that is structured as follows:

::

    {
        PROJECT_NAME: {
            TEST_NODEID: [
                REQ_RUNS,
                LABEL,
                *FEATURES
            ]
            ...
        },
        ...
    }

- ``PROJECT_NAME`` The name of a subject project.
- ``TEST_NODEID`` The unique identifier of a test case as assigned by `pytest <https://docs.pytest.org/en/6.2.x/>`_.
- ``REQ_RUNS`` The number of runs required to identify this flaky test (0 if non-flaky).
- ``LABEL`` 0 if non-flaky, 1 if NOD flaky, 2 if OD flaky.
- ``*FEATURES`` Values for the 16 features of Flake16, in the order of the ``FEATURE_NAMES`` variable on line 65 of ``experiment.py``.

``scores.pkl``
--------------

This is a pickle file that is structured as follows:

::

    {
        [FLAKY_TYPE, FEATURE_SET, PREPROCESSING, BALANCING, MODEL]: [
            T_TRAIN,
            T_TEST,
            {
                PROJECT_NAME: [
                    FALSE_POSITIVES_PROJECT,
                    FALSE_NEGATIVES_PROJECT,
                    TRUE_POSITIVES_PROJECT,
                    PRECISION_PROJECT,
                    RECALL_PROJECT,
                    F1_SCORE_PROJECT
                ],
                ...
            },
            [
                FALSE_POSITIVES,
                FALSE_NEGATIVES,
                TRUE_POSITIVES,
                PRECISION,
                RECALL,
                F1_SCORE
            ]
        ]
        ...
    }


- ``FLAKY_TYPE`` The type of flaky test being detected, either "NOD" or "OD.
- ``FEATURE_SET`` The feature set used, either "Flake16" or "FlakeFlagger".
- ``PREPROCESSING`` The data preprocessing procedure, either "None", "Scaling", or "PCA".
- ``BALANCING`` The data balancing procedure, either "None", "Tomek Links", "SMOTE", "ENN", "SMOTE ENN", or "SMOTE Tomek".
- ``MODEL`` The machine learning model, either "Extra Trees", "Random Forest", or "Decision Tree".
- ``T_TRAIN`` Training time in seconds.
- ``T_TEST`` Testing time in seconds.
- ``PROJECT_NAME`` The name of a subject project.

    - ``FALSE_POSITIVES_PROJECT`` Number of false positives for this specific project.
    - ``FALSE_NEGATIVES_PROJECT`` Number of false negatives for this specific project.
    - ``TRUE_POSITIVES_PROJECT`` Number of true positives for this specific project.
    - ``PRECISION_PROJECT`` Precision for this specific project.
    - ``RECALL_PROJECT`` Recall for this specific project.
    - ``F1_SCORE_PROJECT`` F1 score for this specific project.

- ``FALSE_POSITIVES`` Number of false positives overall.
- ``FALSE_NEGATIVES`` Number of false negatives overall.
- ``TRUE_POSITIVES`` Number of true positives overall.
- ``PRECISION`` Precision overall.
- ``RECALL`` Recall overall.
- ``F1_SCORE`` F1 score overall.

``shap.pkl``
--------------

This is a pickle file containing two `NumPy <https://numpy.org/>`_ arrays of SHAP values. The first is for the NOD classification and the second is for OD.