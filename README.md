# Effective Testing for Machine Learning Projects

[![CI](https://github.com/edublancas/ml-testing/workflows/CI/badge.svg)](https://github.com/edublancas/ml-testing/workflows/CI/badge.svg)


Code for PyData Global 2021 Presentation by [@edublancas](https://twitter.com/intent/follow?screen_name=edublancas). [Slides available here](https://blancas.io/talks/pydata-21.pdf).

The project is developed using [Ploomber](https://github.com/ploomber/ploomber); check it out! :)

If you have questions, [ping me on Slack](https://ploomber.io/community/).

## Blog post series

1. [Part I](https://ploomber.io/blog/ml-testing-i/)
2. [Part II](https://ploomber.io/blog/ml-testing-ii/)
3. [Part III](https://ploomber.io/blog/ml-testing-iii/)

*Follow [@ploomber](https://twitter.com/intent/follow?screen_name=ploomber) on Twitter, or subscribe to our [newsletter](https://www.getrevue.co/profile/ploomber) for more amazing content!*

## Organization

The talk describes five stages of testing, from the most basic one to the most robust. The idea is to make progress and add more robust tests continuously. You can navigate through the branches of this repository to see how each time, it becomes more robust as we add more tests and modularize the code. Here are the links for each level:

1. [Smoke testing (1-smoke-testing)](https://github.com/edublancas/ml-testing/tree/1-smoke-testing)
2. [Integration and unit testing (2-integration-and-unit)](https://github.com/edublancas/ml-testing/tree/2-integration-and-unit)
3. [Variable distributions and inference pipeline (3-distribution-and-inference)](https://github.com/edublancas/ml-testing/tree/3-distribution-and-inference)
4. [Training-serving skew (4-train-serve-skew)](https://github.com/edublancas/ml-testing/tree/4-train-serve-skew)
5. [Model quality (5-model-quality)](https://github.com/edublancas/ml-testing/tree/5-model-quality)

Tests are run automatically on each push using GitHub Actions; you can see the configuration file at [.github/workflows/ci.yml](.github/workflows/ci.yml)

## Setup

```sh
# get the code
git clone https://github.com/edublancas/ml-testing

# move to one of the branches
git checkout branch-name

# example
git checkout 1-smoke-testing

# install dependencies
# conda
conda env create -f environment.yml
# pip
pip install -r requirements.txt

# build the pipeline
ploomber build

# run unit tests (added on level 2)
pytest
```

## Resources

* [Slides](https://blancas.io/talks/pydata-21.pdf).
* Recording will be available here after the PyData Global 2021 conference.
* [Based on this Kaggle notebook](https://www.kaggle.com/roshansharma/heart-diseases-analysis).

