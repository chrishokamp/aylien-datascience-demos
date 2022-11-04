# Aylien Datascience Demos

This is a multi-page streamlit app containing demo apps built on top of the Aylien NewsAPI.

This project is intended for data scientists using the Aylien NewsAPI, but also for anyone interested in

For each of these demos, we provide a sample dataset that will allow them to run without a NewsAPI key. 

Also check out our open source bootstrapper for projects and demos: [datascience-project-quickstarter](https://github.com/AYLIEN/datascience-project-quickstarter)

analysis of news feed data. 

We also include some patterns that we re-use and modify regularly, such as a simple interface to the 
newsAPI, and caching responses for demos. 

If you have an idea, feel free to submit a PR, or open an issue for discussion. 

### Models

Some demos show workflows that are possible with off-the-shelf models. But note that the internal 
models at Aylien are probably better than these ;-) 


#### New environment

Run `conda create -n <env-name> python=3.8` if you're using Anaconda, alternatively `python3.8 -m venv <env-path>`

Activate: <br>
Run `conda activate <env-name>` or `source <env-path>/bin/activate`

#### Install library
Run `make dev`

### New Demo

Within in a project, you can initialize a new demo as follows: <br>
`quickstart-streamlit --project . --name super-cool-demo`

or just run `quickstart-streamlit` and follow the instructions.

A demo directory with the given name and running streamlit skeleton will be created in [/demos](demos).

You can checkout the README generated in the new demo directory for further guidance.


