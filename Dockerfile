From sinzlab/pytorch:v3.8-torch1.7.0-cuda11.0-dj0.12.7

# Install essential Ubuntu packages
# and upgrade pip
RUN apt-get update &&\
    apt-get install -y build-essential \
    apt-transport-https \
    lsb-release  \
    ca-certificates \
    curl

# Update nodejs
RUN curl -sL https://deb.nodesource.com/setup_12.x | bash -
RUN apt-get install -y nodejs

# install third-party libraries
RUN python -m pip install --upgrade pip
RUN python -m pip --no-cache-dir install \
    hiplot \
    slacker\
    ax-platform \
    tensorboard \
    ptvsd \
    jupyterlab>=2 \
    xeus-python \
    streamlit \
    nb_black \
    ipdb \
    statsmodels

RUN python -m pip --no-cache-dir install \
    torch==1.7.1+cu110 \
    torchvision==0.8.2+cu110 \
    torchaudio==0.7.2 \
    -f https://download.pytorch.org/whl/torch_stable.html

# install jax
RUN python -m pip install --upgrade jax jaxlib==0.1.64+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html

# install jupyter debugger extentions
RUN jupyter labextension install @jupyterlab/debugger


# isntall captum
RUN git clone --single-branch --branch optim-wip https://github.com/mohammadbashiri/captum.git /src/captum
RUN pip install -e /src/captum

## set the name of the working directory ##################
# this is by default "notebooks" (set in sinzlab/pytorch image)
# WORKDIR /notebooks

## install the current project as a library ###############
# create a directoy in the docker environment (same name as your package)
RUN mkdir /notebooks/behavior_model_fitting

# copy the content from local package folder to the docker package folder
COPY behavior_model_fitting /notebooks/behavior_model_fitting
COPY setup.py /notebooks

# install current project
RUN python3 setup.py develop

# all the dependencies should be placed in a folder called lib
RUN mkdir /notebooks/lib
COPY lib /notebooks/lib

# install lab-specific public packages
RUN python -m pip install "git+https://github.com/sinzlab/nnfabrik.git@master"
RUN python -m pip install "git+https://github.com/mohammadbashiri/neuralpredictors.git@brain_state"

# install internal unpublished libraries
# RUN python -m pip install -e /notebooks/lib/nndichromacy
# RUN python -m pip install -e /notebooks/lib/data_port
# RUN python -m pip install -e /notebooks/lib/vivid