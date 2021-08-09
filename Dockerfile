FROM continuumio/miniconda3

RUN mkdir -p /orgasegment
COPY . /orgasegment
RUN conda env create -f /orgasegment/conf/environment.yml

# Pull the environment name out of the environment.yml
RUN echo "source activate $(head -1 /orgasegment/conf/environment.yml | cut -d' ' -f2)" > ~/.bashrc
ENV PATH /opt/conda/envs/$(head -1 /orgasegment/conf/environment.yml | cut -d' ' -f2)/bin:$PATH