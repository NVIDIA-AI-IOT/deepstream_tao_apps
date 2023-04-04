FROM nvcr.io/nvidia/deepstream-l4t:6.2-triton as libnvds_infercustomparser_builder

# Install Custom BBox Parser
# Install Git LFS
WORKDIR /workspace
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install git-lfs
RUN git lfs install

# Install Custom BBox Parser
WORKDIR /opt/nvidia/deepstream/deepstream/sources/deepstream_tao_apps
COPY . .
ENV CUDA_VER=11.4