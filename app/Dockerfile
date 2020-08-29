FROM centos:7

RUN set -x

# setup
RUN yum clean all && yum -y update
RUN yum install -y git vim wget unzip make swig gcc gcc-c++ \
                   cmake boost boost-devel bzip2 bzip2-devel

# install python3.6
RUN yum install -y https://centos7.iuscommunity.org/ius-release.rpm
RUN yum install -y python36u python36u-libs python36u-devel python36u-pip

# locale
RUN localedef -f UTF-8 -i ja_JP ja_JP
ENV LC_ALL ja_JP.utf8

# mkdir
RUN mkdir /home/data
WORKDIR /home

# pip install
RUN pip3.6 install torch==1.0.0
RUN pip3.6 install tqdm numpy flask flask-bootstrap emoji regex neologdn gunicorn

# git clone
WORKDIR /home
RUN git clone https://db07bc1dc6b5ced230d48b4dc0bf4be3e6cff2f2:x-oauth-basic@github.com/youichiro/transformer-copy

# run app
WORKDIR /home/transformer-copy/app
EXPOSE 5003
CMD [ "gunicorn", "-b", "0.0.0.0:5003", "app:app" ]

