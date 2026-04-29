# Include the base image for the docker
FROM python:3.12

# Setting working directory to /opt, rather than doing all the work in root.
# Copying the /code directory into /opt
WORKDIR /opt

COPY requirements* /opt
# Running pip install to download required packages
RUN apt-get update ##[edited]
RUN pip install -r requirements_base.txt
RUN pip install -r requirements_extra.txt
RUN pip install jupyter

COPY . /opt

# Setting the default code to run when a container is launced with this image.
CMD ["jupyter", "notebook", "--port=8899", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
