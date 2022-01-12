# Testing entropy viscosity stabilization using deal.ii

For running the program, a docker container could be used.

**Pull the docker image**

`docker pull dealii/dealii:v9.3.0-focal`

**Start the container and mount the directory**

`docker run -i -t -v $(pwd):/home/dealii/shared/ dealii/dealii:v9.3.0-focal`

**Compile and run the program**

`cd shared/`

`cmake CMakeLists.txt`

`make run`

`python3 plot_solutions.py`
