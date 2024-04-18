# With Docker

## Installing Docker

Follow the instructions from the [Docker website](https://www.docker.com/products/docker-desktop/).

## Installing ba_pipeline

The `ba_pipeline` program will both be stored in, and run by Docker.
To download the program into Docker, follow the instructions below:

**Step 1:**
Connect to the RDS

**Step 2:**
Open a terminal or command line, which should look like one of the windows in the figure below.

<p align="center">
    <img src="../../figures/terminals.png" alt="terminals" title="terminals" style="width:70%">
</p>

**Step 3:**
Navigate to the following folder in the terminal with the following command. An image is shown to visualise the location of this folder.

For Mac/Linux

```zsh
cd /Volumes/PRJ-BowenLab/TimLee/releases/ba_pipeline
```

For Windows

```zsh
cd Z:\PRJ-BowenLab\TimLee\releases\ba_pipeline
```

<p align="center">
    <img src="../../figures/mac_docker_location.png" alt="mac_docker_location" title="mac_docker_location" style="width:60%">
</p>

**Step 4:**
Install the `ba_pipeline` program into the system by running the following terminal command. This may take around 10 minutes.

```zsh
docker load -i ba_pipeline.tar.gz
```

**Step 5:**
To verify that the program has been correctly installed, you can open the Docker Desktop application and select "Images" on the left hand pane. There should be a row named `ba_pipeline` with the tag, `core`. An image of this is shown below.

<p align="center">
    <img src="../../figures/docker_window.png" alt="docker_window" title="docker_window" style="width:70%">
</p>
