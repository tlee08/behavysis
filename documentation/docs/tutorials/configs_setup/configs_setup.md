# The Configs File

The config file for an experiment stores all the parameters for how the experiment was recorded (e.g., the frames per second of the raw video, the experiment duration, etc.), and the parameters for how we want to process the data (e.g., the intended frames per second to format the video to, the DLC model to use to analyse, the likeliness pcutoff to interpolate points, etc.)

An example of a config file is shown below. NOTE: DO NOT use the config file shown here as a template. This example is shown only to explain the key parameters in the config file.
Instead, you can use one of these config templates:

- [configs for Open Field](configs_OF.md)

```json title="example configs file with descriptions. DO NOT USE."
--8<-- "docs/tutorials/configs_setup/configs_setup.json"
```

Update the config files of all experiments in your project by copying a template to another file and updating all experiment configs inside your Jupyter notebook with the commands found [here](configs_update.md).
