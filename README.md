# DimBridge

## New Jupyter Lab Widget
See a Jupyter Lab version of this work [[here]](https://github.com/tiga1231/dimbridge-jupyter)

## Breaking Changes

- The latest front end (https://observablehq.com/@tiga1231/dim-bridge) expect images of data points stored in a `image_url` column in the data file (csv), e.g., http://host:port/path/to/image.png

## Getting started

1. Clone this repo:
```git clone git@github.com:tiga1231/dim-bridge.git```

1. ```cd dim-bridge```

~~1.Download dataset from Box as a zip file to the root of this repo:~~
~~https://vanderbilt.box.com/s/kkhbiql67pc4n2wqp11o2wpxy1aim7a0~~
~~1. Unzip:~~
~~```unzip dimbridge-dataset.zip```~~
~~A new dataset/ directory will show up~~

~~1. Download images of animal5 dataset from Box, unzip it, and place the images folder to the dataset/animals5/images:~~
~~https://vanderbilt.app.box.com/s/jmmdgqsiqlyeacfv7453v1hu6ql2brrh ~~
1. Download images of animals https://drive.google.com/drive/folders/1x1Ptvpoay4YsM6IrtuDr11iYtkrv8nzI
unzip, and copy it to dataset/


1. Use your favorite virtualenv*, install python dependencies:
    (note from Jen: I use `python3 -m venv venv` and `source venv/bin/activate`)
```pip install -r requirements.txt```

1. Start the server on port 9001: 
```python app.py```

1. Open the DimBridge observable notebook:

    - WebGL version:
    https://observablehq.com/@tiga1231/dim-bridge

    - Original version:
    https://observablehq.com/d/bc84ced61d90006e

    - Jen and Vanessa's version accounting for Scaled Data:
    https://observablehq.com/@jrogerthat/dim_jen_version

1. If this predicate server is not localhost...
    1. If the remote server machine does not have a public ip but you have ssh access, you can forward port 9001 of the remote server to the same port on the local machine that you run obserable notebook on:
```ssh -L 9001:localhost:9001 <username>@<your-machine-domain-or-ip>```


## Using DimBridge with Docker
TODO


## Supported data format
### CSV file
TODO

