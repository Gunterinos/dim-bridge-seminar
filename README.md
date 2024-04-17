## Breaking Changes

- The latest front end (https://observablehq.com/@tiga1231/dim-bridge) expect images of data points stored in a `image_url` column in the data file (csv), e.g., http://host:port/path/to/image.png

## Supported data format

### CSV file
TODO

## Using DimBridge with Docker
TODO

## Getting started

1. Clone this repo:
```git clone git@github.com:tiga1231/dim-bridge.git```

1. ```cd dim-bridge```

1. Download dataset from Box as a zip file to the root of this repo:
https://vanderbilt.box.com/s/kkhbiql67pc4n2wqp11o2wpxy1aim7a0
TODO update demo datasets

1. Unzip: 
```unzip dimbridge-dataset.zip```
A new dataset/ directory will show up

1. Use your favorite virtualenv*, install python dependencies:
    (note from Jen: I use `python3 -m venv venv` and `source venv/bin/activate`)
```pip install -r requirements.txt```

1. Start the server on port 9001: 
```python app.py```

1. Open the DimBridge observable notebook:
WebGL version:
https://observablehq.com/@tiga1231/dim-bridge

Original version:
https://observablehq.com/d/bc84ced61d90006e

Jen's version:
https://observablehq.com/d/5c05aecd48c4eed5

Jen's backup version
https://observablehq.com/d/1e642c90d6c45083

1. If this predicate server is not localhost...
    1. If the remote server machine does not have a public ip but you have ssh access, you can forward port 9001 of the remote server to the same port on the local machine that you run obserable notebook on:
```ssh -L 9001:localhost:9001 <username>@<your-machine-domain-or-ip>```

