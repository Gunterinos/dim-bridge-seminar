# Get started

1. Clone this repo:
```git clone git@github.com:tiga1231/dim-bridge.git```

1. Download dataset from Box as a zip file to the root of this repo:
https://vanderbilt.box.com/s/kkhbiql67pc4n2wqp11o2wpxy1aim7a0

1. unzip
```unzip dimbridge-dataset.zip```
A new dataset/ directory will show up

1. install python dependencies
```pip install requirements.txt```
[TODO requirements.txt]

1. Start the server on port 9001
```python app.py```

1. Open the DimBridge observable notebook:
https://observablehq.com/d/bc84ced61d90006e

1. If this predicate server is not localhost...
    1. If the remote server machine does not have a public ip but you have ssh access, you can forward port 9001 to your local machine which you run obserable notebook on:

