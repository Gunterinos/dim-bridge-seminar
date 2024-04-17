Dataset goes here.

The dataset directories is structured as:

```
.
├── README.md
├── animals5
│   ├── animals5.csv
│   └── images
└── gait2
    └── gait2.csv
```

where images are usually named in the order of the corresponding data items, e.g.:
```
<dataset_name>/<dataset_name>.csv
<dataset_name>/images/0.png
<dataset_name>/images/1.png
...
<dataset_name>/images/1000.png
```
But it is not neccessary, as long as the dataset stores the full URL to the images in the `image_url` column.
The dataset csv can be named anything, but it has to be the only one csv file under the <dataset_name>/ directory (grabbed by `glob('*.csv')[0]` by `app.py`)

