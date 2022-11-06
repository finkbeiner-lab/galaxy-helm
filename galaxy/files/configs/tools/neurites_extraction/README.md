# Neurite Tracing

Neurite Tracing script for measuring overlap and counting pixels between associated cellmasks and neurites.

## Prerequisites

What things you need to install the software and how to install them

```
import os
import re
import pandas as pd
import fnmatch
import numpy as np
import cv2
import argparse
import time
```

Folder Hierarchy:
All three paths (neurite masks, soma masks, and output) must be specified now.
Deprecated:
- If the folders 'neurite_masks' and 'soma_masks' do not exist in the directory of the path specified, an error will be thrown. This requirement has been lifted.

## Running the script
Example
```
python NeuriteTracing.py /path/to/neurite_masks path/to/soma_masks path/to/output --min_cell 10 --width 3 --radius 1.2
```

neurite_masks must contain binary neurite masks.
soma_masks must contain both binary and encoded soma masks.

--min_cell 10 - The minimum cell size that is worth considering. In this case, any cell that has an area less than 10, will be ignored

--width 3 - The width that will be used when drawing the filaments and cells for our QC and PROCESSED images

--radius 1.2 - The amount that we are multiplying the initial radius of each cell by. If 1 is passed into --radius, this will not change the initial radius at all.


### Output

3 separate folders created in the directory of the path provided.
- associated_neurite_csvs
- qc_images
- processed_images

## Authors

* **Stephen Cannon** - *Initial work* -
