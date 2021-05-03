### Generate Makefile

The following command assumes 100% availabillity of a Tesla T4 (16GB of RAM)

#### Demo Documents:
```bash
DB_ROOT=./data/fake_db_overlaid/
PHOCNET=./models/phocnet_0x0.pt
BINNET=./models/srunet.pt
BOXNET=./models/box_iou.pt
IDX_ROOT=./data/fake_db_idx
PYTHONPATH="./" bin/cb_create_chronicle_makefile -db_root $DB_ROOT  -archive_location $IDX_ROOT -chronicle_paths $DB_ROOT/chronicle/*/* -binnet_path $BINNET -boxnet_path $BOXNET -phocnet_path $PHOCNET   
```

#### Performance Evaluation:

Indexing the train and test set chronicles
```bash
DB_ROOT=./data/fake_db_overlaid/
PHOCNET=./models/phocnet_0x0.pt
BINNET=./models/srunet.pt
BOXNET=./models/box_iou.pt
IDX_ROOT=./data/fake_db_idx
PYTHONPATH="./" bin/cb_create_chronicle_makefile -db_root $DB_ROOT  -archive_location $IDX_ROOT -chronicle_paths $DB_ROOT/chronicle/*/* -aux_proposals_postfix .words.json    
```

- If a smaller or a different GPU is available, change the -max_device_mp to __1/(GPU MegaPixels)__.
- If more than on indexes can be built for the same chronicle, each one should have its own Makefile, set the 
  __-make\_filename__ parameter apropriatly. 



### Make all documents

Assuming IDX_ROOT was set as in previous block
```bash
make -f $IDX_ROOT/Makefile  
```
