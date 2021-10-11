```bash
DBROOT=/data/storage/new_root/queue_cb91/public/
PHOCNET=./models/phocnet_0x0.pt
BINNET=./models/srunet.pt
BOXNET=./models/box_iou.pt
IDX_ROOT=./data/fake_db_idx
PYTHONPATH="./" bin/cb_create_chronicle_makefile -db_root $DB_ROOT  -archive_location $IDX_ROOT -chronicle_paths $DB_ROOT/chronicle/*/* -binnet_path $BINNET -boxnet_path $BOXNET -phocnet_path $PHOCNET   
```
