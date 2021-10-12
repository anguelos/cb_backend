## Create a Queue to be run on a single computation node

### 1. Create a Queue and assign it the Chudenice archive (a few hours)
```bash
WORKING_DIR=/data/storage/new_root
QUEUE_NAME=queue_cb92
QUEUE_ARCHIVE=soap-ch
QUEUE_SRC=/data/archiv/public/chronicle/"$QUEUE_ARCHIVE"
QUEUE_DST="$WORKING_DIR"/"$QUEUE_NAME"/public/chronicle/"$QUEUE_ARCHIVE"
IDX_ROOT="$WORKING_DIR"/data/"$QUEUE_NAME"_idx

mkdir -p "$WORKING_DIR"/"$QUEUE_NAME"/public/chronicle/
mkdir -p "$IDX_ROOT"

cp -Rp "$QUEUE_SRC" "$QUEUE_DST"  # this can take long everything else is instant
```

### 2. Create/Update Makefiles for Queue (few seconds)
```bash
WORKING_DIR=/data/storage/new_root
QUEUE_NAME=queue_cb92
DB_ROOT="$WORKING_DIR"/"$QUEUE_NAME"/public/
PHOCNET="$WORKING_DIR"/models/phocnet_0x0.pt
BINNET="$WORKING_DIR"/models/srunet.pt
BOXNET="$WORKING_DIR"/models/box_iou.pt
IDX_ROOT="$WORKING_DIR"/data/"$QUEUE_NAME"_idx
PYTHONPATH="./" bin/cb_create_chronicle_makefile -db_root $DB_ROOT  -archive_location $IDX_ROOT -chronicle_paths $DB_ROOT/chronicle/*/* -binnet_path $BINNET -boxnet_path $BOXNET -phocnet_path $PHOCNET
```

### 3. Run/Resume one by one all items in the queue (weeks)
```bash
WORKING_DIR=/data/storage/new_root
QUEUE_NAME=queue_cb92
IDX_ROOT="$WORKING_DIR"/data/"$QUEUE_NAME"_idx

make -f "$IDX_ROOT"/Makefile
```
