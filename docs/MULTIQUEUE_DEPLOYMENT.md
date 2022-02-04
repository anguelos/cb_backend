1. Create Queue Makefiles
```bash
export PYTHONPATH="$HOME/cb_binarize/"
PATH="$PATH:$PYTHONPATH/bin"

CB_ROOT="/data/storage/cb_root"
PHOCNET="${CB_ROOT}"/models/phocresnet_0x0_v2.pt
BINNET="${CB_ROOT}"/models/srunet.pt
BOXNET="${CB_ROOT}"/models/box_iou.pt
IDX_ROOT=/data/storage/cb_root/indexes

QUEUE_ROOT=/data/storage/cb_root/cache/queue_cb91/public/

cb_create_chronicle_makefile  -db_root "${QUEUE_ROOT}" -archive_location $IDX_ROOT -chronicle_paths "${QUEUE_ROOT}"/chronicle/*/* -aux_proposals_postfix .words.json -binnet_path "${BINNET}" -boxnet_path "${BOXNET}" -phocnet_path "${PHOCNET}"        
```



```bash
URL=http://test1.portafontium.eu/chronicle/soap-so/00012-mesto-brezova-1965-1974
```


```bash
PYTHONPATH="./" ./bin/cb_cliqbs -query hraz -db_root /data/archiv/public/
```

```bash
PORT=8080
IDX_DIR=/data/storage/cb_root/indexes/
NET=/data/storage/cb_root/models/phocresnet_0x0_v2.pt   
PYTHONPATH="./" ./bin/cb_service -indexed_documents  $IDX_DIR/*pickle -embeding_net $NET -port $PORT -document_root '/data/archiv/public/'

```
