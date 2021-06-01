### Lanching the demo servers

#### Serve test-set indexed by phocnet
```bash
PORT=8080
IDX_DIR=/data/storage/new_root/idx_phocnet/
NET=/data/storage/new_root/models/phocnet_0x0.pt   
PYTHONPATH="./" ./bin/cb_service -indexed_documents  $IDX_DIR/*pickle -embeding_net $NET -port $PORT -document_root '/data/archiv/public/'

```

#### Serve test-set indexed by phocresnet
```bash
PORT=8081
IDX_DIR=/data/storage/new_root/idx_demo_simple_resnet/
NET=/data/storage/new_root/models/phocresnet_0x0.pt   
PYTHONPATH="./" ./bin/cb_service -indexed_documents  $IDX_DIR/*pickle -embeding_net $NET -port $PORT -document_root '/data/archiv/public/'

```

#### Serve train and test-set indexed by phocresnet
```bash
PORT=8082
IDX_DIR=/data/storage/new_root/fake_db_idx/
NET=/data/storage/new_root/models/phocnet_0x0.pt   
PYTHONPATH="./" ./bin/cb_service -indexed_documents  $IDX_DIR/*pickle -embeding_net $NET -port $PORT -document_root '/data/archiv/public/'

```

#### Serve test-set indexed by phocresnet
```bash
PORT=8083
IDX_DIR=/data/storage/new_root/fake_db_test_idx/
NET=/data/storage/new_root/models/phocnet_0x0.pt   
PYTHONPATH="./" python3  ./bin/cb_service -indexed_documents  $IDX_DIR/*pickle -embeding_net $NET -port $PORT -document_root '/data/archiv/public/'

```