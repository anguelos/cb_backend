Train components
```bash
PYTHONPATH="./:./thirdparty/iunets" ./bin/cb_train_component_classifier -binary_images ./data/annotated/*/*bin.png -annotations ./data/annotated/*/*.gt.json
```

create proposals
```bash
PYTHONPATH="./:./thirdparty/iunets" ./bin/cb_propose_words -prob_images ./data/fake_db/*/*bin.png -target_postfix .words.json
```

evaluate proposals
```bash
PYTHONPATH="./:./thirdparty/iunets" ./bin/cb_evaluate_proposals.py -proposals ./data/annotated/blovice/*words.json -gt ./data/annotated/blovice/*gt.json -iou_threshold=.5
```


RND score NMS Recall Rate
-------------------------
THR          | RLSA        | NMS IOU |PROPOSALS |R@50 |R@75 |R@90
-------------|-------------|---------|----------|--------|-------|-----|
128          |0,4,8,16     |75%      | 1064784  | 80.5 % | 43.9 %| 6.4%|
128,224      |0,4,8,16     |90%      | 1121611  | 81.0 % | 45.9 %| 6.9%|
32,64,128,224|0,4,8,16     |75%      | 1621626  | 81.0 % | 43.1 %| 5.6%|
128          |0,2,4,6,12,18,24| 75%| 683156     | 80.4 % | 43.4% | 6.2%|

