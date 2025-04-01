# Code Structure

```
├── conf/
│   ├── ASTGCN/
│   │   └── ASTGCN.conf
│   ├── general_conf/
│   │   ├── global_baselines.conf
│   │   └── pretrain.conf
│   ├── STSGCN/
│   │   └── STSGCN.conf
│   ├── STWA/
│   │   └── STWA.conf
│   └── TGCN/
│       └── TGCN.conf
├── data/
│   ├── generate_ca_data.py
│   └── README.md
├── lib/
│   ├── data_process.py
│   ├── logger.py
│   ├── metrics.py
│   ├── Params_predictor.py
│   ├── Params_pretrain.py
│   ├── predifineGraph.py
│   └── TrainInits.py
├── model/
│   ├── ASTGCN/
│   │   ├── args.py
│   │   └── ASTGCN.py
│   ├── KDMLP/
│   │   ├── args.py
│   │   └── KDMLP.py
│   ├── ST_WA/
│   │   ├── args.py
│   │   ├── attention.py
│   │   └── ST_WA.py
│   ├── STSGCN/
│   │   ├── args.py
│   │   └── STSGCN.py
│   └── TGCN/
│       ├── args.py
│       └── TGCN.py
│   ├── Model.py
│   ├── BasicTrainer.py
│   ├── Run.py
└── model_weights/
    ├── OpenCity/
    └── README.md
```
# Environment

```
pip install -r requirements.txt
```

# How to run

cd model

python3 Run.py -mode ori -model KDMLP -batch_size 128 --real_value False -device cuda:0
