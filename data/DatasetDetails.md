### Evaluating Language Model Agency through Negotiations

This repository containing the raw negotiation transcripts from the paper `Evaluating Language Model Agency through Negotiations` [[1]](#citation) is made available here: `https://zenodo.org/records/10254697``. The data includes both transcripts from self-play (a model plays against itself; corresponding to Section 4.1 of paper) and cross-play (a model plays against another model; Section 4.2). In total, this encompasses **2926 transcripts** (942 self-play, 1984 cross-play).

The transcripts are structured in the following way:
```sh
transcripts/
├── self_play/
│   └── {model_name}/
│       ├── processed_negotiation.csv
│       ├── negotiations.csv
│       ├── interrogations.csv
│       └── .hydra/
│           ├── config.yaml
│           ├── hydra.yaml
│           └── overrides.yaml
└── cross_play/
    └── {model_1_name}_{model_2_name}/
        ├── processed_negotiation.csv
        ├── negotiations.csv
        └── .hydra/
            ├── config.yaml
            ├── hydra.yaml
            └── overrides.yaml
```
We also include metadata with rules and game setup in the metadata folder. 
```sh
metadata/
├── agents/
│   └── anon.yaml
└── game/
    ├── generic-rental-agreement.yaml
    ├── generic_game_rules.yaml
    └── issues/
        ├── {issue_i}
        ├── ...
```

# Citation
Please cite the paper when using data. 

`[1] T.R. Davidson, V. Veselovsky, M. Josifoski, M. Peyrard, A. Bosselut, M. Kosinski, R. West (2023). 
Evaluating Language Model Agency through Negotiations.`


```bib
@article{davidson23,
  title={Evaluating Language Model Agency through Negotiations},
  author={Davidson, Tim R. and 
          Veselovsky, Veniamin and
          Josifoski, Martin and
          Peyrard, Maxim and
          Bosselut, Antoine and
          Kosinski, Michal and
          West, Robert
          },
  journal={arXiv preprint},
  year={2023}
}
```