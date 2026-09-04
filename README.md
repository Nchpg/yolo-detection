# Traffic Vision — détection de véhicules et plaques

Détection de 7 classes sur le dataset Kaggle
[Traffic vehicles Object Detection](https://www.kaggle.com/datasets/saumyapatel/traffic-vehicles-object-detection) :
`Car`, `Number Plate`, `Blur Number Plate`, `Two Wheeler`, `Auto`, `Bus`, `Truck`.

Entraînement sur Colab, puis une page statique où l'on dépose une vidéo et où
l'inférence tourne dans le navigateur. Aucune vidéo ne quitte la machine.

## Parcours

```bash
uv sync
uv run python src/extract_archive.py     # archive.zip -> data/raw
uv run python src/prepare_dataset.py     # data.yaml + contrôle des annotations
# -> notebooks/train_colab.ipynb sur Colab (GPU T4, ~35 min)
# -> best.onnx dans web/models/, best.pt à la racine
uv run python web/serve.py               # http://localhost:8000
```

## Modèle en place

`yolo11s`, 100 époques à 640 px. Mesuré sur les 184 images de validation :

| classe | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Car | 0.906 | 0.920 | 0.948 | 0.783 |
| Two Wheeler | 0.819 | 0.870 | 0.898 | 0.639 |
| Number Plate | 0.744 | 0.837 | 0.886 | 0.547 |
| Bus | 0.691 | 0.818 | 0.844 | 0.643 |
| Truck | 0.794 | 0.752 | 0.824 | 0.623 |
| Auto | 0.772 | 0.691 | 0.735 | 0.388 |
| Blur Number Plate | 0.622 | 0.776 | 0.723 | 0.383 |
| **global** | | | **0.837** | **0.572** |

```bash
uv run python src/evaluate.py --split val    # refaire la mesure
```

## Données

L'archive fournit déjà le découpage. Le dossier `test` n'a pas d'annotations : ce sont
267 images brutes et 18 vidéos, traitées comme jeu de démonstration et rangées dans
`data/demo/`.

| split | images | boîtes |
|---|---|---|
| train | 732 | 9 153 |
| val | 184 | 1 980 |

`prepare_dataset.py` apparie chaque image avec son `.txt`, valide les annotations
(nombre de champs, id de classe, bornes 0-1, boîtes dégénérées) et journalise les
anomalies dans `data/dataset/label_issues.log`.

## Le front

```bash
uv run python web/serve.py
```

`web/models/best.onnx` est chargé automatiquement au démarrage. Déposez une vidéo,
cliquez sur *Lancer*. Deux vidéos du dataset sont fournies dans `web/demo/`.

Seuils de confiance et de NMS réglables en direct, traitement d'une image sur N,
activation par classe avec compteurs, latence et cadence en continu, capture PNG.

Le backend WebGPU est essayé en premier, avec repli sur WASM. `serve.py` envoie les
en-têtes `Cross-Origin-Opener-Policy` et `Cross-Origin-Embedder-Policy: credentialless`,
ce qui débloque le WASM multithread. Un simple `python -m http.server` marche aussi,
mais en mono-thread.

### Performance

Le modèle fait 38 Mo et tourne en ~1,2 s par image en WASM sur CPU. Sur WebGPU,
comptez 30 à 60 ms. Si ça saccade, le curseur *1 image sur N* du panneau 03 est là
pour ça.

### Déploiement

`web/` est un site statique sans build. Sur Vercel, ajoutez les en-têtes d'isolation
dans `vercel.json` pour garder le multithread :

```json
{ "headers": [{ "source": "/(.*)", "headers": [
  { "key": "Cross-Origin-Opener-Policy", "value": "same-origin" },
  { "key": "Cross-Origin-Embedder-Policy", "value": "credentialless" }
]}]}
```

Chaque visiteur télécharge les 38 Mo du modèle avant la première détection.

## Structure

```
best.pt                       poids PyTorch (évaluation, ré-export)
notebooks/train_colab.ipynb   entraînement GPU + export ONNX
src/
├── config.py                 classes, chemins
├── extract_archive.py        archive.zip -> data/raw
├── prepare_dataset.py        data.yaml + validation des labels
├── evaluate.py               mAP par classe sur un split
└── export_onnx.py            best.pt -> web/models/best.onnx
web/
├── index.html, style.css
├── detector.js               letterbox, décodage, NMS
├── app.js                    interface
├── serve.py                  serveur avec isolation cross-origin
├── models/best.onnx
└── demo/                     vidéos d'exemple
```

## Notes

- `override-dependencies` dans `pyproject.toml` neutralise `opencv-python` (build GUI,
  qui réclame `libxcb`/`libGL`) au profit de `opencv-python-headless`, qui fournit le
  même module `cv2`. Nécessaire sur NixOS et dans les conteneurs sans X11.
- Torch est installé en version CPU. Pour une machine CUDA, remplacez l'index
  `pytorch-cpu` par `https://download.pytorch.org/whl/cu124`.
- YOLO11 d'Ultralytics est sous AGPL-3.0.
