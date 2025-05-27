# Segmentation Automatisée des Tumeurs Cérébrales à partir d'IRM

> **Encadrants :** M. El Ossmani & M. Berrada

Un projet de segmentation automatique des tumeurs cérébrales utilisant l'apprentissage profond et l'analyse radiomique sur les images IRM.

## 📋 Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Fonctionnalités](#fonctionnalités)
- [Architecture](#architecture)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure du projet](#structure-du-projet)
- [Données](#données)
- [Modèles](#modèles)
- [Métriques d'évaluation](#métriques-dévaluation)
- [Dashboard](#dashboard)
- [Déploiement](#déploiement)
- [Contribution](#contribution)
- [Licence](#licence)

## 🎯 Vue d'ensemble

Ce projet implémente une solution complète de segmentation automatisée des tumeurs cérébrales à partir d'images IRM. Il combine plusieurs approches d'apprentissage profond avec des techniques de post-traitement et d'analyse radiomique pour fournir des segmentations précises et des insights cliniques.

### Objectifs principaux
- Segmentation automatique des tumeurs cérébrales
- Extraction de caractéristiques radiomiques
- Analyse statistique et corrélation avec le grade tumoral
- Interface utilisateur pour les professionnels de santé

## ✨ Fonctionnalités

- **Segmentation multi-modèles** : U-Net, nnU-Net, Mask R-CNN
- **Préprocessing avancé** : Correction N4, normalisation Z-score
- **Augmentation de données** : Rotation, zoom, ajustement de contraste
- **Post-traitement** : Watershed, Conditional Random Fields (CRF)
- **Analyse radiomique** : Extraction de features avec PyRadiomics
- **Dashboard interactif** : Interface web pour visualisation et analyse
- **Évaluation complète** : Dice, IoU, Hausdorff Distance

## 🏗️ Architecture

```
├── data/
│   ├── raw/                 # Données brutes (DICOM)
│   ├── processed/           # Données préprocessées (NIfTI)
│   └── augmented/           # Données augmentées
├── models/
│   ├── unet/               # Implémentation U-Net
│   ├── nnunet/             # Configuration nnU-Net
│   └── mask_rcnn/          # Implémentation Mask R-CNN
├── src/
│   ├── preprocessing/       # Scripts de préprocessing
│   ├── augmentation/        # Augmentation de données
│   ├── training/           # Entraînement des modèles
│   ├── evaluation/         # Métriques et évaluation
│   ├── postprocessing/     # Post-traitement
│   └── radiomics/          # Extraction de features
├── dashboard/              # Interface utilisateur
├── deployment/             # Scripts de déploiement
├── notebooks/              # Jupyter notebooks d'analyse
└── tests/                  # Tests unitaires
```

## 🚀 Installation

### Prérequis
- Python 3.8+
- CUDA 11.0+ (pour GPU)
- 16GB RAM minimum
- 50GB d'espace disque

### Installation des dépendances

```bash
# Cloner le repository
git clone https://github.com/votre-username/brain-tumor-segmentation.git
cd brain-tumor-segmentation

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt

# Installation spécifique pour nnU-Net
pip install nnunet

# Pour le développement
pip install -r requirements-dev.txt
```

### Configuration

```bash
# Configurer les variables d'environnement
cp .env.example .env
# Éditer .env avec vos paramètres

# Initialiser la base de données (si applicable)
python src/setup_database.py
```

## 📖 Utilisation

### 1. Préparation des données

```bash
# Télécharger les datasets
python src/data_collection/download_datasets.py

# Préprocessing complet
python src/preprocessing/main_preprocessing.py --input data/raw --output data/processed

# Augmentation des données
python src/augmentation/augment_data.py --input data/processed --output data/augmented
```

### 2. Entraînement des modèles

```bash
# U-Net
python src/training/train_unet.py --config configs/unet_config.yaml

# nnU-Net
nnUNet_plan_and_preprocess -t 001 --verify_dataset_integrity
nnUNet_train 3d_fullres nnUNetTrainerV2 001 0

# Mask R-CNN
python src/training/train_mask_rcnn.py --config configs/mask_rcnn_config.yaml
```

### 3. Évaluation

```bash
# Évaluation complète
python src/evaluation/evaluate_models.py --models unet,nnunet,mask_rcnn

# Comparaison inter-modèles
python src/evaluation/compare_models.py --results_dir results/
```

### 4. Extraction de features radiomiques

```bash
# Extraction avec PyRadiomics
python src/radiomics/extract_features.py --input data/processed --masks results/segmentations/

# Analyse statistique
python src/radiomics/statistical_analysis.py --features results/radiomics_features.csv
```

### 5. Lancement du dashboard

```bash
# Streamlit
streamlit run dashboard/streamlit_app.py

# Flask
python dashboard/flask_app.py

# React (développement)
cd dashboard/react-frontend
npm start
```

## 📁 Structure du projet

### Modules principaux

#### `src/preprocessing/`
- `dicom_to_nifti.py` : Conversion DICOM → NIfTI
- `n4_correction.py` : Correction du biais N4
- `normalization.py` : Normalisation Z-score
- `quality_check.py` : Vérification qualité des images

#### `src/models/`
- `unet_model.py` : Architecture U-Net
- `nnunet_wrapper.py` : Interface nnU-Net
- `mask_rcnn_model.py` : Implémentation Mask R-CNN

#### `src/postprocessing/`
- `watershed.py` : Segmentation Watershed
- `crf_refinement.py` : Raffinement CRF
- `morphological_ops.py` : Opérations morphologiques

## 📊 Données

### Sources de données
- **TCGA-GBM** : Glioblastomes de haut grade
- **TCGA-LGG** : Gliomes de bas grade
- **Kaggle Brain Tumor Dataset** : Dataset complémentaire

### Format des données
```
patient_id/
├── t1.nii.gz          # Image T1
├── t1ce.nii.gz        # Image T1 avec contraste
├── t2.nii.gz          # Image T2
├── flair.nii.gz       # Image FLAIR
└── seg.nii.gz         # Masque de segmentation
```

### Classes de segmentation
- **Classe 0** : Arrière-plan
- **Classe 1** : Nécrose et cavité kystique
- **Classe 2** : Œdème péritumoral
- **Classe 4** : Tumeur rehaussée

## 🤖 Modèles

### U-Net
```python
# Configuration exemple
model_config = {
    'input_channels': 4,
    'num_classes': 4,
    'base_filters': 32,
    'depth': 5,
    'dropout': 0.1
}
```

### nnU-Net
- Configuration automatique basée sur les données
- Augmentation adaptive
- Optimisation des hyperparamètres

### Mask R-CNN
```python
# Configuration exemple
config = {
    'backbone': 'resnet50',
    'num_classes': 4,
    'roi_pool_size': 7,
    'anchor_scales': [32, 64, 128, 256, 512]
}
```

## 📈 Métriques d'évaluation

### Métriques de segmentation
- **Dice Coefficient** : Similarité des régions
- **IoU (Intersection over Union)** : Chevauchement des masques
- **Hausdorff Distance** : Distance maximale entre contours
- **95% Hausdorff Distance** : Version robuste aux outliers

### Métriques radiomiques
- **Features de forme** : Volume, surface, sphéricité
- **Features de premier ordre** : Moyenne, variance, asymétrie
- **Features textuelles** : GLCM, GLRLM, GLSZM

## 🖥️ Dashboard

### Fonctionnalités du dashboard
- **Visualisation 3D** des segmentations
- **Comparaison** entre modèles
- **Ajustement interactif** des paramètres
- **Upload** de nouvelles images
- **Export** des résultats

### Technologies utilisées
- **Streamlit** : Interface principale
- **Plotly** : Visualisations interactives
- **ITK-Snap** : Visualisation médicale
- **React** : Interface avancée (optionnel)

## 🚀 Déploiement

### Docker

```dockerfile
# Exemple Dockerfile
FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "dashboard/streamlit_app.py"]
```

### Cloud (AWS/GCP)

```bash
# Déploiement sur AWS
python deployment/deploy_aws.py

# Déploiement sur GCP
python deployment/deploy_gcp.py
```

## 🧪 Tests

```bash
# Tests unitaires
python -m pytest tests/unit/

# Tests d'intégration
python -m pytest tests/integration/

# Tests de performance
python tests/performance/benchmark_models.py
```

## 📝 Configuration

### Fichiers de configuration principaux

#### `configs/unet_config.yaml`
```yaml
model:
  name: "unet"
  input_channels: 4
  num_classes: 4
  base_filters: 32

training:
  batch_size: 4
  learning_rate: 0.001
  epochs: 100
  optimizer: "adam"

data:
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
```

## 🤝 Contribution

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Commit vos changements (`git commit -am 'Ajout nouvelle fonctionnalité'`)
4. Push vers la branche (`git push origin feature/nouvelle-fonctionnalite`)
5. Créer une Pull Request

### Standards de code
- **PEP 8** pour Python
- **Type hints** requis
- **Docstrings** pour toutes les fonctions
- **Tests unitaires** pour les nouvelles fonctionnalités

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 📞 Contact

- **Encadrants** : M. El Ossmani & M. Berrada
- **Email** : [votre-email@exemple.com]
- **Issues** : [Lien vers les issues GitHub]

## 🙏 Remerciements

- TCGA pour les datasets
- Communauté nnU-Net
- PyRadiomics team
- Contributeurs open source

## 📚 Références

1. Ronneberger, O., et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation"
2. Isensee, F., et al. "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation"
3. He, K., et al. "Mask R-CNN"
4. van Griethuysen, J.J.M., et al. "Computational Radiomics System to Decode the Radiographic Phenotype"

---

**Version :** 1.0.0  
**Dernière mise à jour :** [Date]