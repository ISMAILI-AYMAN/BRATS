# BRATS - Segmentation de Tumeurs Cérébrales par Deep Learning

Ce projet propose une application web (Flask) permettant de segmenter automatiquement des tumeurs cérébrales à partir d’IRM, en utilisant un modèle 3D pré-entraîné sur le dataset BRATS.

## 🚀 Fonctionnalités

- **Upload** d’images médicales au format `.npy`, `.nii`, `.nii.gz` ou `.dcm`
- **Prétraitement** automatique (redimensionnement, normalisation, gestion des canaux)
- **Prédiction** de la segmentation tumorale via un modèle Keras 3D
- **Visualisation** de la coupe IRM et du masque de prédiction
- **Téléchargement** des résultats

## 📁 Structure du projet

```
.
├── app.py                  # Application Flask principale
├── models/
│   └── brats_3d.hdf5       # Modèle pré-entraîné
├── static/
│   └── uploaded/           # Images uploadées et résultats
├── templates/
│   ├── index.html
│   └── result.html
├── requirements.txt
└── README.md
```

## ⚙️ Installation

1. **Clone le dépôt :**
   ```bash
   git clone https://github.com/ISMAILI-AYMAN/BRATS.git
   cd BRATS
   ```

2. **Installe les dépendances :**
   ```bash
   pip install -r requirements.txt
   ```

3. **Place le modèle pré-entraîné** dans `models/brats_3d.hdf5` (voir section suivante).

## 📦 Modèle

Le modèle Keras (`brats_3d.hdf5`) doit être placé dans le dossier `models/`.  
Si tu n’as pas ce fichier, entraîne-le ou demande-le à l’auteur du projet.

## 🖥️ Utilisation

1. **Lance l’application Flask :**
   ```bash
   python app.py
   ```
2. **Ouvre ton navigateur** à l’adresse [http://127.0.0.1:5000](http://127.0.0.1:5000)
3. **Upload** une image médicale et visualise la segmentation.

## 📝 Exemple d’utilisation

- Upload d’un fichier `.nii.gz` ou `.dcm`
- Sélection automatique ou manuelle de la coupe à visualiser
- Affichage côte à côte de l’image IRM et du masque de prédiction

## 🧪 Tests

Pour tester l’application, utilise des fichiers d’exemple du dataset BRATS ou des images médicales anonymisées.

## 🤝 Contribution

Les contributions sont les bienvenues !  
Merci de proposer une *issue* ou une *pull request*.

## 📄 Licence

Projet open source sous licence MIT.

---

**Contact** : [ISMAILI-AYMAN](https://github.com/ISMAILI-AYMAN)