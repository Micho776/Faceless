# Cheatsheet - Paskee (Gestionnaire de mots de passe)

## Vue d'ensemble du projet

Paskee est un gestionnaire de mots de passe en ligne de commande, inspiré de KeePass. Il permet de stocker, rechercher, modifier et supprimer des identifiants pour différentes plateformes.

---

## Architecture du projet

```txt
projet_final/
├── AnnuaireMDP.h++      # Déclaration de la classe et structures
├── annuaire_mdp.c++     # Implémentation des méthodes
├── main.cpp             # Tests et interface utilisateur
├── vault.txt            # Fichier de sauvegarde (généré)
└── Paskee.exe           # Exécutable compilé
```

### 📄 Description de chaque fichier

#### 1. `AnnuaireMDP.h++` (Fichier header)

**Rôle :** Déclare l'interface publique de la classe et la structure de données.

**Contenu :**

- Structure `EntreeMDP` : définit les champs d'une entrée (plateforme, utilisateur, motdepasse, note)
- Classe `AnnuaireMDP` : déclare toutes les méthodes publiques
- Attribut privé `std::map<std::string, EntreeMDP> entrees`

**Pourquoi ce fichier ?**

- Sépare l'interface (ce qu'on peut faire) de l'implémentation (comment on le fait)
- Permet à d'autres fichiers d'utiliser la classe sans voir le code interne
- Les guards `#ifndef _ANNUAIRE_MDP_H` évitent les inclusions multiples

**Inclus par :** `annuaire_mdp.c++` et `main.cpp`

---

#### 2. `annuaire_mdp.c++` (Fichier source d'implémentation)

**Rôle :** Contient le code de toutes les méthodes déclarées dans le header.

**Contenu :**

- Implémentation des 10 méthodes de la classe `AnnuaireMDP`
- Logique de gestion de la map (ajout, suppression, recherche)
- Code de sauvegarde/chargement avec parsing de fichiers
- Constructeur de copie

**Pourquoi ce fichier ?**

- Contient la "logique métier" du gestionnaire de mots de passe
- Peut être modifié sans toucher au header (tant que les signatures restent identiques)
- Compilé séparément puis lié avec `main.cpp`

**Dépendances :**

- `#include "AnnuaireMDP.h++"` pour les déclarations
- `<iostream>` pour l'affichage
- `<fstream>` pour les fichiers
- `<sstream>` pour le parsing
  ✅ Architecture modulaire (header/source)

## Compilation et exécution

### Compiler

```powershell
cd projet_final
g++ -O -Wall -std=c++17 annuaire_mdp.c++ main.cpp -o Paskee.exe
```

**Options expliquées :**

- `-O` : Optimisation du code
- `-Wall` : Affiche tous les warnings
- `-std=c++17` : Utilise le standard C++17
- `-o Paskee.exe` : Nom de l'exécutable

### Exécuter

```powershell
.\Paskee.exe
```

**Ce qui se passe :**

1. Exécution de `test()` → Vérifie le code
2. Affiche "Tests OK"
3. Charge `vault.txt` (s'il existe)
4. Lance l'interface utilisateur
5. Attend les commandes de l'utilisateur

---

## Points clés à expliquer au professeur

### 1. Choix de `std::map`

### 2. Séparation header/source

### 3. Gestion des erreurs

### 4. Tests automatisés

### 5. Format de fichier simple

---

## Améliorations possibles

### 1. Chiffrement

- Utiliser AES pour chiffrer `vault.txt`
- Demander un mot de passe maître au démarrage
- Protection contre lecture non autorisée

### 2. Validation des mots de passe

- Vérifier la complexité (longueur, caractères spéciaux)
- Comparer avec liste de mots de passe courants
- Suggérer des améliorations

### 3. Générateur de mots de passe

- Créer des mots de passe aléatoires
- Paramètres : longueur, types de caractères
- Utiliser `std::random`

### 4. Recherche avancée

- Recherche par utilisateur
- Recherche partielle (ex: "micro" trouve "microsoft")
- Filtrer par note

### 5. Export/Import

- Exporter en JSON ou CSV
- Importer depuis d'autres formats
- Compatibilité avec autres gestionnaires

---

## Résumé technique

**Paradigme :** Programmation orientée objet
**Conteneur principal :** `std::map<std::string, EntreeMDP>`
**Persistance :** Fichiers texte avec parsing manuel
**Interface :** CLI textuelle avec boucle infinie
**Tests :** Assertions automatisées au démarrage
**Standard :** C++17

Ce projet démontre :

- ✅ Maîtrise des classes et encapsulation
- ✅ Utilisation de conteneurs STL (map)
- ✅ Gestion de fichiers (ifstream/ofstream)
- ✅ Parsing de données (stringstream)
- ✅ Tests unitaires (assert)
- ✅ Interface utilisateur interactive
- ✅ Architecture modulaire (header/source)

```cpp
struct EntreeMDP
{
    std::string plateforme;   // Clé unique (ex: "microsoft", "google")
    std::string utilisateur;  // Email ou nom d'utilisateur
    std::string motdepasse;   // Mot de passe de l'utilisateur
    std::string note;         // Information supplémentaire (2FA, etc.)
};
```

**Pourquoi cette structure ?**

- Regroupe toutes les informations d'une entrée
- Facilite le passage de données entre fonctions
- Initialisation par défaut avec `{}`

### Classe `AnnuaireMDP`

```cpp
class AnnuaireMDP
{
private:
    std::map<std::string, EntreeMDP> entrees;  // Stockage clé-valeur

public:
    // Méthodes publiques...
};
```

**Pourquoi `std::map` ?**

- Recherche rapide par clé (O(log n))
- Tri automatique par nom de plateforme
- Pas de doublons possibles
- Accès direct : `entrees["microsoft"]`

## Méthodes principales

### 1. `add()` - Ajouter une entrée

```cpp
void add(const std::string &plateforme,
         const std::string &utilisateur,
         const std::string &motdepasse,
         const std::string &note = "")
```

**Fonctionnement :**

1. Vérifie que les champs obligatoires ne sont pas vides
2. Crée une nouvelle `EntreeMDP`
3. L'insère dans la map avec la plateforme comme clé
4. Si la plateforme existe déjà, elle est écrasée (mise à jour)

**Exemple :**

```cpp
vault.add("microsoft", "michel@efrei.net", "pass123", "compte pro");
```

### 2. `get()` - Récupérer une entrée

```cpp
EntreeMDP get(const std::string &plateforme) const
```

**Fonctionnement :**

1. Cherche la plateforme dans la map avec `find()`
2. Si trouvée, retourne la copie de l'entrée
3. Sinon, retourne une `EntreeMDP` vide `{}`

**Pourquoi `const` ?**

- La méthode ne modifie pas l'objet
- Permet d'appeler `get()` sur un objet constant

### 3. `remove()` - Supprimer une entrée

```cpp
void remove(const std::string &plateforme)
```

**Fonctionnement :**

1. Cherche la plateforme avec `find()`
2. Si trouvée, utilise `erase()` pour la supprimer
3. Affiche un message de confirmation ou d'erreur

### 4. `exists()` - Vérifier l'existence

```cpp
bool exists(const std::string &plateforme) const
```

**Fonctionnement :**

- Retourne `true` si `find()` ne retourne pas `end()`
- Utilisé avant `get()` pour éviter les erreurs

### 5. `print()` - Afficher toutes les entrées

```cpp
void print() const
```

**Fonctionnement :**

1. Vérifie si la map est vide
2. Parcourt toutes les entrées avec une boucle `for` range-based
3. Affiche : plateforme | utilisateur | mot de passe | note (optionnel)

**Syntaxe utilisée :**

```cpp
for (const auto &kv : entrees)
{
    const auto &e = kv.second;  // Récupère l'EntreeMDP
    // Affichage...
}
```

### 6. `save()` - Sauvegarder dans un fichier

```cpp
bool save(const std::string &nomFichier) const
```

**Format du fichier :**

```txt
plateforme:utilisateur:motdepasse:note
microsoft:michel@efrei.net:machkar776:compte principal
google:user@gmail.com:pass123:
```

**Fonctionnement :**

1. Ouvre le fichier en écriture avec `std::ofstream`
2. Pour chaque entrée, écrit les champs séparés par `:`
3. Ferme le fichier et retourne `true` si succès

**Pourquoi ce format ?**

- Simple à parser
- Lisible par un humain
- Chaque ligne = une entrée

### 7. `load()` - Charger depuis un fichier

```cpp
bool load(const std::string &nomFichier)
```

**Fonctionnement :**

1. Ouvre le fichier en lecture avec `std::ifstream`
2. Lit chaque ligne avec `std::getline()`
3. Parse la ligne avec `std::stringstream` et `std::getline(ss, var, ':')`
4. Reconstitue les `EntreeMDP` et les ajoute à la map

**Utilisation de `stringstream` :**

```cpp
std::stringstream ss(ligne);
std::getline(ss, plateforme, ':');  // Lit jusqu'au premier ':'
std::getline(ss, utilisateur, ':'); // Lit jusqu'au suivant
std::getline(ss, motdepasse, ':');
std::getline(ss, note);             // Lit le reste
```

### 8. Constructeur de copie

```cpp
AnnuaireMDP(const AnnuaireMDP &other) : entrees(other.entrees)
```

**Fonctionnement :**

- Utilise la liste d'initialisation pour copier la map
- La map fait une copie profonde automatiquement
- Affiche un message pour traçabilité

## Interface utilisateur (CLI)

### Fonction `test()`

```cpp
void test()
{
    // Série d'assertions pour valider le code
    assert(!vault.exists("microsoft"));
    vault.add("microsoft", "...", "...", "...");
    assert(vault.exists("microsoft"));
    // ...
}
```

**Pourquoi des tests ?**

- Vérifie que chaque méthode fonctionne correctement
- Détecte les bugs avant l'utilisation
- Si un `assert` échoue, le programme s'arrête

### Fonction `ui()`

**Boucle principale :**

```cpp
while (true)
{
    // Afficher menu
    // Lire commande
    // Traiter commande avec if/else if
    // Si "quitter", break
}
```

**Commandes disponibles :**

- `ajouter` : Demande plateforme, utilisateur, mot de passe, note
- `rechercher` : Cherche et affiche une entrée
- `supprimer` : Supprime une entrée
- `lister` : Affiche toutes les entrées
- `sauvegarder` : Enregistre dans vault.txt
- `quitter` : Propose de sauvegarder puis quitte

**Utilisation de `std::getline()` :**

```cpp
std::string cmd;
std::getline(std::cin, cmd);  // Lit toute la ligne (avec espaces)
```

**Pourquoi `getline()` et pas `cin >>` ?**

- Permet de lire des phrases avec espaces
- Évite les problèmes de buffer
- Plus fiable pour les interfaces utilisateur

## Concepts C++ utilisés

### 1. Classes et encapsulation

```cpp
class AnnuaireMDP
{
private:
    std::map<std::string, EntreeMDP> entrees;  // Données privées
public:
    // Méthodes publiques pour accéder aux données
};
```

**Principe :**

- Les données (`entrees`) sont privées → protection
- Les méthodes publiques contrôlent l'accès → sécurité
- On ne peut pas modifier `entrees` directement de l'extérieur

### 2. Références constantes (`const &`)

```cpp
void add(const std::string &nom, ...)
```

**Avantages :**

- `&` : Pas de copie → performance
- `const` : Ne peut pas être modifié → sécurité
- Idéal pour les paramètres string

### 3. `std::map` (conteneur associatif)

```cpp
std::map<std::string, EntreeMDP> entrees;
```

**Opérations :**

- Insertion : `entrees[key] = value`
- Recherche : `entrees.find(key)`
- Suppression : `entrees.erase(iterator)`
- Parcours : boucle range-based

### 4. Gestion de fichiers

**Écriture :**

```cpp
std::ofstream f("fichier.txt");
f << "texte" << std::endl;
f.close();
```

**Lecture :**

```cpp
std::ifstream f("fichier.txt");
std::string ligne;
while (std::getline(f, ligne))
{
    // Traiter ligne
}
f.close();
```

### 5. `std::stringstream` (parsing)

```cpp
std::stringstream ss("mot1:mot2:mot3");
std::string a, b, c;
std::getline(ss, a, ':');  // a = "mot1"
std::getline(ss, b, ':');  // b = "mot2"
std::getline(ss, c);        // c = "mot3"
```

**Utilité :**

- Parser des chaînes complexes
- Séparer par délimiteur
- Extraire des données structurées

### 6. Auto et range-based for

```cpp
for (const auto &kv : entrees)
{
    // kv.first = clé (std::string)
    // kv.second = valeur (EntreeMDP)
}
```

**Pourquoi `auto` ?**

- Le compilateur déduit le type
- Code plus court et lisible
- Type exact : `std::pair<const std::string, EntreeMDP>`
