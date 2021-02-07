# lime

### Skopiowana wersja oryginalnego repozytorium LIME. Zawiera autorskie modyfikacje oraz przeprowadzone badania efektywności w ramach wykonanej pracy magisterskiej.

This is the forked version of original LIME repository. Contains modified version of the library and efficiciency study done during preparing thesis. 


## Najważniejsze zmienione elementy biblioteki

- `doc/mod/` - katalog zawierający środowisko testowe, czyli wszystkie zmiany związane z prowadzeniem badań efektywności
<br />
<br />
    - `doc/mod/data/` - katalog przechowujący zbiory danych w oryginalnej formie, wykorzystane w środowisku testowym 
        - `doc/mod/data/img/` - katalog zawierający obrazki przedstawiające wygenerowane wyjaśnienia
<br />
<br />      
    - `doc/mod/saved_results/` - katalog przechowujący surowe wyniki przeprowadzonych eksperymentów (pliki o przyrostku `*_v4.npy`)
<br />
<br />
    - `doc/mod/notebooks/` - katalog przechowujący notatniki `jupyter notebook` wykorzystane w prowadzonych badaniach:
        - `doc/mod/notebooks/EfficiencyTest_*` - pliki o tym przedrostku to notatniki, które opisują przeprowadzone badania efektywności
        - `doc/mod/notebooks/DatasetLookup.ipynb` - notatnik służący przeglądaniu zbiorów danych składających się na środowisko testowe
        - `doc/mod/notebooks/fidelity_comparison.ipynb` - notatnik w którym porównano uzyskane wyniki wierności odwzorowania
        - `doc/mod/notebooks/NormalizationMethods.ipynb` - notatnik prezentujący zaprojektowaną metodę normalizacji
        - `doc/mod/notebooks/test_*.ipynb` - pliki o tym przedrostku to notatniki zawierające proste testy użycia zaimplementowanych modyfikacji
        - `doc/mod/notebooks/tree_explanation_multiregressor.ipynb` - plik generujący zaimplementowane nowe formy wyjaśnienia w postaci grafu reguł (obrazki) oraz tekstowej
<br />
<br />      
    - `doc/mod/utils/` - katalog do przechowywania pomocniczych modułów wykorzystywanych w środowisku testowym 
<br />
<br />      
- `lime/` - pakiet stanowiący serce biblioteki, przechowujący moduły służące do generowania wyjaśnień
    - `lime/explanation.py`, `lime/explanation_mod.py` - klasy reprezentujące wyjaśnienie predykcji
    - `lime/lime_base.py`, `lime/lime_base_mod.py` - klasy bazowe, odpowiadające za trening modelu zastępującego
    - `lime/lime_base_multiclassifier.py`, `lime/lime_base_multiregressor.py`, `lime/lime_base_singleclassifier.py` - klasy podrzędne, realizujące trening modeli zastępujących zgodnie z zaprojektowanymi modyfikacjami
    - `lime/tabular.py`, `lime_tabular_mod.py` - klasy bazowe, których instancje przygotowują argumenty do wyjaśnienia predykcjia na danych tabelarycznych
    - `lime/lime_tabular_multiclassifier.py`, `lime/lime_tabular_multiregressor.py`, `lime/lime_tabular_singleclassifier.py` - klasy podrzędne, generujące wyjaśnienia predykcji na danych tabelarycznych
    

## Przygotowanie środowiska
W celu poprawnego uruchomienia, wskazane jest utworzenie wirtualnego środowiska `virtualenv` z Python 3.7:
```
virtualenv -p <ścieżka prowadząca do Python 3.7> <nazwa środowiska>
```

Po utworzeniu wirtualnego środowiska należy je uruchomić:
```
source <nazwa środowiska>/bin/activate
```

Aby zapewnić wszystkie niezbędne pakiety, można je automatycznie zainstalować zgodnie z plikiem `requirements.txt`:
```
pip install -r requirements.txt
```