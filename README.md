# nn
A repository egy egy rejtett rétegű (feedforward) neurális háló tanításához és a MNIST adatbázison történő kiértékeléséhez használt Python kódot tartalmaz. A repositoryban található fájlok:

* [mlp.py](mlp.py): a használt neurális háló osztály

* [utils.py](utils.py): különféle segédfüggvények

* egy [jupyter notebook](notebook.ipynb) ami a futtatást és az eredményeket - beleértve a feltanult háló súlyait -  szemlélteti (egyelőre nem történt hiperparaméter optimalizáció)

# Használat 
~~~
python mlp.py [-h] [-s HIDDEN] [-b BATCH_SIZE] [-i ITERATIONS]
              [-l LEARNING_RATE] [-d LR_DECAY_RATE] [-r REG_LAMBDA]
              [-c {softmax,mse}] [-p PLOT_WEIGHTS_FN]
~~~
Jelenleg a rejtett réteg és a minibatchek mérete, a learning rate és
(exponenciális) decay, a regularizáció (weight decay) lambda súlya, illetve a
használt célfüggvény paraméterezhető.
A feltanult háló súlyairól készült png egy megadott fájlba menthető (jelenleg csak 100 méretű rejtett réteg esetén).


# Függőségek:
* sklearn: a MNIST adatbázis beolvasásához

* matplotlib: a súlyok szemléltetéséhez

* jupyter: a notebook futtatásához

# További fejlesztési lehetőségek:
* hiperparaméter-optimalizáció

* további rejtett réteg hozzáadása a hálóhoz

* epoch-onkénti accuracy report

