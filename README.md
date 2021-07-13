# ARL-Model-RL-Unsicherheit

## Setup
Um OpenAI Baselines zu verwenden wird Tensorflow Version 1.13 benötigt

> pip install tensorflow==1.13.2    
> pip install stable-baselines3 

Außerdem kann es unter Windows zu Problemen mit MPI kommen. Um diese zu umgehen, muss man den MPI Installer von der folgenden Seite downloaden und installieren:
> https://www.microsoft.com/download/details.aspx?id=100593

Falls LunarLander-v2 als Environment benutzt werden soll, müssen die folgenden Packages installiert werden:
> conda install swig    
> pip install box2d-py
 
Um MarioAI zu installieren, müssen die folgenden Schritte ausgeführt werden:
> git clone https://github.com/micheltokic/marioai.git  
> cd marioai/gym-marioai    
> pip install .     
> (Den Punkt nicht vergessen!)


## Organisatorisches

- Was müssen wir genau abgeben? Code + Ausarbeitung? Welcher Umfang? Formale Vorgaben? Einladung zum Repo? Welche Sprache?
    - Abgabe bestehend aus Vortrag, Video und Ausarbeitung (~10 Seiten)
    - Repository
    - Evtl. auch Jupyter  Notebook für Übung).
- Welche Email? Michel.Tokic@lmu.de?
- Am besten über Moodle
    - Github: micheltokic
    - Aufgabenstellung konkretisieren

## Aufgabenstellung
- Idee: Trainieren eines Agenten basierend auf einer Environment die partiell unsicher ist (z.B. mithilfe von verrauschten Daten), so dass der Agent unsicheren Zuständen gezielt vermeidet
- Ziel: Framework für RL mit Unsicherheit -> Sollte dann für verschiedene Gyms funktionieren (z.B. CartPole, LunarLander, Mario)
 - Aufgabenschritte:
     1. Environment kapseln, so dass Unsicherheit für bestimmte Bereiche gegeben werden kann (z.B. verrauschte Daten -> wir müssen das herausfinden oder ein Programm schreiben, dass das für uns herausfindet ^^)
     2. ...    

## TODO
- [ ] Framework entwickeln
- [ ] Framework mit CartPole testen
- [ ] Framework auf Mario anwenden
- [ ] Was meint er mit kapseln?

## Begriffsdefinitionen
- **Ein-Schritt-Dynamik:** Ich befinde mich in einem Markov-Zustand und führe eine Aktion aus. In welchem Zustand befinde ich mich danach? Diese Frage wird an das Dynamik-Modell gestellt.
- **Aleatorische und epistemische Unsicherheit:** Aleatorische Unsicherheit ist eine grundsätzliche Unsicherheit, die nicht reduzierbar ist. Im Gegensatz dazu kann die epistemische Unsicherheit durch mehr Daten verringert werden.
- **Ensemble-Modellierung:** Erwartungswert + Streuung als Rückgabewert. Entweder über BNN oder viele MLPs.
- Dynamik-Modell = Einschritt-Model

### Links
- [Neural Net with negative log likelihood as a loss](https://fairyonice.github.io/Create-a-neural-net-with-a-negative-log-likelihood-as-a-loss.html)
- [Regression with probabilisitic layers](https://blog.tensorflow.org/2019/03/regression-with-probabilistic-layers-in.html)
- [Modeling Epistemic and Aleatoric Uncertainty
with Bayesian Neural Networks and Latent Variables](https://mediatum.ub.tum.de/doc/1482483/1482483.pdf)

