# ARL-Model-RL-Unsicherheit

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


### Fragen für Michel:

- Was müssen wir genau abgeben? Code + Ausarbeitung? Welcher Umfang? Formale Vorgaben? Einladung zum Repo? Welche Sprache?
-> Abgabe bestehend aus Vortrag und Ausarbeitung (~10 Seiten), sowie Codeo und Video (evtl. auch Jupyter  Notebook für Übung).

- Welche Email? Michel.Tokic@lmu.de?
-> Am besten über Moodle
-> Github: micheltokic

- Aufgabenstellung konkretisieren
-> 

- Big Picture nochmal erklären. Wie hängen RL-Policy und probabilistische Neuronales Netzwerke zusammen? 
- Welche Daten sollen verauscht werden?
- Welche Environment? Carpole oder Mario?
-> Mit CartPole starten, dann auf Mario umbiegen
- Was meint er mit kapseln?
- Was ist mit Ein-Schritt-Dynamik gemeint? 
- aleatorischer vs epistemischer Unsicherheit?
- Ensemble-Modellierung?
- Dynamik-Modell?
- Weiterführende Links/Literatur/Hilfestellungen?

### Links

https://fairyonice.github.io/Create-a-neural-net-with-a-negative-log-likelihood-as-a-loss.html

https://blog.tensorflow.org/2019/03/regression-with-probabilistic-layers-in.html


