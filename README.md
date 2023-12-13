# karpathyGPT

Implementation of https://www.youtube.com/watch?v=kCc8FmEb1nY

See: https://github.com/karpathy/ng-video-lecture

# Dataset
Dataset used is the "German Political Speeches Corpus" from https://politische-reden.eu/

# Train
```python train.py``` (approx. 5h for 8000 epochs on a single 80 GB NVIDIA H100)

```
step    0: train loss 5.0921, val loss 5.0914
step  500: train loss 1.9477, val loss 1.9567
step 1000: train loss 1.1632, val loss 1.1790
step 1500: train loss 1.0260, val loss 1.0490
step 2000: train loss 0.9574, val loss 0.9901
step 2500: train loss 0.9098, val loss 0.9518
step 3000: train loss 0.8733, val loss 0.9268
step 3500: train loss 0.8440, val loss 0.9119
step 4000: train loss 0.8163, val loss 0.8941
step 4500: train loss 0.7918, val loss 0.8844
step 5000: train loss 0.7682, val loss 0.8780
step 5500: train loss 0.7460, val loss 0.8722
step 6000: train loss 0.7223, val loss 0.8710
step 6500: train loss 0.7022, val loss 0.8665
step 7000: train loss 0.6815, val loss 0.8757
step 7500: train loss 0.6593, val loss 0.8716
step 8000: train loss 0.6385, val loss 0.8718
```

# Generate
## Usage
```python generate.py -h```
```
usage: generate.py [-h] [-m MODEL] [-c CONTEXT] [-n NUMTOKENS]

Load a GPT model

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Path to the model file (default: gpt.model)
  -c CONTEXT, --context CONTEXT
                        Initial context text (default: '.')
  -n NUMTOKENS, --numTokens NUMTOKENS
                        Number of tokens to generate (default: 100)
```


## With context
```python generate.py -m gpt.model -n 500 -c "Heute ist ein Tag"```

Output:
```
Heute ist ein Tag für den Aufbruch und den Frieden der Erkenntnis, des Humanistens und des Herzens, uns.
Ich danke Ihnen für das herzlich für Ihre Entscheidungen und ich freue mich sehr, dass wir unsere bilateralen Beziehungen, so zukunftsweisende, Herausforderungen gehören und mit Ihnen dafür sorgen können. Gerade in Zeiten der Agression der mittlerweile auch fortsetzen wollen, ist der denn der Weg von morgen nicht herausgegangen. Diese Erfahrungen, so meine ich, habe ich mitgehört. Zunächst möchte ich großen R
```

```python generate.py -m gpt.model -n 500 -c "Deutschland ist "```

Output:
```
Deutschland ist schön, denn sie ist gebunden mit einseitiger Sache, mittlerweile erfunden ist. Sie haben erkannt, weil Freiheit ist.Die Freiheit ist Mitmenschlichkeit auch schon gemacht. Sie haben ein Vorurteil geboten, Anschläge in Artikel 1 des Gedankens. Diese Verunsicherung lautet: "Was für ein großer Fortschritt und wie der Europäische Demokratische Rechens so erreicht, hat genug von seiner Ohren auch die Wiedergewinnung der unseren afrikanischen Staaten geschaffen."
Dennoch lebten die Verarmenden eine sch
```

## Without context
```python generate.py -m gpt.model -n 500```

Output:
```
. Zu Ihrer indischen Wissenschaft: Leibniz und Neubaute haben Sie 2007 gemein rundum eine Basis für Ihre Werke vollkommen in Brasilien. Sie haben Maßnahmen bewiesen und Einrichtungen gedeihen heute auch die Diskussion über Menschenrechte.
Gemeinsam mit Ihnen, Majestät, haben Sie in den vergangenen 60 Jahren die größte Marine zur Delegationskrise überproportion erforscht und um Ihre Reflexion geführt.
Liebe Frau Schavan, Ihnen herzlichen Dank. Ich freue mich sehr über die beiden Musikerinnen und M
```

```
. Sie können sich verarbeiten. Sie können sicher warm pflegen, ermutigen und erfolgen.
Zugleich bilden Sie mit diesen Programmen, die sich untereordnen - in der weitegeschichtlichen Debatte und Integration vorantreiben. Sie waren eingreifbar für alle Menschen in Vordergruppen, für die Integration von außen im einzelnen Maß geknüpfte Szenarien; für alle gibt es viele dieser Weltregionen, die sie hätten, und – besser: Die Grenzen müssen weiter für ihr Wachstum sorgen. Von dieser Frage gehören für m
```

```
Freilich verband sich überwiegend Einsätze. Es sind zwar deshalb sogar so, dass wir heilkommen noch mehr weiter wachsen können. Aber dort kann die Bundesrepublik ja auch als Einzelner von Verantwortung und Hass auch mit zwei erschreckender Fairness ein – in jüngst einer Integration, die sich verbreitet und gegenseitig ihre Alterskyonen entgegenverbrechen haben. Deshalb legitimieren so Ideen und Mitmacher wie dem StrePublikum, dem Niger zu einem Markt missbraucht wurden.
Integration, liebe Frau
```

# Ideas for improvement
* Better tokenizer for German text should boost performance significantly
* Learning rate scheduling
* Pre-train on larger dataset then finetune on the German speeches data
