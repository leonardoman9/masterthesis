# Master Thesis Template (AI @ UNIBO)

Template LaTeX (non ufficiale) per la tesi magistrale in Artificial Intelligence.

## Requisiti

- Distribuzione LaTeX con `latexmk`, `xelatex`, `biber` (es. TeX Live o MacTeX)
- Font `Times New Roman` installato sul sistema

Nel file `.tex` principale ricordati di usare:

```tex
\setmainfont{Times New Roman}
```

## Build veloce

Dalla cartella `thesis/`:

```bash
latexmk -xelatex example.tex
```

Sostituisci `example.tex` con il tuo file principale.

## Dove finisce l'output

- File intermedi in `thesis/build/`
- PDF finale in `thesis/output.pdf`

Questo comportamento è definito in `thesis/latexmkrc`.

## Compilazione continua (watch)

Ricompila automaticamente a ogni modifica:

```bash
latexmk -pvc -xelatex example.tex
```

Interrompi con `Ctrl+C`.

## Pulizia artefatti

Rimuove file temporanei:

```bash
latexmk -c example.tex
```

Pulizia completa (inclusi output finali generati):

```bash
latexmk -C example.tex
```

## Build manuale (senza latexmk)

Se vuoi compilare a mano:

```bash
xelatex example.tex
biber example
xelatex example.tex
xelatex example.tex
```

## Risoluzione problemi rapida

- `biber: command not found`: installa `biber` dalla tua distribuzione TeX.
- Errore font `Times New Roman`: installa il font o cambia `\setmainfont{...}`.
- Citazioni non aggiornate: riesegui `latexmk -xelatex example.tex` (o il flusso manuale completo).
