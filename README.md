# When Credit AI Is Wrong - reproducibility 

**Status: complete, executable reconstruction of the historical-credit protocol;
not recovery of the original numerical experiment.**

Authors of the accompanying manuscript: Israel Cedillo-Lazcano and Edwin
Montes-Orozco, Universidad de las Americas Puebla. The supplied manuscript is
preserved unchanged in `manuscript/manuscript_supplied.pdf`.

## Inicio rapido

Usa Python **3.12** (version verificada: 3.12.14, Linux). Descomprime el ZIP y abre
una terminal dentro de esta carpeta. Los datos y los resultados ejecutados ya
estan incluidos; no necesitas descargar UCI para correr el experimento.

```bash
python -m venv .venv
```

Activa el entorno:

```bash
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
python -m pip install -r requirements.lock.txt
python verify_package.py
python run_experiments.py --out results/my_run
```

La instalacion requiere internet. La ejecucion del experimento usa solo los datos
locales. El tiempo depende del equipo; son 45 ajustes y 60 explicaciones LIME.
El script rechaza una carpeta de salida que ya existe para proteger resultados.

Para las pruebas y el notebook:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
python -m jupyter lab notebooks/Explorar_reproducibilidad.ipynb
```
## English reviewer guide

Run `python run_experiments.py --out results/my_run` after installing the locked
dependencies. All data are bundled with attribution. Read `docs/PROTOCOL.md` for
the exact methodological choices and ambiguities, `docs/OUTPUTS.md` for the file
dictionary, and `results/reference_run/RUN_REPORT.md` for the executed-run summary.
The supplied manuscript's rounded values are isolated in `manuscript/reported_*.csv`
and are used **only** for comparison, never for fitting or figure generation.

## Attribution and rights

Data: South German Credit (2020), UCI Machine Learning Repository,
https://doi.org/10.24432/C5QG88, CC BY 4.0. Original ASCII, R documentation and
code table are redistributed unchanged. Canonical CSV translates column names and
explicitly recodes the target; no rows are removed or reordered.

The manuscript remains the authors' work. No new license for the manuscript or
research code is asserted by this reconstruction; see `docs/RIGHTS_AND_DATA.md`.
