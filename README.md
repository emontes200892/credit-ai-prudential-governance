# When Credit AI Is Wrong - reproducibility package v2

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

## Que reproduce

| Manuscrito | Implementacion y salida |
|---|---|
| Secciones 3-4 | 1,000 registros; 17 predictores; particiones 550/150/100/200 congeladas |
| Tabla 1 | LR, RF, GB; standard, balanced, IDCS; 5 bootstraps; politicas U y E |
| Tabla 2 | 2,000 bootstraps pareados sobre casos de prueba; referencia LR-standard-U |
| Tabla 3 | SHAP agrupado en el logit calibrado; overlap y cambios de decision |
| Seccion 6.2 | 20 casos x 3 semillas LIME, 1,500 muestras por explicacion |
| Tabla 4 | Revision de 5%, 10%, 20%; aleatoria, cercania al umbral, inestabilidad y combinada |
| Seccion 5 | Sin mejora, revision imperfecta y cota oraculo; costo cobrado por revision |
| Tabla 5 | 9 escenarios de costos, con ensamble RF-standard fijo |
| Tabla 6 | Auditoria descriptiva de edad; denominadores y conteos incluidos |
| Figuras 1-2 | Graficas generadas desde las salidas de esta ejecucion |
| Seccion 7 | Dossiers tecnicos con score, politica, costos y factores; sin validacion legal empirica |

`results/reference_run/` contiene una ejecucion real, los 45 modelos serializados,
las predicciones, las contribuciones SHAP, los coeficientes LIME, las tablas CSV y
fragmentos LaTeX. `docs/VALIDATION.md` documenta las verificaciones realizadas.

## Correcciones relevantes respecto al ZIP anterior

1. En UCI, `kredit = 0` significa **malo**, y `1` **bueno**. Ahora se convierte
   explicitamente a `bad_credit = 1 - kredit`; hay 300 malos y 700 buenos.
2. El acceso actual verificado es UCI **573**, DOI `10.24432/C5QG88`. Se conserva
   el archivo oficial y su tabla de codigos. El ID 522 del paper es una referencia
   anterior al repositorio; no se usa para la adquisicion de este paquete.
3. El preprocesamiento se ajusta una sola vez en los 550 registros de fitting,
   antes del bootstrap. Edad, estatus personal/sexo y trabajador extranjero quedan
   fuera del modelo; edad se guarda solo para la auditoria.
4. La huella de datos incluye el archivo fuente y el target, no solo X.
5. Se distingue perdida por etiqueta realizada de perdida esperada por el modelo.
6. Se implementan los experimentos faltantes, con salidas por caso y verificaciones.

## Limite importante para el paper

No se conservaron las particiones ni los modelos originales que produjeron las
tablas del PDF. Este paquete congela un **nuevo** split y documenta decisiones
faltantes (regularizacion de calibracion, semillas LIME, referencias y empates).
Las cifras originales **no quedan certificadas** por esta reconstruccion.

`results/reference_run/manuscript_comparison.csv` compara 203 celdas/numeros con
el manuscrito; las discrepancias permanecen visibles. No se buscaron semillas para
forzar coincidencias. `docs/MANUSCRIPT_UPDATE_REQUIRED.md` describe los cambios
necesarios antes de afirmar que el paper es numericamente reproducible con este ZIP.

LendingClub y HMDA son extensiones planeadas, sin resultados en este paquete.
Tampoco hay una validacion juridica automatica, ganancias monetarias observadas,
estimaciones de capital regulatorio ni efectos sistemicos estimados.

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
