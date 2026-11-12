# Quantum machine learning for finance - student notebook

QubitHub Qiskit Fall Fest 2026. One real financial question - *will a bank client subscribe to a term deposit after a telemarketing call?* - answered twice, under the same rules: by a classical logistic regression (scikit-learn) and by a variational quantum classifier built with Qiskit. Then the two are compared honestly: accuracy, precision, recall, F1, ROC-AUC, confusion matrices, runtime, and whether a gap is bigger than noise.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hastikacheddy/qiskit-fall-fest-finance-qml-participant/blob/main/student_finance_qml_workshop.ipynb)

> This is an educational experiment, **not** a demonstration of quantum advantage and **not** financial advice. Every number and figure is computed by the notebook; none is typed in. The quantum model is simulated exactly on an ordinary CPU.

Everything runs on an ordinary laptop or on free Google Colab (CPU only): **no API keys, no paid services and no quantum hardware.** No quantum background is needed.

## How the workshop works

Your instructor presents; you follow along in `student_finance_qml_workshop.ipynb` and write a few short functions yourself.

* All the boilerplate is provided (installation, data download, plots, the optimiser loop). You write **8 short TODO functions**, T1 to T8, plus an optional stretch task S1.
* After each TODO there is a **check cell** that tells you right away what is right and what is wrong. **Hints** open one level at a time.
* Later sections need your earlier TODOs. Every section starts with a **gate** that tells you exactly what is missing, so you never get a mysterious error from an unrelated cell.
* If you are stuck, ask. Your instructor may hand out a file called `instructor_solutions.py`. Put it **next to the notebook** and type `use_reference("T4")` (use the number of your TODO) to load the instructor's version of that function. The file is deliberately not part of this repository.

## Google Colab (nothing to install)

1. Click the **Open in Colab** badge above. Colab warns that the notebook was not authored by Google - that is the standard message for every notebook that comes from GitHub; choose *Run anyway*.
2. Run the setup cell. It installs Qiskit if it is missing (about a minute). If Colab asks for a restart, choose *Runtime, then Restart session*, and run the notebook again from the top.
3. If your instructor gives you `instructor_solutions.py`, upload it with the *Files* pane (folder icon on the left) so that it sits next to the notebook.

Colab does not save your changes to GitHub, and a recycled runtime forgets everything: use *File, then Save a copy in Drive* (or *Download .ipynb*) before you leave.

## Local Jupyter

Python 3.11 or newer.

```bash
python -m venv .venv
# Windows (PowerShell):   .venv\Scripts\Activate.ps1
# macOS / Linux:          source .venv/bin/activate
pip install -r requirements.txt
jupyter lab student_finance_qml_workshop.ipynb
```

Qiskit 2.x needs **NumPy 2 or newer**. `requirements.txt` pins the exact versions the notebook was tested with (Qiskit 2.6.0, NumPy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, and so on). Other versions usually work, but the first code cell prints a notice when yours differ, and numbers may then differ in the last decimals.

If your computer is slow, set `QUICK_MODE = True` in the configuration cell: a smaller experiment whose numbers differ from the full run by design. If you see joblib or "insufficient system resources" errors, set `N_JOBS = 1`.

## Data

UCI Machine Learning Repository, *Bank Marketing* (`bank-additional-full.csv`, 41,188 contacts of a Portuguese bank, 2008 to 2010), licence **CC BY 4.0**, [DOI 10.24432/C5K306](https://doi.org/10.24432/C5K306). The notebook downloads it once (about 1 MB) into `data/`, checks its SHA-256 and re-uses it; nothing is shipped in this repository. Please cite Moro, Cortez and Rita (2014), *A data-driven approach to predict the success of bank telemarketing*, Decision Support Systems 62, 22-31, [doi:10.1016/j.dss.2014.03.001](https://doi.org/10.1016/j.dss.2014.03.001).

The call duration is dropped (it is only known after the call, so keeping it would leak the answer, as the authors warn) and 12 exact duplicate rows are removed.

## Method in brief

* One stratified 75/25 split with a fixed seed. The test set is sealed: preprocessing, PCA and decision thresholds are fitted on training data only.
* Three models next to a do-nothing reference: logistic regression on all rows and features, a **matched** logistic regression that sees exactly the rows and features the circuit sees, and the **variational quantum classifier** (4 qubits, angle encoding, 24 trainable angles, trained with COBYLA, simulated exactly without noise).
* Uncertainty: bootstrap intervals, and a paired bootstrap for the differences between models.

## What is in this repository

| File | What it is |
|---|---|
| `student_finance_qml_workshop.ipynb` | the workshop notebook with the TODOs |
| `requirements.txt` | the exact library versions the notebook was tested with |
