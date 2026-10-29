# Quitting after failure in Eedi maths quizzes

Learning Analytics / EDM course project (IIT Bombay). Team: Melvin Varghese (24b2174), Varun Lalwani (24b2564), Sandhya Kedi (26m1027).

We use the **Eedi CausalEdu** event log to ask three questions:
1. Is the lesson step linked to learning?
2. Where in a session does quitting concentrate?
3. Is quitting persistent, and do quitters return?

## Folder layout
| Folder | What is inside |
|---|---|
| `analysis/` | All code. `00_build_dataset.py` builds the event table; `05` change point; `10` RQ1; `20`, `21`, `22` RQ2; `30`, `31` RQ3; `40` figures; `50` LaTeX numbers and tables. `common.py` has shared helpers. |
| `results/tables`, `results/figures` | Every table and figure used in the report. |
| `report/` | `report.tex` (the paper), `auto_numbers.tex` and `tab_*.tex` (generated), `report.pdf`. |
| `*/archive_old_rqs/` | Earlier version of the project (topic hotspots, year groups). Not used in the report. |

## How to run
1. Download the Eedi CausalEdu data and put the CSV files in `Datasets/CausalEduDataset/data/` (the data is not in this repo).
2. `pip install -r analysis/requirements.txt`
3. Run everything (about 8 minutes): `bash analysis/run_all.sh` (Linux/Mac) or `analysis\run_all.bat` (Windows).
4. Build the PDF: `cd report && latexmk -pdf report.tex`

All random steps use seed 610.
