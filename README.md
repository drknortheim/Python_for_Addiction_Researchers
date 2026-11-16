# Python for Addiction Researchers
### Project INTEGRATE — UNT Health Science Center, Fort Worth
**Dr. Kari Northeim**

This repository supports the *Python for Addiction Researchers* course delivered through MoodleCloud. It contains the dataset, Python scripts, and Jupyter notebooks used throughout the course.

---

## Course Overview

This course teaches addiction researchers how to use Python for real-world research tasks including data cleaning, statistical analysis, and meta-analysis. No prior programming experience is required.

**Python 1 — Introduction to Python**
- What Python is and why researchers use it
- Setting up your environment
- Basic syntax, variables, and data types
- Reading and writing files
- Working with research data using pandas

**Python 2 — Advanced Python and Introduction to GitHub**
- Data cleaning and transformation
- Working with REDCap exports
- Descriptive statistics
- Data visualization
- Automating repetitive research tasks
- Meta-analysis (replicating Mun et al., 2022)
- Version control with GitHub

---

## Repository Structure
├── data/
│ └── dat_all_PS.csv # Project INTEGRATE dataset (Mun et al., 2022)
├── notebooks/
│ ├── python1/ # Intro to Python notebooks
│ └── python2/ # Advanced Python notebooks
├── code/
│ └── project_integrate_meta.py # IPD meta-analysis script
└── README.md
---

## Dataset

`dat_all_PS.csv` contains 9,471 participant observations from Project INTEGRATE, a multi-study individual participant data (IPD) meta-analysis of brief alcohol interventions in college students.

**Citation:** Mun, E.-Y., Zhou, Z., Huh, D., Tan, L., Li, D., Tanner-Smith, E. E., Walters, S. T., & Larimer, M. E. (2022). Brief alcohol interventions are effective through six months. *Prevention Science.*

---

## Getting Started

Load the dataset directly in Python or JupyterLite:

```python
import pandas as pd
DATA_URL = "https://raw.githubusercontent.com/drknortheim/Python_for_Addiction_Researchers/main/data/dat_all_PS.csv"
df = pd.read_csv(DATA_URL)
print(df.head())
```
