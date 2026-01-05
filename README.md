# Personal Finance Assistant

Personal Finance Assistant is a Python-based command-line utility that helps users track income and expenses, organize transactions into categories, and gain a simple overview of their personal finances. It is designed as a lightweight, extensible starting point for building more advanced finance tools such as budgeting dashboards.

---

## Features

- Add and manage income and expense  from the command line.
- Track key attributes such as amount, date, category, and description.
- View summarized reports such as total income, total expenses, and net savings.
- Easily extendable to include budgeting, visualization, or AI-based recommendation modules.

---

## Project Structure

The repository currently has a minimal structure focused on a single application entrypoint and a dependency file.

```bash
personal-finance-assistant-/
├─ app.py            # Main application script (core logic and CLI/entrypoint)
└─ requirements.txt  # Python dependencies
```

- `app.py`: Contains the main logic for handling user input, storing transactions (in memory or simple storage), and generating summaries.
- `requirements.txt`: Lists all external Python libraries required to run the project.

As the project grows, you can split `app.py` into modules such as `models.py`, `services.py`, and `reports.py`, and add a `data/` directory for persistent storage.

---

##  Live Demo
🔗 https://personalfin.streamlit.app/

---

##  Tech Stack
- **Frontend**: Streamlit
- **Backend**: Python
- **ML**: Scikit-learn
- **Visualization**: Plotly, Matplotlib
- **Deployment**: Streamlit Cloud

---

##  Installation & Setup

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
streamlit run app.py

### Prerequisites

- Python 3.10 or higher installed on your system.
- Git installed to clone the repository.

#### Clone the repository

```bash
git clone https://github.com/tejas-0905/personal-finance-assistant-.git
cd personal-finance-assistant-
```

#### Create and activate a virtual environment (recommended)

```bash
python -m venv .venv

# On Linux/macOS
source .venv/bin/activate

# On Windows
.venv\Scripts\activate
```

#### Install dependencies

```bash
pip install -r requirements.txt
```

This command installs all required packages listed in `requirements.txt`.

---

## Usage

The exact commands depend on how `app.py` is implemented; adjust this section to match your argument parsing or menu system.

### Basic run

```bash
python app.py
```

Possible interaction patterns:

- Follow an on-screen menu to:
  - Add income or expense entries.
  - List all transactions.
  - View summaries (total income, total expenses, balance).

If `app.py` supports command-line arguments, document them here, for example:

```bash
python app.py --add --type expense --amount 500 --category "Food" --note "Dinner"
python app.py --report monthly
```

Replace these examples with the actual flags and options implemented in your script.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.


![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)


