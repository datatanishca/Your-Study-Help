# Your Study Help 🤝

Study your PDFs with clear answers, summary, quizzes and keep track of your progress. **Your Study Help** is a Windows app that uses [Ollama](https://ollama.com/) to run its language and embedding models locally. No Gemini or OpenAI API key is required.

## Features

- **Ask your PDFs:** Ask questions in plain language and get a direct answer with a short reference to the PDF and page.
- **Study summary:** Create plain-language notes for each processed PDF, with page references. Download the notes as Markdown.
- **Practice quiz:** Generate multiple-choice questions from the PDFs. Check answers to see the correct choice, explanation, and source pages.
- **Progress:** Track quiz accuracy, answers submitted, and topics and questions to review.

## Requirements

- Windows 10 or later
- Python 3.14
- Ollama for Windows
- Internet access for installing packages and downloading the Ollama models. After setup, the AI features run locally.

## Setup on Windows

### 1. Install Ollama and download its models

Install Ollama from [ollama.com/download/windows](https://ollama.com/download/windows) and open it. In PowerShell, download the two models used by the app:

```powershell
ollama pull qwen3:4b
ollama pull embeddinggemma:300m
```

### 2. Open PowerShell in the project folder

In File Explorer, open this project folder, click the address bar, type `powershell`, and press Enter.

### 3. Create a Python environment and install the packages

Run these commands in PowerShell, one at a time:

```powershell
py -3.14 -m venv .venv314
.\.venv314\Scripts\python.exe -m pip install --upgrade pip
.\.venv314\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Start the app

Make sure Ollama is open, then run:

```powershell
.\.venv314\Scripts\python.exe -m streamlit run app.py
```

Your browser opens the app. Upload PDFs in **Your documents** and click **Process PDFs** before using the study tabs.

The app tabs are **Ask your PDFs**, **Study summary**, **Practice quiz**, and **Progress**.

## Privacy and local files

The app sends PDF text to Ollama running on your own computer at `http://localhost:11434`. It does not send documents to Gemini or OpenAI. Ollama models take several gigabytes of disk space. Local processing may be slower on computers without a capable graphics card.

Quiz history is saved locally in `study_progress.json`. The `.gitignore` file excludes that progress file, Python environments, `.env` files, and PDFs so personal data is not accidentally uploaded to GitHub.

## Project files

- `app.py` — the Streamlit app
- `requirements.txt` — Python packages
- `.env.example` — optional local Ollama address setting
- `readme.md` — setup and usage guide
