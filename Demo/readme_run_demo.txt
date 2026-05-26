py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m uvicorn app:app --reload

clear:

deactivate
Remove-Item -Recurse -Force .\.venv