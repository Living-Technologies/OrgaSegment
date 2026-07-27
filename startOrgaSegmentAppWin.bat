call C:\ProgramData\Miniconda3\Scripts\activate.bat
call conda activate OrgaSegment
REM Uncomment the following to use GPU if possible.
REM SET USE_GPU=True
call streamlit run app.py
PAUSE
