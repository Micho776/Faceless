@echo off
echo ============================================================
echo FACELESS - Check Model Status
echo ============================================================
echo.

call C:\Users\miche\miniconda3\Scripts\activate.bat facerec

echo Current Model Status:
echo ------------------------------------------------------------
python -c "import pickle; data=pickle.load(open('data/encodings.pickle','rb')); print(f'\nTotal encodings: {len(data[\"encodings\"])}'); print('\nPeople:'); [print(f'  - {n}: {data[\"names\"].count(n)} encoding(s)') for n in set(data['names'])]"

echo.
echo ------------------------------------------------------------
echo Learning Samples:
echo ------------------------------------------------------------
if exist "learning_samples\*" (
    for /d %%D in (learning_samples\*) do (
        for /f %%C in ('dir /b "%%D\*.pkl" 2^>nul ^| find /c ".pkl"') do (
            echo   - %%~nxD: %%C sample(s)
        )
    )
) else (
    echo   [No learning samples collected yet]
)

echo.
echo ------------------------------------------------------------
echo Backups:
echo ------------------------------------------------------------
if exist "data\backups\*.pickle" (
    dir /b /o-d data\backups\*.pickle | findstr /n "^" | findstr "^[1-5]:"
    echo   [Showing 5 most recent backups]
) else (
    echo   [No backups yet - created after first improvement]
)

echo.
echo ============================================================
pause
