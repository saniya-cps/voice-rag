@echo off
cd /d "%~dp0"
echo ==================================================
echo Cleaning .env and __pycache__ from GitHub repo...
echo Repository: https://github.com/saniya-cps/voice-rag
echo ==================================================

git rm -r --cached .venv 2>nul
git rm --cached .env 2>nul
git rm --cached .env.example 2>nul
git rm -r --cached __pycache__ 2>nul
git rm -r --cached src/__pycache__ 2>nul
git rm -r --cached src/rag/__pycache__ 2>nul
git rm -r --cached src/audio/__pycache__ 2>nul
git rm -r --cached src/llm/__pycache__ 2>nul
git rm --cached data/faiss_index.bin 2>nul
git rm --cached data/passages_cache.json 2>nul

git add -A
git commit -m "Remove .env and cache files from repository"
git push origin main

echo ==================================================
echo Done! .env and all cache folders removed from GitHub!
echo ==================================================
pause
