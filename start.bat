@echo off
title GPT Persona MCP
echo === GPT Persona MCP Server ===
echo Port: 8767
cd /d "%~dp0"
python server.py
pause
