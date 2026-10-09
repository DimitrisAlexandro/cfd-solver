#!/usr/bin/env bash
# AeroLab: the interactive front end. Opens http://localhost:8501
cd "$(dirname "$0")"
exec streamlit run app/AeroLab.py "$@"
