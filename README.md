# AutoReview

## An Explainable AI Framework for Automated Research Paper Review Using Large Language Models and Retrieval-Augmented Generation

AutoReview is an AI-based framework designed to assist researchers in reviewing research papers using Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and multiple specialized review agents.

The system analyzes different aspects of a research paper and generates an overall review containing scores, strengths, weaknesses, and improvement suggestions.

## Features

- Research paper PDF text extraction
- Automatic section identification
- Reference literature retrieval using RAG
- TF-IDF based document representation
- Multi-agent paper evaluation
- Structure analysis
- Novelty analysis
- Methodology analysis
- Literature analysis
- Experiment analysis
- Writing quality analysis
- Citation analysis
- Weighted overall scoring
- Final consolidated review using Gemini
- Automated review report generation

## System Workflow

Research Paper  
↓  
PDF Text Extraction  
↓  
Section Identification  
↓  
Reference Paper Retrieval  
↓  
RAG Context Preparation  
↓  
7 Specialized Review Agents  
↓  
Score Calculation  
↓  
Final Reviewer  
↓  
Automated Review Report

## Technologies Used

- Python
- Google Gemini API
- Retrieval-Augmented Generation (RAG)
- TF-IDF
- Scikit-learn
- PyMuPDF
- Pandas
- NumPy
- ReportLab

## Project Structure

```text
AutoReview/
├── agents/
├── core/
├── data/
├── evaluation/
├── llm/
├── outputs/
├── vector_db/
├── main.py
├── requirements.txt
├── README.md
└── .gitignore