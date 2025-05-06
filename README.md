# AI\_Driven\_Visualisation

## Project Description

**AI\_Driven\_Visualisation** is a cutting-edge initiative that combines Retrieval-Augmented Generation (RAG), Multi-Modal Content Processing (MCP), and robust automation pipelines via n8n to build an intelligent system capable of dynamically interpreting, visualizing, and narrating datasets. This system is not only context-aware but also capable of learning from structured and unstructured data to provide meaningful, interactive visual outputs in real-time. This project bridges the gap between human intent and machine understanding in the context of exploratory data analysis.

## Objectives

* Train and fine-tune a foundation model for content-aware data visualization.
* Implement Retrieval-Augmented Generation (RAG) to provide external knowledge to the model.
* Use Multi-modal Content Processing (MCP) to handle different data formats (text, tables, PDFs, images).
* Orchestrate the pipeline using n8n for automation, logging, and user feedback loops.
* Deploy a local or cloud-based app interface for users to upload data and receive tailored visualizations with natural language explanations.

---

## System Architecture

### 1. Data Ingestion

* Accepts CSV, Excel, JSON, SQL dumps, PDFs, and images via a front-end or file upload pipeline.
* n8n handles file watching and uploading events.

### 2. Preprocessing Layer

* Extract structured data using pandas, PyMuPDF, and Tesseract OCR (if needed).
* Standardize data schema.

### 3. Knowledge Retrieval (RAG)

* Documents and datasets are chunked and embedded using `LangChain` + `FAISS`.
* RAG framework connects to a vector database.
* User queries and data context are matched to relevant chunks.

### 4. Multi-modal Content Processing (MCP)

* Data type detection module.
* Image and table processing using `Pandas-Profiling`, `Plotly`, and `OpenCV`.
* Natural language interface to describe insights.

### 5. AI Engine

* Fine-tuned LLM (e.g., Mistral or Phi-2) trained on visualization and data analysis prompts.
* Generates both code (matplotlib/seaborn/Plotly) and explanations.
* RAG-enhanced prompting to inject external context.

### 6. Automation & Orchestration with n8n

* Automates pipeline stages: ingestion, processing, LLM inference, and visualization rendering.
* Logs inputs, outputs, and user feedback.
* Sends final results to users via email or dashboard.

### 7. Output Layer

* Generates interactive charts.
* Provides downloadable reports (PDF/HTML).
* Creates narrative insights using natural language.

---

## Future Enhancements

* Integrate voice command interpretation.
* Support for real-time streaming data.
* User feedback loop for fine-tuning model behavior.

---

## Project Roadmap

| Phase   | Goal                               | Timeline |
| ------- | ---------------------------------- | -------- |
| Phase 1 | MVP: Upload & Visualize            | Week 1-2 |
| Phase 2 | Add RAG Knowledge Integration      | Week 3   |
| Phase 3 | Automate with n8n                  | Week 4   |
| Phase 4 | Fine-tune AI for better narratives | Week 5   |
| Phase 5 | Launch beta interface              | Week 6   |

---

## Tech Stack

* **LLMs:** Mistral, Phi-2
* **RAG:** LangChain, FAISS
* **Automation:** n8n
* **Visualization:** Plotly, Seaborn, Matplotlib
* **Preprocessing:** pandas, PyMuPDF, Tesseract
* **Interface:** Streamlit or Gradio

---

> This project exemplifies the power of composability in AI workflows by bringing together retrieval systems, automation pipelines, and multimodal understanding to democratize data storytelling.
