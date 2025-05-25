# NeuroTutor AI - Advanced Multi-Agent Learning System

## Overview

NeuroTutor AI is a sophisticated AI-powered tutoring system designed to provide an advanced and interactive learning experience. It leverages a multi-agent architecture based on the ReAct (Reasoning and Acting) framework, enabling specialized AI agents to collaborate and solve complex queries across various subjects. The system features a rich web interface for users to interact with the AI, explore task details, and view the AI's reasoning process.

## Features

### Frontend (`index.html`)
* **Interactive Chat Interface:** Real-time messaging with the AI tutor.
* **Dynamic Message Display:** Shows user and AI messages with sender information, agent badges, timestamps, and associated task IDs.
* **Specialized AI Agent Selection:** Users can conceptually see different AI specialists (Master Tutor, Math Expert, Physics Guru, Chem Wizard, History Sage, Bio Specialist).
* **Session Management:**
    * Start new learning sessions.
    * View and load chat history.
* **Advanced Dashboard Panels:**
    * **Neural Analytics:** Displays real-time system intelligence including active sessions, messages processed, tasks completed, max ReAct steps, task status distribution (visualized with rings), and agent performance matrix.
    * **Task Explorer:** Provides a deep dive into specific tasks, showing task ID, status, the original query, metadata (assigned agent, number of steps, verification attempts), and a detailed timeline of the AI's actions for that task.
    * **ReAct Trace Viewer:** Offers a step-by-step view of the AI's thinking process (Thought, Action, Action Input, Observation) for a given task, including the final result.
* **Rich Text Processing:** Messages can display formatted mathematical expressions (fractions, exponents, subscripts, square roots), lists, code blocks, and subject-specific highlighted terms (e.g., physics constants, chemical elements).
* **User Input:** Auto-resizing textarea for message input with send-on-enter functionality.
* **System Status:**
    * Live neural activity indicator.
    * Typing indicator when the AI is processing.
    * Detailed error message display with options to retry or dismiss.
* **Responsive Design:** Adapts to various screen sizes for usability on desktop and mobile devices.
* **Aesthetics:** Features an animated background and a subtle neural network background effect.

### Backend (`tutor.py`)
* **Flask-Based API:** Provides endpoints for chat, history, agent information, task details, ReAct traces, statistics, and health checks.
* **Multi-Agent System:**
    * **`MainTutorAgent`:** Orchestrates query processing, classifies queries by subject, delegates to specialist agents, and can verify their results.
    * **Specialist Agents:** `MathAgent`, `PhysicsAgent`, `ChemistryAgent`, `HistoryAgent`, each equipped with specific knowledge and tools for their domain.
    * **`BaseAgent`:** Provides common functionalities for all agents, including LLM interaction and tool execution.
* **ReAct (Reasoning and Acting) Framework:** Agents perform tasks by iteratively going through a Think -> Act -> Observe loop, allowing for complex problem-solving.
* **Google Gemini Integration:** Utilizes the Gemini API for advanced language understanding, generation, and function calling.
* **Function Calling:** Agents can use predefined tools by describing their intent, and the LLM generates the appropriate function calls. Tools include:
    * **`Calculator`:** Performs mathematical calculations (arithmetic, expressions, unary operations).
    * **`PhysicsConstants`:** Retrieves values and units for fundamental physics constants.
    * **`ChemistryData`:** Provides information about chemical elements (symbol, atomic number, atomic mass).
* **Task Management:**
    * Defines and tracks tasks with statuses (Pending, In Progress, Completed, Failed, Needs Verification).
    * Logs detailed ReAct steps for each task.
* **Conversation Management:** Maintains conversation history for each user session.
* **Result Verification:** The `MainTutorAgent` can verify the responses from specialist agents to ensure accuracy and completeness.
* **Logging and Error Handling:** Comprehensive logging for monitoring and debugging.

## Architecture

NeuroTutor AI is built on a client-server model:

* **Frontend (Client):** A single-page web application (`index.html`) providing the user interface. It communicates with the backend via HTTP requests.
* **Backend (Server):** A Python Flask application (`tutor.py`) that hosts the AI agents, manages business logic, and exposes API endpoints.

The core of the AI logic revolves around a **Multi-Agent System** implementing the **ReAct Framework**:
1.  User submits a query through the frontend.
2.  The backend's `MainTutorAgent` receives the query.
3.  The `MainTutorAgent` classifies the query to determine the subject.
4.  If the query is general, the `MainTutorAgent` handles it directly. Otherwise, it delegates the query as a task to the appropriate **Specialist Agent** (e.g., `MathAgent`, `PhysicsAgent`).
5.  The Specialist Agent uses the **ReAct loop** (Think -> Act -> Observe) to process the task.
    * **Think:** The agent reasons about the query and its current state.
    * **Act:** The agent decides on an action. This might involve calling a **Tool** (e.g., `Calculator`) using Gemini's **Function Calling** capability, or concluding the task.
    * **Observe:** The agent processes the result of its action (e.g., output from a tool) and updates its state.
6.  This loop continues until the agent concludes it has a final answer.
7.  The `MainTutorAgent` may verify the specialist agent's result. If rejected, the task can be re-attempted with feedback.
8.  The final response is sent back to the frontend.

## Technologies Used

* **Frontend:**
    * HTML5
    * CSS3 (with Google Fonts, custom properties, animations, glassmorphism effects)
    * JavaScript (Vanilla JS for DOM manipulation, event handling, API calls)
* **Backend:**
    * Python 3.x
    * Flask (for web framework and API endpoints)
    * Flask-CORS (for Cross-Origin Resource Sharing)
* **AI & Machine Learning:**
    * Google Gemini API (`google-generativeai` Python SDK)
* **Development & Tooling:**
    * Standard Python libraries (`os`, `json`, `re`, `logging`, `datetime`, `dataclasses`, `enum`)

## Setup and Installation

### Prerequisites
* Python 3.7+
* A Google Gemini API Key

### Backend Setup
1.  **Clone the repository (if applicable) or place `tutor.py` in your project directory.**
2.  **Install Python dependencies:**
    ```bash
    pip install Flask Flask-CORS google-generativeai
    ```
3.  **Set up Environment Variables:**
    Create a `.env` file in the backend directory or set the environment variable directly:
    ```
    GEMINI_API_KEY='YOUR_GEMINI_API_KEY'
    ```
    Replace `YOUR_GEMINI_API_KEY` with your actual Gemini API key. The backend code in `tutor.py` will attempt to read this key.

### Frontend Setup
1.  Place the `index.html` file in a directory that can be served by a web server or directly opened in a browser (though for API interactions, serving it via the Flask app is recommended).
2.  The `index.html` file is self-contained in terms of CSS and JavaScript and makes API calls to the backend (assumed to be running on the same origin or configured for CORS).

## Running the Application

1.  **Start the Backend Server:**
    Navigate to the directory containing `tutor.py` and run:
    ```bash
    python tutor.py
    ```
    By default, the Flask server will start on `http://0.0.0.0:5000`.

2.  **Access the Frontend:**
    Open your web browser and navigate to `http://localhost:5000/`. The `index.html` page will be served.

## API Endpoints (`tutor.py`)

* `GET /`: Serves the main web interface (`index.html`).
* `POST /api/chat`: Handles chat requests. Expects JSON payload with `message` and `session_id`.
* `GET /api/history/<session_id>`: Retrieves conversation history for the given session ID.
* `GET /api/agents`: Provides information about available AI agents and their tools.
* `GET /api/tasks/<task_id>`: Returns detailed information for a specific task ID.
* `GET /api/react-trace/<task_id>`: Returns the ReAct reasoning trace (steps) for a specific task ID.
* `POST /api/clear-session/<session_id>`: Clears the conversation history and related tasks for the specified session.
* `GET /api/statistics`: Returns system usage statistics.
* `GET /health`: Provides a health check of the backend service, including LLM connectivity.

## File Structure

The project is organized as follows:

* `tutor.py`: Main application file containing the Flask API endpoints and AI agent logic.
* `templates/index.html`: The main HTML template for the web interface.
* `static/`: Directory for static files (CSS, JavaScript, images).