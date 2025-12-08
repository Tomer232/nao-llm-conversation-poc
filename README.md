# NAO LLM Conversation System

This project is a proof-of-concept (POC) system that connects a NAO humanoid robot (or a Webots simulation of NAO) to a Large Language Model (LLM).
It enables fully spoken conversations via:

* Automatic Speech Recognition (ASR) using OpenAI Whisper API
* LLM responses (via an external model, e.g. Grok, called through HTTP)
* Text-to-Speech (TTS) back to the robot or PC speakers
* A simple control layer and optional Web UI

Current focus: simulation mode with Webots + NAO model.
Real-robot integration (physical NAO over NAOqi) will be documented later.

---

## High-Level Architecture

The system consists of three main layers:

1. Backend (Python)

   * Manages the conversation loop (`conversation_loop.py`)
   * Talks to:

     * Whisper API (speech-to-text)
     * LLM API (text-to-text)
     * TTS engine (text-to-speech)
   * Exposes a small control server (`control_server.py`) that the Webots controller (`nao_talk`) can call to trigger listening / speaking cycles
   * Maintains a simple state file (`robot_state.txt`) for “idle / listening / thinking / speaking”

2. Webots Simulation (NAO + `nao_talk` controller)

   * Webots world (e.g. `nao_demo`) contains a NAO robot
   * The NAO robot uses a controller script called `nao_talk` (maintained in your Webots project)
   * `nao_talk` communicates with the Python backend (via `control_server.py` or by updating `control_flags.txt`) to:

     * Tell the backend when the robot should listen
     * Optionally read back status (for LEDs, animations, etc.)

3. Web UI (React, optional)

   * Located in `webui/`
   * Provides an optional browser-based interface for monitoring or controlling the simulation
   * Talks to the backend (currently minimal / experimental)

---

## Repository Structure

nao-llm-conversation-poc/
├─ audio/
│  ├─ `__init__.py`
│  ├─ `asr_whisper.py`     (ASR using OpenAI Whisper API)
│  └─ `tts.py`             (Text-to-speech via PC speakers)

├─ backends/
│  ├─ `simulation_backend.py`  (Backend for Webots / PC simulation)
│  └─ `real_backend.py`        (Skeleton for real NAO / NAOqi integration)

├─ webui/                  (React-based web UI, optional)
│  ├─ src/
│  ├─ public/
│  ├─ package.json
│  └─ …

├─ `control_flags.txt`      (Simple flags used for controlling the loop)
├─ `control_server.py`      (HTTP/socket server used by Webots `nao_talk`)
├─ `conversation_loop.py`   (Main conversation loop orchestrator)
├─ `nao_interface.py`       (Common interface for NAO / simulation backends)
├─ `requirements.txt`       (Python dependencies)
├─ `robot_state.py`         (Writes `robot_state.txt` — idle/listening/etc.)
├─ `robot_state.txt`        (Generated at runtime, ignored by git)
├─ `settings.json`          (Runtime configuration: ports, modes, etc.)
├─ `.gitignore`
└─ `README.md`

Note: Webots world files and the `nao_talk` controller script live in your Webots project, not in this repo.

---

## Prerequisites

Backend (Python)

* Python 3.11+ (recommended current 3.x)
* pip
* Internet access (for Whisper + LLM API calls)

Web UI (optional)

* Node.js LTS (e.g. 18.x or 20.x)
* npm

Simulation

* Webots installed
* NAO model available in your Webots installation

---

## Installation – Backend (Python)

Clone the repository:

git clone [https://github.com/Tomer232/nao-llm-conversation-poc.git](https://github.com/Tomer232/nao-llm-conversation-poc.git)
cd nao-llm-conversation-poc

Create and activate a virtual environment (Windows):

python -m venv venv
venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

---

## API Keys and Configuration

The backend expects two environment variables, provided via a local `.env` file in the project root.

Create a `.env` file in the root of the project with:

# LLM key (e.g. Grok or other provider)

GROK_API_KEY=your_grok_or_llm_key_here

# OpenAI key for Whisper ASR

OPENAI_API_KEY=your_openai_key_here

Notes:

* `.env` is ignored by git (see `.gitignore`) and must never be committed.
* If you use a different LLM provider instead of Grok, you can keep the env name `GROK_API_KEY` but adapt the Python code to call your provider.

---

## Running – Backend Simulation Mode (no Webots)

This is the simplest way to test that everything works using your PC microphone and speakers.

From the project root:

venv\Scripts\activate
python conversation_loop.py

Behavior (simplified):

* Backend prints a greeting (simulation)
* It listens via your microphone using Whisper API
* It sends the recognized text to the LLM
* It speaks the LLM answer via your PC speakers
* Internal state is written to `robot_state.txt` in the sequence:
  idle → listening → thinking → speaking → idle

You can stop the process with `Ctrl + C`.

---

## Running – Full Simulation with Webots + `nao_talk`

This section assumes you already have a Webots project with:

* A world named `nao_demo` (or similar)
* A NAO robot node whose controller is `nao_talk`

If you do not have this yet, create or adapt a Webots world with a NAO robot and write a controller script named `nao_talk` that can talk to the backend via HTTP or via file/flag mechanism.

### 1. Start the backend

In a terminal:

cd nao-llm-conversation-poc
venv\Scripts\activate
python control_server.py    (optional, if `nao_talk` uses it)

Then, either in another terminal or after starting the control server:

venv\Scripts\activate
python conversation_loop.py

Make sure:

* `settings.json` and/or `control_flags.txt` match what your `nao_talk` controller expects (ports, flags, etc.).
* Example: if `nao_talk` calls `http://localhost:5005/start_listening`, ensure that `control_server.py` is configured to listen on port `5005`.

### 2. Configure Webots to use `nao_talk`

In Webots:

1. Open your NAO world (e.g. `nao_demo`).
2. Select the NAO robot node in the scene tree.
3. Set its Controller field to `nao_talk`.
4. Ensure the `nao_talk` controller script:

   * Connects to the backend (for example via HTTP to `control_server.py`), or
   * Updates `control_flags.txt` and/or reads `robot_state.txt` in order to know when to trigger speech, animations, etc.

Typical logic in `nao_talk`:

* When the user presses a key, clicks a button in Webots, or some condition is met:

  * Send a “start listening” signal to the backend.
* Poll or receive updates for the backend state (from `robot_state.txt` or an API).
* Change NAO posture / eye LEDs depending on state, for example:

  * `listening` → blue
  * `thinking` → yellow
  * `speaking` → green
* Optionally play audio inside Webots if you route audio outputs into the simulation.

### 3. Run the simulation

1. Ensure `conversation_loop.py` (and `control_server.py` if used) are running and ready.
2. Press Play in Webots to start the `nao_talk` controller.
3. Trigger the conversation (key press or any event inside `nao_talk`).

The NAO in Webots should:

* Indicate “listening”
* Wait for your speech (via PC mic and Whisper)
* Wait for the LLM reply
* Perform a “speaking” action (robot gesture, Webots audio, or both)

If nothing happens:

* Confirm `control_server.py` is reachable from `nao_talk`.
* Check that host and port match between `nao_talk` and `settings.json`.
* Check terminal logs from both Webots (controller console) and Python backend.

---

## Web UI (Optional)

The `webui` folder contains a React-based UI. It is optional and is not required for the core simulation to work. You can use it to:

* Display basic status of the conversation
* Provide buttons to trigger listening, reset, or debug (depending on how you extend it)

Setup:

cd webui
npm install

Run:

npm start

This usually opens `http://localhost:3000` in your browser.

If you extend the backend with HTTP endpoints (for status or controls), you can connect this React app to those endpoints.

---

## Configuration Files

### `settings.json`

Used to configure runtime options, such as:

* Backend mode (simulation vs real robot)
* Ports for `control_server.py`
* Other tuning flags

Example (adapt to match your actual file):

{
"mode": "simulation",
"control_server_host": "127.0.0.1",
"control_server_port": 5005
}

Keep `settings.json` free of secrets (no API keys).

---

### `control_flags.txt`

Simple text file read/written by `control_server.py` / `nao_talk` / `conversation_loop.py` to coordinate:

* When to start listening
* When to stop
* Pausing / resuming loops

You can inspect and adapt the logic inside `control_server.py` and `conversation_loop.py` to match your preferred control flow.

---

### `robot_state.txt`

* Automatically written by `robot_state.py`.
* Contains a single word: `idle`, `listening`, `thinking`, or `speaking`.
* Can be read by the Webots controller or other tools to visualize the robot’s internal state.
* Ignored by git and recreated at runtime.

---

## Development Notes

The system is intentionally designed as a POC: simple, readable, and easy to extend.

You can swap:

* The LLM provider (keep the same interface in `conversation_loop.py`).
* The ASR provider (replace Whisper with another engine in `audio/asr_whisper.py`).
* The TTS backend (replace `pyttsx3` in `audio/tts.py` with any other TTS engine).

Recommended extensions:

* Implement the full real NAO backend in `backends/real_backend.py` using NAOqi.
* Add richer NAO gestures and LED patterns based on `robot_state.txt`.
* Improve the Web UI to show:

  * Live transcript
  * LLM responses
  * State timeline
* Add configuration options for switching models (Grok vs others).
