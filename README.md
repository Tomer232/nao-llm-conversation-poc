# NAO LLM Conversation System (POC)

This project integrates a NAO robot (or a simulation) with an LLM to enable full spoken conversations.  
The system includes:
- Speech recognition (ASR)
- LLM response generation
- Text-to-speech output (robot speaker or PC speaker)
- Simulation mode and real robot mode
- State indicator (`idle`, `listening`, `thinking`, `speaking`)

## How to Run (Simulation Mode)
1. Create a virtual environment:
python -m venv venv
venv\Scripts\activate

2. Install dependencies:
pip install -r requirements.txt

3. Create a `.env` file in the project root:
GROK_API_KEY=your_key_here

4. Start the system:
python conversation_loop.py

## Notes
- `.env` is ignored (not uploaded to GitHub).
- `robot_state.txt` is also ignored and created automatically.
- Works out-of-the-box in simulation mode using your PC microphone and speakers.
