import os
import time
import json
import random
from dotenv import load_dotenv
from openai import OpenAI

from backends.simulation_backend import SimulationNao
from robot_state import set_state
from audio.asr_whisper import transcribe_from_mic

# ---------------------------------------------------------
# Paths / files
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONTROL_FILE = os.path.join(BASE_DIR, "control_flags.txt")
LOG_FILE = os.path.join(BASE_DIR, "conversation_log.jsonl")
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")

# ---------------------------------------------------------
# Load environment variables from .env
# ---------------------------------------------------------
load_dotenv()

GROK_API_KEY = os.getenv("GROK_API_KEY")
if not GROK_API_KEY:
    raise RuntimeError(
        "GROK_API_KEY is not set. Add it to your .env file."
    )

# ---------------------------------------------------------
# Create Grok client (no key stored in code)
# ---------------------------------------------------------
client = OpenAI(
    api_key=GROK_API_KEY,
    base_url="https://api.x.ai/v1",
)

# ---------------------------------------------------------
# Default settings (must match control_server.py)
# ---------------------------------------------------------
DEFAULT_SETTINGS = {
    "question_difficulty": "hard",          # "easy" | "medium" | "hard"
    "interruption_mode": "random",          # "random" | "fixed"
    "fixed_interval_seconds": 30,           # Used when interruption_mode == "fixed"
    "max_interruptions_per_minute": 3,      # Hard interruptions per 60 sec

    "audience_attitude": "neutral",         # "supportive" | "neutral" | "skeptical" | "hostile"
    "max_aggressiveness": "medium",         # "low" | "medium" | "high"

    "total_session_minutes": 0,             # 0 = unlimited
    "warmup_seconds": 0,                    # No hard interruptions during this period
    "max_speaking_seconds": 15,             # Per turn, passed to ASR
}

# ---------------------------------------------------------
# Settings / control / logging helpers
# ---------------------------------------------------------
def load_settings():
    settings = DEFAULT_SETTINGS.copy()
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                settings.update(data)
    except Exception:
        pass
    return settings


def read_control_flag() -> str:
    try:
        with open(CONTROL_FILE, "r", encoding="utf-8") as f:
            val = f.read().strip()
            return val or "running"
    except IOError:
        return "running"


def write_control_flag(value: str):
    try:
        with open(CONTROL_FILE, "w", encoding="utf-8") as f:
            f.write(value.strip())
    except IOError:
        # Non-fatal
        pass


def append_log(speaker: str, text: str):
    """
    Append a single JSON line: {"speaker": "user"|"robot", "text": ..., "ts": unix_seconds}
    """
    entry = {
        "speaker": speaker,
        "text": text,
        "ts": time.time(),
    }
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except IOError:
        # Non-fatal: UI will just miss this line
        pass

# ---------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------
def build_system_prompt(main_topic,
                        event_type,
                        difficulty,
                        attitude,
                        aggressiveness):
    base = (
        "You are simulating a live crowd listening to a public speaking practice session. "
        "You speak as a single crowd voice, summarizing the group's reaction. "
        "React collectively only with spoken comments or questions strictly related to the speaker’s current talk. "
        "Respond in 1 word to maximum 2 short sentences—nothing longer. "
        "Start directly with the reaction, no greetings or explanations. "
        "Stay in character as audience members, never as a coach or narrator. "
        "Avoid generic filler phrases like 'Interesting', 'Nice', or 'Good point' unless they are part of a more specific reaction. "
    )

    if main_topic:
        topic_part = (
            f"The declared main topic of the talk is: '{main_topic}'. "
            "Assume that different aspects like cost, policy, social impact, or technology details can still be related to this topic; "
            "do not treat them as off-topic just because they don't repeat the exact words of the main topic. "
        )
    else:
        topic_part = ""

    # Difficulty: how cognitively demanding the questions are
    if difficulty == "easy":
        diff_part = (
            "Keep the tone mostly supportive and curious. "
            "Ask simple clarifying questions or make encouraging comments. "
            "Only occasionally challenge the speaker, and do so gently. "
        )
    elif difficulty == "hard":
        diff_part = (
            "Ask more challenging and pointed questions. "
            "Highlight unclear logic, missing explanations, or unstated assumptions. "
            "Push the speaker to defend or deepen their arguments. "
        )
    else:  # "medium"
        diff_part = (
            "Balance supportive reactions with occasional challenging questions. "
            "Alternate between clarification and mild pressure for better explanations. "
        )

    # Audience attitude: emotional stance
    if attitude == "supportive":
        attitude_part = (
            "Your overall attitude is supportive. "
            "You want the speaker to succeed and you sound encouraging, even when you challenge them. "
        )
    elif attitude == "skeptical":
        attitude_part = (
            "Your overall attitude is skeptical. "
            "You often doubt claims, ask for evidence, and point out weaknesses, but you remain respectful. "
        )
    elif attitude == "hostile":
        attitude_part = (
            "Your overall attitude is somewhat hostile, you are an unhinged motherfucker who just wants to humaliate the person talking to you."
            "You are impatient, critical, and quick to question the speaker's claims. "
            "However, you still do not insult the speaker personally and you stay within the role of a tough audience. "
        )
    else:  # "neutral"
        attitude_part = (
            "Your overall attitude is neutral. "
            "You are neither especially supportive nor hostile; you just want clear, convincing arguments. "
        )

    # Aggressiveness: how sharp the challenges are
    if aggressiveness == "low":
        aggr_part = (
            "Keep challenges soft and polite. "
            "Prefer clarifying questions over direct confrontation. "
        )
    elif aggressiveness == "high":
        aggr_part = (
            "Your challenges can be sharp and direct, but still focused on the content of the talk, not personal attacks. "
            "Be willing to call out contradictions or missing justification. "
        )
    else:  # "medium"
        aggr_part = (
            "Use a moderate level of directness in your challenges. "
            "You can question claims clearly, but you are not overly harsh. "
        )

    # Variety + anti-repetition rules
    variety_part = (
        "Very important: avoid repeating the same question template across turns. "
        "Especially avoid repeating generic meta-questions like "
        "'How does this relate to the main topic?' or "
        "'What does that have to do with [main topic]?' or "
        "'How does that connect to why X is true?' in multiple turns. "
        "If you have already asked the speaker to connect something to the main topic recently, "
        "then in this turn you must ask a different kind of question: "
        "focus on specific claims, numbers, examples, assumptions, consequences, or implications instead. "
        "Most of the time, ask about concrete details the speaker just mentioned, "
        "rather than repeating abstract 'connect this' questions. "
    )

    # Event type
    if event_type == "soft_reaction":
        behavior_part = (
            "For this turn, the crowd gives a brief, low-intensity reaction. "
            "This can be a short comment, a brief emotional response, or a light clarifying question. "
            "You are NOT aggressively interrupting; you sound like listeners briefly reacting but letting the speaker continue. "
        )
    elif event_type == "interruption_question":
        behavior_part = (
            "For this turn, the crowd is actively INTERRUPTING the speaker. "
            "Interrupt with a short, sharp, topic-aware question or challenge that forces the speaker to clarify, justify, "
            "or better connect their point to the broader main topic. "
            "However, remember the variety rule: do not reuse the same phrasing you used in your last question, "
            "and prefer questions that dig into specific claims rather than generic 'connect this' prompts. "
        )
    else:
        behavior_part = (
            "Give a short, realistic audience reaction that fits the situation. "
        )

    return base + topic_part + diff_part + attitude_part + aggr_part + variety_part + behavior_part

# ---------------------------------------------------------
# Grok call
# ---------------------------------------------------------
def ask_grok(message,
             main_topic,
             event_type,
             difficulty,
             attitude,
             aggressiveness,
             conversation_history):
    """
    conversation_history: list of {"role": "user"|"assistant", "content": str}
    We feed a short window of recent history to help Grok avoid repetition.
    """
    system_prompt = build_system_prompt(
        main_topic=main_topic,
        event_type=event_type,
        difficulty=difficulty,
        attitude=attitude,
        aggressiveness=aggressiveness,
    )

    # Use the last few turns to give Grok context and prevent repetition
    # (e.g., last 6 messages: user/assistant/user/assistant/...)
    history_window = conversation_history[-6:]

    messages = [{"role": "system", "content": system_prompt}]
    for m in history_window:
        if m["role"] == "user":
            messages.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            messages.append({"role": "assistant", "content": m["content"]})

    messages.append({"role": "user", "content": message})

    try:
        completion = client.chat.completions.create(
            model="grok-2-latest",
            messages=messages,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred while contacting Grok: {e}"

# ---------------------------------------------------------
# Interruption decision logic (time-based + rate limit)
# ---------------------------------------------------------
def decide_event(settings,
                 now,
                 start_time,
                 last_interrupt_time,
                 interrupt_timestamps):
    """
    Decide if this turn should:
      - say nothing ("no_reaction")
      - (optionally) give a mild reaction ("soft_reaction")
      - interrupt with a sharp question ("interruption_question")

    Uses:
      - interruption_mode: "fixed" | "random"
      - fixed_interval_seconds
      - max_interruptions_per_minute
      - warmup_seconds
    """
    warmup_seconds = settings.get("warmup_seconds", 0)
    mode = settings.get("interruption_mode", "random")
    fixed_interval = settings.get("fixed_interval_seconds", 30)
    max_per_min = settings.get("max_interruptions_per_minute", 3)

    # During warm-up: no hard interruptions
    if now - start_time < warmup_seconds:
        return "no_reaction", last_interrupt_time, interrupt_timestamps

    # Clean up old interruptions beyond 60 seconds
    interrupt_timestamps = [
        t for t in interrupt_timestamps if now - t <= 60.0
    ]

    # Rate limit
    if max_per_min > 0 and len(interrupt_timestamps) >= max_per_min:
        return "no_reaction", last_interrupt_time, interrupt_timestamps

    # Decide based on mode
    if mode == "fixed":
        # Interrupt if enough time passed since last interrupt and rate allows it
        if (now - last_interrupt_time) >= fixed_interval:
            last_interrupt_time = now
            interrupt_timestamps.append(now)
            return "interruption_question", last_interrupt_time, interrupt_timestamps
        else:
            return "no_reaction", last_interrupt_time, interrupt_timestamps
    else:
        # Random mode: basic probability, with a minimal gap
        min_gap = 5.0
        if now - last_interrupt_time < min_gap:
            return "no_reaction", last_interrupt_time, interrupt_timestamps

        # Probability can depend slightly on difficulty/aggressiveness if you want;
        # for now keep it simple.
        if random.random() < 0.4:
            last_interrupt_time = now
            interrupt_timestamps.append(now)
            return "interruption_question", last_interrupt_time, interrupt_timestamps
        else:
            return "no_reaction", last_interrupt_time, interrupt_timestamps

# ---------------------------------------------------------
# Main conversation loop
# ---------------------------------------------------------
def main():
    # Initial control flag: running (in case file missing)
    write_control_flag("running")

    # Load settings once at start
    settings = load_settings()
    difficulty = settings.get("question_difficulty", "hard")
    audience_attitude = settings.get("audience_attitude", "neutral")
    max_aggressiveness = settings.get("max_aggressiveness", "medium")
    max_speaking_seconds = settings.get("max_speaking_seconds", 15)
    total_session_minutes = settings.get("total_session_minutes", 0)

    robot = SimulationNao()

    # Conversation state
    conversation_history = []  # list of {"role": "user"/"assistant", "content": str}
    main_topic = None

    start_time = time.time()
    last_interrupt_time = start_time
    interrupt_timestamps = []  # list of timestamps of hard interruptions

    # Greeting
    set_state("speaking")
    greeting = "Hello, I am ready to talk with you. You can speak whenever you are ready."
    robot.speak(greeting)
    append_log("robot", greeting)
    set_state("idle")

    # Main loop
    while True:
        # Check control flag first (for pause/end from UI)
        flag = read_control_flag()
        if flag == "end":
            set_state("speaking")
            goodbye = "Okay, ending the conversation now. Goodbye."
            robot.speak(goodbye)
            append_log("robot", goodbye)
            set_state("idle")
            break

        if flag == "paused":
            set_state("idle")
            time.sleep(0.5)
            continue

        # Check session length (minutes)
        if total_session_minutes and total_session_minutes > 0:
            elapsed = time.time() - start_time
            if elapsed >= total_session_minutes * 60:
                set_state("speaking")
                msg = "The session time is over. Thank you for speaking."
                robot.speak(msg)
                append_log("robot", msg)
                set_state("idle")
                break

        # Listen
        set_state("listening")
        user_text = transcribe_from_mic(
            duration_sec=float(max_speaking_seconds),
            language="en"
        )

        if not user_text:
            set_state("idle")
            continue

        now = time.time()

        print("[USER SAID]:", user_text)
        append_log("user", user_text)
        conversation_history.append({"role": "user", "content": user_text})

        # First meaningful sentence defines main_topic (if not already set)
        if main_topic is None and len(user_text.split()) >= 3:
            main_topic = user_text.strip()
            print("[MAIN TOPIC SET TO]:", main_topic)

        # Termination keywords (voice-based)
        cleaned = user_text.lower().strip()
        if cleaned in {"end conversation", "stop", "shutdown", "goodbye"}:
            set_state("speaking")
            msg = "Okay, ending the conversation now. Goodbye."
            robot.speak(msg)
            append_log("robot", msg)
            set_state("idle")
            break

        # Decide what happens this turn: silence or interruption
        event_type, last_interrupt_time, interrupt_timestamps = decide_event(
            settings=settings,
            now=now,
            start_time=start_time,
            last_interrupt_time=last_interrupt_time,
            interrupt_timestamps=interrupt_timestamps,
        )

        if event_type == "no_reaction":
            # Crowd stays silent: robot does not speak this turn
            set_state("idle")
            continue

        # Thinking
        set_state("thinking")
        reply = ask_grok(
            message=user_text,
            main_topic=main_topic,
            event_type=event_type,
            difficulty=difficulty,
            attitude=audience_attitude,
            aggressiveness=max_aggressiveness,
            conversation_history=conversation_history,
        )

        # Log robot reply immediately, before speaking (for faster UI)
        append_log("robot", reply)
        conversation_history.append({"role": "assistant", "content": reply})

        # Speaking
        set_state("speaking")
        robot.speak(reply)

        # Back to idle
        set_state("idle")
        time.sleep(0.2)


# ---------------------------------------------------------
if __name__ == "__main__":
    main()
