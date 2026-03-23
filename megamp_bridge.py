"""
MegAMP Bridge — Firebase <-> Ollama
Reads device state from Firebase Realtime Database,
sends to llama3 via Ollama, writes AI response back to Firebase.

Usage:
    cd /home/sanjaynb/megamp
    python megamp_bridge.py

Requirements:
    pip install firebase-admin requests
"""

import json
import time
import requests
import firebase_admin
from firebase_admin import credentials, db

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — all pre-filled with your credentials
# ─────────────────────────────────────────────────────────────────────────────
SERVICE_ACCOUNT = "/home/sanjaynb/megamp/serviceAccount.json"
DATABASE_URL    = "https://megamp-29e9e-default-rtdb.asia-southeast1.firebasedatabase.app"
OLLAMA_URL      = "http://localhost:11434/api/generate"
OLLAMA_MODEL    = "llama3"
POLL_INTERVAL   = 5   # seconds between checks

# ─────────────────────────────────────────────────────────────────────────────
# INIT FIREBASE
# ─────────────────────────────────────────────────────────────────────────────
cred = credentials.Certificate(SERVICE_ACCOUNT)
firebase_admin.initialize_app(cred, {"databaseURL": DATABASE_URL})

state_ref   = db.reference("/state")        # browser writes device state here
ai_req_ref  = db.reference("/ai_request")   # browser writes AI trigger here
ai_resp_ref = db.reference("/ai_response")  # we write AI reply here
ai_log_ref  = db.reference("/ai_log")       # we write log entries here

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an AI energy manager for an Indian smart home in Bengaluru. "
    "Device power (BEE certified, 230V/50Hz): "
    "light=9W (Havells LED), fan=28W (Havells BLDC), "
    "ac=1250W (Voltas 1.5T 5-star), geyser=2000W (Havells 15L). "
    "Battery: Sodium-Ion 10kWh. Solar: Tata Power array. "
    "Grid: BESCOM tariff Rs7.5/kWh. "
    "Battery ONLY discharges during power cut (solar OFF + main power OFF). "
    "Grid NEVER charges battery."
)

# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA
# ─────────────────────────────────────────────────────────────────────────────
def check_ollama():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        return r.ok
    except Exception:
        return False


def ask_ollama(prompt, system=""):
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    try:
        res = requests.post(OLLAMA_URL, json={
            "model":   OLLAMA_MODEL,
            "prompt":  full_prompt,
            "stream":  False,
            "options": {"temperature": 0.25, "num_predict": 400}
        }, timeout=30)
        res.raise_for_status()
        return res.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        print("  ⚠  Ollama not running — start with: ollama serve")
        return None
    except requests.exceptions.Timeout:
        print("  ⚠  Ollama timed out after 30s")
        return None
    except Exception as e:
        print(f"  ⚠  Ollama error: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# BUILD PROMPT FROM FIREBASE STATE
# ─────────────────────────────────────────────────────────────────────────────
def build_prompt(state, mode, request_type):
    battery  = state.get("battery", 0)
    solar_w  = state.get("solarW", 0)
    load_w   = state.get("loadW", 0)
    sun_pos  = state.get("sunPosition", 0)
    is_day   = state.get("isDay", False)
    main_pwr = state.get("mainPower", True)
    batt_st  = state.get("battState", "idle")
    solar_on = state.get("solar", True)

    # Firebase stores __ instead of : in device keys
    raw_devs    = state.get("devices", {})
    on_devs     = [k.replace("__", ":") for k, v in raw_devs.items() if v]
    on_devs_str = ", ".join(on_devs) if on_devs else "none"

    solar_status = "ON" if (is_day and solar_on) else "OFF"

    if request_type == "mode_change":
        prompt = (
            f"Mode changed to: {mode}\n"
            f"Battery: {battery:.1f}% ({batt_st})\n"
            f"Solar: {solar_status} at {solar_w/1000:.2f}kW\n"
            f"Main grid power: {'ON' if main_pwr else 'OFF — power cut'}\n"
            f"Total house load: {load_w/1000:.2f}kW\n"
            f"Sun intensity: {sun_pos}%\n"
            f"Active devices: {on_devs_str}\n\n"
            "Reply with ONLY valid JSON — no markdown, no explanation:\n"
            '{"summary": "<1 sentence max 12 words>", '
            '"decisions": ["<action max 10 words>", "<action max 10 words>", "<action max 10 words>"]}'
        )
    else:
        prompt = (
            f"Battery {battery:.1f}% ({batt_st}), "
            f"solar {solar_status} at {solar_w/1000:.2f}kW, "
            f"load {load_w/1000:.2f}kW, "
            f"grid {'on' if main_pwr else 'OFF'}, "
            f"mode: {mode}, sun: {sun_pos}%, Bengaluru. "
            "Give ONE concise energy insight under 15 words. "
            "No quotes. No punctuation at end."
        )

    return prompt

# ─────────────────────────────────────────────────────────────────────────────
# PUSH AI RESPONSE TO FIREBASE
# ─────────────────────────────────────────────────────────────────────────────
def push_response(raw_text, request_type, mode):
    ts = int(time.time() * 1000)

    if request_type == "mode_change":
        try:
            clean  = raw_text.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean)
            payload = {
                "type":      "mode_change",
                "mode":      mode,
                "summary":   parsed.get("summary", raw_text[:120]),
                "decisions": parsed.get("decisions", [])[:3],
                "timestamp": ts
            }
        except (json.JSONDecodeError, ValueError):
            payload = {
                "type":      "mode_change",
                "mode":      mode,
                "summary":   raw_text[:120],
                "decisions": [],
                "timestamp": ts
            }
    else:
        payload = {
            "type":      "insight",
            "message":   raw_text,
            "timestamp": ts
        }

    ai_resp_ref.set(payload)
    ai_log_ref.push({
        "msg":       f"🤖 {raw_text[:200]}",
        "type":      "decision",
        "timestamp": ts
    })
    print(f"  → Firebase ✓  {raw_text[:80]}...")


def push_fallback(request_type, mode):
    ai_resp_ref.set({
        "type":      request_type,
        "mode":      mode,
        "summary":   "AI offline — built-in simulation active",
        "decisions": [],
        "timestamp": int(time.time() * 1000)
    })

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  🔥  MegAMP Bridge — Firebase ↔ Ollama")
    print(f"  Firebase : {DATABASE_URL}")
    print(f"  Ollama   : {OLLAMA_URL}")
    print(f"  Model    : {OLLAMA_MODEL}")
    print(f"  Polling  : every {POLL_INTERVAL}s")
    print("=" * 55)

    if check_ollama():
        print("  ✅  Ollama is running\n")
    else:
        print("  ⚠   Ollama not running")
        print("      Start it with: ollama serve")
        print("      Bridge will keep trying...\n")

    last_request_id = None

    while True:
        try:
            req = ai_req_ref.get()

            if req and req.get("id") != last_request_id:
                req_id          = req.get("id")
                request_type    = req.get("type", "insight")
                mode            = req.get("mode", "normal")
                last_request_id = req_id

                print(f"\n[{time.strftime('%H:%M:%S')}] ← Browser: type={request_type}  mode={mode}")

                state = state_ref.get() or {}
                print(f"  State — Battery: {state.get('battery', 0):.1f}%  "
                      f"Solar: {state.get('solarW', 0)/1000:.2f}kW  "
                      f"Load: {state.get('loadW', 0)/1000:.2f}kW  "
                      f"BattState: {state.get('battState', '?')}")

                prompt   = build_prompt(state, mode, request_type)
                response = ask_ollama(prompt, SYSTEM_PROMPT)

                if response:
                    push_response(response, request_type, mode)
                else:
                    push_fallback(request_type, mode)
                    print("  ⚠  Fallback pushed")

        except KeyboardInterrupt:
            print("\n\n  Stopped. Goodbye!\n")
            break
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Error: {e}")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
