import time
import os
import json
import base64
import asyncio
import websockets
import csv
import threading
import tempfile
import subprocess
import openai
from datetime import datetime
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from fastapi.websockets import WebSocketDisconnect
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Connect
from dotenv import load_dotenv

load_dotenv()

# Define the CSV file path and logging function
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
CSV_LOG_FILE = os.path.join(BASE_DIR, "realtime_logs.csv")

def log_to_csv(event_type, event_data):
    file_exists = os.path.exists(CSV_LOG_FILE)
    with open(CSV_LOG_FILE, "a", newline="") as csvfile:
        fieldnames = ["timestamp", "event_type", "event_data"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "event_data": json.dumps(event_data)
        })

def load_prompt(file_name):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    prompt_path = os.path.join(dir_path, 'prompts', f'{file_name}.txt')
    try:
        with open(prompt_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Could not find file: {prompt_path}")
        raise

# --- Conversion Helper Function Using ffmpeg ---
def convert_ulaw_to_wav(audio_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as raw_file:
        raw_file.write(audio_bytes)
        raw_file.flush()
        raw_filename = raw_file.name

    wav_filename = raw_filename + ".wav"
    try:
        subprocess.run([
            "ffmpeg",
            "-y",
            "-f", "mulaw",
            "-ar", "8000",
            "-i", raw_filename,
            "-acodec", "pcm_s16le",
            wav_filename
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        print("Error converting audio using ffmpeg:", e)
        os.remove(raw_filename)
        return None

    os.remove(raw_filename)
    return wav_filename

# --- Actual Transcription Function using OpenAI Whisper API ---
def transcribe_audio(audio_segments: list) -> str:
    try:
        combined_audio = b"".join([base64.b64decode(segment) for segment in audio_segments])
        wav_filename = convert_ulaw_to_wav(combined_audio)
        if not wav_filename:
            return ""
        try:
            with open(wav_filename, "rb") as audio_file:
                transcript_response = openai.Audio.transcribe("whisper-1", audio_file)
                transcript = transcript_response["text"]
        except Exception as e:
            print("Error during transcription via OpenAI API:", e)
            transcript = ""
        finally:
            os.remove(wav_filename)
        return transcript
    except Exception as e:
        print("General error in transcribe_audio:", e)
        return ""

# --- Function to generate horoscope (example tool) ---
def generate_horoscope(sign: str) -> str:
    horoscope = f"Today, {sign}, you will encounter a delightful surprise that brightens your day!"
    return horoscope

# --- Global Configuration ---
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # requires Whisper API access
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')
NGROK_URL = os.getenv('NGROK_URL')
PORT = int(os.getenv('PORT', 5050))

SYSTEM_MESSAGE = load_prompt('system_prompt')
VOICE = 'sage'
LOG_EVENT_TYPES = [
    'response.content.done', 'rate_limits.updated', 'response.done',
    'input_audio_buffer.committed', 'input_audio_buffer.speech_stopped',
    'input_audio_buffer.speech_started', 'session.created'
]

app = FastAPI()

# Global variables for Twilio and call management
TWILIO_CLIENT = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
CURRENT_CALL_SID = None
CALL_ENDED_BY_FUNCTION = False  # flag to allow immediate next call

if not OPENAI_API_KEY:
    raise ValueError('Missing the OpenAI API key. Please set it in the .env file.')
if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN or not TWILIO_PHONE_NUMBER:
    raise ValueError('Missing Twilio configuration. Please set it in the .env file.')

@app.get("/", response_class=HTMLResponse)
async def index_page():
    return {"message": "Twilio Media Stream Server is running!"}

@app.post("/make-call")
async def make_call(request: Request):
    data = await request.json()
    to_phone_number = data.get("to")
    if not to_phone_number:
        return {"error": "Phone number is required"}
    call = TWILIO_CLIENT.calls.create(
        url=f"{NGROK_URL}/outgoing-call",
        to=to_phone_number,
        from_=TWILIO_PHONE_NUMBER
    )
    return {"call_sid": call.sid}

# Endpoint to manually end a call
@app.post("/end-call")
async def end_call_endpoint(request: Request):
    end_call()
    return {"message": "Call ended if it was active."}

# Outgoing call endpoint that connects directly to the media stream.
@app.api_route("/outgoing-call", methods=["GET", "POST"])
async def handle_outgoing_call(request: Request):
    response = VoiceResponse()
    connect = Connect()
    connect.stream(url=f'wss://{request.url.hostname}/media-stream')
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    print("Client connected")
    await websocket.accept()
    assistant_talking = False
    user_speaking = False

    try:
        async with websockets.connect(
            'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01',
            extra_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta": "realtime=v1"
            }
        ) as openai_ws:
            await send_session_update(openai_ws)
            stream_sid = None
            session_id = None

            async def receive_from_twilio():
                nonlocal stream_sid, user_speaking, assistant_talking
                user_audio_segments = []
                try:
                    async for message in websocket.iter_text():
                        data = json.loads(message)
                        event_type = data.get('event')
                        if event_type == 'start':
                            stream_sid = data['start'].get('streamSid')
                            print(f"Incoming stream has started: {stream_sid}")
                            log_to_csv("stream_started", {"streamSid": stream_sid})
                        elif event_type == 'input_audio_buffer.speech_started':
                            user_speaking = True
                            user_audio_segments = []
                            print("User started speaking")
                            log_to_csv("user.speech_started", data)
                            if assistant_talking:
                                cancel_event = {
                                    "event_id": "cancel_on_interrupt",
                                    "type": "response.cancel"
                                }
                                await openai_ws.send(json.dumps(cancel_event))
                        elif event_type == 'input_audio_buffer.speech_stopped':
                            user_speaking = False
                            print("User stopped speaking")
                            log_to_csv("user.speech_stopped", data)
                            if user_audio_segments:
                                transcript = transcribe_audio(user_audio_segments)
                                print("Audio transcript:", transcript)
                                log_to_csv("user.speech.transcript", {"streamSid": stream_sid, "transcript": transcript})
                            else:
                                print("No audio segments captured for transcription.")
                        elif event_type == 'media' and openai_ws.open:
                            payload = data.get('media', {}).get('payload')
                            if user_speaking and payload:
                                user_audio_segments.append(payload)
                            if assistant_talking:
                                await asyncio.sleep(0.5)
                            audio_append = {
                                "type": "input_audio_buffer.append",
                                "audio": payload
                            }
                            await openai_ws.send(json.dumps(audio_append))
                except WebSocketDisconnect:
                    print("Client disconnected.")
                    log_to_csv("client_disconnected", {"message": "Client disconnected"})
                    if openai_ws.open:
                        await openai_ws.close()
                except Exception as e:
                    print(f"Error in receive_from_twilio: {e}")
                    log_to_csv("error_receive", {"error": str(e)})

            async def send_to_twilio():
                nonlocal stream_sid, session_id, assistant_talking
                try:
                    async for openai_message in openai_ws:
                        response_msg = json.loads(openai_message)
                        event_type = response_msg.get('type')
                        if event_type in LOG_EVENT_TYPES:
                            print(f"Received event: {event_type}", response_msg)
                            log_to_csv(event_type, response_msg)
                        if event_type == 'session.created':
                            session_id = response_msg['session']['id']
                        elif event_type == 'session.updated':
                            print("Session updated successfully:", response_msg)
                            log_to_csv("session.updated", response_msg)
                        elif event_type == 'response.audio.delta' and response_msg.get('delta'):
                            if not assistant_talking:
                                assistant_talking = True
                            try:
                                audio_payload = base64.b64encode(
                                    base64.b64decode(response_msg['delta'])
                                ).decode('utf-8')
                                audio_delta = {
                                    "event": "media",
                                    "streamSid": stream_sid,
                                    "media": {"payload": audio_payload}
                                }
                                await websocket.send_json(audio_delta)
                            except Exception as e:
                                print(f"Error processing audio data: {e}")
                                log_to_csv("error_audio", {"error": str(e), "data": response_msg})
                        elif event_type == 'response.done':
                            outputs = response_msg.get("response", {}).get("output", [])
                            if outputs and outputs[0].get("type") == "function_call":
                                func_call = outputs[0]
                                func_name = func_call.get("name")
                                call_id = func_call.get("call_id")
                                arguments = func_call.get("arguments")
                                print(f"Model requested function call: {func_name} with arguments {arguments}")
                                if func_name == "connect_with_recruiter":
                                    try:
                                        _ = json.loads(arguments) if arguments else {}
                                        # End the call via Twilio API
                                        end_call()
                                        # Send function output event back to the realtime API
                                        function_output = {
                                            "type": "conversation.item.create",
                                            "item": {
                                                "type": "function_call_output",
                                                "call_id": call_id,
                                                "output": json.dumps({"message": "Call ended. Connecting you with a recruiter."})
                                            }
                                        }
                                        await openai_ws.send(json.dumps(function_output))
                                        await openai_ws.send(json.dumps({"type": "response.create"}))
                                    except Exception as e:
                                        print("Error handling connect_with_recruiter function call:", e)
                                        log_to_csv("function_call_error", {"error": str(e)})
                                elif func_name == "generate_horoscope":
                                    try:
                                        args = json.loads(arguments)
                                        sign = args.get("sign")
                                        horoscope = generate_horoscope(sign)
                                        function_output = {
                                            "type": "conversation.item.create",
                                            "item": {
                                                "type": "function_call_output",
                                                "call_id": call_id,
                                                "output": json.dumps({"horoscope": horoscope})
                                            }
                                        }
                                        await openai_ws.send(json.dumps(function_output))
                                        await openai_ws.send(json.dumps({"type": "response.create"}))
                                    except Exception as e:
                                        print("Error handling function call:", e)
                                        log_to_csv("function_call_error", {"error": str(e)})
                            else:
                                print("Assistant finished speaking.")
                                log_to_csv("response.done", response_msg)
                                await asyncio.sleep(1)
                                assistant_talking = False
                        elif event_type == 'conversation.item.created':
                            print(f"conversation.item.created event: {response_msg}")
                            log_to_csv("conversation.item.created", response_msg)
                except Exception as e:
                    print(f"Error in send_to_twilio: {e}")
                    log_to_csv("error_in_send_to_twilio", {"error": str(e)})

            await asyncio.gather(receive_from_twilio(), send_to_twilio())
    except Exception as outer_e:
        print(f"Critical error in media stream handling: {outer_e}")
        log_to_csv("critical_error", {"error": str(outer_e)})
        await websocket.close()

async def send_session_update(openai_ws):
    session_update = {
        "type": "session.update",
        "session": {
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "voice": VOICE,
            "instructions": SYSTEM_MESSAGE,
            "modalities": ["text", "audio"],
            "temperature": 0.8,
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 500,
                "create_response": True
            },
            "tools": [
                {
                    "type": "function",
                    "name": "generate_horoscope",
                    "description": "Give today's horoscope for an astrological sign.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sign": {
                                "type": "string",
                                "description": "The sign for the horoscope.",
                                "enum": [
                                    "Aries", "Taurus", "Gemini", "Cancer",
                                    "Leo", "Virgo", "Libra", "Scorpio",
                                    "Sagittarius", "Capricorn", "Aquarius", "Pisces"
                                ]
                            }
                        },
                        "required": ["sign"]
                    }
                },
                {
                    "type": "function",
                    "name": "connect_with_recruiter",
                    "description": "Ends the current call and connects the user with a recruiter, but only after saying you will connect them with a recruiter. If you are sent to voicemail, leave the user a voicemaill and then end the call after.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            ],
            "tool_choice": "auto"
        }
    }
    print('Sending session update:', json.dumps(session_update))
    log_to_csv("session.update", session_update)
    await openai_ws.send(json.dumps(session_update))

def end_call():
    """
    Ends the current Twilio call by updating its status to 'completed'
    and sets a flag so the console loop can immediately accept a new number.
    """
    global CURRENT_CALL_SID, CALL_ENDED_BY_FUNCTION
    if CURRENT_CALL_SID:
        try:
            updated_call = TWILIO_CLIENT.calls(CURRENT_CALL_SID).update(status="completed")
            print(f"Call {CURRENT_CALL_SID} ended successfully with status {updated_call.status}.")
            log_to_csv("end_call", {"status": updated_call.status, "call_sid": CURRENT_CALL_SID})
            CURRENT_CALL_SID = None
            CALL_ENDED_BY_FUNCTION = True
        except Exception as e:
            print(f"Error ending call: {e}")
            log_to_csv("end_call_error", {"error": str(e)})
    else:
        print("No active call to end.")
        log_to_csv("end_call", {"message": "No active call"})

# ----- Main Execution -----
if __name__ == "__main__":
    import uvicorn

    def run_server():
        uvicorn.run(app, host="0.0.0.0", port=PORT)

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    while True:
        to_phone_number = input("Please enter the phone number to call (or press Enter to exit): ").strip()
        if not to_phone_number:
            print("No phone number entered. Exiting.")
            break
        try:
            call = TWILIO_CLIENT.calls.create(
                url=f"{NGROK_URL}/outgoing-call",
                to=to_phone_number,
                from_=TWILIO_PHONE_NUMBER
            )
            CURRENT_CALL_SID = call.sid
            print(f"Call initiated with SID: {call.sid}")
            # Poll the call status until it ends, or if ended by function call.
            while True:
                updated_call = TWILIO_CLIENT.calls(call.sid).fetch()
                if updated_call.status in ["completed", "canceled", "failed"] or CALL_ENDED_BY_FUNCTION:
                    print("Call ended.")
                    CALL_ENDED_BY_FUNCTION = False  # reset flag for next call
                    break
                time.sleep(5)
        except Exception as e:
            print(f"Error initiating call: {e}")