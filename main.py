# main.py

import eel
import soundcard as sc
import threading
import queue
import logging
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import numpy as np
import torch
import torch.nn as nn
import librosa
import ollama
import pythoncom  # Import pywin32's COM module
import json
import os
import warnings
import time

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=UserWarning, module='transformers')
warnings.filterwarnings("ignore", category=RuntimeWarning, module='soundcard')

# Constants
SETTINGS_FILE = 'settings.json'
SUPPORTED_LANGUAGES = ["English"]  # Fixed to English for transcription

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Eel
eel.init('web')

# Function to pad or truncate mel features
def pad_or_truncate(input_features, target_length=3000):
    current_length = input_features.shape[-1]
    if current_length > target_length:
        input_features = input_features[..., :target_length]
    elif current_length < target_length:
        pad_width = target_length - current_length
        input_features = torch.nn.functional.pad(input_features, (0, pad_width))
    return input_features

def detect_silence(audio, samplerate, frame_duration=0.05, energy_threshold=0.05, silence_duration=0.5):
    frame_length = int(frame_duration * samplerate)
    hop_length = frame_length  # Non-overlapping frames

    energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    silent_frames = energy < energy_threshold
    min_silence_frames = int(silence_duration / frame_duration)

    for i in range(len(silent_frames) - min_silence_frames, -1, -1):
        if all(silent_frames[i:i + min_silence_frames]):
            split_point = i * hop_length
            return split_point
    return None

# --------------------
# Settings Class
# --------------------
class Settings:
    def __init__(self, filepath=SETTINGS_FILE):
        self.filepath = filepath
        self.default_settings = {
            "selected_device": "cpu",
            "selected_audio_source": 0,
            "selected_ollama_model": ""  # New setting added
        }
        self.settings = self.load_settings()

    def load_settings(self):
        if not os.path.exists(self.filepath):
            return self.default_settings.copy()
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            # Ensure all default keys are present
            for key, value in self.default_settings.items():
                if key not in settings:
                    settings[key] = value
            return settings
        except Exception as e:
            logging.error(f"Error loading settings: {e}")
            return self.default_settings.copy()

    def save_settings(self):
        try:
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=4)
            logging.info("Settings saved successfully.")
        except Exception as e:
            logging.error(f"Error saving settings: {e}")

    def get_setting(self, key):
        return self.settings.get(key, self.default_settings.get(key))

    def set_setting(self, key, value):
        self.settings[key] = value
        self.save_settings()

# --------------------
# AudioRecorder Class
# --------------------
class AudioRecorder(threading.Thread):
    def __init__(self, mic, source_type, voice_activity_event, is_muted_event, samplerate=16000, chunk_duration=0.5, audio_queue=None):
        super().__init__()
        self.mic = mic
        self.source_type = source_type  # 'Microphone' or 'System Audio'
        self.samplerate = samplerate
        self.chunk_size = int(samplerate * chunk_duration)
        self.running = False
        self.audio_queue = audio_queue
        self.voice_activity_event = voice_activity_event  # Shared event
        self.is_muted_event = is_muted_event  # Event to handle mute functionality
        self.energy_threshold = 0.02  # Adjust as needed
        self.silence_duration = 0.5  # Duration in seconds to consider as silence
        self.last_voice_time = None

    def run(self):
        pythoncom.CoInitialize()  # Initialize COM for this thread
        self.running = True
        try:
            with self.mic.recorder(samplerate=self.samplerate) as recorder:
                while self.running:
                    try:
                        data = recorder.record(numframes=self.chunk_size)
                        # If stereo, convert to mono
                        if len(data.shape) == 2 and data.shape[1] == 2:
                            data = data.mean(axis=1)

                        # Skip processing if muted
                        if self.is_muted_event.is_set():
                            continue

                        if self.audio_queue and not self.audio_queue.full():
                            self.audio_queue.put((data.copy(), self.source_type))

                        # Voice Activity Detection
                        energy = np.mean(np.abs(data))
                        if energy > self.energy_threshold:
                            if not self.voice_activity_event.is_set():
                                logging.debug("Voice activity detected.")
                            self.voice_activity_event.set()
                            self.last_voice_time = time.time()
                        else:
                            if self.last_voice_time and (time.time() - self.last_voice_time) > self.silence_duration:
                                if self.voice_activity_event.is_set():
                                    logging.debug("Voice activity ended.")
                                self.voice_activity_event.clear()
                                self.last_voice_time = None
                    except sc.exceptions.SoundcardRuntimeWarning:
                        logging.warning("Data discontinuity in recording.")
                        continue
                    except Exception as e:
                        logging.error(f"Recording error: {e}")
                        eel.display_error(f"Recording error: {e}")
                        continue
        except Exception as e:
            logging.error(f"Failed to initialize recorder: {e}")
            eel.display_error(f"Failed to initialize recorder: {e}")
        finally:
            pythoncom.CoUninitialize()  # Uninitialize COM before thread exits

    def stop(self):
        self.running = False

# --------------------
# AIResponseHandler Class
# --------------------
class AIResponseHandler(threading.Thread):
    def __init__(self, ollama_client, selected_ollama_model, device, voice_activity_event, lock, conversation_history):
        super().__init__()
        self.ollama_client = ollama_client
        self.selected_ollama_model = selected_ollama_model
        self.device = device
        self.voice_activity_event = voice_activity_event
        self.lock = lock
        self.conversation_history = conversation_history
        self.stop_event = threading.Event()
        self.ai_response = ""
        self.current_response_cancelled = False
        self.cancel_lock = threading.Lock()
        self.new_prompt_event = threading.Event()

    def handle_new_prompt(self):
        with self.cancel_lock:
            self.current_response_cancelled = True  # Cancel any ongoing response
        self.new_prompt_event.set()

    def run(self):
        while not self.stop_event.is_set():
            # Wait for a new prompt
            if not self.new_prompt_event.wait(timeout=0.1):
                continue  # No new prompt, continue waiting

            self.new_prompt_event.clear()
            self.ai_response = ""

            # Before starting to process the new prompt, reset current_response_cancelled
            with self.cancel_lock:
                self.current_response_cancelled = False

            # Wait until voice_activity_event is cleared before starting AI response
            if self.voice_activity_event.is_set():
                logging.info("Waiting for user to finish speaking before starting AI response.")
                start_wait_time = time.time()
                while self.voice_activity_event.is_set() and not self.stop_event.is_set():
                    if time.time() - start_wait_time > 30:
                        logging.info("User has been speaking for more than 30 seconds. Not starting AI response.")
                        break  # Do not start AI response
                    time.sleep(0.1)
                if self.voice_activity_event.is_set():
                    continue  # Skip starting AI response

            # Fetch the latest conversation history
            with self.lock:
                prompt_messages = self.conversation_history.copy()

            try:
                response_interrupted = False  # Flag to indicate if response was interrupted
                ai_message_started = False  # Flag to check if we have started the AI message
                response = self.ollama_client.chat(
                    model=self.selected_ollama_model,
                    messages=prompt_messages,
                    stream=True
                )
                for token_obj in response:
                    if self.stop_event.is_set():
                        logging.info("AI response interrupted by user.")
                        break
                    # Check for cancellation
                    with self.cancel_lock:
                        if self.current_response_cancelled:
                            logging.info("AI response cancelled due to new prompt.")
                            response_interrupted = True
                            break

                    if token_obj is None:
                        break
                    logging.debug(f"Received token object: {token_obj}")

                    # Check if voice activity is ongoing
                    if self.voice_activity_event.is_set():
                        logging.info("Voice activity detected. Stopping AI response.")
                        response_interrupted = True
                        break  # Stop the AI response immediately

                    # Extract 'content' from 'message'
                    token_text = ''
                    if isinstance(token_obj, dict):
                        token_text = token_obj.get('message', {}).get('content', '')
                    elif isinstance(token_obj, str):
                        token_text = token_obj

                    if token_text:
                        if not ai_message_started:
                            eel.add_message('AI', token_text)
                            ai_message_started = True
                        else:
                            eel.update_last_message(token_text)
                        self.ai_response += token_text
                    else:
                        logging.warning(f"Unexpected token type: {type(token_obj)} - {token_obj}")

                # After response is done or interrupted
                if self.ai_response.strip():
                    with self.lock:
                        self.conversation_history.append({"role": "assistant", "content": self.ai_response.strip()})
                    logging.info(f"AI Response: {self.ai_response.strip()}")
                    print(f"AI Response: {self.ai_response.strip()}")  # Print to console for debugging
            except Exception as e:
                logging.error(f"Error in AIResponseHandler: {e}")
                eel.display_error(f"Error in AI response: {e}")

    def stop(self):
        self.stop_event.set()
        with self.cancel_lock:
            self.current_response_cancelled = True

# --------------------
# TranscriptionWorker Class
# --------------------
class TranscriptionWorker(threading.Thread):
    def __init__(self, audio_queue, ai_response_handler, device='cpu', samplerate=16000):
        super().__init__()
        self.audio_queue = audio_queue
        self.running = False
        self.samplerate = samplerate
        self.device = device  # Device string, e.g., 'cpu', 'cuda'
        self.ollama_client = None
        self.conversation_history = []
        self.lock = threading.Lock()
        self.ai_response_handler = ai_response_handler  # Reference to AIResponseHandler

        # Initialize the Transformers model and processor
        try:
            model_id = "openai/whisper-large-v3-turbo"  # Using whisper turbo model
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
            self.model.to(self.device)
            
            if self.device.startswith('cuda'):
                if torch.cuda.device_count() > 1:
                    device_ids = list(range(torch.cuda.device_count()))
                    self.model = nn.DataParallel(self.model, device_ids=device_ids)
                    logging.info(f"Using GPUs: {device_ids}")
                else:
                    logging.info("Using single GPU.")
            else:
                logging.info("Using CPU for transcription.")
            
            self.model.eval()  # Set model to evaluation mode
            if self.device.startswith('cuda'):
                self.model.half()  # Use mixed precision if GPU is available

            self.ollama_client = ollama.Client()  # Adjust host if necessary
            logging.info("TranscriptionWorker initialized successfully.")
        except Exception as e:
            logging.error(f"Initialization error: {e}")
            eel.display_error(f"Initialization error: {e}")
            self.model = None
            self.processor = None

    def generate(self, *args, **kwargs):
        """
        Helper method to call generate on the underlying model,
        handling DataParallel wrapping.
        """
        if isinstance(self.model, nn.DataParallel):
            return self.model.module.generate(*args, **kwargs)
        else:
            return self.model.generate(*args, **kwargs)

    def run(self):
        self.running = True
        buffer = {}
        overlap_samples = int(1 * self.samplerate)  # 1 second overlap

        while self.running and self.model and self.processor:
            try:
                data, source_type = self.audio_queue.get(timeout=1)
                if source_type not in buffer:
                    buffer[source_type] = np.array([], dtype=np.float32)
                buffer[source_type] = np.concatenate((buffer[source_type], data))

                # Detect silence
                split_point = detect_silence(
                    buffer[source_type],
                    self.samplerate,
                    frame_duration=0.05,
                    energy_threshold=0.02,
                    silence_duration=0.5
                )

                if split_point is not None and split_point > 0:
                    # Split at silence
                    segment = buffer[source_type][:split_point]
                    buffer[source_type] = buffer[source_type][split_point:]

                    if np.mean(np.abs(segment)) < 0.01:
                        continue  # Skip silent segments

                    self.process_segment(segment, source_type)
                else:
                    # No silence detected; check if buffer exceeds chunk length
                    chunk_length_s = 30
                    overlap_duration_s = 1
                    overlap_samples = int(overlap_duration_s * self.samplerate)

                    if len(buffer[source_type]) >= (chunk_length_s + overlap_duration_s) * self.samplerate:
                        # Extract chunk with overlap
                        chunk_end = chunk_length_s * self.samplerate
                        chunk = buffer[source_type][:chunk_end + overlap_samples]
                        buffer[source_type] = buffer[source_type][chunk_end:]

                        if np.mean(np.abs(chunk)) < 0.01:
                            continue  # Skip silent segments

                        self.process_segment(chunk, source_type)
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Processing error: {e}")
                eel.display_error(f"Processing error: {e}")

        # Process any remaining audio in the buffer when stopping
        for source_type, audio_data in buffer.items():
            if len(audio_data) > 0:
                self.process_segment(audio_data, source_type)

    def process_segment(self, segment, source_type):
        try:
            # Convert the segment to float32 if it's not
            if segment.dtype != np.float32:
                segment = segment.astype(np.float32)

            # Prepare input features with language set to English
            with torch.no_grad():
                input_features = self.processor(
                    segment, 
                    sampling_rate=self.samplerate, 
                    return_tensors="pt", 
                    padding="longest", 
                    language='en'  # Ensure language is set to English
                ).input_features
                input_features = pad_or_truncate(input_features)
                input_features = input_features.to(self.device)
                if self.device.startswith('cuda'):
                    input_features = input_features.half()  # Convert to float16 for mixed precision

                # Generate transcription
                predicted_ids = self.generate(
                    input_features,
                    max_length=3000,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )
                transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                logging.debug(f"Transcription: {transcription}")

            # Append to conversation history
            with self.lock:
                self.conversation_history.append({"role": "user", "content": transcription})

            # Send the transcription to the frontend
            eel.add_message('User', transcription)

            # Signal AIResponseHandler to process the new prompt
            self.ai_response_handler.handle_new_prompt()

        except Exception as e:
            logging.error(f"Error during transcription/AI response: {e}")
            eel.display_error(f"Error during transcription/AI response: {e}")

    def stop(self):
        self.running = False
        # Stop AIResponseHandler if active
        if self.ai_response_handler and self.ai_response_handler.is_alive():
            self.ai_response_handler.stop()
            self.ai_response_handler.join()

# --------------------
# TranscriptionApp Class
# --------------------
class TranscriptionApp:
    def __init__(self, settings):
        self.audio_queue = queue.Queue(maxsize=200)
        self.recorder_thread = None
        self.transcription_thread = None
        self.recording_devices = []
        self.available_devices = self.get_available_devices()
        self.settings = settings
        self.voice_activity_event = threading.Event()
        self.is_muted_event = threading.Event()
        self.ai_response_handler = None

    def get_available_devices(self):
        devices = ['cpu']
        if torch.cuda.is_available():
            devices.append('cuda')  # 'cuda' will utilize all available GPUs via DataParallel
        return devices

    def get_audio_sources(self):
        try:
            self.recording_devices = sc.all_microphones(include_loopback=True)
            sources = []
            for mic in self.recording_devices:
                device_name = mic.name.lower()
                if 'loopback' in device_name or 'stereo mix' in device_name or 'what u hear' in device_name:
                    label = f"System Audio - {mic.name}"
                    source_type = 'System Audio'
                else:
                    label = f"Microphone - {mic.name}"
                    source_type = 'Microphone'
                sources.append({'label': label, 'index': self.recording_devices.index(mic), 'type': source_type})
            eel.receive_audio_sources(sources)
            logging.info("Audio sources sent to frontend.")
            return True
        except Exception as e:
            logging.error(f"Error fetching audio sources: {e}")
            eel.display_error(f"Error fetching audio sources: {e}")
            return False

    def get_device_list(self):
        try:
            devices = self.available_devices
            eel.receive_device_list(devices)
            logging.info("Device list sent to frontend.")
            return True
        except Exception as e:
            logging.error(f"Error fetching device list: {e}")
            eel.display_error(f"Error fetching device list: {e}")
            return False

    def get_ollama_models(self):
        try:
            client = ollama.Client()
            models_response = ollama.list()
            
            # Log the type and content of the 'models_response'
            logging.info(f"Models response: {models_response} (Type: {type(models_response)})")
            
            # Check if 'models' key exists in the response and is a list
            if isinstance(models_response, dict) and 'models' in models_response:
                models = models_response['models']  # Access the list of models
                model_names = [model['name'] for model in models]
                eel.receive_ollama_models(model_names)
                logging.info("Ollama models sent to frontend.")
                return True
            else:
                logging.error("The response from ollama.list() does not contain a 'models' list.")
                return False
        except Exception as e:
            logging.error(f"Error fetching Ollama models: {e}")
            eel.display_error(f"Error fetching Ollama models: {e}")
            return False

    def start_transcription(self, selected_index, selected_device, selected_ollama_model):
        if self.recorder_thread and self.recorder_thread.is_alive():
            eel.display_error("Transcription is already running.")
            return False

        try:
            mic = self.recording_devices[selected_index]
            source_type = 'System Audio' if ('loopback' in mic.name.lower() or 
                                            'stereo mix' in mic.name.lower() or 
                                            'what u hear' in mic.name.lower()) else 'Microphone'

            # Update settings
            self.settings.set_setting("selected_device", selected_device)
            self.settings.set_setting("selected_audio_source", selected_index)  # Save selected audio source
            self.settings.set_setting("selected_ollama_model", selected_ollama_model)  # Save selected ollama model

            # Initialize shared voice_activity_event and is_muted_event
            self.voice_activity_event.clear()
            self.is_muted_event.clear()

            # Initialize AIResponseHandler
            conversation_history = []
            lock = threading.Lock()
            self.ai_response_handler = AIResponseHandler(
                ollama_client=ollama.Client(),  # Adjust host if necessary
                selected_ollama_model=selected_ollama_model,
                device=selected_device,
                voice_activity_event=self.voice_activity_event,
                lock=lock,
                conversation_history=conversation_history
            )
            self.ai_response_handler.start()
            logging.info("AIResponseHandler thread started.")

            # Initialize TranscriptionWorker with reference to AIResponseHandler
            self.transcription_thread = TranscriptionWorker(
                audio_queue=self.audio_queue,
                ai_response_handler=self.ai_response_handler,
                device=selected_device
            )
            self.transcription_thread.conversation_history = conversation_history
            self.transcription_thread.lock = lock
            self.transcription_thread.start()
            logging.info("TranscriptionWorker thread started.")

            # Initialize AudioRecorder
            self.recorder_thread = AudioRecorder(
                mic=mic,
                source_type=source_type,
                voice_activity_event=self.voice_activity_event,
                is_muted_event=self.is_muted_event,
                audio_queue=self.audio_queue
            )
            self.recorder_thread.start()
            logging.info("AudioRecorder thread started.")

            eel.transcription_started()
            return True
        except Exception as e:
            logging.error(f"Error starting transcription: {e}")
            eel.display_error(f"Error starting transcription: {e}")
            return False

    def stop_transcription(self):
        try:
            if self.recorder_thread:
                self.recorder_thread.stop()
                self.recorder_thread.join()
                self.recorder_thread = None
                logging.info("AudioRecorder thread stopped.")

            if self.transcription_thread:
                self.transcription_thread.stop()
                self.transcription_thread.join()
                self.transcription_thread = None
                logging.info("TranscriptionWorker thread stopped.")

            if self.ai_response_handler:
                self.ai_response_handler.stop()
                self.ai_response_handler.join()
                self.ai_response_handler = None
                logging.info("AIResponseHandler thread stopped.")

            eel.transcription_stopped()
            return True
        except Exception as e:
            logging.error(f"Error stopping transcription: {e}")
            eel.display_error(f"Error stopping transcription: {e}")
            return False

    def get_current_settings(self):
        try:
            eel.receive_current_settings(self.settings.settings)
            logging.info("Current settings sent to frontend.")
            return True
        except Exception as e:
            logging.error(f"Error sending current settings: {e}")
            eel.display_error(f"Error sending current settings: {e}")
            return False

    def mute_transcription(self):
        self.is_muted_event.set()
        eel.update_mute_button(True)
        logging.info("Transcription muted.")

    def unmute_transcription(self):
        self.is_muted_event.clear()
        eel.update_mute_button(False)
        logging.info("Transcription unmuted.")

    def send_user_message(self, message):
        # Process typed user message
        if not message.strip():
            return
        with self.transcription_thread.lock:
            self.transcription_thread.conversation_history.append({"role": "user", "content": message.strip()})
        # Send the message to the frontend
        eel.add_message('User', message.strip())
        # Signal AIResponseHandler to process the new prompt
        self.ai_response_handler.handle_new_prompt()

# --------------------
# Instantiate Settings and TranscriptionApp
# --------------------
settings = Settings()
app = TranscriptionApp(settings)

# --------------------
# Define Eel-exposed functions outside the class
# --------------------
@eel.expose
def get_audio_sources():
    return app.get_audio_sources()

@eel.expose
def get_device_list():
    return app.get_device_list()

@eel.expose
def get_ollama_models():
    return app.get_ollama_models()

@eel.expose
def start_transcription(selected_index, selected_device, selected_ollama_model):
    return app.start_transcription(selected_index, selected_device, selected_ollama_model)

@eel.expose
def stop_transcription():
    return app.stop_transcription()

@eel.expose
def get_current_settings():
    return app.get_current_settings()

@eel.expose
def mute_transcription():
    app.mute_transcription()

@eel.expose
def unmute_transcription():
    app.unmute_transcription()

@eel.expose
def send_user_message(message):
    app.send_user_message(message)

# --------------------
# Main Function
# --------------------
def main():
    eel.start(
        'index.html',
        mode='chrome',  # Specify the browser mode (e.g., 'chrome', 'edge', etc.)
        chrome_args=[
            '--resizable',               # Allow window resizing
            '--start-maximized'         # Start maximized
        ],
        block=False
    )
    app.get_ollama_models()
    app.get_audio_sources()
    app.get_device_list()
    app.get_current_settings()
    eel.sleep(1)  # Allow time for frontend to initialize

    # Keep the main thread alive
    try:
        while True:
            eel.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        logging.info("Shutting down application.")
        app.stop_transcription()

if __name__ == "__main__":
    main()
