// web/script.js

document.addEventListener('DOMContentLoaded', () => {
    // Fetch ollama models, audio sources, device list, and current settings on load
    eel.get_ollama_models();
    eel.get_audio_sources();
    eel.get_device_list();
    eel.get_current_settings();

    // Elements
    const ollamaModelSelector = document.getElementById('ollamaModelSelector');
    const audioSourceSelect = document.getElementById('audioSource');
    const deviceSelector = document.getElementById('deviceSelector');
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const muteBtn = document.getElementById('muteBtn');
    const conversationContainer = document.getElementById('conversationContainer');
    const statusBar = document.getElementById('statusBar');
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');

    let isMuted = false;

    // Function to display status messages
    function showStatus(message, type = 'info') {
        statusBar.className = ''; // Reset classes
        statusBar.classList.add('alert', `alert-${type}`, 'mt-3');
        statusBar.textContent = message;
        statusBar.classList.remove('d-none'); // Show the status bar
        // Automatically hide after 5 seconds
        setTimeout(() => {
            statusBar.classList.add('d-none');
        }, 5000);
    }

    // Start Button Click
    startBtn.addEventListener('click', () => {
        const selectedAudioIndex = audioSourceSelect.value;
        const selectedDevice = deviceSelector.value;
        const selectedOllamaModel = ollamaModelSelector.value;

        if (!selectedOllamaModel) {
            showStatus('Please select an Ollama model.', 'warning');
            return;
        }

        if (selectedAudioIndex === undefined || selectedAudioIndex === null || selectedAudioIndex === "") {
            showStatus('Please select an audio source.', 'warning');
            return;
        }

        if (!selectedDevice) {
            showStatus('Please select a processing device.', 'warning');
            return;
        }

        eel.start_transcription(parseInt(selectedAudioIndex), selectedDevice, selectedOllamaModel);
    });

    // Stop Button Click
    stopBtn.addEventListener('click', () => {
        eel.stop_transcription();
    });

    // Mute Button Click
    muteBtn.addEventListener('click', () => {
        if (isMuted) {
            eel.unmute_transcription();
        } else {
            eel.mute_transcription();
        }
    });

    // Send Button Click
    sendBtn.addEventListener('click', () => {
        const message = userInput.value.trim();
        if (message !== '') {
            eel.send_user_message(message);
            userInput.value = '';
        }
    });

    // Enter Key Press in Input Box
    userInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            event.preventDefault();
            sendBtn.click();
        }
    });
});

// Eel functions to receive data from backend
eel.expose(receive_ollama_models);
function receive_ollama_models(models) {
    console.log("Received Ollama models:", models);  // Debug log
    const ollamaModelSelector = document.getElementById('ollamaModelSelector');
    ollamaModelSelector.innerHTML = ''; // Clear existing options
    models.forEach(model => {
        const option = document.createElement('option');
        option.value = model;
        option.textContent = model;
        ollamaModelSelector.appendChild(option);
    });
}

eel.expose(receive_audio_sources);
function receive_audio_sources(sources) {
    console.log("Received audio sources:", sources);  // Debug log
    const audioSourceSelect = document.getElementById('audioSource');
    audioSourceSelect.innerHTML = ''; // Clear existing options
    sources.forEach(source => {
        const option = document.createElement('option');
        option.value = source.index;
        option.textContent = source.label;
        audioSourceSelect.appendChild(option);
    });
}

eel.expose(receive_device_list);
function receive_device_list(devices) {
    console.log("Received device list:", devices);  // Debug log
    const deviceSelector = document.getElementById('deviceSelector');
    deviceSelector.innerHTML = ''; // Clear existing options
    devices.forEach(device => {
        const option = document.createElement('option');
        option.value = device;
        option.textContent = device.toUpperCase();
        deviceSelector.appendChild(option);
    });
}

eel.expose(receive_current_settings);
function receive_current_settings(settings) {
    console.log("Received current settings:", settings);  // Debug log
    const ollamaModelSelector = document.getElementById('ollamaModelSelector');
    const deviceSelector = document.getElementById('deviceSelector');
    const audioSourceSelect = document.getElementById('audioSource');

    // Set selected Ollama model
    if (settings.selected_ollama_model) {
        ollamaModelSelector.value = settings.selected_ollama_model;
    }

    // Set selected device
    if (settings.selected_device) {
        deviceSelector.value = settings.selected_device;
    }

    // Set selected audio source
    if (settings.selected_audio_source !== undefined && settings.selected_audio_source !== "") {
        audioSourceSelect.value = settings.selected_audio_source;
    }
}

eel.expose(add_message);
function add_message(sender, message) {
    console.log("Adding message from", sender, ":", message);  // Debug log
    const conversationContainer = document.getElementById('conversationContainer');

    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', sender.toLowerCase());

    // For AI messages, we will render markdown
    if (sender.toLowerCase() === 'ai') {
        messageDiv.setAttribute('data-markdown', message);
        messageDiv.innerHTML = marked.parse(message);
    } else {
        messageDiv.textContent = message;
    }

    conversationContainer.appendChild(messageDiv);
    conversationContainer.scrollTop = conversationContainer.scrollHeight;
}

eel.expose(update_last_message);
function update_last_message(text) {
    console.log("Updating last message with:", text);  // Debug log
    const conversationContainer = document.getElementById('conversationContainer');
    const lastMessageDiv = conversationContainer.lastElementChild;

    if (lastMessageDiv && lastMessageDiv.classList.contains('ai')) {
        // Append new text and re-render markdown
        const currentContent = lastMessageDiv.getAttribute('data-markdown') || '';
        const updatedContent = currentContent + text;
        lastMessageDiv.setAttribute('data-markdown', updatedContent);
        lastMessageDiv.innerHTML = marked.parse(updatedContent);
        conversationContainer.scrollTop = conversationContainer.scrollHeight;
    } else {
        console.error("No AI message to update.");
    }
}

eel.expose(display_error);
function display_error(message) {
    console.log("Received error message:", message);  // Debug log
    showStatus(message, 'danger');
}

eel.expose(transcription_started);
function transcription_started() {
    console.log("Transcription started.");  // Debug log
    document.getElementById('startBtn').disabled = true;
    document.getElementById('stopBtn').disabled = false;
    document.getElementById('muteBtn').disabled = false;
    showStatus('Transcription started.', 'success');
}

eel.expose(transcription_stopped);
function transcription_stopped() {
    console.log("Transcription stopped.");  // Debug log
    document.getElementById('startBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
    document.getElementById('muteBtn').disabled = true;
    showStatus('Transcription stopped.', 'secondary');
}

eel.expose(update_mute_button);
function update_mute_button(isMuted) {
    console.log("Mute status updated:", isMuted);  // Debug log
    const muteBtn = document.getElementById('muteBtn');
    if (isMuted) {
        muteBtn.textContent = 'Unmute';
    } else {
        muteBtn.textContent = 'Mute';
    }
    window.isMuted = isMuted;
}

// Function to display status messages
function showStatus(message, type = 'info') {
    const statusBar = document.getElementById('statusBar');
    if (!statusBar) return;
    statusBar.className = ''; // Reset classes
    statusBar.classList.add('alert', `alert-${type}`, 'mt-3');
    statusBar.textContent = message;
    statusBar.classList.remove('d-none'); // Show the status bar
    // Automatically hide after 5 seconds
    setTimeout(() => {
        statusBar.classList.add('d-none');
    }, 5000);
}
