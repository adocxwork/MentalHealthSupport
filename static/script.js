// Simple JavaScript for Mental Health Support App

document.addEventListener('DOMContentLoaded', () => {
    initEmotionDetection();
    initAppointmentBooking();
    initStatusActions();
    initChat();
    initUPIModal();
    initPrescriptions();
});

// Emotion Detection
function initEmotionDetection() {
    const textInput = document.getElementById('textInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const topKSelect = document.getElementById('topK');
    const resultsSection = document.getElementById('resultsSection');
    const resultsContainer = document.getElementById('resultsContainer');
    const errorSection = document.getElementById('errorSection');
    const errorText = document.getElementById('errorText');

    if (!textInput || !analyzeBtn) return;

    analyzeBtn.addEventListener('click', () => {
        const text = textInput.value.trim();
        if (!text) {
            showError('Please enter some text to analyze.');
            return;
        }
        analyzeEmotions(text);
    });

    textInput.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            analyzeBtn.click();
        }
    });

    async function analyzeEmotions(text) {
        hideError();
        resultsSection.style.display = 'none';
        analyzeBtn.disabled = true;
        analyzeBtn.textContent = 'Analyzing...';
        
        try {
            const topK = parseInt(topKSelect.value);
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text, top_k: topK })
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Failed to analyze emotions');
            if (data.success) {
                displayResults(data.emotions);
            } else {
                throw new Error('Invalid response from server');
            }
        } catch (error) {
            showError(error.message || 'An error occurred while analyzing emotions.');
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = 'Analyze';
        }
    }

    function displayResults(emotions) {
        resultsContainer.innerHTML = '';
        if (!emotions || emotions.length === 0) {
            resultsContainer.innerHTML = '<p class="muted">No emotions detected.</p>';
        } else {
            emotions.forEach((emotion) => {
                const card = createEmotionCard(emotion);
                resultsContainer.appendChild(card);
            });
        }
        resultsSection.style.display = 'block';
    }

    function createEmotionCard(emotionData) {
        const card = document.createElement('div');
        card.className = 'emotion-card';
        const confidence = (emotionData.confidence * 100).toFixed(1);
        const emotionName = emotionData.emotion.charAt(0).toUpperCase() + emotionData.emotion.slice(1);
        card.innerHTML = `
            <div class="emotion-header">
                <span class="emotion-name">${emotionName}</span>
                <span class="emotion-confidence">${confidence}%</span>
            </div>
            <div class="confidence-bar-container">
                <div class="confidence-bar" style="width: ${confidence}%"></div>
            </div>
        `;
        return card;
    }

    function showError(message) {
        if (errorText) errorText.textContent = message;
        if (errorSection) errorSection.style.display = 'block';
    }

    function hideError() {
        if (errorSection) errorSection.style.display = 'none';
    }
}

// Appointment Booking
function initAppointmentBooking() {
    const form = document.getElementById('appointmentForm');
    if (!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const doctorId = form.querySelector('[name="doctor_id"]').value;
        const slotTime = form.querySelector('[name="slot_time"]').value;
        const reason = form.querySelector('[name="reason"]').value;
        
        if (!doctorId || !slotTime) {
            alert('Please fill all required fields.');
            return;
        }

        try {
            const res = await fetch('/appointments', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ doctor_id: doctorId, slot_time: slotTime, reason })
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'Could not book appointment');
            alert('Appointment requested successfully.');
            window.location.href = '/appointments';
        } catch (err) {
            alert(err.message);
        }
    });
}

// Status Actions (Accept/Reject)
function initStatusActions() {
    document.querySelectorAll('.status-btn').forEach((btn) => {
        btn.addEventListener('click', async () => {
            const appointmentId = btn.dataset.appointmentId;
            const status = btn.dataset.status;
            if (!confirm(`Are you sure you want to ${status} this appointment?`)) return;
            
            try {
                const res = await fetch(`/appointments/${appointmentId}/status`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ status })
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.error || 'Failed to update status');
                alert(`Appointment ${status} successfully.`);
                window.location.reload();
            } catch (err) {
                alert(err.message);
            }
        });
    });
    
    // Delete appointment actions
    document.querySelectorAll('.delete-appointment-btn').forEach((btn) => {
        btn.addEventListener('click', async () => {
            const appointmentId = btn.dataset.appointmentId;
            if (!confirm('Are you sure you want to delete this appointment? This will also delete all related chat messages and prescriptions. This action cannot be undone.')) return;
            
            try {
                const res = await fetch(`/appointments/${appointmentId}`, {
                    method: 'DELETE',
                    headers: { 'Content-Type': 'application/json' }
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.error || 'Failed to delete appointment');
                alert('Appointment deleted successfully.');
                window.location.reload();
            } catch (err) {
                alert(err.message);
            }
        });
    });
}

// Chat
let chatPoller = null;
let lastMessageId = 0;
let currentAppointmentId = null;

function initChat() {
    const container = document.querySelector('.chat-container');
    if (!container) return;

    const selector = document.getElementById('chatAppointmentSelector');
    const messagesBox = document.getElementById('chatMessages');
    const chatForm = document.getElementById('chatForm');
    const chatInput = document.getElementById('chatInput');
    const chatHeader = document.getElementById('chatHeader');

    // Don't auto-select appointment from URL - let user select manually
    // This ensures consistent behavior whether coming from Dashboard->Chat or Dashboard->Appointments->Chat

    function startPolling(appointmentId) {
        if (chatPoller) clearInterval(chatPoller);
        lastMessageId = 0;
        currentAppointmentId = appointmentId;
        if (messagesBox) messagesBox.innerHTML = '';
        if (!appointmentId) return;
        fetchMessages(appointmentId);
        chatPoller = setInterval(() => fetchMessages(appointmentId), 3000);
    }

    async function fetchMessages(appointmentId) {
        try {
            const res = await fetch(`/api/chat/${appointmentId}/messages?after=${lastMessageId}`);
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'Failed to load messages');
            data.forEach((msg) => appendMessage(msg));
        } catch (err) {
            console.error(err);
        }
    }

    function appendMessage(msg) {
        if (!messagesBox) return;
        lastMessageId = Math.max(lastMessageId, msg.id);
        const item = document.createElement('div');
        item.className = 'chat-message';
        // Format: Name on first line, then Role | UPI ID on second line
        const senderName = msg.sender_name || 'Unknown';
        const roleAndUpi = msg.sender_display || '';
        item.innerHTML = `
            <div class="chat-meta">
                <strong>${senderName}</strong><br>
                <span class="muted">${roleAndUpi}</span> • ${new Date(msg.created_at).toLocaleString()}
            </div>
            <div>${msg.message}</div>
        `;
        messagesBox.appendChild(item);
        messagesBox.scrollTop = messagesBox.scrollHeight;
    }

    if (selector) {
        selector.addEventListener('change', () => {
            const apptId = selector.value;
            const selectedOption = selector.options[selector.selectedIndex];
            const oppositeUpi = selectedOption.dataset.oppositeUpi;
            const oppositeName = selectedOption.dataset.oppositeName;
            
            if (apptId && oppositeUpi && oppositeName) {
                const userInfo = document.getElementById('chatUserInfo');
                const userName = document.getElementById('chatUserName');
                const userLabel = document.getElementById('chatUserLabel');
                const upiClick = document.getElementById('chatUpiClick');
                
                if (userInfo && userName && userLabel && upiClick) {
                    userName.textContent = oppositeName;
                    // Determine role: if option text starts with "Patient:", current user is doctor, so opposite is Patient
                    // If option text starts with "Dr.", current user is patient, so opposite is Doctor
                    const optionText = selectedOption.textContent.trim();
                    const roleText = optionText.startsWith('Patient:') ? 'Patient' : 'Doctor';
                    userLabel.textContent = `${roleText} | ${oppositeUpi}`;
                    upiClick.dataset.upi = oppositeUpi;
                    upiClick.dataset.userName = oppositeName;
                    upiClick.classList.add('upi-click');
                    userInfo.style.display = 'block';
                }
            } else {
                const userInfo = document.getElementById('chatUserInfo');
                if (userInfo) userInfo.style.display = 'none';
            }
            
            startPolling(apptId);
        });
    }

    if (chatForm && chatInput) {
        chatForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const apptId = selector ? selector.value : currentAppointmentId;
            const message = chatInput.value.trim();
            if (!apptId) return alert('Select an appointment to chat.');
            if (!message) return;
            
            try {
                const res = await fetch(`/api/chat/${apptId}/messages`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message })
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.error || 'Failed to send message');
                chatInput.value = '';
                fetchMessages(apptId);
            } catch (err) {
                alert(err.message);
            }
        });
    }
}

// UPI Modal
function initUPIModal() {
    const modal = document.getElementById('upiModal');
    const img = document.getElementById('upiQrImg');
    const label = document.getElementById('upiLabel');
    const closeBtn = document.getElementById('closeUpiModal');
    if (!modal || !img || !label) return;

    function open(upId, userName) {
        const upiPayload = `upi://pay?pa=${encodeURIComponent(upId)}&pn=${encodeURIComponent(userName)}`;
        img.src = `https://api.qrserver.com/v1/create-qr-code/?size=220x220&data=${encodeURIComponent(upiPayload)}`;
        label.textContent = `${userName} | ${upId}`;
        modal.classList.remove('hidden');
    }

    function close() {
        modal.classList.add('hidden');
    }

    // Handle UPI clicks from various sources
    document.addEventListener('click', (e) => {
        if (e.target.classList.contains('upi-click') || e.target.closest('.upi-click')) {
            const el = e.target.classList.contains('upi-click') ? e.target : e.target.closest('.upi-click');
            if (el.dataset.upi && el.dataset.userName) {
                open(el.dataset.upi, el.dataset.userName);
            }
        }
    });
    
    if (closeBtn) closeBtn.addEventListener('click', close);
    modal.addEventListener('click', (e) => {
        if (e.target === modal) close();
    });
}

// Prescriptions
function initPrescriptions() {
    const form = document.getElementById('prescriptionForm');
    const historyBox = document.getElementById('prescriptionHistory');
    const select = document.getElementById('prescriptionAppointment');

    async function loadHistory(appointmentId) {
        if (!historyBox || !appointmentId) return;
        historyBox.innerHTML = '<p class="muted">Loading prescriptions...</p>';
        try {
            const res = await fetch(`/api/appointments/${appointmentId}/prescriptions`);
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || 'Failed to load prescriptions');
            if (!data.length) {
                historyBox.innerHTML = '<p class="muted">No prescriptions yet.</p>';
                return;
            }
            historyBox.innerHTML = data.map(p => `
                <div class="prescription-item">
                    <strong>${new Date(p.created_at).toLocaleString()}</strong>
                    <p class="muted">Notes: ${p.notes}</p>
                    ${p.tests ? `<p class="muted">Tests: ${p.tests}</p>` : ''}
                </div>
            `).join('');
        } catch (err) {
            historyBox.innerHTML = `<p class="muted">${err.message}</p>`;
        }
    }

    if (select) {
        // Check for appointment_id in URL (for patient view)
        const urlParams = new URLSearchParams(window.location.search);
        const appointmentIdFromUrl = urlParams.get('appointment_id');
        if (appointmentIdFromUrl) {
            setTimeout(() => {
                if (select.querySelector(`option[value="${appointmentIdFromUrl}"]`)) {
                    select.value = appointmentIdFromUrl;
                    select.dispatchEvent(new Event('change', { bubbles: true }));
                }
            }, 100);
        }
        
        select.addEventListener('change', () => loadHistory(select.value));
    }

    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const appointmentId = select ? select.value : '';
            const notes = document.getElementById('prescriptionNotes').value;
            const tests = document.getElementById('prescriptionTests').value;
            if (!appointmentId) return alert('Select an appointment first.');
            if (!notes) return alert('Prescription notes are required.');
            
            try {
                const res = await fetch(`/appointments/${appointmentId}/prescription`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ notes, tests })
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.error || 'Failed to save prescription');
                alert('Prescription saved successfully.');
                document.getElementById('prescriptionNotes').value = '';
                document.getElementById('prescriptionTests').value = '';
                loadHistory(appointmentId);
            } catch (err) {
                alert(err.message);
            }
        });
    }
}
