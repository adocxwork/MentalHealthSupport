/**
 * Mental Health Support Application - Client-Side JavaScript
 * 
 * This module handles all client-side interactions including:
 * - Emotion detection and analysis
 * - Appointment booking and management
 * - Real-time chat with AJAX polling
 * - UPI payment QR code modal
 * - Prescription management
 */

document.addEventListener('DOMContentLoaded', () => {
    initEmotionDetection();
    initAppointmentBooking();
    initStatusActions();
    initChat();
    initUPIModal();
    initPrescriptions();
});


// =============================================================================
// Emotion Detection Module
// =============================================================================

/**
 * Initialize the emotion detection feature.
 * Sets up event listeners for analyzing text and displaying results.
 */
function initEmotionDetection() {
    const textInput = document.getElementById('textInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const topKSelect = document.getElementById('topK');
    const resultsSection = document.getElementById('resultsSection');
    const resultsContainer = document.getElementById('resultsContainer');
    const errorSection = document.getElementById('errorSection');
    const errorText = document.getElementById('errorText');

    if (!textInput || !analyzeBtn) return;

    // Click handler for analyze button
    analyzeBtn.addEventListener('click', () => {
        const text = textInput.value.trim();
        if (!text) {
            showError('Please enter some text to analyze.');
            return;
        }
        analyzeEmotions(text);
    });

    // Keyboard shortcut: Ctrl+Enter to analyze
    textInput.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            analyzeBtn.click();
        }
    });

    /**
     * Send text to the API for emotion analysis.
     * @param {string} text - The text to analyze
     */
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
            
            if (!response.ok) {
                throw new Error(data.error || 'Failed to analyze emotions');
            }
            
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

    /**
     * Display emotion analysis results.
     * @param {Array} emotions - Array of emotion predictions
     */
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

    /**
     * Create an emotion result card element.
     * @param {Object} emotionData - Emotion data with name and confidence
     * @returns {HTMLElement} The card element
     */
    function createEmotionCard(emotionData) {
        const card = document.createElement('div');
        card.className = 'emotion-card';
        
        const confidence = (emotionData.confidence * 100).toFixed(1);
        const emotionName = emotionData.emotion.charAt(0).toUpperCase() + 
                           emotionData.emotion.slice(1);
        
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

    /**
     * Display an error message.
     * @param {string} message - Error message to display
     */
    function showError(message) {
        if (errorText) errorText.textContent = message;
        if (errorSection) errorSection.style.display = 'block';
    }

    /**
     * Hide the error message section.
     */
    function hideError() {
        if (errorSection) errorSection.style.display = 'none';
    }
}


// =============================================================================
// Appointment Booking Module
// =============================================================================

/**
 * Initialize the appointment booking form.
 * Handles form submission and API interaction for creating appointments.
 */
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
                body: JSON.stringify({ 
                    doctor_id: doctorId, 
                    slot_time: slotTime, 
                    reason 
                })
            });
            const data = await res.json();
            
            if (!res.ok) {
                throw new Error(data.error || 'Could not book appointment');
            }
            
            alert('Appointment requested successfully.');
            window.location.href = '/appointments';
        } catch (err) {
            alert(err.message);
        }
    });
}


// =============================================================================
// Appointment Status Actions Module
// =============================================================================

/**
 * Initialize status action buttons (Accept/Reject/Delete).
 * Handles appointment status updates and deletion.
 */
function initStatusActions() {
    // Accept/Reject buttons
    document.querySelectorAll('.status-btn').forEach((btn) => {
        btn.addEventListener('click', async () => {
            const appointmentId = btn.dataset.appointmentId;
            const status = btn.dataset.status;
            
            if (!confirm(`Are you sure you want to ${status} this appointment?`)) {
                return;
            }

            try {
                const res = await fetch(`/appointments/${appointmentId}/status`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ status })
                });
                const data = await res.json();
                
                if (!res.ok) {
                    throw new Error(data.error || 'Failed to update status');
                }
                
                alert(`Appointment ${status} successfully.`);
                window.location.reload();
            } catch (err) {
                alert(err.message);
            }
        });
    });

    // Delete appointment buttons
    document.querySelectorAll('.delete-appointment-btn').forEach((btn) => {
        btn.addEventListener('click', async () => {
            const appointmentId = btn.dataset.appointmentId;
            const confirmMessage = 
                'Are you sure you want to delete this appointment? ' +
                'This will also delete all related chat messages and prescriptions. ' +
                'This action cannot be undone.';
            
            if (!confirm(confirmMessage)) {
                return;
            }

            try {
                const res = await fetch(`/appointments/${appointmentId}`, {
                    method: 'DELETE',
                    headers: { 'Content-Type': 'application/json' }
                });
                const data = await res.json();
                
                if (!res.ok) {
                    throw new Error(data.error || 'Failed to delete appointment');
                }
                
                alert('Appointment deleted successfully.');
                window.location.reload();
            } catch (err) {
                alert(err.message);
            }
        });
    });
}


// =============================================================================
// Chat Module
// =============================================================================

// Chat state variables
let chatPoller = null;
let lastMessageId = 0;
let currentAppointmentId = null;

/**
 * Initialize the chat feature.
 * Sets up message polling, sending, and display.
 */
function initChat() {
    const container = document.querySelector('.chat-container');
    if (!container) return;

    const selector = document.getElementById('chatAppointmentSelector');
    const messagesBox = document.getElementById('chatMessages');
    const chatForm = document.getElementById('chatForm');
    const chatInput = document.getElementById('chatInput');

    /**
     * Start polling for new messages.
     * @param {string} appointmentId - The appointment ID to poll messages for
     */
    function startPolling(appointmentId) {
        if (chatPoller) clearInterval(chatPoller);
        lastMessageId = 0;
        currentAppointmentId = appointmentId;
        
        if (messagesBox) messagesBox.innerHTML = '';
        if (!appointmentId) return;
        
        fetchMessages(appointmentId);
        chatPoller = setInterval(() => fetchMessages(appointmentId), 3000);
    }

    /**
     * Fetch messages from the server.
     * @param {string} appointmentId - The appointment ID
     */
    async function fetchMessages(appointmentId) {
        try {
            const res = await fetch(
                `/api/chat/${appointmentId}/messages?after=${lastMessageId}`
            );
            const data = await res.json();
            
            if (!res.ok) {
                throw new Error(data.error || 'Failed to load messages');
            }
            
            data.forEach((msg) => appendMessage(msg));
        } catch (err) {
            console.error(err);
        }
    }

    /**
     * Append a message to the chat display.
     * @param {Object} msg - Message object from the server
     */
    function appendMessage(msg) {
        if (!messagesBox) return;
        
        lastMessageId = Math.max(lastMessageId, msg.id);
        
        const item = document.createElement('div');
        item.className = 'chat-message';
        
        const senderName = msg.sender_name || 'Unknown';
        const roleAndUpi = msg.sender_display || '';
        const timestamp = new Date(msg.created_at).toLocaleString();
        
        item.innerHTML = `
            <div class="chat-meta">
                <strong>${senderName}</strong><br>
                <span class="muted">${roleAndUpi}</span> - ${timestamp}
            </div>
            <div>${msg.message}</div>
        `;
        
        messagesBox.appendChild(item);
        messagesBox.scrollTop = messagesBox.scrollHeight;
    }

    // Handle appointment selection change
    if (selector) {
        selector.addEventListener('change', () => {
            const apptId = selector.value;
            const selectedOption = selector.options[selector.selectedIndex];
            const oppositeUpi = selectedOption.dataset.oppositeUpi;
            const oppositeName = selectedOption.dataset.oppositeName;

            // Update chat header with user info
            if (apptId && oppositeUpi && oppositeName) {
                updateChatUserInfo(selectedOption, oppositeName, oppositeUpi);
            } else {
                hideChatUserInfo();
            }

            startPolling(apptId);
        });
    }

    /**
     * Update the chat header with the selected user's info.
     */
    function updateChatUserInfo(selectedOption, oppositeName, oppositeUpi) {
        const userInfo = document.getElementById('chatUserInfo');
        const userName = document.getElementById('chatUserName');
        const userLabel = document.getElementById('chatUserLabel');
        const upiClick = document.getElementById('chatUpiClick');

        if (userInfo && userName && userLabel && upiClick) {
            userName.textContent = oppositeName;
            
            // Determine role from option text
            const optionText = selectedOption.textContent.trim();
            const roleText = optionText.startsWith('Patient:') ? 'Patient' : 'Doctor';
            
            userLabel.textContent = `${roleText} | ${oppositeUpi}`;
            upiClick.dataset.upi = oppositeUpi;
            upiClick.dataset.userName = oppositeName;
            upiClick.classList.add('upi-click');
            userInfo.style.display = 'block';
        }
    }

    /**
     * Hide the chat user info section.
     */
    function hideChatUserInfo() {
        const userInfo = document.getElementById('chatUserInfo');
        if (userInfo) userInfo.style.display = 'none';
    }

    // Handle message form submission
    if (chatForm && chatInput) {
        chatForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const apptId = selector ? selector.value : currentAppointmentId;
            const message = chatInput.value.trim();
            
            if (!apptId) {
                alert('Select an appointment to chat.');
                return;
            }
            if (!message) return;

            try {
                const res = await fetch(`/api/chat/${apptId}/messages`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message })
                });
                const data = await res.json();
                
                if (!res.ok) {
                    throw new Error(data.error || 'Failed to send message');
                }
                
                chatInput.value = '';
                fetchMessages(apptId);
            } catch (err) {
                alert(err.message);
            }
        });
    }
}


// =============================================================================
// UPI Modal Module
// =============================================================================

/**
 * Initialize the UPI payment QR code modal.
 * Handles displaying QR codes for UPI payment.
 */
function initUPIModal() {
    const modal = document.getElementById('upiModal');
    const img = document.getElementById('upiQrImg');
    const label = document.getElementById('upiLabel');
    const closeBtn = document.getElementById('closeUpiModal');
    
    if (!modal || !img || !label) return;

    /**
     * Open the UPI modal with QR code.
     * @param {string} upId - UPI payment ID
     * @param {string} userName - User's name for display
     */
    function openModal(upId, userName) {
        const upiPayload = `upi://pay?pa=${encodeURIComponent(upId)}&pn=${encodeURIComponent(userName)}`;
        const qrUrl = `https://api.qrserver.com/v1/create-qr-code/?size=220x220&data=${encodeURIComponent(upiPayload)}`;
        
        img.src = qrUrl;
        label.textContent = `${userName} | ${upId}`;
        modal.classList.remove('hidden');
    }

    /**
     * Close the UPI modal.
     */
    function closeModal() {
        modal.classList.add('hidden');
    }

    // Event delegation for UPI click elements
    document.addEventListener('click', (e) => {
        const target = e.target.classList.contains('upi-click') 
            ? e.target 
            : e.target.closest('.upi-click');
        
        if (target && target.dataset.upi && target.dataset.userName) {
            openModal(target.dataset.upi, target.dataset.userName);
        }
    });

    // Close button handler
    if (closeBtn) {
        closeBtn.addEventListener('click', closeModal);
    }

    // Close on backdrop click
    modal.addEventListener('click', (e) => {
        if (e.target === modal) closeModal();
    });
}


// =============================================================================
// Prescriptions Module
// =============================================================================

/**
 * Initialize the prescriptions feature.
 * Handles prescription creation (doctors) and viewing (patients).
 */
function initPrescriptions() {
    const form = document.getElementById('prescriptionForm');
    const historyBox = document.getElementById('prescriptionHistory');
    const select = document.getElementById('prescriptionAppointment');

    /**
     * Load prescription history for an appointment.
     * @param {string} appointmentId - The appointment ID
     */
    async function loadHistory(appointmentId) {
        if (!historyBox || !appointmentId) return;
        
        historyBox.innerHTML = '<p class="muted">Loading prescriptions...</p>';
        
        try {
            const res = await fetch(`/api/appointments/${appointmentId}/prescriptions`);
            const data = await res.json();
            
            if (!res.ok) {
                throw new Error(data.error || 'Failed to load prescriptions');
            }
            
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

    // Handle appointment selection
    if (select) {
        // Check for appointment_id in URL (for patient view navigation)
        const urlParams = new URLSearchParams(window.location.search);
        const appointmentIdFromUrl = urlParams.get('appointment_id');
        
        if (appointmentIdFromUrl) {
            setTimeout(() => {
                const option = select.querySelector(
                    `option[value="${appointmentIdFromUrl}"]`
                );
                if (option) {
                    select.value = appointmentIdFromUrl;
                    select.dispatchEvent(new Event('change', { bubbles: true }));
                }
            }, 100);
        }

        select.addEventListener('change', () => loadHistory(select.value));
    }

    // Handle prescription form submission (doctors only)
    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const appointmentId = select ? select.value : '';
            const notes = document.getElementById('prescriptionNotes').value;
            const tests = document.getElementById('prescriptionTests').value;
            
            if (!appointmentId) {
                alert('Select an appointment first.');
                return;
            }
            if (!notes) {
                alert('Prescription notes are required.');
                return;
            }

            try {
                const res = await fetch(`/appointments/${appointmentId}/prescription`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ notes, tests })
                });
                const data = await res.json();
                
                if (!res.ok) {
                    throw new Error(data.error || 'Failed to save prescription');
                }
                
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
