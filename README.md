# Mental Health Support Web Application

A full-stack web application designed to facilitate mental health support through secure doctor-patient interactions, appointment management, real-time chat, prescription management, and AI-powered emotion detection.

## Features

- **Role-Based Authentication**: Separate interfaces for doctors and patients
- **Appointment Management**: Book, accept, reject, and manage appointments
- **Real-Time Chat**: Secure messaging between doctors and patients via AJAX polling
- **Prescription Management**: Create and view prescriptions with notes and recommended tests
- **Emotion Detection**: AI-powered emotion analysis using RoBERTa-based model
- **UPI Integration**: Display UPI IDs and generate QR codes for payments
- **Responsive Design**: Modern, clean UI with mobile-friendly layout

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for cloning the repository)

### Installation

1. **Clone or download the repository**
   ```bash
   git clone <repository-url>
   cd MentalHealthSupport
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify model files**
   - Ensure the `emotion_roberta_model` directory exists with all model files:
     - `config.json`
     - `model.safetensors`
     - `tokenizer_config.json`
     - `vocab.json`
     - `merges.txt`
     - `special_tokens_map.json`
     - `best_config.json`

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the application**
   - Open your browser and navigate to: `http://localhost:5000`
   - The application will automatically create the database on first run

## Requirements

The following packages are required (automatically installed via `requirements.txt`):

- Flask 3.0.0
- Flask-CORS 4.0.0
- Flask-Login 0.6.3
- Flask-SQLAlchemy 3.1.1
- PyTorch (>=2.0.0)
- Transformers (>=4.30.0)
- Pandas (>=2.0.0)
- NumPy (>=1.24.0)
- Scikit-learn (>=1.3.0)
- tqdm (>=4.65.0)

## Usage

### For Patients

1. **Register**: Create an account with role "patient"
2. **Book Appointment**: Select a doctor and schedule an appointment
3. **View Appointments**: Check appointment status (pending/accepted/rejected)
4. **Chat**: Communicate with your doctor through the chat interface
5. **View Prescriptions**: Access prescriptions provided by your doctor
6. **Emotion Analysis**: Analyze emotional sentiment from text input

### For Doctors

1. **Register**: Create an account with role "doctor" and specialization
2. **Manage Appointments**: Accept or reject appointment requests
3. **Chat**: Communicate with patients
4. **Create Prescriptions**: Write prescriptions with notes and recommended tests
5. **View History**: Access appointment and prescription history

## Project Structure

```
MentalHealthSupport/
├── app.py                      # Main Flask application
├── step_inference.py           # Emotion detection model inference
├── requirements.txt            # Python dependencies
├── emotion_mapping.json        # Emotion label mappings
├── mental_health.db            # SQLite database (auto-generated)
├── emotion_roberta_model/      # Pre-trained emotion detection model
├── static/
│   ├── style.css               # Application styles
│   └── script.js               # Client-side JavaScript
├── templates/
│   ├── base.html               # Base template
│   ├── home.html               # Home page
│   ├── dashboard.html          # User dashboard
│   ├── chat.html               # Chat interface
│   ├── emotion.html            # Emotion detection page
│   ├── appointments_doctor.html # Doctor appointments view
│   ├── appointments_patient.html # Patient appointments view
│   ├── book_appointment.html   # Appointment booking
│   ├── prescriptions.html      # Doctor prescriptions view
│   ├── prescriptions_patient.html # Patient prescriptions view
│   └── auth/
│       ├── login.html          # Login page
│       └── register.html       # Registration page
└── training/                   # Training data and scripts
```

## Configuration

### Environment Variables

- `FLASK_SECRET`: Secret key for Flask sessions (default: "change-me-in-prod")
  - **Important**: Change this in production!

### Database

The application uses SQLite by default. The database file (`mental_health.db`) is automatically created on first run.

## Development

### Running in Debug Mode

The application runs in debug mode by default. For production:

1. Set `FLASK_SECRET` environment variable
2. Disable debug mode in `app.py`:
   ```python
   app.run(debug=False, host="0.0.0.0", port=5000)
   ```

### Database Reset

To reset the database, delete `mental_health.db` and restart the application.

## Notes

- The emotion detection model requires the `emotion_roberta_model` directory with all model files
- First-time model loading may take a few moments
- Chat messages are polled every 3 seconds for real-time updates
- Appointments can be deleted, which automatically deletes related chats and prescriptions
- UPI QR codes are generated using an external API (qrserver.com)

## Contributing

This is a project for mental health support. Contributions and improvements are welcome!

## Acknowledgments

- RoBERTa model for emotion detection
- Flask framework
- All open-source contributors

---

For detailed documentation, see [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)
