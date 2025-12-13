# Project Documentation - Mental Health Support Web Application

## Table of Contents

1. [Project Overview](#project-overview)
2. [Objectives](#objectives)
3. [Tech Stack](#tech-stack)
4. [System Architecture](#system-architecture)
5. [Database Schema](#database-schema)
6. [Detailed Working](#detailed-working)
7. [API Endpoints](#api-endpoints)
8. [Frontend Components](#frontend-components)
9. [Emotion Detection System](#emotion-detection-system)
10. [Security Features](#security-features)
11. [Usage Guide](#usage-guide)
12. [Installation & Setup](#installation--setup)
13. [Configuration](#configuration)
14. [Troubleshooting](#troubleshooting)

---

## Project Overview

The Mental Health Support Web Application is a comprehensive full-stack web platform designed to bridge the gap between mental health professionals and patients. The application provides a secure, user-friendly environment for appointment scheduling, real-time communication, prescription management, and AI-powered emotion analysis.

### Key Highlights

- **Dual Role System**: Separate interfaces for doctors and patients with role-based access control
- **Real-Time Communication**: AJAX-based chat system for instant messaging
- **AI Integration**: Advanced emotion detection using RoBERTa transformer model
- **Payment Integration**: UPI QR code generation for seamless transactions
- **Responsive Design**: Modern, mobile-friendly user interface

---

## Objectives

### Primary Objectives

1. **Facilitate Mental Health Care Access**
   - Enable patients to easily find and book appointments with mental health professionals
   - Provide a centralized platform for managing healthcare interactions

2. **Secure Communication Channel**
   - Implement encrypted, appointment-based chat system
   - Ensure privacy and confidentiality of patient-doctor communications

3. **Digital Prescription Management**
   - Allow doctors to create and manage digital prescriptions
   - Enable patients to access their prescription history

4. **Emotion Analysis Support**
   - Provide AI-powered emotion detection to assist in mental health assessment
   - Offer real-time emotional sentiment analysis from text input

5. **Streamlined Payment Process**
   - Integrate UPI payment system with QR code generation
   - Simplify financial transactions between patients and doctors

### Secondary Objectives

- Implement robust authentication and authorization
- Ensure data integrity with proper database relationships
- Provide intuitive user experience across all devices
- Maintain scalability for future enhancements

---

## Tech Stack

### Backend

#### Core Framework
- **Flask 3.0.0**: Lightweight Python web framework
  - Handles routing, request/response cycle
  - Template rendering with Jinja2
  - Session management

#### Database & ORM
- **Flask-SQLAlchemy 3.1.1**: SQL toolkit and ORM
  - Object-relational mapping
  - Database migrations support
  - Relationship management

- **SQLite**: Lightweight relational database
  - File-based database (`mental_health.db`)
  - No separate server required
  - ACID compliant

#### Authentication & Security
- **Flask-Login 0.6.3**: User session management
  - Login/logout functionality
  - User session tracking
  - Protected route decorators

- **Werkzeug**: Password hashing utilities
  - Secure password storage (bcrypt-based hashing)
  - Password verification

#### CORS & API
- **Flask-CORS 4.0.0**: Cross-Origin Resource Sharing
  - Enables API access from different origins
  - Configurable CORS policies

### AI/ML Components

#### Deep Learning Framework
- **PyTorch (>=2.0.0)**: Deep learning framework
  - Model inference
  - Tensor operations
  - GPU/CPU support

#### NLP & Transformers
- **Transformers (>=4.30.0)**: Hugging Face transformers library
  - Pre-trained RoBERTa model loading
  - Tokenization
  - Sequence classification

#### Data Processing
- **Pandas (>=2.0.0)**: Data manipulation
- **NumPy (>=1.24.0)**: Numerical computations
- **Scikit-learn (>=1.3.0)**: Machine learning utilities

#### Utilities
- **tqdm (>=4.65.0)**: Progress bars for model operations

### Frontend

#### Core Technologies
- **HTML5**: Semantic markup
- **CSS3**: Styling and layout
  - Custom CSS variables for theming
  - Responsive design with media queries
  - Flexbox and Grid layouts

- **JavaScript (ES6+)**: Client-side interactivity
  - DOM manipulation
  - AJAX requests (Fetch API)
  - Event handling
  - Real-time polling

#### External Services
- **Google Fonts (Roboto)**: Typography
- **QR Server API**: UPI QR code generation

### Development Tools
- **Python 3.8+**: Programming language
- **pip**: Package management

---

## System Architecture

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Client Browser                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   HTML/CSS    │  │  JavaScript   │  │   Templates  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP/HTTPS
                            │ AJAX Requests
┌───────────────────────────▼─────────────────────────────────┐
│                    Flask Application (app.py)                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Routes      │  │   Models     │  │  Middleware   │     │
│  │  - Auth       │  │  - User      │  │  - Login      │     │
│  │  - Appointments│  │  - Appointment│  │  - CORS      │     │
│  │  - Chat       │  │  - Prescription│  │  - Security  │     │
│  │  - Prescriptions│ │  - ChatMessage│  │              │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
┌───────▼────────┐                    ┌─────────▼──────────┐
│  SQLite DB     │                    │  Emotion Model      │
│  (mental_health│                    │  (RoBERTa)         │
│   .db)         │                    │  - step_inference.py│
│                │                    │  - emotion_roberta_ │
│  - Users       │                    │    model/           │
│  - Appointments│                    │                     │
│  - Prescriptions│                   │                     │
│  - ChatMessages│                    │                     │
└────────────────┘                    └─────────────────────┘
```

### Request Flow

1. **User Request**: Browser sends HTTP request to Flask server
2. **Route Matching**: Flask matches URL to appropriate route handler
3. **Authentication Check**: Flask-Login verifies user session (if protected route)
4. **Business Logic**: Route handler processes request, interacts with database
5. **Response Generation**: Template rendered or JSON response sent
6. **Client Update**: Browser receives response and updates UI

### Data Flow

1. **User Registration/Login**: Credentials → Hashed → Stored in DB
2. **Appointment Booking**: Patient input → Validation → Database insert
3. **Chat Messages**: User input → AJAX POST → Database → Polling → Display
4. **Emotion Detection**: Text input → Tokenization → Model inference → Results

---

## Database Schema

### Entity Relationship Diagram

```
User (1) ────< (N) Appointment (N) ────> (1) User
  │                    │
  │                    ├───< (N) Prescription
  │                    │
  │                    └───< (N) ChatMessage
  │
  └───< (N) Appointment (as doctor)
```

### Tables

#### 1. User Table

**Purpose**: Stores user accounts (both doctors and patients)

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | Integer | Primary Key, Auto-increment | Unique user identifier |
| name | String(120) | NOT NULL | User's full name |
| email | String(120) | UNIQUE, NOT NULL | User's email address |
| password_hash | String(255) | NOT NULL | Hashed password (bcrypt) |
| role | String(20) | NOT NULL | 'doctor' or 'patient' |
| upi_id | String(120) | NOT NULL | UPI payment ID |
| specialization | String(120) | NULLABLE | Doctor's specialization |
| created_at | DateTime | NOT NULL | Account creation timestamp |

**Relationships**:
- One-to-Many with Appointment (as patient)
- One-to-Many with Appointment (as doctor)

**Indexes**:
- Primary key on `id`
- Unique index on `email`

#### 2. Appointment Table

**Purpose**: Manages appointment bookings between patients and doctors

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | Integer | Primary Key, Auto-increment | Unique appointment identifier |
| patient_id | Integer | Foreign Key → User.id, NOT NULL | Patient user ID |
| doctor_id | Integer | Foreign Key → User.id, NOT NULL | Doctor user ID |
| slot_time | DateTime | NOT NULL | Appointment date and time |
| reason | Text | NULLABLE | Appointment reason/notes |
| status | String(20) | NOT NULL, Default: 'pending' | 'pending', 'accepted', 'rejected' |
| created_at | DateTime | NOT NULL | Appointment creation timestamp |

**Relationships**:
- Many-to-One with User (patient)
- Many-to-One with User (doctor)
- One-to-Many with Prescription
- One-to-Many with ChatMessage

**Constraints**:
- Unique constraint on (`doctor_id`, `slot_time`) - prevents double booking
- Cascade delete: Deleting appointment deletes related prescriptions and messages

**Indexes**:
- Primary key on `id`
- Foreign key indexes on `patient_id` and `doctor_id`
- Unique index on (`doctor_id`, `slot_time`)

#### 3. Prescription Table

**Purpose**: Stores prescriptions created by doctors for patients

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | Integer | Primary Key, Auto-increment | Unique prescription identifier |
| appointment_id | Integer | Foreign Key → Appointment.id, NOT NULL | Related appointment ID |
| doctor_id | Integer | Foreign Key → User.id, NOT NULL | Doctor who created prescription |
| patient_id | Integer | Foreign Key → User.id, NOT NULL | Patient for whom prescription is created |
| notes | Text | NOT NULL | Prescription notes/medication details |
| recommended_tests | Text | NULLABLE | Recommended tests or scans |
| created_at | DateTime | NOT NULL | Prescription creation timestamp |

**Relationships**:
- Many-to-One with Appointment
- Many-to-One with User (doctor)
- Many-to-One with User (patient)

**Cascade Behavior**:
- Deleted when parent appointment is deleted

#### 4. ChatMessage Table

**Purpose**: Stores chat messages between doctors and patients

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | Integer | Primary Key, Auto-increment | Unique message identifier |
| appointment_id | Integer | Foreign Key → Appointment.id, NOT NULL | Related appointment ID |
| sender_id | Integer | Foreign Key → User.id, NOT NULL | Message sender user ID |
| receiver_id | Integer | Foreign Key → User.id, NOT NULL | Message receiver user ID |
| message | Text | NOT NULL | Message content |
| created_at | DateTime | NOT NULL | Message timestamp |

**Relationships**:
- Many-to-One with Appointment
- Many-to-One with User (sender)
- Many-to-One with User (receiver)

**Cascade Behavior**:
- Deleted when parent appointment is deleted

**Indexes**:
- Primary key on `id`
- Foreign key indexes on `appointment_id`, `sender_id`, `receiver_id`
- Used for efficient message retrieval with `id > after_id` queries

---

## Detailed Working

### 1. Authentication System

#### Registration Flow

1. **User Input**: User fills registration form with:
   - Name, Email, Password
   - Role (doctor/patient)
   - UPI ID
   - Specialization (for doctors)

2. **Validation**:
   - Check all required fields are filled
   - Validate role is 'doctor' or 'patient'
   - Check email uniqueness in database

3. **Password Hashing**:
   ```python
   password_hash = generate_password_hash(password)
   ```
   - Uses Werkzeug's secure password hashing (bcrypt)
   - Salt is automatically added

4. **User Creation**:
   - Create User object with hashed password
   - Save to database
   - Flash success message

5. **Redirect**: Redirect to login page

#### Login Flow

1. **User Input**: Email and password submitted
2. **User Lookup**: Query database for user with matching email
3. **Password Verification**:
   ```python
   if user.check_password(password):
       login_user(user)
   ```
4. **Session Creation**: Flask-Login creates user session
5. **Redirect**: Redirect to dashboard

#### Session Management

- **Flask-Login** manages user sessions
- Session stored in encrypted cookie
- `@login_required` decorator protects routes
- Automatic session expiration on logout

### 2. Appointment Management System

#### Patient: Booking Appointment

1. **Doctor Selection**:
   - Patient views list of all available doctors
   - Selects doctor from dropdown

2. **Slot Selection**:
   - Patient selects date and time using datetime-local input
   - Format: `YYYY-MM-DDTHH:MM`

3. **Validation**:
   - Check doctor exists and is active
   - Verify slot time is in future
   - Check for existing appointment at same slot (prevent double booking)

4. **Appointment Creation**:
   ```python
   appointment = Appointment(
       patient_id=current_user.id,
       doctor_id=doctor.id,
       slot_time=slot_time,
       reason=reason,
       status="pending"
   )
   db.session.add(appointment)
   db.session.commit()
   ```

5. **Notification**: Success message displayed

#### Doctor: Managing Appointments

1. **View Pending Requests**:
   - Query: `Appointment.query.filter_by(doctor_id=current_user.id, status="pending")`
   - Displayed in separate "Pending Requests" section

2. **Accept/Reject**:
   - Doctor clicks Accept or Reject button
   - Status updated via AJAX POST request
   - Database updated: `appointment.status = "accepted"` or `"rejected"`

3. **View All Appointments**:
   - All appointments (pending, accepted, rejected) displayed
   - Sorted by slot_time (descending)

#### Appointment Deletion

1. **Permission Check**:
   - Doctor can delete their appointments
   - Patient can delete their appointments

2. **Cascade Delete**:
   - Deleting appointment automatically deletes:
     - All related chat messages
     - All related prescriptions
   - Handled by SQLAlchemy cascade relationships

### 3. Chat System

#### Architecture

- **Real-Time Communication**: AJAX polling (not WebSockets)
- **Polling Interval**: 3 seconds
- **Message Storage**: All messages stored in database
- **Appointment-Based**: Chat tied to specific appointment

#### Sending Messages

1. **User Input**: User types message in textarea
2. **Form Submission**: JavaScript prevents default, sends AJAX POST
3. **Server Processing**:
   ```python
   # Determine receiver
   if current_user.id == appt.doctor_id:
       receiver_id = appt.patient_id
   else:
       receiver_id = appt.doctor_id
   
   # Create message
   msg = ChatMessage(
       appointment_id=appt.id,
       sender_id=current_user.id,
       receiver_id=receiver_id,
       message=message
   )
   db.session.add(msg)
   db.session.commit()
   ```

4. **Response**: Server returns success with message ID
5. **UI Update**: Message immediately displayed in chat

#### Receiving Messages (Polling)

1. **Polling Setup**: JavaScript sets interval (3 seconds)
2. **Request**: GET request to `/api/chat/<appointment_id>/messages?after=<last_message_id>`
3. **Server Query**:
   ```python
   messages = ChatMessage.query.filter(
       ChatMessage.appointment_id == appt.id,
       ChatMessage.id > after_id
   ).order_by(ChatMessage.id.asc()).all()
   ```
4. **Response**: JSON array of new messages
5. **UI Update**: New messages appended to chat display

#### Message Display

- **Format**: 
  - Sender name (bold)
  - Role | UPI ID (muted)
  - Timestamp
  - Message content
- **Auto-scroll**: Chat automatically scrolls to bottom on new messages

### 4. Prescription Management

#### Doctor: Creating Prescription

1. **Appointment Selection**: Doctor selects appointment from dropdown
2. **Form Input**:
   - Prescription notes (required)
   - Recommended tests (optional)
3. **Validation**: Check notes are not empty
4. **Creation**:
   ```python
   presc = Prescription(
       appointment_id=appointment.id,
       doctor_id=current_user.id,
       patient_id=appointment.patient_id,
       notes=notes,
       recommended_tests=tests
   )
   db.session.add(presc)
   db.session.commit()
   ```
5. **History Update**: Prescription history reloaded via AJAX

#### Patient: Viewing Prescriptions

1. **Appointment Selection**: Patient selects appointment
2. **AJAX Request**: GET `/api/appointments/<appointment_id>/prescriptions`
3. **Server Response**: JSON array of prescriptions
4. **Display**: Prescriptions shown with:
   - Creation timestamp
   - Notes
   - Recommended tests (if any)

### 5. Emotion Detection System

#### Model Architecture

- **Base Model**: RoBERTa (Robustly Optimized BERT Pretraining Approach)
- **Task**: Sequence Classification (Multi-label)
- **Input**: Text string
- **Output**: Emotion probabilities for 28 emotion classes

#### Emotion Classes

28 emotions based on GoEmotions dataset:
- admiration, amusement, anger, annoyance, approval, caring
- confusion, curiosity, desire, disappointment, disapproval, disgust
- embarrassment, excitement, fear, gratitude, grief, joy
- love, nervousness, optimism, pride, realization, relief
- remorse, sadness, surprise, neutral

#### Inference Process

1. **Text Input**: User enters text in textarea
2. **Preprocessing**:
   - Text tokenized using RoBERTa tokenizer
   - Max length: 256 tokens
   - Padding/truncation as needed

3. **Model Inference**:
   ```python
   inputs = tokenizer(text, return_tensors="pt", 
                     max_length=256, truncation=True, padding=True)
   inputs = {k: v.to(device) for k, v in inputs.items()}
   
   with torch.no_grad():
       outputs = model(**inputs)
       logits = outputs.logits
   ```

4. **Post-processing**:
   - Apply sigmoid to get probabilities
   - Apply thresholds (if configured)
   - Select top-k emotions

5. **Response**: JSON with emotions and confidence scores
6. **Display**: Emotions shown as cards with confidence bars

#### Model Loading

- **Location**: `emotion_roberta_model/` directory
- **Files Required**:
  - `config.json`: Model configuration
  - `model.safetensors`: Model weights
  - `tokenizer_config.json`: Tokenizer settings
  - `vocab.json`: Vocabulary
  - `merges.txt`: BPE merges
  - `special_tokens_map.json`: Special tokens
  - `best_config.json`: Training configuration

- **Loading Process**:
  ```python
   tokenizer = RobertaTokenizer.from_pretrained(model_path)
   model = RobertaForSequenceClassification.from_pretrained(model_path)
   model.to(device)
   model.eval()
   ```

### 6. UPI Payment Integration

#### QR Code Generation

1. **UPI ID Display**: User's UPI ID displayed in various places
2. **Click Handler**: Clicking UPI ID triggers modal
3. **QR Generation**:
   ```javascript
   const upiPayload = `upi://pay?pa=${encodeURIComponent(upiId)}&pn=${encodeURIComponent(userName)}`;
   const qrUrl = `https://api.qrserver.com/v1/create-qr-code/?size=220x220&data=${encodeURIComponent(upiPayload)}`;
   ```
4. **Display**: QR code shown in modal
5. **Payment**: User scans QR code with UPI app

#### UPI Payload Format

- **Format**: `upi://pay?pa=<UPI_ID>&pn=<USER_NAME>`
- **Parameters**:
  - `pa`: Payee address (UPI ID)
  - `pn`: Payee name

---

## API Endpoints

### Public Endpoints

#### GET `/`
- **Description**: Home page
- **Response**: HTML (home.html template)

#### GET `/emotion`
- **Description**: Emotion detection page
- **Response**: HTML (emotion.html template)

#### GET `/register`
- **Description**: Registration page
- **Response**: HTML (register.html template)

#### POST `/register`
- **Description**: User registration
- **Request Body**: Form data (name, email, password, role, upi_id, specialization)
- **Response**: Redirect to login or error flash

#### GET `/login`
- **Description**: Login page
- **Response**: HTML (login.html template)

#### POST `/login`
- **Description**: User login
- **Request Body**: Form data (email, password)
- **Response**: Redirect to dashboard or error flash

### Protected Endpoints (Require Authentication)

#### GET `/dashboard`
- **Description**: User dashboard
- **Authentication**: Required
- **Response**: HTML (dashboard.html template)

#### GET `/appointments`
- **Description**: Appointments list page
- **Authentication**: Required
- **Response**: HTML (appointments_doctor.html or appointments_patient.html)

#### POST `/appointments`
- **Description**: Create new appointment
- **Authentication**: Required (Patient only)
- **Request Body**: JSON (doctor_id, slot_time, reason)
- **Response**: JSON (success, appointment_id)

#### POST `/appointments/<id>/status`
- **Description**: Update appointment status
- **Authentication**: Required (Doctor only)
- **Request Body**: JSON (status: "accepted" or "rejected")
- **Response**: JSON (success, status)

#### DELETE `/appointments/<id>`
- **Description**: Delete appointment
- **Authentication**: Required
- **Response**: JSON (success)

#### GET `/appointments/book`
- **Description**: Book appointment page
- **Authentication**: Required (Patient only)
- **Response**: HTML (book_appointment.html template)

#### GET `/chat`
- **Description**: Chat page
- **Authentication**: Required
- **Response**: HTML (chat.html template)

#### GET `/api/chat/<appointment_id>/messages`
- **Description**: Get chat messages
- **Authentication**: Required
- **Query Parameters**: `after` (message ID to fetch messages after)
- **Response**: JSON array of messages

#### POST `/api/chat/<appointment_id>/messages`
- **Description**: Send chat message
- **Authentication**: Required
- **Request Body**: JSON (message)
- **Response**: JSON (success, id)

#### GET `/prescriptions`
- **Description**: Prescriptions page
- **Authentication**: Required
- **Response**: HTML (prescriptions.html or prescriptions_patient.html)

#### POST `/appointments/<id>/prescription`
- **Description**: Create prescription
- **Authentication**: Required (Doctor only)
- **Request Body**: JSON (notes, tests)
- **Response**: JSON (success)

#### GET `/api/appointments/<id>/prescriptions`
- **Description**: Get prescriptions for appointment
- **Authentication**: Required
- **Response**: JSON array of prescriptions

#### POST `/api/predict`
- **Description**: Predict emotions from text
- **Authentication**: Not required
- **Request Body**: JSON (text, top_k)
- **Response**: JSON (success, text, emotions)

#### GET `/api/health`
- **Description**: Health check endpoint
- **Response**: JSON (status, model_loaded)

#### GET `/api/emotions`
- **Description**: Get available emotion classes
- **Response**: JSON (emotions, count)

#### POST `/account/delete`
- **Description**: Delete user account
- **Authentication**: Required
- **Response**: Redirect to home

#### GET `/logout`
- **Description**: Logout user
- **Authentication**: Required
- **Response**: Redirect to home

---

## Frontend Components

### Base Template (base.html)

**Purpose**: Base template for all pages

**Components**:
- Navigation bar with:
  - Brand logo/name
  - Emotion Analysis link
  - Dashboard link (if authenticated)
  - User name chip
  - Login/Logout links
- Flash message container
- UPI QR modal
- Script includes

### Dashboard (dashboard.html)

**Features**:
- Role-based card grid
- Doctor view: Appointments, Prescriptions, Chat, Emotion Detection
- Patient view: Prescriptions, Book Appointment, My Appointments, Chat, Emotion Detection
- Account settings section

### Chat Interface (chat.html)

**Components**:
- Appointment selector dropdown
- Chat header (shows selected user info)
- Messages container (scrollable)
- Message input form

**JavaScript Functions**:
- `initChat()`: Initialize chat system
- `startPolling()`: Start message polling
- `fetchMessages()`: Fetch new messages via AJAX
- `appendMessage()`: Display message in UI
- `sendMessage()`: Send message via AJAX

### Emotion Detection (emotion.html)

**Components**:
- Text input textarea
- Top-k selector (3, 5, 10, All)
- Analyze button
- Results container
- Error display

**JavaScript Functions**:
- `initEmotionDetection()`: Initialize emotion detection
- `analyzeEmotions()`: Send text to API, display results
- `displayResults()`: Show emotion cards with confidence bars

### Appointment Pages

#### Doctor View (appointments_doctor.html)
- Pending requests table
- All appointments table
- Accept/Reject buttons
- Chat and Delete buttons

#### Patient View (appointments_patient.html)
- Appointments table
- Chat, Prescription, Delete buttons
- Book new appointment button

### Prescription Pages

#### Doctor View (prescriptions.html)
- Appointment selector
- Prescription form (notes, tests)
- Prescription history display

#### Patient View (prescriptions_patient.html)
- Appointment selector
- Prescription history display (read-only)

---

## Emotion Detection System

### Model Details

**Architecture**: RoBERTa-base for sequence classification

**Input Processing**:
1. Text tokenization using RoBERTa tokenizer
2. Maximum sequence length: 256 tokens
3. Padding and truncation as needed
4. Conversion to PyTorch tensors

**Inference**:
1. Model forward pass
2. Logits extraction
3. Sigmoid activation for multi-label probabilities
4. Threshold application (if configured)
5. Top-k selection

**Output Format**:
```json
{
  "success": true,
  "text": "input text",
  "emotions": [
    {
      "emotion": "joy",
      "confidence": 0.85
    },
    ...
  ]
}
```

### Integration

**Loading**:
- Model loaded at application startup
- Global `detector` variable
- Error handling if model files missing

**API Endpoint**:
- `/api/predict` accepts POST requests
- Validates text input
- Returns emotion predictions

**Frontend Integration**:
- Text input from user
- AJAX POST to `/api/predict`
- Results displayed as cards with confidence bars

---

## Security Features

### Password Security

- **Hashing**: Bcrypt-based password hashing via Werkzeug
- **Salt**: Automatic salt generation
- **Storage**: Only hashed passwords stored, never plain text

### Authentication

- **Session Management**: Flask-Login handles user sessions
- **Protected Routes**: `@login_required` decorator
- **Session Cookies**: Encrypted session cookies
- **Logout**: Proper session cleanup

### Authorization

- **Role-Based Access**: 
  - `ensure_doctor()`: Doctor-only routes
  - `ensure_patient()`: Patient-only routes
- **Resource Ownership**: Users can only access their own data
- **Appointment Access**: Only doctor and patient of appointment can access

### Data Validation

- **Input Validation**: Server-side validation of all inputs
- **SQL Injection Prevention**: SQLAlchemy ORM prevents SQL injection
- **XSS Prevention**: Jinja2 auto-escapes template variables

### CORS Configuration

- **Flask-CORS**: Configured for API access
- **Security Headers**: Can be added for production

---

## Usage Guide

### For Patients

#### Getting Started

1. **Register Account**:
   - Go to registration page
   - Fill in: Name, Email, Password
   - Select role: "patient"
   - Enter UPI ID
   - Submit form

2. **Login**:
   - Enter email and password
   - Click Login
   - Redirected to dashboard

#### Booking Appointment

1. **Navigate**: Dashboard → Book Appointment
2. **Select Doctor**: Choose from available doctors
3. **Select Date/Time**: Use datetime picker
4. **Add Reason**: Optional notes
5. **Submit**: Appointment request created (status: pending)

#### Managing Appointments

1. **View Appointments**: Dashboard → My Appointments
2. **Check Status**: See pending/accepted/rejected status
3. **Chat**: Click Chat button (if not rejected)
4. **View Prescription**: Click Prescription button
5. **Delete**: Click Delete button (removes appointment and related data)

#### Using Chat

1. **Access**: Dashboard → Chat or Appointments → Chat
2. **Select Appointment**: Choose appointment from dropdown
3. **View Messages**: Previous messages displayed
4. **Send Message**: Type in textarea, click Send
5. **Real-Time Updates**: Messages auto-refresh every 3 seconds

#### Viewing Prescriptions

1. **Access**: Dashboard → My Prescriptions
2. **Select Appointment**: Choose appointment from dropdown
3. **View Details**: See prescription notes and recommended tests
4. **Multiple Prescriptions**: All prescriptions for appointment shown

#### Emotion Analysis

1. **Access**: Dashboard → Emotion Detection or Navbar → Emotion Analysis
2. **Enter Text**: Type or paste text to analyze
3. **Select Top-K**: Choose number of emotions to show (3, 5, 10, All)
4. **Analyze**: Click Analyze button
5. **View Results**: See emotions with confidence percentages

### For Doctors

#### Getting Started

1. **Register Account**:
   - Go to registration page
   - Fill in: Name, Email, Password
   - Select role: "doctor"
   - Enter UPI ID
   - Enter Specialization
   - Submit form

2. **Login**: Same as patient

#### Managing Appointments

1. **View Pending**: Dashboard → Appointments
2. **Accept/Reject**: Click Accept or Reject button
3. **View All**: See all appointments (all statuses)
4. **Chat**: Click Chat button to communicate with patient
5. **Delete**: Remove appointment if needed

#### Creating Prescriptions

1. **Access**: Dashboard → Prescriptions
2. **Select Appointment**: Choose accepted appointment
3. **Enter Notes**: Required prescription details
4. **Enter Tests**: Optional recommended tests
5. **Save**: Prescription created and visible to patient

#### Using Chat

- Same as patient workflow
- Can chat with any patient who has appointment

#### Emotion Analysis

- Same as patient workflow
- Can use for patient assessment support

---

## Installation & Setup

### Step-by-Step Installation

#### 1. Prerequisites Check

```bash
# Check Python version (need 3.8+)
python --version

# Check pip
pip --version
```

#### 2. Clone/Download Project

```bash
# If using Git
git clone <repository-url>
cd v2_WebApp

# Or download and extract ZIP file
```

#### 3. Create Virtual Environment

**Windows**:
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac**:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

**Expected Output**: All packages installed successfully

**Note**: PyTorch installation may take time depending on system

#### 5. Verify Model Files

Check that `emotion_roberta_model/` directory contains:
- config.json
- model.safetensors
- tokenizer_config.json
- vocab.json
- merges.txt
- special_tokens_map.json
- best_config.json

#### 6. Run Application

```bash
python app.py
```

**Expected Output**:
```
==============================================================
🚀 Starting Mental Health Support Web Application
==============================================================

✅ Server starting on http://localhost:5000
📝 Open your browser and navigate to http://localhost:5000
==============================================================
```

#### 7. Access Application

- Open browser: `http://localhost:5000`
- Database (`mental_health.db`) created automatically on first run

### Troubleshooting Installation

#### Issue: Python not found
**Solution**: Install Python 3.8+ from python.org

#### Issue: pip not found
**Solution**: 
```bash
python -m ensurepip --upgrade
```

#### Issue: PyTorch installation fails
**Solution**: 
- Check system requirements
- Try CPU-only version: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

#### Issue: Model files missing
**Solution**: Ensure `emotion_roberta_model/` directory exists with all files

#### Issue: Port 5000 already in use
**Solution**: 
- Change port in `app.py`: `app.run(debug=True, host="0.0.0.0", port=5001)`
- Or stop other service using port 5000

---

## Configuration

### Environment Variables

#### FLASK_SECRET

**Purpose**: Secret key for Flask session encryption

**Setting**:
```bash
# Windows
set FLASK_SECRET=your-secret-key-here

# Linux/Mac
export FLASK_SECRET=your-secret-key-here
```

**Default**: "change-me-in-prod" (change in production!)

**Generation**:
```python
import secrets
secrets.token_hex(32)
```

### Database Configuration

**Current**: SQLite (file-based)

**Location**: `mental_health.db` in project root

**Changing Database** (e.g., PostgreSQL):
```python
# In app.py
app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://user:password@localhost/dbname"
```

### Model Configuration

**Location**: `emotion_roberta_model/best_config.json`

**Parameters**:
- `max_length`: Maximum token length (default: 256)
- `batch_size`: Inference batch size (default: 16)
- `device`: 'cuda' or 'cpu' (auto-detected)

### Application Settings

**Debug Mode**: 
- Current: `debug=True`
- Production: Set to `False`

**Host**:
- Current: `0.0.0.0` (all interfaces)
- Development: `127.0.0.1` (localhost only)

**Port**:
- Default: 5000
- Change in `app.py`: `port=5001`

---

## Troubleshooting

### Common Issues

#### 1. Database Errors

**Issue**: "Table doesn't exist"
**Solution**: Delete `mental_health.db` and restart application

**Issue**: "Database locked"
**Solution**: Close other connections, restart application

#### 2. Model Loading Errors

**Issue**: "Model directory not found"
**Solution**: Verify `emotion_roberta_model/` exists with all files

**Issue**: "CUDA out of memory"
**Solution**: Model will fall back to CPU automatically

#### 3. Chat Not Working

**Issue**: Messages not appearing
**Solution**: 
- Check browser console for errors
- Verify AJAX requests in Network tab
- Check appointment_id is valid

#### 4. Authentication Issues

**Issue**: "Invalid credentials"
**Solution**: 
- Verify email and password
- Check user exists in database
- Try registering new account

**Issue**: Session expires quickly
**Solution**: Check Flask-Login configuration

#### 5. CORS Errors

**Issue**: "CORS policy blocked"
**Solution**: Flask-CORS should handle this, check configuration

#### 6. Emotion Detection Not Working

**Issue**: "Model not loaded"
**Solution**: 
- Check model files exist
- Check console for loading errors
- Restart application

**Issue**: Slow predictions
**Solution**: 
- Normal for first prediction (model loading)
- Consider GPU for faster inference

### Debug Mode

**Enable Debug**:
- Already enabled by default
- Shows detailed error messages
- Auto-reloads on code changes

**Disable for Production**:
```python
app.run(debug=False, host="0.0.0.0", port=5000)
```

### Logging

**Add Logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**View Logs**: Check console output for errors and warnings

---

## Additional Notes

### Performance Considerations

- **Database**: SQLite suitable for small-medium scale
- **Chat Polling**: 3-second interval balances responsiveness and server load
- **Model Inference**: First prediction slower (model loading), subsequent faster
- **Caching**: Consider caching model predictions for repeated text

### Scalability

**Current Limitations**:
- SQLite for single-server deployment
- AJAX polling (not WebSockets)
- Single-threaded Flask (development mode)

**Production Recommendations**:
- Use PostgreSQL/MySQL for database
- Implement WebSockets for real-time chat
- Use Gunicorn/uWSGI with multiple workers
- Add Redis for session management
- Implement proper logging and monitoring

### Security Recommendations

1. **Change Secret Key**: Set strong `FLASK_SECRET`
2. **HTTPS**: Use HTTPS in production
3. **Rate Limiting**: Add rate limiting for API endpoints
4. **Input Sanitization**: Additional validation for user inputs
5. **SQL Injection**: Already protected by SQLAlchemy ORM
6. **XSS**: Already protected by Jinja2 auto-escaping

### Future Enhancements

- Email notifications
- File uploads (prescription attachments)
- Video call integration
- Mobile app
- Advanced analytics
- Multi-language support
- Payment gateway integration
- Appointment reminders

---

## Conclusion

This Mental Health Support Web Application provides a comprehensive platform for connecting patients with mental health professionals. With features like appointment management, real-time chat, prescription handling, and AI-powered emotion detection, it offers a complete solution for digital mental health care.

The application is built with modern web technologies, ensuring scalability, security, and user-friendliness. The detailed documentation provided here should help developers understand, deploy, and extend the application.

For questions or issues, refer to the troubleshooting section or check the code comments for implementation details.

---

**Last Updated**: [Current Date]
**Version**: 1.0.0
**Maintainer**: [Your Name/Team]
