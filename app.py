"""
Full-stack Mental Health Support Web Application with Emotion Detection.

Features:
- Auth with Doctor/Patient roles
- Appointment booking + status management
- Chat between doctor & patient (AJAX polling)
- Prescriptions and history
- Emotion detection powered by existing model
- UPI display + QR generation link
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Optional

from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_cors import CORS
from flask_login import (
    LoginManager,
    UserMixin,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import UniqueConstraint
from werkzeug.security import check_password_hash, generate_password_hash

from step_inference import ImprovedEmotionDetector

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(BASE_DIR, "mental_health.db")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET", "change-me-in-prod")
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
CORS(app)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"


# ---------- Models ----------
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # doctor | patient
    upi_id = db.Column(db.String(120), nullable=False)
    specialization = db.Column(db.String(120))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    appointments_as_patient = db.relationship(
        "Appointment",
        foreign_keys="Appointment.patient_id",
        backref="patient",
        cascade="all, delete",
    )
    appointments_as_doctor = db.relationship(
        "Appointment",
        foreign_keys="Appointment.doctor_id",
        backref="doctor",
        cascade="all, delete",
    )

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    @property
    def display_label(self) -> str:
        return f"{self.role.title()} | {self.upi_id}"


class Appointment(db.Model):
    __table_args__ = (UniqueConstraint("doctor_id", "slot_time", name="uq_doctor_slot"),)

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    slot_time = db.Column(db.DateTime, nullable=False)
    reason = db.Column(db.Text)
    status = db.Column(db.String(20), default="pending")  # pending | accepted | rejected
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    prescriptions = db.relationship(
        "Prescription", backref="appointment", cascade="all, delete"
    )
    messages = db.relationship("ChatMessage", backref="appointment", cascade="all, delete")


class Prescription(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    appointment_id = db.Column(db.Integer, db.ForeignKey("appointment.id"), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    patient_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    notes = db.Column(db.Text, nullable=False)
    recommended_tests = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    appointment_id = db.Column(db.Integer, db.ForeignKey("appointment.id"), nullable=False)
    sender_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    receiver_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    message = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# ---------- Auth helpers ----------
@login_manager.user_loader
def load_user(user_id: str) -> Optional[User]:
    return User.query.get(int(user_id))


def ensure_doctor():
    if not current_user.is_authenticated or current_user.role != "doctor":
        abort(403)


def ensure_patient():
    if not current_user.is_authenticated or current_user.role != "patient":
        abort(403)


# ---------- Emotion model loading ----------
detector: Optional[ImprovedEmotionDetector] = None


def load_model() -> bool:
    global detector
    try:
        model_path = "emotion_roberta_model"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        detector = ImprovedEmotionDetector(model_path=model_path, max_length=256, batch_size=16)
        return True
    except Exception as exc:  # pragma: no cover - logging only
        print(f"Error loading model: {exc}")
        return False


# ---------- Routes: Public ----------
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/emotion")
def emotion_page():
    return render_template("emotion.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").lower().strip()
        password = request.form.get("password", "")
        role = request.form.get("role", "")
        upi_id = request.form.get("upi_id", "").strip()
        specialization = request.form.get("specialization", "").strip()

        if not all([name, email, password, role, upi_id]):
            flash("All required fields must be filled.", "error")
            return redirect(url_for("register"))

        if role not in ("doctor", "patient"):
            flash("Invalid user type.", "error")
            return redirect(url_for("register"))

        if User.query.filter_by(email=email).first():
            flash("Email already registered.", "error")
            return redirect(url_for("register"))

        user = User(name=name, email=email, role=role, upi_id=upi_id, specialization=specialization)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash("Registration successful. Please log in.", "success")
        return redirect(url_for("login"))
    return render_template("auth/register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").lower().strip()
        password = request.form.get("password", "")
        user = User.query.filter_by(email=email).first()
        if not user or not user.check_password(password):
            flash("Invalid credentials.", "error")
            return redirect(url_for("login"))
        login_user(user)
        return redirect(url_for("dashboard"))
    return render_template("auth/login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out.", "info")
    return redirect(url_for("home"))


@app.route("/account/delete", methods=["POST"])
@login_required
def delete_account():
    """Delete user account"""
    user_id = current_user.id
    user_name = current_user.name
    logout_user()
    user = User.query.get(user_id)
    if user:
        db.session.delete(user)
        db.session.commit()
    flash(f"Account for {user_name} has been deleted.", "info")
    return redirect(url_for("home"))


# ---------- Routes: Dashboard ----------
@app.route("/dashboard")
@login_required
def dashboard():
    """Simple dashboard with links to separate pages"""
    return render_template("dashboard.html")


# ---------- Routes: Appointments ----------
@app.route("/appointments")
@login_required
def appointments_page():
    """Appointments list page"""
    if current_user.role == "doctor":
        pending = (
            Appointment.query.filter_by(doctor_id=current_user.id, status="pending")
            .order_by(Appointment.slot_time)
            .all()
        )
        all_appointments = (
            Appointment.query.filter_by(doctor_id=current_user.id)
            .order_by(Appointment.slot_time.desc())
            .all()
        )
        return render_template(
            "appointments_doctor.html",
            pending=pending,
            appointments=all_appointments,
        )
    else:
        my_appointments = (
            Appointment.query.filter_by(patient_id=current_user.id)
            .order_by(Appointment.slot_time.desc())
            .all()
        )
        return render_template("appointments_patient.html", appointments=my_appointments)


@app.route("/appointments/book")
@login_required
def book_appointment_page():
    """Book appointment page for patients"""
    ensure_patient()
    # Get all doctors - ensure we're getting the complete list
    doctors = db.session.query(User).filter(User.role == "doctor").order_by(User.name).all()
    return render_template("book_appointment.html", doctors=doctors)


@app.route("/appointments", methods=["POST"])
@login_required
def create_appointment():
    ensure_patient()
    data = request.get_json() or request.form
    doctor_id = data.get("doctor_id")
    slot_time_raw = data.get("slot_time")
    reason = data.get("reason", "").strip()

    if not doctor_id or not slot_time_raw:
        return jsonify({"error": "Doctor and slot time are required."}), 400

    try:
        slot_time = datetime.fromisoformat(slot_time_raw.replace("Z", ""))
    except ValueError:
        return jsonify({"error": "Invalid slot time format."}), 400

    doctor = User.query.filter_by(id=int(doctor_id), role="doctor").first()
    if not doctor:
        return jsonify({"error": "Doctor not found."}), 404

    # Prevent double booking for doctor slot
    existing = Appointment.query.filter_by(doctor_id=doctor.id, slot_time=slot_time).first()
    if existing:
        return jsonify({"error": "Slot already booked with this doctor."}), 409

    appointment = Appointment(
        patient_id=current_user.id,
        doctor_id=doctor.id,
        slot_time=slot_time,
        reason=reason,
        status="pending",
    )
    db.session.add(appointment)
    db.session.commit()
    return jsonify({"success": True, "appointment_id": appointment.id})


@app.route("/appointments/<int:appointment_id>/status", methods=["POST"])
@login_required
def update_appointment_status(appointment_id: int):
    ensure_doctor()
    data = request.get_json() or request.form
    status = data.get("status")
    if status not in {"accepted", "rejected"}:
        return jsonify({"error": "Invalid status."}), 400

    appointment = Appointment.query.filter_by(id=appointment_id, doctor_id=current_user.id).first()
    if not appointment:
        return jsonify({"error": "Appointment not found."}), 404

    appointment.status = status
    db.session.commit()
    return jsonify({"success": True, "status": status})


@app.route("/appointments/<int:appointment_id>", methods=["DELETE"])
@login_required
def delete_appointment(appointment_id: int):
    """Delete an appointment (and related chats and prescriptions via cascade)"""
    appointment = Appointment.query.get_or_404(appointment_id)
    
    # Check permissions: doctor can delete their appointments, patient can delete their appointments
    if current_user.role == "doctor":
        if appointment.doctor_id != current_user.id:
            abort(403)
    else:
        if appointment.patient_id != current_user.id:
            abort(403)
    
    # Cascade delete will automatically handle messages and prescriptions
    db.session.delete(appointment)
    db.session.commit()
    return jsonify({"success": True})


# ---------- Routes: Prescriptions ----------
@app.route("/prescriptions")
@login_required
def prescriptions_page():
    """Prescriptions page - different views for doctors and patients"""
    if current_user.role == "doctor":
        appointments = (
            Appointment.query.filter_by(doctor_id=current_user.id, status="accepted")
            .order_by(Appointment.slot_time.desc())
            .all()
        )
        return render_template("prescriptions.html", appointments=appointments)
    else:
        # Patient view: show all their appointments with prescriptions
        appointments = (
            Appointment.query.filter_by(patient_id=current_user.id)
            .filter(Appointment.status != "rejected")
            .order_by(Appointment.slot_time.desc())
            .all()
        )
        return render_template("prescriptions_patient.html", appointments=appointments)


@app.route("/appointments/<int:appointment_id>/prescription", methods=["POST"])
@login_required
def create_prescription(appointment_id: int):
    ensure_doctor()
    appointment = Appointment.query.filter_by(id=appointment_id, doctor_id=current_user.id).first()
    if not appointment:
        return jsonify({"error": "Appointment not found."}), 404

    data = request.get_json() or request.form
    notes = data.get("notes", "").strip()
    tests = data.get("tests", "").strip()
    if not notes:
        return jsonify({"error": "Prescription notes required."}), 400

    presc = Prescription(
        appointment_id=appointment.id,
        doctor_id=current_user.id,
        patient_id=appointment.patient_id,
        notes=notes,
        recommended_tests=tests,
    )
    db.session.add(presc)
    db.session.commit()
    return jsonify({"success": True})


@app.route("/api/appointments/<int:appointment_id>/prescriptions")
@login_required
def list_prescriptions(appointment_id: int):
    appointment = Appointment.query.get_or_404(appointment_id)
    if current_user.id not in {appointment.doctor_id, appointment.patient_id}:
        abort(403)

    prescs = (
        Prescription.query.filter_by(appointment_id=appointment.id)
        .order_by(Prescription.created_at.desc())
        .all()
    )
    return jsonify(
        [
            {
                "id": p.id,
                "notes": p.notes,
                "tests": p.recommended_tests,
                "created_at": p.created_at.isoformat(),
            }
            for p in prescs
        ]
    )


# ---------- Routes: Chat ----------
@app.route("/chat")
@login_required
def chat_page():
    """Chat page"""
    if current_user.role == "doctor":
        appointments = (
            Appointment.query.filter_by(doctor_id=current_user.id)
            .filter(Appointment.status != "rejected")
            .order_by(Appointment.slot_time.desc())
            .all()
        )
    else:
        appointments = (
            Appointment.query.filter_by(patient_id=current_user.id)
            .filter(Appointment.status != "rejected")
            .order_by(Appointment.slot_time.desc())
            .all()
        )
    # Get opposite user info for each appointment (for QR code)
    appointment_data = []
    for appt in appointments:
        if current_user.role == "doctor":
            opposite_user = appt.patient
        else:
            opposite_user = appt.doctor
        appointment_data.append({
            "appointment": appt,
            "opposite_user": opposite_user
        })
    return render_template("chat.html", appointment_data=appointment_data)


def _get_participating_appointment(appointment_id: int) -> Appointment:
    appt = Appointment.query.get_or_404(appointment_id)
    if current_user.id not in {appt.doctor_id, appt.patient_id}:
        abort(403)
    return appt


@app.route("/api/chat/<int:appointment_id>/messages")
@login_required
def get_chat_messages(appointment_id: int):
    appt = _get_participating_appointment(appointment_id)
    after_id = request.args.get("after", default=0, type=int)
    messages = (
        ChatMessage.query.filter(
            ChatMessage.appointment_id == appt.id, ChatMessage.id > after_id
        )
        .order_by(ChatMessage.id.asc())
        .all()
    )
    return jsonify(
        [
            {
                "id": m.id,
                "message": m.message,
                "created_at": m.created_at.isoformat(),
                "sender_id": m.sender_id,
                "receiver_id": m.receiver_id,
                "sender_name": User.query.get(m.sender_id).name,
                "sender_display": User.query.get(m.sender_id).display_label,
            }
            for m in messages
        ]
    )


@app.route("/api/chat/<int:appointment_id>/messages", methods=["POST"])
@login_required
def send_chat_message(appointment_id: int):
    appt = _get_participating_appointment(appointment_id)
    payload = request.get_json() or {}
    message = payload.get("message", "").strip()
    if not message:
        return jsonify({"error": "Message cannot be empty."}), 400

    if current_user.id == appt.doctor_id:
        receiver_id = appt.patient_id
    else:
        receiver_id = appt.doctor_id

    msg = ChatMessage(
        appointment_id=appt.id,
        sender_id=current_user.id,
        receiver_id=receiver_id,
        message=message,
    )
    db.session.add(msg)
    db.session.commit()
    return jsonify({"success": True, "id": msg.id})


# ---------- Routes: Emotion APIs ----------
@app.route("/api/predict", methods=["POST"])
def predict():
    if detector is None:
        return jsonify({"error": "Model not loaded. Please restart the server."}), 500

    data = request.get_json() or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Text input is required"}), 400

    top_k = data.get("top_k", 5)
    predictions = detector.predict_single(text, top_k=top_k)
    return jsonify({"success": True, "text": text, "emotions": predictions})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": detector is not None})


@app.route("/api/emotions", methods=["GET"])
def get_emotions():
    if detector is None:
        return jsonify({"error": "Model not loaded"}), 500
    return jsonify({"emotions": detector.emotions, "count": len(detector.emotions)})


# ---------- Bootstrap DB & Model ----------
def init_db():
    db.create_all()
    # Remove demo doctor if exists
    demo_doc = User.query.filter_by(email="doctor@example.com").first()
    if demo_doc:
        db.session.delete(demo_doc)
        db.session.commit()


with app.app_context():
    init_db()
    load_model()


if __name__ == "__main__":
    print("=" * 70)
    print("🚀 Starting Mental Health Support Web Application")
    print("=" * 70)
    print("\n✅ Server starting on http://localhost:5000")
    print("📝 Open your browser and navigate to http://localhost:5000")
    print("=" * 70)
    app.run(debug=True, host="0.0.0.0", port=5000)
