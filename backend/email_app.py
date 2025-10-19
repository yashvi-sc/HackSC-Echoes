from flask import Flask, request, jsonify
import os, smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = Flask(__name__)

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))          # 465=SSL, 587=STARTTLS
SMTP_USER = "kuchbhi.naamhaimera@gmail.com"                     # e.g. youracct@gmail.com OR Mailtrap user
SMTP_PASS = "junoo2324"                      # Gmail App Password OR Mailtrap pass
FROM_EMAIL ="kuchbhi.naamhaimera@gmail.com"

def send_simple_email(subject: str, body_text: str, to_email: str, cc_email: str|None=None):
    if not (SMTP_USER and SMTP_PASS and FROM_EMAIL):
        raise RuntimeError("Missing SMTP_USER/SMTP_PASS/FROM_EMAIL env vars")

    msg = MIMEMultipart()
    msg["From"] = FROM_EMAIL
    msg["To"] = to_email
    if cc_email:
        msg["Cc"] = cc_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body_text, "plain", "utf-8"))

    # ---- SSL (465). For STARTTLS on 587, see the comment below. ----
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context) as s:
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)

@app.route("/health", methods=["GET"])
def health():
    ok = bool(SMTP_USER and SMTP_PASS and FROM_EMAIL)
    return jsonify({"ok": ok, "email_ready": ok})

@app.route("/email_report", methods=["POST"])
def email_report():
    try:
        data = request.get_json(force=True)
        doctor = (data.get("doctor_email") or "").strip()
        patient = (data.get("patient_email") or "").strip()
        subject = data.get("subject") or "Speech Therapy Session Report"
        report_text = (data.get("report_text") or "").strip()

        if not doctor:
            return jsonify({"ok": False, "error": "doctor_email required"}), 400
        if not report_text:
            return jsonify({"ok": False, "error": "report_text required"}), 400

        send_simple_email(subject, report_text, doctor, patient or None)
        return jsonify({"ok": True, "message": "sent"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == "__main__":
    # If you prefer STARTTLS:
    #   set SMTP_PORT=587 and replace SMTP_SSL block with:
    #   s = smtplib.SMTP(SMTP_HOST, SMTP_PORT); s.starttls(context=ssl.create_default_context())
    app.run(host="0.0.0.0", port=5057, debug=True)


