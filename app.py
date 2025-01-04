import cv2
import easyocr
import os
import re
import threading
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
from fpdf import FPDF
import json
from datetime import datetime

app = Flask(__name__)

# Upload folder configuration
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure challans folder exists
CHALLAN_FOLDER = "static/challans"
os.makedirs(CHALLAN_FOLDER, exist_ok=True)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Haarcascade file path
harcascade = "model/haarcascade_russian_plate_number.xml"

# Dictionary to store car numbers and owners
car_data = {
    "HR26BP3543": "Aditya Das",
    "HR26CT4063": "Sima Paul",
    "DL3CAY9324": "Prithviraj Singh",
    "MH20EJ0364": "Sayam Dogra",
    "MH14EU3498": "Abhigyan Sengupta",
    "TS03EN6675": "VINAY",
}

detected_plates = []  # List to store detected plates
detected_plates_lock = threading.Lock()  # Lock for thread-safe access

# Path for the challan history JSON file
CHALLAN_HISTORY_FILE = 'static/challan_history.json'

# Function to process uploaded images
def process_image(img_path):
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initialize plate cascade classifier
    plate_cascade = cv2.CascadeClassifier(harcascade)
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    min_area = 500
    result_text = ""
    count = 0

    for (x, y, w, h) in plates:
        area = w * h

        if area > min_area:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
            img_roi = img[y: y + h, x: x + w]

            # Perform OCR on the detected plate region
            result = reader.readtext(img_roi)

            if result:
                plate_text = re.sub(r'\W+', '', result[0][1])
                owner_name = car_data.get(plate_text, "Unknown")
                cv2.putText(img, f"Owner: {owner_name}", (x, y - 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 2)
                cv2.putText(img, f"Plate: {plate_text}", (x, y - 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 2)
                result_text += f"Plate {count + 1}: {plate_text} (Owner: {owner_name})\n"
            count += 1

    # Save processed image
    result_image_path = os.path.join("static", "result.jpg")
    cv2.imwrite(result_image_path, img)

    return result_text


# Function to process live detection frames
def detect_and_draw(frame):
    global detected_plates
    plate_cascade = cv2.CascadeClassifier(harcascade)
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    for (x, y, w, h) in plates:
        img_roi = frame[y: y + h, x: x + w]
        result = reader.readtext(img_roi)

        if result:
            plate_text = re.sub(r'\W+', '', result[0][1])  # Clean plate text
            owner_name = car_data.get(plate_text, "Unknown")

            # Add detected plate to the list if not already added
            with detected_plates_lock:
                if plate_text not in detected_plates:
                    detected_plates.append(f"{plate_text} ({owner_name})")

            # Draw rectangle and text on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{plate_text} ({owner_name})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame


# Routes
@app.route("/")
def default_redirect():
    """Redirect to the home page."""
    return redirect(url_for("home_page"))


@app.route("/home")
def home_page():
    """Render the home page with options."""
    return render_template("home.html")


@app.route("/live")
def live_detection_page():
    """Render the live detection page."""
    return render_template("live.html")


@app.route("/live-detection-feed")
def live_detection_feed():
    """Stream video frames for live detection."""
    def generate_frames():
        cap = cv2.VideoCapture(0)  # Open webcam
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Process the frame for detection
            frame = detect_and_draw(frame)

            # Encode the frame as JPEG
            _, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()

            # Yield the frame in HTTP response format
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

        cap.release()

    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/detected-plates")
def get_detected_plates():
    """Return the list of detected plates as JSON."""
    with detected_plates_lock:
        return jsonify(detected_plates)


@app.route("/image-upload", methods=["GET", "POST"])
def image_plate_detection():
    """Handle image upload for plate detection."""
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            if image.filename == "":
                return redirect(request.url)

            img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(img_path)
            result_text = process_image(img_path)
            return render_template("result.html", result_text=result_text, result_image="static/result.jpg")
    return render_template("index.html")


@app.route("/echallan")
def echallan_page():
    """Render the E-Challan form page."""
    return render_template("echallan.html")

@app.route("/generate_challan", methods=["POST"])
def generate_challan():
    """Generate and save an E-Challan as a PDF and add it to the history."""
    # Get form data
    vehicle_number = request.form.get("vehicle_number")
    violation_date = request.form.get("violation_date")
    violation_type = request.form.get("violation_type")
    penalty_amount = request.form.get("penalty")

    # Create PDF object
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Add only regular and bold DejaVu fonts
    pdf.add_font('DejaVu', '', 'static/fonts/DejaVuSans.ttf', uni=True)  # Regular font
    pdf.add_font('DejaVu', 'B', 'static/fonts/DejaVuSans-Bold.ttf', uni=True)  # Bold font
    
    pdf.set_font('DejaVu', 'B', 16)  # Use bold font for title
    pdf.cell(200, 10, txt="E-Challan", ln=True, align='C')
    pdf.ln(10)  # Line break

    # Set Details Section
    pdf.set_font('DejaVu', '', 12)  # Use regular font for content

    # Add table headers
    pdf.cell(95, 10, 'Vehicle Number', border=1, align='C')
    pdf.cell(95, 10, 'Violation Date', border=1, align='C')
    pdf.ln(10)  # Line break after headers

    # Add data in grid format
    pdf.cell(95, 10, vehicle_number, border=1, align='C')
    pdf.cell(95, 10, violation_date, border=1, align='C')
    pdf.ln(10)  # Line break

    pdf.cell(95, 10, 'Violation Type', border=1, align='C')
    pdf.cell(95, 10, 'Penalty Amount (INR)', border=1, align='C')
    pdf.ln(10)  # Line break after headers

    pdf.cell(95, 10, violation_type, border=1, align='C')
    pdf.cell(95, 10, f'₹ {penalty_amount}', border=1, align='C')
    pdf.ln(20)  # Line break after content

    # Footer Section with instructions or other info (optional)
    pdf.set_font('DejaVu', '', 10)  # Use regular font for footer
    pdf.cell(200, 10, 'Note: Please pay the fine within the specified time period.', ln=True, align='C')

    # Save the PDF in the challans folder
    challan_filename = f"{vehicle_number}_challan.pdf"
    challan_path = os.path.join(CHALLAN_FOLDER, challan_filename)
    pdf.output(challan_path)

    # Store the E-Challan record in JSON file
    challan_record = {
        "vehicle_number": vehicle_number,
        "violation_date": violation_date,
        "violation_type": violation_type,
        "penalty_amount": penalty_amount,
        "generated_at": str(datetime.now())  # Timestamp of when it was generated
    }

    # Load existing history or create a new one
    if os.path.exists(CHALLAN_HISTORY_FILE):
        with open(CHALLAN_HISTORY_FILE, 'r') as file:
            challan_history = json.load(file)
    else:
        challan_history = []

    # Add new record to the history
    challan_history.append(challan_record)

    # Write the updated history back to the JSON file
    with open(CHALLAN_HISTORY_FILE, 'w') as file:
        json.dump(challan_history, file, indent=4)

    return f"E-Challan Generated Successfully! <a href='/{challan_path}' target='_blank'>Download Here</a>"

@app.route("/challan-history")
def challan_history():
    """Display the history of all generated E-Challans."""
    # Check if the file exists and is readable
    if os.path.exists(CHALLAN_HISTORY_FILE):
        with open(CHALLAN_HISTORY_FILE, 'r') as file:
            # Read the existing challan history
            challan_history = json.load(file)
    else:
        challan_history = []  # If the file doesn't exist, initialize as an empty list

    # Render the template to display the history
    return render_template("challan_history.html", challan_history=challan_history)


if __name__ == "__main__":
    app.run(debug=True)
