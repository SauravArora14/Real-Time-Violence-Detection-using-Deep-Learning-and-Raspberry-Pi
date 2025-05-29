#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  yolo_detectfinal.py
#  
#  Copyright 2025  <arora@arora>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

import os
import sys
import argparse
import time
import threading
import cv2
import numpy as np
import smtplib
from email.message import EmailMessage
from ultralytics import YOLO

# Email configuration
EMAIL_SENDER = 'raspii1408@gmail.com'
EMAIL_RECEIVER = 'arorasaurav2003@gmail.com'
EMAIL_PASSWORD = 'elsgfwugallulvxn'  # Use app password if using Gmail
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587

def send_email_alert(subject, body, attachment_path):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg.set_content(body)

    with open(attachment_path, 'rb') as f:
        file_data = f.read()
        file_name = os.path.basename(attachment_path)
    msg.add_attachment(file_data, maintype='video', subtype='avi', filename=file_name)

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as smtp:
            smtp.starttls()
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(msg)
            print(f"[INFO] Email alert sent to {EMAIL_RECEIVER}")
    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")

def record_clip(video_capture, width, height, duration=10, filename="alert.avi"):
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"XVID"), 20.0, (width, height))
    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = video_capture.read()
        if ret:
            out.write(frame)
    out.release()
    print(f"[INFO] Saved alert clip: {filename}")
    send_email_alert("?? Alert: Gun/Knife Detected", "A dangerous object was detected.", filename)

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--source', required=True)
parser.add_argument('--thresh', default=0.5, type=float)
parser.add_argument('--resolution', default=None)
args = parser.parse_args()

# Load model
model = YOLO(args.model, task='detect')
labels = model.names
target_classes = ['Knive', 'Gun']

# Input source
img_source = args.source
resize = False

if args.resolution:
    resW, resH = map(int, args.resolution.split('x'))
    resize = True

if 'usb' in img_source:
    cap_arg = int(img_source[3:])
    cap = cv2.VideoCapture(cap_arg)
elif os.path.isfile(img_source):
    cap = cv2.VideoCapture(img_source)
else:
    print(f"Unsupported source: {img_source}")
    sys.exit(1)

if resize:
    cap.set(3, resW)
    cap.set(4, resH)

alert_active = False

# Detection loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    results = model(frame, verbose=False)
    detections = results[0].boxes

    alert_this_frame = False
    for det in detections:
        class_id = int(det.cls.item())
        class_name = labels[class_id]
        conf = det.conf.item()

        if conf >= args.thresh and class_name in target_classes:
            alert_this_frame = True
            xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if alert_this_frame and not alert_active:
        alert_active = True
        print("[INFO] Alert triggered. Recording and sending email...")
        threading.Thread(target=record_clip, args=(cap, frame.shape[1], frame.shape[0], 10)).start()
        threading.Timer(15, lambda: setattr(sys.modules[__name__], 'alert_active', False)).start()

    cv2.imshow("YOLO Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
cap.release()
cv2.destroyAllWindows()
